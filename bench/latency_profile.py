"""시간이 어디서 드는지 분리 측정: 고정 비용, 입력 처리(prefill), 출력 생성(decode), prefix cache, 동시 처리량.

    (JEV_* 환경변수 설정 후) python bench/latency_profile.py [--reps 3] [--skip-throughput]

[A] 입력 고정, 출력 토큰 수만 변경 → 출력 1토큰당 비용 (전체 시간의 기울기로 간접 추정)
[B] 출력 1토큰 고정, 입력 길이만 변경 → 입력 1토큰당 비용 (기울기)
[C] 같은 긴 state 뒤에 질문만 바꿔 연속 호출 → 두 번째부터 빨라지면 서버 prefix cache가 동작 중
[D] 고정 비용: GET /models 왕복(네트워크+HTTP), 아주 짧은 프롬프트의 1토큰 요청(고정 비용 + 최소 계산)
[E] 스트리밍으로 직접 분리: TTFT(첫 토큰까지 = 고정 비용 + 입력 처리) vs ITL(토큰 사이 간격 = 출력 1토큰 시간).
    짧은 입력과 긴 입력으로 재서, 입력 처리 비용과 출력 비용을 각각 떼어낸다.
    이 값으로 "Jev(출력 1토큰)"와 "plain(출력 N토큰)"의 건당 시간을 이 서버 기준으로 계산한다.
[F] 동시 처리량: 동시 요청 수를 바꿔가며 초당 처리 건수를 잰다. Jev형(출력 1토큰) vs 생성형(출력 N토큰).
    vLLM은 동시 요청을 묶어(배칭) 처리하므로 동시성이 늘면 처리량이 오른다. 가짜 서버는 한 번에 하나만 처리한다.

출력 길이를 고정하려고 vLLM 확장 옵션 `min_tokens`/`ignore_eos`를 보낸다. 게이트웨이가 거부하면 빼고 다시 보내며,
그때는 실제로 나온 토큰 수를 그대로 보고한다.

집 PC 가짜 서버(transformers, 최적화 커널 없음, 배칭 없음) 결과: 출력 약 46~50ms/토큰, 입력 약 0.3~0.5ms/토큰
→ 출력 1토큰 ≈ 입력 100~160토큰. prefix cache와 배칭이 없어 [C], [F]의 개선이 없다.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev_api import APIError, SystemOneAPI, http_json  # noqa: E402

LINE = "RAS KERNEL INFO instruction cache parity error corrected. "
FORCE_LEN = {"min_tokens": None, "ignore_eos": True}  # min_tokens는 호출 시 채움


def timed(fn, reps: int) -> tuple[float, object]:
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), out


def slope(xs: list[float], ys: list[float]) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


class Forcer:
    """출력 길이 고정 옵션(min_tokens/ignore_eos)을 서버가 받는지 기억한다."""

    def __init__(self):
        self.ok = True

    def extra(self, n: int) -> dict:
        return {"min_tokens": n, "ignore_eos": True} if self.ok else {}


def gen(api: SystemOneAPI, forcer: Forcer, content: str, n: int) -> dict:
    """비스트리밍 생성, 출력 n토큰 고정 시도. {'in','out'} 반환."""
    body = {"model": api.model, "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": 0, **forcer.extra(n)}
    if api.mode == "chat":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        r = http_json("POST", f"{api.base_url}/chat/completions", body, api_key=api.api_key,
                      timeout=300, insecure=api.insecure)
    except APIError:
        if not forcer.ok:
            raise
        forcer.ok = False
        print("  (서버가 min_tokens/ignore_eos를 거부해서 빼고 다시 보냄. 출력 길이는 모델이 정한다)")
        return gen(api, forcer, content, n)
    u = r.get("usage") or {}
    return {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}


def stream_chat(api: SystemOneAPI, forcer: Forcer, content: str, n: int) -> dict:
    """SSE 스트리밍. 보낸 시각 기준으로 각 출력 조각이 도착한 시각(ms)을 기록한다."""
    body = {"model": api.model, "messages": [{"role": "user", "content": content}], "max_tokens": n,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, **forcer.extra(n)}
    if api.mode == "chat":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api.api_key:
        headers["Authorization"] = f"Bearer {api.api_key}"
    req = urllib.request.Request(f"{api.base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    ctx = ssl._create_unverified_context() if api.insecure else None
    t0 = time.perf_counter()
    arrivals, usage = [], {}
    try:
        with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        arrivals.append((time.perf_counter() - t0) * 1000)
    except urllib.error.HTTPError:
        if not forcer.ok:
            raise
        forcer.ok = False
        print("  (서버가 min_tokens/ignore_eos를 거부해서 빼고 다시 보냄. 출력 길이는 모델이 정한다)")
        return stream_chat(api, forcer, content, n)
    total = (time.perf_counter() - t0) * 1000
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return {"ttft": arrivals[0] if arrivals else total, "gaps": gaps, "chunks": len(arrivals), "total": total,
            "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens", len(arrivals))}


def section_a_b_c(api: SystemOneAPI, forcer: Forcer, reps: int) -> tuple[float, float]:
    print("\n[A] 입력 고정, 출력 토큰 수만 변경 (전체 시간의 기울기)")
    base = "Explain in detail, at length, what this log line means:\n" + LINE * 3
    xs, ys = [], []
    for mt in (1, 5, 10, 20, 40):
        if mt == 1:
            ms, r = timed(lambda: api.top_tokens(base), reps)
            pin, pout = r[2], 1
        else:
            ms, r = timed(lambda: gen(api, forcer, base, mt), reps)
            pin, pout = r["in"], r["out"]
        xs.append(pout)
        ys.append(ms)
        print(f"  out={pout:3}  in={pin:5}  {ms:7.0f}ms")
    per_out = slope(xs, ys)
    print(f"  → 출력 1토큰당 약 {per_out:.1f}ms")

    print("\n[B] 출력 1토큰 고정, 입력 길이만 변경")
    xs, ys = [], []
    for rep in (3, 30, 150, 300):
        content = "Is this log line a failure? Answer A or B.\n" + LINE * rep
        ms, r = timed(lambda: api.top_tokens(content), reps)
        xs.append(r[2])
        ys.append(ms)
        print(f"  out=  1  in={r[2]:5}  {ms:7.0f}ms")
    per_in = slope(xs, ys)
    print(f"  → 입력 1토큰당 약 {per_in:.3f}ms  (출력 1토큰 ≈ 입력 {per_out / max(per_in, 1e-6):.0f}토큰)")

    print("\n[C] prefix cache: 같은 긴 state(약 1500토큰) + 질문만 변경, 순서대로 1회씩")
    state = "State:\n" + LINE * 150 + "\n\n"
    qs = [f"Question: {q}\nA. Yes\nB. No\n\nAnswer with the option letter only."
          for q in ("Is this a failure?", "Is this routine?", "Should an operator act?", "Is the hardware at fault?")]
    times = []
    for q in qs:
        ms, _ = timed(lambda: api.top_tokens(state + q), 1)
        times.append(ms)
        print(f"  {ms:7.0f}ms")
    ratio = statistics.median(times[1:]) / times[0]
    print(f"  → 두 번째 이후 / 첫 번째 = {ratio:.2f}  "
          + ("(prefix cache 동작 중으로 보임)" if ratio < 0.6 else "(prefix cache 효과 없음 또는 꺼짐)"))
    return per_in, per_out


def section_d(api: SystemOneAPI, reps: int) -> float:
    print("\n[D] 고정 비용")
    ms_models, _ = timed(lambda: http_json("GET", f"{api.base_url}/models", api_key=api.api_key,
                                           insecure=api.insecure), max(reps, 5))
    ms_tiny, r = timed(lambda: api.top_tokens("Answer A or B.\nA. Yes\nB. No"), max(reps, 5))
    print(f"  GET /models 왕복 (네트워크 + HTTP, 모델 계산 없음): {ms_models:6.1f}ms")
    print(f"  아주 짧은 프롬프트(in={r[2]}) 1토큰 판단:         {ms_tiny:6.1f}ms  ← Jev 호출 1회의 바닥값")
    return ms_tiny


def section_e(api: SystemOneAPI, forcer: Forcer, reps: int, n_out: int) -> dict:
    print(f"\n[E] 스트리밍 분리 측정 (출력 {n_out}토큰 고정 시도, 중앙값 {reps}회)")
    res = {}
    for name, content in (("short", "Explain what this log line means:\n" + LINE),
                          ("long", "Explain what this log line means:\n" + LINE * 150)):
        runs = [stream_chat(api, forcer, content, n_out) for _ in range(reps)]
        ttft = statistics.median(r["ttft"] for r in runs)
        gaps = [g for r in runs for g in r["gaps"]]
        itl50, itl_mean = (statistics.median(gaps), statistics.mean(gaps)) if gaps else (0.0, 0.0)
        total = statistics.median(r["total"] for r in runs)
        r0 = runs[0]
        res[name] = {"ttft": ttft, "itl": itl50, "in": r0["in"], "out": r0["out"], "total": total}
        print(f"  {name:5} in={r0['in']!s:>5} out={r0['out']!s:>3} (조각 {r0['chunks']})  "
              f"TTFT {ttft:7.1f}ms   ITL p50 {itl50:6.1f}ms / mean {itl_mean:6.1f}ms / p95 {pct(gaps, 0.95) if gaps else 0:6.1f}ms"
              f"   전체 {total:7.0f}ms")
    s, l = res["short"], res["long"]
    if s["in"] and l["in"] and l["in"] != s["in"]:
        per_in = (l["ttft"] - s["ttft"]) / (l["in"] - s["in"])
        print(f"  → 입력 처리: 1토큰당 약 {per_in:.3f}ms (TTFT 차이 / 입력 차이)")
        res["per_in"] = per_in
    itl = statistics.median([s["itl"], l["itl"]])
    res["itl"] = itl
    print(f"  → 출력 생성: 1토큰당 약 {itl:.1f}ms (ITL)")
    return res


def section_f(api: SystemOneAPI, forcer: Forcer, levels: list[int], n_req: int, n_out: int):
    print(f"\n[F] 동시 처리량 (요청 {n_req}건씩, 입력 약 100토큰)")
    content = "State:\n" + LINE * 8 + "\n\nQuestion: Is this a failure?\nA. Yes\nB. No\n\nAnswer with the option letter only."
    workloads = {"jev (출력 1)": lambda: api.top_tokens(content),
                 f"생성 (출력 {n_out})": lambda: gen(api, forcer, content, n_out)}
    base, rows = {}, []
    print(f"  {'작업':14} {'동시':>4} {'전체(s)':>8} {'건/초':>7} {'건당 p50(ms)':>12} {'1동시 대비':>9}")
    for name, fn in workloads.items():
        for c in levels:
            lat = []

            def one(_):
                t = time.perf_counter()
                fn()
                lat.append((time.perf_counter() - t) * 1000)

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as ex:
                list(ex.map(one, range(n_req)))
            wall = time.perf_counter() - t0
            rps = n_req / wall
            base.setdefault(name, rps)
            print(f"  {name:14} {c:4} {wall:8.2f} {rps:7.2f} {statistics.median(lat):12.0f} {rps / base[name]:8.2f}x")
            rows.append({"workload": name, "concurrency": c, "wall_s": wall, "req_per_s": rps,
                         "latency_p50_ms": statistics.median(lat)})
    return rows


def fmt_ms(ms: float) -> str:
    return f"{ms / 1000:,.1f}초" if ms >= 10000 else f"{ms:,.0f}ms"


def bar(frac: float, width: int = 10) -> str:
    k = round(frac * width)
    return "█" * k + "░" * (width - k)


def breakdown(e: dict, n_plain: int, n_think: int) -> dict:
    """전체 시간 = ① 첫 토큰까지(고정 비용 + 입력 처리, TTFT) + ② 출력 생성((N-1) × 토큰 간격).
    Jev는 출력이 1토큰이라 ①만 든다. 실측한 TTFT와 ITL로 계산한다."""
    print(f"\n[요약] 전체 시간 중 출력이 차지하는 몫 (이 서버 실측값: TTFT와 토큰 간격 {e['itl']:.1f}ms로 계산)")
    out = {}
    methods = [("Jev (출력 1토큰)", 1), (f"plain (출력 {n_plain}토큰)", n_plain),
               (f"thinking (출력 {n_think:,}토큰)", n_think)]
    for key, label in (("short", "짧은 입력"), ("long", "긴 입력")):
        first = e[key]["ttft"]
        print(f"\n  {label} ({e[key]['in']}토큰)")
        print(f"  {'방식':24} {'전체':>10} {'① 첫 토큰까지':>14} {'② 출력 생성':>12} {'출력 비중':>9}  {'Jev 대비':>8}")
        rows = {}
        for name, n in methods:
            gen_ms = (n - 1) * e["itl"]
            total = first + gen_ms
            share = gen_ms / total if total else 0.0
            rows[name] = {"total_ms": total, "first_ms": first, "gen_ms": gen_ms, "output_share": share}
            print(f"  {name:24} {fmt_ms(total):>10} {fmt_ms(first):>14} {fmt_ms(gen_ms):>12} "
                  f"{share * 100:7.1f}%  {bar(share)}  {total / first:6.1f}배")
        plain = rows[f"plain (출력 {n_plain}토큰)"]
        print(f"  → plain과 Jev의 절대 차이: {fmt_ms(plain['gen_ms'])} (= 출력 생성 시간 전부)")
        out[key] = {"input_tokens": e[key]["in"], "rows": rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stream-out", type=int, default=32, help="[E] 스트리밍 출력 토큰 수")
    ap.add_argument("--gen-out", type=int, default=20, help="요약과 [F]에 쓰는 plain(생성형) 출력 토큰 수")
    ap.add_argument("--think-out", type=int, default=3000, help="요약에 쓰는 thinking 출력 토큰 수")
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--n-requests", type=int, default=16)
    ap.add_argument("--skip-throughput", action="store_true")
    ap.add_argument("--out", default=None, help="결과 JSON 경로 (기본: bench/out/latency_<시각>.json)")
    args = ap.parse_args()
    api = SystemOneAPI()
    forcer = Forcer()
    print(f"endpoint={api.base_url} model={api.model} mode={api.mode}")

    per_in, per_out = section_a_b_c(api, forcer, args.reps)
    floor = section_d(api, args.reps)
    e = section_e(api, forcer, args.reps, args.stream_out)
    summary = breakdown(e, args.gen_out, args.think_out)

    throughput = None
    if not args.skip_throughput:
        throughput = section_f(api, forcer, [int(x) for x in args.concurrency.split(",")], args.n_requests,
                               args.gen_out)

    # 모델 이름·주소는 저장하지 않는다 (public 저장소에 결과를 옮길 때 사내 정보가 섞이지 않게)
    result = {"per_input_token_ms_slope": per_in, "per_output_token_ms_slope": per_out, "jev_floor_ms": floor,
              "stream": {k: v for k, v in e.items()}, "breakdown": summary, "throughput": throughput, "length_forcing_accepted": forcer.ok,
              "args": vars(args)}
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                                    f"latency_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
