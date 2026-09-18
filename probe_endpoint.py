"""사내 LLM 엔드포인트가 Jev 방식(logprobs 기반 판단)을 지원하는지 점검한다.

    set JEV_BASE_URL=http://.../v1
    set JEV_MODEL=qwen3.8
    set JEV_API_KEY=...        (필요시)
    set JEV_INSECURE=1         (자체서명 인증서일 때)
    python probe_endpoint.py

표준 라이브러리만 사용. 회사 데이터는 보내지 않는다(고정된 예시 문장만 사용).
"""
from __future__ import annotations

import os
import statistics
import sys
import time

from jev_api import QWEN_NO_THINK, APIError, SystemOneAPI, choice, http_json, noul, score

SAMPLE = "Help! My payouts have been failing for 3 days."
AB_PROMPT = (f"State:\n{SAMPLE}\n\nQuestion: Does this message convey urgency?\n"
             "A. Yes\nB. No\n\nAnswer with the option letter only.")

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001 — 점검 스크립트라 모든 실패를 기록
            ok, msg = False, f"{type(e).__name__}: {e}"
        results.append((name, ok, msg))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {msg}")
        return ok
    return deco


def main() -> int:
    base = os.environ.get("JEV_BASE_URL", "").rstrip("/")
    model = os.environ.get("JEV_MODEL", "")
    key = os.environ.get("JEV_API_KEY")
    insecure = os.environ.get("JEV_INSECURE") == "1"
    if not base or not model:
        print("JEV_BASE_URL, JEV_MODEL 환경변수를 설정하세요.")
        return 2
    kw = dict(api_key=key, insecure=insecure, timeout=60)
    print(f"endpoint={base}  model={model}\n")

    def chat(extra: dict, top: int = 5) -> dict:
        body = {"model": model, "messages": [{"role": "user", "content": AB_PROMPT}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": top, **extra}
        return http_json("POST", f"{base}/chat/completions", body, **kw)

    def first_and_top(r: dict) -> tuple[str, list]:
        lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
        if not lp:
            return "", []
        return lp[0]["token"], [(t["token"], round(t["logprob"], 3)) for t in lp[0].get("top_logprobs") or []]

    @check("1. GET /models")
    def _():
        r = http_json("GET", f"{base}/models", **kw)
        ids = [m.get("id") for m in r.get("data", [])]
        return model in ids, f"served models: {ids}"

    def is_label(tok: str) -> bool:
        # thinking이 켜져 있으면 첫 토큰이 "<think>", "Thinking" 등 라벨이 아닌 것이 나온다
        return tok.strip() in ("A", "B")

    state = {"no_think_kwarg": False, "default_think": None, "max_top": 0, "completions": False}

    @check("2. chat + logprobs + chat_template_kwargs(enable_thinking=False)")
    def _():
        first, top = first_and_top(chat({"chat_template_kwargs": {"enable_thinking": False}}))
        if not top:
            return False, "logprobs가 비어 있음 — 게이트웨이가 logprobs를 제거하거나 모델이 미지원"
        state["no_think_kwarg"] = is_label(first)
        return state["no_think_kwarg"], (f"first token={first!r}, top={top}"
                                         + ("" if is_label(first) else " → 라벨이 아님: thinking이 안 꺼졌을 수 있음"))

    @check("3. chat 기본값(kwargs 없음)에서 thinking 여부")
    def _():
        first, top = first_and_top(chat({}))
        state["default_think"] = not is_label(first)
        return True, (f"first token={first!r} → "
                      + ("기본이 thinking ON (끄는 설정 필요)" if state["default_think"] else "기본이 thinking OFF"))

    @check("4. top_logprobs 최대 허용치")
    def _():
        for n in (20, 10, 5, 1):
            try:
                _, top = first_and_top(chat({"chat_template_kwargs": {"enable_thinking": False}}, top=n))
                if top:
                    state["max_top"] = n
                    return True, f"top_logprobs={n} OK (받은 개수 {len(top)})"
            except APIError as e:
                last = str(e)[:200]
        return False, f"모든 값 실패: {last}"

    @check("5. /completions + ChatML 직접 구성 (chat_template_kwargs 대안)")
    def _():
        prompt = f"<|im_start|>user\n{AB_PROMPT}<|im_end|>\n<|im_start|>assistant\n{QWEN_NO_THINK}"
        r = http_json("POST", f"{base}/completions",
                      {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 5}, **kw)
        lp = r["choices"][0].get("logprobs") or {}
        tops = lp.get("top_logprobs") or []
        state["completions"] = bool(tops)
        return bool(tops), f"first token={(lp.get('tokens') or [''])[0]!r}, top={tops[:1]}"

    # 추천 설정 결정
    if state["no_think_kwarg"] and state["max_top"]:
        mode = "chat"
    elif state["completions"]:
        mode = "completions"
    else:
        mode = None

    if mode:
        eng = SystemOneAPI(base, model, key, mode=mode, top_logprobs=max(state["max_top"], 5), insecure=insecure)
        qs = {
            "urgent": noul("Does this message convey urgency?"),
            "intent": choice("What is the customer's main request?", {
                "refund": "The customer wants money returned.",
                "technical_help": "The customer needs a bug fixed.",
                "other": "None of the other options clearly fits."}),
            "frustration": score("How frustrated does the customer appear?",
                                 ["Calm and neutral", "Concerned but civil", "Very angry"]),
        }

        @check(f"6. SystemOneAPI end-to-end (mode={mode})")
        def _():
            r = eng.system_one(SAMPLE, qs)
            a = r["answers"]
            masses = {k: round(v["label_mass"], 3) for k, v in a.items()}
            ok = all(m > 0.5 for m in masses.values())
            return ok, (f"urgent={a['urgent']['noul']:.3f}, intent={a['intent']['choice']} "
                        f"({a['intent']['confidence']:.2f}), frustration={a['frustration']['level']}; "
                        f"label_mass={masses}; first_tokens={[v['first_token'] for v in a.values()]}")

        @check("7. 지연시간 (질문 1개 x5, 질문 3개 병렬 x5)")
        def _():
            one = [eng.system_one(SAMPLE, {"u": qs["urgent"]})["usage"]["latency_ms"] for _ in range(5)]
            three = [eng.system_one(SAMPLE, qs)["usage"]["latency_ms"] for _ in range(5)]
            return True, (f"1Q median {statistics.median(one):.0f}ms (min {min(one):.0f}), "
                          f"3Q parallel median {statistics.median(three):.0f}ms (min {min(three):.0f})")

    print("\n=== 요약 ===")
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if mode:
        print(f"\n추천 설정: SystemOneAPI(mode={mode!r}, top_logprobs={max(state['max_top'], 5)})")
    else:
        print("\nlogprobs를 받을 수 있는 경로가 없음 — 게이트웨이 담당자에게 logprobs 통과 여부 문의 필요")
    return 0 if mode else 1


if __name__ == "__main__":
    t = time.time()
    code = main()
    print(f"\n(elapsed {time.time() - t:.1f}s)")
    sys.exit(code)
