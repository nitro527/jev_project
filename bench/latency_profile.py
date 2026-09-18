"""시간이 어디서 드는지 분리 측정: 입력 처리(prefill) vs 출력 생성(decode), 그리고 prefix cache 여부.

    (JEV_* 환경변수 설정 후) python bench/latency_profile.py [--reps 3]

[A] 입력 고정, 출력 토큰 수만 변경 → 출력 1토큰당 비용 (기울기)
[B] 출력 1토큰 고정, 입력 길이만 변경 → 입력 1토큰당 비용 (기울기)
[C] 같은 긴 state 뒤에 질문만 바꿔서 연속 호출 → 두 번째부터 빨라지면 서버 prefix cache가 동작 중
    (Jev 방식은 질문마다 state를 다시 보내므로, 사내 서버에서 prefix cache가 켜져 있는지가 중요하다)

집 PC 가짜 서버(transformers, 최적화 커널 없음) 결과: 출력 약 46~50ms/토큰, 입력 약 0.3~0.5ms/토큰 → 출력 1토큰 ≈ 입력 100~160토큰.
가짜 서버에는 prefix cache가 없어서 [C]의 차이가 없다.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev_api import SystemOneAPI  # noqa: E402

LINE = "RAS KERNEL INFO instruction cache parity error corrected. "


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    api = SystemOneAPI()
    print(f"endpoint={api.base_url} model={api.model} mode={api.mode}")

    print("\n[A] 입력 고정, 출력 토큰 수만 변경")
    base = "Explain in detail, at length, what this log line means:\n" + LINE * 3
    xs, ys = [], []
    for mt in (1, 5, 10, 20, 40):
        if mt == 1:
            ms, r = timed(lambda: api.top_tokens(base), args.reps)
            pin, pout = r[2], 1
        else:
            ms, r = timed(lambda: api.chat([{"role": "user", "content": base}], max_tokens=mt), args.reps)
            pin, pout = r["usage"]["input_tokens"], r["usage"]["output_tokens"]
        xs.append(pout)
        ys.append(ms)
        print(f"  out={pout:3}  in={pin:5}  {ms:7.0f}ms")
    per_out = slope(xs, ys)
    print(f"  → 출력 1토큰당 약 {per_out:.1f}ms")

    print("\n[B] 출력 1토큰 고정, 입력 길이만 변경")
    xs, ys = [], []
    for rep in (3, 30, 150, 300):
        content = "Is this log line a failure? Answer A or B.\n" + LINE * rep
        ms, r = timed(lambda: api.top_tokens(content), args.reps)
        xs.append(r[2])
        ys.append(ms)
        print(f"  out=  1  in={r[2]:5}  {ms:7.0f}ms")
    per_in = slope(xs, ys)
    print(f"  → 입력 1토큰당 약 {per_in:.2f}ms  (출력 1토큰 ≈ 입력 {per_out / max(per_in, 1e-6):.0f}토큰)")

    print("\n[C] prefix cache: 같은 긴 state(약 1500토큰) + 질문만 변경, 순서대로 1회씩")
    state = "State:\n" + LINE * 150 + "\n\n"
    qs = ["Question: Is this a failure?\nA. Yes\nB. No\n\nAnswer with the option letter only.",
          "Question: Is this routine?\nA. Yes\nB. No\n\nAnswer with the option letter only.",
          "Question: Should an operator act?\nA. Yes\nB. No\n\nAnswer with the option letter only.",
          "Question: Is the hardware at fault?\nA. Yes\nB. No\n\nAnswer with the option letter only."]
    times = []
    for q in qs:
        ms, _ = timed(lambda: api.top_tokens(state + q), 1)
        times.append(ms)
        print(f"  {ms:7.0f}ms")
    ratio = statistics.median(times[1:]) / times[0]
    print(f"  → 두 번째 이후 / 첫 번째 = {ratio:.2f}  "
          + ("(prefix cache 동작 중으로 보임)" if ratio < 0.6 else "(prefix cache 효과 없음 또는 꺼짐)"))


if __name__ == "__main__":
    main()
