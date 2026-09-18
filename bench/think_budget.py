"""사고 예산 실험: thinking을 얼마나 허락하느냐에 따라 1토큰 판단이 어떻게 바뀌는가.

조건
  off   : thinking 끔 (raw, 기본 Jev 방식)
  B<n>  : thinking으로 최대 n토큰 풀이를 쓰게 한 뒤, 풀이 끝 자리에서 라벨 확률을 읽음 (think-then-decide)
          예산 안에 풀이가 끝나지 않으면 잘린 풀이 뒤에 강제로 답 자리를 붙인다 (cut 표시)

    (JEV_* 환경변수 설정 후) python bench/think_budget.py [--budgets 256,1024]

문항: 여러 단계 계산이 필요한 사실 판단 10개(산술, 날짜, 글자 세기, 함정). 공개 가능한 일반 문장.
집 PC(Qwen3.5-4B) 결과: off 7/10, 256토큰(대부분 잘림) 8/10, 1024토큰 10/10 (평균 29초).
생각을 끝까지 마쳐야 효과가 있고, 마친 뒤에는 확률이 0/1로 쏠린다.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import improve_eval as E  # noqa: E402
from jev_api import SystemOneAPI  # noqa: E402

ITEMS = [
    ("17 * 23 = 391", "yes"),
    ("A week has 168 hours.", "yes"),
    ("한글은 세종대왕이 창제했다.", "yes"),
    ("A bat and ball cost $1.10; the bat costs $1 more. The ball costs $0.10.", "no"),
    ("9.11 is greater than 9.9.", "no"),
    ("2의 10제곱은 1000이다.", "no"),
    ("37 * 43 = 1591", "yes"),
    ("If today is Wednesday, 100 days from today is a Friday.", "yes"),
    ("The number of letter r's in 'strawberry' is 2.", "no"),
    ("1234 * 5 = 6160", "no"),
]
SPEC = {"kind": "noul", "instructions": "Is the statement true?"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="256,1024")
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    caller = E.Caller(SystemOneAPI())
    print(f"endpoint={caller.api.base_url} model={caller.api.model}\n")

    conds = ["off"] + [f"B{b}" for b in budgets]
    stats = {c: {"ok": 0, "conf": [], "ms": [], "tok": [], "cut": 0} for c in conds}
    for text, gold in ITEMS:
        cells = []
        p, rec = E.dist(caller, text, SPEC, [0, 1])
        results = {"off": (p[0], rec["ms"], 0, False)}
        for b in budgets:
            p, c = E.think_then_decide(caller, text, SPEC, budget=b)
            think_tok = c["out"] - 1
            results[f"B{b}"] = (p[0], c["ms"], think_tok, think_tok >= b)
        for cnd in conds:
            p_yes, ms, tok, cut = results[cnd]
            ok = ("yes" if p_yes >= 0.5 else "no") == gold
            s = stats[cnd]
            s["ok"] += ok
            s["conf"].append(max(p_yes, 1 - p_yes))
            s["ms"].append(ms)
            s["tok"].append(tok)
            s["cut"] += cut
            cells.append(f"{cnd}: P(yes)={p_yes:.2f} {'OK' if ok else 'XX'} t={tok}{'(cut)' if cut else ''} "
                         f"{ms / 1000:.1f}s")
        print(f"{text[:42]:42} gold={gold:3} | " + " | ".join(cells), flush=True)

    print()
    n = len(ITEMS)
    for c in conds:
        s = stats[c]
        print(f"{c:6} acc {s['ok']}/{n}  mean conf {sum(s['conf']) / n:.2f}  min conf {min(s['conf']):.2f}  "
              f"mean think tokens {sum(s['tok']) / n:.0f}  cut {s['cut']}/{n}  mean time {sum(s['ms']) / n / 1000:.1f}s")


if __name__ == "__main__":
    main()
