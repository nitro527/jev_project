"""말로 한 확률(verbalized) vs logprob으로 읽은 확률 비교.

같은 문제에 대해
  - logprob : raw 방식. max_tokens=1 + logprobs로 A(Yes)/B(No) 확률을 직접 읽는다 (측정값)
  - verbal  : "0~100 숫자로 Yes일 확률만 답해"라고 시키고, 모델이 쓴 숫자를 받는다 (자기 보고)
를 비교한다. 두 방식의 프롬프트는 마지막 지시문 한 줄만 다르다.

    (JEV_* 환경변수 설정 후) python bench/verbal_vs_logprob.py [--tasks bgl,boolq]

집 PC(Qwen3.5-4B) 결과 요약: verbal은 거의 0 또는 100만 썼고(BGL은 두 값뿐), 오답에도 전부 100%를 붙였다.
형식은 강제할 수 있지만(파싱 실패 0) 숫자의 정직성은 강제할 수 없다 → 확률에는 logprobs가 필요하다.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import datasets as D  # noqa: E402
import improve_eval as E  # noqa: E402
from jev_api import SystemOneAPI, build_user_content  # noqa: E402

VERBAL_INSTRUCTION = "Reply with ONLY a number from 0 to 100: the probability (%) that the answer is Yes."


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def run(task: str, caller: E.Caller) -> dict:
    api = caller.api
    spec = D.TASKS[task]
    if spec["kind"] != "noul":
        raise ValueError("verbal 비교는 예/아니오(noul) 태스크만 지원")
    _, test = D.load(task, *E.SIZES[task])
    y = [1 if it["gold"] == "yes" else 0 for it in test]

    lp = [E.dist(caller, it["state"], spec, [0, 1])[0][0] for it in test]  # P(yes), raw와 같은 호출(캐시 공유)
    lp_ms = [E.dist(caller, it["state"], spec, [0, 1])[1]["ms"] for it in test]

    verbal, ms, outs, fails = [], [], [], 0
    for it in test:
        content = build_user_content(it["state"], {"type": "choice", "instructions": spec["instructions"]},
                                     ["A", "B"], ["Yes", "No"])
        content = content.replace("Answer with the option letter only.", VERBAL_INSTRUCTION)

        def call():
            t0 = time.perf_counter()
            g = api.chat([{"role": "user", "content": content}], thinking=False, max_tokens=8)
            return {"text": g["content"], "out": g["usage"]["output_tokens"],
                    "ms": (time.perf_counter() - t0) * 1000}

        r = caller.memo(f"verbal|{content}", call)
        m = re.search(r"\d+(\.\d+)?", r["text"])
        if m:
            verbal.append(min(float(m.group()), 100.0) / 100)
        else:
            fails += 1
            verbal.append(0.5)
        ms.append(r["ms"])
        outs.append(r["out"])

    def acc(s):
        return sum((v >= 0.5) == bool(l) for v, l in zip(s, y)) / len(y)

    def wrong_conf(s):
        return sorted(round(max(v, 1 - v), 2) for v, l in zip(s, y) if (v >= 0.5) != bool(l))

    return {
        "task": task, "n": len(y),
        "logprob": {"acc": acc(lp), "auroc_vs_gold": auroc(lp, y), "distinct_values": len({round(v, 3) for v in lp}),
                    "out_tokens": 1, "ms_p50": sorted(lp_ms)[len(lp_ms) // 2], "wrong_conf": wrong_conf(lp)},
        "verbal": {"acc": acc(verbal), "auroc_vs_gold": auroc(verbal, y), "distinct_values": len(set(verbal)),
                   "out_tokens": sum(outs) / len(outs), "ms_p50": sorted(ms)[len(ms) // 2], "parse_fail": fails,
                   "most_common": collections.Counter(round(v * 100) for v in verbal).most_common(6),
                   "wrong_conf": wrong_conf(verbal)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="bgl,boolq")
    args = ap.parse_args()
    caller = E.Caller(SystemOneAPI())
    print(f"endpoint={caller.api.base_url} model={caller.api.model}")
    for t in args.tasks.split(","):
        r = run(t, caller)
        print(f"\n{t} (n={r['n']})")
        for k in ("logprob", "verbal"):
            v = r[k]
            print(f"  {k:8} acc {v['acc']:.3f}  AUROC {v['auroc_vs_gold']:.3f}  distinct {v['distinct_values']:3}  "
                  f"out_tok {v['out_tokens']:.1f}  p50 {v['ms_p50']:.0f}ms  "
                  f"wrong={len(v['wrong_conf'])} (top conf {v['wrong_conf'][-5:]})")
        print(f"  verbal most common values: {r['verbal']['most_common']}  parse_fail={r['verbal']['parse_fail']}")


if __name__ == "__main__":
    main()
