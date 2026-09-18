"""jev 판단 성능 개선 기법 비교 벤치마크 (공개 라벨 데이터, dev/test 분할).

    (JEV_* 환경변수 설정 후) python bench/improve_eval.py [--tasks bgl,boolq,agnews] [--plain] [--cascade N]

기법 (모두 같은 1토큰 logprob 호출 위에서 동작):
  raw        현재 구현 그대로 (선택지 순서 고정)
  perm       선택지 순서를 순환시켜 여러 번 묻고 평균 (위치/라벨 편향 제거, 호출 수 = 선택지 수)
  cc         contextual calibration: 내용 없는 입력("N/A" 등)에 대한 편향을 측정해 나눔 (라벨 불필요)
  perm+cc    둘 다
  bc         batch calibration: 테스트 배치 평균 확률을 사전분포로 보고 나눔 (라벨 불필요, 추가 호출 0)
  fewshot    dev에서 클래스별 예시를 뽑아 프롬프트에 넣음
  devcal     dev 라벨로 temperature + 클래스별 bias를 학습해 적용 (라벨 필요)
  plain      (--plain) 일반 생성으로 JSON 답 (비교 기준)
  cascade    (--cascade N) raw 확신도 < 0.7인 건만 think-then-decide로 재판단 (태스크당 최대 N건)

지표: acc(+95% bootstrap CI), macro-F1, Brier, ECE, AUROC(확신도로 정오 구분), AURC, 상위 50% 확신도 정확도,
      noul의 yes 비율(치우침), 항목당 호출 수/지연.
결과: bench/out/improve_<시각>.json, 호출 캐시: bench/out/cache.jsonl (재실행 시 재사용)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import datasets as D  # noqa: E402
from jev_api import (SystemOneAPI, build_user_content, http_json, parse_plain_answers,  # noqa: E402
                     _norm_token, LETTERS)

OUT = os.path.join(HERE, "out")
CONTENT_FREE = ["N/A", "", "[MASK]"]
CASCADE_TAU = 0.7


# --- 호출 + 캐시 ---

class Caller:
    def __init__(self, api: SystemOneAPI):
        self.api = api
        os.makedirs(OUT, exist_ok=True)
        self.path = os.path.join(OUT, "cache.jsonl")
        self.cache = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    self.cache[r["k"]] = r
        self.fh = open(self.path, "a", encoding="utf-8")

    def label_logprobs(self, state: str, instructions: str, shown: list[str]) -> dict:
        """선택지 표시문 목록(표시 순서) → 각 라벨(A,B,..)의 logprob. 캐시됨."""
        labels = list(LETTERS[: len(shown)])
        content = build_user_content(state, {"type": "choice", "instructions": instructions}, labels, shown)
        k = hashlib.sha1(f"{self.api.model}|{content}".encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        t0 = time.perf_counter()
        top, first, n_tok = self.api.top_tokens(content)
        rec = {"k": k, "lps": [top.get(l, -math.inf) for l in labels], "first": first, "in": n_tok,
               "ms": (time.perf_counter() - t0) * 1000}
        rec["lps"] = [x if x != -math.inf else -1e9 for x in rec["lps"]]  # JSON 저장용
        self.cache[k] = rec
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        return rec


def softmax(xs, T=1.0):
    m = max(xs)
    e = [math.exp((x - m) / T) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def normalize(p):
    s = sum(p)
    return [v / s for v in p] if s > 0 else [1 / len(p)] * len(p)


# --- 태스크 표현 ---

def task_options(spec: dict) -> tuple[list[str], list[str]]:
    """(답 이름, 표시문). noul은 yes/no 두 선택지로 다룬다 (표시문은 운영 코드와 동일)."""
    if spec["kind"] == "noul":
        return ["yes", "no"], ["Yes", "No"]
    names = list(spec["options"])
    return names, [f"{n}: {d}" for n, d in spec["options"].items()]


def perms_for(n: int, kind: str) -> list[list[int]]:
    if kind == "noul":
        return [[0, 1], [1, 0]]
    return [[(i + s) % n for i in range(n)] for s in range(n)]  # 순환 이동


def dist(caller: Caller, state: str, spec: dict, perm: list[int]) -> tuple[list[float], dict]:
    """perm 순서로 표시해 묻고, 원래 이름 순서의 확률 벡터로 돌려준다."""
    names, shown = task_options(spec)
    rec = caller.label_logprobs(state, spec["instructions"], [shown[i] for i in perm])
    p_disp = softmax(rec["lps"])
    p = [0.0] * len(names)
    for pos, idx in enumerate(perm):
        p[idx] = p_disp[pos]
    return p, rec


def fewshot_state(examples: list[dict], spec: dict, state: str) -> str:
    names, shown = task_options(spec)
    lines = ["Here are labeled examples of this task:\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:\n{ex['state']}\nCorrect answer: {shown[names.index(ex['gold'])]}\n")
    lines.append(f"Now judge this input:\n{state}")
    return "\n".join(lines)


def pick_fewshot(dev: list[dict], names: list[str], k_per_class: int, seed=0) -> list[dict]:
    rng = random.Random(seed)
    ex = []
    for n in names:
        pool = [d for d in dev if d["gold"] == n]
        ex += rng.sample(pool, min(k_per_class, len(pool)))
    rng.shuffle(ex)
    return ex


# --- 학습형 보정 (dev 라벨 사용) ---

def fit_devcal(logps: list[list[float]], gold: list[int], iters=500, lr=0.1) -> tuple[float, list[float]]:
    """NLL 최소화: p = softmax((log p_raw + b) / T). 단순 경사하강."""
    n = len(logps[0])
    logT, b = 0.0, [0.0] * n
    for _ in range(iters):
        T = math.exp(logT)
        gT, gb = 0.0, [0.0] * n
        for lp, y in zip(logps, gold):
            z = [(lp[j] + b[j]) / T for j in range(n)]
            p = softmax(z)
            for j in range(n):
                d = p[j] - (1 if j == y else 0)  # dNLL/dz_j
                gb[j] += d / T
                gT += d * (-(lp[j] + b[j]) / T)  # dz/dlogT = -z
        m = len(gold)
        logT -= lr * gT / m
        b = [bj - lr * g / m for bj, g in zip(b, gb)]
    return math.exp(logT), b


def apply_devcal(p_raw: list[float], T: float, b: list[float]) -> list[float]:
    return softmax([(math.log(max(v, 1e-12)) + bj) / T for v, bj in zip(p_raw, b)])


# --- 지표 ---

def metrics(probs: list[list[float]], gold: list[int], names: list[str], kind: str, seed=0) -> dict:
    n = len(gold)
    pred = [max(range(len(p)), key=p.__getitem__) for p in probs]
    conf = [p[k] for p, k in zip(probs, pred)]
    correct = [int(a == b) for a, b in zip(pred, gold)]
    acc = sum(correct) / n
    rng = random.Random(seed)
    boots = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(1000))
    f1s = []
    for c in range(len(names)):
        tp = sum(1 for p, g in zip(pred, gold) if p == c and g == c)
        fp = sum(1 for p, g in zip(pred, gold) if p == c and g != c)
        fn = sum(1 for p, g in zip(pred, gold) if p != c and g == c)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    brier = sum(sum((p[j] - (1 if j == g else 0)) ** 2 for j in range(len(p))) for p, g in zip(probs, gold)) / n
    ece, bins = 0.0, 10
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n) if lo < conf[i] <= hi or (b == 0 and conf[i] == 0)]
        if idx:
            ece += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    pos = [c for c, k in zip(conf, correct) if k]
    neg = [c for c, k in zip(conf, correct) if not k]
    auroc = (sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
             if pos and neg else None)
    order = sorted(range(n), key=lambda i: -conf[i])
    errs, aurc = 0, 0.0
    for r, i in enumerate(order, 1):
        errs += 1 - correct[i]
        aurc += errs / r
    aurc /= n
    top = order[: n // 2]
    out = {"n": n, "acc": acc, "acc_ci95": [boots[25], boots[974]], "macro_f1": sum(f1s) / len(f1s),
           "brier": brier, "ece": ece, "auroc": auroc, "aurc": aurc,
           "acc_top50conf": sum(correct[i] for i in top) / len(top)}
    if kind == "noul":
        out["yes_rate"] = sum(1 for p in pred if p == 0) / n
    return out


# --- plain / cascade ---

def plain_pred(api: SystemOneAPI, state: str, spec: dict, names: list[str]) -> tuple[list[float] | None, float]:
    q = ({"type": "noul", "instructions": spec["instructions"]} if spec["kind"] == "noul"
         else {"type": "choice", "instructions": spec["instructions"], "choices": spec["options"]})
    t0 = time.perf_counter()
    r = api.plain_decide(state, {"answer": q})
    ms = (time.perf_counter() - t0) * 1000
    ans = r["answers"].get("answer", {}).get("answer")
    if ans is None:  # 키 불일치 관대 처리 (HANDOFF TODO #1)
        o = r["output"]
        try:
            d = json.loads(o[o.find("{"):o.rfind("}") + 1])
            if len(d) == 1:
                ans = parse_plain_answers(json.dumps({"answer": next(iter(d.values()))}), {"answer": q})[0]["answer"]["answer"]
        except (json.JSONDecodeError, ValueError):
            pass
    if ans is None:
        return None, ms
    return [1.0 if n == ans else 0.0 for n in names], ms


def think_then_decide(api: SystemOneAPI, state: str, spec: dict, budget: int) -> tuple[list[float], float, int]:
    names, shown = task_options(spec)
    labels = list(LETTERS[: len(shown)])
    content = build_user_content(state, {"type": "choice", "instructions": spec["instructions"]}, labels, shown)
    t0 = time.perf_counter()
    g = api.chat([{"role": "user", "content": content}], thinking=True, max_tokens=budget)
    reasoning = g["reasoning"] or g["content"]
    prompt = (f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n"
              f"{reasoning.strip()}\n</think>\n\n")
    r = http_json("POST", f"{api.base_url}/completions",
                  {"model": api.model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 20},
                  api_key=api.api_key, insecure=api.insecure, timeout=300)
    top = {}
    for t, lp in r["choices"][0]["logprobs"]["top_logprobs"][0].items():
        k = _norm_token(t)
        top[k] = math.log(math.exp(top[k]) + math.exp(lp)) if k in top else lp
    p = softmax([top.get(l, -1e9) for l in labels])
    return p, (time.perf_counter() - t0) * 1000, g["usage"]["output_tokens"]


# --- 실행 ---

SIZES = {"bgl": (20, 80), "boolq": (40, 120), "agnews": (40, 120)}


def run_task(task: str, api: SystemOneAPI, caller: Caller, do_plain: bool, cascade_cap: int) -> dict:
    spec = D.TASKS[task]
    names, _ = task_options(spec)
    dev, test = D.load(task, *SIZES[task])
    gold = [names.index(it["gold"]) for it in test]
    ident = list(range(len(names)))
    perms = perms_for(len(names), spec["kind"])
    res, calls, lat = {}, {}, {}

    def timed(label, fn):
        t0 = time.perf_counter()
        out = fn()
        lat[label] = (time.perf_counter() - t0) * 1000 / len(test)
        return out

    print(f"\n### {task} (dev {len(dev)}, test {len(test)})", flush=True)
    raw = timed("raw", lambda: [dist(caller, it["state"], spec, ident)[0] for it in test])
    res["raw"], calls["raw"] = raw, 1
    print("  raw done", flush=True)

    # 호출 한 번의 실제 지연(캐시 무관): 캐시 기록의 ms 사용
    per_call_ms = [caller.label_logprobs(it["state"], spec["instructions"], task_options(spec)[1])["ms"]
                   for it in test]
    per_call_ms.sort()

    by_perm = timed("perm", lambda: [[dist(caller, it["state"], spec, pm)[0] for pm in perms] for it in test])
    res["perm"] = [normalize([sum(x) / len(x) for x in zip(*ps)]) for ps in by_perm]
    calls["perm"] = len(perms)
    print("  perm done", flush=True)

    # contextual calibration: 순서별로 내용 없는 입력의 분포를 구해 나눔
    cf = {tuple(pm): normalize([sum(x) / len(CONTENT_FREE) for x in
                                zip(*[dist(caller, s, spec, pm)[0] for s in CONTENT_FREE])]) for pm in perms}
    cc = lambda p, pm: normalize([v / max(c, 1e-6) for v, c in zip(p, cf[tuple(pm)])])  # noqa: E731
    res["cc"] = [cc(p, ident) for p in raw]
    calls["cc"] = 1
    res["perm+cc"] = [normalize([sum(x) / len(x) for x in zip(*[cc(p, pm) for p, pm in zip(ps, perms)])])
                      for ps in by_perm]
    calls["perm+cc"] = len(perms)

    prior = normalize([sum(x) / len(raw) for x in zip(*raw)])  # batch calibration
    res["bc"] = [normalize([v / max(c, 1e-6) for v, c in zip(p, prior)]) for p in raw]
    calls["bc"] = 1

    shots = pick_fewshot(dev, names, 2 if spec["kind"] == "noul" else 1)
    res["fewshot"] = timed("fewshot", lambda: [dist(caller, fewshot_state(shots, spec, it["state"]), spec, ident)[0]
                                               for it in test])
    calls["fewshot"] = 1
    print("  fewshot done", flush=True)

    dev_gold = [names.index(it["gold"]) for it in dev]
    dev_raw = [dist(caller, it["state"], spec, ident)[0] for it in dev]
    T, b = fit_devcal([[math.log(max(v, 1e-12)) for v in p] for p in dev_raw], dev_gold)
    res["devcal"] = [apply_devcal(p, T, b) for p in raw]
    calls["devcal"] = 1

    # fewshot 위에 보정 얹기 (fewshot은 기준선이 크게 이동하므로)
    fs = res["fewshot"]
    fs_prior = normalize([sum(x) / len(fs) for x in zip(*fs)])
    res["fewshot+bc"] = [normalize([v / max(c, 1e-6) for v, c in zip(p, fs_prior)]) for p in fs]
    calls["fewshot+bc"] = 1
    shot_ids = {s["id"] for s in shots}
    dev_fs_items = [it for it in dev if it["id"] not in shot_ids]
    dev_fs = [dist(caller, fewshot_state(shots, spec, it["state"]), spec, ident)[0] for it in dev_fs_items]
    T2, b2 = fit_devcal([[math.log(max(v, 1e-12)) for v in p] for p in dev_fs],
                        [names.index(it["gold"]) for it in dev_fs_items])
    res["fewshot+devcal"] = [apply_devcal(p, T2, b2) for p in fs]
    calls["fewshot+devcal"] = 1

    extra = {"devcal_T": T, "devcal_bias": b, "cf_prior_identity": cf[tuple(ident)], "batch_prior": prior,
             "call_ms_p50": per_call_ms[len(per_call_ms) // 2], "fewshot_ids": [s["id"] for s in shots]}

    if do_plain:
        preds, ms_list, fails = [], [], 0
        for it in test:
            p, ms = plain_pred(api, it["state"], spec, names)
            ms_list.append(ms)
            if p is None:
                fails += 1
                p = [1 / len(names)] * len(names)  # 파싱 실패 = 무작위 (오답 처리 효과)
            preds.append(p)
        res["plain"], calls["plain"] = preds, 1
        extra["plain_parse_fail"] = fails
        extra["plain_ms_p50"] = sorted(ms_list)[len(ms_list) // 2]
        print("  plain done", flush=True)

    if cascade_cap:
        casc, n_esc, think_ms, think_tok = list(raw), 0, [], []
        low = sorted([i for i, p in enumerate(raw) if max(p) < CASCADE_TAU], key=lambda i: max(raw[i]))
        for i in low[:cascade_cap]:
            p, ms, tok = think_then_decide(api, test[i]["state"], spec, budget=768)
            casc[i] = p
            n_esc += 1
            think_ms.append(ms)
            think_tok.append(tok)
        res["cascade"] = casc
        calls["cascade"] = 1 + n_esc / len(test)
        extra.update({"cascade_low_conf": len(low), "cascade_escalated": n_esc,
                      "cascade_think_ms_mean": sum(think_ms) / len(think_ms) if think_ms else 0,
                      "cascade_think_tokens_mean": sum(think_tok) / len(think_tok) if think_tok else 0})
        print(f"  cascade done ({n_esc}/{len(low)} low-conf escalated)", flush=True)

    table = {m: {**metrics(p, gold, names, spec["kind"]), "calls_per_item": calls[m]} for m, p in res.items()}
    return {"task": task, "metrics": table, "extra": extra}


def print_table(r: dict):
    t = r["metrics"]
    has_yes = "yes_rate" in next(iter(t.values()))
    print(f"\n{r['task']}: n={next(iter(t.values()))['n']}  (1 call p50 {r['extra']['call_ms_p50']:.0f}ms)")
    head = f"  {'method':9} {'acc':>6} {'95%CI':>13} {'F1':>5} {'Brier':>6} {'ECE':>5} {'AUROC':>6} {'AURC':>5} {'top50':>6}"
    print(head + (f" {'yes%':>5}" if has_yes else "") + f" {'calls':>5}")
    for m, v in t.items():
        au = f"{v['auroc']:.3f}" if v["auroc"] is not None else "  -  "
        line = (f"  {m:9} {v['acc']:6.3f} [{v['acc_ci95'][0]:.2f},{v['acc_ci95'][1]:.2f}] {v['macro_f1']:5.3f} "
                f"{v['brier']:6.3f} {v['ece']:5.3f} {au:>6} {v['aurc']:5.3f} {v['acc_top50conf']:6.3f}")
        if has_yes:
            line += f" {v['yes_rate']:5.2f}"
        print(line + f" {v['calls_per_item']:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="bgl,boolq,agnews")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--cascade", type=int, default=0, help="태스크당 think-then-decide 최대 건수")
    args = ap.parse_args()
    api = SystemOneAPI()
    caller = Caller(api)
    print(f"endpoint={api.base_url} model={api.model} mode={api.mode}")
    results = [run_task(t, api, caller, args.plain, args.cascade) for t in args.tasks.split(",")]
    for r in results:
        print_table(r)
    path = os.path.join(OUT, f"improve_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": api.model, "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
