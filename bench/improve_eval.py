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
  plain+thinking (--plain-thinking N) 평소 방식, 느려서 앞쪽 N건 표본만 (같은 표본에서 다른 기법과 비교)

지표: acc(+95% bootstrap CI), macro-F1, Brier, ECE, AUROC(확신도로 정오 구분), AURC, 상위 50% 확신도 정확도,
      항목당 입력/출력 토큰, 지연 p50/p95, raw 대비 토큰/시간 배수, 1회성 준비 비용(보정용 호출·라벨 수),
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

    def memo(self, key: str, fn) -> dict:
        """plain / thinking 같은 비싼 호출 결과(JSON)를 캐시한다."""
        k = hashlib.sha1(f"{self.api.model}|{key}".encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]["v"]
        v = fn()
        self.cache[k] = {"k": k, "v": v}
        self.fh.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
        self.fh.flush()
        return v


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

def plain_pred(caller: Caller, state: str, spec: dict, names: list[str],
               thinking: bool = False) -> tuple[list[float] | None, dict]:
    """(확률 벡터 또는 파싱 실패 시 None, 비용 {in, out, ms})."""
    api = caller.api
    q = ({"type": "noul", "instructions": spec["instructions"]} if spec["kind"] == "noul"
         else {"type": "choice", "instructions": spec["instructions"], "choices": spec["options"]})

    def run():
        t0 = time.perf_counter()
        r = api.plain_decide(state, {"answer": q}, thinking=thinking)
        return {"answers": r["answers"], "output": r["output"], "in": r["usage"]["input_tokens"],
                "out": r["usage"]["output_tokens"], "ms": (time.perf_counter() - t0) * 1000}

    r = caller.memo(f"plain|{thinking}|{spec['instructions']}|{state}", run)
    cost = {"in": r["in"], "out": r["out"], "ms": r["ms"]}
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
        return None, cost
    return [1.0 if n == ans else 0.0 for n in names], cost


def think_then_decide(caller: Caller, state: str, spec: dict, budget: int) -> tuple[list[float], dict]:
    """thinking으로 풀이를 생성한 뒤, 풀이 끝 자리에서 라벨 확률을 읽는다. (확률, 비용 {in, out, ms})."""
    api = caller.api
    names, shown = task_options(spec)
    labels = list(LETTERS[: len(shown)])
    content = build_user_content(state, {"type": "choice", "instructions": spec["instructions"]}, labels, shown)

    def run():
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
        # 입력: 생성 호출 입력 + 판독 호출 입력(풀이 포함) / 출력: 풀이 토큰 + 1
        return {"p": softmax([top.get(l, -1e9) for l in labels]),
                "in": g["usage"]["input_tokens"] + (r.get("usage") or {}).get("prompt_tokens", 0),
                "out": g["usage"]["output_tokens"] + 1, "ms": (time.perf_counter() - t0) * 1000}

    r = caller.memo(f"think|{budget}|{content}", run)
    return r["p"], {"in": r["in"], "out": r["out"], "ms": r["ms"]}


# --- 실행 ---

SIZES = {"bgl": (20, 80), "boolq": (40, 120), "agnews": (40, 120)}


def cost_of(rec: dict) -> dict:
    """label_logprobs 호출 1회의 비용 (출력은 1토큰)."""
    return {"in": rec["in"], "out": 1, "ms": rec["ms"]}


def add_costs(*cs: dict) -> dict:
    return {k: sum(c[k] for c in cs) for k in ("in", "out", "ms")}


def cost_summary(costs: list[dict]) -> dict:
    ms = sorted(c["ms"] for c in costs)
    n = len(costs)
    return {"in_tok": sum(c["in"] for c in costs) / n, "out_tok": sum(c["out"] for c in costs) / n,
            "ms_p50": ms[n // 2], "ms_p95": ms[min(n - 1, int(n * 0.95))], "ms_mean": sum(ms) / n}


def run_task(task: str, caller: Caller, do_plain: bool, cascade_cap: int, cascade_budget: int,
             plain_think_n: int) -> dict:
    spec = D.TASKS[task]
    names, _ = task_options(spec)
    dev, test = D.load(task, *SIZES[task])
    gold = [names.index(it["gold"]) for it in test]
    n = len(test)
    ident = list(range(len(names)))
    perms = perms_for(len(names), spec["kind"])
    res, cost, calls, setup = {}, {}, {}, {}

    print(f"\n### {task} (dev {len(dev)}, test {n})", flush=True)

    # raw
    raw_pairs = [dist(caller, it["state"], spec, ident) for it in test]
    raw = [p for p, _ in raw_pairs]
    raw_cost = [cost_of(r) for _, r in raw_pairs]
    res["raw"], cost["raw"], calls["raw"] = raw, raw_cost, 1
    print("  raw done", flush=True)

    # perm: 순서마다 1회씩 (순차 합산 비용; 병렬로 보내면 지연은 줄어든다)
    perm_pairs = [[dist(caller, it["state"], spec, pm) for pm in perms] for it in test]
    by_perm = [[p for p, _ in ps] for ps in perm_pairs]
    res["perm"] = [normalize([sum(x) / len(x) for x in zip(*ps)]) for ps in by_perm]
    cost["perm"] = [add_costs(*[cost_of(r) for _, r in ps]) for ps in perm_pairs]
    calls["perm"] = len(perms)
    print("  perm done", flush=True)

    # contextual calibration: 내용 없는 입력은 태스크당 한 번 (준비 비용)
    cf_pairs = {tuple(pm): [dist(caller, s, spec, pm) for s in CONTENT_FREE] for pm in perms}
    cf = {k: normalize([sum(x) / len(CONTENT_FREE) for x in zip(*[p for p, _ in v])]) for k, v in cf_pairs.items()}
    setup["cc"] = {"calls": len(CONTENT_FREE)}
    setup["perm+cc"] = {"calls": len(CONTENT_FREE) * len(perms)}
    cc = lambda p, pm: normalize([v / max(c, 1e-6) for v, c in zip(p, cf[tuple(pm)])])  # noqa: E731
    res["cc"], cost["cc"], calls["cc"] = [cc(p, ident) for p in raw], raw_cost, 1
    res["perm+cc"] = [normalize([sum(x) / len(x) for x in zip(*[cc(p, pm) for p, pm in zip(ps, perms)])])
                      for ps in by_perm]
    cost["perm+cc"], calls["perm+cc"] = cost["perm"], len(perms)

    # batch calibration: 추가 호출 없음
    prior = normalize([sum(x) / n for x in zip(*raw)])
    res["bc"] = [normalize([v / max(c, 1e-6) for v, c in zip(p, prior)]) for p in raw]
    cost["bc"], calls["bc"] = raw_cost, 1

    # few-shot: 프롬프트가 길어져 입력 토큰이 는다
    shots = pick_fewshot(dev, names, 2 if spec["kind"] == "noul" else 1)
    fs_pairs = [dist(caller, fewshot_state(shots, spec, it["state"]), spec, ident) for it in test]
    fs = [p for p, _ in fs_pairs]
    fs_cost = [cost_of(r) for _, r in fs_pairs]
    res["fewshot"], cost["fewshot"], calls["fewshot"] = fs, fs_cost, 1
    print("  fewshot done", flush=True)

    # devcal: dev 라벨로 한 번 학습 (준비 비용), 적용은 무료
    dev_gold = [names.index(it["gold"]) for it in dev]
    dev_pairs = [dist(caller, it["state"], spec, ident) for it in dev]
    T, b = fit_devcal([[math.log(max(v, 1e-12)) for v in p] for p, _ in dev_pairs], dev_gold)
    res["devcal"], cost["devcal"], calls["devcal"] = [apply_devcal(p, T, b) for p in raw], raw_cost, 1
    setup["devcal"] = {"calls": len(dev), "labels": len(dev)}

    fs_prior = normalize([sum(x) / n for x in zip(*fs)])
    res["fewshot+bc"] = [normalize([v / max(c, 1e-6) for v, c in zip(p, fs_prior)]) for p in fs]
    cost["fewshot+bc"], calls["fewshot+bc"] = fs_cost, 1
    shot_ids = {s["id"] for s in shots}
    dev_fs_items = [it for it in dev if it["id"] not in shot_ids]
    dev_fs_pairs = [dist(caller, fewshot_state(shots, spec, it["state"]), spec, ident) for it in dev_fs_items]
    T2, b2 = fit_devcal([[math.log(max(v, 1e-12)) for v in p] for p, _ in dev_fs_pairs],
                        [names.index(it["gold"]) for it in dev_fs_items])
    res["fewshot+devcal"] = [apply_devcal(p, T2, b2) for p in fs]
    cost["fewshot+devcal"], calls["fewshot+devcal"] = fs_cost, 1
    setup["fewshot+devcal"] = {"calls": len(dev_fs_items), "labels": len(dev)}

    extra = {"devcal_T": T, "devcal_bias": b, "cf_prior_identity": cf[tuple(ident)], "batch_prior": prior,
             "fewshot_ids": [s["id"] for s in shots], "setup_costs": setup}

    if do_plain:
        preds, pc, fails = [], [], 0
        for it in test:
            p, c = plain_pred(caller, it["state"], spec, names)
            pc.append(c)
            if p is None:
                fails += 1
                p = [1 / len(names)] * len(names)  # 파싱 실패 = 무작위 (오답 처리 효과)
            preds.append(p)
        res["plain"], cost["plain"], calls["plain"] = preds, pc, 1
        extra["plain_parse_fail"] = fails
        print("  plain done", flush=True)

    if cascade_cap:
        # raw 확신도가 낮은 것부터 최대 cascade_cap건을 thinking 재판단. 나머지는 raw 그대로.
        casc, cc_cost, n_esc = list(raw), list(raw_cost), 0
        low = sorted([i for i, p in enumerate(raw) if max(p) < CASCADE_TAU], key=lambda i: max(raw[i]))
        for i in low[:cascade_cap]:
            p, c = think_then_decide(caller, test[i]["state"], spec, budget=cascade_budget)
            casc[i] = p
            cc_cost[i] = add_costs(raw_cost[i], c)
            n_esc += 1
        res["cascade"], cost["cascade"], calls["cascade"] = casc, cc_cost, 1 + n_esc / n
        extra.update({"cascade_low_conf": len(low), "cascade_escalated": n_esc, "cascade_budget": cascade_budget})
        print(f"  cascade done ({n_esc}/{len(low)} low-conf escalated)", flush=True)

    table = {m: {**metrics(p, gold, names, spec["kind"]), **cost_summary(cost[m]), "calls_per_item": calls[m]}
             for m, p in res.items()}

    # plain+thinking(평소 방식)은 느려서 앞쪽 N건 표본만. 같은 표본에서 다른 기법과 비교한다.
    subset = None
    if plain_think_n:
        idx = list(range(min(plain_think_n, n)))
        preds, pc, fails = [], [], 0
        for i in idx:
            p, c = plain_pred(caller, test[i]["state"], spec, names, thinking=True)
            pc.append(c)
            if p is None:
                fails += 1
                p = [1 / len(names)] * len(names)
            preds.append(p)
        g = [gold[i] for i in idx]
        subset = {"n": len(idx), "plain_think_parse_fail": fails, "methods": {}}
        subset["methods"]["plain+thinking"] = {**metrics(preds, g, names, spec["kind"]), **cost_summary(pc)}
        for m in ("raw", "fewshot+devcal", "plain", "cascade"):
            if m in res:
                subset["methods"][m] = {**metrics([res[m][i] for i in idx], g, names, spec["kind"]),
                                        **cost_summary([cost[m][i] for i in idx])}
        print(f"  plain+thinking done (n={len(idx)})", flush=True)

    return {"task": task, "metrics": table, "subset": subset, "extra": extra}


def print_table(r: dict):
    t = r["metrics"]
    raw = t["raw"]
    has_yes = "yes_rate" in raw
    print(f"\n{r['task']}: n={raw['n']}   (cost = mean per item; x = multiple of raw)")
    head = (f"  {'method':15} {'acc':>6} {'95%CI':>11} {'ECE':>5} {'AUROC':>6} {'AURC':>5}"
            + (f" {'yes%':>5}" if has_yes else "")
            + f" {'calls':>5} {'in_tok':>7} {'out_tok':>7} {'p50ms':>7} {'p95ms':>7} {'tok_x':>6} {'ms_x':>6}")
    print(head)
    for m, v in t.items():
        au = f"{v['auroc']:.3f}" if v["auroc"] is not None else "  -  "
        tok_x = (v["in_tok"] + v["out_tok"]) / (raw["in_tok"] + raw["out_tok"])
        ms_x = v["ms_mean"] / raw["ms_mean"]
        line = (f"  {m:15} {v['acc']:6.3f} [{v['acc_ci95'][0]:.2f},{v['acc_ci95'][1]:.2f}] {v['ece']:5.3f} "
                f"{au:>6} {v['aurc']:5.3f}" + (f" {v['yes_rate']:5.2f}" if has_yes else "")
                + f" {v['calls_per_item']:5.2f} {v['in_tok']:7.0f} {v['out_tok']:7.1f} {v['ms_p50']:7.0f}"
                f" {v['ms_p95']:7.0f} {tok_x:6.1f} {ms_x:6.1f}")
        print(line)
    su = r["extra"]["setup_costs"]
    print("  one-time setup: " + ", ".join(
        f"{k}: {v['calls']} calls" + (f" / {v['labels']} labels" if "labels" in v else "") for k, v in su.items()))
    if r.get("subset"):
        s = r["subset"]
        print(f"  --- sample of {s['n']} items (incl. plain+thinking) ---")
        for m, v in s["methods"].items():
            print(f"  {m:15} acc {v['acc']:.3f}  in {v['in_tok']:6.0f}  out {v['out_tok']:7.1f}  "
                  f"p50 {v['ms_p50']:7.0f}ms  mean {v['ms_mean']:7.0f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="bgl,boolq,agnews")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--cascade", type=int, default=0, help="max think-then-decide items per task")
    ap.add_argument("--cascade-budget", type=int, default=768, help="thinking token budget for cascade")
    ap.add_argument("--plain-thinking", type=int, default=0, help="plain+thinking sample size per task")
    args = ap.parse_args()
    api = SystemOneAPI()
    caller = Caller(api)
    print(f"endpoint={api.base_url} model={api.model} mode={api.mode}")
    results = [run_task(t, caller, args.plain, args.cascade, args.cascade_budget, args.plain_thinking)
               for t in args.tasks.split(",")]
    for r in results:
        print_table(r)
    path = os.path.join(OUT, f"improve_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": api.model, "args": vars(args), "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
