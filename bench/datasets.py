"""공개 라벨 데이터셋 로더 — 받아서 bench/data/*.jsonl로 캐시한다 (gitignore됨, 라이선스상 커밋하지 않음).

각 태스크는 dev(보정/튜닝용)와 test(평가용)로 고정 시드 분할된다.
item = {"id", "state", "gold"}; 태스크 스펙 = {"kind": "noul"|"choice", "instructions", "options"}

- bgl     : Loghub BGL 2k (실제 슈퍼컴퓨터 로그, 관리자 알림 라벨). 한 줄이 알림(장애)인가 → noul
- boolq   : BoolQ validation (지문 + 예/아니오 질문) → noul
- agnews  : AG News test (뉴스 4분류) → choice
"""
from __future__ import annotations

import io
import json
import os
import random
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

TASKS = {
    "bgl": {
        "kind": "noul",
        "instructions": ("This is one line from a supercomputer (Blue Gene/L) system log. Does this line report "
                         "an actual failure or alert that operators should act on, rather than routine or "
                         "self-corrected informational activity?"),
    },
    "boolq": {
        "kind": "noul",
        "instructions": "Based on the passage, is the answer to the question yes?",
    },
    "agnews": {
        "kind": "choice",
        "instructions": "What is the topic of this news article?",
        "options": {"world": "World news, politics, international affairs",
                    "sports": "Sports",
                    "business": "Business, economy, companies, markets",
                    "scitech": "Science and technology"},
    },
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _bgl() -> list[dict]:
    raw = _get("https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log").decode("utf-8", "replace")
    seen, items = set(), []
    for line in raw.splitlines():
        parts = line.split(" ", 9)
        if len(parts) < 10:
            continue
        label, comp, level, content = parts[0], parts[7], parts[8], parts[9]
        text = f"{parts[6]} {comp} {level} {content}"  # 라벨/타임스탬프/노드 ID 제외
        if text in seen:  # 완전히 같은 줄 반복은 하나로 (반복이 점수를 부풀리지 않게)
            continue
        seen.add(text)
        items.append({"state": text, "gold": "no" if label == "-" else "yes", "alert_type": label,
                      "level": level})
    return items


def _parquet(url: str):
    import pandas as pd  # bench 그룹 의존성 (집 PC 전용)
    return pd.read_parquet(io.BytesIO(_get(url)))


def _boolq() -> list[dict]:
    df = _parquet("https://huggingface.co/api/datasets/google/boolq/parquet/default/validation/0.parquet")
    return [{"state": f"Passage: {r.passage}\n\nQuestion: {r.question}?", "gold": "yes" if r.answer else "no"}
            for r in df.itertuples()]


def _agnews() -> list[dict]:
    df = _parquet("https://huggingface.co/api/datasets/fancyzhx/ag_news/parquet/default/test/0.parquet")
    names = ["world", "sports", "business", "scitech"]
    return [{"state": r.text, "gold": names[r.label]} for r in df.itertuples()]


LOADERS = {"bgl": _bgl, "boolq": _boolq, "agnews": _agnews}


def load(task: str, n_dev: int, n_test: int, seed: int = 0, balanced: bool = True) -> tuple[list, list]:
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"{task}.jsonl")
    if not os.path.exists(path):
        items = LOADERS[task]()
        with open(path, "w", encoding="utf-8") as f:
            for i, it in enumerate(items):
                f.write(json.dumps({"id": f"{task}-{i}", **it}, ensure_ascii=False) + "\n")
    with open(path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f]
    rng = random.Random(seed)
    rng.shuffle(items)
    if balanced:  # 라벨별로 고르게 뽑는다 (소수 클래스가 적으면 가능한 만큼)
        by: dict[str, list] = {}
        for it in items:
            by.setdefault(it["gold"], []).append(it)
        per = (n_dev + n_test) // len(by)
        if task == "bgl":
            # 함정: 알림은 전부 FATAL이지만 정상 줄에도 FATAL이 많다 → 정상의 절반은 FATAL 정상 줄로
            neg = by["no"]
            hard = [it for it in neg if it.get("level") == "FATAL"]
            easy = [it for it in neg if it.get("level") != "FATAL"]
            by["no"] = hard[: per // 2] + easy[: per - per // 2]
        picked = []
        for g in sorted(by):
            picked += by[g][:per]
        rng.shuffle(picked)
        items = picked
    items = items[: n_dev + n_test]
    return items[:n_dev], items[n_dev:]


if __name__ == "__main__":
    for t in TASKS:
        dev, test = load(t, 40, 120)
        gold = {}
        for it in dev + test:
            gold[it["gold"]] = gold.get(it["gold"], 0) + 1
        print(f"{t:7} dev={len(dev)} test={len(test)} labels={gold}")
