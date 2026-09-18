"""Jev 스타일 판단 — 로컬 transformers 버전 (전체 logits 직접 사용).

프롬프트/답 구성은 jev_api.py와 공유한다. 사내 API 버전과 결과를 비교하는 기준용.

    engine = SystemOne("models/Qwen3.5-4B")
    engine.system_one(state, {"urgent": noul("Does this convey urgency?")})
"""
from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jev_api import (build_answer, build_options, build_user_content, choice, noul,  # noqa: F401
                     score, state_to_text)


class SystemOne:
    def __init__(self, model_dir: str, device: str = "cuda", dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map=device)
        self.model.eval()
        self.device = device

    def _prompt(self, content: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def _label_ids(self, labels: list[str]) -> list[int]:
        ids = []
        for l in labels:
            enc = self.tok.encode(l, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"label {l!r} is not a single token")
            ids.append(enc[0])
        return ids

    @torch.inference_mode()
    def _last_logits(self, prompts: list[str], batch: bool) -> tuple[torch.Tensor, int]:
        if batch:
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**enc).logits[:, -1].float()
            return logits, int(enc.attention_mask.sum())
        rows, n = [], 0
        for p in prompts:
            enc = self.tok(p, return_tensors="pt").to(self.device)
            rows.append(self.model(**enc).logits[0, -1].float())
            n += enc.input_ids.shape[1]
        return torch.stack(rows), n

    def system_one(self, state: Any, questions: dict[str, dict], *,
                   batch: bool = True, temperature: float = 1.0) -> dict:
        state_text = state_to_text(state)
        keys = list(questions)
        specs = [build_options(questions[k]) for k in keys]
        prompts = [self._prompt(build_user_content(state_text, questions[k], lab, shown))
                   for k, (lab, _, shown) in zip(keys, specs)]

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, n_tokens = self._last_logits(prompts, batch)
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000

        full = torch.softmax(logits, -1)
        answers = {}
        for row, (k, (labels, names, _)) in enumerate(zip(keys, specs)):
            ids = self._label_ids(labels)
            probs = torch.softmax(logits[row, ids] / temperature, -1).tolist()
            answers[k] = build_answer(questions[k], names, probs, full[row, ids].sum().item())

        return {"answers": answers,
                "usage": {"input_tokens": n_tokens, "questions": len(keys), "latency_ms": latency_ms}}
