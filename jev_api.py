"""Jev(TypeSafe System One) 스타일 판단 — OpenAI 호환 API(vLLM 등) 버전.

표준 라이브러리만 사용한다(사내망에서 pip 설치 없이 동작).
질문마다 max_tokens=1 요청을 보내고, 첫 토큰의 top_logprobs에서 선택지 라벨의
확률만 모아 정규화한다. 질문들은 스레드로 병렬 전송한다.

    export JEV_BASE_URL=http://llm-gateway.internal/v1
    export JEV_MODEL=qwen3.8
    export JEV_API_KEY=...            # 필요할 때만
    export JEV_INSECURE=1             # 사내 자체서명 인증서일 때만

    engine = SystemOneAPI()
    engine.system_one(state, {"urgent": noul("Does this convey urgency?")})
"""
from __future__ import annotations

import json
import math
import os
import re
import ssl
import string
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

LETTERS = string.ascii_uppercase
# Qwen 계열 chat template에서 thinking을 끈 assistant 시작부 (completions 모드에서 직접 붙인다)
QWEN_NO_THINK = "<think>\n\n</think>\n\n"


# --- 질문 빌더 (JS SDK의 noul/choice/score와 같은 모양) ---

def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, choices: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "choices": choices}


def score(instructions: str, scale: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "scale": scale}


# --- 프롬프트/답 구성 (로컬 jev_local.py와 공유) ---

def state_to_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


def build_options(q: dict) -> tuple[list[str], list, list[str]]:
    """(라벨, 답 이름, 프롬프트 표시문). 라벨은 단일 토큰이 되도록 A-Z / 1-9를 쓴다."""
    t = q["type"]
    if t == "noul":
        return ["A", "B"], ["yes", "no"], ["Yes", "No"]
    if t == "choice":
        names = list(q["choices"])
        if len(names) > len(LETTERS):
            raise ValueError(f"choice supports up to {len(LETTERS)} options")
        return (list(LETTERS[: len(names)]), names,
                [f"{n}: {d}" for n, d in q["choices"].items()])
    if t == "score":
        scale = q["scale"]
        if not 2 <= len(scale) <= 9:
            raise ValueError("score scale must have 2-9 levels")
        return [str(i + 1) for i in range(len(scale))], list(range(len(scale))), scale
    raise ValueError(f"unknown question type: {t}")


def build_user_content(state_text: str, q: dict, labels: list[str], shown: list[str]) -> str:
    # state를 맨 앞에 둬서 같은 state의 질문들이 서버 prefix cache를 공유하게 한다
    kind = "number" if q["type"] == "score" else "letter"
    return (
        f"State:\n{state_text}\n\n"
        f"Question: {q['instructions']}\n"
        + "\n".join(f"{l}. {s}" for l, s in zip(labels, shown))
        + f"\n\nAnswer with the option {kind} only."
    )


def build_answer(q: dict, names: list, probs: list[float], label_mass: float) -> dict:
    t = q["type"]
    if t == "noul":
        ans = {"noul": probs[0]}
    elif t == "choice":
        dist = dict(zip(names, probs))
        best = max(dist, key=dist.get)
        ans = {"choice": best, "confidence": dist[best], "probabilities": dist}
    else:
        n = len(names)
        expected = sum(i * p for i, p in enumerate(probs))
        ans = {"score": expected / (n - 1),  # 0~1로 정규화한 기대값
               "level": q["scale"][max(range(n), key=probs.__getitem__)],
               "distribution": dict(zip(q["scale"], probs))}
    # 라벨 토큰에 실린 원래 확률 합 — 낮으면 모델이 형식을 안 따른 것(신뢰도 낮음)
    ans["label_mass"] = label_mass
    return ans


def restricted_softmax(label_logprobs: list[float], temperature: float = 1.0) -> list[float]:
    finite = [lp for lp in label_logprobs if lp != -math.inf]
    if not finite:
        raise ValueError("none of the option labels appeared in top_logprobs")
    m = max(finite)
    ws = [math.exp((lp - m) / temperature) if lp != -math.inf else 0.0 for lp in label_logprobs]
    s = sum(ws)
    return [w / s for w in ws]


# --- 비교용: 일반 생성 방식 (모델이 JSON으로 답 + 자기 확신도를 쓰게 함) ---

def build_plain_prompt(state_text: str, questions: dict[str, dict]) -> str:
    lines = [f"State:\n{state_text}\n", "Answer every question below."]
    for k, q in questions.items():
        t = q["type"]
        if t == "noul":
            lines.append(f'- "{k}": {q["instructions"]} Answer "yes" or "no".')
        elif t == "choice":
            opts = "; ".join(f"{n} = {d}" for n, d in q["choices"].items())
            lines.append(f'- "{k}": {q["instructions"]} Options: {opts}. Answer with one option name.')
        else:
            lvls = "; ".join(f"{i + 1} = {s}" for i, s in enumerate(q["scale"]))
            lines.append(f'- "{k}": {q["instructions"]} Levels: {lvls}. Answer with the level number.')
    lines.append('\nReply with ONLY a JSON object, no other text, in this form: '
                 '{"<question key>": {"answer": <answer>, "confidence": <0-100>}, ...}')
    return "\n".join(lines)


def parse_plain_answers(text: str, questions: dict[str, dict]) -> tuple[dict, str | None]:
    """모델이 쓴 JSON을 파싱해 질문 타입별로 정규화. (answers, error)"""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e < s:
        return {}, "no JSON object in output"
    try:
        raw = json.loads(text[s:e + 1])
    except json.JSONDecodeError as ex:
        return {}, f"invalid JSON: {ex}"
    out = {}
    for k, q in questions.items():
        item = raw.get(k)
        if isinstance(item, dict):
            ans, conf = item.get("answer"), item.get("confidence")
        else:
            ans, conf = item, None
        norm = None
        if q["type"] == "noul" and isinstance(ans, (str, bool)):
            a = str(ans).strip().lower()
            norm = "yes" if a in ("yes", "true", "y") else "no" if a in ("no", "false", "n") else None
        elif q["type"] == "choice" and isinstance(ans, str):
            lut = {n.lower(): n for n in q["choices"]}
            norm = lut.get(ans.strip().lower())
        elif q["type"] == "score":
            try:
                i = int(str(ans).strip())
                norm = q["scale"][i - 1] if 1 <= i <= len(q["scale"]) else None
            except ValueError:
                lut = {l.lower(): l for l in q["scale"]}
                norm = lut.get(str(ans).strip().lower())
        try:
            conf = float(conf) / 100 if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        out[k] = {"answer": norm, "raw_answer": ans, "self_confidence": conf}
    return out, None


def jev_label(q: dict, ans: dict) -> str:
    """jev 답을 plain 답과 비교 가능한 값으로."""
    if q["type"] == "noul":
        return "yes" if ans["noul"] >= 0.5 else "no"
    if q["type"] == "choice":
        return ans["choice"]
    return ans["level"]


def split_thinking(content: str, reasoning: str = "") -> tuple[str, str]:
    """서버에 reasoning parser가 없으면 <think>...</think>가 content에 섞여 온다."""
    m = re.search(r"<think>(.*?)</think>", content, re.S)
    if m:
        return content[m.end():].strip(), (reasoning or m.group(1)).strip()
    if "</think>" in content:  # template이 <think>를 프롬프트에 넣은 경우 닫는 태그만 온다
        r, c = content.split("</think>", 1)
        return c.strip(), (reasoning or r).strip()
    return content.strip(), reasoning.strip()


def _norm_token(t: str) -> str:
    # 서버에 따라 " A", "ĠA"(byte-level BPE), "▁A"(sentencepiece)로 올 수 있다
    return t.replace("Ġ", " ").replace("▁", " ").strip()


# --- HTTP ---

class APIError(RuntimeError):
    pass


def http_json(method: str, url: str, body: dict | None = None, *, api_key: str | None = None,
              timeout: float = 60.0, insecure: bool = False) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    ctx = ssl._create_unverified_context() if insecure else None
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise APIError(f"HTTP {e.code} {url}: {e.read().decode(errors='replace')[:500]}") from None


# --- 엔진 ---

class SystemOneAPI:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, *, mode: str = "chat", top_logprobs: int = 20,
                 disable_thinking: bool = True, timeout: float = 60.0, max_workers: int = 8,
                 insecure: bool | None = None, extra_body: dict | None = None):
        """mode="chat": /chat/completions + chat_template_kwargs로 thinking 끔.
        mode="completions": /completions에 Qwen ChatML 프롬프트를 직접 만들어 보냄
        (게이트웨이가 chat_template_kwargs를 막을 때의 대안)."""
        self.base_url = (base_url or os.environ["JEV_BASE_URL"]).rstrip("/")
        self.model = model or os.environ["JEV_MODEL"]
        self.api_key = api_key if api_key is not None else os.environ.get("JEV_API_KEY")
        self.insecure = insecure if insecure is not None else os.environ.get("JEV_INSECURE") == "1"
        if mode not in ("chat", "completions"):
            raise ValueError("mode must be 'chat' or 'completions'")
        self.mode = mode
        self.top_logprobs = top_logprobs
        self.disable_thinking = disable_thinking
        self.timeout = timeout
        self.max_workers = max_workers
        self.extra_body = extra_body or {}

    def top_tokens(self, content: str) -> tuple[dict[str, float], str, int]:
        """user content 하나를 보내고 (정규화 토큰 -> logprob, 첫 생성 토큰, prompt 토큰 수)."""
        if self.mode == "chat":
            body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                    "max_tokens": 1, "temperature": 0, "logprobs": True,
                    "top_logprobs": self.top_logprobs}
            if self.disable_thinking:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            body.update(self.extra_body)
            r = http_json("POST", f"{self.base_url}/chat/completions", body, api_key=self.api_key,
                          timeout=self.timeout, insecure=self.insecure)
            c = r["choices"][0]
            lp = (c.get("logprobs") or {}).get("content") or []
            if not lp:
                raise APIError("response has no logprobs — gateway/model may not support them")
            first = lp[0]["token"]
            pairs = [(t["token"], t["logprob"]) for t in lp[0].get("top_logprobs") or []]
        else:
            prompt = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
            if self.disable_thinking:
                prompt += QWEN_NO_THINK
            body = {"model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                    "logprobs": self.top_logprobs}
            body.update(self.extra_body)
            r = http_json("POST", f"{self.base_url}/completions", body, api_key=self.api_key,
                          timeout=self.timeout, insecure=self.insecure)
            c = r["choices"][0]
            lp = c.get("logprobs") or {}
            if not lp.get("top_logprobs"):
                raise APIError("response has no logprobs — gateway/model may not support them")
            first = lp["tokens"][0]
            pairs = list(lp["top_logprobs"][0].items())

        # " A"와 "A"처럼 같은 라벨로 정규화되는 토큰은 확률을 합친다
        merged: dict[str, float] = {}
        for tok, logprob in pairs:
            k = _norm_token(tok)
            merged[k] = math.log(math.exp(merged[k]) + math.exp(logprob)) if k in merged else logprob
        return merged, first, (r.get("usage") or {}).get("prompt_tokens", 0)

    def chat(self, messages: list[dict], *, max_tokens: int = 1024, thinking: bool = False,
             temperature: float = 0.0) -> dict:
        """일반 텍스트 생성. thinking 제어는 chat_template_kwargs로 (mode='chat'일 때만 전송)."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature}
        if self.mode == "chat":
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        body.update(self.extra_body)
        t0 = time.perf_counter()
        r = http_json("POST", f"{self.base_url}/chat/completions", body, api_key=self.api_key,
                      timeout=max(self.timeout, 300), insecure=self.insecure)
        latency_ms = (time.perf_counter() - t0) * 1000
        c = r["choices"][0]
        msg = c.get("message") or {}
        content, reasoning = split_thinking(msg.get("content") or "",
                                            msg.get("reasoning_content") or msg.get("reasoning") or "")
        u = r.get("usage") or {}
        return {"content": content, "reasoning": reasoning, "finish_reason": c.get("finish_reason"),
                "usage": {"input_tokens": u.get("prompt_tokens", 0),
                          "output_tokens": u.get("completion_tokens", 0),
                          "latency_ms": latency_ms}}

    def plain_decide(self, state: Any, questions: dict[str, dict], *, thinking: bool = False,
                     max_tokens: int | None = None) -> dict:
        """비교 기준: 모든 질문을 한 프롬프트에 넣고 모델이 JSON으로 답을 '생성'하게 한다."""
        prompt = build_plain_prompt(state_to_text(state), questions)
        r = self.chat([{"role": "user", "content": prompt}], thinking=thinking,
                      max_tokens=max_tokens or (4096 if thinking else 512))
        answers, err = parse_plain_answers(r["content"], questions)
        out = {"answers": answers, "usage": r["usage"], "finish_reason": r["finish_reason"],
               "output": r["content"][:2000]}
        if r["reasoning"]:
            out["reasoning_chars"] = len(r["reasoning"])
        if err:
            out["parse_error"] = err
        return out

    def compare(self, state: Any, questions: dict[str, dict], *, thinking: bool = False,
                temperature: float = 1.0) -> dict:
        """같은 입력을 jev 방식과 일반 생성 방식으로 각각 판단해 나란히 반환."""
        j = self.system_one(state, questions, temperature=temperature)
        p = self.plain_decide(state, questions, thinking=thinking)
        rows, agree = {}, 0
        for k, q in questions.items():
            ja = j["answers"][k]
            pa = p["answers"].get(k, {})
            jl = jev_label(q, ja)
            jconf = ja["noul"] if q["type"] == "noul" else ja.get("confidence",
                                                               max(ja.get("distribution", {0: 0}).values()))
            if q["type"] == "noul" and jl == "no":
                jconf = 1 - jconf
            same = jl == pa.get("answer")
            agree += same
            rows[k] = {"jev": jl, "jev_prob": jconf, "plain": pa.get("answer"),
                       "plain_self_confidence": pa.get("self_confidence"), "agree": same}
        return {"comparison": rows, "agreement": f"{agree}/{len(questions)}",
                "cost": {"jev": j["usage"], "plain": p["usage"]},
                "jev": j["answers"], "plain": p}

    def _ask(self, state_text: str, q: dict, temperature: float) -> tuple[dict, int]:
        labels, names, shown = build_options(q)
        top, first, n_tok = self.top_tokens(build_user_content(state_text, q, labels, shown))
        label_lps = [top.get(l, -math.inf) for l in labels]
        if all(lp == -math.inf for lp in label_lps):
            raise APIError(f"no option label in top_logprobs (first token={first!r}); "
                           "thinking may be on — check disable_thinking / mode='completions'")
        label_mass = sum(math.exp(lp) for lp in label_lps if lp != -math.inf)
        ans = build_answer(q, names, restricted_softmax(label_lps, temperature), label_mass)
        ans["first_token"] = first  # "<think>" 등이 나오면 thinking이 안 꺼진 것
        return ans, n_tok

    def system_one(self, state: Any, questions: dict[str, dict], *,
                   temperature: float = 1.0) -> dict:
        state_text = state_to_text(state)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(questions))) as ex:
            futs = {k: ex.submit(self._ask, state_text, q, temperature) for k, q in questions.items()}
            results = {k: f.result() for k, f in futs.items()}
        return {"answers": {k: a for k, (a, _) in results.items()},
                "usage": {"input_tokens": sum(n for _, n in results.values()),
                          "output_tokens": len(questions),  # 질문당 1토큰
                          "questions": len(questions),
                          "latency_ms": (time.perf_counter() - t0) * 1000}}
