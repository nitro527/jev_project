"""Jev 스타일 판단 엔진을 MCP(stdio) 도구로 노출한다. 표준 라이브러리만 사용.

Claude Code / opencode 같은 에이전트가 "정해진 선택지 중 판단 + 확률"이 필요한
대량·반복 판단(로그 윈도 분류, 트리아지 등)을 사내 LLM에 싸게 위임하는 용도.

환경변수: JEV_BASE_URL, JEV_MODEL, JEV_API_KEY, JEV_INSECURE=1,
         JEV_MODE(chat|completions), JEV_TOP_LOGPROBS(20), JEV_DISABLE_THINKING(1),
         JEV_TEMPERATURE(1.0, 보정용), JEV_BATCH_WORKERS(4)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jev_api import SystemOneAPI, build_options  # noqa: E402

SERVER_INFO = {"name": "jev", "version": "0.1.0"}
SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_BATCH = 200

QUESTIONS_SCHEMA = {
    "type": "object",
    "description": (
        "Map of question_key -> question. Keys are identifiers only (never shown to the model). "
        "All questions are answered independently and in parallel."),
    "additionalProperties": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["noul", "choice", "score"],
                     "description": "noul = yes/no (returns P(yes)); choice = pick one option; "
                                    "score = rate on an ordered scale"},
            "instructions": {"type": "string", "description": "The question, in plain language."},
            "choices": {"type": "object", "additionalProperties": {"type": "string"},
                        "description": "choice only: option_name -> description (2-26 options). "
                                       "Include an 'other'/'normal' option when none may fit."},
            "scale": {"type": "array", "items": {"type": "string"},
                      "description": "score only: 2-9 level descriptions, lowest first."},
        },
        "required": ["type", "instructions"],
    },
}

RESULT_NOTE = (
    "Result per question: noul -> {noul: P(yes)}; choice -> {choice, confidence, probabilities}; "
    "score -> {score: expected value normalized 0-1, level, distribution}. Every answer has "
    "label_mass (prob. the model put on valid option tokens; < 0.9 means it did not follow the format, "
    "treat as unreliable). Probabilities are not calibrated: use them for ranking and confidence gating "
    "(e.g. auto-accept >= 0.9, discard <= 0.1, escalate the middle to a stronger model), not as exact odds.")

TOOLS = [
    {
        "name": "jev_decide",
        "description": (
            "Fast, cheap typed decision by the internal LLM (Jev / 'System One' style): no text is "
            "generated; each question is answered from a single next-token distribution restricted to "
            "the allowed options, so you get an answer plus probabilities in ~0.1-0.5s. Use for "
            "classification, routing, yes/no checks and scoring over a given state (e.g. a log window). "
            "Do not use for open-ended reasoning or text generation. " + RESULT_NOTE),
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {"type": "string",
                          "description": "The input to judge (log lines, ticket text, JSON text...)."},
                "questions": QUESTIONS_SCHEMA,
                "temperature": {"type": "number", "description": "Softmax temperature over options "
                                "(>1 flattens overconfident probabilities). Default from server config."},
            },
            "required": ["state", "questions"],
        },
    },
    {
        "name": "jev_decide_batch",
        "description": (
            "Same as jev_decide, but applies the same questions to many states in parallel "
            f"(max {MAX_BATCH}). Use to triage/map over many log windows or records at once, then "
            "inspect only the flagged or low-confidence ones yourself. " + RESULT_NOTE),
        "inputSchema": {
            "type": "object",
            "properties": {
                "states": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                           "maxItems": MAX_BATCH, "description": "Inputs to judge, one per item."},
                "questions": QUESTIONS_SCHEMA,
                "temperature": {"type": "number"},
            },
            "required": ["states", "questions"],
        },
    },
    {
        "name": "jev_compare",
        "description": (
            "Experiment tool: judge the same state + questions two ways with the same internal LLM and "
            "return them side by side. (1) jev: one-token logprob decision per question (fast, "
            "probabilities). (2) plain: the model generates a JSON answer with a self-reported "
            "confidence, like normal LLM usage (optionally with thinking). Returns per-question answers, "
            "agreement, and latency/token cost of each. Use to evaluate whether jev-style decisions "
            "match normal generation on real data."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "description": "The input to judge."},
                "questions": QUESTIONS_SCHEMA,
                "thinking": {"type": "boolean",
                             "description": "Let the plain side think before answering (slower). Default false."},
                "temperature": {"type": "number"},
            },
            "required": ["state", "questions"],
        },
    },
    {
        "name": "llm_chat",
        "description": (
            "Plain text generation with the internal LLM (no jev restriction). Use for comparison or when "
            "a free-form answer from the internal model is needed."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "system": {"type": "string", "description": "Optional system prompt."},
                "thinking": {"type": "boolean", "description": "Enable thinking. Default false."},
                "max_tokens": {"type": "integer", "description": "Default 1024 (4096 with thinking)."},
            },
            "required": ["prompt"],
        },
    },
]

_engine: SystemOneAPI | None = None
_engine_lock = threading.Lock()
_write_lock = threading.Lock()


def log(msg: str) -> None:
    print(f"[jev-mcp] {msg}", file=sys.stderr, flush=True)


def engine() -> SystemOneAPI:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = SystemOneAPI(
                mode=os.environ.get("JEV_MODE", "chat"),
                top_logprobs=int(os.environ.get("JEV_TOP_LOGPROBS", "20")),
                disable_thinking=os.environ.get("JEV_DISABLE_THINKING", "1") == "1",
            )
            log(f"engine: {_engine.base_url} model={_engine.model} mode={_engine.mode}")
        return _engine


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def _validate(questions) -> None:
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    for k, q in questions.items():
        if not isinstance(q, dict) or "instructions" not in q:
            raise ValueError(f"question {k!r} needs 'type' and 'instructions'")
        if q.get("type") == "choice" and (not isinstance(q.get("choices"), dict) or len(q["choices"]) < 2):
            raise ValueError(f"question {k!r}: choice needs 'choices' with >= 2 options")
        if q.get("type") == "score" and not isinstance(q.get("scale"), list):
            raise ValueError(f"question {k!r}: score needs 'scale' list")
        build_options(q)  # 타입/개수 검증


def _temperature(args: dict) -> float:
    return float(args.get("temperature") or os.environ.get("JEV_TEMPERATURE", "1.0"))


def call_tool(name: str, args: dict) -> dict:
    if name == "llm_chat":
        msgs = ([{"role": "system", "content": args["system"]}] if args.get("system") else [])
        msgs.append({"role": "user", "content": args["prompt"]})
        thinking = bool(args.get("thinking"))
        r = engine().chat(msgs, thinking=thinking,
                          max_tokens=int(args.get("max_tokens") or (4096 if thinking else 1024)))
        r["reasoning"] = r["reasoning"][:4000]  # 컨텍스트 절약
        return _round(r)
    questions = args.get("questions")
    _validate(questions)
    t = _temperature(args)
    if name == "jev_decide":
        r = engine().system_one(args["state"], questions, temperature=t)
        return _round(r)
    if name == "jev_compare":
        return _round(engine().compare(args["state"], questions,
                                       thinking=bool(args.get("thinking")), temperature=t))
    if name == "jev_decide_batch":
        states = args.get("states")
        if not isinstance(states, list) or not states or len(states) > MAX_BATCH:
            raise ValueError(f"states must be a list of 1-{MAX_BATCH} items")
        eng = engine()
        workers = int(os.environ.get("JEV_BATCH_WORKERS", "4"))

        def one(i_s):
            i, s = i_s
            try:
                r = eng.system_one(s, questions, temperature=t)
                return {"index": i, "answers": r["answers"], "latency_ms": r["usage"]["latency_ms"]}
            except Exception as e:  # noqa: BLE001 — 건별 실패는 결과에 담고 계속
                return {"index": i, "error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(one, enumerate(states)))
        return _round({"results": results,
                       "errors": sum(1 for r in results if "error" in r),
                       "count": len(results)})
    raise ValueError(f"unknown tool: {name}")


# --- JSON-RPC over stdio ---

def send(msg: dict) -> None:
    with _write_lock:
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def handle_tool_call(req_id, params: dict) -> None:
    try:
        out = call_tool(params.get("name"), params.get("arguments") or {})
        send({"jsonrpc": "2.0", "id": req_id, "result": {
            "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}],
            "isError": False}})
    except Exception as e:  # noqa: BLE001 — 도구 에러는 isError로 에이전트에게 돌려준다
        log(traceback.format_exc())
        send({"jsonrpc": "2.0", "id": req_id, "result": {
            "content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}})


def handle(msg: dict) -> None:
    method, req_id, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if req_id is None:  # notification (initialized, cancelled 등) — 응답 없음
        return
    if method == "initialize":
        v = params.get("protocolVersion")
        send({"jsonrpc": "2.0", "id": req_id, "result": {
            "protocolVersion": v if v in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0],
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO}})
    elif method == "ping":
        send({"jsonrpc": "2.0", "id": req_id, "result": {}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        # LLM 호출은 느리므로 별도 스레드 — 에이전트의 병렬 도구 호출을 막지 않는다
        # non-daemon: stdin이 닫혀도 진행 중인 호출은 응답을 보내고 끝난다
        threading.Thread(target=handle_tool_call, args=(req_id, params)).start()
    else:
        send({"jsonrpc": "2.0", "id": req_id,
              "error": {"code": -32601, "message": f"method not found: {method}"}})


def main() -> None:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    log("started")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        for m in msg if isinstance(msg, list) else [msg]:
            handle(m)


if __name__ == "__main__":
    main()
