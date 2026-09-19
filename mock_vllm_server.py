"""로컬 검증용: Qwen3.5-4B를 vLLM의 OpenAI 호환 응답 형식으로 흉내 내는 서버.

max_tokens=1 + logprobs(판단) 요청과 chat 일반 생성을 지원한다 (jev_api / probe_endpoint 테스트용).
chat은 `stream: true`(SSE, 토큰마다 한 조각)와 vLLM 확장 `min_tokens` / `ignore_eos`도 받는다 (지연 측정용).
동시 요청은 락으로 한 번에 하나씩 처리한다(vLLM과 달리 배칭하지 않음).
    python mock_vllm_server.py --port 8000
"""
import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_LOGPROBS = 20  # vLLM 기본 max_logprobs

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="models/Qwen3.5-4B")
ap.add_argument("--served-name", default="qwen-local")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
lock = threading.Lock()


@torch.inference_mode()
def next_token(prompt: str, k: int):
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with lock:
        logits = model(**ids).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    vals, idx = lp.topk(max(k, 1))
    top = [(tok.decode([i]), v) for v, i in zip(vals.tolist(), idx.tolist())]
    return top[0], top[:k], ids.input_ids.shape[1]


@torch.inference_mode()
def generate(prompt: str, max_new: int, temperature: float):
    ids = tok(prompt, return_tensors="pt").to("cuda")
    kw = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
    with lock:
        out = model.generate(**ids, max_new_tokens=max_new, **kw)
    new = out[0, ids.input_ids.shape[1]:]
    finish = "stop" if len(new) < max_new else "length"
    return tok.decode(new, skip_special_tokens=True), ids.input_ids.shape[1], len(new), finish


def _eos_ids() -> set[int]:
    e = model.generation_config.eos_token_id
    ids = set(e if isinstance(e, list) else [e]) if e is not None else set()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int):
        ids.add(im_end)
    return ids


EOS_IDS = _eos_ids()


@torch.inference_mode()
def stream_tokens(prompt: str, max_new: int, min_new: int = 0, ignore_eos: bool = False):
    """토큰을 하나씩 직접 생성해 (텍스트, 입력 토큰 수)를 yield한다 (greedy, KV 캐시 사용)."""
    ids = tok(prompt, return_tensors="pt").to("cuda")
    n_in = ids.input_ids.shape[1]
    with lock:
        out = model(**ids, use_cache=True)
        past = out.past_key_values
        for i in range(max_new):
            logits = out.logits[0, -1]
            if i < min_new or ignore_eos:  # 끝내지 못하게 EOS를 막는다
                logits = logits.clone()
                logits[list(EOS_IDS)] = -float("inf")
            nxt = int(logits.argmax())
            if nxt in EOS_IDS:
                return
            yield tok.decode([nxt]), n_in
            if i + 1 < max_new:
                out = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
                past = out.past_key_values


def usage(n, m=1):
    return {"prompt_tokens": n, "completion_tokens": m, "total_tokens": n + m}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def stream_chat(self, prompt: str, max_tokens: int, body: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())

        def emit(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        def chunk(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": args.served_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        emit(chunk({"role": "assistant", "content": ""}))
        n_in, m = 0, 0
        for text, n_in in stream_tokens(prompt, max_tokens, body.get("min_tokens") or 0,
                                        bool(body.get("ignore_eos"))):
            m += 1
            emit(chunk({"content": text}))
        emit(chunk({}, "length" if m >= max_tokens else "stop"))
        if (body.get("stream_options") or {}).get("include_usage"):
            emit({"id": cid, "object": "chat.completion.chunk", "created": created, "model": args.served_name,
                  "choices": [], "usage": usage(n_in, m)})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": args.served_name, "object": "model"}]})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model") != args.served_name:
            return self.send(404, {"error": {"message": f"model {body.get('model')} not found"}})
        if self.path == "/v1/chat/completions":
            k = body.get("top_logprobs") or 0
            if k > MAX_LOGPROBS:
                return self.send(400, {"error": {"message": f"top_logprobs must be <= {MAX_LOGPROBS}"}})
            kwargs = body.get("chat_template_kwargs") or {}
            prompt = tok.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=kwargs.get("enable_thinking", True))
            max_tokens = body.get("max_tokens") or 1024
            if body.get("stream"):  # SSE 스트리밍 (logprobs 미지원)
                return self.stream_chat(prompt, max_tokens, body)
            if max_tokens > 1 and (body.get("min_tokens") or body.get("ignore_eos")):
                # 출력 길이를 고정해야 하는 측정 요청: 스트리밍과 같은 경로로 생성
                parts = list(stream_tokens(prompt, max_tokens, body.get("min_tokens") or 0,
                                           bool(body.get("ignore_eos"))))
                n = parts[0][1] if parts else len(tok(prompt).input_ids)
                return self.send(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                    "created": int(time.time()), "model": args.served_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(p for p, _ in parts)},
                                 "logprobs": None, "finish_reason": "length" if len(parts) >= max_tokens else "stop"}],
                    "usage": usage(n, len(parts))})
            if max_tokens > 1:  # 일반 생성 (logprobs 미지원)
                text, n, m, finish = generate(prompt, max_tokens, body.get("temperature", 0) or 0)
                return self.send(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                    "created": int(time.time()), "model": args.served_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "logprobs": None, "finish_reason": finish}],
                    "usage": usage(n, m)})
            (t, lp), top, n = next_token(prompt, k)
            entry = lambda tt, ll: {"token": tt, "logprob": ll, "bytes": list(tt.encode())}  # noqa: E731
            logprobs = ({"content": [{**entry(t, lp), "top_logprobs": [entry(a, b) for a, b in top] if k else []}]}
                        if body.get("logprobs") else None)
            return self.send(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                "model": args.served_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": t},
                             "logprobs": logprobs, "finish_reason": "length"}],
                "usage": usage(n)})
        if self.path == "/v1/completions":
            k = body.get("logprobs")
            if k is not None and k > MAX_LOGPROBS:
                return self.send(400, {"error": {"message": f"logprobs must be <= {MAX_LOGPROBS}"}})
            (t, lp), top, n = next_token(body["prompt"], k or 0)
            logprobs = ({"tokens": [t], "token_logprobs": [lp], "top_logprobs": [dict(top)], "text_offset": [0]}
                        if k is not None else None)
            return self.send(200, {
                "id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion", "created": int(time.time()),
                "model": args.served_name,
                "choices": [{"index": 0, "text": t, "logprobs": logprobs, "finish_reason": "length"}],
                "usage": usage(n)})
        self.send(404, {"error": "not found"})


print(f"serving {args.served_name} on http://127.0.0.1:{args.port}/v1", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
