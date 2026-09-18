"""로컬 검증용: Qwen3.5-4B를 vLLM의 OpenAI 호환 응답 형식으로 흉내 내는 서버.

max_tokens=1 + logprobs 요청만 지원한다 (jev_api / probe_endpoint 테스트용).
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


def usage(n):
    return {"prompt_tokens": n, "completion_tokens": 1, "total_tokens": n + 1}


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
