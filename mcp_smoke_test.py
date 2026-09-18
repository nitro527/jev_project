"""jev_mcp_server.py를 stdio로 띄워 MCP 핸드셰이크와 도구 호출을 검증한다.

    (JEV_* 환경변수 설정 후) python mcp_smoke_test.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

proc = subprocess.Popen([sys.executable, os.path.join(HERE, "jev_mcp_server.py")],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")


def rpc(i, method, params=None):
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def notify(method):
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
    proc.stdin.flush()


QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does this message convey urgency?"},
    "intent": {"type": "choice", "instructions": "What is the customer's main request?",
               "choices": {"refund": "The customer wants money returned.",
                           "technical_help": "The customer needs a bug fixed.",
                           "other": "None of the other options clearly fits."}},
    "frustration": {"type": "score", "instructions": "How frustrated does the customer appear?",
                    "scale": ["Calm and neutral", "Concerned but civil", "Very angry"]},
}

try:
    init = rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "smoke", "version": "0"}})
    print("initialize:", init["result"]["serverInfo"], init["result"]["protocolVersion"])
    notify("notifications/initialized")
    tools = rpc(2, "tools/list")["result"]["tools"]
    print("tools:", [t["name"] for t in tools])

    t = time.time()
    r = rpc(3, "tools/call", {"name": "jev_decide", "arguments": {
        "state": "Help! My payouts have been failing for 3 days.", "questions": QUESTIONS}})["result"]
    print(f"\njev_decide ({(time.time() - t) * 1000:.0f}ms) isError={r['isError']}")
    print(r["content"][0]["text"])

    t = time.time()
    r = rpc(4, "tools/call", {"name": "jev_decide_batch", "arguments": {
        "states": ["Where can I download my invoice?",
                   "I was charged twice, refund the duplicate NOW or I'm calling my bank!!",
                   "The export button crashes the app every time I click it."],
        "questions": QUESTIONS}})["result"]
    out = json.loads(r["content"][0]["text"])
    print(f"\njev_decide_batch ({(time.time() - t) * 1000:.0f}ms) errors={out['errors']}")
    for item in out["results"]:
        a = item["answers"]
        print(f"  [{item['index']}] urgent={a['urgent']['noul']:.2f} intent={a['intent']['choice']}"
              f"({a['intent']['confidence']:.2f}) frustration={a['frustration']['level']}")

    r = rpc(5, "tools/call", {"name": "jev_decide", "arguments": {
        "state": "x", "questions": {"bad": {"type": "choice", "instructions": "?"}}}})["result"]
    print("\nvalidation error case: isError =", r["isError"], "|", r["content"][0]["text"])
finally:
    proc.stdin.close()
    proc.wait(timeout=30)
