"""thinking을 끈 1토큰 판단으로도 '단순 답' 이상을 하는 패턴 데모.

  [1] 추출      : 후보(로그 줄)를 choice 선택지로 주고 근본 원인 줄 / 첫 증상 줄을 고르게 한다
  [2] 재정렬    : 문서마다 noul "질문 답에 도움 되나?"의 P(yes)로 정렬한다 (Qwen3-Reranker와 같은 원리)
  [3] 계층 분류 : 대분류를 고른 뒤 세부를 고른다 (선택지 26개 제한 우회 + 해석 가능)

    (JEV_* 환경변수 설정 후) python bench/patterns_demo.py

데이터는 공개 가능한 예시 문장이다.
집 PC(Qwen3.5-4B) 결과: 첫 증상 줄 line5(0.88), 원인 줄은 line4(0.51)/line2(0.40)로 갈려 애매함을 표시,
재정렬 0.95/0.80/0.41/0.00/0.00, 계층 분류 account(1.00) → login_session(1.00).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev_api import SystemOneAPI, choice, noul  # noqa: E402

LOG = [
    "10:00:01 INFO  api   request id=77 GET /orders",
    "10:00:01 INFO  db    pool size=20 active=20 waiting=14",
    "10:00:02 WARN  db    connection acquire took 4800ms",
    "10:00:06 ERROR db    could not acquire connection within 5000ms (pool exhausted)",
    "10:00:06 ERROR api   request id=77 failed: 500 Internal Server Error",
    "10:00:06 INFO  api   retrying request id=77",
    "10:00:07 ERROR api   request id=77 failed: 500 Internal Server Error",
]
QUERY = "How do I fix 'could not acquire connection' errors in my Java service?"
DOCS = [
    "Tuning HikariCP: increase maximumPoolSize and fix connection leaks with leakDetectionThreshold.",
    "Our cafeteria menu for next week includes pasta on Tuesday.",
    "Java 21 introduced virtual threads which change how blocking I/O scales.",
    "Pool exhaustion usually means connections are not returned; always close them in finally or try-with-resources.",
    "How to configure Nginx rate limiting for public APIs.",
]
TICKET = "After the last update the app logs me out every 5 minutes and I have to enter my 2FA code again."
SUBCATEGORIES = {"account": {"login_session": "being logged out, session expiry, cannot stay signed in",
                             "password_reset": "forgot or reset password",
                             "2fa_setup": "setting up or losing a 2FA device",
                             "profile": "changing name, email, avatar"}}


def main():
    api = SystemOneAPI()
    print(f"endpoint={api.base_url} model={api.model}")

    t = time.perf_counter()
    opts = {f"line{i + 1}": l for i, l in enumerate(LOG)}
    r = api.system_one("\n".join(LOG), {
        "root_cause_line": choice("Which log line shows the root cause of the failures (not a symptom)?", opts),
        "first_symptom_line": choice("Which log line is the first user-visible failure?", opts),
    })
    print(f"\n[1] 추출 ({(time.perf_counter() - t) * 1000:.0f}ms)")
    for k, a in r["answers"].items():
        top = sorted(a["probabilities"].items(), key=lambda x: -x[1])[:3]
        print(f"  {k}: {a['choice']} ({a['confidence']:.2f})  top3={[(n, round(p, 2)) for n, p in top]}")

    t = time.perf_counter()
    scores = []
    for d in DOCS:
        a = api.system_one(f"Query: {QUERY}\nDocument: {d}",
                           {"rel": noul("Does the document help answer the query?")})["answers"]["rel"]
        scores.append((a["noul"], d))
    print(f"\n[2] 재정렬 ({(time.perf_counter() - t) * 1000:.0f}ms, 문서 {len(DOCS)}개 순차)")
    for s, d in sorted(scores, reverse=True):
        print(f"  {s:.3f}  {d[:80]}")

    t = time.perf_counter()
    l1 = api.system_one(TICKET, {"area": choice("Which area is this ticket about?", {
        "billing": "payments, invoices, refunds", "account": "login, authentication, profile, security",
        "product": "features, bugs in app functionality", "other": "anything else"})})["answers"]["area"]
    line = f"[3] 계층 분류: L1={l1['choice']} ({l1['confidence']:.2f})"
    sub = SUBCATEGORIES.get(l1["choice"])
    if sub:
        l2 = api.system_one(TICKET, {"sub": choice("Which specific account issue is this?", sub)})["answers"]["sub"]
        line += f" → L2={l2['choice']} ({l2['confidence']:.2f})"
    print(f"\n{line}  ({(time.perf_counter() - t) * 1000:.0f}ms)")


if __name__ == "__main__":
    main()
