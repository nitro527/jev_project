"""jev vs plain(일반 생성) 정량 비교 — 라벨 데이터셋 60건 (감성/메시지 분류/사실 판단 x 20).

    (JEV_* 환경변수 설정 후) python bench/bench_compare.py [--out results.json] [--thinking]

데이터는 공개 가능한 일반 문장만 사용한다 (사내 데이터 아님).
기준 결과: bench/results_qwen3.5-4b_local.json (집 PC, Qwen3.5-4B, thinking off).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev_api import SystemOneAPI, parse_plain_answers  # noqa: E402

TASKS = {
    "sentiment": {
        "key": "sentiment",
        "q": {"type": "choice", "instructions": "Overall sentiment of this review?",
              "choices": {"positive": "mostly positive", "negative": "mostly negative",
                          "mixed": "clearly contains both positive and negative points"}},
        "data": [
            ("Absolutely love it. Works perfectly and arrived early!", "positive"),
            ("Broke after two days. Complete waste of money.", "negative"),
            ("Great sound quality, but the battery dies in an hour.", "mixed"),
            ("최고예요! 재구매 의사 100%입니다.", "positive"),
            ("배송도 늦고 제품도 불량이라 너무 실망했어요.", "negative"),
            ("디자인은 예쁜데 사이즈가 너무 작게 나왔어요.", "mixed"),
            ("Customer service was rude and never solved my issue.", "negative"),
            ("Exactly as described, super comfortable, highly recommend.", "positive"),
            ("The food was delicious but we waited 90 minutes for it.", "mixed"),
            ("가성비 최고, 부모님도 만족하셨어요.", "positive"),
            ("냄새가 너무 심해서 환불 요청했습니다.", "negative"),
            ("화면은 선명한데 발열이 좀 심하네요.", "mixed"),
            ("Not worth it. Cheap plastic and it squeaks.", "negative"),
            ("My kids adore this toy, best purchase this year.", "positive"),
            ("Setup was a nightmare, though once running it's fast.", "mixed"),
            ("포장 꼼꼼하고 품질 좋아요. 강추!", "positive"),
            ("두 번 썼는데 고장났어요. 다신 안 삽니다.", "negative"),
            ("맛은 있는데 양이 너무 적어요.", "mixed"),
            ("Terrible fit, the seams ripped on first wash.", "negative"),
            ("Five stars. Beautiful craftsmanship.", "positive"),
        ],
    },
    "message": {
        "key": "message_type",
        "q": {"type": "choice", "instructions": "Classify this message.",
              "choices": {"legit": "legitimate normal message",
                          "spam": "unsolicited marketing / advertising",
                          "phishing": "scam trying to steal credentials, money or personal data"}},
        "data": [
            ("Your package has shipped and will arrive Tuesday.", "legit"),
            ("URGENT: account suspended. Verify your password at http://paypa1-secure.co/login", "phishing"),
            ("Hey, are we still on for lunch tomorrow?", "legit"),
            ("Invoice #2231 attached for September services. Payment due in 30 days.", "legit"),
            ("Your bank detected unusual activity. Reply with your SSN to confirm identity.", "phishing"),
            ("Reminder: team standup moved to 10:30.", "legit"),
            ("Send 0.1 BTC to this wallet and we'll send back 0.2 BTC, guaranteed!", "phishing"),
            ("Limited offer: 90% off designer watches, free shipping worldwide!", "spam"),
            ("Big summer sale at MegaMart! All shoes 50% off this weekend only.", "spam"),
            ("[국외발신] 택배 주소 불일치, 확인하세요 http://cj-dlvr.xyz", "phishing"),
            ("엄마 나 폰 고장나서 친구폰으로 연락해. 급하게 문화상품권 좀 사서 번호 보내줘", "phishing"),
            ("(광고) 신규 가입시 치킨 쿠폰 증정! 수신거부 080-123-4567", "spam"),
            ("내일 회의 자료 공유드립니다. 확인 부탁드려요.", "legit"),
            ("Subscribe now to our newsletter for weekly deals on gadgets!", "spam"),
            ("Your Netflix payment failed. Update your card details here: netf1ix-billing.com", "phishing"),
            ("Dentist appointment confirmed for Oct 3 at 2pm.", "legit"),
            ("(광고) 이번 주 한정 헬스장 회원권 50% 할인!", "spam"),
            ("주문하신 상품이 출고되었습니다. 운송장번호 1234-5678", "legit"),
            ("Hot singles in your area! Click to see photos.", "spam"),
            ("IRS notice: you owe back taxes. Pay immediately with gift cards to avoid arrest.", "phishing"),
        ],
    },
    "fact": {
        "key": "is_true",
        "q": {"type": "noul", "instructions": "Is the statement true?"},
        "data": [
            ("17 * 23 = 391", "yes"),
            ("The capital of Australia is Sydney.", "no"),
            ("Water boils at 100 degrees Celsius at sea level.", "yes"),
            ("144 is a prime number.", "no"),
            ("The Pacific is the largest ocean on Earth.", "yes"),
            ("A bat and ball cost $1.10; the bat costs $1 more. The ball costs $0.10.", "no"),
            ("대한민국의 수도는 서울이다.", "yes"),
            ("한글은 세종대왕이 창제했다.", "yes"),
            ("2의 10제곱은 1000이다.", "no"),
            ("Light travels faster than sound.", "yes"),
            ("Spiders are insects.", "no"),
            ("sqrt(169) = 13", "yes"),
            ("The Great Wall of China is visible from the Moon with the naked eye.", "no"),
            ("Python lists are immutable.", "no"),
            ("HTTP status code 404 means Not Found.", "yes"),
            ("If all cats are mammals and Tom is a cat, Tom is a mammal.", "yes"),
            ("0.1 + 0.2 == 0.3 evaluates to True in IEEE-754 double arithmetic.", "no"),
            ("지구에서 가장 높은 산은 K2이다.", "no"),
            ("A week has 168 hours.", "yes"),
            ("9.11 is greater than 9.9.", "no"),
        ],
    },
}


def lenient_answer(output: str, key: str, qs: dict):
    """모델이 JSON 키를 질문 키 대신 질문 문장 등으로 쓴 경우, 값이 하나뿐이면 그 값을 답으로 본다."""
    try:
        d = json.loads(output[output.find("{"):output.rfind("}") + 1])
        if isinstance(d, dict) and len(d) == 1:
            return parse_plain_answers(json.dumps({key: next(iter(d.values()))}), qs)[0][key]["answer"]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results.json")
    ap.add_argument("--thinking", action="store_true", help="plain 쪽 thinking 켜기 (매우 느림)")
    args = ap.parse_args()

    api = SystemOneAPI()
    print(f"endpoint={api.base_url} model={api.model} mode={api.mode} thinking={args.thinking}\n")
    rows = []
    for task, spec in TASKS.items():
        key, qs = spec["key"], {spec["key"]: spec["q"]}
        for i, (text, gold) in enumerate(spec["data"]):
            r = api.compare(text, qs, thinking=args.thinking)
            c = r["comparison"][key]
            plain_lenient = c["plain"] if c["plain"] is not None else lenient_answer(r["plain"]["output"], key, qs)
            rows.append({
                "task": task, "i": i, "text": text, "gold": gold,
                "jev": c["jev"], "jev_prob": c["jev_prob"], "label_mass": r["jev"][key]["label_mass"],
                "plain": c["plain"], "plain_lenient": plain_lenient, "plain_conf": c["plain_self_confidence"],
                "jev_ms": r["cost"]["jev"]["latency_ms"], "plain_ms": r["cost"]["plain"]["latency_ms"],
                "jev_in": r["cost"]["jev"]["input_tokens"], "plain_in": r["cost"]["plain"]["input_tokens"],
                "plain_out": r["cost"]["plain"]["output_tokens"],
                "parse_error": r["plain"].get("parse_error"), "plain_raw": r["plain"]["output"],
            })
            print(f"{task:9} {i:2} gold={gold:8} jev={c['jev']:8} ({c['jev_prob']:.2f}) "
                  f"plain={c['plain']} / lenient={plain_lenient}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    print("\n=== 요약 ===")
    for t in list(TASKS) + ["ALL"]:
        rs = [r for r in rows if t == "ALL" or r["task"] == t]
        print(f"{t:9} jev {sum(r['jev'] == r['gold'] for r in rs):2}/{len(rs)}  "
              f"plain(strict) {sum(r['plain'] == r['gold'] for r in rs):2}  "
              f"plain(lenient) {sum(r['plain_lenient'] == r['gold'] for r in rs):2}")
    med = lambda k: statistics.median(r[k] for r in rows)  # noqa: E731
    print(f"latency median: jev {med('jev_ms'):.0f}ms, plain {med('plain_ms'):.0f}ms")
    print(f"tokens median: jev in {med('jev_in'):.0f}/out 1, plain in {med('plain_in'):.0f}/out {med('plain_out'):.0f}")
    print(f"label_mass min: {min(r['label_mass'] for r in rows):.3f}")
    low = [r for r in rows if r["jev_prob"] < 0.7]
    print(f"jev prob < 0.7: {len(low)}건 — " + ", ".join(f"{r['text'][:25]!r}({r['jev_prob']:.2f})" for r in low))
    wrong = [r for r in rows if r["plain_lenient"] != r["gold"]]
    print(f"plain(lenient) 오답: {len(wrong)}건 — "
          + ", ".join(f"{r['text'][:25]!r} plain={r['plain_lenient']}(self {r['plain_conf']}) "
                      f"jev={r['jev']}({r['jev_prob']:.2f})" for r in wrong))
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
