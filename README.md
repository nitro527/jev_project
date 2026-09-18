# jev_project

사내 LLM(Qwen 등, OpenAI 호환 API)을 **Jev(TypeSafe "System One") 방식의 판단 엔진**으로 쓰고,
MCP 서버로 감싸 Claude Code / opencode에서 도구로 호출하기 위한 프로젝트.

텍스트를 생성하지 않고, 질문마다 `max_tokens=1` 요청을 보내 첫 토큰의 `logprobs`에서
선택지 라벨(A/B/C…, 1/2/3…)의 확률만 모아 정규화한다. 결과는 답 + 확률.

> ⚠️ 이 저장소는 public이다. 사내 엔드포인트 주소, API 키, 실제 로그는 절대 커밋하지 말 것.
> 실제 설정은 `.mcp.json` / `opencode.json`(gitignore됨)이나 환경변수에만 둔다.

## 파일

| 파일 | 용도 | 의존성 |
|---|---|---|
| `jev_api.py` | 핵심 엔진 `SystemOneAPI` + 질문 빌더 `noul/choice/score` | 표준 라이브러리 |
| `jev_mcp_server.py` | MCP stdio 서버 (`jev_decide`, `jev_decide_batch`) | 표준 라이브러리 |
| `probe_endpoint.py` | 사내 엔드포인트가 이 방식을 지원하는지 점검 | 표준 라이브러리 |
| `mcp_smoke_test.py` | MCP 서버를 stdio로 띄워 핸드셰이크/호출 검증 | 표준 라이브러리 |
| `bench/bench_compare.py` | jev vs 일반 생성 60건 정량 비교 (기준 결과: `bench/results_qwen3.5-4b_local.json`) | 표준 라이브러리 |
| `bench/improve_eval.py`, `bench/datasets.py` | 개선 기법(few-shot, 보정, 순서 섞기, cascade) 비교 — 공개 실제 라벨 데이터(BGL 로그, BoolQ, AG News) | 표준 라이브러리 (데이터 받을 때만 pandas) |
| `docs/TECHNIQUE_GUIDE.md` | **문제 유형별 기법 선택 가이드** (쉬운 설명) | – |
| `docs/IMPROVE_AND_BENCHMARK.md` | 개선 기법 조사·실험 결과·벤치마크 설계 | – |
| `docs/HANDOFF.md` | **인수인계 문서** — 설계 이유, 실패/해결 기록, 벤치마크, 사내 적용 절차, TODO | – |
| `jev_local.py` | (집 PC) transformers로 전체 logits를 직접 읽는 기준 구현 | torch, transformers |
| `mock_vllm_server.py` | (집 PC) 로컬 모델로 vLLM 응답 형식을 흉내 내는 테스트 서버 | torch, transformers |

회사에서는 위 4개 표준 라이브러리 파일만 있으면 된다 (Python 3.10+).

## 회사에서 할 일

### 1. 엔드포인트 점검
```bash
# Windows: set 대신 PowerShell이면 $env:JEV_BASE_URL="..."
export JEV_BASE_URL=http://<사내-게이트웨이>/v1
export JEV_MODEL=<모델 이름, 예: qwen3.8>
export JEV_API_KEY=<필요시>
export JEV_INSECURE=1        # 자체서명 인증서일 때만
python probe_endpoint.py
```
확인 항목:
1. `/models`에 모델이 있는지
2. chat 응답에 `logprobs`가 통과되는지 ← **가장 중요**. 게이트웨이가 제거하면 이 방식은 불가
3. 기본 thinking 여부 / `chat_template_kwargs`로 끌 수 있는지
4. `top_logprobs` 최대 허용치 (vLLM 기본 20)
5. `/completions` 대안 경로 (게이트웨이가 `chat_template_kwargs`를 막을 때)
6. end-to-end 결과와 `label_mass`
7. 지연시간

마지막에 추천 설정(`mode`, `top_logprobs`)이 출력된다. `mode=completions`가 추천되면 `JEV_MODE=completions`로 설정.

프록시 환경이면 `NO_PROXY`에 게이트웨이 호스트를 넣어야 할 수 있다 (urllib은 `HTTP(S)_PROXY`를 따른다).

### 2. MCP 서버 검증
```bash
python mcp_smoke_test.py
```

### 3. 에이전트에 연결

**Claude Code** — CLI로 등록:
```bash
claude mcp add jev -e JEV_BASE_URL=http://<게이트웨이>/v1 -e JEV_MODEL=<모델> -- python /path/to/jev_project/jev_mcp_server.py
```
또는 `.mcp.json.example`을 프로젝트 루트에 `.mcp.json`으로 복사 (환경변수 `${JEV_BASE_URL}` 확장 사용).

**opencode** — `opencode.json.example`을 참고해 `opencode.json`의 `mcp` 항목에 추가.

### 설정 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `JEV_BASE_URL` | (필수) | OpenAI 호환 API base (`.../v1`) |
| `JEV_MODEL` | (필수) | 모델 이름 |
| `JEV_API_KEY` | 없음 | Bearer 토큰 |
| `JEV_INSECURE` | `0` | `1`이면 TLS 검증 끔 |
| `JEV_MODE` | `chat` | `chat` 또는 `completions` |
| `JEV_TOP_LOGPROBS` | `20` | 서버 허용치 이하로 |
| `JEV_DISABLE_THINKING` | `1` | Qwen thinking 끄기 |
| `JEV_TEMPERATURE` | `1.0` | 선택지 softmax 온도 (보정용, >1이면 과신 완화) |
| `JEV_BATCH_WORKERS` | `4` | `jev_decide_batch` 동시 처리 state 수 |

## MCP 도구

- **`jev_decide`** `{state, questions, temperature?}` — state 하나에 여러 질문을 병렬로.
- **`jev_decide_batch`** `{states[], questions, temperature?}` — 같은 질문을 최대 200개 state에. 로그 윈도 대량 트리아지용.
- **`jev_compare`** `{state, questions, thinking?, temperature?}` — 같은 입력을 ① jev(1토큰 logprob) ② 일반 생성
  (모델이 JSON으로 답 + 자기 확신도를 씀, thinking 선택)으로 각각 판단해 답/일치 여부/지연/토큰을 나란히 반환. 실험용.
- **`llm_chat`** `{prompt, system?, thinking?, max_tokens?}` — 제한 없는 일반 텍스트 생성.

질문 형식 (Jev API와 동일한 모양):
```json
{
  "failure_type": {"type": "choice", "instructions": "What is the primary failure pattern?",
                   "choices": {"rach_fail": "PRACH/RAR failure", "rlf": "Radio link failure",
                               "normal": "No significant issue"}},
  "needs_review": {"type": "noul", "instructions": "Should an engineer look at this window?"},
  "severity":     {"type": "score", "instructions": "How severe is the issue?",
                   "scale": ["No impact", "Minor", "Degraded", "Connection lost"]}
}
```
답: `noul → {noul: P(yes)}`, `choice → {choice, confidence, probabilities}`,
`score → {score: 0~1 기대값, level, distribution}`. 공통으로 `label_mass`, `first_token`.

## 로컬(RTX 4070 SUPER, Qwen3.5-4B)에서 확인한 것

- 선택지 라벨로 제한한 forward 1회: 질문당 약 70~80ms, 질문 3개 병렬 약 200~260ms.
- 형식 준수는 매우 좋음: `label_mass`가 대부분 0.99 이상. **0.9 미만이면 형식 이탈이니 신뢰하지 말 것.**
- **thinking이 켜져 있으면 동작하지 않는다**: 첫 토큰이 `<think>`/`Thinking`이 되어 라벨이 top_logprobs에 없음.
  `enable_thinking=False`(chat) 또는 빈 `<think></think>` prefill(completions)이 필수.
- chat 모드와 completions(ChatML 직접 구성) 모드의 결과가 동일함을 확인.
- **확률은 순위로는 쓸 만하지만 보정되어 있지 않다.** 예/아니오 질문에서 "예" 쪽으로 치우침
  (정상 케이스도 0.56~0.71). 0.5 기준은 부정확했고, 문제/정상의 순서는 정확히 갈렸다.
  → 실제 데이터 수십 건으로 기준값(예: 0.9)과 `JEV_TEMPERATURE`를 정할 것.
- 경계 사례(재전송 몇 번, BLER 8% 등)는 선택지 설명에 판단 기준을 적어야 정확해진다
  (예: "BLER < 10%이고 재시도 후 성공하면 normal").

## 로컬 개발 환경 (집 PC)

```bash
uv sync                                   # Python 3.12, torch cu126, transformers
python mock_vllm_server.py --port 8000    # models/Qwen3.5-4B 필요
JEV_BASE_URL=http://127.0.0.1:8000/v1 JEV_MODEL=qwen-local python probe_endpoint.py
```
