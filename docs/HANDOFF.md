# jev_project 인수인계 문서 (집 PC → 사내 Claude Code)

> **이 문서를 읽는 사내 Claude Code에게**
> 이 문서는 집 PC(RTX 4070 SUPER, 로컬 Qwen3.5-4B)에서 Jev 방식 판단 엔진과 MCP 서버를 만들고
> 검증한 전체 과정을 기록한 것이다. 네 목표는 이 코드를 **사내 Qwen3.8 엔드포인트에 붙여서
> MCP로 Claude Code / opencode에서 쓸 수 있게 하는 것**이다.
> 7장 "사내 적용 절차"부터 순서대로 진행하고, 판단이 필요하면 4장(고민/실패/해결)과 6장(알려진 한계)을 참고해라.
> 이 저장소는 **public**이다. 사내 엔드포인트 주소, 키, 실제 로그, 사내 데이터를 절대 커밋하지 마라.

---

## 목차
1. 배경과 목표
2. 핵심 원리: 1토큰 logprob 판단
3. 구성: 파일, 데이터 흐름, 두 종류의 "서버"
4. 구현 과정: 고민, 실패, 해결 (시간순)
5. 검증 및 벤치마크 결과
6. 알려진 한계와 미해결 과제
7. 사내 적용 절차 (단계별 + 분기표)
8. 사내 Claude Code에게 요청하는 작업 목록
9. 부록: 명령어 모음, 용어

---

## 1. 배경과 목표

### 1.1 Jev란
- TypeSafe AI가 2026-09-15에 발표한 "System One" 모델. 텍스트를 생성하지 않고,
  `state + 구조화된 질문들 → 타입이 있는 답 + 확률`을 한 번의 forward로 병렬 반환한다.
- 질문 타입은 세 가지: `noul`(예/아니오 → P(yes)), `choice`(선택지 중 하나), `score`(척도 점수).
- 주장: 20~200배 빠르고 40~400배 저렴. 벤치마크는 자사 평가이며 과장 가능성이 있다.
- 비판(Sean Goedecke 등): 기존 LLM도 응답을 미리 채우고 **선택지로 제한된 토큰 하나만 보면**
  비슷한 속도와 확률을 얻을 수 있다. 즉 기술보다 "LLM을 코드 속 fuzzy if문으로 쓰자"는
  **인터페이스 전환**이 핵심이다.

### 1.2 왜 직접 만드나
- Jev 본체는 웨이트리스트 기반 외부 호스팅이라, 사내 로그를 외부로 보낼 수 없다.
- 상용 API는 이 방식에 필요한 **logprobs를 주지 않는다**.
  - Claude(Anthropic Messages API): logprobs를 아예 제공하지 않는다.
  - Gemini 3.x: 파라미터는 남아 있지만 logprobs를 반환하지 않는다(의도된 동작이라고 함).
  - OpenAI 최신 계열: json_schema와 함께 쓰면 비어 있거나 top_logprobs가 에러를 낸다는 보고가 있다.
- 반면 vLLM/SGLang/llama.cpp로 **자체 서빙하는 오픈 모델은 logprobs를 준다.**
  사내 Qwen3.8(vLLM류 추정)이 바로 이 경우다.
- 참고한 오픈소스 대안: `open-alternative-jev`(Qwen3.6-27B로 검증, HF/vLLM), `OpenJev`(Qwen3.5-4B),
  `openjev`(DiffusionGemma, Jev API 호환). 이 프로젝트는 이들과 같은 원리를
  **OpenAI 호환 API의 logprobs만으로** 구현한다. 사내에서는 모델 가중치를 직접 불러올 수 없기 때문이다.

### 1.3 최종 목표
```
Claude Code / opencode  ──MCP(stdio)──▶  jev_mcp_server.py  ──HTTP(OpenAI 호환)──▶  사내 Qwen3.8 (vLLM)
     (질문 설계, 에스컬레이션)                (판단 엔진)                              (logprobs 제공)
```
- 역할 분담: Claude가 **무엇을 물을지(질문·선택지) 설계**하고 → jev가 대량 판단을 **싸고 빠르게** 처리하고 →
  Claude가 **확신도가 낮은 건만** 다시 들여다본다(confidence gate).
- 1차 적용 도메인: 모뎀 로그 분석. 로그 윈도 분류(RACH 실패, HARQ 폭주, RLF, 정상), 리뷰 필요 여부, 심각도 판단.

---

## 2. 핵심 원리: 1토큰 logprob 판단

### 2.1 프롬프트 형식 (`jev_api.build_user_content`)
질문마다 다음 user 메시지를 만든다. **state를 맨 앞에 둔다.** 같은 state에 대한 질문들이
서버의 prefix cache를 공유하게 하기 위해서다.
```
State:
<state 텍스트 또는 JSON(indent=2)>

Question: <instructions>
A. <option_name>: <description>
B. ...

Answer with the option letter only.      ← score는 "option number"
```
thinking은 반드시 꺼야 한다. chat 모드에서는 `chat_template_kwargs={"enable_thinking": false}`로 끄고,
completions 모드에서는 ChatML을 직접 만들고 assistant 시작부에 `<think>\n\n</think>\n\n`를 미리 채운다.

### 2.2 라벨 설계
- choice는 `A`~`Z`(최대 26개), noul은 `A=Yes / B=No`, score는 `1`~`9`(2~9단계).
- **선택지 이름 대신 단일 문자 라벨을 쓰는 이유**: 이름은 여러 토큰으로 쪼개질 수 있고
  (`technical_help` 등), 첫 토큰이 겹칠 수 있다(`rach_fail`과 `rlf`가 모두 `r`로 시작).
  첫 토큰의 확률만으로 구분하려면 **라벨 하나가 토큰 하나이고, 라벨끼리 겹치지 않아야** 한다.
- 로컬 구현(`jev_local`)은 라벨이 단일 토큰인지 토크나이저로 검증한다. API 구현은 토크나이저가 없으므로
  top_logprobs의 토큰 문자열을 정규화해서 매칭한다(2.4).

### 2.3 확률 계산 (`restricted_softmax`, `build_answer`)
1. `max_tokens=1, temperature=0, logprobs=true, top_logprobs=K`로 요청한다.
2. 첫 토큰의 top-K에서 각 라벨의 logprob을 찾는다. 없으면 `-inf`로 둔다.
3. 라벨 logprob끼리만 softmax를 한다. 이때 `temperature`로 나눌 수 있어서 보정에 쓴다.
4. 답 형태:
   - noul → `{"noul": P(yes)}`
   - choice → `{"choice", "confidence", "probabilities"}`
   - score → `{"score": 기대값을 0~1로 정규화, "level": 최빈 단계, "distribution"}`
5. 공통 필드:
   - **`label_mass`** = 원래 분포에서 라벨 토큰들이 차지한 확률의 합. 모델이 형식을 따랐는지 보여준다.
     **0.9 미만이면 신뢰하지 마라.**
   - `first_token` = 실제 1순위 토큰. `<think>`나 `Thinking`이 나오면 thinking이 안 꺼진 것이다.

### 2.4 토큰 문자열 정규화 (`_norm_token`)
서버와 토크나이저에 따라 라벨 토큰이 `"A"`, `" A"`, `"ĠA"`(byte-level BPE), `"▁A"`(sentencepiece)로 온다.
`Ġ`와 `▁`를 공백으로 바꾸고 strip한 뒤, 같은 라벨로 정규화되는 토큰들의 확률은 **합친다**(logsumexp).
실측에서는 `" A"`가 logprob -10.45 정도로 무시할 수준이었다.

---

## 3. 구성

### 3.1 파일
| 파일 | 역할 | 의존성 | 사내 필요 |
|---|---|---|---|
| `jev_api.py` | 핵심 엔진 `SystemOneAPI` (판단, 일반 생성, 비교), 질문 빌더, 프롬프트/파서 | 표준 라이브러리 | ✅ |
| `jev_mcp_server.py` | MCP stdio 서버. 도구 4개 | 표준 라이브러리 | ✅ |
| `probe_endpoint.py` | 엔드포인트가 이 방식을 지원하는지 7단계로 점검 | 표준 라이브러리 | ✅ |
| `mcp_smoke_test.py` | MCP 서버를 stdio로 띄워 핸드셰이크와 전 도구 호출 검증 | 표준 라이브러리 | ✅ |
| `bench/bench_compare.py` | jev vs 일반 생성 60건 정량 비교 | 표준 라이브러리 | ✅ |
| `bench/results_qwen3.5-4b_local.json` | 집 PC 기준 결과(비교 기준선) | – | 참고 |
| `jev_local.py` | transformers로 전체 logits를 직접 읽는 기준 구현 | torch, transformers | ❌ |
| `mock_vllm_server.py` | 로컬 모델로 vLLM 응답 형식을 흉내 내는 테스트 서버 | torch, transformers | ❌ |
| `.mcp.json.example`, `opencode.json.example` | 에이전트 등록 예시 | – | ✅ |

사내용 파일은 모두 **표준 라이브러리만** 쓴다. Python 3.10 이상이면 되고 pip 설치가 필요 없다.

### 3.2 MCP 도구
| 도구 | 입력 | 용도 |
|---|---|---|
| `jev_decide` | `state, questions, temperature?` | state 하나에 여러 질문을 병렬로 |
| `jev_decide_batch` | `states[], questions, temperature?` | 같은 질문을 최대 200개 state에 (로그 윈도 대량 트리아지) |
| `jev_compare` | `state, questions, thinking?, temperature?` | 같은 입력을 jev와 일반 생성으로 각각 판단해 나란히 비교 (실험용) |
| `llm_chat` | `prompt, system?, thinking?, max_tokens?` | 제약 없는 일반 텍스트 생성 |

### 3.3 두 종류의 "서버"를 구분할 것
| | 무엇 | 누가 띄우나 |
|---|---|---|
| MCP 서버 (`jev_mcp_server.py`) | 에이전트와 LLM 사이의 어댑터 | **Claude Code가 세션을 시작할 때 자동으로 실행하고 종료한다.** 설정만 해두면 된다 |
| LLM 서버 | 실제로 확률을 계산하는 모델 | 사내에서는 **이미 떠 있는 Qwen3.8 API**. 집에서는 `mock_vllm_server.py`를 직접 띄웠다 |

---

## 4. 구현 과정: 고민, 실패, 해결 (시간순)

### 4.1 환경 구축 (집 PC: Windows 11, Ryzen 7800X3D, RAM 32GB, RTX 4070 SUPER 12GB)
| 문제 | 원인 | 해결 |
|---|---|---|
| 모델 크기 결정 | VRAM 12GB. 27B는 Q4로도 약 16GB, 8bit면 약 28GB라 올라가지 않는다 | **4B**로 결정. FP16으로 약 8GB라 원본 정밀도 그대로 확률을 볼 수 있다 |
| vLLM/Docker 사용 불가 | vLLM은 Windows 네이티브를 지원하지 않고, WSL과 Docker는 설치되어 있지 않다 | **HF transformers로 logits를 직접 계산**하는 방식을 택했다 |
| Python 3.14만 설치됨 | ML wheel 호환이 불확실하다 | uv로 3.12 venv를 만들기로 했다 |
| `uv python install 3.12` 실패 | "Missing expected target directory for Python minor version link". 버전 단축 링크(junction)를 만들지 못하고 **조용히 3.14로 넘어갔다** | 인터프리터는 설치돼 있었다. **전체 경로로 `uv venv --python <path>`를 실행**하고 `requires-python = ">=3.12,<3.13"`으로 고정했다 |
| CUDA torch 설치 | 기본 PyPI의 torch는 CPU 빌드다 | `pyproject.toml`에 `[[tool.uv.index]] pytorch-cu126 (explicit)` + `[tool.uv.sources] torch`를 추가했다. 드라이버 560.94가 CUDA 12.6을 지원한다 |

### 4.2 모델: Qwen3.5-4B
- HF API로 Qwen 4B 계열을 조회한 뒤 `Qwen/Qwen3.5-4B`를 골랐다. Apache-2.0이고 gated가 아니며 8.7GB다.
  OpenJev가 쓴 것과 같은 모델이다.
- 메타데이터상 `image-text-to-text`(`Qwen3_5ForConditionalGeneration`)지만, `AutoModelForCausalLM`으로
  불러오면 `Qwen3_5ForCausalLM`이 되어 텍스트 전용으로 잘 동작했다.
- **하이브리드 linear attention 구조**라서 `flash-linear-attention`과 `causal_conv1d`가 없으면
  PyTorch 참조 구현으로 느리게 돈다. 경고가 나오지만 결과는 정확하다.
  짧은 입력은 약 74ms로 문제가 없었다. Windows에서는 이 커널들을 설치하기 어려워 생략했다.
- 로드 4.7초, VRAM 8.4GB. A/B 긴급도 질문 1회 forward에 74ms가 걸렸고
  top-1이 `A`(0.993), 라벨 확률 합은 0.9996이었다.

### 4.3 thinking 문제 (가장 중요한 함정)
- Qwen3.5는 **기본적으로 thinking이 켜져 있다.** 켜진 상태에서는 첫 토큰이 `<think>`나 `Thinking`이 되어
  **라벨이 top_logprobs에 아예 들어오지 않는다.** 그러면 판단 자체가 불가능하다.
- 해결:
  - chat: `chat_template_kwargs={"enable_thinking": False}`를 보낸다. 그러면 template이 빈 `<think></think>`를 채운다.
  - completions: ChatML을 직접 만들고 `<think>\n\n</think>\n\n`를 미리 채운다.
- **실패 1**: `probe_endpoint.py`의 첫 버전은 thinking 여부를 `"think" in first_token`으로 판정했다.
  그런데 실제 첫 토큰은 대문자 **`Thinking`**이라 매칭되지 않았고, 켜져 있는 것을 "OFF"로 잘못 보고했다.
  → **첫 토큰이 선택지 라벨(A/B)인지**로 판정하도록 바꿨다.
- **실패 2**: thinking이 켜진 채로 호출하면 원인을 알 수 없는 `ValueError`가 났다.
  → 라벨이 하나도 없으면 `first_token`을 포함해서 "thinking may be on" 힌트를 주는 `APIError`를 던지도록 바꿨다.

### 4.4 로컬 배치 처리
- 로컬 구현은 질문들을 left padding으로 한 배치에 묶어 forward한다.
- 배치 결과와 순차 결과의 확률 차이는 **최대 0.03**이었다. 하이브리드 linear attention의 패딩 처리와
  bf16 수치 오차 때문으로 보인다. 실사용에는 문제없는 수준이다.
- 지연: 질문 3개 기준으로 배치 262ms, 순차 315ms. API 버전은 배치 대신 **질문별 병렬 HTTP 요청**을 쓴다.

### 4.5 합성 모뎀 로그 테스트 (결과는 버리고 교훈만 남김)
- 직접 만든 LTE 로그 6건으로 테스트했다. 뚜렷한 장애 3건(RACH, HARQ, RLF), 정상 1건, 헷갈리는 정상 2건이다.
- failure_type은 5/6이었다. 틀린 건은 BLER 8%라 정상인데 harq_storm(0.85)으로 본 사례다.
- needs_review는 0.5 기준으로 3/6이었지만, **순서는 완벽했다**. 장애는 0.97 이상, 정상은 0.56~0.71이라
  기준을 0.9로 잡으면 6/6이 된다.
- 교훈:
  1. **예/아니오는 "예" 쪽으로 치우친다.** 0.5 기준은 믿을 수 없고 실제 데이터로 기준값을 정해야 한다.
  2. 경계 사례는 **선택지 설명 안에 판단 기준을 적어야** 한다 (예: "BLER < 10%이고 재시도 후 성공이면 normal").
- 사용자 판단: 가짜 로그로 하는 테스트는 의미가 없다. 그래서 테스트 파일은 삭제했다. **실제 로그로 다시 해야 한다.**

### 4.6 API 버전 설계 (`jev_api.SystemOneAPI`)
| 고민 | 결정과 이유 |
|---|---|
| 사내망에서 pip이 막혀 있을 수 있다 | **표준 라이브러리만** 쓴다(urllib, json, ssl, concurrent.futures). httpx, openai SDK를 쓰지 않는다 |
| 전체 logits를 받을 수 없다 | top-K logprobs만 쓴다. 라벨이 top-K 밖이면 확률 0(`-inf`)으로 본다. `label_mass`는 top-K 기준이라 하한값이다 |
| top_logprobs 한도 | vLLM 기본 `max_logprobs=20`이다. 기본값을 20으로 두고 probe로 실제 한도를 확인한다 |
| 게이트웨이가 `chat_template_kwargs`를 없앨 수 있다 | **completions 모드**를 대안으로 둔다. ChatML을 직접 만들고 빈 think를 미리 채운다. 단, 이는 Qwen의 ChatML 형식을 가정한 것이다 |
| 질문 여러 개 | 질문마다 요청 1개를 보내고 ThreadPoolExecutor로 병렬 처리한다. 한 프롬프트에 묶는 방식은 옆 질문의 영향을 받는다(open-alternative-jev가 6~9%라고 보고). 대신 입력 토큰이 늘어나므로 state를 앞에 둬서 prefix cache를 탄다 |
| 사내 인증서 | `JEV_INSECURE=1`이면 TLS 검증을 끈다 |
| 프록시 | urllib은 `HTTP(S)_PROXY`를 따른다. 사내 게이트웨이가 프록시 뒤에 있으면 `NO_PROXY`가 필요할 수 있다 |
| 로컬과 사내의 프롬프트 차이 | 프롬프트, 옵션, 답 구성 함수를 `jev_api.py`로 모으고 `jev_local.py`가 이를 import한다. **두 구현이 같은 프롬프트를 쓴다** |

### 4.7 가짜 vLLM 서버로 API 경로 검증
- 사내에서는 API로만 붙을 수 있으므로, 로컬 모델을 vLLM 응답 형식으로 감싼 `mock_vllm_server.py`를 만들었다.
  - `/v1/models`
  - `/v1/chat/completions`: `logprobs.content[0].top_logprobs[{token, logprob, bytes}]`, `chat_template_kwargs` 지원
  - `/v1/completions`: `logprobs.top_logprobs[0]`가 `{token: logprob}` 형태
  - `top_logprobs > 20`이면 400을 돌려준다
- 결과:
  - **chat 모드, completions 모드, 로컬 직접 계산이 소수점 4자리까지 같았다** (urgent 0.9841, intent 0.7058).
  - probe 7단계가 모두 통과했다.
- 주의: 이 서버는 vLLM의 형식을 **흉내 낸 것**이다. 실제 사내 게이트웨이의 응답은 다를 수 있으니 반드시 probe로 확인해라.

### 4.8 MCP 서버 (`jev_mcp_server.py`)
| 고민/실패 | 해결 |
|---|---|
| `mcp` Python SDK는 pip 설치가 필요하다 | stdio JSON-RPC를 **직접 구현**했다. `initialize`, `ping`, `tools/list`, `tools/call`을 지원하고 notification은 무시한다 |
| 프로토콜 버전 | 클라이언트가 요청한 버전이 `2025-06-18`, `2025-03-26`, `2024-11-05` 중 하나면 그대로 쓰고, 아니면 최신 버전으로 응답한다 |
| stdout 오염 | stdout에는 JSON 메시지만 쓴다. 로그는 **stderr**로 보낸다 |
| Windows 인코딩과 개행 | `sys.stdin/stdout.reconfigure(encoding="utf-8", newline="\n")` |
| 에이전트가 도구를 병렬로 부를 때 막힘 | `tools/call`을 **스레드**로 처리하고, 출력은 write lock으로 보호한다 |
| **실패**: stdin이 닫히면 진행 중이던 호출의 응답이 사라짐 | 처음에는 daemon 스레드여서 main이 끝나면 함께 죽었다 → **non-daemon 스레드**로 바꿨다 |
| 잘못된 입력 | 사전 검증(`_validate`)을 하고 에러를 `isError: true` 결과로 돌려준다. JSON-RPC error가 아니라서 에이전트가 읽고 고칠 수 있다 |
| 컨텍스트 낭비 | 결과의 float를 소수점 4자리로 반올림하고, `llm_chat`의 reasoning은 4000자로 자른다 |
| 에이전트가 결과를 잘못 해석함 | 도구 description에 `label_mass` 해석법과 gate 기준(0.9 이상은 자동 처리, 0.1 이하는 버림, 중간은 상위 모델로)을 적었다 |

### 4.9 Windows/PowerShell 함정
- **PowerShell 5.1은 네이티브 exe에 넘기는 인자의 큰따옴표를 제거한다.**
  - `python -c "..."`로 긴 스크립트를 넘기면 SyntaxError가 난다 → 스크립트를 파일로 만들어 실행한다.
  - `claude -p "<프롬프트>"`로 넘기면 프롬프트가 `"Help!"`에서 잘린다. 새 세션이 엉뚱한 질문을 받았다 → **프롬프트를 stdin으로** 넘긴다(`$p | claude -p ...`).
- `git commit -F -`에 here-string을 넘기면 실패한다 → 메시지 파일을 만들어 `-F <file>`로 넘긴다.
- CRLF 경고는 무시해도 된다.

### 4.10 MCP 연결 운영
- MCP 서버는 **세션을 시작할 때만** 불러온다. 도구를 추가하거나 설정을 바꾸면 새 세션을 열어야 반영된다.
- 프로젝트의 `.mcp.json`은 새 세션에서 승인 창이 뜬다. 실제 설정 파일은 gitignore했다(`.mcp.json`, `opencode.json`).
- 헤드리스 검증 명령:
  `$p | claude -p --mcp-config .mcp.json --strict-mcp-config --allowedTools "mcp__jev__jev_decide,mcp__jev__jev_decide_batch"`
  → 새 세션이 도구를 호출해서 정상적인 결과를 받았다.
- 집에서는 가짜 서버가 Claude Code 세션의 백그라운드 작업이었다. 그래서 그 세션을 닫으면 LLM 서버도 꺼졌다. 사내에서는 해당하지 않는다.

### 4.11 비교 도구 (`jev_compare`, `llm_chat`)
- 목적: jev 방식이 일반 생성과 같은 답을 내는지, 비용은 어떤지 실제 데이터로 판단하기 위해서다.
- plain 방식(`build_plain_prompt`): 모든 질문을 한 프롬프트에 넣는다. noul은 yes/no, choice는 옵션 이름,
  score는 단계 번호로 답하게 하고, 출력은
  `{"<key>": {"answer": ..., "confidence": 0-100}}` JSON 하나만 내도록 요구한다.
- thinking 출력 분리(`split_thinking`)는 세 경우를 처리한다.
  1. 서버에 reasoning parser가 있어서 `reasoning_content`가 따로 오는 경우
  2. content에 `<think>…</think>`가 섞여 오는 경우
  3. template이 `<think>`를 프롬프트에 넣어서 **닫는 태그 `</think>`만** 오는 경우
- 가짜 서버는 `max_tokens > 1`이면 일반 생성(`model.generate`)을 하도록 확장했다.
- **실패 (벤치마크 중 발견)**: plain 쪽 채점에서 모델이 JSON 키를 질문 키(`"a"`, `"is_true"`)가 아니라
  **질문 문장 그 자체**로 쓰는 일이 잦았다. 예: `{"9.11 is greater than 9.9": {"answer": "no", ...}}`.
  `parse_plain_answers`는 키가 정확히 맞아야 해서 답을 None으로 처리했다.
  - 질문 키를 `"a"`로 뒀던 첫 실행에서는 **60건 중 26건**이 이렇게 실패했다.
  - 키를 의미 있는 이름으로 바꾸니 3건으로 줄었다.
  - 벤치마크에서는 "키가 어긋나도 값이 하나면 그 값을 쓰는" **관대한 채점(lenient)**을 따로 집계했다.
  - `jev_api.py` 본체는 **아직 고치지 않았다** (6장 TODO #1).

---

## 5. 검증 및 벤치마크 결과

> 모든 수치는 **집 PC, 로컬 Qwen3.5-4B, transformers 기반 가짜 서버** 기준이다.
> 가짜 서버는 최적화 커널이 없어서 **생성 속도가 특히 느리다**. 사내 vLLM에서는 절대 시간이 훨씬 짧을 것이다.
> 대신 출력 토큰 수의 차이(1개 대 수십~수천 개)는 그대로 유지된다.

### 5.1 60건 정량 비교 (`bench/bench_compare.py`, plain은 thinking off)
| 과제 (각 20건) | jev | plain (엄격 채점) | plain (관대 채점) |
|---|---|---|---|
| 감성 (긍정/부정/혼합, 한·영) | 20 | 20 | 20 |
| 메시지 (정상/스팸/피싱, 한·영) | 20 | 20 | 20 |
| 사실 판단 (예/아니오) | 20 | 14 | 17 |
| **합계** | **60** | 54 | 57 |

| 지표 | jev | plain |
|---|---|---|
| 지연 (건당 중앙값) | **86ms** | 952ms (약 11배) |
| 토큰 (중앙값) | 입력 69 / 출력 1 | 입력 113 / 출력 21 |
| 확신도 범위 | 0.51 ~ 1.00 | 자기 보고 0.85 ~ 1.00 |
| 형식 준수 | label_mass 최소 0.953 | 파싱 에러 0건, 단 키 불일치 3건 |

저장소에 넣은 `bench/bench_compare.py`로 다시 돌렸을 때도 정확도, 오답 목록, 저확신 목록이 **완전히 같게 재현**됐다
(temperature 0, 결정적 실행). 지연은 84ms / 966ms로 오차 범위 안이었다.

**확신도 분석 (가장 중요한 발견)**
- plain이 틀린 3건은 `17*23=391`, `한글은 세종대왕이 창제했다`, `A week has 168 hours`이다.
  세 건 모두 정답이 "예"인데 plain은 **자기 확신도 1.0으로 "아니오"**라고 답했다.
- jev는 같은 3건을 모두 맞혔지만 확신도가 낮았다(0.56, 0.84, 0.62).
- jev 확신도가 0.7 미만인 건은 60건 중 4건이었다: 인보이스 0.51, "Hot singles" 스팸 0.60, 17*23 0.56, 168시간 0.62.
  **4건 모두 정답이었고**, 모델이 실제로 헷갈려 하는 사례를 골라냈다.
- 결론: **plain의 자기 보고 확신도는 판단 근거로 쓸 수 없다.** 틀린 답에도 1.0을 붙인다.
  jev의 확률은 넓게 퍼져 있어서 confidence gate로 쓸 수 있다.

### 5.2 소규모 비교 (다른 세션에서 MCP 도구로 실행)
| 테스트 | 두 방식 일치 | jev | plain |
|---|---|---|---|
| DB 커넥션 풀 장애 로그 (장애 여부 / 원인 / 심각도) | 2/3 | 573ms | 3,730ms |
| 한국어 상품 리뷰 (감성 / 재구매 / 가격 불만) | 3/3 | 312ms | 1,096ms |
| 고객지원 티켓 (담당팀 / 복수 이슈 / 긴급도) | 2/3 | 349ms | 2,375ms |
| 방망이·공 가격 문제 (plain은 thinking 켬) | 1/1 | 104ms | 119,680ms (2,699토큰) |
- 불일치는 모두 **score 질문**에서 났다. 심각도는 jev가 critical 0.65 / major 0.35, 긴급도는 jev가 high 0.47이었다.
  분포를 보면 원래 경계 사례였다. **score는 level 하나만 보지 말고 기대값과 분포를 함께 봐야 한다.**
- thinking을 켜면 약 2분에 2,699토큰이 들었지만, 이 테스트에서는 jev보다 나은 점이 없었다.

### 5.3 배치 (메일 10건을 정상/스팸/피싱으로)
- 총 약 2.9초, 건당 약 270ms였다. label_mass는 모두 0.99 이상이었다.
- 명확한 오분류는 1건이었다. 정상 인보이스를 피싱으로 판정했는데 확신도가 0.51이었다.
  0.7 미만 기준으로 걸러내면 이 오분류가 걸러진다.

### 5.4 MCP 도구 스모크 (`mcp_smoke_test.py`)
- 핸드셰이크 OK, 도구 4개 모두 확인했다. 단건 질문 3개가 0.6~0.9초, 배치 3건이 약 0.7초였다.
  비교 도구는 thinking off에서 3.0초, on에서 72초였다. 잘못된 입력은 `isError=true`로 돌아왔다.

### 5.5 이 결과를 어디까지 믿을 수 있나
- 60건은 작은 표본이다. 60 대 57은 통계적으로 유의하다고 보기 어렵다.
- 문항과 정답 라벨은 직접 만들었고 대체로 쉽다. **실제 도메인 데이터(모뎀 로그)로는 아직 측정하지 않았다.**
- 4B 모델 하나로만 측정했다. Qwen3.8에서는 정확도, 확신도 분포, 치우침이 모두 다를 수 있다.

---

## 6. 알려진 한계와 미해결 과제

| # | 항목 | 내용 | 제안 |
|---|---|---|---|
| 1 | **plain 파서 키 불일치** | `jev_api.py`의 `parse_plain_answers`는 모델이 질문 키가 아닌 다른 키를 쓰면 None으로 처리한다. 그래서 `jev_compare`가 plain을 실제보다 나쁘게 보고할 수 있다 | 질문이 하나이고 키가 하나면 그 값을 쓴다. 여러 개면 순서나 퍼지 매칭으로 대응한다. 또는 JSON schema로 키를 강제한다(vLLM `response_format`/guided json, 게이트웨이가 지원할 때만) |
| 2 | 확률 미보정 | noul이 "예" 쪽으로 치우친다. 0.5 기준은 부정확하다 | 실제 라벨 데이터 50~100건으로 질문별 기준값과 `JEV_TEMPERATURE`를 정한다. 신뢰도 구간별 실제 정확도(reliability)를 확인한다 |
| 3 | 입력 토큰 증가 | 질문 N개면 state를 N번 보낸다 | vLLM prefix caching으로 완화된다(state를 앞에 둔 이유). 필요하면 한 프롬프트에 묶는 방식을 실험하되 교차 영향을 측정한다 |
| 4 | top-K 밖 라벨 | top_logprobs 한도(보통 20) 밖의 라벨은 확률 0이 된다 | 선택지를 한도보다 충분히 적게 둔다. `label_mass`로 감지한다 |
| 5 | 위치 편향 미검증 | A나 1번 선택지를 선호할 가능성이 있다 | 선택지 순서를 섞어서 결과가 바뀌는지 측정한다 |
| 6 | score 기대값 | 단계 사이 간격이 같다고 가정한다 | 해석할 때 분포를 함께 본다 |
| 7 | Qwen3.8 template 미확인 | completions 모드는 Qwen ChatML과 빈 think를 가정한다 | probe 5번 결과와 첫 토큰으로 확인한다. 필요하면 prefix를 수정한다 |
| 8 | 게이트웨이 rate limit | 배치는 동시 요청 수가 `JEV_BATCH_WORKERS × 질문 수`까지 늘어난다 | 한도를 확인한 뒤 `JEV_BATCH_WORKERS`를 조정한다. 429 재시도는 **아직 구현하지 않았다** |
| 9 | 긴 로그 | 로그 윈도가 길면 입력 토큰과 지연이 커진다 | 윈도 크기와 요약 전처리 전략이 필요하다 |
| 10 | 재시도/타임아웃 | HTTP 실패 시 재시도가 없다 | 사내 환경에서 필요하면 추가한다 |

---

## 7. 사내 적용 절차

### 7.1 준비
```bash
git clone https://github.com/nitro527/jev_project.git
cd jev_project
python --version        # 3.10 이상
```

### 7.2 환경변수 (셸 또는 에이전트 설정에만. 절대 커밋하지 않는다)
```bash
export JEV_BASE_URL=http://<사내-게이트웨이>/v1
export JEV_MODEL=<모델 이름>          # /v1/models에 나오는 id
export JEV_API_KEY=<필요시>
export JEV_INSECURE=1                 # 자체서명 인증서일 때만
export NO_PROXY=<게이트웨이 호스트>    # 프록시 환경일 때
```
PowerShell이면 `$env:JEV_BASE_URL="..."` 형식으로 설정한다.

### 7.3 엔드포인트 점검
```bash
python probe_endpoint.py
```
**결과별 대응 분기표**
| 결과 | 의미 | 조치 |
|---|---|---|
| 1 FAIL (모델 없음) | 모델 이름 불일치 | 출력된 served models 중 하나로 `JEV_MODEL`을 설정한다 |
| 1 FAIL (연결/SSL) | 네트워크, 인증서, 프록시 문제 | `JEV_INSECURE=1`, `NO_PROXY`, URL 끝의 `/v1`을 확인한다 |
| 2 FAIL "logprobs가 비어 있음" | **게이트웨이가 logprobs를 제거하거나 모델이 지원하지 않음** | 5번 결과를 본다. 둘 다 안 되면 **이 방식은 불가**하다. 게이트웨이 담당자에게 logprobs 통과를 요청해야 한다 |
| 2 FAIL "라벨이 아님" | `chat_template_kwargs`가 무시되어 thinking이 켜진 상태 | 5번이 PASS면 `JEV_MODE=completions`로 설정한다 |
| 3 "기본이 thinking ON" | 참고 정보 | 2번이 PASS면 괜찮다 |
| 4 top_logprobs 한도 < 20 | 게이트웨이 제한 | `JEV_TOP_LOGPROBS=<한도>`로 설정하고, 선택지 수를 그보다 적게 설계한다 |
| 5 FAIL | `/completions` 미지원 | chat 모드만 쓴다 |
| 6 label_mass < 0.5 | 형식을 따르지 않음 | `first_token`을 보고 template이나 thinking 문제인지 확인한다 |
| 7 지연 | 참고 | 로컬 기준은 질문 1개 약 80ms, 3개 병렬 약 200ms였다 |

### 7.4 MCP 검증
```bash
python mcp_smoke_test.py
```
도구 4개가 나오고 `jev_decide`, `jev_decide_batch`, `jev_compare`(thinking off/on), `llm_chat`, 에러 케이스가 모두 정상이어야 한다.
thinking on 비교는 오래 걸릴 수 있다.

### 7.5 에이전트 등록
**Claude Code**
```bash
claude mcp add jev -e JEV_BASE_URL=http://<게이트웨이>/v1 -e JEV_MODEL=<모델> -- python /abs/path/jev_project/jev_mcp_server.py
```
또는 `.mcp.json.example`을 `.mcp.json`으로 복사해서 쓴다. 환경변수는 `${VAR}`로 확장된다. 등록한 뒤에는 **새 세션**을 열어야 한다.

**opencode**: `opencode.json.example`을 참고해서 `opencode.json`의 `mcp` 항목에 추가한다.

### 7.6 벤치마크 재측정 (사내 모델 기준선 만들기)
```bash
python bench/bench_compare.py --out bench_qwen38.json
```
결과를 `bench/results_qwen3.5-4b_local.json`과 비교한다. 결과 파일에는 공개 데이터만 들어가지만,
**사내 모델의 이름이나 주소가 들어갈 수 있으니 커밋하기 전에 확인해라.**

### 7.7 실제 데이터로 보정 (핵심 다음 단계)
1. 반복되는 판단 하나를 고른다 (예: 로그 윈도 failure_type).
2. 선택지와 **판단 기준이 담긴 설명**을 고정한다. `other`나 `normal` 같은 탈출구를 반드시 넣는다.
3. 엔지니어가 라벨을 붙인 실제 로그 윈도를 50~100건 모은다. **사내에만 보관하고 커밋하지 않는다.**
4. `jev_decide_batch`로 돌려서 정확도와 신뢰도 구간별 실제 정확도를 계산한다.
5. 기준값을 정한다. 예: confidence 0.9 이상은 자동 처리, 0.1 이하는 무시, 중간은 사람이나 상위 모델로 보낸다.
   필요하면 `JEV_TEMPERATURE`를 조정한다.
6. 같은 데이터로 `jev_compare`를 돌려 일반 생성과 비교한다 (6장 TODO #1을 먼저 고칠 것).

---

## 8. 사내 Claude Code에게 요청하는 작업 목록 (우선순위 순)

1. **7.3 probe를 실행하고** 결과를 사용자에게 분기표 기준으로 보고한다. logprobs가 불가능하면 즉시 멈추고 알린다.
2. probe 결과에 맞춰 `JEV_MODE`와 `JEV_TOP_LOGPROBS`를 정하고, `mcp_smoke_test.py`를 통과시킨다.
3. Claude Code에 MCP를 등록한다. **사용자 설정 파일을 바꾸기 전에 확인을 받는다.**
4. **TODO #1**: `parse_plain_answers`의 키 불일치 처리를 수정하고 `bench_compare.py`로 확인한다.
5. `bench/bench_compare.py`로 사내 모델의 기준선을 측정하고 집 PC 결과와 비교해 보고한다.
6. 사용자가 실제 모뎀 로그 라벨 데이터를 주면 7.7의 보정 절차를 수행한다.
7. 필요에 따라 6장의 3, 5, 8, 10번(프롬프트 묶기, 위치 편향, rate limit, 재시도)을 개선한다.

코드 규칙:
- 사내용 파일(`jev_api.py`, `jev_mcp_server.py`, `probe_endpoint.py`, `mcp_smoke_test.py`, `bench/`)은 **표준 라이브러리만** 쓴다.
- 프롬프트 형식을 바꾸면 `jev_local.py`도 영향을 받는다. 두 구현이 `jev_api`의 함수를 공유하기 때문이다.
- 이 저장소는 public이다. 커밋 전에 `git diff`에서 사내 정보가 있는지 확인한다.

---

## 9. 부록

### 9.1 설정 환경변수
| 변수 | 기본값 | 설명 |
|---|---|---|
| `JEV_BASE_URL` | (필수) | OpenAI 호환 API base (`.../v1`) |
| `JEV_MODEL` | (필수) | 모델 이름 |
| `JEV_API_KEY` | 없음 | Bearer 토큰 |
| `JEV_INSECURE` | `0` | `1`이면 TLS 검증을 끈다 |
| `JEV_MODE` | `chat` | `chat` 또는 `completions` |
| `JEV_TOP_LOGPROBS` | `20` | 서버 허용치 이하로 설정 |
| `JEV_DISABLE_THINKING` | `1` | thinking 끄기 (판단에는 필수) |
| `JEV_TEMPERATURE` | `1.0` | 선택지 softmax 온도. 1보다 크면 과신이 완화된다 |
| `JEV_BATCH_WORKERS` | `4` | 배치에서 동시에 처리하는 state 수 |

### 9.2 질문 작성 요령
- **noul**: 가장 범용적이다. 질문은 "~인가?"처럼 명확하게 쓴다. 0.5가 아니라 보정된 기준값을 쓴다.
- **choice**: 모델은 이름보다 **설명**을 보고 판단한다. 경계 기준을 설명에 적고, 탈출구 선택지를 반드시 넣는다. 최대 26개이고 top_logprobs 한도보다 적어야 한다.
- **score**: 단계 설명을 구체적으로 쓴다(2~9단계). level 하나만 보지 말고 기대값과 분포를 함께 본다.
- 반복해서 쓰는 판단은 질문과 선택지를 **고정**해야 결과가 일관되고 통계를 낼 수 있다.
  에이전트가 즉석에서 만든 선택지는 호출할 때마다 달라질 수 있다.

### 9.3 호출 예시 (모뎀 로그)
```json
{
  "state": "<로그 윈도>",
  "questions": {
    "failure_type": {"type": "choice", "instructions": "What is the primary failure pattern in this modem log window?",
      "choices": {
        "rach_fail":  "Random access (PRACH/RAR) failure: repeated RAR timeouts until preambleTransMax",
        "harq_storm": "Sustained HARQ NACKs with high BLER (>= 10%) on the data channel",
        "rlf":        "Radio link failure: out-of-sync, T310 expiry, re-establishment",
        "normal":     "No significant issue. Occasional retransmissions or a RACH retry that succeeds are normal"}},
    "needs_review": {"type": "noul", "instructions": "Should an engineer look at this log window?"},
    "severity": {"type": "score", "instructions": "How severe is the issue?",
      "scale": ["No impact", "Minor, self-recovered", "Degraded service", "Connection lost"]}
  }
}
```

### 9.4 용어
- **logprob**: 토큰 확률의 로그값. `top_logprobs`는 확률이 높은 상위 K개 토큰과 그 logprob이다.
- **label_mass**: 원래 분포에서 선택지 라벨 토큰들이 차지한 확률의 합. 형식 준수 지표다.
- **confidence gate**: 확신도에 따라 자동 처리, 무시, 에스컬레이션으로 나누는 패턴.
- **thinking**: Qwen 계열의 추론 모드. 켜져 있으면 첫 토큰이 `<think>`가 되어 이 방식이 동작하지 않는다.
- **prefix cache**: 앞부분이 같은 프롬프트의 KV를 재사용하는 서버 기능. vLLM V1에서 기본으로 켜져 있다.
