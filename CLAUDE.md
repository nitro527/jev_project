# jev_project

사내 LLM(OpenAI 호환 API, Qwen 계열)을 Jev 방식(1토큰 logprob 판단: noul/choice/score + 확률)으로 쓰고
MCP 서버로 Claude Code / opencode에 노출하는 프로젝트.

## 먼저 읽을 문서 (순서대로)
1. **`docs/HANDOFF.md`**: 설계 이유, 고민·실패·해결 기록, 사내 적용 절차(분기표), TODO, 작업 목록(8장)
2. `docs/CONCEPTS.md`: 개념 정리. raw와 plain의 차이, 출력 1토큰(속도)과 logprobs(확률)의 역할, 말로 한 확률이 안 되는 이유,
   입력/출력 시간, thinking, 보정, cascade, 진짜 Jev와의 차이
3. **`docs/TECHNIQUE_GUIDE.md`**: 문제 유형별 기법 선택. **질문을 설계하거나 기법을 고를 때 먼저 확인한다**
4. `docs/IMPROVE_AND_BENCHMARK.md`: 개선 기법 실험 결과(비용 포함), 벤치마크 설계, 재현 방법

## 규칙
- 이 저장소는 **public**이다. 사내 엔드포인트 주소, API 키, 실제 로그, 사내 데이터, 사내 모델 결과 파일을 커밋하지 마라.
  실제 설정은 환경변수나 gitignore된 `.mcp.json` / `opencode.json`에만 둔다. 커밋 전에 `git diff`를 확인한다.
- 사내용 파일(`jev_api.py`, `jev_mcp_server.py`, `probe_endpoint.py`, `mcp_smoke_test.py`, `bench/`)은
  **표준 라이브러리만** 쓴다 (사내망에서 pip 없이 동작해야 함). `bench/datasets.py`만 데이터 첫 다운로드 때 pandas를 쓴다.
- 프롬프트, 옵션, 답 구성 로직은 `jev_api.py` 한 곳에 있다. `jev_local.py`가 이를 import하므로 바꾸면 양쪽에 영향이 있다.
- 판단 호출에서 thinking은 반드시 꺼져 있어야 한다. 첫 토큰이 라벨이 아니면 동작하지 않는다.
- **logprobs가 이 방식의 전제다.** 없으면 확률·보정·cascade가 모두 불가능하다. "확률을 숫자로 답해"로 대체하지 않는다.
- 비용 비교는 지연보다 **토큰 수**를 기준으로 본다(지연은 서버마다 크게 다르다).
- `jev_local.py`와 `mock_vllm_server.py`는 집 PC(GPU) 개발용이다. 사내에서는 필요 없다.
- 사용자에게 설명할 때는 용어보다 **비유, 실제 프롬프트, 실측 숫자**를 쓴다. 용어에는 한 줄 설명을 붙인다.

## 자주 쓰는 명령
```bash
python probe_endpoint.py                     # 엔드포인트 점검 (JEV_BASE_URL, JEV_MODEL 필요)
python mcp_smoke_test.py                     # MCP 서버 전 도구 검증
python bench/latency_profile.py              # 고정 비용·TTFT·ITL·출력 비중 표·prefix cache·동시 처리량
python bench/improve_eval.py --plain --cascade 10 --cascade-budget 2048 --plain-thinking 5   # 기법 비교(비용 포함)
python bench/verbal_vs_logprob.py            # 말로 한 확률 vs logprobs
python bench/think_budget.py --budgets 512,2048   # 사고 예산 실험
python bench/patterns_demo.py                # 추출·재정렬·계층 분류 데모
python bench/bench_compare.py --out x.json   # jev vs 일반 생성 60건 비교
```
