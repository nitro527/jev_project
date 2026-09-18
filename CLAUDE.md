# jev_project

사내 LLM(OpenAI 호환 API, Qwen 계열)을 Jev 방식(1토큰 logprob 판단: noul/choice/score + 확률)으로 쓰고
MCP 서버로 Claude Code / opencode에 노출하는 프로젝트.

**작업 전에 `docs/HANDOFF.md`를 먼저 읽어라.** 설계 이유, 실패와 해결 기록, 벤치마크, 사내 적용 절차와 TODO가 있다.
성능 개선 기법과 벤치마크 설계는 `docs/IMPROVE_AND_BENCHMARK.md`에 있다.

## 규칙
- 이 저장소는 **public**이다. 사내 엔드포인트 주소, API 키, 실제 로그, 사내 데이터, 사내 모델 결과 파일을 커밋하지 마라.
  실제 설정은 환경변수나 gitignore된 `.mcp.json` / `opencode.json`에만 둔다. 커밋 전에 `git diff`를 확인한다.
- 사내용 파일(`jev_api.py`, `jev_mcp_server.py`, `probe_endpoint.py`, `mcp_smoke_test.py`, `bench/`)은
  **표준 라이브러리만** 쓴다 (사내망에서 pip 없이 동작해야 함).
- 프롬프트, 옵션, 답 구성 로직은 `jev_api.py` 한 곳에 있다. `jev_local.py`가 이를 import하므로 바꾸면 양쪽에 영향이 있다.
- 판단 호출에서 thinking은 반드시 꺼져 있어야 한다. 첫 토큰이 라벨이 아니면 동작하지 않는다.
- `jev_local.py`와 `mock_vllm_server.py`는 집 PC(GPU) 개발용이다. 사내에서는 필요 없다.

## 자주 쓰는 명령
```bash
python probe_endpoint.py                  # 엔드포인트 점검 (JEV_BASE_URL, JEV_MODEL 필요)
python mcp_smoke_test.py                  # MCP 서버 전 도구 검증
python bench/bench_compare.py --out x.json   # jev vs 일반 생성 60건 비교
python bench/improve_eval.py --plain         # 개선 기법 비교 (BGL/BoolQ/AG News, dev/test)
```
