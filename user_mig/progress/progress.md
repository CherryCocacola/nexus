# Project Nexus — 개발 진행 상황

> 이 파일은 세션 간 진행 상황 공유를 위해 매 작업마다 업데이트된다.

---

## 현재 상태

- **현재 Phase**: Phase 8.0 완료 + 웹 UI + GPU E2E + DocumentProcess
- **마지막 업데이트**: 2026-04-15
- **브랜치**: main
- **총 테스트**: 400개 (전부 통과) — 단위 316 + 통합 64 + E2E 20
- **전체 파일**: ~155개 Python/HTML 모듈

---

## Phase 0.5: Foundation (완료)

### v0.1.0 — 프로젝트 스켈레톤 (2026-04-14)
**커밋**: `1678f2e`

- `pyproject.toml` — 의존성, ruff/pytest/mypy 설정 통합
- `.gitignore` — Python, data/, models/, checkpoints/, logs/ 제외
- `requirements.txt`, `requirements-train.txt`, `requirements-dev.txt`
- `.env.example` — 환경 변수 템플릿
- 디렉토리 구조 생성 (Ch.21 기준 16개 패키지 + `__init__.py`)
- `config/*.yaml` 5종: nexus_config, tool_mappings, model_profiles, permission_rules, logging_config
- `tests/conftest.py` — 공통 fixture (mock GPU/Redis/PG)

### v0.1.1 — 핵심 부트스트랩 모듈 (2026-04-14)
**커밋**: `9746efb`

- `core/state.py` — GlobalState 싱글톤 (DAG leaf 격리, 스레드 안전 토큰 카운터)
- `core/config.py` — NexusConfig Pydantic v2 설정 시스템 (YAML 로딩, 에어갭 검증)
- `core/bootstrap.py` — Phase 1 초기화 (로깅, 종료 핸들러, GPU 사전연결, 플랫폼 감지)
- 21개 테스트 통과

---

## Phase 1.0: Model Layer (완료)

### v0.2.0 — 메시지 타입 + 추론 클라이언트 + GPU 감지 (2026-04-14)
**커밋**: `35b9a2e`

- `core/message.py` — 전체 타입 시스템 구현
  - StreamEvent (17종 이벤트 타입), Message (factory method 4종)
  - ContentBlock (TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock)
  - Conversation (compact_boundary 기반 컨텍스트 관리)
  - TokenUsage (합산 연산자, vLLM prefix caching 지원)
- `core/model/inference.py` — ModelProvider ABC + LocalModelProvider
  - vLLM OpenAI 호환 /v1/chat/completions SSE 스트리밍
  - Nexus Message → OpenAI message 자동 변환
  - tool_calls 증분 누적 + 최종 완성 로직
  - 연결 실패 시 graceful error event yield
- `core/model/gpu_detector.py` — GPU 티어 감지
  - GPUTier 4종 (RTX_5090, H100, H200, MULTI_GPU)
  - 티어별 최적 설정 테이블 (양자화, 컨텍스트, 배치, LoRA)
- `core/model/prompt_formatter.py` — 모델별 프롬프트 포매터
  - Gemma 4 (<start_of_turn>), ExaOne ([|system|]), ChatML 폴백
  - 도구 스키마 XML 주입
- `core/model/model_manager.py` — Machine A 측 모델 매니저
  - hot-swap 조율, LoRA 로드/언로드, 헬스 체크
- 61개 테스트 통과 (신규 40개)

---

## Phase 2.0a: Tool System 프레임워크 + 핵심 도구 (완료)

### v0.3.0 — BaseTool + Registry + Executor + 핵심 도구 8개 (2026-04-14)
**커밋**: `995ed5a`

- `core/tools/base.py` — BaseTool ABC + ToolResult + ToolUseContext
  - fail-closed 기본값, 7개 카테고리 ~25개 멤버
- `core/tools/registry.py` — ToolRegistry (등록, alias 조회, deny 필터, cache-stable 정렬)
- `core/tools/executor.py` — 13단계 실행 파이프라인
- 핵심 도구 8개:
  - Read, Write, Edit, MultiEdit (파일 시스템)
  - Bash (실행), Glob, Grep (검색), LS (디렉토리)
- 82개 테스트 통과 (신규 21개)

---

## Phase 2.0b: 나머지 도구 16개 (완료)

### v0.4.0 — 도구 24개 완성 (2026-04-14)
**커밋**: `9fe5ba3`

- `git_tools.py`: Git 도구 6개 (Log, Diff, Status, Commit, Branch, Checkout)
- `notebook_tools.py`: Notebook 도구 2개 (Read, Edit)
- `task_tools.py`: Task 도구 3개 (TodoRead, TodoWrite, Task)
- `agent_tool.py`: Agent 도구 1개 (서브 에이전트 stub)
- `memory_tools.py`: Memory 도구 2개 (인메모리 폴백)
- `docker_tools.py`: Docker 도구 2개 (Build, Run)
- 91개 테스트 통과

---

## Phase 3.0: Orchestrator (완료)

### v0.5.0 — QueryLoop + 스트리밍 + 컨텍스트 관리 (2026-04-14)
**커밋**: `c7f32d9`

- `query_loop.py`: while(True) 에이전트 턴 루프
  - 7가지 ContinueReason 전환, 4 Phase per iteration
  - LoopState 명시적 상태 관리, max_output_tokens 에스컬레이션
- `stream_handler.py`: StreamingToolExecutor
  - 모델 스트리밍 중 도구 병렬 실행, Semaphore 동시성 제한
- `context_manager.py`: 4단계 압축 파이프라인
  - tool_result 예산, snip/micro/auto compact, 긴급 압축
- `stop_resolver.py`: 종료/계속 판단 + 잘림 감지 휴리스틱
- 106개 테스트 통과

---

## Phase 4.0: Security & Permission (완료)

### v0.6.0 — 5계층 권한 + 보안 + 훅 시스템 (2026-04-14)
**커밋**: `525b096`

- `core/permission/types.py`: 전체 권한 타입 (7 Mode, 7 Category, 7x7 MAP)
- `core/permission/pipeline.py`: 5계층 파이프라인 (deny→tool→mode→hook→resolve)
- `core/security/path_guard.py`: 경로 보호 (순회, UNC, 보호 경로 21개 패턴)
- `core/security/command_filter.py`: 명령어 필터 (allowlist + 30+ 위험 패턴)
- `core/security/audit.py`: JSONL 감사 로그 (10MB 로테이션)
- `core/hooks/hook_manager.py`: 훅 매니저 (4 이벤트, 3 결정)
- `core/hooks/builtin_hooks.py`: 내장 훅 2개
- 124개 테스트 통과

---

## Phase 5.0: Thinking & Memory (완료)

### v0.7.0 — Thinking Engine + Memory System (2026-04-14)
**커밋**: `47a715f`

- Thinking Engine (core/thinking/ 6모듈):
  - ComplexityAssessor (키워드+증폭패턴 → 0.0~1.0)
  - ThinkingStrategy 4종 (DIRECT/HIDDEN_COT/SELF_REFLECT/MULTI_AGENT)
  - HiddenCoTEngine (2-pass), SelfReflectionEngine (3-pass)
  - ThinkingOrchestrator (복잡도→전략→실행→캐시)
  - ThinkingCache (SHA-256, LRU, TTL)
- Memory System (core/memory/ 6모듈):
  - MemoryType 5종, MemoryEntry, DECAY_HALF_LIFE
  - ShortTermMemory (Redis/인메모리), LongTermMemory (PG+pgvector/인메모리)
  - MemoryManager (턴 생명주기), ImportanceAssessor, MemoryDecayManager
- 236개 테스트 통과 (신규 112개)

---

## Phase 6.0 + 7.0: Training + CLI/Web/Deployment (완료)

### v0.8.0 — 학습 파이프라인 + 인터페이스 + 배포 (2026-04-14)
**커밋**: `3fc9787`

- Training Pipeline (training/ 6모듈):
  - 5-Phase 전략 (PROMPT→BOOTSTRAP→SELF_DATA→REASONING→DOMAIN)
  - 부트스트랩 데이터 생성 (도구 70% + 추론 30%)
  - LoRA/QLoRA 트레이너, 데이터 수집 (PII 마스킹), 자동 학습 루프
  - 체크포인트 관리 (활성화, 롤백, 최고성능)
- CLI (cli/ 3모듈): Rich REPL, Click 명령어, 출력 포매터
- Web (web/ 2모듈): FastAPI (chat/stream/sessions/tools/health), 미들웨어
- Deployment (deployment/ 2모듈): SHA256 무결성, 에어갭 번들 준비
- 288개 테스트 통과 (신규 52개)

---

## Phase 8.0: Integration & Polish (완료)

### v0.9.0 — 통합 테스트 64개 + 모듈 간 연결 검증 (2026-04-15)

- **conftest.py 확장**:
  - EnhancedMockModelProvider — tool_calls 시뮬레이션 가능한 ModelProvider mock
  - MockResponse dataclass — 응답 시나리오 정의 (text, tool_calls, stop_reason)
  - basic_tools, tool_use_context fixture 추가
- **tests/integration/test_full_pipeline.py** (22개 테스트):
  - TestQueryLoopIntegration: query_loop → tool → response 풀 플로우 (5개)
  - TestToolChainIntegration: 다중 도구 체인 실행 (2개)
  - TestSecurityIntegration: PathGuard + CommandFilter + PermissionPipeline (11개)
  - TestThinkingIntegration: 복잡도 평가 → 전략 선택 → 캐시 (4개)
  - TestMemoryIntegration: ShortTerm + LongTerm + Manager 인메모리 폴백 (4개)
- **tests/integration/test_training_integration.py** (12개 테스트):
  - BootstrapGenerator: JSONL 생성, 7:3 비율, seed 결정성 (4개)
  - DataCollector: PII 마스킹 (email/phone), 민감 경로 필터링 (5개)
  - CheckpointManager: 목록 조회, activate/rollback (httpx mock) (3개)
- **tests/integration/test_web_integration.py** (9개 테스트):
  - 7개 엔드포인트: health, chat, stream(SSE), sessions, tools, models, metrics
- **tests/integration/test_airgap_integration.py** (16개 테스트):
  - IntegrityVerifier: compute_hash, verify_file/directory, generate_manifest (12개)
  - AirGapPrep: generate_manifest, verify_manifest (4개)
- 352개 테스트 통과 (기존 288 + 신규 64)

---

---

## TODO 연동 작업 (완료)

### v0.9.1 — QueryEngine + 모듈 간 배선 연동 (2026-04-15)

- **core/orchestrator/query_engine.py** (신규):
  - Tier 1 세션 오케스트레이터 — query_loop (Tier 2) 래퍼
  - submit_message() → AsyncGenerator[StreamEvent | Message]
  - 대화 히스토리 관리, 세션 사용량 추적
- **core/bootstrap.py — Phase 2 초기화**:
  - init_phase2(): ToolRegistry(24개 도구) + MemoryManager + QueryEngine 초기화
  - _create_tool_registry(): 24개 도구 일괄 등록
- **web/app.py 연동**:
  - /v1/chat: QueryEngine.submit_message() → ChatResponse
  - /v1/chat/stream: QueryEngine → SSE 스트리밍
  - /v1/tools: ToolRegistry.get_all_tools() → 도구 목록
- **cli/repl.py + cli/commands.py 연동**:
  - _bootstrap()에서 Phase 2 초기화 (QueryEngine 자동 생성)
  - ask 커맨드: 비대화형 QueryEngine 호출
- **core/orchestrator/query_loop.py — HookManager Transition 5**:
  - hook_manager 파라미터 추가
  - HookEvent.STOP → BLOCK이면 강제 다음 턴 (stop_hook_blocking)
- **core/tools/implementations/memory_tools.py**:
  - MemoryRead: MemoryManager.search_relevant() 벡터+텍스트 검색
  - MemoryWrite: MemoryManager.add_semantic() 장기 메모리 저장
- **core/tools/implementations/agent_tool.py**:
  - 서브 에이전트: 독립 QueryEngine 생성 + DISALLOWED_TOOLS 필터링
  - max_turns=10 제한, 부모 context 연결
- 352개 테스트 통과 (기존 테스트 모두 유지)

---

## TaskManager 구현 (완료)

### v0.9.2 — TaskManager + task_tools 연동 (2026-04-15)

- **core/task.py** (신규):
  - TaskType (7종): LOCAL_BASH, LOCAL_AGENT, REMOTE_AGENT, TEAMMATE, WORKFLOW, MONITOR, TRAINING
  - TaskStatus (5종): PENDING, RUNNING, COMPLETED, FAILED, KILLED
  - TaskState (Pydantic BaseModel): id, type, status, description, progress, error_message, result
  - TaskManager: create, run, run_background, kill, update_progress, get_active, on_complete, cleanup_old
- **core/tools/implementations/task_tools.py** (수정):
  - TodoRead/TodoWrite/TaskTool → TaskManager API 연동 (폴백 유지)
  - _handle_stop() → async + TaskManager.kill() 사용
  - killed 상태 아이콘 [K] 추가
  - TaskState ↔ dict 변환 유틸 추가
- **core/bootstrap.py** (수정):
  - init_phase2()에 TaskManager 인스턴스 생성 + context.options 주입
- **tests/unit/test_task_manager.py** (신규, 28개 테스트):
  - TaskType/TaskStatus enum 검증 (5개)
  - TaskManager 라이프사이클: create, run, run_background, kill, progress, callback, cleanup (18개)
  - task_tools + TaskManager 연동 검증 (5개)
- 380개 테스트 통과 (기존 352 + 신규 28)

---

## GPU E2E + 벤치마크 + Soak Test (완료)

### v0.9.3 — 실 GPU 서버 연결 + 성능 측정 (2026-04-15)

- **서버 포트 분리**:
  - config/nexus_config.yaml: gpu_server.url → :8001, embedding_url → :8002
  - core/config.py: GPUServerConfig에 embedding_url 추가
  - core/model/inference.py: embedding_base_url 파라미터 + /v1/embed 엔드포인트 적용
  - core/bootstrap.py: embedding_base_url 전달
- **tests/e2e/test_gpu_e2e.py** (12개 테스트):
  - 서버 연결, 모델 확인, 텍스트 생성, SSE 스트리밍, tool_calling
  - 임베딩 생성, 배치 임베딩, LocalModelProvider 통합, 3턴 대화
- **tests/e2e/test_benchmark.py** (5개 테스트):
  - Simple Response Time: 0.728s (목표 <1.5s) — 통과
  - Complex Response Time: 7.369s (목표 <8s) — 통과
  - TTFT: 0.031s, TPS: 69.4 tokens/s, Embedding: 32.8ms
- **tests/e2e/test_soak.py** (3개 테스트):
  - 100-Turn 대화: 100/100 완료, 에러 0, 평균 0.314s
  - 50회 연속 추론: 50/50 완료, 성능 변화 +18.6%
  - 동시 임베딩+추론: 성능 저하 +1.7% (목표 <20%) — 통과
- 400개 테스트 통과 (기존 380 + E2E 20)

---

## 웹 채팅 UI + DocumentProcess (완료)

### v0.9.4 — 웹 UI 구현 + 문서 분석 도구 (2026-04-15)

- **web/static/index.html** — Claude 스타일 채팅 UI:
  - idino 색상 팔레트 (#00479d 기반), Nexus 브랜딩 + idino 로고
  - SSE 스트리밍 실시간 응답, 타이핑 커서 애니메이션
  - 사이드바 (세션 목록, 기본 닫힘, 햄버거 버튼으로 오버레이)
  - 파일 첨부 (텍스트 직접 포함 + 바이너리 서버 업로드)
  - 토큰 카운터 (IN/OUT 실시간 표시)
  - PWA 지원 (manifest.json, HTTPS 자체 서명 SSL)
  - 응답 대기 스피너 ("생각하는 중...")
- **web/app.py** — 웹 서버 연동:
  - 세션별 히스토리 격리 (QueryEngine messages 매 요청 초기화)
  - 히스토리 복원 시 토큰 예산 기반 제한 (tool_result 제외, user/assistant만)
  - 파일 업로드 API (POST /v1/upload → 임시 디렉토리 저장)
  - 웹 전용 QueryEngine (도구 8개, 토큰 예산 최적화)
  - HTTPS 자체 서명 SSL 인증서 (config/ssl/)
- **core/tools/implementations/document_tool.py** (신규) — DocumentProcess 도구:
  - PDF (pypdf), DOCX (python-docx), XLSX (openpyxl) 파싱
  - 청크 분할 (2,500자 단위) — 컨텍스트 초과 방지
  - 문서 캐시 — 같은 파일 재파싱 방지
  - chunk_index 파라미터로 순차 청크 읽기
- **core/bootstrap.py** — 웹용 도구 레지스트리:
  - `_create_web_tool_registry()` — 핵심 도구 8개 (Read, Write, Edit, Bash, Glob, Grep, LS, DocumentProcess)
  - 토큰 예산: ~2,051 토큰 (24개 6,102 토큰 대비 67% 절약)
- **core/model/inference.py** — 컨텍스트 초과 자동 처리:
  - 400 에러에서 input_tokens 추출 → max_tokens 자동 축소 재시도 (3회)
  - 재시도 모두 실패 시 "입력이 너무 길어 분석할 수 없습니다" 에러 메시지
  - SSE 파싱을 async with 블록 안으로 이동 (StreamClosed 수정)
- **core/orchestrator/query_loop.py** — 동적 max_tokens:
  - 입력 토큰 추정 후 max_tokens = max(512, max_context - estimated_input - 200)
  - 입력 초과 시 자동 truncate
  - 재시도 시 system 메시지 추가하지 않음 (토큰 증가 방지)
- **cli/repl.py** — Message 객체 필터링:
  - QueryEngine이 yield하는 Message 객체를 건너뛰고 StreamEvent만 표시
- **config** 변경:
  - gpu_server.url: 8001, embedding_url: 8002
  - max_context_tokens: 8192, default_max_tokens: 4096
  - gpu_server_url → property로 전환 (YAML gpu_server.url 자동 동기화)
- **GPU 서버** (192.168.22.28):
  - vLLM: --max-model-len 8192, --gpu-memory-utilization 0.90
  - 도구 호출 활성화: --enable-auto-tool-choice --tool-call-parser gemma4

---

## RTX 5090 (8192 ctx) 실측 제약 사항

```
VRAM 최대 max-model-len:
  8,192 ✅ (GPU 30.7GB/32.6GB)
  16,384 ❌ (OOM 실패)

토큰 예산 (도구 8개):
  도구 스키마:    ~2,500 토큰 (Gemma4 BPE 실측)
  시스템 프롬프트:   ~72 토큰
  고정 비용 합계: ~2,572 토큰

  가용 (입력+출력): ~5,620 토큰
  문서 분석 한계:  ~2,500자 (청크 1개) → A4 1~2페이지

대용량 문서:
  청크 분할로 순차 분석 가능 (무제한)
  단, 각 청크는 독립 분석 (이전 청크 맥락 유실)
```

---

## v7.0 적응형 멀티모델 오케스트레이션 (완료)

### v7.0.0 — Phase 9.0a~c + 9.5 (2026-04-16)

- **설계 개정안**: `user_mig/PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` 작성
  - 3가지 불변 전제: Claude Code 설계 유지, GPU 업그레이드 시 성능 향상, 동일 사용자 경험
  - HardwareTier 적응형 (TIER_S/M/L), TIER_M/L에서는 v6.1과 100% 동일 동작
- **Phase 9.0a: 도구 스키마 축소**
  - 11개 CLI 도구 description 영문 축소 (1,588 → 1,472 토큰)
- **Phase 9.0b: TurnState 상태 외부화**
  - `core/orchestrator/turn_state.py` (신규): TurnState, TurnStateStore, extract_turn_state()
  - query_loop에 on_turn_complete 콜백 추가
  - QueryEngine에서 TurnState 컨텍스트 복원 (이전 messages 대신 요약)
- **Phase 9.0c: HardwareTier 감지 + ModelDispatcher**
  - `core/model/hardware_tier.py` (신규): HardwareTier enum, detect_hardware_tier()
  - `core/orchestrator/model_dispatcher.py` (신규): ModelDispatcher (Scout→Worker 분배)
  - bootstrap.py: 티어별 자동 도구 선택 + TurnState + Scout 초기화
- **Phase 9.5: Scout 통합 (llama.cpp CPU 모델)**
  - GPU 서버에 llama.cpp b8808 + Gemma 4 E4B (Q4_K_M, 4.7GB) 설치
  - Scout 서버: :8003 (CPU, ~16 TPS), 방화벽 8003 포트 개방
  - `core/model/scout_provider.py` (신규): ScoutModelProvider
  - `core/config.py`: ScoutConfig + hardware_tier 필드 추가
  - Scout → Worker 핸드오프: _run_scout() 구현 (탐색→계획→Worker 실행)
  - Scout 실패 시 Worker 단독 fallback
- **버그 수정**:
  - 토큰 0/0 수정 (inference.py: finish_reason 후 usage 청크 대기)
  - CONTEXT_OVERFLOW 즉시 종료 (query_loop.py: 무한 재시도 방지)
  - GPU health 경고 수정 (bootstrap.py: vLLM 빈 body 대응)
  - 웹 시스템 프롬프트 수정 (파일 분석/코드 리뷰 중심)
- **Chrome 스타일 웹 UI**: `web/static/chrome.html` (시연용)
- **453개 테스트 통과** (기존 380 + v7.0 신규 73)

---

## v7.0 추가 구현 (2026-04-16 후반)

### Ch 7 WithRetry + StreamWatchdog
- `core/orchestrator/retry.py` (신규): ErrorCategory(9종), classify_error(), with_retry() AsyncGenerator
- `core/orchestrator/stream_watchdog.py` (신규): StreamWatchdog, stream_with_watchdog()
- query_loop에 stream_with_watchdog 통합 + StreamWatchdogTimeout 처리

### RAG 파이프라인
- `core/rag/indexer.py` (신규): ProjectIndexer (파일 탐색→청크→임베딩→저장)
- `core/rag/retriever.py` (신규): RAGRetriever (쿼리 임베딩→벡터 검색→컨텍스트 주입)
- QueryEngine.submit_message()에서 RAG 컨텍스트 자동 주입
- bootstrap에서 백그라운드 인덱싱 (fire-and-forget)

### Redis/PostgreSQL 실 연결
- PostgreSQL: 192.168.10.39:5440 (nexus/idino@12, DB=nexus)
  - pgvector v0.8.0 소스 빌드 설치 (docutil-postgres 컨테이너)
  - tb_memories 테이블 + 인덱스 5개 (idx_ 규칙)
  - 네이밍 규칙: tb_, idx_, vtb_, vw_, fn_, proc_
- Redis: 192.168.10.39:6340 (pw=docutil_redis_2024, db=6)
- bootstrap.py: _create_redis_client(), _create_pg_pool() (실패 시 인메모리 폴백)
- long_term.py: 테이블명 memories → tb_memories

### 모델 전환: Gemma 4 31B → Qwen 3.5 27B
- **이유**: Gemma 4는 vLLM LoRA/AWQ 미지원. Qwen은 전부 지원.
- **AWQ 서빙**: Qwen 3.5 27B AWQ (21GB) — vLLM :8001, VRAM 29GB
- **LoRA 학습**: 원본 Qwen 3.5 27B (52GB) + unsloth 4bit QLoRA
  - 1,000 샘플 (24개 도구 + 6개 추론 카테고리), 61분, loss 0.073
  - 체크포인트: /opt/nexus-gpu/checkpoints/qwen35-phase1/ (153MB)
- **LoRA 핫로딩 성공**: vLLM `--enable-lora --lora-modules nexus-phase1=...`
  - `"model": "qwen3.5-27b"` (기본) / `"model": "nexus-phase1"` (LoRA) 동적 전환
- **tool-call-parser**: `qwen3_xml` (hermes 아님 — Qwen 3.5 전용)
- **코드 수정**: 14개 파일 + 테스트 Gemma→Qwen 일괄 변경
- **Bootstrap 템플릿**: 7개 도구 → 24개 도구 전체 확장
- httpx 로그 레벨 WARNING으로 변경 (CLI 프롬프트 덮어쓰기 방지)
- 시스템 프롬프트: 범용 AI 어시스턴트로 변경 + thinking 출력 억제

### 테스트: 532개 통과

---

## Part 2.5.8 지식 RAG 파이프라인 구현 (2026-04-21 저녁)

### 배경
Part 2.5 라우팅으로 KNOWLEDGE_MODE에서 베이스 Qwen이 훨씬 정확해졌지만,
한국어 인문학·전문 지식은 여전히 부족. **외부 지식 베이스(위키)를 pgvector로
인덱싱하고 KNOWLEDGE 질의 시 자동 주입**하는 RAG 계층을 추가하여 근본 해결.

### 신규 파일
- `core/rag/knowledge_store.py` — KnowledgeStore + KnowledgeEntry + split_into_chunks
  - 인메모리 폴백 지원 (pg_pool=None)
  - add_many / search_by_vector / search_by_text / count / list_sources
  - build_vector_index (ivfflat, 대량 적재 후 1회)
- `core/rag/knowledge_retriever.py` — 검색 결과 → 시스템 프롬프트 블록 포매터
  - 토큰 예산 내 청크 연결, 임베딩 실패 시 텍스트 검색 폴백
- `scripts/prepare_kowiki.py` — kowiki 덤프 적재 스크립트 (운영 코드와 격리)
  - bz2 스트리밍, 카테고리 필터, 1,200자 청크, 배치 임베딩, UPSERT
- `tests/unit/test_knowledge_store.py` — 17 단위 테스트

### 스키마 (tb_knowledge)
```
id (text PK) / source / title / section / content / chunk_index /
total_chunks / tags (text[]) / embedding (vector(1024)) / created_at / metadata (jsonb)
인덱스: idx_knowledge_source/title/tags(GIN) + idx_knowledge_embed (ivfflat cosine)
```
tb_memories와 **별도 테이블** — 대화 EPISODIC 데이터와 혼재 방지.

### 수정 파일
- `core/orchestrator/query_engine.py` — `knowledge_retriever` 인자 + KNOWLEDGE 분류
  시 `effective_system_prompt`에 검색 결과 자동 주입
- `core/bootstrap.py` — Phase 2 ⑨-b에 KnowledgeStore 초기화 + ensure_schema()
- `web/app.py` — 웹 QueryEngine에 knowledge_retriever 주입

### 에어갭 준수
운영 Nexus 코드에는 외부 URL 없음. 덤프 다운로드는 GPU 서버에서 별도 wget으로
수행한 뒤 `scripts/prepare_kowiki.py`로 적재. `anti-patterns.md` #10 위반 없음.

### E2E 검증 (샘플 3건)
수동 큐레이션 위키 스타일 요약(차라투스트라/변신/니체)을 tb_knowledge에 적재 후
웹 요청으로 검증:

| 질의 | 라우팅 | RAG | 결과 |
|---|---|---|---|
| "차라투스트라를 세 가지 변신 중심으로" | KNOWLEDGE | 1,190자 주입 | 낙타-사자-아이 정확 (41.9s, 790토큰) |
| "카프카의 변신 줄거리와 주제는?" | KNOWLEDGE | 1,190자 주입 | 그레고르 잠자/소외 정확 (48.8s, 958토큰) |

실 로그:
```
라우팅: class=KNOWLEDGE, model=qwen3.5-27b, temp=0.20, max_tokens=2048
지식 RAG 주입: ~1190자
```

### 테스트
- 17/17 통과 (split/id 결정론성/폴백 CRUD/벡터 정렬/텍스트 검색/retriever 포맷·예산·폴백)
- 회귀 **649/649** 통과 (632 + 17)
- ruff: 신규 파일 clean

### 사양서
`PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` Part 2.5.8 섹션 전면 재작성 (스키마/적재
스크립트/E2E 결과/한계 및 향후 과제).

### 남은 작업
- **kowiki 실 덤프 적재** (GPU 서버, 1~수 시간) — 사용자가 백그라운드로 수행
- **Phase 10.0 RAG 심볼 인덱싱** — 우선순위 낮음

---

## Ch 6 ContextManager 티어별 전략 공식화 (2026-04-21 저녁)

### 배경
TurnStateStore(TIER_S 컨텍스트 요약)는 이미 구현되어 있었으나, ContextManager는
여전히 모든 티어에서 4단계 압축 파이프라인을 돌리도록 설계되어 있어 사양서
Part 5 Ch 6의 "티어별 전략" 의도와 어긋남. QueryEngine이 bootstrap에서
`context_manager=None`으로 초기화되어 있어 실질적으로는 사용되지 않았지만,
미래에 ContextManager 주입이 필요해질 때 TIER_S에서 중복 압축이 발생할 위험.

### 구현
- `core/orchestrator/context_manager.py`
  - `__init__`에 `tier: HardwareTier | None = None` 파라미터 추가
  - `_passthrough` 플래그: TIER_S이면 True
  - `apply_all`, `auto_compact_if_needed`, `emergency_compact` 각각에 pass-through
    분기 추가 (TIER_S면 TurnStateStore가 담당하므로 no-op 또는 최근 1턴만 추출)
  - `stats` 프로퍼티에 `tier`, `passthrough` 필드 노출
- `core/bootstrap.py` — Phase 2 ⑪단계에서 ContextManager 생성 + QueryEngine 주입
- `web/app.py` — 웹 QueryEngine에도 동일 인스턴스 공유

### 테스트
- `tests/unit/test_context_manager_tier.py` 신규 8건
  - TIER_S passthrough 검증 (apply_all/auto_compact/emergency)
  - TIER_M/L/None 기존 동작 유지 확인
  - stats 필드 노출 확인

### 회귀
- 632/632 통과 (624 + 신규 8)
- ruff: 신규 파일 전부 clean
- 서버 기동 로그 실측: `ContextManager 초기화: tier=small, passthrough=True`

### 하드웨어 업그레이드 시 자동 수렴
`detect_hardware_tier()`가 H100을 감지하면 `_passthrough=False`로 자동 전환되어
v6.1 4단계 압축 파이프라인이 활성화. 설정 변경 없이 수렴.

### 사양서 반영
`PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` Ch 6 섹션을 구현 실태에 맞게 재작성.

---

## Ch 16 세션 영속화 구현 (2026-04-21 오후)

### 배경
진단 결과 웹 채팅 히스토리가 `_app_state["chat_histories"]` 인메모리 dict에만
저장되고 있어 서버 재기동 시 전부 소실. Redis/PG 연결은 성공했지만 **QueryEngine이
MemoryManager를 호출하지 않아** 저장 기능이 연결되지 않은 "유령 구성" 상태였음.

### 구현

**신규 모듈**:
- `core/memory/transcript.py` — SessionTranscript, list_transcript_sessions
  - `{sessions_dir}/{session_id}/transcript.jsonl`에 append-only 기록
  - 턴 단위 JSON Lines (ts, role, content, turn, usage)

**수정**:
- `core/memory/short_term.py` — `list_sessions(limit)` 추가 (Redis SCAN)
- `core/orchestrator/query_engine.py` — `memory_manager`, `transcript` 파라미터 추가,
  `_finalize_turn()` 내부 헬퍼로 submit_message 말미에서 호출
- `core/bootstrap.py` — CLI용 SessionTranscript 생성 + QueryEngine에 주입
- `web/app.py`
  - `_app_state["memory_manager"]` 노출
  - `/v1/chat`, `/v1/chat/stream` 핸들러에 session_id 세팅 + Redis 복원 + write-through
  - `/v1/sessions` 엔드포인트가 실제 저장된 세션 목록 반환

**저장 3단 구조**:
| 매체 | 용도 | TTL |
|---|---|---|
| 인메모리 chat_histories | 프로세스 내 빠른 접근 | 소멸 |
| Redis session:{id}:context | 재기동 후 복원 | 24h |
| {sessions_dir}/{id}/transcript.jsonl | 영구 감사 기록 | 영구 |
| tb_memories (중요 턴) | 의미 검색/장기 승격 | 영구 |

### E2E 검증

| 단계 | 결과 |
|---|---|
| 턴 1 "내 이름은 홍길동이야" (SID=persist-e2e-001) | 응답 2.7s, Redis+JSONL 기록 |
| 턴 2 "내 이름이 뭐였지?" | 응답 0.7s, "홍길동" 정확 복기 |
| **서버 재기동 후** 같은 SID로 재질의 | **messages=3개 Redis 복원** → "홍길동입니다" 1.2s |
| `/v1/sessions` | 디스크 1 + Redis-only 1 총 2개 반환 |

### 테스트
- `tests/unit/test_session_persistence.py` 신규 (11 케이스)
  - SessionTranscript 기록/비활성/목록 조회
  - ShortTermMemory.list_sessions 인메모리 폴백
  - QueryEngine._finalize_turn (memory 호출, swallow 에러, noop, 실파일 기록)
- 회귀 624/624 통과 (기존 613 + 신규 11)

### 사양서
- Part 5 Ch 16 섹션 전면 업데이트 (아키텍처·저장 매체·E2E 결과)

### 남은 작업 (후순위)
- **Ch 6 ContextManager 티어별 전략 분기** — TIER_S는 TurnStateStrategy,
  TIER_M/L은 기존 4단계 압축 (현재 TurnStateStore는 있지만 context_manager 공식 분기 미구현)
- **Phase 10.0 RAG 심볼 인덱싱** (선택, 큰 작업)
- **Part 2.5.8 인문학 RAG 지식 베이스** (사용자 결정 대기)

---

## Scout 29자 수렴 근본 해결 (2026-04-21 오후)

### 배경
Part 2.5(쿼리 라우팅) 도입 후 실측에서 Scout가 여전히 "29자"만 Worker에 전달하는
증상 지속. α 진단(30분)으로 세 층의 누적 버그 확정.

### α 진단 결과
Scout 서버(:8003)에 직접 curl로 6개 케이스 비교:

| # | enable_thinking | finish_reason | out tokens | content 길이 |
|---|---|---|---|---|
| A | 미주입 | stop | 255 | 478자 (정상) |
| **B** | **False (Nexus 기본)** | **tool_calls** | **28** | **0** (버그 재현) |
| C | True | stop | 348 | 451자 (정상) |

→ `chat_template_kwargs={"enable_thinking": False}`가 Qwen3.5-4B에서
  거짓 tool_call 1개 뱉고 28토큰 조기 종료 유발. Worker 27B에서는 무해하지만
  4B에서는 치명적.

**추가 발견 (E2E 재현 중)**:
- Part 2.5 라우팅이 Scout 서브에이전트 QueryEngine에도 적용되어 Scout 서버에
  `model=nexus-phase3`(존재하지 않는 모델명) 주입
- llama.cpp가 `<think>` 블록을 reasoning_content로 자동 분리하는데 Nexus SSE
  파서가 이 필드를 버려서 Scout 출력의 대부분이 Worker에게 전달되지 않음

### 해결 (코드 6곳, 모델 교체 없음)

1. `core/model/inference.py` — `stream()` 시그니처에 `enable_thinking: bool | None`
   추가, `None`이면 `chat_template_kwargs` 생략
2. `core/model/scout_provider.py` — `ScoutModelProvider.stream()` 오버라이드로
   `enable_thinking`을 항상 `None`으로 강제
3. `core/orchestrator/query_engine.py` — `routing_config.enabled=False`일 때
   `model_override/temperature/max_tokens_cap/enable_thinking` 모두 기본값으로
   돌려놓는 분기 추가
4. `core/tools/implementations/agent_tool.py` — 서브에이전트 QueryEngine 생성 시
   `routing_config=RoutingConfig(enabled=False)` 주입
5. `core/model/inference.py` — `delta.reasoning_content`를 THINKING_DELTA로 yield
   (프로바이더 플래그 `_include_reasoning_as_text=True`면 TEXT_DELTA로 병합)
6. `core/model/scout_provider.py` — Scout는 `_include_reasoning_as_text=True`
   설정하여 reasoning을 Worker에게 전달

### 실 E2E 검증 (13:04)

`"core/config.py 파일에 어떤 설정 클래스들이 정의되어 있는지 알려줘"`:

| 지표 | 수정 전 | 수정 후 |
|---|---|---|
| Scout → Worker 전달 길이 | 29자 | **1,211자** (42배) |
| Scout 출력 토큰 | 28 | 335 |
| 응답 성공 | ❌ timeout 300s | ✅ **75.7s, 완전한 답변** |
| 라우팅 로그 | `model=nexus-phase3` | "라우팅 비활성" |

Worker가 Scout의 4섹션 마크다운 리포트를 받아 247토큰짜리 정돈된 답변 생성
(GPUServerConfig/RedisConfig/PostgreSQLConfig/ModelConfig/SessionConfig/
ScoutConfig/RoutingProfile/RoutingConfig 8개 클래스 나열).

### 테스트
- `tests/unit/test_scout_provider.py` 신규 (7개 테스트)
- 회귀 613/613 통과 (기존 606 + 신규 7)

### 결정: γ(Scout 모델 교체) 철회
α 단독으로 해결됐으므로 4B → 7B 교체는 불필요. Qwen3.5-4B Q4_K_M 유지.

### 사양서 반영
`user_mig/PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` Part 2.3에 **3차 개정 (2026-04-21)**
절을 추가하여 위 6개 수정 지점과 근거를 문서화.

---

## v7.0 Part 2.5 — 쿼리 라우팅 도입 (2026-04-21)

### 배경
사용자 실사용 중 "차라투스트라는 이렇게 말했다 설명해줘" 질의에서 모델이
"카프카의 소설"로 오답. 원인 분석 결과:
- Phase 3 LoRA가 도구 호출·기술 지식을 강화하는 대신 일반 교양 지식의
  표현을 좁히는 부작용 발생
- `chat_template_kwargs={"enable_thinking": False}` + temperature 0.7로
  자체 검증 없이 첫 연상을 그대로 출력

### 0단계 진단 (A/B/C/D 실측)

같은 질문(`차라투스트라는 이렇게 말했다`)에 대한 4가지 조합 curl 비교:

| # | model | thinking | 언어 | 결과 |
|---|---|---|---|---|
| A | nexus-phase3 + thinking=False (현 운영) | OFF | KO | 정답 + 경미한 할루시네이션 ("알렉산더 폰 훔볼트 풍자" 등) |
| B | qwen3.5-27b (LoRA OFF) + thinking=False | OFF | KO | **완벽 답변** — 낙타/사자/아기 3변신 상세 설명 |
| C | nexus-phase3 + thinking=True | ON | KO | A보다 개선, B보다 약함 |
| D | qwen3.5-27b + thinking=True | ON | EN | content에 thinking leak되어 답변 잘림 |

**결론**:
- LoRA OFF(B)가 일반 지식에서 압도적 우위 → LoRA가 원흉 확정
- `enable_thinking=True`는 leak 이슈로 당분간 사용 보류
- 도구 호출 질의는 여전히 Phase 3 LoRA 필요

### 조치 — 쿼리 라우팅 분기 구현

**신규 분류기**:
- `core/orchestrator/query_engine.py` — `classify_query()`, `_resolve_profile()`
- 규칙: `enabled=False` → TOOL / 길이 ≥500 → TOOL / tool_keywords 포함 → TOOL /
  그 외 → KNOWLEDGE

**신규 Pydantic 모델**:
- `core/config.py` — `RoutingConfig`, `RoutingProfile`
- `config/nexus_config.yaml` — `routing:` 섹션

**프로필**:
| 프로필 | model | temperature | max_tokens | enable_thinking |
|---|---|---|---|---|
| knowledge_mode | `qwen3.5-27b` (LoRA OFF) | 0.2 | 2048 | false |
| tool_mode | `nexus-phase3` (LoRA ON) | 0.3 | 4096 | false |

**전파 경로** (4-Tier 파라미터 추가, 기본값은 기존 동작과 동일):
```
QueryEngine.submit_message
  → classify + profile 선택
  → ModelDispatcher.route(model_override, temperature, max_tokens_cap, enable_thinking)
    → query_loop(… 동일 파라미터 …)
      → model_provider.stream(… payload["model"] = model_override …)
```

**수정 파일** (7개):
- `core/config.py` — RoutingConfig/RoutingProfile 신규
- `config/nexus_config.yaml` — routing 섹션 추가
- `core/model/inference.py` — `stream()` 시그니처 확장
- `core/orchestrator/query_loop.py` — 파라미터 전파 + max_tokens_cap 적용
- `core/orchestrator/model_dispatcher.py` — route() 시그니처 확장
- `core/orchestrator/query_engine.py` — 분류기 + submit_message 분기
- `core/bootstrap.py`, `web/app.py` — QueryEngine 생성 시 routing_config 주입

**테스트**:
- `tests/unit/test_query_routing.py` (신규, 28개 케이스)
- `tests/conftest.py` — EnhancedMockModelProvider.stream() 시그니처 동기화
- **회귀 600개 전부 통과** (단위 535 + 통합 65)

### 사양서 개정
`user_mig/PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md`에 **Part 2.5 신규 추가**:
- 배경, 진단 근거, 설계, 데이터 모델, 4-Tier 영향, 하드웨어 업그레이드 시
  자동 비활성화 조항, 한계와 향후 과제

### TIER_M 이상 업그레이드 시 자동 복귀
- RTX 5090 → H100으로 업그레이드 시 `routing.enabled: false`로 전환하면
  단일 경로(베이스 모델 + 24개 도구)로 복귀 가능
- Phase 3 LoRA 자체가 8K 컨텍스트 우회책이라 TIER_M 이상에서는 불필요

### 다음 단계 (2단계 RAG 지식 베이스 — 보류)
사용자 결정에 따라 2단계(한국어 위키 덤프 → pgvector 인덱싱 → knowledge_mode
진입 시 자동 검색 주입)는 1단계 실측 효과 확인 후 착수.

### 실측 검증 (사용자 확인 필요)
웹 서버 재기동 후 `차라투스트라`/`니체` 등 지식 질의 → 로그에서
`라우팅: class=KNOWLEDGE, model=qwen3.5-27b, temp=0.20` 확인 + 응답 품질 개선.

---

## 사양서 원본 회귀 (경로 ⓐ) — 부분 성공 + docx 이슈 (2026-04-18~19)

### 배경
경로 B 전환(2026-04-17) 후 사용자가 실사용에서 다음 지적:
1. Scout가 짧은 요약만 뱉음 (4B가 27B가 해야 할 요약 작업 수행)
2. Worker가 시키지 않은 파일을 임의 생성(Write logs/debug.log)
3. Worker가 바이너리 파일에 Read fallback 시도

사양서 v7.0 Part 2.3(SCOUT_TOOLS·JSON 출력)과 Part 2.4(Worker는 실행 전용·
Read 필요 없음)와 현재 구현을 전수 대조한 결과 **3가지 중대 이탈** 발견:
- Worker 도구 풀에 Read/Glob/Grep/LS 포함 (Part 2.4 위반)
- Worker system_prompt에 `{scout_plan}` 슬롯 구조 부재 (Part 2.4 위반)
- Scout가 JSON 아닌 자유 텍스트 요약 반환 (Part 2.3 위반)

### 조치 (경로 ⓐ 사양서 원본 회귀)
**1) Worker 도구 축소** — Part 2.4 원본 `{Edit, Write, Bash, GitCommit, GitDiff}` 복원
- CLI Worker: 11개 → 6개 (Edit/Write/Bash/GitCommit/GitDiff/Agent)
- Web Worker: 8개 → 4개 (Edit/Write/Bash/Agent)
- Read/Glob/Grep/LS/DocumentProcess는 **Scout 전용**

**2) SCOUT_AGENT.system_prompt 재작성**
- Worker(27B)가 분석, Scout(4B)는 "탐색·계획"만
- 출력 형식: Part 2.3 원본 JSON → 실측 결과 4B가 JSON 규율 못 지킴 →
  **마크다운 4섹션** (`## relevant_files`, `## file_summaries`, `## plan`,
  `## requires_tools`)으로 **2차 완화** (사양서 Part 2.3 2차 개정으로 문자화)

**3) Worker system_prompt 재작성**
- "You are the 27B brain; Scout is a 4B helper"
- Scout 마크다운 섹션 해석 지시, `## plan` 본문을 factual ground truth로 사용
- "Scout 1회 호출 제한, 재호출 금지" 규칙 명시

**4) 사양서 Part 2.3 2차 개정 추가**
- JSON→마크다운 완화 근거 문서화 (Qwen3.5-4B 능력 한계 반영)
- 정신(4섹션 구조)은 유지, 형식만 완화

### 검증 결과 3/4 통과

| # | 시나리오 | 결과 | 비고 |
|---|---|---|---|
| 1 | 인사 | ✅ 1초, 10토큰 | 깔끔 |
| 2 | 일반 지식 (짜라투스트라) | ✅ 10초, 139토큰 | 구조화된 자세한 답변 |
| 3 | 텍스트 첨부 (DB 로그 분석) | ✅ 44초, 309토큰 | 원인 진단 + 조치 추천 |
| 4 | docx 업로드 → Scout 분석 | ❌ 180~240초 타임아웃 | **아래 미해결 이슈 참조** |

### 미해결: docx 파일 분석 무한 호출 루프

**증상**: docx 업로드 시 Scout가 매 호출마다 **정확히 29자** 짧은 응답을
반환하고 Worker가 7회 이상 재호출하며 타임아웃.

**시도한 조치 (모두 효과 없음)**:
- Scout system_prompt에 "DocumentProcess 사용 강제" 규칙
- Scout 출력 형식 JSON → 마크다운 헤더 완화
- Worker system_prompt에 "Scout 1회 호출 제한" 명시
- max_turns 확대

**추정 원인 (프롬프트 레벨 아님)**:
1. Scout 서버(llama.cpp `--jinja`)가 Qwen3.5-4B의 tool_call을 제대로 파싱 못 함
2. `chat_template_kwargs={"enable_thinking": false}` 파라미터가 llama.cpp에
   전달은 되지만 실제 효과 미지수
3. Scout 응답이 29자로 고정 수렴하는 건 시스템 레벨 stop 조건의 조기 발동
   (LLM 자유 응답이 이렇게 규칙적이지 않음)

**다음 세션 작업 후보**:
- α. Scout 서버에 직접 curl로 최소 요청 보내서 실제 tool_call 생성·파싱 과정 관찰 (30분~1시간)
- β. DocumentProcess를 Worker에 임시 복귀 (사양서 정신 타협, 15KB 이하만 지원)
- γ. Scout 모델을 Qwen3.5-4B → 7B로 교체 (CPU 자원 여유 확인 필요, GGUF 다운로드 + llama.cpp 재기동, 1~2시간)

현재는 α(디버깅)를 먼저 하는 것이 정보 확보 관점에서 권장됨.

### 커밋 상태
- 경로 ⓐ 변경 전체를 한 커밋으로 정리 (docx 이슈 주석 포함)
- 회귀 572/572 통과, 시나리오 1~3 실작동 확인

---

## Scout를 문서 분석 전담자로 복원 — 경로 B (2026-04-17)

### 배경
Phase 3 LoRA 성공 후 실사용 중 15KB docx 파일 업로드 시 "입력 내용이
너무 길어 분석할 수 없습니다" 에러 발생. 로그 분석 결과:
- 1차 시도 input=4097, max_tokens=4096 (합 8193 > 8192 한계)
- 재시도 시 input이 4097 → 4198 → 4299로 증가 (vLLM prefix cache 영향 추정)
- 근본 원인: Worker가 DocumentProcess를 직접 호출하면서 큰 문서 원문이
  Worker 컨텍스트(8K)에 직접 들어가 한계 초과

### 설계 판단: Scout 원래 목적 복원
사용자 지적 — "Scout가 원래 토큰 줄이기 위한 거 아니냐".
사양서 v7.0 Part 2.4의 핵심 원칙 재확인:
> Scout가 이미 읽은 파일 내용을 요약으로 전달한다. Worker는 파일을 다시
> 읽지 않으므로 Read 도구가 필요 없다. Worker 컨텍스트를 실행에만 집중.

현재 B 방식 구현은 이 원칙을 훼손 — DocumentProcess가 Worker 쪽에 있어
큰 문서가 Worker로 직통. 이를 **Scout 전용으로 이관**하여 본래 설계
의도 복원.

### 변경
**1. `core/orchestrator/agent_definition.py`**:
- `SCOUT_AGENT.allowed_tools`: 4개 → 5개 (DocumentProcess 추가)
- `max_turns`: 3 → 5 (문서 청크 순차 처리 위해)
- `description`: "analyzing an uploaded document" 문구로 파일 업로드 시
  Worker의 Scout 자율 선택 유도
- `system_prompt`: "absorb large data sources on behalf of the Worker"
  명시, DocumentProcess 청크 워크플로우 안내

**2. `core/bootstrap.py`**:
- `_create_scout_tool_registry`: DocumentProcess 추가 (5개)
- `_create_cli_tool_registry`: DocumentProcess 제거 (Worker는 Scout 경유)
- `_create_web_tool_registry`: DocumentProcess 제거 (스키마 ~200토큰 절감)

**3. `web/static/index.html`**:
- 파일 업로드 시 자동 생성 메시지를 "DocumentProcess 도구 사용" →
  "Agent(scout)로 분석하고 요약 받아 주세요"로 교체

**4. `core/model/inference.py`**:
- 재시도 마진을 100 → 300으로 확대 (vLLM input 증가 완화)

**5. 사양서 v7.0 AMENDMENT Part 2.3 개정**:
- SCOUT_AGENT 선언 갱신 + 개정 근거 명시 + 하드웨어 업그레이드 시
  자동 복귀 조항 추가

**6. 테스트**: `test_agent_definition.py`와 `test_agent_tool.py`의
`allowed_tools`/`max_turns` 단언 갱신. `_verify_phase.py`에 파일 분석
시나리오(6번째) 추가.

### 중기 계획 — 하드웨어 업그레이드 시 Scout 복잡도 철회

Scout 인프라는 **RTX 5090 8K 컨텍스트 제약을 우회하기 위한 임시 장치**다.
GPU 업그레이드 시 자동 무력화되어야 한다:

| 대상 하드웨어 | Worker 컨텍스트 | Scout 상태 | DocumentProcess |
|---|---|---|---|
| RTX 5090 (현재, TIER_S) | 8K | **필수** (이 개정 적용) | Scout 전용 |
| H100 80GB (TIER_M) | 32K | 비활성 | Worker 직접 사용 가능 |
| H200 141GB (TIER_L) | 128K | 비활성 | Worker 직접 사용 가능 |

**TIER_M/L 복귀 경로**:
1. `core/model/hardware_tier.py::detect_hardware_tier()`가 VRAM 기반으로
   TIER를 판정 → H100 감지 시 자동 TIER_M으로 진입
2. TIER_M/L에서는 `_create_tool_registry()` (24개 전체) 사용 — DocumentProcess
   가 Worker 풀에 자동 포함
3. `components["scout_provider"]`는 None으로 설정되고 ModelDispatcher의
   `scout_available=False`. AgentTool이 Scout 호출 시 명확한 에러 반환
4. Claude Code 원래 구조(단일 Worker + 24 도구)로 자동 수렴 — 추가 수정
   불필요

**공공 납품 방향 권고 (2026-04-17)**:
사용자와 함께 관찰한 통찰 — 현재 겪는 일련의 문제(Phase 2 실패, Phase 3
재학습, Scout 이동 논의)는 전부 "GPU가 8K로 묶여 있어 생기는 파생 문제"다.
본질은 RTX 5090의 KV 캐시 한계.
- 공공 납품 사양 논의 초기에 **H100 80GB 포함 여부**를 테이블에 올릴 것
- 포함되면 Scout 복잡도를 점진적으로 제거 가능하며, 코드 변경 없이 티어
  감지만으로 단순화
- 포함되지 않으면 본 개정(Part 2.3) 상태가 계속 유효

### 실측 검증 계획
웹 서버 재기동 후 15KB docx 파일 업로드:
- Worker가 Agent(subagent_type="scout") 호출 확인
- Scout가 DocumentProcess로 청크 순차 처리
- 요약(200~500토큰)만 Worker에 반환
- 최종 응답 완성 + `/metrics agents.scout.calls` 증가

---

## Phase 3 LoRA 학습 성공 (2026-04-17, 1차 통과)

### 배경
Phase 2 LoRA에서 두 가지 회귀 발견:
1. tool_call 직렬화가 원시 JSON이라 vLLM qwen3_xml 파서가 인식 못 함 →
   assistant 본문에 `{"name": "Agent", "arguments": ...}` 그대로 노출
2. direct_answer 샘플이 짧은 인사·단답 중심이라 장문 설명 능력 퇴보

### 개선
**bootstrap_generator 재구성** (비율: 도구 45% / 추론 25% / 서브에이전트 15% / 지식 15%):
- `_KNOWLEDGE_TEMPLATES` 신규 16건 — OOP/REST/GIL/ACID/HTTP2/CAP/TCP-UDP/
  JWT/K8s/Git/Docker/SQL/ML/테스트/async/GraphQL 각 3~6단락 구조화 답변
- `_SUBAGENT_TEMPLATES` use_scout 프롬프트 10→30 확대 (프로젝트 구조/다중
  파일 검색/리팩토링/영문 프롬프트 다양화)
- `_generate_knowledge_sample()` 추가

**train_qwen_lora_phase3.py (신규)**:
- `tokenizer.apply_chat_template(messages, tools=[AGENT_SCHEMA])` 사용
  → tool_calls가 Qwen3.5 공식 XML 포맷으로 자동 직렬화
    `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
- vLLM `qwen3_xml` 파서와 완전 호환
- 학습 결과: **train_loss=0.06946** (Phase 1 0.073, Phase 2 0.168 대비 최저)
- 학습 시간: 71분, 1000 샘플 × 3 epoch

**자동 검증 스크립트 (`scripts/_verify_phase.py`)**:
5개 시나리오 회귀 검사 — 인사/장문 지식/단일 도구/대규모 탐색/tool_call leak

### 1차 검증 결과 (nexus-phase3 적용 후)

| # | 시나리오 | 결과 | 상세 |
|---|---|---|---|
| 1 | 짧은 인사 | **PASS** | 3.4초, 10 토큰 |
| 2 | 장문 지식 (GIL) | **PASS** | 21.3초, **252 토큰** (Phase 2 대비 6배) |
| 3 | 단일 파일 탐색 | **PASS** | 9.0초, 81 토큰, 도구+답변 |
| 4 | 대규모 탐색 → Agent(scout) | **PASS** | 57.8초, scout_calls 0→1 증가 |
| 5 | tool_call JSON/XML 누출 없음 | **PASS** | 4개 시나리오 모두 clean |

**통과 5/5 — 1차 학습만으로 전 시나리오 통과.** Phase 1 롤백 없이 Phase 3 유지.

### 설정 반영
- `config/nexus_config.yaml`: `primary_model: "nexus-phase3"`
- vLLM: `--lora-modules nexus-phase1=... nexus-phase2=... nexus-phase3=...`
  (세 어댑터 모두 핫로드, 런타임 전환 가능)

### 교훈
- Qwen3.5 chat template의 tool_call 공식 포맷은 중첩 XML(nested function/parameter)
- 학습 데이터 직렬화는 반드시 `tokenizer.apply_chat_template`으로 → 수동 조립은
  포맷 drift 유발
- bootstrap_generator의 카테고리 분포 편중이 응답 스타일 퇴보를 직접 야기함
  (Phase 2의 "짧은 답변 편중" = 장문 능력 퇴보)

---

## Scout 모델 Qwen3.5-4B 전환 + Phase 2 LoRA 학습 완료 (2026-04-17)

### 동기
B 방식 전환 후 초기 실측에서 두 이슈 발견:
1. Scout 모델(Gemma 4 E4B)이 Qwen3.5-27B Worker와 이기종 — 토크나이저/
   chat template 불일치로 hand-off 비용 발생
2. 기존 Phase 1 LoRA는 Agent 도구 사용 경험이 없어 Worker가 필요 상황에서도
   Scout를 호출하지 않고 직접 LS/Glob/Read로 탐색 (컨텍스트 소진)

### 해결 — 두 축 동시 교체
**축 1: Scout 모델 동종화 (Gemma 4 E4B → Qwen3.5-4B)**
- 선택: `unsloth/Qwen3.5-4B-GGUF` Q4_K_M (2.6GB, Worker와 동일 패밀리)
- GPU 서버에 `/opt/nexus-gpu/models/qwen3.5-4b-gguf/` 배치
- llama.cpp 재기동: `--jinja` 플래그로 ChatML + tool_calls 지원
- 코드 변경: `ScoutConfig.model_id = "qwen3.5-4b"`, SCOUT_AGENT description 갱신

**축 2: Phase 2 LoRA 학습 (Agent 도구 사용 학습)**
- Bootstrap 데이터 1000개 재생성 (도구 60% / 추론 30% / **서브에이전트 10%**)
- 서브에이전트 시나리오 세분화:
    subagent_use_scout: 22건 (긍정 — 대규모 탐색)
    subagent_direct_answer: 28건 (부정 — 인사/일반 지식)
    subagent_single_tool: 50건 (부정 — 단일 파일 작업)
  → 부정 샘플이 긍정의 3.5배로 Scout 남용 억제
- GPU 서버에서 Phase 2 학습: `scripts/train_qwen_lora_phase2.py`
  - Worker vLLM 중단 → unsloth 4bit + LoRA r=8, lr=3e-4, 3 epoch
  - `train_runtime: 3566초 (59분)`, `train_loss: 0.1681` (Phase 1 0.073 대비
    카테고리 다양화로 소폭 상승, 정상 범위)
  - 체크포인트: `/opt/nexus-gpu/checkpoints/qwen35-phase2/` (159MB)
- vLLM 재기동: `--lora-modules nexus-phase1=... nexus-phase2=...` 둘 다 노출
- `config/nexus_config.yaml`: `primary_model: "nexus-phase2"`로 전환

### 실환경 검증 (Phase 2 LoRA 적용 후)

| 시나리오 | Phase 1 + Gemma 4 E4B | **Phase 2 + Qwen3.5-4B** |
|---|---|---|
| "Hi there" (인사) | 4초, Agent 0회 ✅ | 4초, Agent 0회 ✅ |
| "core/orchestrator 개요" | 23초, 불완전 응답 | **23초, 한글 요약 완성** |
| "5계층 권한 시스템 전수 조사" | 시도 안 함 | **45초, Agent(scout) 자율 호출** ✅ |

중요 관찰: 5계층 권한 탐색 같은 대규모 요청에서 Worker가 **스스로**
"scout sub-agent를 사용하는 것이 적합합니다"라고 판단 후 Agent 도구를
호출함. Phase 2 학습 전에는 일어나지 않던 행동.

### Scout 메트릭 (/metrics)
```json
"agents": {
    "scout": {
        "calls": 1,
        "total_latency_ms": 25156.0,
        "avg_latency_ms": 25156.0
    }
}
```
Scout 지연 시간 25초 — Gemma 4 E4B 대비 **약 28% 단축** (35초 → 25초).

### 알려진 후속 이슈
- Scout 응답 품질: 25초에 29자 짧은 응답. Qwen3.5-4B Q4_K_M의 CPU 추론
  한계. 개선 방안: 양자화 완화(Q5/Q6) 또는 Scout 전용 후속 LoRA.
- 서브에이전트 학습 샘플 22개는 여전히 적음. 실 사용 데이터 누적 후
  Phase 3 재학습으로 보강 예정.

### 사양서/코드 영향
- `config/nexus_config.yaml` — primary_model을 nexus-phase2로
- `core/config.py`, `core/model/scout_provider.py` — model_id = qwen3.5-4b
- `core/orchestrator/agent_definition.py` — description에 Qwen3.5-4B 명시

---

## v7.0 Phase 9 B 방식 전환 — Scout를 서브에이전트로 승격 (2026-04-17)

### 배경
A 방식(Dispatcher 자동 Scout 전처리) 배선 직후 실측해보니, TIER_S에서는
"안녕" 같은 단순 인사에도 Scout가 조건 없이 선행 실행되어 ~33초가 걸렸다.
Scout는 CPU 4B(llama.cpp) 모델이라 모든 요청에 고정 오버헤드를 더하는 구조가
실사용에서 불합리했다.

사용자 결정: **B 방식으로 전환** — Scout를 자동 전처리기가 아니라
"Worker가 필요할 때 호출하는 서브에이전트"로 승격시킨다. 납품 일정보다 올바른
아키텍처가 우선.

### 구현 (10단계, 각 Phase별 커밋 분리)

**Phase 1 (e134a4c)**: `core/orchestrator/agent_definition.py` 신규
- `AgentDefinition` frozen dataclass (name/desc/system_prompt/allowed_tools/max_turns/model_override)
- `AgentRegistry` — register/get/list_names/list_descriptions
- `SCOUT_AGENT` 상수 — Read/Glob/Grep/LS, max_turns=3, model_override="scout"
- `build_default_agent_registry()` 팩토리
- 21개 테스트 통과

**Phase 2 (4f57de5)**: `core/tools/implementations/agent_tool.py` 확장
- `subagent_type` 파라미터 추가, AgentRegistry 조회
- `model_override="scout"` → `context.options["scout_provider"]` 선택
- `allowed_tools` 기반 도구 필터링 (+ DISALLOWED 이중 보호)
- `AgentTool._stats` 클래스 레벨 통계 + `get_stats()/reset_stats()`
- 하위 호환: subagent_type 없으면 description 기반 ad-hoc 동작
- 14개 테스트 통과

**Phase 3 (242e02b)**: ModelDispatcher의 Scout 자동 전처리 제거
- `_run_scout()` 메서드 완전 제거
- `route()`는 모든 티어에서 passthrough (항상 Worker 직행)
- `stats`는 하위 호환 키 유지하되 값 항상 0, `note` 필드로 AgentTool 안내
- 기존 테스트 3개 제거, 1개 재작성(`test_route_tier_s_now_passthrough`)

**Phase 4 (098896d)**: Worker에게 AgentTool 노출
- `_create_cli_tool_registry`, `_create_web_tool_registry`에 AgentTool() 등록
- `ToolUseContext.options`에 agent_registry/scout_provider/available_tools 주입
- `_build_default_system_prompt(agent_registry)` — 동적 서브에이전트 목록 +
  "NEVER invoke scout for trivial tasks — it is slow" 경고 삽입
- 웹 전용 QueryEngine에도 동일 처리

**Phase 5 (b64a959)**: `/metrics`에 `agents` 섹션 노출
- `AgentTool.get_stats()` → `result["agents"]`
- 하위 호환 `result["scout"]`은 AgentTool 값 평탄화 + note 필드

**Phase 6**: 테스트 정비 (Phase 2/3에 포함)

**Phase 7 (8ca8610)**: `training/bootstrap_generator.py` 서브에이전트 시나리오
- `_SUBAGENT_TEMPLATES` 추가 (use_scout/direct_answer/single_tool 3가지)
- 생성 비율 조정: 도구 60% / 추론 30% / 서브에이전트 10%
- 부정 샘플(direct_answer + single_tool) 비중 높여 남용 억제
- `_generate_subagent_sample()` 추가

**Phase 8 (GPU 서버 작업, 이 저장소 범위 밖)**: LoRA 재학습
- scripts/train_qwen_lora.py 재실행 → qwen35-phase2 체크포인트
- vLLM `--lora-modules nexus-phase2=...`로 핫로드 예정

**Phase 9**: 사양서 v7.0 AMENDMENT 개정
- Part 2 (멀티모델 디스패치 계층) — B 방식 재설계 내용으로 갱신
- Part 5 Ch 14 — AgentDefinition/AgentRegistry/SCOUT_AGENT 정식 명세
- Part 5 Ch 17 — AgentTool.get_stats 기반 메트릭 구조 명시

**Phase 10**: 이 문서 업데이트

### 실환경 검증 (웹 서버 재시작 후)

"Hi, how are you today?" 요청:
- **3초 내 응답 완료**, `tool_calls: []`
- Worker 내부 추론: "simple greeting question. I should respond directly without using any tools"
- Scout 미호출 ✅

"Give me a high-level overview of the core/orchestrator directory" 요청:
- Worker가 `Agent(subagent_type="scout")` 정확히 선택
- 로그: `Agent: 서브에이전트 실행 시작 (type=scout, tools=4개, model_override=scout)`
- Scout 35.3초 실행, Worker가 최종 응답 생성
- `/metrics` 응답: `agents.scout.calls=1, avg_latency_ms=35266`

### 알려진 후속 이슈

- Scout(Gemma 4 E4B) tool_call 실행 품질이 낮음 — 29자 짧은 응답 반환 사례
  Worker가 fallback으로 직접 도구 사용으로 전환되는 합리적 대처 확인
  해결: Phase 8 재학습 이후 Scout 모델을 Qwen 계열 소형으로 교체 고려
- Phase 8 재학습 전에는 Worker의 Scout 선택이 시스템 프롬프트에만 의존 →
  "NEVER invoke for trivial" 경고로 1차 억제, 실측 결과 충분히 보수적으로 동작

### 테스트: 572개 통과 (기존 571 + /metrics agents 섹션 1개)

---

## v7.0 Phase 9 실 배선 + Ch 17 Scout 메트릭 (2026-04-17)

### 발견한 문제
v7.0 Phase 9 코드(HardwareTier/TurnState/ModelDispatcher/Scout)는 모듈
단위로 구현됐고 487줄의 유닛테스트까지 통과했으나, **실제 실행 경로에
연결되지 않은 상태**였다:
- bootstrap에서 `scout_provider`는 만들지만 `ModelDispatcher`는 생성 안 됨
- `QueryEngine.submit_message`가 `model_provider`를 받아 `query_loop`를 직접
  호출 → Scout는 한 번도 실행되지 않음
- 웹 서버 로그에는 "Scout 초기화 성공"이 찍히지만 실 사용자 요청 처리 시
  Scout는 호출되지 않는 "유령 구성" 상태

### 해결
1. **ModelDispatcher 배선** — bootstrap.py에 ⑩ 단계로 Dispatcher 생성 추가,
   `components["model_dispatcher"]`로 저장, QueryEngine 생성자에 주입
2. **QueryEngine 이중 경로** — `model_dispatcher` 파라미터가 주입되면
   `dispatcher.route()` 호출, 없으면 기존 `query_loop()` 직접 경로로 폴백
   (하위 호환 유지)
3. **Scout 전용 도구 팩토리** — `_create_scout_tool_registry()` 신규
   (Read/Glob/Grep/LS 4개, 사양서 Part 2.3 SCOUT_TOOLS와 일치)
4. **웹 전용 Dispatcher** — `web/app.py`에서 웹 도구 8개로 별도 Dispatcher
   생성 (scout_provider/scout_tools는 bootstrap 인스턴스 재사용)
5. **Ch 17 Scout 메트릭 노출** — `ModelDispatcher.stats`에
   `scout_avg_latency_ms`, `scout_fallback_count` 추가, `_run_scout` 성공 시
   elapsed_ms 누적. `/metrics` 엔드포인트에 `result["scout"]` 섹션 추가
6. **사양서 동기화** — v7.0 AMENDMENT의 Ch 3 / Ch 5.9 / Ch 17 블록을 실제
   구현과 일치하도록 갱신 (dispatcher= → model_dispatcher= 등)

### 신규 테스트
- `tests/unit/test_model_dispatcher.py::TestModelDispatcherStatsFields` — 4개
- `tests/unit/test_query_engine_dispatcher.py` — 4개 (주입 경로/폴백/프로퍼티)

### 테스트: 540개 통과 (기존 532 + 신규 8)

---

## 다음 세션 시작 시 참고

1. **Phase 0.5~8.0 + v7.0 전체 완료 + 모델 Qwen 3.5 27B 전환**
2. **기본 참조 문서**: `user_mig/PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md`
3. **웹 서버 실행**: `python -m uvicorn web.app:app --host 0.0.0.0 --port 8443 --ssl-keyfile config/ssl/key.pem --ssl-certfile config/ssl/cert.pem`
4. **CLI 실행**: `python -m cli.commands chat`
5. **서버 정보**:
   - GPU Worker: 192.168.22.28:8001 (Qwen 3.5 27B AWQ + LoRA, qwen3_xml parser)
   - GPU Scout: 192.168.22.28:8003 (Gemma 4 E4B, llama.cpp CPU)
   - Embedding: 192.168.22.28:8002 (e5-large)
   - DB: 192.168.10.39:5440 (PostgreSQL nexus), :6340 (Redis db=6)
   - 웹: https://192.168.22.223:8443
6. **GPU 서버 모델 목록**:
   - /opt/nexus-gpu/models/qwen3.5-27b-awq/ (21GB, 서빙용)
   - /opt/nexus-gpu/models/qwen3.5-27b/ (52GB, 학습용)
   - /opt/nexus-gpu/models/gemma-4-e4b-it-gguf/ (4.7GB, Scout)
   - /opt/nexus-gpu/models/gemma-4-31b-it-awq/ (20GB, 백업)
   - /opt/nexus-gpu/models/gemma-4-31b-it/ (59GB, 학습용 백업)
   - /opt/nexus-gpu/models/e5-large/ (9GB, 임베딩)
   - /opt/nexus-gpu/checkpoints/qwen35-phase1/ (153MB, LoRA 어댑터)
7. **vLLM 시작 명령 (GPU 서버)**:
   ```
   /opt/nexus-gpu/.venv/bin/python3.12 -m vllm.entrypoints.openai.api_server \
     --model /opt/nexus-gpu/models/qwen3.5-27b-awq \
     --max-model-len 8192 --gpu-memory-utilization 0.90 \
     --port 8001 --host 0.0.0.0 --trust-remote-code \
     --served-model-name qwen3.5-27b \
     --enable-prefix-caching --enforce-eager \
     --enable-auto-tool-choice --tool-call-parser qwen3_xml \
     --enable-lora --max-lora-rank 16 \
     --lora-modules nexus-phase1=/opt/nexus-gpu/checkpoints/qwen35-phase1
   ```
8. **추가 가능 작업**:
   - Ch 16 세션 관리 (JSONL 트랜스크립트, 세션 재개)
   - Qwen 3.5 thinking 출력 제어 (chat_template 수정)
   - Scout 모델을 Qwen 계열 소형으로 교체
   - 24시간 Soak Test
   - 웹 UI 문서 분석 개선 (청크 순차 분석 UX)
9. **알려진 제한**:
   - Qwen 3.5 thinking 텍스트가 응답에 포함될 수 있음
   - 8,192 컨텍스트 한계 (RTX 5090 32GB)
   - Scout(Gemma 4 E4B)는 Qwen 계열이 아님 (향후 교체 고려)
10. 532개 테스트 전부 통과
6. **서버 정보**:
   - GPU: 192.168.22.28 (Worker :8001, Embedding :8002, Scout :8003), max-model-len=8192
   - Scout: Gemma 4 E4B (Q4_K_M) on llama.cpp CPU, ~16 TPS
   - DB: 192.168.10.39 (PostgreSQL + Redis, 현재 미연결)
   - 웹: https://192.168.22.223:8443 (자체 서명 SSL)
7. 453개 테스트 전부 통과
