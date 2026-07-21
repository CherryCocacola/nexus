# Nexus Architecture Rules

## P1. 4-Tier AsyncGenerator Chain — 절대 불변 구조

모든 데이터 흐름은 4계층 AsyncGenerator 체인을 통과한다.
각 계층은 `async def ... yield StreamEvent` 형태이며, 이벤트는 하위→상위로 전파된다.

```
Tier 1: QueryEngine.submit_message()  — 세션 오케스트레이터
Tier 2: query() / query_loop()        — while(True) 에이전트 턴 루프
Tier 3: query_model_streaming()       — SSE 스트림 파싱
Tier 4: with_retry()                  — 재시도 + httpx 클라이언트
```

- 새 기능을 추가할 때 이 체인을 우회하거나 단축하지 않는다
- 각 Tier는 자기 하위 Tier의 AsyncGenerator만 소비한다 (Tier 1이 Tier 4를 직접 호출 금지)
- StreamEvent는 frozen Pydantic 모델 — 생성 후 수정 불가 (`core/message.py`, `frozen=True`)

### 구현 현황 주석 (2026-07-09 사양 감사 반영)

Tier 4 `with_retry`(`core/orchestrator/retry.py`)는 지수 백오프 재시도 **프리미티브**로
정의·단위테스트되어 있으나, **현재 프로덕션 스트리밍 경로에는 아직 배선되지 않았다**.
실제 일시 오류 처리는 다음 두 지점이 담당한다.
- **Tier 2(`query_loop`)**: `StreamWatchdog` 타임아웃(스트림 정지/GPU 행)을
  `model_error_count`/`MAX_MODEL_ERROR_RETRY`로 재시도.
- **Tier 3(`inference`)**: 컨텍스트 초과 재시도 루프 + 연결 오류를 `ERROR` StreamEvent로
  변환(즉시 상위 전파, 연결 레벨 백오프는 미적용).

즉 이름상 4-Tier이나 Tier 4의 백오프 재시도는 아직 프리미티브 상태다. httpx 왕복부를
`with_retry`로 감싸 연결 오류 백오프를 실제 적용하는 것은 **후속 과제**(TODO). 이 문단은
문서-구현 불일치(감사 MED)를 정직하게 명시하기 위한 것이다.

### 비활성 서브시스템 주석 (2026-07-21 확인)

아래 두 서브시스템은 **구현되어 있으나 현재 배포에서 도달 불가**다. 문서만 보고
"동작 중"이라고 오해하지 않도록 명시한다(코드는 남겨 둔다 — 사양 정의 구조이며
TIER_S 재사용 가능성이 있다).

**① Scout (4B 보조 모델) — tier=large에서 비활성**
- `bootstrap.py`는 `if tier == HardwareTier.TIER_S and config.scout.enabled` 일 때만
  `scout_provider`를 만든다. 그런데 **설정 3본(`nexus_config.yaml`/`.pc.yaml`/`.112.yaml`)이
  전부 `hardware_tier: "large"`** 이므로 조건이 참이 되지 않는다 → `scout_provider`는 항상 `None`.
- 따라서 도달 불가: `_create_cli_tool_registry()`(TIER_S 7개 풀),
  `_create_scout_tool_registry()`, `core/model/scout_provider.py`,
  `_build_default_system_prompt()`의 TIER_S 분기.
- `AgentTool`은 남아 있으나 웹 TIER_L 풀에만 있고, 호출돼도 `_resolve_model_provider`가
  **부모 Worker 모델로 폴백**한다(2026-07-13 수정). 유일한 서브에이전트 정의가 `scout`
  하나뿐이라 실질 위임 대상이 없다.
- CLI TIER_M/L 풀(23개)에는 `AgentTool`이 없다. 프롬프트도 이를 명시적으로 금지한다
  (`_build_expanded_system_prompt`) — 프롬프트↔도구 풀 불일치를 막기 위함.

**② core/thinking/ — 전체 미배선**
- `ComplexityAssessor` / `ThinkingStrategy`(DIRECT·HIDDEN_COT·SELF_REFLECT·MULTI_AGENT) /
  `hidden_cot` / `self_reflection` / `cache` 가 구현되어 있으나, **`core/thinking/` 밖의
  어떤 모듈도 이들을 import하지 않는다**(bootstrap·query_engine·web 전부 미참조).
- 즉 복잡도 기반 사고 전략 선택은 **현재 동작하지 않는다**. CLI의 `/thinking` 명령은
  이와 무관하며 모델이 낸 thinking 블록의 **화면 표시 토글**일 뿐이다.

## P2. 디렉토리 구조 및 의존성 방향

```
project-nexus/
  config/           # YAML 설정 파일
  core/
    orchestrator/   # query_loop, stream_handler, context_manager, stop_resolver
    model/          # model_manager, inference, gpu_detector, prompt_formatter
    tools/
      registry.py, executor.py, result_formatter.py
      implementations/   # 24개 도구 구현
      validation/        # path_validator, command_validator, schema_validator
    permission/     # 5계층 권한 파이프라인
    security/       # sandbox, path_guard, command_filter, audit
    hooks/          # hook_manager, hook_runner
    thinking/       # assessor, orchestrator, hidden_cot
    memory/         # manager, short_term(Redis), long_term(PG+pgvector)
    system_prompt/  # builder, templates/(Jinja2)
  training/         # strategy, trainer, data_collector, feedback_loop
  deployment/       # airgap_prep, offline_packages, integrity
  cli/              # repl(Rich), commands(Click), formatters
  web/              # app(FastAPI), middleware
  tests/            # unit/, integration/, e2e/
```

### 의존성 방향 (단방향만 허용)

```
cli/, web/ → core/              (진입점 → 코어)
core/orchestrator/ → core/model/, core/tools/, core/thinking/
core/tools/ → core/security/, core/permission/
training/ → core/              (core는 training을 절대 import하지 않음)
deployment/ → training/(부분), core/config
```

- 순환 의존(circular import) 발생 시: lazy import로 해결
- 계층을 역방향으로 import하는 코드는 즉시 거부

## P3. 표준 내부 계약 — OpenAI tool_calls 형식

모든 내부 컴포넌트(query_loop, tool_execution, streaming parser, training data)는
OpenAI `tool_calls` 형식만 사용한다.

- query_loop은 XML 형식을 절대 보지 않는다
- XML 폴백 파싱은 ModelAdapter 내부에 캡슐화
- 새 모델 추가 시에도 이 계약을 유지

## P4. 2-Machine 토폴로지

```
Machine A (Orchestrator): Python 3.11+ / asyncio
  — CLI, bootstrap, query loop, tools, permissions, memory, hooks
Machine B (GPU Server): Python 3.11+ / vLLM / FastAPI
  — Qwen 3.5 27B, ExaOne 7.8B, e5-large, LoRA hot-loading
```

- 두 머신은 LAN-only HTTP/SSE로 통신
- Machine A의 코드가 GPU/CUDA를 직접 호출하지 않는다
- 모든 모델 추론은 Machine B의 OpenAI 호환 API를 통해서만 수행

## P5. Pydantic v2 데이터 모델

- 모든 데이터 구조는 Pydantic BaseModel 또는 frozen dataclass 사용
- Factory method 패턴: `Message.user()`, `ToolResult.success()`, `ToolResult.error()`
- 불변 객체 우선: StreamEvent, PermissionContext, AgentDefinition은 `frozen=True`
- Union 타입은 `type` 필드로 discriminated union 구성

## P6. Fail-Closed 기본값

- BaseTool의 모든 behavior flag는 가장 제한적인 값이 기본값
  - `is_read_only=False`, `is_concurrency_safe=False`, `requires_confirmation=False`
- 권한 시스템: 5개 레이어 모두 통과해야 실행 허용
- 새 도구 추가 시 명시적으로 flag를 완화하지 않으면 쓰기/순차실행/확인필요로 동작

## P7. 설정은 YAML, 로그는 JSONL

- 설정: `config/*.yaml` (nexus_config, tool_mappings, model_profiles, permission_rules, logging)
- 감사 로그: JSONL 형식 (10MB 로테이션)
- 에이전트 기록: `~/.nexus/agent_transcripts/{name}_{timestamp}.jsonl`
- 학습 데이터: JSONL 형식
- 시스템 프롬프트 템플릿: Jinja2
