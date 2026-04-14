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
- StreamEvent는 frozen dataclass — 생성 후 수정 불가

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
  — Gemma 4 31B, ExaOne 7.8B, e5-large, LoRA hot-loading
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
