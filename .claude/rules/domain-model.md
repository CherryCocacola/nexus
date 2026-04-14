# Nexus Domain Model Definitions

코드, 변수, 주석에서 아래 정의된 용어를 정확히 사용한다. 동의어 혼용 금지.

## Core 엔티티

### StreamEvent (스트리밍 이벤트)
- frozen dataclass — 생성 후 수정 불가
- 16가지 EventType: TEXT_DELTA, TOOL_USE_START, TOOL_USE_DELTA, TOOL_USE_STOP,
  THINKING_DELTA, ERROR, TOOL_RESULT, CONTEXT_COMPACT, TURN_START, TURN_END 등
- 4-Tier 체인 전체에서 사용하는 유일한 데이터 전달 단위

### Message (메시지)
- Pydantic v2 BaseModel
- Role: USER, ASSISTANT, TOOL_RESULT, SYSTEM
- ContentBlock: TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock (discriminated union)
- Factory: `Message.user()`, `Message.assistant()`, `Message.tool_result()`, `Message.system()`

### Conversation (대화)
- Message의 컨테이너
- `compact_boundary`: 컨텍스트 압축 경계 인덱스
- `get_active_messages()`: 압축 경계 이후의 메시지만 반환

### BaseTool (도구)
- ABC, 24개 구현체
- Identity: name, description, aliases, group
- Schema: input_schema (JSON Schema dict)
- Behavior: is_read_only, is_concurrency_safe, is_destructive 등 (fail-closed 기본값)
- Lifecycle: validate_input() → check_permissions() → call() → map_result()

### ToolResult (도구 결과)
- Pydantic model: data, is_error, error_message, metadata
- Factory: `ToolResult.success(data)`, `ToolResult.error(message)`

### QueryEngine (쿼리 엔진)
- 세션 오케스트레이터 (Tier 1)
- 시스템 프롬프트 조립, 입력 전처리, 사용량 추적, 예산 강제

### TurnResult (턴 결과)
- 한 턴의 누적 결과: assistant_text, tool_calls, stop_reason, usage, continue_reason
- ContinueReason: COMPLETED, NEXT_TURN, REACTIVE_COMPACT, MAX_OUTPUT_RECOVERY 등 7가지

### AgentDefinition (에이전트 정의)
- frozen Pydantic model
- name, description, system_prompt, tools (필터 리스트), max_turns, model_override
- 서브 에이전트는 독립된 QueryEngine을 가짐 (messages[] 격리)

### TaskState (태스크 상태)
- TaskType: LOCAL_BASH, LOCAL_AGENT, REMOTE_AGENT, WORKFLOW, MONITOR, TRAINING
- TaskStatus: PENDING, RUNNING, COMPLETED, FAILED, KILLED
- progress: 0.0 ~ 1.0

### GlobalState (전역 상태)
- 싱글톤 — 모듈 레벨에서 하나만 존재
- ~100개 필드: 세션, 설정, 모델 상태, 도구 레지스트리 등

## Permission 엔티티

### PermissionMode (7가지)
DEFAULT, ACCEPT_EDITS, BYPASS_PERMISSIONS, DONT_ASK, PLAN, AUTO, BUBBLE

### PermissionBehavior (3가지)
ALLOW, DENY, ASK

### ToolCategory (7가지)
READONLY, FILE_WRITE, BASH, DANGEROUS, NETWORK, AGENT, MCP

### PermissionPipeline (5계층)
Layer 1: DenyRuleFilter — 도구 자체를 제거 (모델이 호출 불가)
Layer 2: FilePermissionChecker / BashPermissionChecker — 경로/명령어 분석
Layer 3: CanUseToolHandler — 모드별 사용자 확인
Layer 4: HookPermissionChecker — Hook 기반 승인/차단
Layer 5: apply_context_resolution() — 최종 모드 보정

## 용어 규칙

| 정확한 용어 | 금지 용어 |
|---|---|
| Tool (도구) | plugin, extension, action |
| Query Loop (쿼리 루프) | main loop, agent loop, run loop |
| StreamEvent | event, message (StreamEvent과 Message는 구분) |
| Turn (턴) | iteration, cycle, round |
| Tier (계층) | layer (Permission의 Layer와 혼동 방지) |
| Permission Layer (권한 레이어) | tier (4-Tier 체인의 Tier와 혼동 방지) |
| Compact (압축) | summarize, truncate |
| Air-gap (에어갭) | offline, disconnected |
