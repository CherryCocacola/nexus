<!-- Point 4.4 — TodoWrite식 계획·진행 추적 도구(체크리스트) 상세 설계 문서 -->

# Point 4.4 — TodoWrite식 계획·진행 추적 도구 설계

- 작성일: 2026-07-08
- 상태: 설계(구현 전)
- 대상 브랜치: feature/b200-bakeoff
- 관련 규칙: architecture.md P2/P5/P6, anti-patterns #5/#7/#12, domain-model.md, agent-collaboration.md

---

## 1. 개요·동기

### 1.1 목표

Claude Code의 TodoWrite처럼, 모델이 장기·다단계 작업 중 **계획 체크리스트**
(항목 + 상태 `pending`/`in_progress`/`completed`)를 스스로 생성·갱신하며
작업을 추적하는 도구를 추가한다.

기대 효과.

1. **일관성** — 장기 자율 작업(수십 턴)에서 모델이 원래 계획을 잊지 않는다.
   특히 TIER_S(8K 컨텍스트) 환경에서 compact 이후에도 체크리스트가 시스템
   프롬프트로 재주입되어 작업 맥락이 유지된다.
2. **가시성** — 사용자(웹/CLI)가 "지금 몇 번째 단계를 하고 있는지"를 실시간
   체크리스트로 확인할 수 있다. 최근 커밋 "채팅 UX 클로드화"(d1041bd)의
   activity trail UI와 자연스럽게 결합된다.
3. **학습 데이터 품질** — 턴별 계획-실행-완료 궤적이 구조화되어 QLoRA 학습
   데이터(JSONL)에 그대로 남는다.

### 1.2 현재 상태와 문제

현재 `core/orchestrator/turn_state.py`의 `TurnState.todo`가 유일한 "할 일"
추적 수단인데, 이는 **어시스턴트 텍스트에서 키워드("TODO", "해야", "need to")를
휴리스틱으로 긁어내는 수동 추출**이다(turn_state.py `extract_turn_state()` 3단계).
모델이 능동적으로 계획을 선언·갱신하는 수단이 없고, 상태(진행 중/완료) 개념도 없다.

---

## 2. 기존 task_tools와의 차이 — 왜 별도 설계가 필요한가

### 2.1 기존 도구 분석 (core/tools/implementations/task_tools.py)

현재 `TodoRead`/`TodoWrite`/`Task` 3종이 등록되어 있으나(core/bootstrap.py
889~891행), 이들의 실체는 **비동기 백그라운드 태스크 관리자**다.

| 관점 | 기존 TodoWrite (task_tools.py) | 목표 (Claude Code식 체크리스트) |
|---|---|---|
| 백엔드 | `core.task.TaskManager` — asyncio.Task 실행체 관리 | 세션별 인메모리 계획 상태 (실행체 없음) |
| 단위 | 태스크 **1건** 생성/수정 (task_id 기반) | 체크리스트 **전체 목록** 원자적 교체 |
| 상태 | TaskStatus: PENDING/RUNNING/COMPLETED/FAILED/KILLED (실행 상태) | pending/in_progress/completed (계획 진행 상태) |
| 용도 | LOCAL_BASH/LOCAL_AGENT/MONITOR/TRAINING 등 실행 제어(kill 포함) | 모델의 사고 정리 + 사용자 가시성 (순수 메타데이터) |
| 권한 | ASK (사용자 확인 필요) | ALLOW (자율 작업 중 무확인 갱신 필수) |
| 세션 격리 | 없음 (전역 TaskManager / 모듈 전역 `_fallback_tasks`) | 세션·에이전트별 격리 필수 |

즉 **기존 TodoWrite는 이름만 Todo일 뿐 TaskTool(create/update/list/get/stop)의
부분집합**이며, 계획 체크리스트가 아니다. 기능적으로도 TaskTool과 거의 완전
중복이다(task_tools.py 자체 주석에도 "TodoRead와 동일한 방식의 목록 조회" 명시).

### 2.2 사양서 원문 대조 — "중복"이 아니라 "사양 복귀"

v6.1 사양서의 24개 도구 표(28165행)는 이렇게 정의한다.

```
| 9  | TodoRead  | Active | Read task list  |
| 10 | TodoWrite | Active | Write task list |
```

원 사양의 의도는 "task **list**를 읽고/쓴다"이며, 이는 Claude Code TodoWrite의
전체-목록 교체 시맨틱과 일치한다. 현재 구현(단건 CRUD + TaskManager 결합)이
오히려 사양에서 드리프트된 상태다. **따라서 본 설계는 신규 중복 도구 추가가
아니라, 기존 TodoRead/TodoWrite 두 이름의 시맨틱을 사양 의도(체크리스트)로
교정하는 것**이다. 백그라운드 태스크 제어는 이미 `Task` 도구(create/update/
list/get/stop)가 온전히 담당하므로 기능 손실이 없다.

### 2.3 대안 비교

| 안 | 내용 | 장점 | 단점 |
|---|---|---|---|
| **A안 (권장)** | 기존 TodoRead/TodoWrite를 체크리스트 시맨틱으로 교체. 태스크 제어는 Task 도구로 일원화 | 24개 도구 수 유지, Claude Code 관례·v6.1 표와 이름 일치, 중복 제거 | 기존 학습 데이터(bootstrap_generator.py 309~343행)·테스트 갱신 필요 |
| B안 | 새 이름(예: `PlanWrite`) 추가, 기존 3종 유지 | 기존 코드 무변경 | 도구 25개로 증가(사양 이탈), Todo라는 이름이 체크리스트가 아닌 혼란 영속, TaskTool과의 중복 방치 |

**권장: A안.** 리스크(9장)에서 B안 대비 마이그레이션 비용을 상세히 다룬다.

---

## 3. 도구 설계

### 3.1 단일 도구 vs Read/Write 분리 → **Write 중심 + 경량 Read 유지**

- `TodoWrite` — 체크리스트 **전체를 원자적으로 교체**하는 유일한 쓰기 경로.
  Claude Code와 동일하게 부분 수정(patch) API를 두지 않는다.
  - 왜 전체 교체인가: (1) 멱등적 — 같은 호출을 재시도해도 결과 동일,
    (2) id 관리 불필요 — 소형 모델(Qwen 27B)이 task_id를 기억·전달하다
    틀리는 오류 모드를 원천 제거, (3) 순서가 곧 계획 순서 — 배열 그대로 렌더.
- `TodoRead` — 현재 체크리스트 조회(읽기 전용). 24개 도구 수 유지 겸,
  compact 직후나 서브에이전트 시작 시 모델이 스스로 상태를 복원할 때 쓴다.
  단, 매 `TodoWrite` 결과에 목록 전문이 에코되고(3.4) 시스템 프롬프트에도
  주입되므로(6장) 호출 빈도는 낮을 것으로 예상한다.

### 3.2 TodoWrite input_schema

```python
@property
def input_schema(self) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "체크리스트 전체 (이 배열이 기존 목록을 통째로 대체합니다)",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "할 일 내용 (명령형, 예: '권한 파이프라인 테스트 실행')",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "항목 상태",
                        },
                        "active_form": {
                            "type": "string",
                            "description": "진행 중일 때 UI에 표시할 현재진행형 문구 (예: '권한 파이프라인 테스트 실행 중')",
                        },
                    },
                    "required": ["content", "status", "active_form"],
                },
            },
        },
        "required": ["todos"],
    }
```

주의점.

- 키 이름은 프로젝트 파이썬 관례에 맞춰 `active_form`(snake_case)로 한다.
  Claude Code 원형은 `activeForm`이지만, Nexus 도구 스키마는 전부
  snake_case(`file_path`, `status_filter` 등)이므로 일관성이 우선이다.
  다만 소형 모델이 Claude Code 학습 잔재로 `activeForm`을 보낼 가능성에
  대비해 `validate_input()`에서 `activeForm` → `active_form` 관용 매핑을
  허용한다(에러 대신 정규화).
- TodoRead의 input_schema는 `{"type": "object", "properties": {}, "required": []}`
  (인자 없음)로 단순화한다. 기존 `status_filter`는 체크리스트 규모(≤50항목)에서
  불필요하다.

### 3.3 Behavior Flag (core/tools/base.py 기준)

| 플래그 | TodoWrite | TodoRead | 근거 |
|---|---|---|---|
| `is_read_only` | **False** (기본값 유지) | **True** | TodoWrite는 세션 상태를 변경하므로 읽기 전용이라 선언하면 거짓. fail-closed 기본값을 그대로 둔다 |
| `is_concurrency_safe` | **False** (기본값 유지) | **True** | 전체 교체 쓰기라 동시 실행 시 마지막 쓰기가 이기는 레이스 발생 가능. anti-pattern #12에 따라 순차 파티션에 남긴다 |
| `is_destructive` | False | False | 파일·시스템 영향 없음 |
| `requires_confirmation` | False | False | 자율 갱신이 목적 — 확인 강제 금지 |
| `should_defer` | False | False | 계획 갱신은 즉시 반영되어야 UI 가시성이 산다 |
| `group` | `"task"` | `"task"` | 기존 그룹 유지 |
| `timeout_seconds` | 기본 120s (실측 <1ms라 무의미하지만 오버라이드 불필요) | 동일 | 최소 변경 원칙 |

즉 **TodoWrite는 플래그를 하나도 완화하지 않는다**(P6 fail-closed 그대로).
자동 허용은 플래그가 아니라 권한 카테고리(7장)에서 해결한다.

### 3.4 결과(ToolResult) 계약

```python
# 성공 시 — data는 사람이 읽는 체크리스트 전문, metadata에 구조화 원본
return ToolResult.success(
    _render_checklist(items),          # "[x] A\n[~] B ← 진행 중\n[ ] C" 형태
    todos=[i.model_dump() for i in items],   # UI가 파싱할 구조화 데이터
    stats={"total": n, "completed": c, "in_progress": p},
)
```

- **data(모델이 다시 읽는 내용)**: 갱신된 목록 전문을 에코한다. 모델이
  다음 턴에서 자기 계획을 재확인하는 비용이 0이 된다(TodoRead 불필요).
- **metadata.todos**: 웹 서버가 tool_result에서 구조화 데이터를 추출해 UI
  프레임을 만든다. 이는 이미 DocumentExport 다운로드 URL에서 검증된 패턴
  ("다운로드 URL은 tool_result에서 서버 추출, 모델 텍스트 X")과 동일하다.

### 3.5 도메인 검증 (validate_input)

JSON Schema 통과 후 추가 검증.

1. 항목 수 상한: 50개 초과 시 에러 문자열 반환 (컨텍스트 폭주 방지).
2. `content` 공백 문자열 금지, 500자 초과 시 에러.
3. `in_progress` 항목이 2개 이상이면 **에러가 아니라 경고 메타데이터**로만
   기록한다(soft rule). 이유: 병렬 하위 작업 표현을 하드 차단하면 모델이
   재시도 루프에 빠질 수 있다. 시스템 프롬프트에서 "동시에 하나만"을
   지시(6장)하고, 위반은 학습 피드백 신호로 수집한다.
4. 알 수 없는 status 값은 에러(스키마 enum이 1차 방어, 여긴 2차).

---

## 4. 데이터 모델 (Pydantic v2)

새 파일 `core/task/todo_store.py`에 정의한다.

배치 근거: 의존성 방향(P2). 도구(core/tools)가 저장소 타입을 import해야
하는데, `core/orchestrator/`에 두면 tools→orchestrator 역방향 import가 되어
순환이 생긴다(orchestrator는 이미 tools를 import). `core/task/`는
task_tools.py가 이미 import하는 안전한 하위 계층이며, TaskManager와 나란히
"작업 상태" 도메인으로 응집된다.

```python
# core/task/todo_store.py — 세션별 계획 체크리스트 상태 저장소 (한글 헤더 주석 필수)

class TodoStatus(str, Enum):
    """체크리스트 항목의 진행 상태. TaskStatus(실행 상태)와 별개의 enum이다."""
    PENDING = "pending"          # 아직 시작 전
    IN_PROGRESS = "in_progress"  # 지금 진행 중 (원칙적으로 동시 1개)
    COMPLETED = "completed"      # 완료

class TodoItem(BaseModel):
    """체크리스트 항목 1개. frozen으로 두어 목록 교체 시 새 객체만 생성한다."""
    model_config = {"frozen": True}
    content: str                 # 할 일 내용 (명령형)
    status: TodoStatus = TodoStatus.PENDING
    active_form: str = ""        # 진행 중 표시용 현재진행형 문구

class TodoListState(BaseModel):
    """한 (세션, 에이전트) 쌍의 체크리스트 스냅샷."""
    model_config = {"frozen": True}
    items: tuple[TodoItem, ...] = ()
    revision: int = 0            # 교체 횟수 (UI 증분 갱신·디버깅용)
    updated_at: float = 0.0      # epoch 초
```

용어 규칙(domain-model.md) 준수: `TaskState`/`TaskStatus`(실행)와
`TodoItem`/`TodoStatus`(계획)를 이름부터 구분한다. 주석에도 혼용 금지를 명시한다.

### 4.1 TodoStore

```python
class TodoStore:
    """
    체크리스트 저장소 — TurnStateStore(core/orchestrator/turn_state.py)와
    동일한 '세션 키 인메모리 dict' 패턴. 키는 (session_id, agent_id or "main").
    """
    def __init__(self) -> None:
        self._lists: dict[tuple[str, str], TodoListState] = {}

    def replace(self, session_id: str, agent_id: str | None,
                items: list[TodoItem]) -> TodoListState: ...
    def get(self, session_id: str, agent_id: str | None) -> TodoListState: ...
    def clear_session(self, session_id: str) -> None: ...   # 세션 삭제 시 호출
    def to_context_string(self, session_id: str) -> str: ...  # 프롬프트 주입용
```

- **저장 위치 결정: GlobalState 필드가 아니라 독립 컴포넌트.**
  GlobalState(core/state.py)는 이미 ~100 필드 싱글톤이라 더 불리지 않는다.
  대신 `TurnStateStore`·`TaskManager` 선례를 그대로 따라 bootstrap이 생성해
  `components["todo_store"]`로 보관하고, 도구에는
  `context.options["todo_store"]`로 주입한다(task_tools.py의
  `_get_task_manager()` 주입 패턴과 동일).
- **폴백**: 주입이 없으면(단위 테스트 등) 모듈 전역 인메모리 dict 폴백 —
  task_tools의 `_fallback_tasks` 관례를 따르되, 세션 키를 유지한다.
- **영속성**: 1차 구현은 인메모리(프로세스 수명). 계획 체크리스트는 본질적으로
  세션 수명 데이터라 Redis 승격은 불필요하다. 단, 웹 세션 복원(재접속) 요구가
  생기면 TurnStateStore와 함께 short_term(Redis) 직렬화를 후속 과제로 남긴다
  (`# TODO(nexus)` 주석으로 표기).

---

## 5. 상태 저장·조회 흐름

### 5.1 턴 간 유지

```
[턴 N] 모델 → tool_calls: TodoWrite(todos=[...])
      → executor(13단계) → TodoWriteTool.call()
      → TodoStore.replace(session_id, agent_id, items)   # 원자적 교체
      → ToolResult(data=체크리스트 전문, metadata.todos=[...])
      → TOOL_RESULT StreamEvent로 상위 전파 (4-Tier 체인 준수, 우회 없음)

[턴 N+1] prompt_assembler가 TodoStore.to_context_string(session_id) 주입
      → 모델은 compact 이후에도 항상 최신 계획을 본다
```

- **4-Tier 체인 무결성**: 도구는 Tier와 무관한 executor 경로로 실행되고 결과는
  기존 TOOL_RESULT 이벤트로만 전파된다. 새 이벤트 타입을 추가하지 않는다
  (StreamEvent 16종 불변 — 웹 프레임 변환은 web/app.py 책임, 5.4 참조).
- **prompt_assembler 연동**: `core/orchestrator/prompt_assembler.py`는 이미
  `turn_state_store`를 선택 주입받아 이전 턴 요약을 붙인다(178~183행). 같은
  방식으로 `todo_store`를 선택 주입받아 "## 현재 작업 체크리스트" 섹션을
  주입한다(토큰 예산 상한 예: 300토큰, 초과 시 completed 항목은 개수만 축약).
- **TurnState.todo와의 관계**: `extract_turn_state()`의 키워드 휴리스틱(3단계)은
  유지하되, TodoStore에 목록이 존재하면 그것을 우선 사용하도록 후속 개선
  항목으로 남긴다(이번 범위에서는 두 경로가 공존해도 충돌 없음 — TurnState는
  요약, TodoStore는 원본).

### 5.2 세션 격리

키가 `(session_id, agent_id)`이므로 웹의 다중 세션·nexus_G2 공유 시나리오에서도
목록이 섞이지 않는다. 세션 삭제 API(web/app.py `delete_session`)에서
`TodoStore.clear_session()`을 함께 호출한다.

### 5.3 서브에이전트 격리

- `ToolUseContext.agent_id`(base.py 148행)가 이미 서브에이전트 구분자를
  제공한다. 서브에이전트의 TodoWrite는 자기 키에만 쓰므로 **메인 목록을
  덮어쓸 수 없다.**
- UI에는 메인 에이전트(agent_id=None → "main") 목록만 체크리스트 패널로
  노출한다. 서브에이전트 목록은 해당 에이전트 transcript에만 남는다.

### 5.4 웹 노출 흐름

web/app.py의 SSE 변환 루프(1108~1129행, TOOL_RESULT 분기)에서
tool 이름이 `TodoWrite`이고 `metadata.todos`가 있으면 기존 `tool_result`
프레임에 더해 **`todo_update` 웹 프레임**(웹 전용, StreamEvent 아님)을 추가로
내보낸다. 서버가 metadata에서 추출하므로 모델 텍스트 파싱이 필요 없다.

---

## 6. UI 노출

### 6.1 웹 (web/static/index.html)

최근 커밋 d1041bd의 activity 영역(`.activity-area`: trail/results/live) 위에
**고정 체크리스트 패널**을 추가한다.

- `todo_update` 프레임 수신 → `#todoPanel` 갱신 (전체 교체라 diff 불필요,
  revision으로 중복 프레임 무시).
- 렌더 형식: `[x]` 완료(취소선), `[~]` 진행 중(스피너 + `active_form` 문구
  강조), `[ ]` 대기. 접기/펼치기 토글, 완료 n/전체 m 카운터.
- 스트리밍 종료 후에도 패널은 유지한다(activity trail이 흐려질 때도 체크리스트는
  최종 상태를 보여주는 것이 목적).

### 6.2 CLI (cli/formatters.py, cli/repl.py)

- `formatters.py`에 `format_todo_list(items) -> Panel` 추가 — Rich Panel에
  체크박스 목록 렌더 (`✔`/`◐`/`○` + 진행 중 항목 굵게).
- `repl.py`의 TOOL_RESULT 처리 지점에서 도구명이 TodoWrite일 때 결과 본문
  대신 이 Panel을 출력한다(중복 텍스트 방지).

---

## 7. 권한 파이프라인 통과 설계

### 7.1 카테고리 결정

`core/permission/pipeline.py::_categorize_tool()`(229행)은 1단계 이름 매칭이
최우선이다. 현재 `TodoWrite`는 어느 집합에도 없어 3단계 기본값
**FILE_WRITE로 분류 → DEFAULT 모드에서 ASK**가 된다. 이 상태로는 자율 작업 중
매 갱신마다 확인이 떠서 도구가 무용지물이다.

**결정: 이름 기반 READONLY 분류에 `"todowrite"`, `"todoread"`를 명시 추가한다.**

```python
readonly_tools = {
    "read", "glob", "grep", "ls", "cat", "head", "tail",
    "taskget", "tasklist",
    # TodoRead/TodoWrite: 세션 내부의 계획 메타데이터만 갱신하며
    # 파일·프로세스·네트워크 등 외부 부작용이 전혀 없다.
    # PLAN 모드에서도 계획 수립 자체는 허용되어야 하므로 READONLY로 취급한다.
    # (도구 자체의 is_read_only 플래그는 False로 정직하게 유지 —
    #  executor의 동시성 파티셔닝은 플래그를, 권한은 이 카테고리를 본다)
    "todoread", "todowrite",
}
```

근거.

1. **MODE_BEHAVIOR_MAP 불변식과 정합** — "READONLY 열은 항상 ALLOW"
   (types.py 183행). 계획 갱신은 모든 모드(특히 PLAN 모드 — 계획 수립이
   본업)에서 허용되어야 하며, 이 불변식이 정확히 그 요구를 충족한다.
   Claude Code도 plan 모드에서 TodoWrite를 허용한다(동형성).
2. **fail-closed 원칙 훼손 없음** — 카테고리 완화는 "외부 부작용 없음"이
   구조적으로 보장되는 도구에 한한 명시적·주석 첨부 결정이다(P6이 요구하는
   "명시적 완화"의 형태). 도구 플래그(`is_read_only=False`,
   `is_concurrency_safe=False`)는 기본값 그대로라 executor 순차 실행·감사
   로그에는 쓰기 도구로 정확히 기록된다.
3. Layer 1(deny rule)·Layer 4(hook)는 그대로 적용된다 — 운영자가
   `permission_rules.yaml`에서 TodoWrite를 deny하면 여전히 차단 가능
   (레이어 건너뛰기 없음, anti-pattern #11 준수).

### 7.2 check_permissions 구현

```python
async def check_permissions(self, input_data, context) -> PermissionResult:
    """계획 메타데이터 갱신은 외부 부작용이 없으므로 항상 ALLOW.
    (Layer 3 — 도구 고유 판정. 파이프라인의 다른 레이어는 정상 통과한다)"""
    return PermissionResult(behavior=PermissionBehavior.ALLOW)
```

### 7.3 서브에이전트 허용 여부

**허용한다.** `DISALLOWED_TOOLS_FOR_AGENTS`(agent_tool.py 61행)에 넣지 않는다.

- 금지 목록의 취지는 재귀(Agent)와 부모 전권 행위(TaskCreate/TaskStop/
  Training/Checkpoint) 차단이다. TodoWrite는 어느 쪽도 아니다.
- 서브에이전트도 자기 다단계 작업의 일관성이 필요하며, 5.3의 agent_id 격리로
  부모 목록 오염이 불가능하다.
- 현재도 TodoWrite는 금지 목록에 없으므로 **변경 없음**(회귀 위험 0).

---

## 8. 시스템 프롬프트 지시 초안

`web/prompts/worker_system.md`(웹)와 CLI 시스템 프롬프트 템플릿에 아래 섹션을
추가한다. (Jinja2 템플릿 규칙 P7 준수, 한국어 본문.)

```markdown
## 작업 체크리스트 (TodoWrite)

복잡한 작업은 TodoWrite로 체크리스트를 만들어 진행 상황을 추적하십시오.

**사용해야 할 때**
- 3단계 이상이 필요한 작업
- 여러 파일을 수정하는 작업
- 사용자가 여러 요구사항을 한 번에 제시했을 때
- 긴 자율 작업(테스트-수정 반복, 마이그레이션 등)

**사용하지 않아도 될 때**
- 단일 도구 호출로 끝나는 단순 요청
- 순수 질의응답

**규칙**
1. 작업 시작 시 전체 계획을 pending 항목으로 등록하십시오.
2. 항목을 시작할 때 그 항목만 in_progress로 바꾸십시오 — 동시에 하나만.
3. 항목이 끝나면 **즉시** completed로 갱신하십시오. 여러 개를 몰아서
   갱신하지 마십시오.
4. TodoWrite는 항상 목록 전체를 보내 기존 목록을 교체합니다.
5. 계획이 바뀌면 남은 항목을 수정·추가·삭제해 목록을 현실과 일치시키십시오.
6. 테스트 실패 등으로 완료가 확인되지 않은 항목은 completed로 바꾸지 말고
   블로커를 새 항목으로 추가하십시오.
```

주입 위치: prompt_assembler의 도구 안내 섹션(정적) + 5.1의 현재 체크리스트
스냅샷(동적, 매 턴). 정적 지시는 프롬프트 캐시 안정 구간에, 동적 스냅샷은
turn_state 요약과 같은 가변 구간에 배치해 캐시 무효화를 최소화한다
(anti-pattern #5의 취지와 동일한 캐시 안정성 고려).

---

## 9. 구현 단계 (파일별 체크리스트)

3개 이상 파일 수정이므로 agent-collaboration.md에 따라 이 문서가 한글
계획서를 겸한다. 구현 착수 전 사용자 승인 필요.

### Phase 1 — 데이터 모델·저장소
- [ ] `core/task/todo_store.py` **신규** — TodoStatus/TodoItem/TodoListState/
      TodoStore (+ 한글 헤더 주석, 폴백 전역 dict)
- [ ] `core/task/__init__.py` — export 추가

### Phase 2 — 도구 교체
- [ ] `core/tools/implementations/todo_tools.py` **신규** —
      TodoWriteTool/TodoReadTool (체크리스트 시맨틱, 3장 설계대로)
- [ ] `core/tools/implementations/task_tools.py` — 기존 TodoReadTool/
      TodoWriteTool 클래스 **제거**, TaskTool만 존치 (모듈 docstring 갱신)
- [ ] `core/bootstrap.py` — (a) `components["todo_store"] = TodoStore()`,
      (b) context.options 주입(304행 부근 task_manager와 나란히),
      (c) 레지스트리 등록을 todo_tools의 새 클래스로 교체(860·889행)
- [ ] `core/permission/pipeline.py` — `_categorize_tool()` readonly_tools에
      `"todoread", "todowrite"` 추가 (7.1 주석 포함)
- [ ] `config/tool_mappings.yaml` — 이름 변경 없음, 주석만 "계획 체크리스트"로
      정정

### Phase 3 — 오케스트레이터·프롬프트
- [ ] `core/orchestrator/prompt_assembler.py` — todo_store 선택 주입 +
      체크리스트 섹션 조립(토큰 상한 포함)
- [ ] `core/orchestrator/query_engine.py` — prompt_assembler에 todo_store 전달
- [ ] `web/prompts/worker_system.md` — 8장 지시 섹션 추가

### Phase 4 — UI
- [ ] `web/app.py` — TOOL_RESULT 분기에서 TodoWrite metadata.todos 추출 →
      `todo_update` 프레임 송출, `delete_session`에서 clear_session 호출
- [ ] `web/static/index.html` — `#todoPanel` 렌더/갱신 JS + CSS
- [ ] `cli/formatters.py` — `format_todo_list()` 추가
- [ ] `cli/repl.py` — TodoWrite 결과를 Panel로 표시

### Phase 5 — 학습 데이터·테스트
- [ ] `training/bootstrap_generator.py` — 309~343행 TodoRead/TodoWrite 예제를
      새 스키마(전체 목록 교체)로 재작성
- [ ] 테스트 (아래 9.1)
- [ ] `ruff check . && ruff format` (변경 파일만), `pytest tests/ -x`
- [ ] progress.md 결정 기록

### 9.1 단위 테스트 계획 (`tests/unit/test_todo_tools.py` 신규)

네이밍 규칙(testing.md) 준수.

```
test_todo_write_full_replace_overwrites_previous_list
test_todo_write_empty_array_clears_list
test_todo_write_over_50_items_returns_error
test_todo_write_blank_content_returns_error
test_todo_write_multiple_in_progress_warns_but_succeeds
test_todo_write_camelcase_activeform_normalized
test_todo_write_result_metadata_contains_structured_todos
test_todo_read_returns_current_checklist
test_todo_store_session_isolation_no_cross_talk
test_todo_store_subagent_isolation_separate_from_main
test_todo_store_clear_session_removes_list
test_todo_write_permission_always_allow
test_categorize_todowrite_as_readonly            # pipeline 테스트에 추가
test_todo_write_not_concurrency_safe_flag        # #12 파티셔닝 전제 검증
test_prompt_assembler_injects_todo_section       # 통합(별도 파일)
```

기존 테스트 갱신.
- `tests/unit/test_task_manager.py` 376~400행 — 구 TodoWrite/TodoRead 테스트
  제거 또는 TaskTool 기반으로 이전.
- `tests/unit/test_tools_all24.py` — import 경로만 todo_tools로 교체
  (도구 수 24 불변이므로 카운트 검증은 그대로 통과해야 함).

---

## 10. 리스크 및 사양 대조 결과

### 10.1 사양 대조 (필수 항목)

- `PROJECT_NEXUS_SPEC_v7.1_AMENDMENT.md`, `v7.2_AMENDMENT.md`:
  todo/plan/checklist/진행 추적 관련 언급 **없음**(grep 0건) → 상충 없음.
- `v7.0_AMENDMENT.md`: `TurnState.todo`(807행)만 존재 — 턴 요약 필드로,
  본 설계와 역할이 다르며 공존한다(5.1).
- `v6.1_EN.md` 28165행: TodoRead/TodoWrite = "Read/Write task **list**" —
  본 설계는 이 원 정의로의 **복귀**다. 다만 input_schema·저장소·권한
  카테고리의 구체 형태는 사양서에 없던 상세화이므로, **v7.3 AMENDMENT에
  본 문서를 근거로 1개 섹션(도구 9·10번 시맨틱 확정 + TodoStore)을 추가**하는
  것을 권고한다(사양서-구현 동기화 원칙).

### 10.2 리스크

| # | 리스크 | 심각도 | 완화 |
|---|---|---|---|
| 1 | **학습 데이터 드리프트** — bootstrap_generator의 구 스키마(task_id/description) 예제로 학습된 LoRA가 구 형식으로 호출 | 중 | (a) validate_input에서 구 형식 감지 시 "새 스키마 안내" 에러 메시지 반환(모델 자가 교정 유도), (b) Phase 5에서 학습 예제 동시 교체, (c) 차기 LoRA 사이클에 반영 |
| 2 | 소형 모델(Qwen 27B)이 전체 목록 재전송을 누락하고 1건만 보냄 → 목록 축소 사고 | 중 | 결과 에코에 "n→m개로 줄었습니다" 카운트 명시 + 급감(50%↑) 시 결과에 경고 문구, 시스템 프롬프트 규칙 4 |
| 3 | 매 갱신마다 목록 전문 에코 → TIER_S(8K)에서 토큰 압박 | 중 | 항목 50개·내용 500자 상한, prompt_assembler 주입 300토큰 상한, completed 축약 |
| 4 | READONLY 카테고리 분류를 "권한 우회 선례"로 오해 | 저 | 7.1 근거 주석을 코드에 그대로 남기고, 감사 로그에는 실제 플래그(쓰기)로 기록됨을 명시 |
| 5 | 구 TodoWrite(ASK) 동작에 의존하던 흐름 회귀 | 저 | 전수 grep 결과 의존처는 bootstrap 등록·테스트·학습 예제뿐(2.1) — 전부 Phase 2·5에서 갱신. TaskTool이 동일 기능 제공 |
| 6 | 웹 프레임 추가로 구 프론트엔드와 불일치 | 저 | `todo_update`는 신규 프레임 — 구 클라이언트는 무시(하위 호환), index.html은 같은 배포 단위 |

### 10.3 권장안 요약

- **A안 채택**: 기존 TodoRead/TodoWrite 이름을 체크리스트 시맨틱으로 교체
  (v6.1 원 정의 복귀), 백그라운드 태스크는 Task 도구로 일원화.
- TodoWrite = 전체 목록 원자 교체 + 결과 에코, TodoRead = 경량 조회 유지
  (24개 도구 수 불변).
- 저장소는 `core/task/todo_store.py`의 세션·에이전트 키 인메모리 TodoStore,
  bootstrap 컴포넌트 + context.options 주입(기존 task_manager 패턴).
- 권한은 이름 기반 READONLY 카테고리(모든 모드 ALLOW, PLAN 모드 포함),
  도구 플래그는 fail-closed 기본값 유지, 서브에이전트 허용(격리 저장).
- 사양서 v7.3 AMENDMENT에 본 설계 반영 섹션 추가를 함께 진행.
