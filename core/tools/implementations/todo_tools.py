# 계획 체크리스트(Claude Code TodoWrite식) 도구 — TodoWrite/TodoRead 2종
"""
Todo 도구 모음 — 모델이 다단계 작업의 "계획 체크리스트"를 스스로 관리하는 도구.

[역할]
Claude Code의 TodoWrite처럼, 모델이 장기·다단계 작업 중 계획 항목(내용 + 상태
pending/in_progress/completed)을 만들고 갱신하며 진행을 추적한다. 백그라운드
실행체(asyncio.Task) 제어는 이 도구가 아니라 Task 도구(create/update/list/get/
stop)가 담당한다 — 역할이 명확히 분리돼 있다.

[제공 도구]
  - TodoWriteTool ("TodoWrite") : 체크리스트 "전체"를 원자적으로 교체(유일한 쓰기 경로)
  - TodoReadTool  ("TodoRead")  : 현재 체크리스트 조회(읽기 전용)

[전체-교체 시맨틱을 쓰는 이유]
  (1) 멱등적 — 같은 호출을 재시도해도 결과가 같다.
  (2) id 관리 불필요 — 소형 모델(Qwen 27B)이 task_id를 기억·전달하다 틀리는
      오류 모드를 원천 제거한다.
  (3) 순서가 곧 계획 순서 — 배열을 그대로 렌더한다.

[저장소]
실제 목록은 core.todo_store.TodoStore가 (세션, 에이전트) 키로 보관한다.
context.options["todo_store"]에 주입돼 오며(bootstrap), 없으면 모듈 전역 폴백을
쓴다(get_todo_store 헬퍼). task_tools의 task_manager 주입 패턴과 동일하다.

[권한]
계획 메타데이터 갱신은 외부 부작용(파일·프로세스·네트워크)이 전혀 없으므로
check_permissions는 항상 ALLOW다. 자율 작업 중 매 갱신마다 확인이 뜨면 도구가
무용지물이 되기 때문이다. 단, 도구 플래그(is_read_only 등)는 정직하게 유지한다
(TodoWrite는 세션 상태를 바꾸므로 read_only=False). PLAN 모드 등에서의 자동
허용은 권한 파이프라인의 이름 기반 READONLY 카테고리(pipeline._categorize_tool)가
담당한다 — 그쪽 주석 참조.

의존:
  - core.todo_store : TodoItem/TodoStatus/TodoStore/render_checklist/get_todo_store
  - core.tools.base : BaseTool 및 권한/결과 타입

작성자: Nexus / 작성일: 2026-07-08
"""

from __future__ import annotations

import logging
from typing import Any

from core.todo_store import (
    MAX_CONTENT_LEN,
    MAX_TODO_ITEMS,
    TodoItem,
    TodoStatus,
    get_todo_store,
    render_checklist,
)
from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 이 모듈 전용 로거. 규칙상 "nexus.{module}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.tools.todo")

# 유효한 status 문자열 집합(2차 방어 — 1차는 JSON Schema enum).
_VALID_STATUS = {s.value for s in TodoStatus}


def _stats(items: list[TodoItem]) -> dict[str, int]:
    """체크리스트의 상태별 개수를 요약한다(결과 metadata·에코용)."""
    total = len(items)
    completed = sum(1 for i in items if i.status == TodoStatus.COMPLETED)
    in_progress = sum(1 for i in items if i.status == TodoStatus.IN_PROGRESS)
    return {"total": total, "completed": completed, "in_progress": in_progress}


# ─────────────────────────────────────────────
# TodoWriteTool — 체크리스트 전체 원자 교체
# ─────────────────────────────────────────────
class TodoWriteTool(BaseTool):
    """
    계획 체크리스트 "전체"를 원자적으로 교체하는 쓰기 도구("TodoWrite").

    입력 todos 배열이 기존 목록을 통째로 대체한다(부분 patch API 없음 — Claude
    Code와 동일). 세션·에이전트별로 격리된 TodoStore에 저장하고, 결과에는 갱신된
    목록 전문을 에코하며(모델이 다음 턴에 자기 계획을 0비용으로 재확인) metadata에
    구조화 원본(todos)을 담는다(웹/CLI가 UI 프레임을 만들 때 서버에서 추출).

    [Behavior Flag — P6 fail-closed 그대로]
    is_read_only=False(세션 상태를 바꿈), is_concurrency_safe=False(전체-교체라
    동시 실행 시 마지막 쓰기가 이기는 레이스 가능 — anti-pattern #12에 따라 순차
    파티션에 남긴다). 어떤 플래그도 완화하지 않는다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "TodoWrite"

    @property
    def description(self) -> str:
        return (
            "계획 체크리스트 전체를 교체합니다. todos 배열이 기존 목록을 통째로 "
            "대체합니다(항상 전체를 보내세요). 각 항목은 content, "
            "status(pending/in_progress/completed), active_form을 가집니다."
        )

    @property
    def group(self) -> str:
        return "task"

    # ═══ 2. Schema ═══

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
                                "description": "할 일 내용 (명령형, 예: '테스트 실행')",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "항목 상태",
                            },
                            "active_form": {
                                "type": "string",
                                "description": (
                                    "진행 중일 때 UI에 표시할 현재진행형 문구 "
                                    "(예: '권한 파이프라인 테스트 실행 중')"
                                ),
                            },
                        },
                        "required": ["content", "status", "active_form"],
                    },
                },
            },
            "required": ["todos"],
        }

    # ═══ 3. Behavior Flags ═══
    # 쓰기 도구 — 기본 fail-closed(is_read_only=False, is_concurrency_safe=False) 유지.

    # ═══ 5. Lifecycle ═══

    @staticmethod
    def _normalize_todo(raw: Any) -> dict[str, Any]:
        """
        입력 항목 하나를 관용적으로 정규화한다(검증 전 전처리).

        - Claude Code 학습 잔재로 camelCase `activeForm`을 보낼 수 있어
          `active_form`으로 매핑한다(에러 대신 정규화).
        dict가 아니면 그대로 돌려줘 validate_input이 걸러내게 한다.
        """
        if not isinstance(raw, dict):
            return raw
        item = dict(raw)
        # activeForm → active_form 관용 매핑(원본에 snake_case가 없을 때만).
        if "activeForm" in item and "active_form" not in item:
            item["active_form"] = item.pop("activeForm")
        return item

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        JSON Schema 통과 후 도메인 검증(권한/실행 전 호출).

        구 스키마(단건 task_id/action/description) 감지 시 새 스키마 안내 에러를
        반환해 모델이 자가 교정하게 유도한다(학습 데이터 드리프트 완화). 그 외:
        항목 수/내용 길이 상한, 공백 content 금지, status enum 2차 검증을 수행한다.
        in_progress가 2개 이상이어도 에러가 아니라 경고(soft rule)로만 다룬다.
        """
        # ── 구 스키마 감지(드리프트 완화) ──
        # 예전 TodoWrite는 action/task_id/description을 받았다. todos가 없고 이
        # 구 키들이 보이면, 모델이 옛 형식으로 부른 것이므로 새 형식을 안내한다.
        if "todos" not in input_data and any(
            k in input_data for k in ("action", "task_id", "description")
        ):
            return (
                "TodoWrite는 이제 계획 체크리스트 전체를 교체합니다. "
                "task_id/action/description 대신 todos 배열을 보내세요. "
                "예: {\"todos\": [{\"content\": \"...\", \"status\": \"pending\", "
                "\"active_form\": \"...\"}]}. "
                "(백그라운드 태스크 제어는 Task 도구를 사용하세요.)"
            )

        todos = input_data.get("todos")
        if not isinstance(todos, list):
            return "todos는 배열이어야 합니다."

        # 항목 수 상한(컨텍스트 폭주 방지).
        if len(todos) > MAX_TODO_ITEMS:
            return f"체크리스트 항목은 최대 {MAX_TODO_ITEMS}개입니다 (현재 {len(todos)}개)."

        for idx, raw in enumerate(todos):
            item = self._normalize_todo(raw)
            if not isinstance(item, dict):
                return f"{idx + 1}번째 항목이 객체가 아닙니다."
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                return f"{idx + 1}번째 항목의 content가 비어 있습니다."
            if len(content) > MAX_CONTENT_LEN:
                return f"{idx + 1}번째 항목의 content가 너무 깁니다 (최대 {MAX_CONTENT_LEN}자)."
            status = item.get("status")
            if status not in _VALID_STATUS:
                return f"{idx + 1}번째 항목의 status가 올바르지 않습니다: {status!r}"
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        계획 메타데이터 갱신은 외부 부작용이 없으므로 항상 ALLOW.

        (Layer 3 — 도구 고유 판정. 파이프라인의 다른 레이어는 정상 통과한다.)
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        체크리스트 전체를 원자적으로 교체하고, 갱신된 목록 전문을 에코한다.

        처리 순서:
          1. 입력 todos를 정규화(activeForm 매핑) 후 TodoItem 리스트로 변환.
          2. TodoStore.replace(session_id, agent_id, items)로 원자 교체.
          3. data에 목록 전문(모델이 다시 읽는 내용) + 통계/경고를 에코,
             metadata.todos에 구조화 원본을 담아 반환(웹/CLI가 서버에서 추출).

        4-Tier 체인 준수: 도구는 executor 경로로 실행되고 결과는 기존 TOOL_RESULT
        이벤트로만 전파된다(새 StreamEvent 타입을 만들지 않는다).
        """
        store = get_todo_store(context)

        # 입력 → TodoItem 변환. validate_input을 통과했으므로 필드는 유효하다.
        items: list[TodoItem] = []
        for raw in input_data.get("todos", []):
            item = self._normalize_todo(raw)
            items.append(
                TodoItem(
                    content=item["content"],
                    status=TodoStatus(item["status"]),
                    active_form=item.get("active_form", "") or "",
                )
            )

        # 교체 직전 이전 개수를 확보(급감 감지용).
        prev_count = len(store.get(context.session_id, context.agent_id).items)
        store.replace(context.session_id, context.agent_id, items)

        stats = _stats(items)
        body = render_checklist(tuple(items))
        # 요약 카운트를 함께 에코해, 소형 모델이 목록이 줄었는지 스스로 인지하게 한다.
        summary = f"체크리스트 갱신됨 — 완료 {stats['completed']}/{stats['total']}"
        if stats["in_progress"]:
            summary += f", 진행 중 {stats['in_progress']}"

        warnings: list[str] = []
        # soft rule: 동시에 여러 개가 in_progress면 경고만(하드 차단 시 재시도 루프 위험).
        if stats["in_progress"] > 1:
            warnings.append(
                f"진행 중(in_progress) 항목이 {stats['in_progress']}개입니다 — "
                "원칙적으로 동시에 하나만 두는 것이 좋습니다."
            )
        # 목록이 절반 이상 급감했으면(전체 재전송 누락 의심) 경고한다.
        if prev_count and stats["total"] <= prev_count // 2:
            warnings.append(
                f"항목이 {prev_count}→{stats['total']}개로 줄었습니다 — "
                "TodoWrite는 항상 목록 전체를 보내야 합니다(부분 전송이 아닌지 확인)."
            )

        data = f"{summary}\n{body}"
        if warnings:
            data += "\n\n주의:\n" + "\n".join(f"- {w}" for w in warnings)

        logger.info(
            "TodoWrite: session=%s agent=%s items=%d (completed=%d, in_progress=%d)",
            context.session_id,
            context.agent_id or "main",
            stats["total"],
            stats["completed"],
            stats["in_progress"],
        )
        # metadata.todos — 웹/CLI가 파싱할 구조화 데이터(모델 텍스트 파싱 불필요).
        return ToolResult.success(
            data,
            todos=[i.model_dump(mode="json") for i in items],
            stats=stats,
            warnings=warnings,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Updating checklist..."


# ─────────────────────────────────────────────
# TodoReadTool — 현재 체크리스트 조회
# ─────────────────────────────────────────────
class TodoReadTool(BaseTool):
    """
    현재 계획 체크리스트를 조회하는 읽기 전용 도구("TodoRead").

    데이터를 바꾸지 않는 순수 조회라 is_read_only=True, is_concurrency_safe=True.
    매 TodoWrite 결과에 목록 전문이 에코되고 시스템 프롬프트에도 주입되므로 호출
    빈도는 낮을 것으로 예상하지만, compact 직후나 서브에이전트 시작 시 상태를
    복원하는 용도로 24개 도구 수를 유지하며 존치한다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "TodoRead"

    @property
    def description(self) -> str:
        return "현재 계획 체크리스트를 조회합니다(인자 없음)."

    @property
    def group(self) -> str:
        return "task"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        # 인자 없음 — 체크리스트 규모(≤50항목)에서 필터는 불필요하다.
        return {"type": "object", "properties": {}, "required": []}

    # ═══ 3. Behavior Flags ═══

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """조회는 부작용이 없으므로 항상 ALLOW."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """현재 (세션, 에이전트)의 체크리스트를 전문으로 돌려준다."""
        store = get_todo_store(context)
        state = store.get(context.session_id, context.agent_id)
        stats = _stats(list(state.items))
        body = render_checklist(state.items)
        logger.debug("TodoRead: session=%s items=%d", context.session_id, stats["total"])
        return ToolResult.success(
            body,
            todos=[i.model_dump(mode="json") for i in state.items],
            stats=stats,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Loading checklist..."
