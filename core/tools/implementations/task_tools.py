"""
Task 도구 모음 — 태스크 관리 도구.

3개의 태스크 도구를 제공한다:
  - TodoRead: 태스크 목록 조회 (읽기 전용)
  - TodoWrite: 태스크 생성/업데이트
  - TaskTool: 태스크 관리 (create/update/list/stop)

실제 TaskManager는 아직 구현되지 않았으므로 (Phase 3.0+),
context.options에서 task_manager를 가져오되
없으면 인메모리 딕셔너리 기반의 간단한 구현을 사용한다.

각 태스크는 고유 ID, 상태(pending/in_progress/completed/failed),
설명, 우선순위 등의 정보를 가진다.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.task")

# 태스크 상태 상수
TASK_STATUS_PENDING = "pending"
TASK_STATUS_IN_PROGRESS = "in_progress"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 유효한 태스크 상태 목록
VALID_STATUSES = {
    TASK_STATUS_PENDING,
    TASK_STATUS_IN_PROGRESS,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
}

# 인메모리 태스크 저장소 (TaskManager가 없을 때 폴백으로 사용)
# 키: task_id, 값: task 딕셔너리
_fallback_tasks: dict[str, dict[str, Any]] = {}


def _get_task_manager(context: ToolUseContext) -> Any:
    """
    context.options에서 task_manager를 가져온다.
    없으면 None을 반환하여 인메모리 폴백을 사용하게 한다.
    """
    return context.options.get("task_manager")


def _create_task(
    description: str,
    priority: str = "medium",
    status: str = TASK_STATUS_PENDING,
) -> dict[str, Any]:
    """
    새 태스크를 생성하고 인메모리 저장소에 저장한다.

    Args:
        description: 태스크 설명
        priority: 우선순위 (low/medium/high)
        status: 초기 상태

    Returns:
        생성된 태스크 딕셔너리
    """
    task_id = str(uuid.uuid4())[:8]  # 간결한 ID (8자)
    task = {
        "id": task_id,
        "description": description,
        "status": status,
        "priority": priority,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    _fallback_tasks[task_id] = task
    return task


def _format_task(task: dict[str, Any]) -> str:
    """태스크를 사람이 읽기 쉬운 한 줄 형식으로 변환한다."""
    status_icon = {
        TASK_STATUS_PENDING: "[ ]",
        TASK_STATUS_IN_PROGRESS: "[~]",
        TASK_STATUS_COMPLETED: "[x]",
        TASK_STATUS_FAILED: "[!]",
    }.get(task["status"], "[?]")
    priority = task.get("priority", "medium")
    return f"#{task['id']} {status_icon} [{priority}] {task['description']}"


# ─────────────────────────────────────────────
# TodoReadTool — 태스크 목록 조회
# ─────────────────────────────────────────────
class TodoReadTool(BaseTool):
    """
    태스크 목록을 조회하는 도구.
    전체 태스크 또는 상태별 필터링된 목록을 보여준다.
    읽기 전용이므로 병렬 실행이 안전하다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "TodoRead"

    @property
    def description(self) -> str:
        return "태스크 목록을 조회합니다. 전체 또는 상태별로 필터링할 수 있습니다."

    @property
    def group(self) -> str:
        return "task"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "description": "상태 필터 (pending/in_progress/completed/failed)",
                    "enum": ["pending", "in_progress", "completed", "failed"],
                },
            },
            "required": [],
        }

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
        """읽기 전용이므로 항상 허용한다."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        태스크 목록을 조회한다.

        처리 순서:
          1. TaskManager가 있으면 사용, 없으면 인메모리 폴백
          2. 상태 필터 적용 (선택)
          3. 포맷하여 반환
        """
        manager = _get_task_manager(context)
        status_filter = input_data.get("status_filter")

        # TaskManager가 있으면 위임
        # TODO(nexus): TaskManager 구현 후 연동
        if manager and hasattr(manager, "list_tasks"):
            tasks = manager.list_tasks(status=status_filter)
        else:
            # 인메모리 폴백 사용
            tasks = list(_fallback_tasks.values())
            if status_filter:
                tasks = [t for t in tasks if t["status"] == status_filter]

        if not tasks:
            filter_msg = f" (상태: {status_filter})" if status_filter else ""
            return ToolResult.success(f"태스크가 없습니다{filter_msg}.", count=0)

        # 생성 시간순 정렬 (최신 먼저)
        tasks.sort(key=lambda t: t.get("created_at", 0), reverse=True)
        lines = [_format_task(t) for t in tasks]
        result_text = "\n".join(lines)

        logger.debug("TodoRead: %d tasks found", len(tasks))
        return ToolResult.success(result_text, count=len(tasks))

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Loading tasks..."


# ─────────────────────────────────────────────
# TodoWriteTool — 태스크 생성/업데이트
# ─────────────────────────────────────────────
class TodoWriteTool(BaseTool):
    """
    태스크를 생성하거나 업데이트하는 도구.
    task_id가 주어지면 업데이트, 없으면 새로 생성한다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "TodoWrite"

    @property
    def description(self) -> str:
        return (
            "태스크를 생성하거나 업데이트합니다. "
            "task_id가 주어지면 기존 태스크를 업데이트하고, "
            "없으면 새 태스크를 생성합니다."
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
                "task_id": {
                    "type": "string",
                    "description": "업데이트할 태스크 ID (생략 시 새로 생성)",
                },
                "description": {
                    "type": "string",
                    "description": "태스크 설명",
                },
                "status": {
                    "type": "string",
                    "description": "태스크 상태",
                    "enum": ["pending", "in_progress", "completed", "failed"],
                },
                "priority": {
                    "type": "string",
                    "description": "우선순위 (기본: medium)",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
            },
            "required": [],
        }

    # ═══ 3. Behavior Flags ═══
    # 쓰기 도구 — 기본 fail-closed 유지

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """새 태스크 생성 시 description이 필수인지 검증한다."""
        task_id = input_data.get("task_id")
        if not task_id and not input_data.get("description"):
            return "새 태스크 생성 시 description은 필수입니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """태스크 쓰기는 사용자에게 확인을 요청한다."""
        task_id = input_data.get("task_id")
        if task_id:
            msg = f"태스크 #{task_id} 업데이트"
        else:
            desc = input_data.get("description", "")[:50]
            msg = f"새 태스크 생성: {desc}"
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=msg,
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        태스크를 생성하거나 업데이트한다.

        처리 순서:
          1. task_id 유무로 생성/업데이트 분기
          2. 생성: 새 태스크 딕셔너리 생성 후 저장
          3. 업데이트: 기존 태스크를 찾아 필드 수정
        """
        manager = _get_task_manager(context)
        task_id = input_data.get("task_id")

        if task_id:
            # 기존 태스크 업데이트
            # TODO(nexus): TaskManager 구현 후 연동
            if manager and hasattr(manager, "update_task"):
                task = manager.update_task(task_id, input_data)
            else:
                # 인메모리 폴백
                task = _fallback_tasks.get(task_id)
                if not task:
                    return ToolResult.error(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")

                # 제공된 필드만 업데이트
                if "description" in input_data:
                    task["description"] = input_data["description"]
                if "status" in input_data:
                    task["status"] = input_data["status"]
                if "priority" in input_data:
                    task["priority"] = input_data["priority"]
                task["updated_at"] = time.time()

            logger.info("TodoWrite: updated task #%s", task_id)
            return ToolResult.success(
                f"태스크 #{task_id}을(를) 업데이트했습니다.\n{_format_task(task)}",
                task_id=task_id,
            )
        else:
            # 새 태스크 생성
            description = input_data["description"]
            priority = input_data.get("priority", "medium")

            # TODO(nexus): TaskManager 구현 후 연동
            if manager and hasattr(manager, "create_task"):
                task = manager.create_task(description=description, priority=priority)
            else:
                task = _create_task(description=description, priority=priority)

            logger.info("TodoWrite: created task #%s", task["id"])
            return ToolResult.success(
                f"태스크를 생성했습니다.\n{_format_task(task)}",
                task_id=task["id"],
            )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        if input_data.get("task_id"):
            return "Updating task..."
        return "Creating task..."


# ─────────────────────────────────────────────
# TaskTool — 종합 태스크 관리
# ─────────────────────────────────────────────
class TaskTool(BaseTool):
    """
    종합 태스크 관리 도구.
    하나의 도구로 태스크의 생성, 조회, 업데이트, 중지를 수행할 수 있다.
    TodoRead/TodoWrite보다 더 세분화된 제어가 가능하다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "Task"

    @property
    def description(self) -> str:
        return (
            "태스크를 관리합니다. "
            "create(생성), update(업데이트), list(목록), get(조회), stop(중지) 작업을 지원합니다."
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
                "action": {
                    "type": "string",
                    "description": "수행할 작업",
                    "enum": ["create", "update", "list", "get", "stop"],
                },
                "task_id": {
                    "type": "string",
                    "description": "태스크 ID (update/get/stop 시 필수)",
                },
                "description": {
                    "type": "string",
                    "description": "태스크 설명 (create 시 필수)",
                },
                "status": {
                    "type": "string",
                    "description": "태스크 상태 (update 시 사용)",
                    "enum": ["pending", "in_progress", "completed", "failed"],
                },
                "priority": {
                    "type": "string",
                    "description": "우선순위 (기본: medium)",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                },
                "status_filter": {
                    "type": "string",
                    "description": "목록 조회 시 상태 필터",
                    "enum": ["pending", "in_progress", "completed", "failed"],
                },
            },
            "required": ["action"],
        }

    # ═══ 3. Behavior Flags ═══
    # 읽기/쓰기가 혼재 — 기본 fail-closed 유지

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """action별 필수 필드를 검증한다."""
        action = input_data.get("action", "")

        if action == "create" and not input_data.get("description"):
            return "create 작업에는 description이 필수입니다."

        if action in ("update", "get", "stop") and not input_data.get("task_id"):
            return f"{action} 작업에는 task_id가 필수입니다."

        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """읽기 작업은 허용, 쓰기 작업은 확인을 요청한다."""
        action = input_data.get("action", "")

        # 읽기 작업은 바로 허용
        if action in ("list", "get"):
            return PermissionResult(behavior=PermissionBehavior.ALLOW)

        # 쓰기 작업은 확인 요청
        task_id = input_data.get("task_id", "")
        if action == "create":
            desc = input_data.get("description", "")[:50]
            msg = f"Task create: {desc}"
        elif action == "update":
            msg = f"Task update: #{task_id}"
        elif action == "stop":
            msg = f"Task stop: #{task_id}"
        else:
            msg = f"Task {action}"

        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=msg,
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        지정된 action에 따라 태스크를 관리한다.

        지원 작업:
          - create: 새 태스크 생성
          - update: 태스크 필드 업데이트
          - list: 태스크 목록 조회
          - get: 특정 태스크 상세 조회
          - stop: 태스크 중지 (상태를 failed로 변경)
        """
        action = input_data["action"]

        # 각 action별 분기 처리
        if action == "create":
            return self._handle_create(input_data, context)
        elif action == "update":
            return self._handle_update(input_data, context)
        elif action == "list":
            return self._handle_list(input_data, context)
        elif action == "get":
            return self._handle_get(input_data, context)
        elif action == "stop":
            return self._handle_stop(input_data, context)
        else:
            return ToolResult.error(f"알 수 없는 작업: {action}")

    def _handle_create(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """새 태스크를 생성한다."""
        description = input_data["description"]
        priority = input_data.get("priority", "medium")

        manager = _get_task_manager(context)
        # TODO(nexus): TaskManager 구현 후 연동
        if manager and hasattr(manager, "create_task"):
            task = manager.create_task(description=description, priority=priority)
        else:
            task = _create_task(description=description, priority=priority)

        logger.info("Task create: #%s", task["id"])
        return ToolResult.success(
            f"태스크를 생성했습니다.\n{_format_task(task)}",
            task_id=task["id"],
        )

    def _handle_update(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """기존 태스크를 업데이트한다."""
        task_id = input_data["task_id"]

        manager = _get_task_manager(context)
        if manager and hasattr(manager, "update_task"):
            task = manager.update_task(task_id, input_data)
        else:
            task = _fallback_tasks.get(task_id)
            if not task:
                return ToolResult.error(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")

            if "description" in input_data:
                task["description"] = input_data["description"]
            if "status" in input_data:
                task["status"] = input_data["status"]
            if "priority" in input_data:
                task["priority"] = input_data["priority"]
            task["updated_at"] = time.time()

        logger.info("Task update: #%s", task_id)
        return ToolResult.success(
            f"태스크 #{task_id}을(를) 업데이트했습니다.\n{_format_task(task)}",
            task_id=task_id,
        )

    def _handle_list(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """태스크 목록을 조회한다."""
        status_filter = input_data.get("status_filter")

        manager = _get_task_manager(context)
        if manager and hasattr(manager, "list_tasks"):
            tasks = manager.list_tasks(status=status_filter)
        else:
            tasks = list(_fallback_tasks.values())
            if status_filter:
                tasks = [t for t in tasks if t["status"] == status_filter]

        if not tasks:
            return ToolResult.success("태스크가 없습니다.", count=0)

        tasks.sort(key=lambda t: t.get("created_at", 0), reverse=True)
        lines = [_format_task(t) for t in tasks]
        return ToolResult.success("\n".join(lines), count=len(tasks))

    def _handle_get(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """특정 태스크를 상세 조회한다."""
        task_id = input_data["task_id"]

        manager = _get_task_manager(context)
        if manager and hasattr(manager, "get_task"):
            task = manager.get_task(task_id)
        else:
            task = _fallback_tasks.get(task_id)

        if not task:
            return ToolResult.error(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")

        # 상세 정보 포맷
        lines = [
            f"ID:       #{task['id']}",
            f"상태:     {task['status']}",
            f"우선순위: {task.get('priority', 'medium')}",
            f"설명:     {task['description']}",
            f"생성:     {task.get('created_at', 'N/A')}",
            f"수정:     {task.get('updated_at', 'N/A')}",
        ]
        return ToolResult.success("\n".join(lines), task_id=task_id)

    def _handle_stop(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """태스크를 중지한다 (상태를 failed로 변경)."""
        task_id = input_data["task_id"]

        manager = _get_task_manager(context)
        if manager and hasattr(manager, "stop_task"):
            manager.stop_task(task_id)
        else:
            task = _fallback_tasks.get(task_id)
            if not task:
                return ToolResult.error(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")
            task["status"] = TASK_STATUS_FAILED
            task["updated_at"] = time.time()

        logger.info("Task stop: #%s", task_id)
        return ToolResult.success(f"태스크 #{task_id}을(를) 중지했습니다.")

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        action = input_data.get("action", "")
        return f"Task {action}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        action = input_data.get("action", "")
        task_id = input_data.get("task_id", "")
        if task_id:
            return f"{action} #{task_id}"
        return action
