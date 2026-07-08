"""
Task 도구 — 에이전트가 백그라운드 "태스크"를 관리할 때 쓰는 종합 도구.

이 파일은 LLM(모델)이 백그라운드 실행체(asyncio.Task — 로컬/원격 에이전트,
Bash, 모니터, 학습 등)를 생성·수정·조회·중지할 수 있도록, 그 라이프사이클을
제어하는 BaseTool 구현을 담는다. Nexus의 도구 레지스트리에 등록되어 모델의
tool_calls로 호출된다.

제공하는 도구(클래스):
  - TaskTool ("Task") : 종합 관리 — create/update/list/get/stop

[변경 이력 — 2026-07-08]
예전에는 이 파일에 TodoReadTool/TodoWriteTool도 있었으나, 이들은 이름만 Todo일
뿐 TaskManager 단건 CRUD로 Task 도구와 기능이 중복이었다. 계획 체크리스트
시맨틱(전체 목록 원자 교체 + pending/in_progress/completed)으로 사양(v6.1의
"task list")을 복구하며 두 도구를 core/tools/implementations/todo_tools.py로
분리·재구현했다. 백그라운드 태스크 제어는 이 Task 도구로 일원화했다.

동작 방식(중요):
  - 실제 태스크 저장/실행은 core.task.TaskManager가 담당한다.
  - TaskManager는 context.options["task_manager"]에 주입되어 전달된다.
  - 주입돼 있으면 그것을 쓰고, 없으면 모듈 전역 인메모리 딕셔너리
    (_fallback_tasks)를 쓰는 "폴백" 경로로 동작한다.

왜 폴백을 유지하는가:
  TaskManager를 붙이지 않은 단위 테스트나 아주 단순한 실행 환경에서도
  기본 CRUD(생성/조회/수정/삭제)가 최소한 돌아가게 하기 위해서다.
  즉 "TaskManager 없어도 죽지 않고 동작"을 보장하는 안전망이다.

의존:
  - core.task            : TaskManager / TaskState / TaskStatus
  - core.tools.base      : BaseTool 및 권한/결과 관련 타입들

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from core.task import TaskManager, TaskState, TaskStatus
from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 이 모듈 전용 로거. 규칙상 "nexus.{module}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.tools.task")

# 인메모리 태스크 저장소 — TaskManager가 없을 때만 쓰는 폴백 공간.
# 프로세스 메모리에만 존재하므로 재시작하면 사라진다(영속성 없음).
# 구조: { task_id(str) -> task(dict) }. 여러 도구가 이 전역을 공유한다.
_fallback_tasks: dict[str, dict[str, Any]] = {}


def _get_task_manager(context: ToolUseContext) -> TaskManager | None:
    """
    실행 컨텍스트에서 TaskManager를 꺼내오는 헬퍼.

    도구 호출 시 넘어오는 context.options 딕셔너리 안에
    "task_manager" 키로 TaskManager가 주입돼 있을 수 있다.
    - 올바른 TaskManager 인스턴스면 그대로 반환한다.
    - 없거나 타입이 맞지 않으면 None을 반환해서,
      호출부가 인메모리 폴백 경로를 타도록 한다.

    Args:
        context: 도구 실행 컨텍스트(옵션·세션 정보 등을 담고 있음).

    Returns:
        TaskManager 인스턴스 또는 None(폴백 사용 신호).
    """
    manager = context.options.get("task_manager")
    if isinstance(manager, TaskManager):
        return manager
    return None


def _create_fallback_task(
    description: str,
    priority: str = "medium",
    status: str = "pending",
) -> dict[str, Any]:
    """
    폴백용 태스크 하나를 만들어 인메모리 저장소(_fallback_tasks)에 넣는다.
    TaskManager가 주입되지 않은 환경에서만 쓰이는 경로다.

    task_id는 uuid4의 앞 8자리만 잘라 짧게 쓴다(사람이 읽기 편하게).
    created_at/updated_at은 생성 시각(time.time(), epoch 초)으로 동일하게 채운다.

    Args:
        description: 태스크 설명(무슨 일인지).
        priority: 우선순위 문자열 (low/medium/high). 기본 medium.
        status: 초기 상태 문자열. 기본 pending.

    Returns:
        방금 생성해 저장소에 저장한 태스크 딕셔너리.
    """
    # uuid4를 만들고 앞 8글자만 사용해 짧은 식별자를 만든다.
    task_id = str(uuid.uuid4())[:8]
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


def _task_state_to_dict(task: TaskState) -> dict[str, Any]:
    """
    TaskManager가 다루는 TaskState 객체를, 이 파일에서 통용되는
    "태스크 딕셔너리" 포맷으로 변환하는 어댑터.

    왜 변환하는가:
      폴백 경로는 처음부터 dict를 쓰고, _format_task()도 dict를 받는다.
      TaskManager 경로에서 온 TaskState를 같은 dict 모양으로 맞춰 주면
      이후 포맷/출력 코드를 한 갈래로 공유할 수 있다.

    주의:
      TaskManager는 priority 개념이 없어서 여기선 "medium"으로 고정한다.
      updated_at은 종료시각(end_time)이 있으면 그걸, 없으면 시작시각을 쓴다.
    """
    return {
        "id": task.id,
        "description": task.description,
        "status": task.status.value,
        "priority": "medium",  # TaskManager는 priority를 관리하지 않음
        "created_at": task.start_time.timestamp(),
        "updated_at": (task.end_time or task.start_time).timestamp(),
    }


def _format_task(task: dict[str, Any]) -> str:
    """
    태스크 딕셔너리 하나를 사람이 읽기 좋은 한 줄 문자열로 만든다.

    형식: "#<id> <상태아이콘> [<우선순위>] <설명>"
    상태 문자열을 보기 쉬운 아이콘으로 매핑하며, 모르는 상태는 "[?]"로 표시.
    """
    # 상태 문자열 -> 표시용 아이콘 매핑. 목록/결과 출력에 공통으로 쓰인다.
    status_icon = {
        "pending": "[ ]",
        "in_progress": "[~]",
        "running": "[~]",  # TaskManager의 running 상태도 동일 아이콘
        "completed": "[x]",
        "failed": "[!]",
        "killed": "[K]",  # 강제 종료 상태
    }.get(task["status"], "[?]")
    # priority 키가 없을 수도 있으니(폴백 외 경로) 기본값 medium으로 보정.
    priority = task.get("priority", "medium")
    return f"#{task['id']} {status_icon} [{priority}] {task['description']}"


# ─────────────────────────────────────────────
# TaskTool — 종합 태스크 관리
# ─────────────────────────────────────────────
class TaskTool(BaseTool):
    """
    하나로 다 하는 종합 태스크 관리 도구("Task").

    입력의 action 값에 따라 다섯 가지 작업을 한 도구에서 처리한다:
      create(생성) / update(수정) / list(목록) / get(상세) / stop(중지).
    stop처럼 백그라운드 실행체의 제어(중지)까지 포함한다. 계획 체크리스트는
    별도 도구(TodoWrite/TodoRead)가 담당한다.

    권한 정책:
      list/get 같은 읽기 작업은 바로 ALLOW,
      create/update/stop 같은 쓰기·제어 작업은 사용자 확인(ASK)을 받는다.
      (읽기/쓰기가 섞여 있어 클래스 단위 플래그는 fail-closed 기본값 유지)

    내부 구조:
      call()이 action을 보고 _handle_* 헬퍼로 위임하는 디스패처 형태다.
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
                    "enum": ["pending", "running", "completed", "failed", "killed"],
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
                    "enum": ["pending", "running", "completed", "failed", "killed"],
                },
            },
            "required": ["action"],
        }

    # ═══ 3. Behavior Flags ═══
    # 읽기/쓰기가 혼재 — 기본 fail-closed 유지

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        action별로 꼭 필요한 필드가 왔는지 미리 검사한다.

        - create: description 필수(무슨 태스크인지 있어야 만든다)
        - update/get/stop: task_id 필수(대상이 있어야 조작한다)
        문제가 있으면 에러 메시지를, 정상이면 None을 반환한다.
        """
        action = input_data.get("action", "")

        if action == "create" and not input_data.get("description"):
            return "create 작업에는 description이 필수입니다."

        if action in ("update", "get", "stop") and not input_data.get("task_id"):
            return f"{action} 작업에는 task_id가 필수입니다."

        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        action 종류에 따라 권한을 다르게 판정한다.

        읽기(list/get)는 부작용이 없으니 ALLOW,
        쓰기·제어(create/update/stop)는 사용자 확인(ASK)을 받는다.
        확인 메시지는 어떤 작업을 어떤 대상에 하는지 알기 쉽게 구성한다.
        """
        action = input_data.get("action", "")

        # 읽기 작업(list/get)은 바로 허용.
        if action in ("list", "get"):
            return PermissionResult(behavior=PermissionBehavior.ALLOW)

        # 여기부터는 쓰기·제어 작업 — 확인용 메시지를 만든다.
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
        action 값을 보고 알맞은 _handle_* 헬퍼로 넘기는 디스패처.

        지원 작업:
          - create: 새 태스크 생성
          - update: 태스크 필드 업데이트
          - list:   태스크 목록 조회
          - get:    특정 태스크 상세 조회
          - stop:   태스크 강제 종료 (TaskManager.kill() 사용)

        stop만 비동기 헬퍼(kill이 async)라서 await로 호출한다.
        알 수 없는 action이면 에러를 반환한다.
        """
        action = input_data["action"]

        if action == "create":
            return self._handle_create(input_data, context)
        elif action == "update":
            return self._handle_update(input_data, context)
        elif action == "list":
            return self._handle_list(input_data, context)
        elif action == "get":
            return self._handle_get(input_data, context)
        elif action == "stop":
            return await self._handle_stop(input_data, context)
        else:
            return ToolResult.error(f"알 수 없는 작업: {action}")

    def _handle_create(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        action="create" 처리 — 새 태스크를 만든다.

        TaskManager가 있으면 local_workflow 타입으로 생성하고,
        없으면 인메모리 폴백에 생성한다. priority는 표시용으로 붙인다.
        """
        description = input_data["description"]
        priority = input_data.get("priority", "medium")

        manager = _get_task_manager(context)
        if manager is not None:
            tid = manager.create("local_workflow", description)
            task_state = manager.tasks[tid]
            task = _task_state_to_dict(task_state)
            task["priority"] = priority
        else:
            task = _create_fallback_task(description=description, priority=priority)

        logger.info("Task create: #%s", task["id"])
        return ToolResult.success(
            f"태스크를 생성했습니다.\n{_format_task(task)}",
            task_id=task["id"],
        )

    def _handle_update(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        action="update" 처리 — 기존 태스크의 일부 필드를 갱신한다.

        입력에 들어온 필드만 부분 수정한다. TaskManager 경로에서는
        progress도 받을 수 있어 update_progress()로 진행률을 갱신한다.
        대상을 못 찾으면 에러를 반환한다.
        """
        task_id = input_data["task_id"]

        manager = _get_task_manager(context)
        if manager is not None:
            task_state = manager.tasks.get(task_id)
            if not task_state:
                return ToolResult.error(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")

            if "description" in input_data:
                task_state.description = input_data["description"]
            if "status" in input_data:
                # 잘못된 상태 문자열이면 조용히 무시(ValueError swallow).
                try:
                    task_state.status = TaskStatus(input_data["status"])
                except ValueError:
                    pass
            if "progress" in input_data:
                # 진행률(0.0~1.0)을 매니저를 통해 갱신한다.
                manager.update_progress(task_id, float(input_data["progress"]))
            task = _task_state_to_dict(task_state)
        else:
            # 폴백 경로: 인메모리 dict를 직접 부분 수정.
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
        """
        action="list" 처리 — 백그라운드 태스크 목록 조회.

        status_filter로 상태를 걸러낼 수 있고, 최신순으로 정렬해 반환한다.
        """
        status_filter = input_data.get("status_filter")

        manager = _get_task_manager(context)
        if manager is not None:
            all_tasks = manager.get_all()
            tasks = [_task_state_to_dict(t) for t in all_tasks]
            if status_filter:
                tasks = [t for t in tasks if t["status"] == status_filter]
        else:
            tasks = list(_fallback_tasks.values())
            if status_filter:
                tasks = [t for t in tasks if t["status"] == status_filter]

        if not tasks:
            return ToolResult.success("태스크가 없습니다.", count=0)

        # 최신순 정렬 후 한 줄씩 포맷해 합친다.
        tasks.sort(key=lambda t: t.get("created_at", 0), reverse=True)
        lines = [_format_task(t) for t in tasks]
        return ToolResult.success("\n".join(lines), count=len(tasks))

    def _handle_get(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        action="get" 처리 — 한 태스크의 상세 정보를 여러 줄로 보여준다.

        TaskManager 경로는 유형/진행률/시작·종료 시각 등 풍부한 정보를,
        폴백 경로는 인메모리 dict가 가진 만큼(우선순위/생성·수정 시각)만 보여준다.
        대상을 못 찾으면 에러를 반환한다.
        """
        task_id = input_data["task_id"]

        manager = _get_task_manager(context)
        if manager is not None:
            task_state = manager.tasks.get(task_id)
            if not task_state:
                return ToolResult.error(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")
            # TaskState의 필드들을 라벨 붙여 정렬된 여러 줄로 만든다.
            lines = [
                f"ID:       #{task_state.id}",
                f"유형:     {task_state.type.value}",
                f"상태:     {task_state.status.value}",
                f"진행률:   {task_state.progress:.0%}",
                f"설명:     {task_state.description}",
                f"시작:     {task_state.start_time.isoformat()}",
                f"종료:     {task_state.end_time.isoformat() if task_state.end_time else 'N/A'}",
            ]
            # 에러/결과는 있을 때만 덧붙인다(진행 중이면 비어 있을 수 있음).
            if task_state.error_message:
                lines.append(f"에러:     {task_state.error_message}")
            if task_state.result:
                # 결과가 매우 길 수 있으니 앞 200자만 잘라서 표시.
                lines.append(f"결과:     {task_state.result[:200]}")
            return ToolResult.success("\n".join(lines), task_id=task_id)
        else:
            # 폴백 경로: 인메모리 dict가 가진 필드만으로 상세 표시.
            task = _fallback_tasks.get(task_id)
            if not task:
                return ToolResult.error(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")

            lines = [
                f"ID:       #{task['id']}",
                f"상태:     {task['status']}",
                f"우선순위: {task.get('priority', 'medium')}",
                f"설명:     {task['description']}",
                f"생성:     {task.get('created_at', 'N/A')}",
                f"수정:     {task.get('updated_at', 'N/A')}",
            ]
            return ToolResult.success("\n".join(lines), task_id=task_id)

    async def _handle_stop(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> ToolResult:
        """
        action="stop" 처리 — 실행 중인 태스크를 강제 종료한다.

        TaskManager가 있으면 kill()을 await로 호출해 실제 asyncio.Task를 취소한다.
        (그래서 이 헬퍼만 async다.) 폴백 경로에는 취소할 실제 실행이 없으므로
        상태만 "failed"로 바꾸고 수정 시각을 갱신하는 것으로 대신한다.
        대상을 못 찾으면 에러를 반환한다.
        """
        task_id = input_data["task_id"]

        manager = _get_task_manager(context)
        if manager is not None:
            if task_id not in manager.tasks:
                return ToolResult.error(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")
            # 실제 실행 중인 태스크를 취소(코루틴/Task 중단).
            await manager.kill(task_id)
        else:
            task = _fallback_tasks.get(task_id)
            if not task:
                return ToolResult.error(f"태스크 #{task_id}을(를) 찾을 수 없습니다.")
            # 폴백에는 실행 실체가 없어 상태만 종료로 표시한다.
            task["status"] = "failed"
            task["updated_at"] = time.time()

        logger.info("Task stop: #%s", task_id)
        return ToolResult.success(f"태스크 #{task_id}을(를) 중지했습니다.")

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 UI에 "Task create...", "Task stop..." 처럼 현재 작업을 보여준다.
        action = input_data.get("action", "")
        return f"Task {action}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출을 한 줄로 요약(로그/UI용). 대상 id가 있으면 함께 표기.
        action = input_data.get("action", "")
        task_id = input_data.get("task_id", "")
        if task_id:
            return f"{action} #{task_id}"
        return action
