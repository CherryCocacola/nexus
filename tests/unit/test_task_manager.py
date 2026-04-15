"""
TaskManager 단위 테스트.

core/task.py의 TaskType, TaskStatus, TaskState, TaskManager를 검증한다.
비동기 태스크 라이프사이클(생성→실행→완료/실패/강제종료)이 올바르게 동작하는지 확인한다.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from core.task import TaskManager, TaskState, TaskStatus, TaskType


# ─────────────────────────────────────────────
# TaskType 테스트
# ─────────────────────────────────────────────
class TestTaskType:
    """TaskType enum과 prefix 프로퍼티를 검증한다."""

    def test_all_types_have_prefix(self):
        """모든 TaskType이 비어있지 않은 prefix를 가져야 한다."""
        for tt in TaskType:
            assert len(tt.prefix) >= 1, f"{tt.value}의 prefix가 비어 있다"

    def test_prefix_values(self):
        """각 TaskType의 prefix가 사양서 정의와 일치해야 한다."""
        assert TaskType.LOCAL_BASH.prefix == "b"
        assert TaskType.LOCAL_AGENT.prefix == "a"
        assert TaskType.REMOTE_AGENT.prefix == "r"
        assert TaskType.TEAMMATE.prefix == "t"
        assert TaskType.WORKFLOW.prefix == "w"
        assert TaskType.MONITOR.prefix == "m"
        assert TaskType.TRAINING.prefix == "tr"

    def test_task_type_count(self):
        """TaskType은 정확히 7가지여야 한다."""
        assert len(TaskType) == 7


# ─────────────────────────────────────────────
# TaskStatus 테스트
# ─────────────────────────────────────────────
class TestTaskStatus:
    """TaskStatus enum을 검증한다."""

    def test_status_count(self):
        """TaskStatus는 정확히 5가지여야 한다."""
        assert len(TaskStatus) == 5

    def test_status_values(self):
        """각 상태의 문자열 값이 올바른지 확인한다."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.KILLED.value == "killed"


# ─────────────────────────────────────────────
# TaskManager 테스트
# ─────────────────────────────────────────────
class TestTaskManager:
    """TaskManager의 전체 라이프사이클을 검증한다."""

    def test_create_generates_unique_id(self):
        """create()가 prefix + 8자 형식의 고유 ID를 생성해야 한다."""
        mgr = TaskManager()
        tid1 = mgr.create(TaskType.LOCAL_BASH, "task 1")
        tid2 = mgr.create(TaskType.LOCAL_BASH, "task 2")

        # prefix 'b' + uuid 8자 = 9자
        assert tid1.startswith("b")
        assert len(tid1) == 9
        # 두 ID가 서로 달라야 한다
        assert tid1 != tid2

    def test_create_with_string_type(self):
        """문자열 task_type이 TaskType으로 올바르게 변환되어야 한다."""
        mgr = TaskManager()
        tid = mgr.create("local_bash", "bash task")

        assert tid.startswith("b")
        assert mgr.tasks[tid].type == TaskType.LOCAL_BASH

    def test_create_invalid_type_defaults_workflow(self):
        """잘못된 task_type 문자열은 WORKFLOW로 폴백되어야 한다."""
        mgr = TaskManager()
        tid = mgr.create("invalid_type", "fallback task")

        assert tid.startswith("w")  # WORKFLOW prefix
        assert mgr.tasks[tid].type == TaskType.WORKFLOW

    def test_create_training_type(self):
        """TRAINING 타입은 'tr' prefix를 가져야 한다."""
        mgr = TaskManager()
        tid = mgr.create(TaskType.TRAINING, "학습 태스크")

        assert tid.startswith("tr")
        # prefix 'tr' + uuid 8자 = 10자
        assert len(tid) == 10
        assert mgr.tasks[tid].type == TaskType.TRAINING

    def test_create_initial_state(self):
        """생성 직후 태스크의 초기 상태가 올바른지 확인한다."""
        mgr = TaskManager()
        tid = mgr.create(TaskType.WORKFLOW, "test task")

        task = mgr.tasks[tid]
        assert task.status == TaskStatus.PENDING
        assert task.progress == 0.0
        assert task.description == "test task"
        assert task.end_time is None
        assert task.error_message is None
        assert task.result is None

    @pytest.mark.asyncio
    async def test_run_completes_successfully(self):
        """정상 완료된 코루틴의 상태가 COMPLETED이고 progress가 1.0이어야 한다."""
        mgr = TaskManager()
        tid = mgr.create(TaskType.WORKFLOW, "success task")

        # 즉시 완료되는 코루틴
        async def success_coro():
            return "done"

        await mgr.run(tid, success_coro())
        # asyncio.Task가 완료될 때까지 대기
        await asyncio.sleep(0.05)

        task = mgr.tasks[tid]
        assert task.status == TaskStatus.COMPLETED
        assert task.progress == 1.0
        assert task.result == "done"
        assert task.end_time is not None

    @pytest.mark.asyncio
    async def test_run_failure_sets_failed(self):
        """예외가 발생한 코루틴의 상태가 FAILED이고 error_message가 설정되어야 한다."""
        mgr = TaskManager()
        tid = mgr.create(TaskType.WORKFLOW, "fail task")

        async def fail_coro():
            raise RuntimeError("something broke")

        await mgr.run(tid, fail_coro())
        await asyncio.sleep(0.05)

        task = mgr.tasks[tid]
        assert task.status == TaskStatus.FAILED
        assert "something broke" in task.error_message
        assert task.end_time is not None

    @pytest.mark.asyncio
    async def test_run_background_fire_and_forget(self):
        """run_background()가 즉시 반환되고 태스크가 배경에서 완료되어야 한다."""
        mgr = TaskManager()
        tid = mgr.create(TaskType.LOCAL_BASH, "bg task")

        async def bg_coro():
            await asyncio.sleep(0.01)
            return "bg done"

        # run_background()는 즉시 반환되어야 한다 (await 아님)
        mgr.run_background(tid, bg_coro())

        # 즉시 확인 — 아직 실행 중이거나 PENDING일 수 있다
        assert mgr.tasks[tid].status in (TaskStatus.PENDING, TaskStatus.RUNNING)

        # 배경 태스크 완료 대기
        await asyncio.sleep(0.1)
        assert mgr.tasks[tid].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_kill_cancels_running_task(self):
        """kill()이 실행 중인 태스크를 KILLED 상태로 만들어야 한다."""
        mgr = TaskManager()
        tid = mgr.create(TaskType.LOCAL_AGENT, "long task")

        async def long_coro():
            await asyncio.sleep(10)  # 오래 걸리는 작업

        await mgr.run(tid, long_coro())
        # 태스크가 RUNNING이 될 때까지 잠시 대기
        await asyncio.sleep(0.05)
        assert mgr.tasks[tid].status == TaskStatus.RUNNING

        # 강제 종료
        await mgr.kill(tid)

        task = mgr.tasks[tid]
        assert task.status == TaskStatus.KILLED
        assert task.end_time is not None
        # 실행 목록에서 제거되어야 한다
        assert tid not in mgr._running

    @pytest.mark.asyncio
    async def test_kill_pending_task(self):
        """실행 전(PENDING) 태스크도 kill()로 KILLED 상태가 되어야 한다."""
        mgr = TaskManager()
        tid = mgr.create(TaskType.WORKFLOW, "pending task")

        assert mgr.tasks[tid].status == TaskStatus.PENDING
        await mgr.kill(tid)
        assert mgr.tasks[tid].status == TaskStatus.KILLED

    def test_update_progress_clamp(self):
        """update_progress()가 0.0~1.0 범위로 클램프해야 한다."""
        mgr = TaskManager()
        tid = mgr.create(TaskType.WORKFLOW, "progress task")

        # 정상 범위
        mgr.update_progress(tid, 0.5)
        assert mgr.tasks[tid].progress == 0.5

        # 상한 초과 → 1.0으로 클램프
        mgr.update_progress(tid, 1.5)
        assert mgr.tasks[tid].progress == 1.0

        # 하한 미달 → 0.0으로 클램프
        mgr.update_progress(tid, -0.3)
        assert mgr.tasks[tid].progress == 0.0

    def test_update_progress_nonexistent_task(self):
        """존재하지 않는 태스크의 진행률 업데이트는 무시되어야 한다."""
        mgr = TaskManager()
        # 에러 없이 무시
        mgr.update_progress("nonexistent", 0.5)

    def test_get_active_filters_correctly(self):
        """get_active()가 PENDING과 RUNNING 태스크만 반환해야 한다."""
        mgr = TaskManager()

        tid1 = mgr.create(TaskType.WORKFLOW, "pending")
        tid2 = mgr.create(TaskType.WORKFLOW, "running")
        tid3 = mgr.create(TaskType.WORKFLOW, "completed")
        tid4 = mgr.create(TaskType.WORKFLOW, "failed")

        # 상태 직접 설정
        mgr.tasks[tid2].status = TaskStatus.RUNNING
        mgr.tasks[tid3].status = TaskStatus.COMPLETED
        mgr.tasks[tid4].status = TaskStatus.FAILED

        active = mgr.get_active()
        active_ids = {t.id for t in active}

        # PENDING, RUNNING만 포함
        assert tid1 in active_ids
        assert tid2 in active_ids
        # COMPLETED, FAILED는 미포함
        assert tid3 not in active_ids
        assert tid4 not in active_ids

    def test_get_all_returns_everything(self):
        """get_all()이 모든 상태의 태스크를 반환해야 한다."""
        mgr = TaskManager()

        for i in range(5):
            mgr.create(TaskType.WORKFLOW, f"task {i}")

        all_tasks = mgr.get_all()
        assert len(all_tasks) == 5

    @pytest.mark.asyncio
    async def test_on_complete_callback(self):
        """on_complete()로 등록한 콜백이 태스크 완료 시 호출되어야 한다."""
        mgr = TaskManager()
        tid = mgr.create(TaskType.WORKFLOW, "callback task")

        # 콜백 결과를 저장할 리스트
        callback_results: list[TaskState] = []

        def sync_callback(task: TaskState):
            callback_results.append(task)

        mgr.on_complete(tid, sync_callback)

        async def simple_coro():
            return "callback done"

        await mgr.run(tid, simple_coro())
        await asyncio.sleep(0.05)

        # 콜백이 호출되었어야 한다
        assert len(callback_results) == 1
        assert callback_results[0].status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_on_complete_async_callback(self):
        """비동기 콜백도 올바르게 호출되어야 한다."""
        mgr = TaskManager()
        tid = mgr.create(TaskType.WORKFLOW, "async callback task")

        callback_results: list[str] = []

        async def async_callback(task: TaskState):
            callback_results.append(f"async:{task.id}")

        mgr.on_complete(tid, async_callback)

        async def simple_coro():
            return "ok"

        await mgr.run(tid, simple_coro())
        await asyncio.sleep(0.05)

        assert len(callback_results) == 1
        assert callback_results[0].startswith("async:")

    def test_cleanup_old_removes_expired(self):
        """cleanup_old()가 만료된 완료 태스크를 정리해야 한다."""
        mgr = TaskManager()

        # 오래된 완료 태스크 생성
        tid_old = mgr.create(TaskType.WORKFLOW, "old task")
        mgr.tasks[tid_old].status = TaskStatus.COMPLETED
        # 25시간 전에 완료된 것으로 설정
        mgr.tasks[tid_old].end_time = datetime.utcnow() - timedelta(hours=25)

        # 최근 완료 태스크 생성
        tid_new = mgr.create(TaskType.WORKFLOW, "new task")
        mgr.tasks[tid_new].status = TaskStatus.COMPLETED
        mgr.tasks[tid_new].end_time = datetime.utcnow() - timedelta(hours=1)

        # PENDING 태스크 (정리 대상 아님)
        tid_pending = mgr.create(TaskType.WORKFLOW, "pending task")

        removed = mgr.cleanup_old(max_age_hours=24)

        assert removed == 1
        assert tid_old not in mgr.tasks  # 정리됨
        assert tid_new in mgr.tasks  # 보존됨
        assert tid_pending in mgr.tasks  # 보존됨

    def test_cleanup_old_no_expired(self):
        """만료된 태스크가 없으면 아무것도 정리하지 않는다."""
        mgr = TaskManager()
        mgr.create(TaskType.WORKFLOW, "active task")

        removed = mgr.cleanup_old(max_age_hours=24)
        assert removed == 0

    @pytest.mark.asyncio
    async def test_run_nonexistent_task_raises(self):
        """존재하지 않는 태스크에 run()을 호출하면 ValueError가 발생해야 한다."""
        mgr = TaskManager()

        async def dummy():
            return "x"

        with pytest.raises(ValueError, match="찾을 수 없습니다"):
            await mgr.run("nonexistent", dummy())


# ─────────────────────────────────────────────
# TaskTools + TaskManager 연동 테스트
# ─────────────────────────────────────────────
class TestTaskToolsWithManager:
    """task_tools.py가 TaskManager와 올바르게 연동되는지 검증한다."""

    def _make_context_with_manager(self, manager: TaskManager):
        """TaskManager가 주입된 ToolUseContext를 생성한다."""
        from core.tools.base import ToolUseContext

        return ToolUseContext(
            cwd="/tmp",
            session_id="test-session",
            options={"task_manager": manager},
        )

    @pytest.mark.asyncio
    async def test_todo_write_create_uses_manager(self):
        """TodoWriteTool의 create가 TaskManager를 사용해야 한다."""
        from core.tools.implementations.task_tools import TodoWriteTool

        mgr = TaskManager()
        ctx = self._make_context_with_manager(mgr)
        tool = TodoWriteTool()

        result = await tool.call({"description": "manager task"}, ctx)

        assert not result.is_error
        assert "태스크를 생성했습니다" in result.data
        # TaskManager에 태스크가 생성되어야 한다
        assert len(mgr.tasks) == 1

    @pytest.mark.asyncio
    async def test_todo_read_uses_manager(self):
        """TodoReadTool이 TaskManager에서 태스크를 조회해야 한다."""
        from core.tools.implementations.task_tools import TodoReadTool

        mgr = TaskManager()
        mgr.create(TaskType.WORKFLOW, "task A")
        mgr.create(TaskType.WORKFLOW, "task B")

        ctx = self._make_context_with_manager(mgr)
        tool = TodoReadTool()

        result = await tool.call({}, ctx)

        assert not result.is_error
        assert "task A" in result.data
        assert "task B" in result.data

    @pytest.mark.asyncio
    async def test_task_tool_stop_uses_kill(self):
        """TaskTool의 stop action이 TaskManager.kill()을 사용해야 한다."""
        from core.tools.implementations.task_tools import TaskTool

        mgr = TaskManager()
        tid = mgr.create(TaskType.WORKFLOW, "stop me")

        ctx = self._make_context_with_manager(mgr)
        tool = TaskTool()

        result = await tool.call({"action": "stop", "task_id": tid}, ctx)

        assert not result.is_error
        assert mgr.tasks[tid].status == TaskStatus.KILLED

    @pytest.mark.asyncio
    async def test_task_tool_get_with_manager(self):
        """TaskTool의 get action이 TaskState 상세 정보를 반환해야 한다."""
        from core.tools.implementations.task_tools import TaskTool

        mgr = TaskManager()
        tid = mgr.create(TaskType.LOCAL_BASH, "detailed task")

        ctx = self._make_context_with_manager(mgr)
        tool = TaskTool()

        result = await tool.call({"action": "get", "task_id": tid}, ctx)

        assert not result.is_error
        assert "유형:" in result.data
        assert "local_bash" in result.data
        assert "진행률:" in result.data
