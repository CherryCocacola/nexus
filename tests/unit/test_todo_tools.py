# 계획 체크리스트(TodoWrite/TodoRead) 도구 + TodoStore 단위 테스트
"""
TodoWrite/TodoRead(계획 체크리스트) + TodoStore 단위 테스트.

검증 범위:
  - 목록 전체 원자 교체 + 상태 전이
  - 세션·에이전트 격리
  - 잘못된 입력 fail-closed(구 스키마/공백/50개 초과/잘못된 status)
  - 결과 metadata 구조화(todos)
  - READONLY 카테고리 분류로 PLAN 모드에서도 ALLOW
  - behavior flag(#12 순차 파티셔닝 전제)
  - prompt_assembler 체크리스트 섹션 주입
"""

from __future__ import annotations

import pytest

from core.todo_store import (
    TodoItem,
    TodoListState,
    TodoStatus,
    TodoStore,
    render_checklist,
)
from core.tools.base import ToolUseContext
from core.tools.implementations.todo_tools import TodoReadTool, TodoWriteTool


def _ctx(session_id: str = "s1", agent_id: str | None = None, store: TodoStore | None = None):
    """TodoStore가 주입된 ToolUseContext를 만든다(주입 없으면 폴백 경로)."""
    options: dict = {}
    if store is not None:
        options["todo_store"] = store
    return ToolUseContext(cwd="/tmp", session_id=session_id, agent_id=agent_id, options=options)


def _todos(*items: tuple[str, str]) -> list[dict]:
    """(content, status) 튜플들을 스키마에 맞는 todos 배열로 변환한다."""
    return [
        {"content": c, "status": s, "active_form": f"{c} 중"} for c, s in items
    ]


# ─────────────────────────────────────────────
# TodoStore — 저장소 자체
# ─────────────────────────────────────────────
class TestTodoStore:
    def test_replace_and_get_roundtrip(self):
        store = TodoStore()
        state = store.replace("s1", None, [TodoItem(content="A", status=TodoStatus.PENDING)])
        assert isinstance(state, TodoListState)
        assert store.get("s1", None).items[0].content == "A"

    def test_replace_increments_revision(self):
        store = TodoStore()
        store.replace("s1", None, [TodoItem(content="A")])
        state2 = store.replace("s1", None, [TodoItem(content="B")])
        assert state2.revision == 2

    def test_get_empty_returns_default_state(self):
        store = TodoStore()
        state = store.get("nope", None)
        assert state.items == ()
        assert state.revision == 0

    def test_todo_store_session_isolation_no_cross_talk(self):
        """세션이 다르면 목록이 섞이지 않아야 한다."""
        store = TodoStore()
        store.replace("s1", None, [TodoItem(content="s1-task")])
        store.replace("s2", None, [TodoItem(content="s2-task")])
        assert store.get("s1", None).items[0].content == "s1-task"
        assert store.get("s2", None).items[0].content == "s2-task"

    def test_todo_store_subagent_isolation_separate_from_main(self):
        """서브에이전트(agent_id)의 목록은 메인 목록을 덮어쓸 수 없다."""
        store = TodoStore()
        store.replace("s1", None, [TodoItem(content="main")])
        store.replace("s1", "sub-agent", [TodoItem(content="sub")])
        assert store.get("s1", None).items[0].content == "main"
        assert store.get("s1", "sub-agent").items[0].content == "sub"

    def test_todo_store_clear_session_removes_list(self):
        store = TodoStore()
        store.replace("s1", None, [TodoItem(content="A")])
        store.replace("s1", "sub", [TodoItem(content="B")])
        store.replace("s2", None, [TodoItem(content="C")])
        store.clear_session("s1")
        # s1의 메인·서브 모두 비고, s2는 남는다.
        assert store.get("s1", None).items == ()
        assert store.get("s1", "sub").items == ()
        assert store.get("s2", None).items[0].content == "C"

    def test_todo_item_frozen(self):
        """TodoItem은 불변이어야 한다."""
        import pydantic

        item = TodoItem(content="A")
        with pytest.raises(pydantic.ValidationError):
            item.content = "B"

    def test_render_checklist_marks(self):
        items = (
            TodoItem(content="done", status=TodoStatus.COMPLETED),
            TodoItem(content="now", status=TodoStatus.IN_PROGRESS, active_form="now 중"),
            TodoItem(content="later", status=TodoStatus.PENDING),
        )
        text = render_checklist(items)
        assert "[x] done" in text
        assert "[~] now ← now 중" in text
        assert "[ ] later" in text


# ─────────────────────────────────────────────
# TodoWrite — 전체 교체 + 상태 전이
# ─────────────────────────────────────────────
class TestTodoWrite:
    @pytest.mark.asyncio
    async def test_todo_write_full_replace_overwrites_previous_list(self):
        store = TodoStore()
        tool = TodoWriteTool()
        await tool.call({"todos": _todos(("A", "pending"), ("B", "pending"))}, _ctx(store=store))
        # 두 번째 호출이 목록 전체를 교체한다.
        await tool.call({"todos": _todos(("C", "in_progress"))}, _ctx(store=store))
        items = store.get("s1", None).items
        assert len(items) == 1
        assert items[0].content == "C"
        assert items[0].status == TodoStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_todo_write_empty_array_clears_list(self):
        store = TodoStore()
        tool = TodoWriteTool()
        await tool.call({"todos": _todos(("A", "pending"))}, _ctx(store=store))
        result = await tool.call({"todos": []}, _ctx(store=store))
        assert not result.is_error
        assert store.get("s1", None).items == ()

    @pytest.mark.asyncio
    async def test_todo_write_result_metadata_contains_structured_todos(self):
        store = TodoStore()
        tool = TodoWriteTool()
        result = await tool.call(
            {"todos": _todos(("A", "completed"), ("B", "in_progress"))}, _ctx(store=store)
        )
        assert not result.is_error
        assert "todos" in result.metadata
        assert result.metadata["todos"][0]["content"] == "A"
        assert result.metadata["stats"] == {"total": 2, "completed": 1, "in_progress": 1}

    @pytest.mark.asyncio
    async def test_todo_write_multiple_in_progress_warns_but_succeeds(self):
        """in_progress가 여러 개여도 에러가 아니라 경고(soft rule)로만 다룬다."""
        store = TodoStore()
        tool = TodoWriteTool()
        result = await tool.call(
            {"todos": _todos(("A", "in_progress"), ("B", "in_progress"))}, _ctx(store=store)
        )
        assert not result.is_error
        assert result.metadata["warnings"]
        assert len(store.get("s1", None).items) == 2

    @pytest.mark.asyncio
    async def test_todo_write_camelcase_activeform_normalized(self):
        """Claude Code 학습 잔재 activeForm(camelCase)을 active_form으로 정규화한다."""
        store = TodoStore()
        tool = TodoWriteTool()
        raw = {"todos": [{"content": "A", "status": "in_progress", "activeForm": "A 하는 중"}]}
        # validate_input이 통과해야 한다(정규화 후 active_form 존재).
        assert tool.validate_input(raw) is None
        result = await tool.call(raw, _ctx(store=store))
        assert not result.is_error
        assert store.get("s1", None).items[0].active_form == "A 하는 중"

    @pytest.mark.asyncio
    async def test_todo_write_result_echoes_checklist(self):
        store = TodoStore()
        tool = TodoWriteTool()
        result = await tool.call({"todos": _todos(("계획 수립", "pending"))}, _ctx(store=store))
        assert "계획 수립" in result.data


# ─────────────────────────────────────────────
# TodoWrite — 잘못된 입력 fail-closed (validate_input)
# ─────────────────────────────────────────────
class TestTodoWriteValidation:
    def test_over_50_items_returns_error(self):
        tool = TodoWriteTool()
        todos = [{"content": f"t{i}", "status": "pending", "active_form": ""} for i in range(51)]
        err = tool.validate_input({"todos": todos})
        assert err is not None and "50" in err

    def test_blank_content_returns_error(self):
        tool = TodoWriteTool()
        err = tool.validate_input(
            {"todos": [{"content": "  ", "status": "pending", "active_form": ""}]}
        )
        assert err is not None

    def test_invalid_status_returns_error(self):
        tool = TodoWriteTool()
        err = tool.validate_input(
            {"todos": [{"content": "A", "status": "running", "active_form": ""}]}
        )
        assert err is not None

    def test_old_schema_returns_guidance_error(self):
        """구 스키마(action/task_id/description)를 감지해 새 스키마 안내 에러를 반환한다."""
        tool = TodoWriteTool()
        err = tool.validate_input({"action": "add", "content": "old style"})
        assert err is not None and "todos" in err

    def test_todos_not_a_list_returns_error(self):
        tool = TodoWriteTool()
        err = tool.validate_input({"todos": "not a list"})
        assert err is not None

    def test_valid_input_passes(self):
        tool = TodoWriteTool()
        assert tool.validate_input({"todos": _todos(("A", "pending"))}) is None


# ─────────────────────────────────────────────
# TodoRead
# ─────────────────────────────────────────────
class TestTodoRead:
    @pytest.mark.asyncio
    async def test_todo_read_returns_current_checklist(self):
        store = TodoStore()
        store.replace("s1", None, [TodoItem(content="읽을 항목", status=TodoStatus.PENDING)])
        result = await TodoReadTool().call({}, _ctx(store=store))
        assert not result.is_error
        assert "읽을 항목" in result.data
        assert result.metadata["todos"][0]["content"] == "읽을 항목"

    @pytest.mark.asyncio
    async def test_todo_read_empty_when_no_list(self):
        result = await TodoReadTool().call({}, _ctx(store=TodoStore()))
        assert not result.is_error
        assert "비어" in result.data


# ─────────────────────────────────────────────
# 권한 — flag + 파이프라인 카테고리
# ─────────────────────────────────────────────
class TestTodoPermissions:
    @pytest.mark.asyncio
    async def test_todo_write_permission_always_allow(self):
        from core.tools.base import PermissionBehavior

        result = await TodoWriteTool().check_permissions({"todos": []}, _ctx())
        assert result.behavior == PermissionBehavior.ALLOW

    def test_todo_write_not_concurrency_safe_flag(self):
        """#12 동시성 파티셔닝 전제 — TodoWrite는 병렬 안전이 아니어야 순차 실행된다."""
        tool = TodoWriteTool()
        assert tool.is_concurrency_safe is False
        assert tool.is_read_only is False

    def test_todo_read_flags(self):
        tool = TodoReadTool()
        assert tool.is_read_only is True
        assert tool.is_concurrency_safe is True

    def test_categorize_todowrite_as_readonly(self):
        """pipeline._categorize_tool이 TodoWrite/TodoRead를 READONLY로 분류해야 한다."""
        from core.permission.pipeline import PermissionPipeline
        from core.permission.types import PermissionContext, PermissionMode, ToolCategory

        pipeline = PermissionPipeline(
            context=PermissionContext(mode=PermissionMode.DEFAULT, working_directory="/tmp")
        )
        assert pipeline._categorize_tool(TodoWriteTool()) == ToolCategory.READONLY
        assert pipeline._categorize_tool(TodoReadTool()) == ToolCategory.READONLY

    @pytest.mark.asyncio
    async def test_plan_mode_allows_todo_write(self):
        """READONLY 분류 → PLAN 모드에서도 계획 수립(TodoWrite)이 ALLOW여야 한다."""
        from core.permission.pipeline import PermissionPipeline
        from core.permission.types import AllowDecision, PermissionContext, PermissionMode

        pipeline = PermissionPipeline(
            context=PermissionContext(mode=PermissionMode.PLAN, working_directory="/tmp")
        )
        decision = await pipeline.check(TodoWriteTool(), {"todos": []}, _ctx())
        assert isinstance(decision, AllowDecision)


# ─────────────────────────────────────────────
# prompt_assembler 통합 — 체크리스트 섹션 주입
# ─────────────────────────────────────────────
class TestPromptAssemblerTodoInjection:
    @pytest.mark.asyncio
    async def test_prompt_assembler_injects_todo_section(self):
        from core.orchestrator.prompt_assembler import PromptAssembler

        store = TodoStore()
        store.replace("s1", None, [TodoItem(content="주입될 항목", status=TodoStatus.IN_PROGRESS)])
        assembler = PromptAssembler(todo_store=store)
        out = assembler._attach_todo_checklist("BASE", "s1")
        assert "현재 작업 체크리스트" in out
        assert "주입될 항목" in out

    def test_prompt_assembler_skips_when_empty(self):
        from core.orchestrator.prompt_assembler import PromptAssembler

        assembler = PromptAssembler(todo_store=TodoStore())
        out = assembler._attach_todo_checklist("BASE", "s1")
        assert out == "BASE"

    def test_prompt_assembler_skips_when_no_store(self):
        from core.orchestrator.prompt_assembler import PromptAssembler

        assembler = PromptAssembler(todo_store=None)
        assert assembler._attach_todo_checklist("BASE", "s1") == "BASE"
