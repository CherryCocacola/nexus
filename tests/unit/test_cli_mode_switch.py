# CLI 런타임 권한 모드 전환(A2) 검증 — 4지점 원자 갱신·순환·거부.
"""
NexusREPL._apply_mode_change / _cmd_mode 계약을 고정한다 (2026-08-05).

[왜 4지점을 한 번에 갱신해야 하나]
  모드는 서로 다른 네 곳이 각자 들고 있다. 하나만 바뀌면 "화면에 보이는 모드"와
  "실제 판정"이 어긋난다(기동 시 GlobalState가 갱신되지 않던 실제 버그).
    ① 파이프라인 PermissionContext.mode  ② ask_handler_v2 주입 여부
    ③ tool_use_context.permission_mode   ④ GlobalState + REPL 표시값

  또한 자동 허용 모드(auto/bypass/trust)에서는 확인 핸들러를 제거해야 프롬프트가
  뜨지 않는다. accept_edits는 자동 허용이 아니므로 핸들러를 유지해야 한다
  (핸들러 안에서 파일 수정만 선별 승인하기 때문).
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from cli.repl import NexusREPL
from core.permission.types import PermissionContext, PermissionMode
from core.state import PermissionModeValue
from core.tools.base import ToolUseContext


class _FakePipeline:
    """update_context/context만 흉내내는 파이프라인 대역."""

    def __init__(self) -> None:
        self._context = PermissionContext(
            mode=PermissionMode.DEFAULT, working_directory="/work", session_id="s1"
        )

    @property
    def context(self) -> PermissionContext:
        return self._context

    def update_context(self, context: PermissionContext) -> None:
        self._context = context


def _make_repl(mode: str = "default"):
    """생성자를 우회해 모드 전환에 필요한 필드만 심은 REPL을 만든다."""
    repl = NexusREPL.__new__(NexusREPL)
    repl._permission_mode = mode
    repl._session_allow = set()
    repl.console = Console(file=io.StringIO(), force_terminal=False, width=200)
    pipeline = _FakePipeline()
    repl._tool_ctx = ToolUseContext(
        cwd="/work", options={"permission_pipeline": pipeline}, permission_mode=mode
    )
    repl._state = SimpleNamespace(permission_mode=PermissionModeValue.DEFAULT)
    return repl, pipeline


class TestApplyModeChange:
    def test_accept_edits_updates_all_four_points(self) -> None:
        """accept_edits 전환이 4지점 전부에 반영된다(핸들러는 유지)."""
        repl, pipeline = _make_repl("default")

        assert repl._apply_mode_change("accept_edits") is True

        assert pipeline.context.mode == PermissionMode.ACCEPT_EDITS  # ①
        assert "ask_handler_v2" in repl._tool_ctx.options  # ② 유지(선별 승인)
        assert repl._tool_ctx.permission_mode == "accept_edits"  # ③
        assert repl._state.permission_mode == PermissionModeValue.ACCEPT_EDITS  # ④
        assert repl._permission_mode == "accept_edits"

    def test_pipeline_context_other_fields_preserved(self) -> None:
        """모드만 갈아끼우고 working_directory·session_id는 보존한다."""
        repl, pipeline = _make_repl("default")

        repl._apply_mode_change("plan")

        assert pipeline.context.mode == PermissionMode.PLAN
        assert pipeline.context.working_directory == "/work"
        assert pipeline.context.session_id == "s1"

    @pytest.mark.parametrize("mode", ["auto", "bypass", "trust"])
    def test_auto_allow_modes_remove_handler(self, mode: str) -> None:
        """자동 허용 모드에서는 확인 핸들러를 제거해 프롬프트가 뜨지 않게 한다."""
        repl, _ = _make_repl("default")
        repl._tool_ctx.options["ask_handler_v2"] = lambda *a: None

        repl._apply_mode_change(mode)

        assert "ask_handler_v2" not in repl._tool_ctx.options

    def test_returning_from_auto_reinjects_handler(self) -> None:
        """자동 허용에서 돌아오면 핸들러를 다시 주입한다(전환이 왕복 가능해야 한다)."""
        repl, _ = _make_repl("default")
        repl._apply_mode_change("bypass")
        assert "ask_handler_v2" not in repl._tool_ctx.options

        repl._apply_mode_change("default")

        assert "ask_handler_v2" in repl._tool_ctx.options

    def test_unknown_mode_rejected_and_state_unchanged(self) -> None:
        """모르는 모드는 거부하고 기존 상태를 건드리지 않는다."""
        repl, pipeline = _make_repl("default")

        assert repl._apply_mode_change("nonexistent") is False

        assert repl._permission_mode == "default"
        assert pipeline.context.mode == PermissionMode.DEFAULT

    def test_works_without_tool_context(self) -> None:
        """부트스트랩 실패로 컨텍스트가 없어도 표시 모드는 바뀌고 예외가 없다."""
        repl, _ = _make_repl("default")
        repl._tool_ctx = None

        assert repl._apply_mode_change("plan") is True
        assert repl._permission_mode == "plan"


class TestCmdMode:
    def test_next_cycles_three_modes(self) -> None:
        """`/mode next`는 default → accept_edits → plan → default로 순환한다."""
        repl, _ = _make_repl("default")

        seen = []
        for _ in range(4):
            asyncio.run(repl._cmd_mode(["next"]))
            seen.append(repl._permission_mode)

        assert seen == ["accept_edits", "plan", "default", "accept_edits"]

    def test_next_from_outside_cycle_jumps_to_first(self) -> None:
        """순환 목록 밖(예: bypass)에서 next를 누르면 첫 칸으로 간다."""
        repl, _ = _make_repl("bypass")

        asyncio.run(repl._cmd_mode(["next"]))

        assert repl._permission_mode == "default"

    def test_explicit_mode_argument(self) -> None:
        """`/mode plan`처럼 직접 지정하면 그 모드로 바뀐다(대소문자 무관)."""
        repl, pipeline = _make_repl("default")

        asyncio.run(repl._cmd_mode(["PLAN"]))

        assert repl._permission_mode == "plan"
        assert pipeline.context.mode == PermissionMode.PLAN

    def test_no_args_shows_table_without_changing(self) -> None:
        """인자 없이 부르면 현재 모드를 표로 보여주기만 한다."""
        repl, _ = _make_repl("accept_edits")

        asyncio.run(repl._cmd_mode([]))

        assert repl._permission_mode == "accept_edits"
        out = repl.console.file.getvalue()
        assert "accept_edits" in out
