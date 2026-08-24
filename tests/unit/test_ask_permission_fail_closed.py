# 비대화형 표면의 ASK 권한 fail-closed 계약 검증.
"""
2026-08-24 이전 동작: 확인 핸들러(ask_handler)가 주입되지 않은 표면에서는 ASK
판정 도구를 **그냥 통과**시켰다. 그 결과 `nexus ask` 한 줄로 Write·Edit·Bash 가
아무 확인 없이 실행됐다(실측). fail-closed 원칙(P6)과 정면으로 어긋나고, 웹에서
Bash 를 보안 사유로 제거한 결정과도 결이 맞지 않았다.

바뀐 계약:
  · 핸들러가 있으면 종전대로 그것으로 판단한다(대화형 REPL).
  · 핸들러가 없으면 `options["ask_auto_approve"]` 가 **명시**돼야 통과한다.
  · 둘 다 없으면 거부한다.

핵심은 "조용히 열리는 대신 조용히 막힌다"로 뒤집은 것이다. 자동 승인이 필요한
표면(웹·CI)이 키를 빠뜨리면 도구가 안 도니 배선 누락이 즉시 드러난다 —
이 리포에서 반복 관측된 "웹이 자체 조립하며 키를 빠뜨린다" 사고 유형에 대한 방어다.
"""

from __future__ import annotations

import pytest

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)
from core.tools.executor import run_tool_use


class _AskTool(BaseTool):
    """항상 ASK 를 돌려주는 최소 도구 — 권한 경로만 검증한다."""

    @property
    def name(self) -> str:
        return "AskProbe"

    @property
    def description(self) -> str:
        return "probe"

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def check_permissions(self, input_data, context) -> PermissionResult:  # noqa: ANN001
        return PermissionResult(behavior=PermissionBehavior.ASK, message="확인 필요")

    async def call(self, input_data, context) -> ToolResult:  # noqa: ANN001
        return ToolResult.success("실행됨")


def _context(**options) -> ToolUseContext:
    return ToolUseContext(cwd=".", session_id="s", options=options)


async def _run(context: ToolUseContext) -> list:
    """도구 하나를 실제 실행 파이프라인에 태우고 흘러나온 항목을 모은다."""
    return [
        item
        async for item in run_tool_use(
            {"id": "tu1", "name": "AskProbe", "input": {}}, [_AskTool()], context
        )
    ]


def _result_text(items: list) -> str:
    """tool_result 본문을 이어 붙인다(블록 리스트 형태도 방어적으로 처리)."""
    parts: list[str] = []
    for item in items:
        content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                text = getattr(block, "content", None) or getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    return " ".join(parts)


def _errored(items: list) -> bool:
    if any(getattr(i, "is_error", False) for i in items):
        return True
    return "tool_use_error" in _result_text(items)


class TestFailClosedByDefault:
    @pytest.mark.asyncio
    async def test_no_handler_no_flag_is_denied(self) -> None:
        """★핵심 — 확인 수단도 자동 승인도 없으면 실행하지 않는다."""
        events = await _run(_context())
        assert _errored(events)
        assert "사용자 확인이 필요합니다" in _result_text(events)

    @pytest.mark.asyncio
    async def test_denial_message_tells_how_to_proceed(self) -> None:
        """막기만 하고 방법을 안 알려주면 사용자가 원인을 못 찾는다."""
        assert "--yes" in _result_text(await _run(_context()))

    @pytest.mark.asyncio
    async def test_explicit_false_is_still_denied(self) -> None:
        events = await _run(_context(ask_auto_approve=False))
        assert _errored(events)


class TestExplicitAutoApprove:
    @pytest.mark.asyncio
    async def test_flag_allows_execution(self) -> None:
        """웹·CI 처럼 확인 화면이 없는 표면은 명시로 통과한다."""
        events = await _run(_context(ask_auto_approve=True))
        assert not _errored(events)
        assert "실행됨" in _result_text(events)


class TestHandlerPathUnaffected:
    """대화형 REPL 경로는 그대로여야 한다(무회귀)."""

    @pytest.mark.asyncio
    async def test_handler_approval_still_runs(self) -> None:
        async def approve(tool_name, message, tool_input):  # noqa: ANN001, ARG001
            return {"approved": True, "feedback": "", "always_allow": False}

        events = await _run(_context(ask_handler_v2=approve))
        assert not _errored(events)

    @pytest.mark.asyncio
    async def test_handler_denial_still_blocks(self) -> None:
        async def deny(tool_name, message, tool_input):  # noqa: ANN001, ARG001
            return {"approved": False, "feedback": "안 됨", "always_allow": False}

        events = await _run(_context(ask_handler_v2=deny))
        assert _errored(events)

    @pytest.mark.asyncio
    async def test_handler_wins_over_missing_flag(self) -> None:
        """핸들러가 있으면 자동 승인 키가 없어도 그 판단을 따른다."""
        async def approve(tool_name, message, tool_input):  # noqa: ANN001, ARG001
            return {"approved": True, "feedback": "", "always_allow": False}

        events = await _run(_context(ask_handler_v2=approve))
        assert not _errored(events)


class TestWebSurfaceWiring:
    """★웹이 키를 빠뜨리면 도구가 통째로 막힌다 — 배선을 테스트로 고정한다."""

    def test_web_base_options_declares_auto_approve(self) -> None:
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "web" / "app.py"
        ).read_text(encoding="utf-8")
        assert '"ask_auto_approve": True' in source, (
            "웹 base_options 에서 ask_auto_approve 가 빠지면 ASK 도구가 전부 거부된다"
        )
