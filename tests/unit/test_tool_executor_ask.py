# core/tools/executor.py ASK 게이트 검증 — 확인 핸들러 배선/거부/무회귀 통과.
"""
run_tool_use의 ASK(사용자 확인) 분기를 검증한다(B-1).

배경: Bash 등은 check_permissions가 항상 ASK를 반환하는데, 과거 executor는 이를
그냥 통과시켰다(확인 없음). 이제 options에 확인 핸들러(ask_handler)가 있으면 물어
거부 시 차단하고, 없으면(웹·비대화형·기존 테스트) 종전대로 통과시킨다(무회귀).

Permission 테스트 필수 시나리오 규칙(.claude/rules/testing.md)에 따라 세 경로를
모두 덮는다: 핸들러 거부→차단 / 핸들러 없음→통과 / 핸들러 허용→실행.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)
from core.tools.executor import run_tool_use


class _FakeAskTool(BaseTool):
    """항상 ASK를 반환하고, call() 실행 여부를 called 플래그로 기록하는 테스트용 도구."""

    def __init__(self) -> None:
        self.called = False

    @property
    def name(self) -> str:
        return "FakeAsk"

    @property
    def description(self) -> str:
        return "테스트용 ASK 도구"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def check_permissions(self, input_data, context) -> PermissionResult:
        return PermissionResult(behavior=PermissionBehavior.ASK, message="확인 필요")

    async def call(self, input_data, context) -> ToolResult:
        self.called = True
        return ToolResult.success("실행됨")


async def _drain(tool, context):
    """run_tool_use 제너레이터를 끝까지 소비하고, 오류 결과가 있었는지 돌려준다."""
    saw_error = False
    async for item in run_tool_use(
        {"id": "toolu_x", "name": "FakeAsk", "input": {}},
        [tool],
        context,
    ):
        # tool_result Message는 is_error 여부를 담는다(거부 시 True).
        if getattr(item, "is_error", False):
            saw_error = True
        # Message.tool_result 형태에서 error 여부를 content로도 확인(방어적).
        content = getattr(item, "content", None)
        if isinstance(content, str) and "사용자가 도구 실행을 거부" in content:
            saw_error = True
    return saw_error


@pytest.mark.asyncio
async def test_executor_ask_handler_deny_blocks():
    """확인 핸들러가 False(거부)를 반환하면 도구를 실행하지 않고 오류로 막는다."""
    calls = []

    async def deny_handler(name, message):
        calls.append((name, message))
        return False

    tool = _FakeAskTool()
    ctx = ToolUseContext(cwd=".", options={"ask_handler": deny_handler})

    saw_error = await _drain(tool, ctx)

    assert tool.called is False  # 실행 차단됨
    assert calls == [("FakeAsk", "확인 필요")]  # 핸들러가 실제로 호출됨
    assert saw_error is True  # 거부 오류가 방출됨


@pytest.mark.asyncio
async def test_executor_ask_no_handler_passthrough():
    """확인 핸들러가 없으면(웹·비대화형) 종전대로 통과시켜 도구를 실행한다(무회귀)."""
    tool = _FakeAskTool()
    ctx = ToolUseContext(cwd=".")  # options 비어 있음 — ask_handler 없음

    await _drain(tool, ctx)

    assert tool.called is True  # 통과 → 실행됨


@pytest.mark.asyncio
async def test_executor_ask_handler_allow_runs():
    """확인 핸들러가 True(허용)를 반환하면 도구를 정상 실행한다."""
    calls = []

    async def allow_handler(name, message):
        calls.append(name)
        return True

    tool = _FakeAskTool()
    ctx = ToolUseContext(cwd=".", options={"ask_handler": allow_handler})

    await _drain(tool, ctx)

    assert tool.called is True  # 허용 → 실행됨
    assert calls == ["FakeAsk"]  # 핸들러가 호출됨
