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


async def _drain_collect_contents(tool, context) -> list[str]:
    """run_tool_use를 끝까지 소비하며 문자열 content들을 수집한다(피드백 검증용)."""
    contents: list[str] = []
    async for item in run_tool_use(
        {"id": "toolu_x", "name": "FakeAsk", "input": {"x": 1}},
        [tool],
        context,
    ):
        content = getattr(item, "content", None)
        if isinstance(content, str):
            contents.append(content)
        # Message.tool_result의 content가 블록 리스트인 경우도 방어적으로 편평화.
        elif isinstance(content, list):
            for block in content:
                text = getattr(block, "content", None) or getattr(block, "text", None)
                if isinstance(text, str):
                    contents.append(text)
    return contents


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
async def test_executor_ask_no_handler_denies():
    """확인 핸들러도 자동 승인도 없으면 **거부**한다 (2026-08-24 계약 변경).

    이전에는 이 경우 그냥 통과시켰다. 그 결과 `nexus ask` 한 줄로 Write·Edit·Bash 가
    아무 확인 없이 실행됐다(실측). fail-closed 원칙(P6)과 어긋나 뒤집었다.
    자동 승인이 필요한 표면은 options["ask_auto_approve"]=True 를 **명시**해야 한다 —
    실수로 빠뜨리면 조용히 열리는 대신 조용히 막혀 배선 누락이 즉시 드러난다.
    """
    tool = _FakeAskTool()
    ctx = ToolUseContext(cwd=".")  # 핸들러도 자동 승인 키도 없음

    await _drain(tool, ctx)

    assert tool.called is False  # 거부 → 실행되지 않음


@pytest.mark.asyncio
async def test_executor_ask_explicit_auto_approve_runs():
    """웹·CI 처럼 확인 화면이 없는 표면은 명시 키로 통과한다."""
    tool = _FakeAskTool()
    ctx = ToolUseContext(cwd=".", options={"ask_auto_approve": True})

    await _drain(tool, ctx)

    assert tool.called is True


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


# ─── ask_handler v2 계약 (CLI Stage 1 A3) ───


@pytest.mark.asyncio
async def test_executor_ask_handler_v2_approve_runs():
    """v2 핸들러가 approved=True를 반환하면 도구를 정상 실행한다."""
    calls = []

    async def v2_allow(name, message, tool_input):
        calls.append((name, message, tool_input))
        return {"approved": True, "feedback": "", "always_allow": False}

    tool = _FakeAskTool()
    ctx = ToolUseContext(cwd=".", options={"ask_handler_v2": v2_allow})

    await _drain(tool, ctx)

    assert tool.called is True  # 허용 → 실행됨
    # v2는 tool_input까지 전달받는다(diff 미리보기 등 입력 기반 표시용 계약).
    assert calls == [("FakeAsk", "확인 필요", {})]


@pytest.mark.asyncio
async def test_executor_ask_handler_v2_deny_feedback_forwarded():
    """v2 핸들러가 거부+피드백을 반환하면 차단하고 피드백을 결과에 실어 전달한다."""

    async def v2_deny(name, message, tool_input):
        return {"approved": False, "feedback": "지금은 하지 마", "always_allow": False}

    tool = _FakeAskTool()
    ctx = ToolUseContext(cwd=".", options={"ask_handler_v2": v2_deny})

    contents = await _drain_collect_contents(tool, ctx)

    assert tool.called is False  # 실행 차단됨
    # 거부 사실 + 사용자 피드백이 모델에 전달될 결과 안에 함께 담겨야 한다.
    joined = "\n".join(contents)
    assert "사용자가 도구 실행을 거부" in joined
    assert "지금은 하지 마" in joined


@pytest.mark.asyncio
async def test_executor_ask_handler_v2_preferred_over_v1():
    """v2와 v1이 모두 주입되면 v2만 호출한다(v2 우선, v1은 폴백 전용)."""
    v1_calls = []
    v2_calls = []

    async def v1_handler(name, message):
        v1_calls.append(name)
        return False  # v1이 불리면 차단될 것 — 불리지 않아야 한다

    async def v2_handler(name, message, tool_input):
        v2_calls.append(name)
        return {"approved": True, "feedback": "", "always_allow": False}

    tool = _FakeAskTool()
    ctx = ToolUseContext(
        cwd=".",
        options={"ask_handler": v1_handler, "ask_handler_v2": v2_handler},
    )

    await _drain(tool, ctx)

    assert tool.called is True  # v2가 허용 → 실행됨
    assert v2_calls == ["FakeAsk"]  # v2 호출됨
    assert v1_calls == []  # v1은 호출되지 않음


@pytest.mark.asyncio
async def test_executor_ask_handler_v2_non_dict_bool_fallback():
    """v2 핸들러가 dict 대신 bool을 반환해도(잘못 구현) 안전하게 근사 처리한다."""

    async def v2_bool_deny(name, message, tool_input):
        return False  # 계약 위반 반환 — bool로 근사되어 거부 처리돼야 한다

    tool = _FakeAskTool()
    ctx = ToolUseContext(cwd=".", options={"ask_handler_v2": v2_bool_deny})

    saw_error = await _drain(tool, ctx)

    assert tool.called is False  # 거부로 근사 → 차단
    assert saw_error is True
