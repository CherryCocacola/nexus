"""
executor 권한 강제 배선 통합 테스트 — run_tool_use의 shadow vs enforce.

무엇을 고정하는가 (구현 사실 core/tools/executor.py Step 6-8a):
    context.options의 permission_enforcement{enabled,mode} / permission_pipeline /
    audit_logger를 사용해 5계층 파이프라인을 배선한다.
      - enabled=False 또는 pipeline None → 파이프라인 check()를 호출조차 안 한다
        (현행 동작 100% — 무회귀). 도구는 그대로 실행된다.
      - enabled=True + mode="shadow" → check()로 판정을 계산해 AuditLogger에
        "기록만" 하고 차단하지 않는다(관측 전용). 도구는 여전히 실행된다.
      - enabled=True + mode="enforce" + 판정 deny → is_error tool_result를 yield하고
        중단한다(실제 차단). 도구는 실행되지 않는다.
      - enforce라도 판정이 ask/allow면 통과한다(도구 실행). ★P3(ASK 강제)는 아직
        미구현이므로 ASK를 차단으로 단언하지 않는다.★

왜 통합 테스트인가:
    executor(파이프라인 호출/감사 기록/차단 게이트) + 실제 AuditLogger + 도구
    실행 경로가 함께 맞물려 동작하는지를 end-to-end로 확인해야 하기 때문이다.
    파이프라인 자체 판정 로직은 mock으로 고정하고, executor의 "배선"만 검증한다.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from core.message import Message
from core.permission.types import PermissionAuditEntry
from core.security.audit import AuditLogger
from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)
from core.tools.executor import run_tool_use


class _Decision:
    """파이프라인 판정 mock — executor는 .type/.message만 참조한다."""

    def __init__(self, decision_type: str, message: str = "정책 위반") -> None:
        self.type = decision_type
        self.message = message


class _SpyPipeline:
    """
    PermissionPipeline mock(스파이).

    executor가 실제로 쓰는 두 메서드만 흉내낸다:
      - check(): 호출 여부를 기록(check_called)하고 미리 정한 판정을 돌려준다.
      - get_recent_audit(): 감사 기록용 엔트리 하나를 돌려준다.
    이렇게 하면 파이프라인 판정 로직과 무관하게 "executor의 배선"만 격리 검증한다.
    """

    def __init__(self, decision_type: str) -> None:
        self._decision = _Decision(decision_type)
        self.check_called = False
        # 감사 로그 경로에서 재사용될 실제 엔트리(AuditLogger가 직렬화할 수 있도록).
        self._entry = PermissionAuditEntry(
            tool_name="Echo",
            decision=decision_type,
            reason="test",
        )

    async def check(self, tool: BaseTool, tool_input: dict, ctx: ToolUseContext) -> _Decision:
        self.check_called = True
        return self._decision

    def get_recent_audit(self, n: int = 1) -> list[PermissionAuditEntry]:
        return [self._entry]


class _CountingTool(BaseTool):
    """
    실행 여부를 셀 수 있는 최소 도구.

    call()이 불릴 때마다 call_count를 올린다 → "차단됐는지(0) 실행됐는지(1)"를
    직접 관찰할 수 있다. check_permissions는 ALLOW라 executor의 '현행' 게이트는
    통과하므로, 실행이 막혔다면 그것은 오직 enforce 차단 때문이다.
    """

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def name(self) -> str:
        return "Echo"

    @property
    def description(self) -> str:
        return "echo tool"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object"}

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        self.call_count += 1
        return ToolResult.success("done")


async def _collect(gen: AsyncGenerator) -> list[Any]:
    """AsyncGenerator의 모든 이벤트(StreamEvent | Message)를 소비해 리스트로 모은다."""
    out: list[Any] = []
    async for item in gen:
        out.append(item)
    return out


def _blocked_by_permission(events: list[Any]) -> bool:
    """'권한 거부' is_error tool_result가 yield됐는지 판정한다(실제 차단 신호)."""
    for e in events:
        if isinstance(e, Message) and e.is_error:
            content = e.content if isinstance(e.content, str) else str(e.content)
            if "권한 거부" in content:
                return True
    return False


def _make_context(tmp_path, *, enabled: bool, mode: str, pipeline, audit_logger) -> ToolUseContext:
    """executor가 읽는 options(permission_enforcement/pipeline/audit_logger)를 조립한다."""
    return ToolUseContext(
        cwd=str(tmp_path),
        options={
            "permission_enforcement": {"enabled": enabled, "mode": mode},
            "permission_pipeline": pipeline,
            "audit_logger": audit_logger,
        },
    )


_TOOL_USE = {"id": "t1", "name": "Echo", "input": {}}


class TestEnforcementDisabled:
    """enabled=False면 파이프라인을 호출조차 하지 않고 도구를 실행한다(무회귀)."""

    async def test_disabled_does_not_call_pipeline_and_runs_tool(self, tmp_path) -> None:
        tool = _CountingTool()
        spy = _SpyPipeline("deny")  # deny로 세팅해도, 비활성이면 아예 호출되면 안 된다
        logger = AuditLogger(log_path=str(tmp_path / "audit.log"))
        ctx = _make_context(
            tmp_path, enabled=False, mode="enforce", pipeline=spy, audit_logger=logger
        )

        events = await _collect(run_tool_use(_TOOL_USE, [tool], ctx))

        # 파이프라인 미호출 + 감사 기록 없음 + 도구는 정상 실행 + 차단 없음.
        assert spy.check_called is False
        assert logger.get_recent() == []
        assert tool.call_count == 1
        assert _blocked_by_permission(events) is False


class TestShadowMode:
    """shadow면 판정을 감사 로그에 기록만 하고 차단하지 않는다(도구는 실행)."""

    async def test_shadow_deny_records_audit_but_runs_tool(self, tmp_path) -> None:
        tool = _CountingTool()
        spy = _SpyPipeline("deny")  # deny 판정이어도 shadow는 절대 차단하지 않는다
        logger = AuditLogger(log_path=str(tmp_path / "audit.log"))
        ctx = _make_context(
            tmp_path, enabled=True, mode="shadow", pipeline=spy, audit_logger=logger
        )

        events = await _collect(run_tool_use(_TOOL_USE, [tool], ctx))

        # 파이프라인 호출됨 + 감사 로그 1건 기록 + 차단 없음 + 도구 실행됨.
        assert spy.check_called is True
        assert len(logger.get_recent()) == 1
        assert _blocked_by_permission(events) is False
        assert tool.call_count == 1


class TestEnforceMode:
    """enforce에서 deny면 실제 차단, ask/allow는 통과(P3 미구현)."""

    async def test_enforce_deny_blocks_and_skips_tool(self, tmp_path) -> None:
        """deny 판정이면 is_error tool_result를 내고 도구를 실행하지 않는다."""
        tool = _CountingTool()
        spy = _SpyPipeline("deny")
        logger = AuditLogger(log_path=str(tmp_path / "audit.log"))
        ctx = _make_context(
            tmp_path, enabled=True, mode="enforce", pipeline=spy, audit_logger=logger
        )

        events = await _collect(run_tool_use(_TOOL_USE, [tool], ctx))

        assert spy.check_called is True
        assert _blocked_by_permission(events) is True  # 실제 차단 신호
        assert tool.call_count == 0  # 도구 미실행
        # 차단 사실도 감사 로그에 남아야 한다(차단 처리 '전에' 기록).
        assert len(logger.get_recent()) == 1

    async def test_enforce_allow_passes_and_runs_tool(self, tmp_path) -> None:
        """allow 판정은 통과 — 도구가 정상 실행된다."""
        tool = _CountingTool()
        spy = _SpyPipeline("allow")
        logger = AuditLogger(log_path=str(tmp_path / "audit.log"))
        ctx = _make_context(
            tmp_path, enabled=True, mode="enforce", pipeline=spy, audit_logger=logger
        )

        events = await _collect(run_tool_use(_TOOL_USE, [tool], ctx))

        assert spy.check_called is True
        assert _blocked_by_permission(events) is False
        assert tool.call_count == 1

    async def test_enforce_ask_passes_and_runs_tool(self, tmp_path) -> None:
        """★P3 미구현★ enforce라도 ask 판정은 아직 통과시킨다(차단으로 단언하지 않는다)."""
        tool = _CountingTool()
        spy = _SpyPipeline("ask")
        logger = AuditLogger(log_path=str(tmp_path / "audit.log"))
        ctx = _make_context(
            tmp_path, enabled=True, mode="enforce", pipeline=spy, audit_logger=logger
        )

        events = await _collect(run_tool_use(_TOOL_USE, [tool], ctx))

        assert spy.check_called is True
        assert _blocked_by_permission(events) is False  # ASK는 아직 차단 안 함
        assert tool.call_count == 1
