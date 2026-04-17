"""
QueryEngine의 ModelDispatcher 주입 경로 단위 테스트.

v7.0 Phase 9 배선 검증:
  - dispatcher가 주입되면 submit_message는 dispatcher.route()를 호출한다.
  - dispatcher가 없으면 기존 query_loop 직접 호출 경로로 폴백한다.
  - model_dispatcher 프로퍼티가 주입된 인스턴스를 그대로 노출한다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.message import StreamEvent, StreamEventType
from core.orchestrator.query_engine import QueryEngine
from core.tools.base import ToolUseContext

from tests.conftest import EnhancedMockModelProvider, MockResponse


# ─────────────────────────────────────────────
# fixture
# ─────────────────────────────────────────────
@pytest.fixture
def provider() -> EnhancedMockModelProvider:
    """폴백 경로에서 쓰이는 mock 프로바이더를 만든다."""
    return EnhancedMockModelProvider(
        responses=[MockResponse(text="fallback")]
    )


@pytest.fixture
def context(tmp_path: Path) -> ToolUseContext:
    """테스트용 도구 실행 컨텍스트를 만든다."""
    return ToolUseContext(
        cwd=str(tmp_path),
        session_id="test-session-dispatcher",
        permission_mode="bypass_permissions",
    )


def _make_fake_dispatcher() -> MagicMock:
    """route()가 StreamEvent 하나를 yield하는 mock dispatcher를 만든다."""

    async def fake_route(**kwargs):
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="from-dispatcher")

    dispatcher = MagicMock()
    dispatcher.route = MagicMock(side_effect=fake_route)
    return dispatcher


# ─────────────────────────────────────────────
# dispatcher 주입 경로
# ─────────────────────────────────────────────
class TestQueryEngineDispatcherInjection:
    """dispatcher 주입 시 submit_message가 dispatcher.route()를 호출하는지 검증한다."""

    async def test_dispatcher_injected_route_is_called(
        self,
        provider: EnhancedMockModelProvider,
        context: ToolUseContext,
    ):
        """dispatcher 주입되면 route()가 호출되고 그 이벤트가 전달된다."""
        dispatcher = _make_fake_dispatcher()
        engine = QueryEngine(
            model_provider=provider,
            tools=[],
            context=context,
            model_dispatcher=dispatcher,
        )

        events = []
        async for event in engine.submit_message("hello"):
            events.append(event)

        # dispatcher.route()가 정확히 1회 호출됐다
        assert dispatcher.route.call_count == 1
        # dispatcher가 yield한 텍스트가 그대로 전달됐다
        text_events = [
            e for e in events
            if isinstance(e, StreamEvent) and e.type == StreamEventType.TEXT_DELTA
        ]
        assert any(e.text == "from-dispatcher" for e in text_events)

    async def test_no_dispatcher_falls_back_to_query_loop(
        self,
        provider: EnhancedMockModelProvider,
        context: ToolUseContext,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """dispatcher 미주입이면 query_loop 경로로 폴백한다."""
        # query_loop을 mock하여 호출 여부를 확인한다
        mock_loop = MagicMock()

        async def fake_loop(**kwargs):
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="from-query-loop")

        mock_loop.side_effect = fake_loop
        monkeypatch.setattr(
            "core.orchestrator.query_engine.query_loop", mock_loop
        )

        engine = QueryEngine(
            model_provider=provider,
            tools=[],
            context=context,
            model_dispatcher=None,
        )

        events = []
        async for event in engine.submit_message("hello"):
            events.append(event)

        # 폴백 경로 — query_loop이 호출됐다
        assert mock_loop.call_count == 1
        text_events = [
            e for e in events
            if isinstance(e, StreamEvent) and e.type == StreamEventType.TEXT_DELTA
        ]
        assert any(e.text == "from-query-loop" for e in text_events)

    def test_dispatcher_property_exposes_injected_instance(
        self,
        provider: EnhancedMockModelProvider,
        context: ToolUseContext,
    ):
        """model_dispatcher 프로퍼티가 주입된 인스턴스를 그대로 반환한다."""
        dispatcher = _make_fake_dispatcher()
        engine = QueryEngine(
            model_provider=provider,
            tools=[],
            context=context,
            model_dispatcher=dispatcher,
        )
        assert engine.model_dispatcher is dispatcher

    def test_dispatcher_property_none_when_not_injected(
        self,
        provider: EnhancedMockModelProvider,
        context: ToolUseContext,
    ):
        """dispatcher 미주입 시 프로퍼티는 None이어야 한다."""
        engine = QueryEngine(
            model_provider=provider,
            tools=[],
            context=context,
        )
        assert engine.model_dispatcher is None
