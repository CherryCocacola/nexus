"""
core/orchestrator/model_dispatcher.py 단위 테스트.

ModelDispatcher의 초기화, 라우팅, 프로퍼티를 검증한다.
query_loop은 mock 처리하여 디스패처의 라우팅 로직만 격리 테스트한다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.message import Message, StreamEvent, StreamEventType
from core.model.hardware_tier import HardwareTier
from core.model.inference import ModelConfig, ModelProvider
from core.tools.base import BaseTool, ToolUseContext

# conftest의 EnhancedMockModelProvider, MockResponse를 재활용한다
from tests.conftest import EnhancedMockModelProvider, MockResponse


# ─────────────────────────────────────────────
# 테스트용 fixture
# ─────────────────────────────────────────────
@pytest.fixture
def mock_worker_provider() -> EnhancedMockModelProvider:
    """Worker용 mock 모델 프로바이더를 생성한다."""
    return EnhancedMockModelProvider(
        responses=[MockResponse(text="worker response")]
    )


@pytest.fixture
def mock_scout_provider() -> EnhancedMockModelProvider:
    """Scout용 mock 모델 프로바이더를 생성한다."""
    return EnhancedMockModelProvider(
        responses=[MockResponse(text="scout plan")]
    )


@pytest.fixture
def mock_tools() -> list[MagicMock]:
    """테스트용 mock 도구 리스트를 생성한다."""
    tools = []
    for name in ["Read", "Write", "Bash"]:
        tool = MagicMock(spec=BaseTool)
        tool.name = name
        tools.append(tool)
    return tools


@pytest.fixture
def mock_context(tmp_path: Path) -> ToolUseContext:
    """테스트용 도구 실행 컨텍스트를 생성한다."""
    return ToolUseContext(
        cwd=str(tmp_path),
        session_id="test-session",
        permission_mode="bypass_permissions",
    )


# ─────────────────────────────────────────────
# ModelDispatcher 초기화 테스트
# ─────────────────────────────────────────────
class TestModelDispatcherInit:
    """ModelDispatcher 초기화 로직을 검증한다."""

    def test_init_tier_s_scout_enabled(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_scout_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """TIER_S + scout_provider가 있으면 scout가 활성화된다."""
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_S,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
            scout_provider=mock_scout_provider,
        )
        assert dispatcher.scout_enabled is True
        assert dispatcher.tier is HardwareTier.TIER_S

    def test_init_tier_s_no_scout_provider_disabled(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """TIER_S이지만 scout_provider가 None이면 scout는 비활성화된다."""
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_S,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
            scout_provider=None,
        )
        assert dispatcher.scout_enabled is False

    def test_init_scout_enabled_follows_provider_presence(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_scout_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """
        v7.0 Phase 9 재설계 이후: scout_enabled는 "Scout 서버 연결 여부"만
        나타낸다. TIER와 무관하게 scout_provider 존재 여부로 결정된다.

        실제 Scout 호출 여부는 Worker가 AgentTool로 판단하므로,
        이 플래그는 단순 관측 지표다.
        """
        from core.orchestrator.model_dispatcher import ModelDispatcher

        # TIER_M + scout_provider 있음 → scout_available=True
        dispatcher_m_with = ModelDispatcher(
            tier=HardwareTier.TIER_M,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
            scout_provider=mock_scout_provider,
        )
        assert dispatcher_m_with.scout_enabled is True

        # TIER_L + scout_provider 없음 → False
        dispatcher_l_without = ModelDispatcher(
            tier=HardwareTier.TIER_L,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
        )
        assert dispatcher_l_without.scout_enabled is False


# ─────────────────────────────────────────────
# ModelDispatcher 프로퍼티 테스트
# ─────────────────────────────────────────────
class TestModelDispatcherProperties:
    """프로퍼티 반환값을 검증한다."""

    def test_tier_property(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """tier 프로퍼티가 초기화 시 전달한 값을 정확히 반환한다."""
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_M,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
        )
        assert dispatcher.tier is HardwareTier.TIER_M

    def test_worker_provider_property(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """worker_provider 프로퍼티가 올바른 인스턴스를 반환한다."""
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_S,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
        )
        assert dispatcher.worker_provider is mock_worker_provider

    def test_worker_tools_property(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """worker_tools 프로퍼티가 초기화 시 전달한 도구 목록을 반환한다."""
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_S,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
        )
        assert dispatcher.worker_tools is mock_tools
        assert len(dispatcher.worker_tools) == 3


# ─────────────────────────────────────────────
# stats 프로퍼티 — Ch 17 SessionMetrics 반영
# ─────────────────────────────────────────────
class TestModelDispatcherStatsFields:
    """stats 프로퍼티가 Ch 17에서 요구하는 키를 모두 노출하는지 검증한다."""

    def test_stats_contains_all_required_keys(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """stats가 Scout 관련 필수 키 5개를 모두 포함한다."""
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_M,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
        )
        stats = dispatcher.stats
        # Ch 17 요구 필드 — SessionMetrics에 그대로 노출된다
        assert "tier" in stats
        assert "scout_enabled" in stats
        assert "scout_calls" in stats
        assert "scout_fallback_count" in stats
        assert "scout_avg_latency_ms" in stats

    def test_stats_initial_zero_values(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """Scout 호출 전에는 통계가 모두 0이어야 한다."""
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_M,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
        )
        stats = dispatcher.stats
        assert stats["scout_calls"] == 0
        assert stats["scout_fallback_count"] == 0
        assert stats["scout_avg_latency_ms"] == 0.0

    def test_stats_backward_compat_keys_always_zero(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """
        v7.0 Phase 9 재설계 이후: stats의 Scout 수치는 항상 0이다.

        실제 Scout 호출 통계는 AgentTool.get_stats()에서 집계한다.
        Dispatcher는 키만 유지하여 기존 대시보드가 깨지지 않게 한다.
        """
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_M,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
        )

        stats = dispatcher.stats
        assert stats["scout_calls"] == 0
        assert stats["scout_fallback_count"] == 0
        assert stats["scout_fallbacks"] == 0
        assert stats["scout_avg_latency_ms"] == 0.0
        # 운영자가 실제 Scout 통계 위치를 찾을 수 있도록 안내 노트를 포함한다
        assert "AgentTool" in stats.get("note", "")


# ─────────────────────────────────────────────
# ModelDispatcher.route() 테스트
# ─────────────────────────────────────────────
class TestModelDispatcherRoute:
    """route() 메서드의 라우팅 로직을 검증한다."""

    async def test_route_tier_m_passthrough_yields_events(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """TIER_M에서 route()는 query_loop을 호출하고 StreamEvent를 yield한다."""
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_M,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
        )

        # query_loop을 mock하여 StreamEvent를 yield하도록 한다
        async def fake_query_loop(**kwargs):
            yield StreamEvent(type=StreamEventType.MESSAGE_START)
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="hello")
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)

        messages = [Message.user("테스트")]

        with patch(
            "core.orchestrator.model_dispatcher.query_loop",
            side_effect=fake_query_loop,
        ):
            events = []
            async for event in dispatcher.route(messages, "system prompt"):
                events.append(event)

        assert len(events) == 3
        assert events[0].type == StreamEventType.MESSAGE_START
        assert events[1].type == StreamEventType.TEXT_DELTA
        assert events[1].text == "hello"
        assert events[2].type == StreamEventType.MESSAGE_STOP

    async def test_route_passes_correct_args_to_query_loop(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """route()가 query_loop에 올바른 인자를 전달하는지 확인한다."""
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_M,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
            max_turns=50,
        )

        captured_kwargs = {}

        async def capture_query_loop(**kwargs):
            captured_kwargs.update(kwargs)
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)

        messages = [Message.user("test")]
        callback = MagicMock()

        with patch(
            "core.orchestrator.model_dispatcher.query_loop",
            side_effect=capture_query_loop,
        ):
            async for _ in dispatcher.route(messages, "sys prompt", on_turn_complete=callback):
                pass

        # query_loop에 전달된 인자를 검증한다
        assert captured_kwargs["messages"] is messages
        assert captured_kwargs["system_prompt"] == "sys prompt"
        assert captured_kwargs["model_provider"] is mock_worker_provider
        assert captured_kwargs["tools"] is mock_tools
        assert captured_kwargs["context"] is mock_context
        assert captured_kwargs["max_turns"] == 50
        assert captured_kwargs["on_turn_complete"] is callback

    async def test_route_tier_s_now_passthrough(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_scout_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """
        v7.0 Phase 9 재설계 이후: TIER_S + scout_provider가 있어도
        route()는 Scout를 자동 호출하지 않고 Worker query_loop만 한 번 부른다.

        Scout는 이제 Worker가 AgentTool로 호출하는 서브에이전트이므로
        Dispatcher의 자동 전처리 경로는 제거됐다.
        """
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_S,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
            scout_provider=mock_scout_provider,
        )
        # scout_available 플래그는 True지만 자동 실행은 안 된다
        assert dispatcher.scout_enabled is True

        call_count = 0

        async def fake_query_loop(**kwargs):
            nonlocal call_count
            call_count += 1
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="worker-direct")
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)

        messages = [Message.user("이 프로젝트 분석해줘")]

        with patch(
            "core.orchestrator.model_dispatcher.query_loop",
            side_effect=fake_query_loop,
        ):
            events = []
            async for event in dispatcher.route(messages, "system prompt"):
                events.append(event)

        # query_loop은 정확히 1회만 호출 (Scout 경유 없음)
        assert call_count == 1
        text_events = [e for e in events if hasattr(e, "text") and e.text]
        assert any("worker-direct" in e.text for e in text_events)

    async def test_route_stats_tracking(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_scout_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """
        stats는 하위 호환 필드를 유지하되 값은 항상 0이다.

        Scout 호출 통계는 이제 AgentTool.get_stats()가 담당한다.
        """
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_S,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
            scout_provider=mock_scout_provider,
        )

        assert dispatcher.stats["scout_calls"] == 0
        assert dispatcher.stats["scout_fallbacks"] == 0
        assert dispatcher.stats["scout_enabled"] is True
        assert dispatcher.stats["tier"] == "small"
        # 안내 노트로 AgentTool 참조를 유도한다
        assert "AgentTool" in dispatcher.stats.get("note", "")

    async def test_route_on_turn_complete_callback_forwarded(
        self,
        mock_worker_provider: EnhancedMockModelProvider,
        mock_tools: list[MagicMock],
        mock_context: ToolUseContext,
    ):
        """on_turn_complete 콜백이 query_loop에 올바르게 전달되는지 확인한다."""
        from core.orchestrator.model_dispatcher import ModelDispatcher

        dispatcher = ModelDispatcher(
            tier=HardwareTier.TIER_M,
            worker_provider=mock_worker_provider,
            worker_tools=mock_tools,
            context=mock_context,
        )

        received_callback = None

        async def capture_callback(**kwargs):
            nonlocal received_callback
            received_callback = kwargs.get("on_turn_complete")
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)

        callback_fn = MagicMock()
        messages = [Message.user("test")]

        with patch(
            "core.orchestrator.model_dispatcher.query_loop",
            side_effect=capture_callback,
        ):
            async for _ in dispatcher.route(messages, "prompt", on_turn_complete=callback_fn):
                pass

        assert received_callback is callback_fn
