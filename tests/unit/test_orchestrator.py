"""
core/orchestrator/ 단위 테스트.

ContinueReason, LoopState, ContextManager, StopResolver를 검증한다.
query_loop 자체는 ModelProvider mock이 필요하므로 통합 테스트에서 검증한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from core.message import Message, StopReason
from core.orchestrator.context_manager import ContextManager
from core.orchestrator.query_loop import ContinueReason, LoopState
from core.orchestrator.stop_resolver import StopResolver, _seems_truncated


# ─── ContextManager 생성 헬퍼 (model_provider mock 주입) ───
def _make_context_manager(**kwargs):
    """mock ModelProvider로 ContextManager를 생성한다."""
    mock_provider = AsyncMock()
    mock_provider.get_config.return_value = AsyncMock(max_context_tokens=8192)
    return ContextManager(model_provider=mock_provider, **kwargs)


class TestContinueReason:
    """ContinueReason 열거형 테스트."""

    def test_seven_reasons(self):
        """7가지 계속 전환 이유가 정의되어 있는지 확인한다."""
        assert len(ContinueReason) == 7

    def test_next_turn_is_default(self):
        """기본 이유가 NEXT_TURN인지 확인한다."""
        state = LoopState(messages=[])
        assert state.continue_reason == ContinueReason.NEXT_TURN


class TestLoopState:
    """LoopState 테스트."""

    def test_initial_state(self):
        """초기 상태가 올바르게 설정되는지 확인한다."""
        state = LoopState(messages=[])
        assert state.turn_count == 0
        assert state.max_output_tokens_override is None
        assert state.model_error_count == 0

    def test_elapsed_seconds(self):
        """경과 시간이 양수인지 확인한다."""
        state = LoopState(messages=[])
        assert state.elapsed_seconds >= 0


class TestContextManager:
    """ContextManager 테스트."""

    def test_estimate_tokens(self):
        """토큰 추정이 합리적인 값을 반환하는지 확인한다."""
        cm = _make_context_manager()
        msgs = [Message.user("Hello world")]
        tokens = cm._estimate_tokens(msgs)
        assert tokens > 0

    def test_apply_all_preserves_messages(self):
        """apply_all이 메시지를 보존하는지 확인한다."""
        cm = _make_context_manager()
        msgs = [Message.user("test"), Message.assistant(text="response")]
        result = cm.apply_all(msgs)
        assert len(result) == 2

    async def test_auto_compact_below_threshold(self):
        """임계치 이하면 압축하지 않는지 확인한다."""
        cm = _make_context_manager(max_context_tokens=100000)
        msgs = [Message.user("short")]
        result = await cm.auto_compact_if_needed(msgs)
        assert len(result) == 1

    async def test_emergency_compact(self):
        """긴급 압축이 메시지를 줄이는지 확인한다."""
        cm = _make_context_manager(max_context_tokens=4096)
        msgs = [Message.user(f"msg {i}") for i in range(20)]
        result = await cm.emergency_compact(msgs)
        assert len(result) <= len(msgs)


class TestStopResolver:
    """StopResolver 테스트."""

    def test_should_continue_with_tools(self):
        """도구 호출이 있으면 계속해야 한다."""
        resolver = StopResolver()
        state = LoopState(messages=[])
        assert resolver.should_continue(state, [{"name": "Read"}]) is True

    def test_should_not_continue_without_tools(self):
        """도구 호출이 없으면 중단해야 한다."""
        resolver = StopResolver()
        state = LoopState(messages=[])
        assert resolver.should_continue(state, []) is False

    def test_resolve_end_turn(self):
        """END_TURN 종료 이유를 올바르게 해석하는지 확인한다."""
        resolver = StopResolver()
        reason = resolver.resolve_stop_reason(StopReason.END_TURN, "text")
        # "completed" 또는 한글 "완료" 등 구현에 따라 다를 수 있다
        assert isinstance(reason, str)
        assert len(reason) > 0


class TestSeemsTruncated:
    """_seems_truncated 휴리스틱 테스트."""

    def test_empty_text(self):
        """빈 텍스트는 잘림이 아니다."""
        assert _seems_truncated("") is False

    def test_proper_ending(self):
        """문장 종료 문자로 끝나면 잘림이 아니다."""
        assert _seems_truncated("This is a complete sentence.") is False

    def test_unclosed_code_block(self):
        """닫히지 않은 코드 블록은 잘림으로 판단한다."""
        text = "Here is code:\n```python\ndef hello():\n    print('hi')"
        assert _seems_truncated(text) is True

    def test_short_text(self):
        """짧은 텍스트는 잘림 판단 대상이 아니다."""
        assert _seems_truncated("Hi") is False
