"""
core/message.py 단위 테스트.

Message, StreamEvent, ContentBlock, Conversation의 생성, 변환, 유틸리티를 검증한다.
"""

from __future__ import annotations

from core.message import (
    Conversation,
    Message,
    Role,
    StopReason,
    StreamEvent,
    StreamEventType,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
)


class TestContentBlocks:
    """콘텐츠 블록 테스트."""

    def test_text_block(self):
        """TextBlock이 올바르게 생성되는지 확인한다."""
        block = TextBlock(text="Hello")
        assert block.type == "text"
        assert block.text == "Hello"

    def test_tool_use_block_auto_id(self):
        """ToolUseBlock의 id가 자동 생성되는지 확인한다."""
        block = ToolUseBlock(name="Read", input={"path": "/tmp/a.txt"})
        assert block.type == "tool_use"
        assert block.id.startswith("toolu_")
        assert block.name == "Read"

    def test_tool_result_block(self):
        """ToolResultBlock이 올바르게 생성되는지 확인한다."""
        block = ToolResultBlock(
            tool_use_id="toolu_abc", content="file content", is_error=False
        )
        assert block.type == "tool_result"
        assert block.tool_use_id == "toolu_abc"

    def test_thinking_block(self):
        """ThinkingBlock이 올바르게 생성되는지 확인한다."""
        block = ThinkingBlock(thinking="Let me analyze this...")
        assert block.type == "thinking"


class TestTokenUsage:
    """토큰 사용량 테스트."""

    def test_total_tokens(self):
        """total_tokens가 input + output인지 확인한다."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_add_usage(self):
        """두 사용량을 합산할 수 있는지 확인한다."""
        u1 = TokenUsage(input_tokens=100, output_tokens=50)
        u2 = TokenUsage(input_tokens=200, output_tokens=100)
        result = u1 + u2
        assert result.input_tokens == 300
        assert result.output_tokens == 150


class TestStreamEvent:
    """스트리밍 이벤트 테스트."""

    def test_text_delta_event(self):
        """TEXT_DELTA 이벤트가 올바르게 생성되는지 확인한다."""
        event = StreamEvent(type=StreamEventType.TEXT_DELTA, text="Hello")
        assert event.type == StreamEventType.TEXT_DELTA
        assert event.text == "Hello"

    def test_message_stop_event(self):
        """MESSAGE_STOP 이벤트에 stop_reason과 usage가 포함되는지 확인한다."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        event = StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            stop_reason=StopReason.END_TURN,
            usage=usage,
        )
        assert event.stop_reason == StopReason.END_TURN
        assert event.usage.total_tokens == 150

    def test_error_event(self):
        """ERROR 이벤트가 올바르게 생성되는지 확인한다."""
        event = StreamEvent(
            type=StreamEventType.ERROR,
            error_code="CONNECT_ERROR",
            message="연결 실패",
        )
        assert event.error_code == "CONNECT_ERROR"

    def test_timestamp_auto_set(self):
        """timestamp가 자동으로 설정되는지 확인한다."""
        event = StreamEvent(type=StreamEventType.MESSAGE_START)
        assert event.timestamp is not None


class TestMessage:
    """메시지 테스트."""

    def test_user_factory(self):
        """Message.user() 팩토리가 올바르게 동작하는지 확인한다."""
        msg = Message.user("안녕하세요")
        assert msg.role == Role.USER
        assert msg.content == "안녕하세요"
        assert msg.text_content == "안녕하세요"

    def test_assistant_factory_text_only(self):
        """텍스트만 있는 assistant 메시지가 올바르게 생성되는지 확인한다."""
        msg = Message.assistant(text="답변입니다")
        assert msg.role == Role.ASSISTANT
        assert isinstance(msg.content, list)
        assert msg.text_content == "답변입니다"
        assert not msg.has_tool_use

    def test_assistant_factory_with_tool_use(self):
        """도구 호출이 포함된 assistant 메시지를 확인한다."""
        msg = Message.assistant(
            text="파일을 읽겠습니다",
            tool_uses=[{"name": "Read", "input": {"path": "/tmp/a.txt"}}],
        )
        assert msg.has_tool_use
        assert len(msg.tool_use_blocks) == 1
        assert msg.tool_use_blocks[0].name == "Read"

    def test_assistant_factory_with_thinking(self):
        """사고 블록이 포함된 assistant 메시지를 확인한다."""
        msg = Message.assistant(
            text="답변입니다",
            thinking="먼저 분석하겠습니다",
        )
        blocks = msg.content
        assert isinstance(blocks, list)
        assert any(isinstance(b, ThinkingBlock) for b in blocks)

    def test_tool_result_factory(self):
        """Message.tool_result() 팩토리가 올바르게 동작하는지 확인한다."""
        msg = Message.tool_result(
            tool_use_id="toolu_abc",
            content="파일 내용",
            is_error=False,
        )
        assert msg.role == Role.TOOL_RESULT
        assert msg.tool_use_id == "toolu_abc"
        assert msg.is_error is False

    def test_system_factory(self):
        """Message.system() 팩토리가 올바르게 동작하는지 확인한다."""
        msg = Message.system("당신은 AI 어시스턴트입니다.")
        assert msg.role == Role.SYSTEM

    def test_estimated_tokens_korean(self):
        """한국어 텍스트의 토큰 추정이 합리적인지 확인한다."""
        msg = Message.user("안녕하세요 만나서 반갑습니다")
        tokens = msg.estimated_tokens()
        # 한국어 11글자 → ~22 토큰 + 알파
        assert tokens > 10

    def test_message_id_auto_generated(self):
        """메시지 id가 자동 생성되는지 확인한다."""
        msg = Message.user("test")
        assert msg.id.startswith("msg_")


class TestConversation:
    """대화 컨테이너 테스트."""

    def test_append_and_get(self):
        """메시지 추가 및 조회가 올바르게 동작하는지 확인한다."""
        conv = Conversation()
        conv.append(Message.user("질문"))
        conv.append(Message.assistant(text="답변"))
        assert len(conv.messages) == 2

    def test_turn_count(self):
        """턴 수가 user 메시지 수와 일치하는지 확인한다."""
        conv = Conversation()
        conv.append(Message.user("질문 1"))
        conv.append(Message.assistant(text="답변 1"))
        conv.append(Message.user("질문 2"))
        conv.append(Message.assistant(text="답변 2"))
        assert conv.turn_count == 2

    def test_compact_boundary(self):
        """compact_boundary가 활성 메시지 범위를 올바르게 제한하는지 확인한다."""
        conv = Conversation()
        for i in range(6):
            conv.append(Message.user(f"메시지 {i}"))

        conv.compact_boundary = 4
        active = conv.get_active_messages()
        assert len(active) == 2

    def test_get_last_n_turns(self):
        """최근 n개 턴 추출이 올바르게 동작하는지 확인한다."""
        conv = Conversation()
        conv.append(Message.user("질문 1"))
        conv.append(Message.assistant(text="답변 1"))
        conv.append(Message.user("질문 2"))
        conv.append(Message.assistant(text="답변 2"))

        last_1 = conv.get_last_n_turns(1)
        assert len(last_1) == 2  # user + assistant
        assert last_1[0].text_content == "질문 2"
