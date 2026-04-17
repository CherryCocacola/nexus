"""
core/model/prompt_formatter.py 단위 테스트.

Qwen, ExaOne, ChatML 포맷이 올바르게 생성되는지 검증한다.
"""

from __future__ import annotations

from core.model.prompt_formatter import format_chat_prompt


class TestQwenFormat:
    """Qwen ChatML 포맷 테스트."""

    def test_basic_prompt(self):
        """기본 Qwen ChatML 프롬프트 형식을 확인한다."""
        messages = [{"role": "user", "content": "안녕하세요"}]
        result = format_chat_prompt(
            messages=messages,
            system_prompt="You are a helpful assistant.",
            tools=[],
            model_name="qwen3.5-27b",
        )
        assert "<|im_start|>system" in result or "<|im_start|>user" in result
        assert "<|im_end|>" in result

    def test_tool_result_format(self):
        """tool_result 메시지가 올바르게 포맷되는지 확인한다."""
        messages = [
            {"role": "user", "content": "파일 읽어줘"},
            {"role": "tool_result", "tool_use_id": "abc", "content": "파일 내용"},
        ]
        result = format_chat_prompt(
            messages=messages,
            system_prompt="",
            tools=[],
            model_name="qwen3.5-27b",
        )
        # ChatML 형식에서 tool_result는 tool 또는 assistant 역할로 포함됨
        assert "파일 내용" in result or "abc" in result

    def test_with_tools(self):
        """도구 스키마가 시스템 프롬프트에 포함되는지 확인한다."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "파일 읽기",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = format_chat_prompt(
            messages=[{"role": "user", "content": "test"}],
            system_prompt="System",
            tools=tools,
            model_name="qwen3.5-27b",
        )
        assert '<tool name="Read">' in result
        assert "tool_use" in result


class TestExaOneFormat:
    """ExaOne 포맷 테스트."""

    def test_basic_prompt(self):
        """기본 ExaOne 프롬프트 형식을 확인한다."""
        messages = [{"role": "user", "content": "안녕하세요"}]
        result = format_chat_prompt(
            messages=messages,
            system_prompt="시스템 프롬프트",
            tools=[],
            model_name="exaone-7.8b",
        )
        assert "[|system|]" in result
        assert "[|user|]" in result
        assert "[|assistant|]" in result
        assert "[|endofturn|]" in result


class TestChatMLFormat:
    """ChatML 범용 포맷 테스트."""

    def test_basic_prompt(self):
        """기본 ChatML 형식을 확인한다."""
        messages = [{"role": "user", "content": "Hello"}]
        result = format_chat_prompt(
            messages=messages,
            system_prompt="System prompt",
            tools=[],
            model_name="unknown-model",
        )
        assert "<|im_start|>system" in result
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "<|im_end|>" in result


class TestModelRouting:
    """모델 이름 기반 라우팅 테스트."""

    def test_qwen_routing(self):
        """qwen이 포함된 이름이 ChatML 포맷으로 라우팅되는지 확인한다."""
        result = format_chat_prompt([], "", [], "Qwen3.5-27B")
        assert "<|im_start|>" in result

    def test_exaone_routing(self):
        """exaone이 포함된 이름이 ExaOne 포맷으로 라우팅되는지 확인한다."""
        result = format_chat_prompt([], "", [], "ExaOne-3.5-7.8B")
        assert "[|system|]" in result

    def test_fallback_routing(self):
        """알 수 없는 모델이 ChatML 폴백으로 라우팅되는지 확인한다."""
        result = format_chat_prompt([], "", [], "llama-3-70b")
        assert "<|im_start|>" in result
