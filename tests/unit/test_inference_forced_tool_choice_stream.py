# 강제 tool_choice 스트리밍에서 finish_reason="stop"이어도 tool_calls를 finalize하는지 검증
"""
강제 tool_choice(named) 스트리밍 tool_calls finalize 회귀 테스트 (2026-07-14).

배경 (DocumentExport guided 재시도가 END_TURN으로 새던 버그):
  vLLM은 tool_choice를 특정 함수로 '강제'하면, tool_calls를 정상 스트리밍하면서도
  finish_reason을 "tool_calls"가 아니라 "stop"으로 반환한다(B200 실측). 기존
  stream() 파서는 finish_reason == "tool_calls"일 때만 _finalize_tool_calls를
  호출했고 [DONE]에서도 pending_stop이 있으면 건너뛰어, 강제 호출의 tool_calls가
  유실되고 응답이 텍스트(END_TURN)로 처리됐다.

수정: finish_reason 문자열이 아니라 '실제 누적된 tool_calls 유무'로 finalize하고,
  tool_calls가 있으면 종료 이유를 TOOL_USE로 바로잡는다.

이 테스트는 httpx 스트림을 가짜로 대체해, finish_reason="stop"과 함께 오는
  tool_calls가 (1) TOOL_USE_STOP으로 finalize되고 (2) MESSAGE_STOP이 TOOL_USE로
  종료되는지 검증한다. 실제 vLLM/GPU 서버는 호출하지 않는다.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

from core.message import StopReason, StreamEventType
from core.model.inference import LocalModelProvider


class _ToolCallStopResponse:
    """finish_reason="stop"과 함께 tool_calls를 스트리밍하는 vLLM 응답 재현(강제 tool_choice)."""

    status_code = 200

    async def aiter_lines(self):
        # tool_call 시작(이름) → 인자 조각들 → finish_reason="stop" → [DONE]
        yield (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            '"function":{"name":"DocumentExport","arguments":""}}]},'
            '"finish_reason":null}]}'
        )
        yield (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"{\\"content\\": \\"본문\\", "}}]},'
            '"finish_reason":null}]}'
        )
        yield (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"\\"format\\": \\"md\\"}"}}]},'
            '"finish_reason":null}]}'
        )
        # 핵심: 강제 tool_choice는 finish_reason="stop"으로 종료한다(‼️ "tool_calls" 아님)
        yield (
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":5}}'
        )
        yield "data: [DONE]"

    async def aiter_bytes(self):
        if False:
            yield b""


class _FakeCtx:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc: Any) -> bool:
        return False


async def _collect_events(response: Any):
    """가짜 응답으로 stream()을 끝까지 소비하고 이벤트 리스트를 반환한다."""
    from core.message import Message

    provider = LocalModelProvider(base_url="http://192.168.21.112:8000")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeCtx:
        return _FakeCtx(response)

    events = []
    with patch.object(provider._client, "stream", side_effect=fake_stream):
        async for ev in provider.stream(
            messages=[Message.user("보고서를 docx로")],
            system_prompt="너는 도우미다.",
            tools=None,
            force_tool_choice="DocumentExport",
        ):
            events.append(ev)
    return events


class _TruncatedToolCallLengthResponse:
    """tool_call arguments가 max_tokens로 절단(finish_reason="length")된 응답 재현."""

    status_code = 200

    async def aiter_lines(self):
        yield (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            '"function":{"name":"DocumentExport","arguments":""}}]},'
            '"finish_reason":null}]}'
        )
        # 문자열이 닫히지 않고 중간에서 끊김(절단) — strict=False로도 복구 불가
        yield (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"{\\"content\\": \\"긴 본문이 중간에서"}}]},'
            '"finish_reason":null}]}'
        )
        yield (
            'data: {"choices":[{"delta":{},"finish_reason":"length"}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":8192}}'
        )
        yield "data: [DONE]"

    async def aiter_bytes(self):
        if False:
            yield b""


async def test_truncated_tool_call_length_preserves_max_tokens_and_flags_parse_error():
    """finish_reason="length"로 tool_call이 절단되면 parse_error=True로 finalize하되
    종료 이유는 MAX_TOKENS를 보존한다(기존 max-output 복구 경로 유지)."""
    events = await _collect_events(_TruncatedToolCallLengthResponse())

    stops = [e for e in events if e.type == StreamEventType.TOOL_USE_STOP]
    assert stops, "절단돼도 누적 tool_call은 finalize돼야 한다(상위가 감지하도록)"
    tu = stops[0].tool_use
    assert tu.name == "DocumentExport"
    # 절단 = 파싱 불가 → 빈 인자 + parse_error 신호(상위 guided 재생성 트리거)
    assert tu.input == {}
    assert tu.parse_error is True

    # length 절단은 MAX_TOKENS를 보존해야 한다(TOOL_USE로 덮어쓰지 않음)
    msg_stops = [e for e in events if e.type == StreamEventType.MESSAGE_STOP]
    assert msg_stops and msg_stops[-1].stop_reason == StopReason.MAX_TOKENS


async def test_forced_tool_choice_stop_finish_still_finalizes_tool_call():
    """finish_reason="stop"이어도 누적 tool_calls가 있으면 finalize되고 TOOL_USE로 종료된다."""
    events = await _collect_events(_ToolCallStopResponse())

    # TOOL_USE_STOP이 방출되고 인자가 온전히 파싱돼야 한다(유실 금지)
    stops = [e for e in events if e.type == StreamEventType.TOOL_USE_STOP]
    assert stops, "finish_reason='stop'이어도 tool_calls를 finalize해야 한다"
    tu = stops[0].tool_use
    assert tu.name == "DocumentExport"
    assert tu.input == {"content": "본문", "format": "md"}
    assert tu.parse_error is False

    # MESSAGE_STOP의 종료 이유는 END_TURN이 아니라 TOOL_USE로 바로잡혀야 한다
    msg_stops = [e for e in events if e.type == StreamEventType.MESSAGE_STOP]
    assert msg_stops, "MESSAGE_STOP이 있어야 한다"
    assert msg_stops[-1].stop_reason == StopReason.TOOL_USE
