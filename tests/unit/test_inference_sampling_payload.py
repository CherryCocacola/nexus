"""
LocalModelProvider.stream() vLLM payload 샘플링 파라미터 단위 테스트.

배경 (degeneration 버그 수정, 2026-06-18):
  vLLM 요청 payload에 top_p/repetition_penalty/frequency_penalty/presence_penalty
  4개 파라미터가 누락되어 동일 문장이 무한 반복(degeneration)되는 결함이 있었다.
  이 4개를 temperature가 흐르던 경로를 그대로 미러링해 추가했고, payload 최상위
  (top-level)에 넣는다(httpx raw POST이므로 extra_body가 아님).

이 테스트는 httpx 클라이언트의 stream()을 mock하여 실제 전송 직전의 `json=payload`를
캡처하고, 4개 파라미터가 올바른 위치/값으로 들어가는지 검증한다. 실제 vLLM/GPU
서버는 호출하지 않는다(단위 테스트).
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from core.message import Message
from core.model.inference import LocalModelProvider


# ─────────────────────────────────────────────
# 헬퍼: httpx AsyncClient.stream()을 흉내내는 가짜 스트림 컨텍스트
# ─────────────────────────────────────────────
class _FakeStreamResponse:
    """vLLM SSE 응답을 최소한으로 흉내내는 가짜 응답 객체.

    status_code=200 + 짧은 SSE 라인(텍스트 1개 + finish + [DONE])만 흘려보내
    stream()이 정상 경로로 끝까지 진행하도록 한다.
    """

    status_code = 200

    async def aiter_lines(self):
        # vLLM이 보내는 SSE 형식을 최소 재현: 텍스트 델타 → 종료 → [DONE]
        yield 'data: {"choices":[{"delta":{"content":"안녕"},"finish_reason":null}]}'
        yield (
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":1}}'
        )
        yield "data: [DONE]"

    async def aiter_bytes(self):
        # 200 경로에서는 호출되지 않지만 인터페이스 호환을 위해 둔다
        if False:
            yield b""


class _FakeStreamCtx:
    """`async with client.stream(...) as response:` 형태를 지원하는 컨텍스트 매니저.

    생성 시 전달된 kwargs(특히 json=payload)를 capture 딕셔너리에 기록하여
    테스트가 전송 직전의 payload를 직접 검사할 수 있게 한다.
    """

    def __init__(self, capture: dict[str, Any], **kwargs: Any) -> None:
        # 호출 시점의 모든 kwargs를 기록 — 테스트는 capture["json"]을 검증한다
        capture.clear()
        capture.update(kwargs)
        self._response = _FakeStreamResponse()

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def _make_provider() -> LocalModelProvider:
    """LAN 주소로 LocalModelProvider를 생성한다(에어갭 준수, 실제 연결 없음)."""
    return LocalModelProvider(base_url="http://192.168.21.112:8000")


async def _capture_payload(provider: LocalModelProvider, **stream_kwargs: Any) -> dict[str, Any]:
    """stream()을 끝까지 소비하면서 vLLM으로 전송되는 payload를 캡처해 반환한다.

    provider._client.stream을 가짜 컨텍스트로 교체하여 실제 httpx POST를 차단한다.
    AsyncGenerator는 반드시 `async for`로 전부 소비해야 payload 구성 코드가 실행된다.
    """
    captured: dict[str, Any] = {}

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamCtx:
        return _FakeStreamCtx(captured, **kwargs)

    messages = [Message.user("테스트 질의")]
    with patch.object(provider._client, "stream", side_effect=fake_stream):
        async for _event in provider.stream(
            messages=messages,
            system_prompt="너는 도우미다.",
            **stream_kwargs,
        ):
            # 모든 StreamEvent를 소비해야 payload 구성/전송 코드가 끝까지 실행된다
            pass

    assert "json" in captured, "stream()이 json=payload로 POST하지 않았다"
    return captured["json"]


# ─────────────────────────────────────────────
# 1) KNOWLEDGE 프로필 값이 payload 최상위에 반영된다
# ─────────────────────────────────────────────
async def test_stream_payload_knowledge_values_included_at_top_level() -> None:
    """KNOWLEDGE 샘플링 값(top_p=0.95, repetition_penalty=1.15 등)이 payload에 실린다."""
    provider = _make_provider()
    payload = await _capture_payload(
        provider,
        top_p=0.95,
        repetition_penalty=1.15,
        frequency_penalty=0.3,
        presence_penalty=0.0,
    )

    assert payload["top_p"] == pytest.approx(0.95)
    assert payload["repetition_penalty"] == pytest.approx(1.15)
    assert payload["frequency_penalty"] == pytest.approx(0.3)
    assert payload["presence_penalty"] == pytest.approx(0.0)


# ─────────────────────────────────────────────
# 2) vLLM 회귀 가드 — repetition_penalty는 top-level, extra_body 사용 금지
# ─────────────────────────────────────────────
async def test_stream_payload_no_extra_body_repetition_penalty_top_level() -> None:
    """repetition_penalty가 extra_body가 아니라 payload 최상위에 있어야 한다.

    이 코드는 openai SDK가 아니라 httpx raw POST이므로 extra_body로 감싸면
    vLLM이 repetition_penalty를 인식하지 못해 degeneration 억제가 무력화된다.
    """
    provider = _make_provider()
    payload = await _capture_payload(
        provider,
        top_p=0.95,
        repetition_penalty=1.15,
    )

    # extra_body로 감싸면 안 된다 (vLLM 인식 실패 → 버그 재발)
    assert "extra_body" not in payload
    # 네 파라미터 모두 최상위 키로 존재해야 한다
    for key in ("top_p", "repetition_penalty", "frequency_penalty", "presence_penalty"):
        assert key in payload, f"{key}가 payload 최상위에 없다"


# ─────────────────────────────────────────────
# 3) 하위 호환 — 신규 인자 없이 호출 시 비활성 기본값이 실린다
# ─────────────────────────────────────────────
async def test_stream_payload_defaults_are_inactive_when_args_omitted() -> None:
    """stream()을 신규 샘플링 인자 없이 호출하면 비활성 기본값이 payload에 들어간다.

    top_p=1.0 / repetition_penalty=1.0 / frequency=0.0 / presence=0.0은 모두
    vLLM이 사실상 무시하는 중립값이라 기존 동작이 바뀌지 않는다(하위 호환).
    """
    provider = _make_provider()
    payload = await _capture_payload(provider)  # 4개 인자 모두 생략

    assert payload["top_p"] == pytest.approx(1.0)
    assert payload["repetition_penalty"] == pytest.approx(1.0)
    assert payload["frequency_penalty"] == pytest.approx(0.0)
    assert payload["presence_penalty"] == pytest.approx(0.0)


# ─────────────────────────────────────────────
# 4) CHAT 프로필 값도 그대로 전달된다
# ─────────────────────────────────────────────
async def test_stream_payload_chat_values_passthrough() -> None:
    """CHAT 샘플링 값(top_p=0.9, repetition_penalty=1.1, freq=0.2)이 그대로 실린다."""
    provider = _make_provider()
    payload = await _capture_payload(
        provider,
        top_p=0.9,
        repetition_penalty=1.1,
        frequency_penalty=0.2,
        presence_penalty=0.0,
    )

    assert payload["top_p"] == pytest.approx(0.9)
    assert payload["repetition_penalty"] == pytest.approx(1.1)
    assert payload["frequency_penalty"] == pytest.approx(0.2)
    assert payload["presence_penalty"] == pytest.approx(0.0)
