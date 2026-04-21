"""
ScoutModelProvider 단위 테스트 — α 해결 (2026-04-21).

검증 대상:
  1. ScoutModelProvider.stream()은 호출자가 enable_thinking에 뭘 넘기든
     HTTP 요청 페이로드에서 chat_template_kwargs가 **빠져** 있어야 한다
     (Qwen3.5-4B on llama.cpp에서 빈 <think></think> 주입 시 28토큰 조기
     종료 버그를 회피).
  2. 반대로 Worker용 LocalModelProvider는 enable_thinking=False를 기본으로
     받아 chat_template_kwargs={"enable_thinking": False}를 주입한다
     (Worker 27B의 답변 누락 회피 목적이 유지되어야 한다).

테스트 방식:
  httpx.AsyncClient를 모킹하여 실제 요청 payload를 가로채고,
  chat_template_kwargs 키 유무를 검사한다.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.message import Message
from core.model.inference import LocalModelProvider
from core.model.scout_provider import ScoutModelProvider


# ─────────────────────────────────────────────
# HTTP 스트림 응답을 흉내내는 헬퍼
# ─────────────────────────────────────────────
class _FakeSSEResponse:
    """httpx.AsyncClient.stream() 컨텍스트 매니저가 돌려주는 응답을 흉내낸다.

    핵심: aiter_lines()가 빈 리스트를 반환하도록 하여 stream()이 조기 종료되도록
    한다. 우리는 payload만 검증하고 싶고, 실제 SSE 파싱은 테스트 범위 밖이다.
    """

    def __init__(self) -> None:
        self.status_code = 200

    async def __aenter__(self) -> _FakeSSEResponse:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def aiter_lines(self):  # type: ignore[override]
        # SSE 파싱할 내용 없음 — 즉시 종료
        if False:
            yield ""  # 제너레이터 표시용, 실제로 yield 안 함
        return


async def _drain(async_gen) -> None:
    """비동기 제너레이터를 끝까지 소비한다 (payload가 실제로 전송되도록)."""
    async for _ in async_gen:
        pass


def _capture_payload(provider: LocalModelProvider) -> dict[str, Any]:
    """
    provider._client.stream(...)을 모킹하여 호출 시 payload를 잡아낸다.

    반환값은 stream 호출 시 사용된 json 페이로드.
    """
    captured: dict[str, Any] = {}

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeSSEResponse:
        # POST /v1/chat/completions 요청의 json= 파라미터를 캡쳐
        if "json" in kwargs:
            captured.update(kwargs["json"])
        return _FakeSSEResponse()

    provider._client.stream = MagicMock(side_effect=fake_stream)  # type: ignore[method-assign]
    return captured


# ─────────────────────────────────────────────
# 테스트 1: ScoutModelProvider는 chat_template_kwargs를 주입하지 않는다
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_scout_provider_omits_chat_template_kwargs_by_default() -> None:
    """
    Scout는 enable_thinking 기본값(False)으로 호출돼도 payload에
    chat_template_kwargs 키가 존재하지 않아야 한다.
    """
    provider = ScoutModelProvider(
        base_url="http://127.0.0.1:65530",  # 실제 요청은 안 나감
        api_key="test",
    )
    captured = _capture_payload(provider)

    await _drain(
        provider.stream(
            messages=[Message.user("hi")],
            system_prompt="test",
        )
    )

    assert "chat_template_kwargs" not in captured, (
        "Scout는 chat_template_kwargs를 보내서는 안 된다 — "
        f"하지만 보내졌다: {captured.get('chat_template_kwargs')}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [False, True, None])
async def test_scout_provider_ignores_enable_thinking_argument(passed: bool | None) -> None:
    """
    호출자가 True/False/None 중 무엇을 넘겨도 Scout는 payload에
    chat_template_kwargs를 싣지 않아야 한다 (Scout.stream 오버라이드가
    인자를 None으로 강제하기 때문).
    """
    provider = ScoutModelProvider(
        base_url="http://127.0.0.1:65530",
        api_key="test",
    )
    captured = _capture_payload(provider)

    await _drain(
        provider.stream(
            messages=[Message.user("hi")],
            system_prompt="test",
            enable_thinking=passed,
        )
    )

    assert "chat_template_kwargs" not in captured


# ─────────────────────────────────────────────
# 테스트 2: Worker용 LocalModelProvider는 여전히 enable_thinking=False를 주입
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_local_provider_injects_enable_thinking_false_by_default() -> None:
    """
    Worker(LocalModelProvider)는 기본 enable_thinking=False로
    chat_template_kwargs={"enable_thinking": False}를 주입한다 —
    Scout 오버라이드가 Worker에 역으로 영향을 주면 안 된다.
    """
    provider = LocalModelProvider(
        base_url="http://127.0.0.1:65530",
        api_key="test",
        model_id="qwen3.5-27b",
    )
    captured = _capture_payload(provider)

    await _drain(
        provider.stream(
            messages=[Message.user("hi")],
            system_prompt="test",
        )
    )

    assert captured.get("chat_template_kwargs") == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_local_provider_skips_kwargs_when_enable_thinking_is_none() -> None:
    """
    Worker에게도 명시적으로 enable_thinking=None을 넘기면 chat_template_kwargs를
    생략한다 (Scout 외 시나리오에서도 옵션으로 쓸 수 있도록).
    """
    provider = LocalModelProvider(
        base_url="http://127.0.0.1:65530",
        api_key="test",
        model_id="qwen3.5-27b",
    )
    captured = _capture_payload(provider)

    await _drain(
        provider.stream(
            messages=[Message.user("hi")],
            system_prompt="test",
            enable_thinking=None,
        )
    )

    assert "chat_template_kwargs" not in captured


@pytest.mark.asyncio
async def test_local_provider_enable_thinking_true_is_injected() -> None:
    """
    enable_thinking=True도 정상적으로 주입되어야 한다
    (KNOWLEDGE_MODE 검증 같은 향후 용도).
    """
    provider = LocalModelProvider(
        base_url="http://127.0.0.1:65530",
        api_key="test",
        model_id="qwen3.5-27b",
    )
    captured = _capture_payload(provider)

    await _drain(
        provider.stream(
            messages=[Message.user("hi")],
            system_prompt="test",
            enable_thinking=True,
        )
    )

    assert captured.get("chat_template_kwargs") == {"enable_thinking": True}
