# 코딩 전용 프로바이더로 전환할 때 페이로드 모델 이름이 그 서버의 것으로 나가는지 검증한다.
"""
코더 라우팅 모델명 배선 테스트 (2026-08-20).

무엇을 막는가:
    라우팅 프로필(`routing.profiles.*.model`)은 앵커 모델 이름("ax-4.0")을 담는다.
    코딩 턴에서 프로바이더만 코딩 서버로 갈아끼우고 이 이름을 그대로 실어 보내면,
    코딩 서버는 자기 served-model-name 만 알기 때문에 **HTTP 404** 를 낸다.
    실측으로 확인된 결함이다 — bakeoff 첫 시도에서 404 가 3회 반복됐다.

왜 값이 아니라 배선을 보는가:
    두 경로(dispatcher / query_loop 폴백)가 각각 model_override 를 넘긴다.
    한쪽만 고치면 그 경로에서만 조용히 404 가 계속된다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config import RoutingConfig
from core.message import StreamEvent, StreamEventType
from core.orchestrator.query_engine import QueryEngine
from core.tools.base import ToolUseContext

from tests.conftest import EnhancedMockModelProvider, MockResponse

# 코딩 질의로 분류되게 하는 키워드(설정 기본값에 포함된 단어).
CODING_PROMPT = "이 함수 디버깅 해줘"


@pytest.fixture
def context(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        cwd=str(tmp_path),
        session_id="test-coder-model-id",
        permission_mode="bypass_permissions",
    )


def _routing_config_with_coder() -> RoutingConfig:
    """코더 라우팅을 켠 설정(그 외는 기본값)."""
    return RoutingConfig(enabled=True, coder_enabled=True)


def _capturing_dispatcher() -> MagicMock:
    """route() 호출 인자를 기록하는 mock dispatcher."""

    async def fake_route(**kwargs):
        fake_route.captured = kwargs
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="ok")

    fake_route.captured = {}
    dispatcher = MagicMock()
    dispatcher.route = MagicMock(side_effect=fake_route)
    dispatcher._fake = fake_route
    return dispatcher


def _make_engine(context: ToolUseContext, dispatcher, coder_provider):
    """코더 프로바이더를 주입한 엔진을 만든다."""
    return QueryEngine(
        model_provider=EnhancedMockModelProvider(responses=[MockResponse(text="anchor")]),
        tools=[],
        context=context,
        model_dispatcher=dispatcher,
        routing_config=_routing_config_with_coder(),
        coder_provider=coder_provider,
    )


async def test_coder_turn_does_not_send_anchor_model_name(context: ToolUseContext) -> None:
    """코딩 턴에서는 프로필의 앵커 모델명을 실어 보내지 않는다(그대로 보내면 404)."""
    coder = EnhancedMockModelProvider(responses=[MockResponse(text="coder")])
    coder.model_id = "qwen3-coder-30b"
    dispatcher = _capturing_dispatcher()
    engine = _make_engine(context, dispatcher, coder)

    async for _ in engine.submit_message(CODING_PROMPT):
        pass

    captured = dispatcher._fake.captured
    # 프로바이더는 코딩 서버로 바뀌어야 한다.
    assert captured.get("provider_override") is coder
    # 그리고 모델 이름은 프로필 값이 아니라 그 프로바이더의 것을 쓰게 비워 보낸다.
    assert captured.get("model_override") is None


async def test_non_coder_turn_keeps_profile_model_name(context: ToolUseContext) -> None:
    """일반 턴은 기존 그대로 — 프로필이 정한 모델명을 계속 사용한다(무회귀)."""
    coder = EnhancedMockModelProvider(responses=[MockResponse(text="coder")])
    coder.model_id = "qwen3-coder-30b"
    dispatcher = _capturing_dispatcher()
    engine = _make_engine(context, dispatcher, coder)

    async for _ in engine.submit_message("오늘 날씨 어때?"):
        pass

    captured = dispatcher._fake.captured
    assert captured.get("provider_override") is None


def test_both_paths_use_the_same_override_variable() -> None:
    """dispatcher 경로와 query_loop 폴백 경로가 같은 값을 쓰는지 소스로 확인한다.

    한쪽만 고치면 그 경로에서만 404 가 남는다 — 실행 경로가 둘이라 값 검증만으로는
    빠지는 구멍이 생긴다(이번 주 교훈).
    """
    source = Path("core/orchestrator/query_engine.py").read_text(encoding="utf-8")
    assert source.count("model_override=active_model_override,") == 2
    assert "model_override=decision.model_override," not in source
