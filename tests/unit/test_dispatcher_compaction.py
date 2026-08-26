# dispatcher 경로에서 컨텍스트 압축이 실제로 호출되는지 동작으로 검증한다.
"""
`ModelDispatcher.route` → `query_loop` 압축 배선 테스트.

왜 배선 확인만으로 부족한가:
    인자를 넘기도록 고쳐 놓고도 query_loop 안에서 쓰이지 않으면 아무 일도 안 난다.
    그래서 여기서는 시그니처가 아니라 **압축 관리자가 실제로 불렸는지**를 본다.

시나리오:
    컨텍스트 초과를 흉내 내는 프로바이더를 주고, query_loop 이 복구를 시도할 때
    context_manager 의 압축 메서드가 호출되는지 확인한다.
"""

from __future__ import annotations

import pytest

from core.message import StreamEvent, StreamEventType


class _RecordingContextManager:
    """압축 요청을 기록만 하는 가짜 관리자(실제로 줄이지는 않는다)."""

    def __init__(self) -> None:
        self.emergency_calls = 0
        self.auto_calls = 0
        self.max_tokens = 10000
        self.adopted_calls = 0

    def mark_result_adopted(self) -> None:
        """실물과 시그니처를 맞춘다 — 없으면 복구 경로가 AttributeError 로 죽는다."""
        self.adopted_calls += 1

    async def emergency_compact(self, messages):
        self.emergency_calls += 1
        return messages

    async def auto_compact_if_needed(self, messages, *args, **kwargs):
        self.auto_calls += 1
        return messages

    def apply_all(self, messages):
        return messages

    def take_last_compaction(self):
        return None


@pytest.mark.asyncio
async def test_route_forwards_context_manager_into_query_loop(monkeypatch) -> None:
    """route() 에 넘긴 context_manager 가 query_loop 인자로 도달하는지 본다."""
    from core.orchestrator import model_dispatcher

    seen: dict = {}

    async def _fake_query_loop(**kwargs):
        seen.update(kwargs)
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="ok")

    monkeypatch.setattr(model_dispatcher, "query_loop", _fake_query_loop)

    cm = _RecordingContextManager()
    d = model_dispatcher.ModelDispatcher(
        tier=_AnyTier(),
        worker_provider=object(),
        worker_tools=[],
        context=object(),
    )
    async for _ in d.route(messages=[], system_prompt="s", context_manager=cm):
        pass

    assert "context_manager" in seen, "query_loop 이 인자를 아예 못 받았다"
    assert seen["context_manager"] is cm, "다른 객체가 전달됐다"


@pytest.mark.asyncio
async def test_route_without_context_manager_is_still_fine() -> None:
    """미주입 호출(기존 코드·테스트)은 그대로 동작해야 한다(무회귀)."""
    import inspect

    from core.orchestrator import model_dispatcher

    sig = inspect.signature(model_dispatcher.ModelDispatcher.route)
    assert sig.parameters["context_manager"].default is None


class _AnyTier:
    """ModelDispatcher 가 로그에 tier.value 만 쓰므로 최소 스텁이면 충분하다."""

    value = "large"


@pytest.mark.asyncio
async def test_engine_to_dispatcher_reaches_compaction(monkeypatch) -> None:
    """★전 구간 검증 — QueryEngine 에 준 압축 관리자가 dispatcher 를 거쳐 도달하는가.

    지금까지의 사고는 전부 "한쪽 경로만 고쳐서" 났다. 그래서 조각이 아니라
    엔진에서 출발해 query_loop 인자에 닿는 것까지 한 번에 본다.
    """
    from core.orchestrator import model_dispatcher
    from core.orchestrator import query_engine as qe_mod

    seen: dict = {}

    async def _fake_query_loop(**kwargs):
        seen.update(kwargs)
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="ok")

    monkeypatch.setattr(model_dispatcher, "query_loop", _fake_query_loop)

    cm = _RecordingContextManager()
    dispatcher = model_dispatcher.ModelDispatcher(
        tier=_AnyTier(),
        worker_provider=object(),
        worker_tools=[],
        context=_StubContext(),
    )
    engine = qe_mod.QueryEngine(
        model_provider=object(),
        tools=[],
        context=_StubContext(),
        model_dispatcher=dispatcher,
        context_manager=cm,
    )

    async for _ in engine.submit_message("안녕"):
        pass

    assert seen.get("context_manager") is cm, (
        "엔진이 준 압축 관리자가 dispatcher 를 거쳐 query_loop 까지 도달하지 못했다"
    )


class _StubContext:
    """ToolUseContext 대용 — 엔진/디스패처가 읽는 최소 속성만 갖춘다."""

    session_id = "t"
    cwd = "."
    options: dict = {}
