# QueryEngine의 지식 RAG 출처 인용(KNOWLEDGE_SOURCES 이벤트) 배선 단위 테스트 (Point 4-2)
"""
QueryEngine.submit_message가 프롬프트 조립 직후, KB에 실제 주입된 청크의 출처가
있으면 KNOWLEDGE_SOURCES StreamEvent 1건을 먼저 yield하는지 검증한다.

검증 전략(격리):
  - 실제 라우팅/분류/모델/DB 미사용. PromptAssembler를 가짜로 갈아끼워
    last_knowledge_citations만 제어하고, dispatcher는 텍스트 이벤트 하나를
    흘리는 mock으로 둔다. 이로써 "citations가 있으면 이벤트를 낸다"는 배선만
    격리 검증한다(4-Tier 체인 준수 — 웹 직참조 대신 이벤트 전달).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.message import KnowledgeCitation, StreamEvent, StreamEventType
from core.orchestrator.query_engine import QueryEngine
from core.tools.base import ToolUseContext
from tests.conftest import EnhancedMockModelProvider, MockResponse


@pytest.fixture
def provider() -> EnhancedMockModelProvider:
    return EnhancedMockModelProvider(responses=[MockResponse(text="ok")])


@pytest.fixture
def context(tmp_path: Path) -> ToolUseContext:
    return ToolUseContext(
        cwd=str(tmp_path),
        session_id="sess-citation",
        permission_mode="bypass_permissions",
    )


class _FakeAssembler:
    """assemble()가 고정 프롬프트를 돌려주고 last_knowledge_citations를 노출하는 가짜."""

    def __init__(self, citations: tuple) -> None:
        self.last_knowledge_citations = citations

    async def assemble(self, **kwargs) -> str:
        # 실제 조립기처럼 last_knowledge_citations는 이미 세팅돼 있다(여기선 고정).
        return "SYS"


def _make_fake_dispatcher() -> MagicMock:
    """route()가 텍스트 이벤트 하나만 yield하는 mock dispatcher."""

    async def fake_route(**kwargs):
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="answer")

    dispatcher = MagicMock()
    dispatcher.route = MagicMock(side_effect=fake_route)
    return dispatcher


@pytest.mark.asyncio
async def test_submit_message_yields_knowledge_sources_event_when_citations_exist(
    provider: EnhancedMockModelProvider,
    context: ToolUseContext,
) -> None:
    """citations가 있으면 KNOWLEDGE_SOURCES 이벤트가 텍스트 이벤트보다 먼저 나온다."""
    citations = (
        KnowledgeCitation(index=1, source="kowiki", title="바흐", section="생애", score=0.9),
        KnowledgeCitation(index=2, source="kowiki", title="헨델", score=0.85),
    )
    engine = QueryEngine(
        model_provider=provider,
        tools=[],
        context=context,
        model_dispatcher=_make_fake_dispatcher(),
    )
    engine._prompt_assembler = _FakeAssembler(citations)

    events = [ev async for ev in engine.submit_message("바흐 알려줘")]

    ks_events = [
        e for e in events
        if isinstance(e, StreamEvent) and e.type == StreamEventType.KNOWLEDGE_SOURCES
    ]
    assert len(ks_events) == 1
    assert ks_events[0].knowledge_sources == list(citations)
    # KNOWLEDGE_SOURCES가 텍스트 이벤트보다 먼저 방출된다(조립 직후 1회).
    ks_pos = events.index(ks_events[0])
    text_positions = [
        i for i, e in enumerate(events)
        if isinstance(e, StreamEvent) and e.type == StreamEventType.TEXT_DELTA
    ]
    assert text_positions and ks_pos < text_positions[0]


@pytest.mark.asyncio
async def test_submit_message_no_event_when_no_citations(
    provider: EnhancedMockModelProvider,
    context: ToolUseContext,
) -> None:
    """citations가 비면(CHAT 질의/인용 비활성) KNOWLEDGE_SOURCES 이벤트를 내지 않는다."""
    engine = QueryEngine(
        model_provider=provider,
        tools=[],
        context=context,
        model_dispatcher=_make_fake_dispatcher(),
    )
    engine._prompt_assembler = _FakeAssembler(())

    events = [ev async for ev in engine.submit_message("안녕")]

    ks_events = [
        e for e in events
        if isinstance(e, StreamEvent) and e.type == StreamEventType.KNOWLEDGE_SOURCES
    ]
    assert ks_events == []
