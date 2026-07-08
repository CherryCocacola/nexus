"""
PromptAssembler 단위 테스트 — Part 2.5.9 (v0.14.6).

조립 순서 검증 + CHAT 분기에서 KB 단계 스킵 검증.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.message import KnowledgeCitation
from core.orchestrator.prompt_assembler import PromptAssembler
from core.orchestrator.routing import RoutingDecision
from core.rag.knowledge_retriever import KnowledgeContext


def _make_decision(query_class: str) -> RoutingDecision:
    """테스트용 최소 RoutingDecision."""
    return RoutingDecision(
        query_class=query_class,
        model_override="qwen3.5-27b",
        temperature=0.2,
        max_tokens_cap=1024,
        enable_thinking=False,
        allowed_knowledge_sources=None,
    )


@pytest.mark.asyncio
async def test_assemble_skips_kb_for_chat_decision() -> None:
    """CHAT 분류일 때 KnowledgeRetriever.get_context()는 호출되지 않는다.

    kowiki 100만 청크 환경에서 인사·잡담에 위키 청크가 주입되는 부작용 차단
    (Part 2.5.9 v0.14.6).
    """
    kr = MagicMock()
    kr.get_context = AsyncMock(return_value="이 텍스트는 절대 주입되어선 안 됨")

    assembler = PromptAssembler(knowledge_retriever=kr)
    out = await assembler.assemble(
        base_prompt="BASE",
        session_id="s1",
        user_input="안녕",
        decision=_make_decision("CHAT"),
    )

    assert out == "BASE"
    kr.get_context.assert_not_called()


@pytest.mark.asyncio
async def test_assemble_skips_kb_for_tool_decision() -> None:
    """TOOL 분류도 KB 주입 안 됨 (기존 동작 유지)."""
    kr = MagicMock()
    kr.get_context = AsyncMock(return_value="off-topic chunk")

    assembler = PromptAssembler(knowledge_retriever=kr)
    out = await assembler.assemble(
        base_prompt="BASE",
        session_id="s1",
        user_input="이 파일 읽어줘",
        decision=_make_decision("TOOL"),
    )

    assert "off-topic" not in out
    kr.get_context.assert_not_called()


@pytest.mark.asyncio
async def test_assemble_injects_kb_for_knowledge_decision() -> None:
    """KNOWLEDGE 분류일 때만 KB 주입 + grounding(불확실 시 추측 금지) 지시 포함."""
    kr = MagicMock()
    kr.get_context_with_citations = AsyncMock(
        return_value=KnowledgeContext(text="kowiki 청크 본문", citations=())
    )

    assembler = PromptAssembler(knowledge_retriever=kr)
    out = await assembler.assemble(
        base_prompt="BASE",
        session_id="s1",
        user_input="니체 철학 요약",
        decision=_make_decision("KNOWLEDGE"),
    )

    assert "kowiki 청크 본문" in out
    # ★1 grounding 보조 지시가 KB 블록 뒤에 함께 들어간다
    # (검증 가능한 사실은 근거 없으면 단정 말고 'not sure'라고 답하라)
    assert "not sure rather than guessing" in out
    kr.get_context_with_citations.assert_awaited_once()


@pytest.mark.asyncio
async def test_assemble_injects_no_material_marker_when_kb_empty() -> None:
    """★2 게이팅 — KNOWLEDGE 질의인데 KB 결과가 비면 '관련 자료 없음'을 명시 주입.

    빈 결과여도 침묵하지 않고 '자료 없음 + 추측 금지'를 주입해, 모델이 KB에
    없는 사실(예: BWV 544)을 지어내는 할루시네이션을 막는다.
    """
    kr = MagicMock()
    kr.get_context_with_citations = AsyncMock(
        return_value=KnowledgeContext(text="", citations=())
    )  # 관련 자료 0건

    assembler = PromptAssembler(knowledge_retriever=kr)
    out = await assembler.assemble(
        base_prompt="BASE",
        session_id="s1",
        user_input="바흐 BWV 544 작품 알려줘",
        decision=_make_decision("KNOWLEDGE"),
    )

    # '관련 자료 없음' 마커 + 추측 금지 지시가 주입된다
    assert "찾지 못했습니다" in out
    assert "do NOT invent" in out
    kr.get_context_with_citations.assert_awaited_once()


@pytest.mark.asyncio
async def test_assemble_kb_failure_is_swallowed() -> None:
    """KB 검색에서 예외가 나도 기본 프롬프트는 정상 반환."""
    kr = MagicMock()
    kr.get_context_with_citations = AsyncMock(
        side_effect=RuntimeError("embedding down")
    )

    assembler = PromptAssembler(knowledge_retriever=kr)
    out = await assembler.assemble(
        base_prompt="BASE",
        session_id="s1",
        user_input="니체 철학",
        decision=_make_decision("KNOWLEDGE"),
    )

    assert out == "BASE"


# ─────────────────────────────────────────────
# 출처 인용 trailer + last_knowledge_citations (Point 4-2)
# ─────────────────────────────────────────────
def _mock_kr(citations: tuple, text: str = "kowiki 청크 본문", label: str = "출처") -> MagicMock:
    """get_context_with_citations가 (text, citations)를 돌려주는 mock retriever."""
    kr = MagicMock()
    kr.citation_label = label
    kr.get_context_with_citations = AsyncMock(
        return_value=KnowledgeContext(text=text, citations=citations)
    )
    return kr


@pytest.mark.asyncio
async def test_attach_knowledge_citation_trailer_present_when_enabled() -> None:
    """citation 활성(citations 비어있지 않음) 시 인용 지시 trailer가 붙는다."""
    kr = _mock_kr(
        citations=(
            KnowledgeCitation(index=1, source="kowiki", title="바흐", score=0.9),
        )
    )
    assembler = PromptAssembler(knowledge_retriever=kr)
    out = await assembler.assemble(
        base_prompt="BASE",
        session_id="s1",
        user_input="바흐 알려줘",
        decision=_make_decision("KNOWLEDGE"),
    )

    # 인용 지시문 + grounding 지시가 함께 들어간다.
    assert "Citation rule (출처 표기)" in out
    assert "[출처N]" in out
    assert "not sure rather than guessing" in out
    # citations가 조립기에 보관되어 Tier 1이 이벤트로 노출할 수 있다.
    assert len(assembler.last_knowledge_citations) == 1
    assert assembler.last_knowledge_citations[0].title == "바흐"


@pytest.mark.asyncio
async def test_attach_knowledge_citation_trailer_absent_when_disabled() -> None:
    """citation 비활성(citations 빈 튜플) 시 인용 trailer가 없고, 본문은 정상 주입된다."""
    kr = _mock_kr(citations=())
    assembler = PromptAssembler(knowledge_retriever=kr)
    out = await assembler.assemble(
        base_prompt="BASE",
        session_id="s1",
        user_input="니체 철학",
        decision=_make_decision("KNOWLEDGE"),
    )

    assert "kowiki 청크 본문" in out
    assert "Citation rule" not in out
    assert assembler.last_knowledge_citations == ()


@pytest.mark.asyncio
async def test_last_knowledge_citations_reset_per_assemble() -> None:
    """assemble()마다 last_knowledge_citations가 리셋되어 이전 턴 값이 누수되지 않는다."""
    kr = _mock_kr(
        citations=(
            KnowledgeCitation(index=1, source="kowiki", title="바흐", score=0.9),
        )
    )
    assembler = PromptAssembler(knowledge_retriever=kr)

    # 1턴: KNOWLEDGE → citations 채워짐
    await assembler.assemble(
        base_prompt="BASE", session_id="s1", user_input="바흐",
        decision=_make_decision("KNOWLEDGE"),
    )
    assert len(assembler.last_knowledge_citations) == 1

    # 2턴: CHAT → KB 스킵 → citations가 ()로 리셋되어야 한다(이전 값 누수 금지).
    await assembler.assemble(
        base_prompt="BASE", session_id="s1", user_input="안녕",
        decision=_make_decision("CHAT"),
    )
    assert assembler.last_knowledge_citations == ()
