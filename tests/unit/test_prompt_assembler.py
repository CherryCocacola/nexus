"""
PromptAssembler 단위 테스트 — Part 2.5.9 (v0.14.6).

조립 순서 검증 + CHAT 분기에서 KB 단계 스킵 검증.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.orchestrator.prompt_assembler import PromptAssembler
from core.orchestrator.routing import RoutingDecision


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
    kr.get_context = AsyncMock(return_value="kowiki 청크 본문")

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
    kr.get_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_assemble_injects_no_material_marker_when_kb_empty() -> None:
    """★2 게이팅 — KNOWLEDGE 질의인데 KB 결과가 비면 '관련 자료 없음'을 명시 주입.

    빈 결과여도 침묵하지 않고 '자료 없음 + 추측 금지'를 주입해, 모델이 KB에
    없는 사실(예: BWV 544)을 지어내는 할루시네이션을 막는다.
    """
    kr = MagicMock()
    kr.get_context = AsyncMock(return_value="")  # 관련 자료 0건

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
    kr.get_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_assemble_kb_failure_is_swallowed() -> None:
    """KB 검색에서 예외가 나도 기본 프롬프트는 정상 반환."""
    kr = MagicMock()
    kr.get_context = AsyncMock(side_effect=RuntimeError("embedding down"))

    assembler = PromptAssembler(knowledge_retriever=kr)
    out = await assembler.assemble(
        base_prompt="BASE",
        session_id="s1",
        user_input="니체 철학",
        decision=_make_decision("KNOWLEDGE"),
    )

    assert out == "BASE"
