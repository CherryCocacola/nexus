"""
시스템 프롬프트 조립기 (2026-04-21 리팩토링).

`QueryEngine.submit_message()`에 뭉쳐 있던 "effective_system_prompt 조립" 로직을
분리한다. 조립 순서와 규칙이 한곳에 모여 있어 디버깅과 유지가 쉽다.

조립 순서 (누적):
  1. base_system_prompt              (상시)
  2. TurnState 이전 턴 요약           (turn_state_store가 있을 때)
  3. RAG 관련 파일 청크               (rag_retriever가 있을 때, 모든 질의)
  4. Knowledge base 청크              (knowledge_retriever가 있고 KNOWLEDGE 질의일 때)
     + tenant.allowed_knowledge_sources 필터 자동 적용

각 단계는 실패해도 조용히 폴백 — 어떤 보조 모듈이 죽어도 본류 응답은 생성된다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.orchestrator.routing import RoutingDecision

logger = logging.getLogger("nexus.orchestrator.prompt_assembler")


class PromptAssembler:
    """system_prompt 조립을 한 객체로 캡슐화.

    주입 가능한 보조 의존성은 전부 `None` 허용 — 테스트·경량 환경에서도 동작.
    """

    def __init__(
        self,
        turn_state_store: Any | None = None,
        rag_retriever: Any | None = None,
        knowledge_retriever: Any | None = None,
    ) -> None:
        self._turn_state_store = turn_state_store
        self._rag_retriever = rag_retriever
        self._knowledge_retriever = knowledge_retriever

    async def assemble(
        self,
        base_prompt: str,
        session_id: str,
        user_input: str,
        decision: RoutingDecision,
    ) -> str:
        """시스템 프롬프트를 최종 조립해 반환한다.

        호출 지점은 `QueryEngine.submit_message()` — 매 턴 한 번만 호출.
        """
        prompt = base_prompt

        # ① TurnState — 이전 턴 요약
        prompt = self._attach_turn_state(prompt, session_id)

        # ② 프로젝트 RAG — 관련 파일 청크 (질의 타입 무관)
        prompt = await self._attach_project_rag(prompt, user_input)

        # ③ Knowledge RAG — KNOWLEDGE 질의에만 (tenant 필터 적용)
        prompt = await self._attach_knowledge_base(prompt, user_input, decision)

        return prompt

    # ─── 내부 스텝 ───────────────────────────────────
    def _attach_turn_state(self, prompt: str, session_id: str) -> str:
        if self._turn_state_store is None:
            return prompt
        try:
            prev = self._turn_state_store.get_context(session_id, max_tokens=1000)
        except Exception as e:
            logger.debug("TurnState 조회 실패 (무시): %s", e)
            return prompt
        if not prev:
            return prompt
        return (
            prompt
            + "\n\n--- Previous context ---\n"
            + prev
        )

    async def _attach_project_rag(self, prompt: str, user_input: str) -> str:
        if self._rag_retriever is None:
            return prompt
        try:
            ctx = await self._rag_retriever.get_context(user_input, max_tokens=1500)
        except Exception as e:
            logger.debug("RAG 검색 실패 (무시): %s", e)
            return prompt
        if not ctx:
            return prompt
        return (
            prompt
            + "\n\n--- Relevant files ---\n"
            + ctx
            + "\n--- End of relevant files ---"
        )

    async def _attach_knowledge_base(
        self,
        prompt: str,
        user_input: str,
        decision: RoutingDecision,
    ) -> str:
        if self._knowledge_retriever is None:
            return prompt
        if decision.query_class != "KNOWLEDGE":
            return prompt
        try:
            kb_ctx = await self._knowledge_retriever.get_context(
                user_input,
                max_tokens=1000,
                allowed_sources=decision.allowed_knowledge_sources,
            )
        except Exception as e:
            logger.debug("지식 RAG 주입 실패 (무시): %s", e)
            return prompt
        if not kb_ctx:
            return prompt
        logger.info("지식 RAG 주입: ~%d자", len(kb_ctx))
        return (
            prompt
            + "\n\n--- Knowledge base ---\n"
            + kb_ctx
            + "\n--- End of knowledge base ---\n"
            + "Answer the user using the information above as your primary reference. "
            "If the knowledge base does not cover the question, state what you know "
            "generally and clearly mark uncertain parts."
        )
