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
     + CHAT/TOOL 질의는 자동 스킵 (Part 2.5.9 v0.14.6)

각 단계는 실패해도 조용히 폴백 — 어떤 보조 모듈이 죽어도 본류 응답은 생성된다.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.orchestrator.routing import RoutingDecision

logger = logging.getLogger("nexus.orchestrator.prompt_assembler")

# ─── 예산 폴백 상수 (하드코딩 외부화 무회귀용) ───────────────
# 생성자에 예산이 주입되지 않으면 아래 현행 값으로 폴백한다. 값의 단일 출처는
# core/config.py ContextBudgetConfig이며, 이 상수는 그 기본값과 반드시 동일하게
# 유지한다(테스트·경량 환경에서 config 없이 생성해도 기존 동작 보장 = 무회귀).
_DEFAULT_TURN_STATE_TOKENS = 1000  # 이전 턴 요약 주입 상한(현행)
_DEFAULT_PROJECT_RAG_TOKENS = 1500  # 프로젝트 RAG 주입 상한(현행)
_DEFAULT_KNOWLEDGE_RAG_TOKENS = 1000  # 지식베이스 RAG 주입 상한(현행)


class PromptAssembler:
    """system_prompt 조립을 한 객체로 캡슐화.

    주입 가능한 보조 의존성은 전부 `None` 허용 — 테스트·경량 환경에서도 동작.
    """

    def __init__(
        self,
        turn_state_store: Any | None = None,
        rag_retriever: Any | None = None,
        knowledge_retriever: Any | None = None,
        # ── 컨텍스트 예산 주입 (하드코딩 외부화, 2026-07-03) ──
        # None이면 현행 상수로 폴백 → 기존 호출부(테스트 포함)는 동작 불변(무회귀).
        # bootstrap→QueryEngine 경로에서 config.context_budgets 값이 주입된다.
        turn_state_tokens: int | None = None,
        project_rag_tokens: int | None = None,
        knowledge_rag_tokens: int | None = None,
    ) -> None:
        self._turn_state_store = turn_state_store
        self._rag_retriever = rag_retriever
        self._knowledge_retriever = knowledge_retriever
        # 예산 확정 — 주입값 우선, 미주입 시 현행 상수 폴백(무회귀).
        self._turn_state_tokens = (
            turn_state_tokens if turn_state_tokens is not None
            else _DEFAULT_TURN_STATE_TOKENS
        )
        self._project_rag_tokens = (
            project_rag_tokens if project_rag_tokens is not None
            else _DEFAULT_PROJECT_RAG_TOKENS
        )
        self._knowledge_rag_tokens = (
            knowledge_rag_tokens if knowledge_rag_tokens is not None
            else _DEFAULT_KNOWLEDGE_RAG_TOKENS
        )

    async def assemble(
        self,
        base_prompt: str,
        session_id: str,
        user_input: str,
        decision: RoutingDecision,
    ) -> str:
        """시스템 프롬프트를 최종 조립해 반환한다.

        호출 지점은 `QueryEngine.submit_message()` — 매 턴 한 번만 호출.

        조립 timing 로그 — 각 step(turn_state/project_rag/knowledge_rag)의
        소요 시간을 ms 단위로 기록한다. 운영 모니터링 및 RAG 병목 진단에 쓴다
        ([ASSEMBLE_TIMING] 키로 grep).
        """
        prompt = base_prompt
        t_start = time.perf_counter()

        # ① TurnState — 이전 턴 요약
        t0 = time.perf_counter()
        prompt = self._attach_turn_state(prompt, session_id)
        ms_turn_state = (time.perf_counter() - t0) * 1000

        # ② 프로젝트 RAG — 관련 파일 청크 (질의 타입 무관)
        t0 = time.perf_counter()
        prompt = await self._attach_project_rag(prompt, user_input)
        ms_project_rag = (time.perf_counter() - t0) * 1000

        # ③ Knowledge RAG — KNOWLEDGE 질의에만 (tenant 필터 적용)
        t0 = time.perf_counter()
        prompt = await self._attach_knowledge_base(prompt, user_input, decision)
        ms_kb = (time.perf_counter() - t0) * 1000

        ms_total = (time.perf_counter() - t_start) * 1000
        # 한 줄로 모아 찍어 grep이 쉽게 — class는 라우팅 결과를 그대로 노출
        logger.info(
            "[ASSEMBLE_TIMING] class=%s total=%.0fms "
            "turn_state=%.0fms project_rag=%.0fms knowledge_rag=%.0fms",
            decision.query_class, ms_total,
            ms_turn_state, ms_project_rag, ms_kb,
        )
        return prompt

    # ─── 내부 스텝 ───────────────────────────────────
    def _attach_turn_state(self, prompt: str, session_id: str) -> str:
        if self._turn_state_store is None:
            return prompt
        try:
            prev = self._turn_state_store.get_context(
                session_id, max_tokens=self._turn_state_tokens
            )
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
            ctx = await self._rag_retriever.get_context(
                user_input, max_tokens=self._project_rag_tokens
            )
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
        # CHAT/TOOL 질의는 KB 단계 자체를 스킵 (Part 2.5.9 v0.14.6).
        # 인사·잡담에 위키 청크가 주입되어 부자연스러운 답변이 나오는 부작용 차단.
        if not decision.inject_knowledge_rag:
            return prompt
        try:
            kb_ctx = await self._knowledge_retriever.get_context(
                user_input,
                max_tokens=self._knowledge_rag_tokens,
                allowed_sources=decision.allowed_knowledge_sources,
            )
        except Exception as e:
            logger.debug("지식 RAG 주입 실패 (무시): %s", e)
            return prompt
        if not kb_ctx:
            # ★2 게이팅 — KNOWLEDGE 질의인데 관련 자료를 못 찾은 경우.
            # 예전엔 그냥 prompt를 반환해 KB 블록 없이 모델이 자유 생성했고,
            # 이것이 "KB에 없는 사실(예: BWV 544)을 자신 있게 지어내는"
            # 할루시네이션의 직접 원인이었다. 빈 결과여도 침묵하지 말고
            # "관련 자료 없음"을 명시 주입해, worker_system.md의 grounding
            # 지침과 결합되어 모델이 추측 대신 "자료에 없어 확실치 않다"고 답하게 한다.
            logger.info("지식 RAG: 관련 자료 없음 — '자료 없음' 마커 주입")
            return (
                prompt
                + "\n\n--- Knowledge base ---\n"
                + "(질의와 직접 관련된 자료를 지식베이스에서 찾지 못했습니다.)\n"
                + "--- End of knowledge base ---\n"
                + "No relevant material was found above. For verifiable facts "
                "(catalog numbers, names, dates, figures), do NOT invent an answer. "
                "If you are not confident from well-established common knowledge, "
                "say honestly in the user's language that the material is not "
                "available and you cannot verify it (예: '제공된 자료에는 없고 "
                "정확히 확인하기 어렵습니다')."
            )
        logger.info("지식 RAG 주입: ~%d자", len(kb_ctx))
        # 검색 결과가 질의와 무관할 때 모델이 무리하게 활용하지 않도록 명시.
        # kowiki 100만 청크 환경에서 어떤 질의든 코사인 유사도로 무언가가 잡히지만
        # 의미적으로 관련이 없을 수 있다. "주어진 컨텍스트 = 정답 재료"로 오인하지
        # 말 것 + 검증 가능한 사실은 근거 없으면 단정 금지(★1 grounding)를 명시.
        return (
            prompt
            + "\n\n--- Knowledge base ---\n"
            + kb_ctx
            + "\n--- End of knowledge base ---\n"
            + "Use the information above ONLY when it is clearly relevant to the "
            "user's question. If the snippets are off-topic or irrelevant, do not "
            "force-fit them. For verifiable facts (catalog numbers, names, dates, "
            "figures), state them as certain ONLY if supported above or by "
            "well-established common knowledge; otherwise say you are not sure "
            "rather than guessing."
        )
