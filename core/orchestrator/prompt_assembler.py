"""
시스템 프롬프트 조립기 (2026-04-21 리팩토링).

[이 파일이 하는 일]
LLM에게 매 턴 전달할 "시스템 프롬프트"를 완성해서 돌려주는 전용 조립기다.
원래는 `QueryEngine.submit_message()` 안에 이 조립 로직이 뒤섞여 있었는데,
조립 순서·규칙을 한곳으로 모아 디버깅과 유지보수를 쉽게 하려고 분리했다.

시스템 프롬프트는 "기본 지침(base) + 이전 대화 맥락 + 관련 파일 + 지식베이스"를
필요에 따라 차곡차곡 이어 붙여 만든다. 이렇게 하면 모델이 현재 상황에 필요한
배경 정보를 미리 갖고 답변을 시작할 수 있다.

[조립 순서 — 앞에서부터 누적으로 이어 붙임]
  1. base_system_prompt              (상시)
  2. TurnState 이전 턴 요약           (turn_state_store가 있을 때)
  3. RAG 관련 파일 청크               (rag_retriever가 있을 때, 모든 질의)
  4. Knowledge base 청크              (knowledge_retriever가 있고 KNOWLEDGE 질의일 때)
     + tenant.allowed_knowledge_sources 필터 자동 적용
     + CHAT/TOOL 질의는 자동 스킵 (Part 2.5.9 v0.14.6)

[핵심 설계 원칙 — Fail-open 폴백]
각 단계(2~4)는 실패해도 예외를 밖으로 던지지 않고 조용히 원본 프롬프트를
그대로 돌려준다. 즉 TurnState·RAG·지식베이스 같은 "보조" 모듈이 죽어도
본류 응답 생성은 절대 막히지 않는다. 보조 정보는 있으면 좋고 없으면 마는 것.

[주요 구성]
  - PromptAssembler         : 조립 전체를 캡슐화한 클래스
  - PromptAssembler.assemble: 외부에서 호출하는 유일한 진입점(매 턴 1회)
  - _attach_turn_state / _attach_project_rag / _attach_knowledge_base
                            : 순서대로 실행되는 내부 조립 스텝 3개

[의존 관계]
  - RoutingDecision(core.orchestrator.routing) : 질의 분류 결과. 어떤 RAG를
    주입할지, 어떤 지식 소스를 허용할지 결정하는 근거로 쓴다.
  - 호출자는 QueryEngine.submit_message() 하나뿐.

작성자: 이현수 / 작성일: 2026-07-05
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
# 계획 체크리스트(TodoWrite) 주입 상한. TIER_S(8K) 토큰 압박을 고려해 작게 잡는다.
# 초과 시 completed 항목은 개수로 축약한다(아래 _attach_todo_checklist 참조).
_DEFAULT_TODO_TOKENS = 300


def _citation_trailer(label: str) -> str:
    """출처 인용 지시문(Point 4-2 §5.1)을 만든다 — KB 블록 뒤 grounding trailer 뒤에 붙는다.

    citation 활성 + 실제 주입 청크가 있을 때만 사용한다. 라벨(기본 "출처")은
    config에서 오며, 헤더의 `[출처N | ...]`과 같은 값이라 본문 마커와 정합한다.
    나머지 지시는 모델 지시 효율을 위해 기존 trailer들처럼 영어로 둔다(라벨만 한글).
    """
    return (
        f"Citation rule ({label} 표기): Each snippet above is labeled [{label}N].\n"
        f"- When you state a fact taken from a snippet, append its label like "
        f"[{label}1] at the end of that sentence. Multiple labels are allowed: "
        f"[{label}1][{label}3].\n"
        f"- Use ONLY the labels that actually appear above. Never invent labels "
        f"or numbers.\n"
        f"- Do NOT restate snippet titles or metadata in the body — the label "
        f"alone is enough.\n"
        f"- Statements from your own general knowledge get NO label; if a "
        f"verifiable fact has no supporting snippet and you are not confident, "
        f"say you are not sure (Grounding rule)."
    )


class PromptAssembler:
    """system_prompt 조립을 한 객체로 캡슐화한 클래스.

    생성 시점에 세 종류의 "보조 조회기(retriever/store)"와 각 단계의
    토큰 예산을 주입받고, 이후 매 턴 `assemble()`이 호출될 때마다
    그 의존성들을 사용해 프롬프트를 만들어 준다.

    [설계 포인트]
    주입 가능한 보조 의존성(turn_state_store / rag_retriever /
    knowledge_retriever)은 전부 `None`을 허용한다. 그래서 단위 테스트나
    보조 서비스가 없는 경량 환경에서도 이 객체를 그대로 만들어 쓸 수 있다.
    (None으로 들어온 스텝은 실행 시 조용히 건너뛴다.)
    """

    def __init__(
        self,
        turn_state_store: Any | None = None,
        rag_retriever: Any | None = None,
        knowledge_retriever: Any | None = None,
        todo_store: Any | None = None,
        # ── 컨텍스트 예산 주입 (하드코딩 외부화, 2026-07-03) ──
        # None이면 현행 상수로 폴백 → 기존 호출부(테스트 포함)는 동작 불변(무회귀).
        # bootstrap→QueryEngine 경로에서 config.context_budgets 값이 주입된다.
        turn_state_tokens: int | None = None,
        project_rag_tokens: int | None = None,
        knowledge_rag_tokens: int | None = None,
        todo_tokens: int | None = None,
    ) -> None:
        """조립기를 초기화하며 보조 의존성과 토큰 예산을 확정한다.

        매개변수:
          turn_state_store    : 이전 턴 요약을 보관·조회하는 저장소.
                                None이면 ① 스텝을 건너뜀.
          rag_retriever       : 질의와 관련된 프로젝트 파일 청크를 찾아오는
                                검색기. None이면 ② 스텝을 건너뜀.
          knowledge_retriever : 지식베이스(위키 등) 청크를 찾아오는 검색기.
                                None이면 ③ 스텝을 건너뜀.
          turn_state_tokens / project_rag_tokens / knowledge_rag_tokens :
                                각 스텝이 프롬프트에 주입할 최대 토큰 예산.
                                None이면 파일 상단의 현행 상수로 폴백한다
                                (아래 "무회귀" 설명 참조).

        운영 경로에서는 bootstrap→QueryEngine을 거쳐 config.context_budgets의
        값이 예산으로 주입된다. 테스트에서는 예산을 생략해도 상수 폴백으로
        기존과 똑같이 동작한다.
        """
        self._turn_state_store = turn_state_store
        self._rag_retriever = rag_retriever
        self._knowledge_retriever = knowledge_retriever
        # 계획 체크리스트 저장소(TodoWrite). None이면 ①.5 스텝을 건너뛴다.
        self._todo_store = todo_store
        # 예산 확정 — 주입값이 있으면 그걸 쓰고, 없으면(None) 현행 상수로 폴백.
        # 이렇게 해야 예산 인자를 넘기지 않는 기존 호출부·테스트가 동작 불변(무회귀).
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
        self._todo_tokens = (
            todo_tokens if todo_tokens is not None else _DEFAULT_TODO_TOKENS
        )
        # 출처 인용(Point 4-2) — 이번 assemble()에서 KB에 '실제로 주입된' 청크의
        # 출처 목록. _attach_knowledge_base가 채우고(citation 활성 시), assemble()
        # 시작마다 ()로 리셋한다. Tier 1(QueryEngine)이 이 값을 읽어
        # KNOWLEDGE_SOURCES StreamEvent로 상위(웹/CLI)에 1회 전달한다.
        self.last_knowledge_citations: tuple = ()

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
        # 출처 인용 — 이번 조립의 citations를 매번 초기화한다(이전 턴 값 누수 차단).
        # KB 주입이 없거나 인용 비활성이면 ()로 남아 Tier 1이 이벤트를 내지 않는다.
        self.last_knowledge_citations = ()

        # ① TurnState — 이전 턴 요약
        t0 = time.perf_counter()
        prompt = self._attach_turn_state(prompt, session_id)
        ms_turn_state = (time.perf_counter() - t0) * 1000

        # ①.5 계획 체크리스트(TodoWrite) — compact 이후에도 계획을 유지시킨다.
        prompt = self._attach_todo_checklist(prompt, session_id)

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
        """① 이전 턴 요약(TurnState)을 프롬프트 뒤에 이어 붙인다.

        같은 세션의 앞선 대화 맥락을 모델에게 다시 상기시켜, 여러 턴에 걸친
        대화의 연속성을 유지하기 위한 단계다.

        흐름:
          - 저장소가 주입되지 않았으면(None) 아무 것도 안 하고 원본 반환.
          - 저장소 조회가 실패하면 경고만 남기고 원본 반환(fail-open).
          - 요약이 비어 있으면(첫 턴 등) 그대로 원본 반환.
          - 요약이 있으면 'Previous context' 구분선과 함께 덧붙여 반환.
        """
        if self._turn_state_store is None:
            return prompt
        try:
            # 세션별 이전 맥락을 예산(max_tokens) 안에서 가져온다.
            prev = self._turn_state_store.get_context(
                session_id, max_tokens=self._turn_state_tokens
            )
        except Exception as e:
            # 보조 저장소 장애가 본류 응답을 막지 않도록 조용히 폴백.
            logger.debug("TurnState 조회 실패 (무시): %s", e)
            return prompt
        if not prev:
            return prompt
        return (
            prompt
            + "\n\n--- Previous context ---\n"
            + prev
        )

    def _attach_todo_checklist(self, prompt: str, session_id: str) -> str:
        """①.5 현재 계획 체크리스트(TodoWrite)를 프롬프트 뒤에 이어 붙인다.

        모델이 다단계 작업의 원래 계획을 잊지 않도록, 매 턴 최신 체크리스트를
        시스템 프롬프트에 재주입한다. 특히 TIER_S(8K)에서 compact 이후에도 계획
        맥락이 유지된다.

        흐름:
          - 저장소가 주입되지 않았으면(None) 원본 반환.
          - 조회 실패 시 경고만 남기고 원본 반환(fail-open — 보조 정보라 본류를 막지 않음).
          - 목록이 비어 있으면 그대로 원본 반환.
          - 있으면 토큰 예산(_todo_tokens) 안에서 축약해 '## 현재 작업 체크리스트'
            섹션으로 덧붙인다. 예산 초과 시 completed 항목은 개수로만 요약한다.
        """
        if self._todo_store is None:
            return prompt
        try:
            # 메인 에이전트(agent_id=None) 목록만 주입한다(서브에이전트 목록은 격리).
            state = self._todo_store.get(session_id, None)
        except Exception as e:
            # 보조 저장소 장애가 본류 응답을 막지 않도록 조용히 폴백.
            logger.debug("체크리스트 조회 실패 (무시): %s", e)
            return prompt
        items = getattr(state, "items", ())
        if not items:
            return prompt

        # 렌더 — core.todo_store.render_checklist는 순수 함수라 여기서 재사용한다.
        from core.todo_store import TodoStatus, render_checklist

        section = render_checklist(items)
        # 토큰 예산 초과 시 completed 항목을 개수로 축약한다(estimated: 문자수//3).
        if len(section) // 3 > self._todo_tokens:
            done = sum(1 for i in items if getattr(i, "status", None) == TodoStatus.COMPLETED)
            active = tuple(
                i for i in items if getattr(i, "status", None) != TodoStatus.COMPLETED
            )
            section = render_checklist(active)
            if done:
                section += f"\n(완료 {done}개 생략)"
        return (
            prompt
            + "\n\n## 현재 작업 체크리스트\n"
            + section
        )

    async def _attach_project_rag(self, prompt: str, user_input: str) -> str:
        """② 질의와 관련된 프로젝트 파일 청크(RAG)를 프롬프트에 이어 붙인다.

        사용자의 이번 입력(user_input)과 의미적으로 관련 있는 파일 조각을
        검색해 주입한다. 질의 타입과 무관하게(모든 질의에서) 시도한다.
        비동기 검색이 필요하므로 async 메서드다.

        흐름:
          - 검색기가 없으면(None) 원본 반환.
          - 검색 실패 시 경고만 남기고 원본 반환(fail-open).
          - 결과가 비면 원본 반환.
          - 결과가 있으면 'Relevant files' 블록으로 감싸 덧붙여 반환.
        """
        if self._rag_retriever is None:
            return prompt
        try:
            # 사용자 입력을 질의로 삼아 관련 파일 청크를 예산 안에서 검색.
            ctx = await self._rag_retriever.get_context(
                user_input, max_tokens=self._project_rag_tokens
            )
        except Exception as e:
            # 검색 장애가 본류 응답을 막지 않도록 조용히 폴백.
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
        """③ 지식베이스(위키 등) 청크를 KNOWLEDGE 질의에 한해 주입한다.

        라우팅 결과(decision)를 보고 "지식 검색이 필요한 질의"일 때만 지식베이스를
        조회해 주입한다. 인사·잡담(CHAT)이나 도구 실행(TOOL) 질의에는 넣지 않는다.

        이 단계는 할루시네이션 저감 로직의 핵심이라 분기가 세 갈래로 나뉜다:
          (a) 검색기 없음/게이팅 스킵/검색 실패 → 원본 그대로 반환.
          (b) 검색 결과가 비었음 → "관련 자료 없음" 마커를 명시 주입(★2).
              → 모델이 근거 없이 사실을 지어내지 않고 "모른다"고 답하게 유도.
          (c) 검색 결과가 있음 → 지식베이스 블록 + grounding 지침을 함께 주입(★1).

        매개변수:
          decision : 질의 분류 결과. inject_knowledge_rag(주입 여부)와
                     allowed_knowledge_sources(테넌트별 허용 소스 필터)를 제공.
        """
        if self._knowledge_retriever is None:
            return prompt
        # CHAT/TOOL 질의는 KB 단계 자체를 스킵 (Part 2.5.9 v0.14.6).
        # 인사·잡담에 위키 청크가 주입되어 부자연스러운 답변이 나오는 부작용 차단.
        if not decision.inject_knowledge_rag:
            return prompt
        try:
            # 지식베이스 조회 — 예산(max_tokens)과 테넌트 허용 소스 필터를 함께 전달.
            # allowed_sources로 이 테넌트가 볼 수 있는 지식 소스만 걸러 검색한다.
            # get_context_with_citations는 (주입 텍스트 + 출처 목록)을 함께 돌려준다
            # (출처 인용 Point 4-2). citation 비활성이면 citations는 () — 무회귀.
            kb = await self._knowledge_retriever.get_context_with_citations(
                user_input,
                max_tokens=self._knowledge_rag_tokens,
                allowed_sources=decision.allowed_knowledge_sources,
            )
        except Exception as e:
            # 지식 검색 장애가 본류 응답을 막지 않도록 조용히 폴백.
            logger.debug("지식 RAG 주입 실패 (무시): %s", e)
            return prompt
        kb_ctx = kb.text
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
        # 출처 인용(Point 4-2) — 실제 주입된 청크의 출처 목록을 보관한다.
        # Tier 1(QueryEngine)이 이 값을 읽어 KNOWLEDGE_SOURCES 이벤트로 노출한다.
        # citation 비활성이거나 주입 청크가 없으면 kb.citations는 ()라 무영향.
        self.last_knowledge_citations = kb.citations
        logger.info(
            "지식 RAG 주입: ~%d자 (citations=%d)", len(kb_ctx), len(kb.citations)
        )
        # 검색 결과가 질의와 무관할 때 모델이 무리하게 활용하지 않도록 명시.
        # kowiki 100만 청크 환경에서 어떤 질의든 코사인 유사도로 무언가가 잡히지만
        # 의미적으로 관련이 없을 수 있다. "주어진 컨텍스트 = 정답 재료"로 오인하지
        # 말 것 + 검증 가능한 사실은 근거 없으면 단정 금지(★1 grounding)를 명시.
        trailer = (
            "Use the information above ONLY when it is clearly relevant to the "
            "user's question. If the snippets are off-topic or irrelevant, do not "
            "force-fit them. For verifiable facts (catalog numbers, names, dates, "
            "figures), state them as certain ONLY if supported above or by "
            "well-established common knowledge; otherwise say you are not sure "
            "rather than guessing."
        )
        # 인용 활성 + 실제 주입 청크가 있을 때(=citations 비어있지 않음)만 인용 지시를
        # grounding trailer 뒤에 덧붙인다. "자료 없음" 분기(b)에는 인용 대상이 없어
        # 넣지 않는다(위에서 이미 return). 라벨은 retriever와 같은 값으로 맞춘다.
        if kb.citations:
            label = getattr(self._knowledge_retriever, "citation_label", "출처")
            trailer = trailer + "\n\n" + _citation_trailer(label)
        return (
            prompt
            + "\n\n--- Knowledge base ---\n"
            + kb_ctx
            + "\n--- End of knowledge base ---\n"
            + trailer
        )
