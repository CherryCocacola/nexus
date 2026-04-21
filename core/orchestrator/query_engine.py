"""
쿼리 엔진 — Tier 1 세션 오케스트레이터.

Claude Code의 QueryEngine.ts를 Python으로 재구현한다.
4-Tier AsyncGenerator 체인의 최상위 레이어로서,
세션 단위의 대화를 관리하고 query_loop (Tier 2)를 호출한다.

핵심 역할:
  1. 대화 히스토리(messages[]) 관리
  2. 시스템 프롬프트 조립
  3. 사용자 입력 전처리
  4. query_loop() 호출 → StreamEvent/Message yield
  5. 세션 사용량 추적

4-Tier 체인에서의 위치:
  Tier 1: QueryEngine.submit_message() ← 여기
  Tier 2: query_loop()
  Tier 3: model_provider.stream()
  Tier 4: httpx 클라이언트

의존성 방향:
  QueryEngine → query_loop, ContextManager, ModelProvider, BaseTool
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from core.config import RoutingConfig, RoutingProfile
from core.message import (
    Message,
    StreamEvent,
    StreamEventType,
    TokenUsage,
)
from core.model.inference import ModelProvider
from core.orchestrator.context_manager import ContextManager
from core.orchestrator.query_loop import query_loop
from core.tools.base import BaseTool, ToolUseContext

logger = logging.getLogger("nexus.orchestrator.query_engine")


# ─────────────────────────────────────────────
# v7.0 Part 2.5 — 쿼리 분류기 (2026-04-21)
# ─────────────────────────────────────────────
# 짜라투스트라 사건(Phase 3 LoRA가 인문학 지식 표현을 좁힘) 이후 추가된 로직.
# 실측 A/B 결과: 베이스 Qwen(LoRA OFF) + temp=0.2가 지식 질문에서 압도적 우위.
# 반대로 도구 호출 질문은 Phase 3 LoRA가 필수 — tool_call 포맷 학습이 담겨 있음.
# 분류기는 단순 휴리스틱(길이 + 키워드) — 복잡한 LLM 분류기는 오버헤드만 추가.
def classify_query(user_input: str, routing: RoutingConfig) -> str:
    """
    사용자 입력을 "KNOWLEDGE" 또는 "TOOL"로 분류한다.

    분류 규칙:
      1. enabled=False → 항상 "TOOL" (라우팅 기능 오프)
      2. 길이가 long_input_threshold 이상 → "TOOL"
         (문서/로그 첨부는 거의 항상 도구 경로로 처리되어야 함)
      3. tool_keywords 중 하나 이상 포함 → "TOOL"
      4. 그 외 → "KNOWLEDGE"

    Args:
        user_input: 사용자 메시지 원문
        routing: RoutingConfig (config.routing)

    Returns:
        "KNOWLEDGE" 또는 "TOOL"
    """
    if not routing.enabled:
        return "TOOL"

    # 1) 길이 기반 — 긴 입력은 첨부/로그 가능성이 높음
    if len(user_input) >= routing.long_input_threshold:
        return "TOOL"

    # 2) 키워드 기반 — 대소문자 무시 단순 포함 검사
    lowered = user_input.lower()
    for kw in routing.tool_keywords:
        if kw.lower() in lowered:
            return "TOOL"

    # 3) 나머지는 일반 지식 질의
    return "KNOWLEDGE"


def _resolve_profile(query_class: str, routing: RoutingConfig) -> RoutingProfile:
    """분류 결과로 실행 프로필을 골라 돌려준다."""
    if query_class == "KNOWLEDGE":
        return routing.knowledge_mode
    return routing.tool_mode


class QueryEngine:
    """
    세션 오케스트레이터 — 4-Tier 체인의 Tier 1.

    하나의 QueryEngine 인스턴스가 하나의 대화 세션을 담당한다.
    사용자 메시지를 받으면 messages[]에 추가하고,
    query_loop()을 호출하여 모델 응답 → 도구 실행 → 결과 반환의
    전체 흐름을 AsyncGenerator로 yield한다.

    사용 예시:
        engine = QueryEngine(
            model_provider=provider,
            tools=tools,
            context=context,
            system_prompt="당신은 도움을 주는 AI입니다.",
        )
        async for event in engine.submit_message("안녕하세요"):
            if isinstance(event, StreamEvent) and event.type == StreamEventType.TEXT_DELTA:
                print(event.text, end="")
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        tools: list[BaseTool],
        context: ToolUseContext,
        system_prompt: str = "",
        context_manager: ContextManager | None = None,
        max_turns: int = 200,
        hook_manager: Any | None = None,
        turn_state_store: Any | None = None,
        rag_retriever: Any | None = None,
        model_dispatcher: Any | None = None,
        routing_config: RoutingConfig | None = None,
        memory_manager: Any | None = None,
        transcript: Any | None = None,
        knowledge_retriever: Any | None = None,
    ) -> None:
        """
        QueryEngine을 초기화한다.

        Args:
            model_provider: LLM 프로바이더 (Tier 3 진입점) — 폴백 경로용
            tools: 사용 가능한 도구 리스트 (폴백 경로에서만 사용)
            context: 도구 실행 컨텍스트 (cwd, session_id 등)
            system_prompt: 시스템 프롬프트 텍스트
            context_manager: 컨텍스트 압축 관리자 (선택)
            max_turns: 최대 턴 수 (기본: 200)
            hook_manager: 훅 매니저 (선택)
            turn_state_store: TurnStateStore (v7.0, 선택) — 턴 상태 외부화
            rag_retriever: RAGRetriever (선택) — 관련 문서 청크 자동 검색
            model_dispatcher: ModelDispatcher (v7.0 Phase 9, 선택)
                주입되면 submit_message()는 dispatcher.route()를 호출한다.
                TIER_S에서는 Scout→Worker 2단계, TIER_M/L에서는 Worker 단독.
                None이면 기존 경로(query_loop 직접 호출)로 폴백한다.
            routing_config: RoutingConfig (v7.0 Part 2.5, 선택)
                None이면 기본 RoutingConfig 사용(enabled=True).
                enabled=False면 분류 없이 항상 tool_mode 프로필 적용.
        """
        self._model_provider = model_provider
        self._tools = tools
        self._context = context
        self._system_prompt = system_prompt
        self._context_manager = context_manager
        self._max_turns = max_turns
        self._hook_manager = hook_manager

        # v7.0: 턴 상태 외부화 저장소
        self._turn_state_store = turn_state_store

        # RAG: 관련 문서 청크 자동 검색
        self._rag_retriever = rag_retriever

        # v7.0 Phase 9: 멀티모델 디스패처 (Scout + Worker)
        # None이면 단일 Worker 경로로 폴백한다 (하위 호환).
        self._model_dispatcher = model_dispatcher

        # v7.0 Part 2.5: 쿼리 라우팅 설정 (None이면 기본값 사용)
        self._routing_config = routing_config or RoutingConfig()

        # Ch 16 세션 영속화: MemoryManager와 트랜스크립트
        # - memory_manager: on_turn_end로 Redis 단기 저장 + tb_memories 장기 승격
        # - transcript: JSONL 파일에 영구 기록 (서버 재기동 후 조회)
        # 둘 다 None 허용 — 테스트/경량 환경에서도 QueryEngine이 동작하도록.
        self._memory_manager = memory_manager
        self._transcript = transcript

        # Part 2.5.8: 지식 RAG — KNOWLEDGE_MODE 진입 시 tb_knowledge에서 검색 주입
        # None이면 주입하지 않음 (tb_knowledge 미구성 환경/테스트)
        self._knowledge_retriever = knowledge_retriever

        # 대화 히스토리 — submit_message() 호출마다 누적
        self._messages: list[Message] = []

        # 세션 사용량 추적
        self._cumulative_usage = TokenUsage()
        self._total_turns: int = 0

        # 세션 ID — ToolUseContext에서 가져오거나 새로 생성
        self._session_id = context.session_id or str(uuid.uuid4())

        logger.info(
            "QueryEngine 초기화: session=%s, tools=%d개, max_turns=%d",
            self._session_id,
            len(tools),
            max_turns,
        )

    async def submit_message(
        self, user_input: str
    ) -> AsyncGenerator[StreamEvent | Message, None]:
        """
        사용자 메시지를 제출하고 스트리밍 응답을 반환한다.

        이 메서드가 4-Tier 체인의 진입점이다.
        사용자 입력을 Message로 변환하여 대화에 추가하고,
        query_loop()을 호출하여 모델 응답을 yield한다.

        Args:
            user_input: 사용자 입력 텍스트

        Yields:
            StreamEvent: 스트리밍 이벤트 (UI 업데이트용)
            Message: assistant/tool_result 메시지 (대화 기록용)
        """
        # v7.0: TurnState 기반 컨텍스트 복원
        # TurnStateStore가 있으면 이전 턴의 요약을 시스템 프롬프트에 주입하고,
        # raw messages는 현재 턴의 사용자 메시지만 유지한다.
        # 이전 대화 맥락은 TurnState 요약으로 대체된다.
        effective_system_prompt = self._system_prompt
        if self._turn_state_store is not None:
            # 이전 턴 요약을 컨텍스트로 가져온다 (최대 1000토큰)
            prev_context = self._turn_state_store.get_context(
                self._session_id, max_tokens=1000
            )
            if prev_context:
                effective_system_prompt = (
                    self._system_prompt
                    + "\n\n--- Previous context ---\n"
                    + prev_context
                )
            # messages를 현재 턴만으로 초기화 (이전 raw messages 제거)
            # 왜: TurnState 요약이 이전 맥락을 대체하므로
            # raw messages 누적이 불필요하다.
            self._messages.clear()

        # 사용자 메시지를 대화 히스토리에 추가
        user_msg = Message.user(user_input)
        self._messages.append(user_msg)

        logger.info(
            "메시지 제출: session=%s, messages=%d개, input='%s'",
            self._session_id,
            len(self._messages),
            user_input[:80],
        )

        # RAG: 관련 문서 청크를 시스템 프롬프트에 주입
        # 사용자 입력을 임베딩으로 변환하여 인덱싱된 청크에서 유사한 것을 검색한다.
        # 결과를 "--- Relevant files ---" 섹션으로 추가한다.
        if self._rag_retriever is not None:
            try:
                rag_context = await self._rag_retriever.get_context(
                    user_input, max_tokens=1500
                )
                if rag_context:
                    effective_system_prompt = (
                        effective_system_prompt
                        + "\n\n--- Relevant files ---\n"
                        + rag_context
                        + "\n--- End of relevant files ---"
                    )
            except Exception as e:
                logger.debug("RAG 검색 실패 (무시): %s", e)

        # v7.0: TurnState 콜백 — query_loop이 턴 완료 시 호출
        def _on_turn_complete(turn_state: Any) -> None:
            if self._turn_state_store is not None:
                self._turn_state_store.save(self._session_id, turn_state)

        # v7.0 Part 2.5: 쿼리 타입 분류 → 프로필 선택
        # KNOWLEDGE → 베이스 Qwen + temp 0.2 + max_tokens 2048 (일반 지식 QA)
        # TOOL      → Phase 3 LoRA + temp 0.3 + max_tokens 4096 (도구 호출/프로젝트)
        # 분류 결과는 로그와 사용량 추적에 남겨 운영자가 오판을 감지할 수 있도록 한다.
        #
        # routing_config.enabled=False는 "라우팅 OFF" 모드. 이때는 프로필/프로바이더
        # 치환을 일체 하지 않고 기본 동작을 따른다. 특히 서브에이전트(Scout 등)는
        # 자기 전용 프로바이더를 쓰므로 라우팅이 개입하면 model_override로 엉뚱한
        # 모델명이 주입되어 오작동한다(2026-04-21 α 재진단).
        if self._routing_config.enabled:
            query_class = classify_query(user_input, self._routing_config)
            profile = _resolve_profile(query_class, self._routing_config)
            active_model_override: str | None = profile.model
            active_temperature = profile.temperature
            active_max_tokens_cap: int | None = profile.max_tokens
            active_enable_thinking: bool | None = profile.enable_thinking

            # 멀티테넌시: tenant.model_override가 있으면 프로필 모델보다 우선 적용.
            # 이유: tenant 전용 LoRA가 존재하면 QueryEngine이 그 어댑터로 응답해야
            # 학교/기업별 커스터마이즈가 의미를 갖는다. TOOL 모드에선 Phase LoRA
            # 호환성이 필요하므로 KNOWLEDGE 질의에만 override를 적용한다.
            tenant = self._context.options.get("tenant") if self._context else None
            if (
                tenant is not None
                and getattr(tenant, "model_override", None)
                and query_class == "KNOWLEDGE"
            ):
                active_model_override = tenant.model_override
                logger.info(
                    "라우팅(tenant): tenant=%s → model=%s",
                    tenant.id, tenant.model_override,
                )

            logger.info(
                "라우팅: class=%s, model=%s, temp=%.2f, max_tokens=%d",
                query_class,
                active_model_override,
                active_temperature,
                active_max_tokens_cap,
            )

            # Part 2.5.8: KNOWLEDGE 분류일 때만 tb_knowledge RAG 주입
            # 일반 지식 질의에 한해 외부 지식 베이스(위키 등)에서 관련 청크를
            # 검색하여 시스템 프롬프트에 붙인다. TOOL 질의는 도구 호출 흐름이
            # 우선이므로 주입하지 않는다(지연/컨텍스트 낭비 회피).
            if query_class == "KNOWLEDGE" and self._knowledge_retriever is not None:
                try:
                    # 멀티테넌시 — tenant.allowed_knowledge_sources가 있으면 그 소스만 검색
                    allowed = None
                    if tenant is not None:
                        src = getattr(tenant, "allowed_knowledge_sources", None) or []
                        # 빈 리스트는 "명시적으로 비허용"이 아니라 "필터 미지정"으로 간주하기
                        # 위해 None 처리. 단, 명시적 격리를 원하면 tenants.yaml에
                        # sources=[특정값]을 정의해야 한다.
                        allowed = list(src) if src else None
                    kb_ctx = await self._knowledge_retriever.get_context(
                        user_input, max_tokens=1000,
                        allowed_sources=allowed,
                    )
                    if kb_ctx:
                        effective_system_prompt = (
                            effective_system_prompt
                            + "\n\n--- Knowledge base ---\n"
                            + kb_ctx
                            + "\n--- End of knowledge base ---\n"
                            + "Answer the user using the information above as your "
                            "primary reference. If the knowledge base does not cover "
                            "the question, state what you know generally and clearly "
                            "mark uncertain parts."
                        )
                        logger.info("지식 RAG 주입: ~%d자", len(kb_ctx))
                except Exception as e:
                    logger.debug("지식 RAG 주입 실패 (무시): %s", e)
        else:
            # 라우팅 비활성 — 프로바이더의 기본 설정을 그대로 사용한다
            active_model_override = None
            active_temperature = 0.7
            active_max_tokens_cap = None
            active_enable_thinking = False
            logger.info("라우팅 비활성 — 프로바이더 기본 설정 사용")

        # v7.0 Phase 9: dispatcher가 있으면 route() 경유, 없으면 query_loop 직접 호출
        # dispatcher 경로는 Scout → Worker 2단계(TIER_S)를 포함하며
        # TIER_M/L에서는 passthrough로 단일 Worker에 직행한다.
        if self._model_dispatcher is not None:
            stream = self._model_dispatcher.route(
                messages=self._messages,
                system_prompt=effective_system_prompt,
                on_turn_complete=_on_turn_complete,
                model_override=active_model_override,
                temperature=active_temperature,
                max_tokens_cap=active_max_tokens_cap,
                enable_thinking=active_enable_thinking,
            )
        else:
            # 폴백 — dispatcher 주입이 없는 경우 기존 단일 Worker 경로
            stream = query_loop(
                messages=self._messages,
                system_prompt=effective_system_prompt,
                model_provider=self._model_provider,
                tools=self._tools,
                context=self._context,
                context_manager=self._context_manager,
                max_turns=self._max_turns,
                on_turn_complete=_on_turn_complete,
                model_override=active_model_override,
                temperature=active_temperature,
                max_tokens_cap=active_max_tokens_cap,
                enable_thinking=active_enable_thinking,
            )

        async for event in stream:
            # 사용량 추적 — USAGE_UPDATE 이벤트에서 누적
            if (
                isinstance(event, StreamEvent)
                and event.type == StreamEventType.USAGE_UPDATE
                and event.usage
            ):
                self._cumulative_usage = event.usage

            yield event

        self._total_turns += 1

        logger.info(
            "메시지 처리 완료: session=%s, 누적 턴=%d, "
            "토큰=%d/%d (input/output)",
            self._session_id,
            self._total_turns,
            self._cumulative_usage.input_tokens,
            self._cumulative_usage.output_tokens,
        )

        # Ch 16: 턴 종료 훅 — 세션 영속화
        # (1) MemoryManager.on_turn_end — Redis 단기 + 중요도 평가 후 tb_memories 승격
        # (2) Transcript.append_entry — JSONL 파일에 user/assistant 쌍 기록
        # 둘 다 실패해도 본류 응답은 이미 yield 완료 상태 — 최대한 안전한 swallow.
        await self._finalize_turn(user_input)

    async def _finalize_turn(self, user_input: str) -> None:
        """턴 종료 시 메모리/트랜스크립트 기록을 수행한다 (실패 시 swallow)."""
        # (1) MemoryManager 연동
        if self._memory_manager is not None:
            try:
                await self._memory_manager.on_turn_end(
                    session_id=self._session_id,
                    messages=self._messages,
                )
            except Exception as e:
                # 메모리 저장 실패는 치명적이지 않다 — 로그만 남기고 진행
                logger.warning(
                    "MemoryManager.on_turn_end 실패 (session=%s): %s",
                    self._session_id, e,
                )

        # (2) 트랜스크립트 기록 — 마지막 user/assistant 쌍만 append
        # 왜 쌍만? 전체 messages를 매번 덮어쓰면 append-only 규칙 위반 + 중복 누적
        if self._transcript is not None:
            try:
                last_user: str | None = user_input
                last_assistant: str | None = None
                # messages 끝부터 역순으로 탐색 — 가장 최근 assistant 텍스트
                for m in reversed(self._messages):
                    role = m.role if isinstance(m.role, str) else m.role.value
                    if role == "assistant":
                        text = m.text_content if hasattr(m, "text_content") else str(m.content)
                        if text:
                            last_assistant = text
                            break
                usage = {
                    "input_tokens": self._cumulative_usage.input_tokens,
                    "output_tokens": self._cumulative_usage.output_tokens,
                }
                if last_user:
                    self._transcript.append_entry(
                        role="user",
                        content=last_user,
                        turn=self._total_turns,
                    )
                if last_assistant:
                    self._transcript.append_entry(
                        role="assistant",
                        content=last_assistant,
                        turn=self._total_turns,
                        usage=usage,
                    )
            except Exception as e:
                logger.warning(
                    "Transcript 기록 실패 (session=%s): %s",
                    self._session_id, e,
                )

    # ─── 대화 상태 조회 메서드 ───

    @property
    def messages(self) -> list[Message]:
        """현재 대화 히스토리를 반환한다 (읽기 전용 복사)."""
        return list(self._messages)

    @property
    def session_id(self) -> str:
        """세션 ID를 반환한다."""
        return self._session_id

    @property
    def usage(self) -> TokenUsage:
        """누적 토큰 사용량을 반환한다."""
        return self._cumulative_usage

    @property
    def total_turns(self) -> int:
        """누적 submit_message 호출 횟수를 반환한다."""
        return self._total_turns

    @property
    def tools(self) -> list[BaseTool]:
        """사용 가능한 도구 리스트를 반환한다."""
        return self._tools

    @property
    def system_prompt(self) -> str:
        """현재 시스템 프롬프트를 반환한다."""
        return self._system_prompt

    @property
    def model_dispatcher(self) -> Any | None:
        """
        주입된 ModelDispatcher를 반환한다 (없으면 None).

        Ch 17 메트릭 엔드포인트가 Scout 통계를 조회할 때 사용한다.
        """
        return self._model_dispatcher

    def update_system_prompt(self, prompt: str) -> None:
        """시스템 프롬프트를 업데이트한다."""
        self._system_prompt = prompt

    def clear_messages(self) -> None:
        """대화 히스토리를 초기화한다 (새 세션 시작)."""
        self._messages.clear()
        logger.info("대화 히스토리 초기화: session=%s", self._session_id)

    def get_last_assistant_text(self) -> str:
        """마지막 assistant 메시지의 텍스트를 반환한다."""
        for msg in reversed(self._messages):
            role = str(msg.role)
            if role == "assistant":
                return msg.text_content
        return ""
