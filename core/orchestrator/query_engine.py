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

        # v7.0 Phase 9: dispatcher가 있으면 route() 경유, 없으면 query_loop 직접 호출
        # dispatcher 경로는 Scout → Worker 2단계(TIER_S)를 포함하며
        # TIER_M/L에서는 passthrough로 단일 Worker에 직행한다.
        if self._model_dispatcher is not None:
            stream = self._model_dispatcher.route(
                messages=self._messages,
                system_prompt=effective_system_prompt,
                on_turn_complete=_on_turn_complete,
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
