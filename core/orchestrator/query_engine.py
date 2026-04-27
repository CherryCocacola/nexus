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

from core.config import RoutingConfig
from core.message import (
    Message,
    StreamEvent,
    StreamEventType,
    TokenUsage,
)
from core.model.inference import ModelProvider
from core.orchestrator.context_manager import ContextManager
from core.orchestrator.prompt_assembler import PromptAssembler
from core.orchestrator.query_loop import query_loop
from core.orchestrator.routing import (
    RoutingDecision,
    RoutingResolver,
    _resolve_profile,  # noqa: F401 — 하위 호환 export
    classify_query,  # noqa: F401 — 하위 호환 export
)
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
        # 라우팅·시스템 프롬프트 조립은 별도 객체로 캡슐화 (2026-04-21 리팩토링)
        # → submit_message()가 조율만 담당, 세부 로직은 Resolver/Assembler가 담당.
        self._router = RoutingResolver(self._routing_config)
        self._prompt_assembler = PromptAssembler(
            turn_state_store=turn_state_store,
            rag_retriever=rag_retriever,
            knowledge_retriever=None,  # 아래 setattr 이후 바인딩
        )

        # Ch 16 세션 영속화: MemoryManager와 트랜스크립트
        # - memory_manager: on_turn_end로 Redis 단기 저장 + tb_memories 장기 승격
        # - transcript: JSONL 파일에 영구 기록 (서버 재기동 후 조회)
        # 둘 다 None 허용 — 테스트/경량 환경에서도 QueryEngine이 동작하도록.
        self._memory_manager = memory_manager
        self._transcript = transcript

        # Part 2.5.8: 지식 RAG — KNOWLEDGE_MODE 진입 시 tb_knowledge에서 검색 주입
        # None이면 주입하지 않음 (tb_knowledge 미구성 환경/테스트)
        self._knowledge_retriever = knowledge_retriever
        # PromptAssembler에 뒤늦게 knowledge_retriever 바인딩 (생성자 순서 이슈 회피)
        self._prompt_assembler._knowledge_retriever = knowledge_retriever

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
        # TurnStateStore가 있으면 raw messages를 비우고 요약만 시스템 프롬프트에 싣는다
        # (Part 3 상태 외부화). 이 처리는 PromptAssembler의 _attach_turn_state 이전에
        # 수행해야 한다 — messages 히스토리 정리를 담당하는 건 QueryEngine이다.
        if self._turn_state_store is not None:
            self._messages.clear()

        # 사용자 메시지를 대화 히스토리에 추가
        user_msg = Message.user(user_input)
        self._messages.append(user_msg)

        logger.info(
            "메시지 제출: session=%s, messages=%d개, input='%s'",
            self._session_id, len(self._messages), user_input[:80],
        )

        # ─── 라우팅 결정 (RoutingResolver) ─────────────────
        tenant = self._context.options.get("tenant") if self._context else None
        decision = self._router.resolve(user_input, tenant)
        if decision.routing_enabled:
            logger.info(
                "라우팅: class=%s, model=%s, temp=%.2f, max_tokens=%s, tenant=%s",
                decision.query_class, decision.model_override,
                decision.temperature, decision.max_tokens_cap, decision.tenant_id,
            )
        else:
            logger.info("라우팅 비활성 — 프로바이더 기본 설정 사용")

        # ─── 시스템 프롬프트 조립 (PromptAssembler) ────────
        effective_system_prompt = await self._prompt_assembler.assemble(
            base_prompt=self._system_prompt,
            session_id=self._session_id,
            user_input=user_input,
            decision=decision,
        )

        # TurnState 저장 콜백 — query_loop이 턴 완료 시 호출
        def _on_turn_complete(turn_state: Any) -> None:
            if self._turn_state_store is not None:
                self._turn_state_store.save(self._session_id, turn_state)

        # ─── Dispatcher / query_loop 경유 ──────────────────
        if self._model_dispatcher is not None:
            stream = self._model_dispatcher.route(
                messages=self._messages,
                system_prompt=effective_system_prompt,
                on_turn_complete=_on_turn_complete,
                model_override=decision.model_override,
                temperature=decision.temperature,
                max_tokens_cap=decision.max_tokens_cap,
                enable_thinking=decision.enable_thinking,
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
                model_override=decision.model_override,
                temperature=decision.temperature,
                max_tokens_cap=decision.max_tokens_cap,
                enable_thinking=decision.enable_thinking,
            )

        # ─── 스트림 소비 ─────────────────────────────────
        # 이 try/finally 블록은 두 가지를 보장한다:
        #  1) 정상 종료든 예외든 _finalize_turn이 **반드시** 호출되어 트랜스크립트가
        #     최소 user 엔트리 한 줄이라도 남는다. 기존 구현은 async for가 예외로
        #     중단되면 `_finalize_turn`이 스킵되어 "폴더만 있고 transcript.jsonl
        #     없는 유령 세션"이 생겼다 (2026-04-23 hang 진단).
        #  2) 예외는 재-raise하여 상위(웹 핸들러)가 인지하게 한다. swallow 금지.
        finalize_error: BaseException | None = None
        try:
            async for event in stream:
                # 사용량 추적 — USAGE_UPDATE 이벤트에서 누적
                if (
                    isinstance(event, StreamEvent)
                    and event.type == StreamEventType.USAGE_UPDATE
                    and event.usage
                ):
                    self._cumulative_usage = event.usage

                yield event
        except BaseException as e:  # noqa: BLE001 — 정말로 모두 잡고 싶다
            # GeneratorExit/CancelledError 포함 — 클라이언트 연결 끊김에도 최소 기록
            finalize_error = e
            raise
        finally:
            self._total_turns += 1

            if finalize_error is None:
                logger.info(
                    "메시지 처리 완료: session=%s, 누적 턴=%d, "
                    "토큰=%d/%d (input/output)",
                    self._session_id,
                    self._total_turns,
                    self._cumulative_usage.input_tokens,
                    self._cumulative_usage.output_tokens,
                )
            else:
                logger.warning(
                    "메시지 처리 중단: session=%s, 누적 턴=%d, error=%s",
                    self._session_id,
                    self._total_turns,
                    type(finalize_error).__name__,
                )

            # Ch 16: 턴 종료 훅 — 세션 영속화
            # (1) MemoryManager.on_turn_end — Redis 단기 + tb_memories 승격
            # (2) Transcript.append_entry — JSONL 파일에 user/assistant 쌍 기록
            # 예외 중에도 호출되며 — finalize_error가 있으면 system 에러 엔트리도 남긴다.
            await self._finalize_turn(user_input, finalize_error=finalize_error)

    async def _finalize_turn(
        self,
        user_input: str,
        finalize_error: BaseException | None = None,
    ) -> None:
        """턴 종료 시 메모리/트랜스크립트 기록을 수행한다 (실패 시 swallow).

        Args:
            user_input: 이번 턴의 user 입력(원문)
            finalize_error: 스트림이 중단된 경우 그 예외. 전달되면 트랜스크립트에
                role="system" 에러 엔트리도 남겨 hang/에러 세션을 구분할 수 있게 한다.
        """
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
                # 스트림이 비정상 종료됐으면 system 에러 엔트리를 추가로 남긴다.
                # 이 엔트리가 있으면 "폴더만 있고 파일 없는 hang 세션"과 구분 가능.
                if finalize_error is not None:
                    self._transcript.append_entry(
                        role="system",
                        content=(
                            f"[stream aborted] {type(finalize_error).__name__}: "
                            f"{finalize_error}"
                        ),
                        turn=self._total_turns,
                        extra={"error_type": type(finalize_error).__name__},
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

    # ─── 요청 단위 바인딩 (2026-04-21 리팩토링) ───
    def bind_request(
        self,
        session_id: str,
        tenant: Any | None = None,
        transcript: Any | None = None,
        restore_messages: list[Message] | None = None,
    ) -> None:
        """
        한 HTTP 요청이 도착했을 때 QueryEngine을 해당 요청에 바인딩한다.

        웹 서버는 QueryEngine을 싱글톤처럼 공유하지만 session/tenant/transcript는
        요청마다 다르다. 예전엔 `web/app.py`가 `engine._session_id = ...` 식으로
        비공개 필드를 직접 덮어써서 race condition 위험이 있었다. 이 공식 메서드는
        그 패턴을 한 지점으로 모아 의도를 명확히 한다.

        **주의**: QueryEngine 인스턴스는 동시 요청에 대해 thread-safe가 아니다.
        현재 설계는 FastAPI/asyncio 단일 프로세스 순차 처리 가정이다. 진정한 동시
        요청 처리가 필요해지면 요청당 QueryEngine을 생성하는 팩토리 패턴으로
        옮겨야 한다 (TODO: core-analysis-specialist 2026-04-21 권고).

        Args:
            session_id: 이 요청의 세션 ID (Memory/Redis 키 정합성에 사용)
            tenant: TenantConfig 또는 None
            transcript: SessionTranscript 인스턴스 또는 None (세션별 동적 주입)
            restore_messages: 이 요청 시작 시 초기 메시지로 얹을 히스토리.
                None이면 기존 messages를 유지, [] 이면 clear.
        """
        self._session_id = session_id
        if tenant is not None and self._context is not None:
            self._context.options["tenant"] = tenant
            logger.debug(
                "tenant 바인딩: session=%s, tenant=%s",
                session_id,
                getattr(tenant, "id", "?"),
            )
        if transcript is not None:
            self._transcript = transcript
        if restore_messages is not None:
            self._messages.clear()
            self._messages.extend(restore_messages)

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
