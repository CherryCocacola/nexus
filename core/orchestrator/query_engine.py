"""
쿼리 엔진 — Tier 1 세션 오케스트레이터.

이 파일은 Nexus 대화 시스템의 "가장 바깥쪽 지휘자"다. 사용자가 메시지 하나를
보내면, 그 메시지를 받아서 아래 계층(query_loop/모델/도구)이 실제 일을 하도록
흐름을 열어 주고, 흘러나오는 이벤트를 웹/CLI 쪽으로 그대로 흘려보낸다.
Claude Code의 QueryEngine.ts를 Python(asyncio)으로 재구현한 것이다.

4-Tier AsyncGenerator 체인의 최상위(Tier 1) 레이어로서, 세션 단위의 대화를
관리하고 query_loop(Tier 2)를 호출한다. "세션"이란 한 사용자와의 연속된 대화
한 묶음을 뜻하며, QueryEngine 인스턴스 하나가 그 한 세션을 책임진다.

핵심 역할(이 클래스가 하는 일):
  1. 대화 히스토리(messages[]) 관리 — 지금까지 오간 메시지를 쌓아 둔다
  2. 시스템 프롬프트 조립 — PromptAssembler에 위임(RAG/지식/턴상태 주입 포함)
  3. 사용자 입력 전처리 및 라우팅 결정 — RoutingResolver에 위임
  4. query_loop()(또는 ModelDispatcher) 호출 → StreamEvent/Message yield
  5. 세션 사용량(토큰) 추적 및 턴 종료 시 메모리/트랜스크립트 영속화

4-Tier 체인에서의 위치(이벤트는 아래→위로 전파된다):
  Tier 1: QueryEngine.submit_message() ← 이 파일
  Tier 2: query_loop()          — while(True) 에이전트 턴 루프
  Tier 3: model_provider.stream() — SSE 스트림 파싱
  Tier 4: httpx 클라이언트        — 재시도 + 실제 HTTP 요청

주요 협력 객체(이 엔진이 조율만 하고 세부 로직은 이들이 담당):
  - RoutingResolver  : 쿼리 분류 + 샘플링 파라미터/모델 선택 결정
  - PromptAssembler  : 최종 시스템 프롬프트 조립(RAG/지식/턴상태 주입)
  - query_loop       : 실제 모델 호출 + 도구 실행 루프(폴백 경로)
  - ModelDispatcher  : Scout→Worker 2단계 멀티모델 경로(주입 시 우선)
  - MemoryManager    : 턴 종료 시 Redis 단기 저장 + 장기 승격
  - Transcript       : 턴 종료 시 JSONL 파일에 대화 영구 기록

의존성 방향(단방향 — 순환 import 금지):
  QueryEngine → query_loop, ContextManager, ModelProvider, BaseTool

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from core.config import ContextBudgetConfig, RoutingConfig
from core.message import (
    Message,
    StreamEvent,
    StreamEventType,
    TokenUsage,
)
from core.model.inference import ModelProvider, StructuredOutputSpec
from core.orchestrator.context_manager import ContextManager
from core.orchestrator.prompt_assembler import PromptAssembler
from core.orchestrator.query_loop import query_loop
from core.orchestrator.routing import (
    RoutingDecision,  # noqa: F401 — 하위 호환 export
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

    설계 메모(온보딩용):
      - 이 클래스는 "무엇을 할지"를 직접 계산하지 않는다. 대신 라우팅은
        RoutingResolver, 프롬프트 조립은 PromptAssembler, 실제 모델/도구
        실행은 query_loop/ModelDispatcher에게 위임하고 조율만 한다.
      - AsyncGenerator라서 return이 아니라 yield로 값을 흘려보낸다. 호출자는
        `async for`로 이벤트를 하나씩 받아 UI에 즉시 반영할 수 있다(스트리밍).
      - 스레드 세이프하지 않다. 웹 서버는 요청마다 bind_request()로 세션/
        트랜스크립트를 갈아끼워 재사용한다(하단 bind_request 주석 참조).

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
        todo_store: Any | None = None,
        rag_retriever: Any | None = None,
        model_dispatcher: Any | None = None,
        routing_config: RoutingConfig | None = None,
        memory_manager: Any | None = None,
        transcript: Any | None = None,
        knowledge_retriever: Any | None = None,
        # 컨텍스트 예산(하드코딩 외부화, 2026-07-03). None이면 PromptAssembler·
        # query_loop이 각자 현행 상수로 폴백 → 기존 호출부(테스트 포함) 동작 불변.
        context_budgets: ContextBudgetConfig | None = None,
        # 코딩 전용 모델 프로바이더(2026-08-05, 선택). 라우팅이 코딩 질의로 판정한
        # 턴에서만 model_provider 대신 이것을 쓴다. None이면 항상 기본 프로바이더를
        # 쓰므로 기존 동작과 동일하다(무회귀).
        coder_provider: ModelProvider | None = None,
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
        # 코딩 전용 프로바이더 — 없으면 코딩 라우팅 자체가 발생하지 않는다.
        self._coder_provider = coder_provider
        self._tools = tools
        self._context = context
        self._system_prompt = system_prompt
        self._context_manager = context_manager
        self._max_turns = max_turns
        self._hook_manager = hook_manager

        # v7.0: 턴 상태 외부화 저장소
        self._turn_state_store = turn_state_store

        # 계획 체크리스트 저장소(TodoWrite) — PromptAssembler가 매 턴 재주입한다.
        self._todo_store = todo_store

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
        # 컨텍스트 예산 보관 — None이면 하위(PromptAssembler/query_loop)가 현행
        # 상수로 폴백한다(무회귀). B200 티어는 bootstrap이 config 값을 주입한다.
        self._context_budgets = context_budgets
        # PromptAssembler에 RAG/프롬프트 예산을 전달. 예산이 없으면 None을 넘겨
        # PromptAssembler 내부 기본 상수(현행값)로 폴백시킨다.
        self._prompt_assembler = PromptAssembler(
            turn_state_store=turn_state_store,
            todo_store=todo_store,
            rag_retriever=rag_retriever,
            knowledge_retriever=None,  # 아래 setattr 이후 바인딩
            turn_state_tokens=(
                context_budgets.turn_state_tokens if context_budgets else None
            ),
            project_rag_tokens=(
                context_budgets.project_rag_tokens if context_budgets else None
            ),
            knowledge_rag_tokens=(
                context_budgets.knowledge_rag_tokens if context_budgets else None
            ),
        )
        # 출력 토큰 에스컬레이션 — query_loop/dispatcher로 넘길 값(없으면 None).
        self._output_token_escalation = (
            context_budgets.output_token_escalation if context_budgets else None
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

        # 진입점 채널(web/cli/api) — 히스토리 저장을 채널별로 격리하기 위한 태그.
        # 진입점이 context.options["channel"]에 심어둔 값을 읽는다(없으면 None=flat).
        # 요청마다 다를 수 있어 bind_request에서 덮어쓸 수 있다.
        _opts = getattr(context, "options", None)
        self._channel: str | None = _opts.get("channel") if isinstance(_opts, dict) else None

        logger.info(
            "QueryEngine 초기화: session=%s, tools=%d개, max_turns=%d",
            self._session_id,
            len(tools),
            max_turns,
        )

    async def submit_message(
        self,
        user_input: str,
        structured_output: StructuredOutputSpec | None = None,
        max_tokens_override: int | None = None,
        append_user_message: bool = True,
        forced_query_class: str | None = None,
        request_id: str | None = None,
    ) -> AsyncGenerator[StreamEvent | Message, None]:
        """
        사용자 메시지를 제출하고 스트리밍 응답을 반환한다.

        이 메서드가 4-Tier 체인의 진입점이다. 웹/CLI가 사용자 입력을 받으면
        이 코루틴을 `async for`로 순회하면서 흘러나오는 이벤트를 화면에 뿌린다.

        전체 처리 흐름(위에서 아래로):
          1. (턴상태 외부화 시) 기존 messages 비우기
          2. 사용자 입력을 Message로 변환해 히스토리에 추가
          3. RoutingResolver로 라우팅 결정(모델/샘플링 파라미터/클래스)
          4. PromptAssembler로 최종 시스템 프롬프트 조립
          5. ModelDispatcher(있으면) 또는 query_loop(폴백)로 스트림 시작
          6. 스트림을 순회하며 이벤트를 그대로 상위로 yield(사용량만 옆에서 추적)
          7. try/finally로 어떤 경우에도 _finalize_turn을 호출해 영속화 보장

        Args:
            user_input: 사용자 입력 텍스트
            max_tokens_override: 이번 호출의 출력 토큰 상한(선택, 2026-08-05).
                OpenAI 호환 API의 `max_tokens`를 실제로 반영하기 위한 통로다.
                이전에는 이 값이 어디에도 전달되지 않아 클라이언트가 출력 크기를
                제어할 수 없었고(실측: max_tokens=16 요청에 1086토큰 생성),
                큰 출력이 엔진 기본 예산에서 잘려도 원인을 알 수 없었다.
                None이면 종전대로 라우팅 결정값(decision.max_tokens_cap)을 쓴다.
            structured_output: 구조화 출력(guided decoding) 스펙(선택). 지정되면
                이번 호출의 응답이 지정 JSON Schema를 강제로 따른다. 세션 상태가
                아니라 "호출 단위 인자"로 받는다 — 세션에 붙이면 다음 턴까지 스키마가
                잔류해 일반 대화가 오염되기 때문이다(리스크 R8). dispatcher/폴백
                query_loop 양쪽 경로로 그대로 전달한다.
            append_user_message: user_input을 히스토리에 새 user 메시지로 추가할지
                여부(기본 True — 종전 동작). False로 주는 경우는 하나뿐이다.
                클라이언트가 도구를 실행하고 그 결과를 들고 다시 호출했을 때다.
                이때 대화는 `user → assistant(tool_calls) → tool(결과)` 로 끝나므로
                **새 사용자 발화가 없다**. 그런데도 user 메시지를 만들어 붙이면
                모델이 사용자가 같은 말을 두 번 한 것으로 읽어 답이 흔들린다.
                False일 때 user_input은 히스토리에 들어가지 않고 라우팅 판정과
                로그에만 쓰인다(라우팅은 "무엇을 하려는 대화인지"를 알아야 한다).

        Yields:
            StreamEvent: 스트리밍 이벤트 (UI 업데이트용 — 텍스트 델타/도구 등)
            Message: assistant/tool_result 메시지 (대화 기록용)
        """
        # TurnStateStore가 있으면 raw messages를 비우고 요약만 시스템 프롬프트에 싣는다
        # (Part 3 상태 외부화). 이 처리는 PromptAssembler의 _attach_turn_state 이전에
        # 수행해야 한다 — messages 히스토리 정리를 담당하는 건 QueryEngine이다.
        if self._turn_state_store is not None:
            self._messages.clear()

        # 사용자 메시지를 대화 히스토리에 추가.
        # (append_user_message=False면 건너뛴다 — 도구 결과로 이어 도는 호출이라
        #  새 사용자 발화가 없다. 자세한 이유는 위 Args 설명 참고.)
        if append_user_message:
            user_msg = Message.user(user_input)
            self._messages.append(user_msg)

        logger.info(
            "메시지 제출: session=%s, messages=%d개, input='%s'",
            self._session_id, len(self._messages), user_input[:80],
        )

        # ─── 프롬프트 인젝션 순응 차단 (2026-08-08) ────────
        # 모델을 부르기 **전에** 판정한다. 사후 경고로는 못 막기 때문이다 —
        # 순응 문자열 자체가 피해이고, CLI·스트리밍은 본문을 이미 흘려보낸 뒤라
        # 되돌릴 수도 없다. 판정 근거와 오탐 방지책은 injection_guard 문서 참조.
        # 웹·CLI·OpenAI 호환 경로가 모두 이 Tier 1 을 지나므로 여기 한 곳이면 된다.
        if append_user_message:
            refusal = self._injection_refusal(user_input)
            if refusal is not None:
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=refusal)
                assistant_msg = Message.assistant(refusal)
                self._messages.append(assistant_msg)
                yield assistant_msg
                return

        # ─── 라우팅 결정 (RoutingResolver) ─────────────────
        tenant = self._context.options.get("tenant") if self._context else None
        # 호출자가 클래스를 지정했으면 분류기를 건너뛴다(요청 단위 고정).
        decision = self._router.resolve(user_input, tenant, forced_class=forced_query_class)
        if decision.routing_enabled:
            logger.info(
                "라우팅: class=%s%s, model=%s, temp=%.2f, top_p=%.2f, "
                "rep_pen=%.2f, max_tokens=%s, tenant=%s",
                decision.query_class,
                "(고정)" if forced_query_class else "",
                decision.model_override,
                decision.temperature, decision.top_p,
                decision.repetition_penalty, decision.max_tokens_cap,
                decision.tenant_id,
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

        # 과거 기억 회상 — 관련 있는 장기 기억을 '시스템 프롬프트'에 덧붙인다.
        # ★messages 에 넣지 않는 이유(실측으로 드러난 버그): 회상 블록을
        #   Message.user() 로 히스토리에 넣었더니 턴 종료 시 그것이 '사용자 발화'로
        #   다시 저장되고, 저장된 것이 다음 턴에 또 회상돼 주입되는 되먹임이 생겼다
        #   ("--- 과거 기억 --- - --- 과거 기억 ---" 중첩이 실제로 관측됐다).
        #   회상은 대화가 아니라 참고 자료이므로 지식 RAG와 같은 자리(시스템 프롬프트)에
        #   둔다. 그러면 저장 대상 자체가 되지 않는다.
        recall_block = await self._recall_memories(user_input)
        if recall_block:
            effective_system_prompt = f"{effective_system_prompt}\n\n{recall_block}"

        # ─── 지식 RAG 출처 인용 (Point 4-2) ─────────────────
        # 프롬프트 조립 직후, 이번 턴에 KB로 '실제 주입된' 청크의 출처 목록이 있으면
        # KNOWLEDGE_SOURCES StreamEvent 1건을 먼저 흘려보낸다(4-Tier 체인 준수 —
        # 웹 핸들러 직참조 대신 이벤트로 전달). citation 비활성/주입 없음이면 목록이
        # 비어 있어 이벤트를 내지 않는다(무회귀). 미지 타입을 무시하는 기존 소비자에는
        # 하위 호환(신규 이벤트 추가일 뿐 기존 이벤트 수정 아님 — anti-patterns #3).
        knowledge_citations = getattr(
            self._prompt_assembler, "last_knowledge_citations", ()
        )
        if knowledge_citations:
            yield StreamEvent(
                type=StreamEventType.KNOWLEDGE_SOURCES,
                knowledge_sources=list(knowledge_citations),
            )

        # TurnState 저장 콜백 — query_loop이 턴 완료 시 호출
        def _on_turn_complete(turn_state: Any) -> None:
            if self._turn_state_store is not None:
                self._turn_state_store.save(self._session_id, turn_state)

        # ─── Dispatcher / query_loop 경유 ──────────────────
        # 두 갈래 중 하나로 스트림을 연다:
        #  - dispatcher 주입 O → Scout+Worker 멀티모델 경로(route)
        #  - dispatcher 주입 X → 단일 Worker 폴백 경로(query_loop)
        # 어느 쪽이든 반환값 stream은 StreamEvent/Message를 내보내는
        # AsyncGenerator라서, 아래 소비 루프는 경로를 신경 쓰지 않아도 된다.
        # 코딩 질의로 판정됐고 코딩 프로바이더가 배선돼 있으면 그것을 쓴다.
        # 둘 중 하나라도 없으면 기본 프로바이더 — 즉 기본 동작은 변하지 않는다.
        active_provider = self._model_provider
        if decision.use_coder and self._coder_provider is not None:
            active_provider = self._coder_provider
            logger.info("라우팅: 코딩 전용 모델로 전환 (class=%s)", decision.query_class)

        if self._model_dispatcher is not None:
            stream = self._model_dispatcher.route(
                # 코딩 턴에만 프로바이더를 갈아끼운다(None이면 dispatcher 기본 Worker).
                provider_override=(
                    active_provider if active_provider is not self._model_provider else None
                ),
                messages=self._messages,
                system_prompt=effective_system_prompt,
                # 프롬프트 덤프 키 — 진단이 켜져 있을 때만 쓰인다.
                # 요청 ID 가 없으면(웹 UI·CLI) 세션 ID 로 떨어진다.
                request_id=request_id,
                session_id=self._session_id,
                on_turn_complete=_on_turn_complete,
                model_override=decision.model_override,
                temperature=decision.temperature,
                # 호출 단위 override(OpenAI max_tokens)가 있으면 우선한다.
                max_tokens_cap=max_tokens_override or decision.max_tokens_cap,
                enable_thinking=decision.enable_thinking,
                # 라우팅이 결정한 샘플링 파라미터를 dispatcher 경로로 전달.
                top_p=decision.top_p,
                repetition_penalty=decision.repetition_penalty,
                frequency_penalty=decision.frequency_penalty,
                presence_penalty=decision.presence_penalty,
                # 출력 토큰 에스컬레이션 단계(config 값, None이면 상수 폴백).
                output_token_escalation=self._output_token_escalation,
                # 구조화 출력 스펙(호출 단위 인자). None이면 일반 경로.
                structured_output=structured_output,
                # 자기일관성(SC) 파라미터 — 게이트 미통과 시 sc_n=1(비활성, 무회귀).
                sc_n=decision.sc_n,
                sc_min_agreement=decision.sc_min_agreement,
                sc_short_answer_max_chars=decision.sc_short_answer_max_chars,
                sc_similarity_threshold=decision.sc_similarity_threshold,
            )
        else:
            # 폴백 — dispatcher 주입이 없는 경우 기존 단일 Worker 경로
            stream = query_loop(
                messages=self._messages,
                system_prompt=effective_system_prompt,
                # 프롬프트 덤프 키 — 진단이 켜져 있을 때만 쓰인다.
                # 요청 ID 가 없으면(웹 UI·CLI) 세션 ID 로 떨어진다.
                request_id=request_id,
                session_id=self._session_id,
                model_provider=active_provider,
                tools=self._tools,
                context=self._context,
                context_manager=self._context_manager,
                max_turns=self._max_turns,
                on_turn_complete=_on_turn_complete,
                model_override=decision.model_override,
                temperature=decision.temperature,
                # 호출 단위 override(OpenAI max_tokens)가 있으면 우선한다.
                max_tokens_cap=max_tokens_override or decision.max_tokens_cap,
                enable_thinking=decision.enable_thinking,
                # 라우팅이 결정한 샘플링 파라미터를 query_loop 폴백 경로로 전달.
                top_p=decision.top_p,
                repetition_penalty=decision.repetition_penalty,
                frequency_penalty=decision.frequency_penalty,
                presence_penalty=decision.presence_penalty,
                # 출력 토큰 에스컬레이션 단계(config 값, None이면 상수 폴백).
                output_token_escalation=self._output_token_escalation,
                # 구조화 출력 스펙(호출 단위 인자). None이면 일반 경로.
                structured_output=structured_output,
                # 자기일관성(SC) 파라미터 — 게이트 미통과 시 sc_n=1(비활성, 무회귀).
                sc_n=decision.sc_n,
                sc_min_agreement=decision.sc_min_agreement,
                sc_short_answer_max_chars=decision.sc_short_answer_max_chars,
                sc_similarity_threshold=decision.sc_similarity_threshold,
            )

        # ─── 스트림 소비 ─────────────────────────────────
        # 이 try/finally 블록은 두 가지를 보장한다:
        #  1) 정상 종료든 예외든 _finalize_turn이 **반드시** 호출되어 트랜스크립트가
        #     최소 user 엔트리 한 줄이라도 남는다. 기존 구현은 async for가 예외로
        #     중단되면 `_finalize_turn`이 스킵되어 "폴더만 있고 transcript.jsonl
        #     없는 유령 세션"이 생겼다 (2026-04-23 hang 진단).
        #  2) 예외는 재-raise하여 상위(웹 핸들러)가 인지하게 한다. swallow 금지.
        # finalize_error에 예외를 담아 finally까지 전달한다 — finally 블록이
        # "정상 종료였는지 / 중단됐는지"를 구분해 로그·트랜스크립트를 다르게
        # 남기기 위해서다. None이면 정상 종료를 의미한다.
        finalize_error: BaseException | None = None
        try:
            # 하위 Tier(stream)가 내보내는 이벤트를 그대로 상위(웹/CLI)로 흘려보낸다.
            # QueryEngine은 이벤트를 가공하지 않고 통과시키되, 토큰 사용량만 옆에서
            # 누적해 둔다(관측용).
            async for event in stream:
                # 사용량 추적 — USAGE_UPDATE 이벤트가 올 때마다 최신값으로 갱신.
                # (누적이 아니라 최신 스냅샷을 그대로 보관 — 하위에서 이미 누적해 보냄)
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

    def _memory_owner(self) -> str | None:
        """이번 대화의 기억 소유자(테넌트) 식별자.

        회상·저장 양쪽에서 같은 값을 써야 "내가 저장한 것만 내가 회상"이 성립한다.
        테넌트를 알 수 없으면 None — 그 경우 저장물에 소유자 표식이 없어 회상에서
        제외된다(fail-closed). 개인정보가 남의 대화로 새는 것보다 회상이 안 되는
        편이 낫다.
        """
        tenant = self._context.options.get("tenant") if self._context else None
        if tenant is None:
            return None
        # 테넌트는 객체(.id)일 수도, 이미 문자열일 수도 있다.
        owner = getattr(tenant, "id", tenant)
        return str(owner) if owner else None

    def _injection_refusal(self, user_input: str) -> str | None:
        """프롬프트 인젝션 순응 시도면 거절문을, 아니면 None 을 돌려준다.

        판정은 `core.verification.injection_guard` 가 한다. 여기서는 호출과 로깅만
        맡는다 — 웹·CLI 가 같은 판정을 받아야 하고, 판정 규칙이 늘어도 이 자리는
        그대로여야 하기 때문이다.

        fail-soft: 가드가 예외를 내면 차단하지 않고 통과시킨다. 보호 장치가 정상
        대화를 막는 쪽이 더 나쁘다. 다만 그 사실은 로그로 남긴다.
        """
        try:
            from core.verification.injection_guard import (
                build_injection_refusal,
                find_dictated_compliance,
            )

            finding = find_dictated_compliance(user_input)
            if finding is None:
                return None
            logger.warning(
                "프롬프트 인젝션 차단: session=%s, 받아쓰기='%s', 탈취어구='%s'",
                self._session_id, finding.dictated[:40], finding.override_hit,
            )
            return build_injection_refusal(finding)
        except Exception as e:  # noqa: BLE001 — 가드 오류가 대화를 막지 않게 한다
            logger.warning("인젝션 가드 실패(통과시킴): %s", e)
            return None

    async def _recall_memories(self, user_input: str) -> str:
        """이번 입력과 관련된 과거 기억을 찾아 주입용 블록으로 만든다.

        [왜 기본 비활성인가]
          2026-08-06 실측 전까지 이 회상 훅(MemoryManager.on_turn_start)은 프로덕션
          어디서도 호출되지 않았다 — 장기 기억은 '쓰기 전용'이었다. 이제 배선했지만,
          매 턴 모든 응답에 과거 기억이 끼어드는 것은 큰 동작 변화라 설정으로 켜게 둔다
          (config: memory.recall_enabled, 기본 false = 기존 동작 그대로 = 무회귀).

        [무엇을 주입하나]
          코드 RAG 청크는 MemoryManager 쪽에서 이미 걸러진다(is_rag_chunk). 여기서는
          너무 긴 기억이 컨텍스트를 잡아먹지 않도록 건수·길이만 제한한다.

        실패는 삼킨다 — 회상은 부가 기능이고, 실패했다고 대화가 막혀선 안 된다.
        """
        if self._memory_manager is None:
            return ""

        # 설정은 값으로 주입받는다(bootstrap/web이 options에 넣어 준다).
        # permission_enforcement 등 기존 주입 패턴과 같은 모양이라 시그니처 변경이 없다.
        recall_cfg = (self._context.options.get("memory_recall") if self._context else None) or {}
        if not recall_cfg.get("enabled", False):
            return ""

        try:
            entries = await self._memory_manager.on_turn_start(
                session_id=self._session_id,
                user_message=user_input,
                owner=self._memory_owner(),
            )
        except Exception as e:  # noqa: BLE001 — 회상 실패가 대화를 막지 않게
            logger.warning("메모리 회상 실패 (session=%s): %s", self._session_id, e)
            return ""

        max_items = recall_cfg.get("max_items", 5)
        max_chars = recall_cfg.get("max_chars", 400)
        lines = [
            f"- {(e.content or '').strip()[:max_chars]}"
            for e in entries[:max_items]
            if (e.content or "").strip()
        ]
        if not lines:
            return ""

        logger.info(
            "메모리 회상: session=%s, 주입 %d건", self._session_id, len(lines)
        )
        # 참고 자료임을 분명히 해, 모델이 이걸 사용자 발화로 오해하지 않게 한다.
        return (
            "--- 과거 기억 (참고용, 사용자가 방금 한 말이 아님) ---\n"
            + "\n".join(lines)
            + "\n관련 있을 때만 활용하고, 무관하면 무시하라."
        )

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
                    channel=self._channel,
                    owner=self._memory_owner(),
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
                # user 쪽은 이번 턴 입력 원문을 그대로 쓴다.
                last_user: str | None = user_input
                last_assistant: str | None = None
                # assistant 쪽은 messages를 끝에서부터 거슬러 올라가며 가장 최근
                # assistant 텍스트를 찾는다. role이 enum/문자열 둘 다일 수 있어
                # .value 유무를 방어적으로 처리하고, text_content 속성이 없으면
                # content를 문자열화해 폴백한다(메시지 구현 차이 흡수).
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
    # 아래 property들은 모두 내부 상태(_messages, _session_id 등)의 단순 조회용이다.
    # 외부(웹 핸들러, 메트릭 엔드포인트, 테스트)가 엔진 내부를 들여다볼 수 있게
    # 하되, 직접 _필드를 만지지 않도록 읽기 창구를 제공한다.

    @property
    def messages(self) -> list[Message]:
        """현재 대화 히스토리를 반환한다 (읽기 전용 복사).

        내부 리스트를 그대로 주면 호출자가 실수로 수정할 수 있으므로,
        list(...)로 얕은 복사본을 만들어 내부 상태를 보호한다.
        """
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
        channel: str | None = None,
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
            channel: 진입점 채널(web/cli/api). 지정되면 이 요청의 히스토리 저장을
                해당 채널로 격리한다(None이면 기존 채널 유지 — 덮어쓰지 않음).
        """
        self._session_id = session_id
        if channel is not None:
            self._channel = channel
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
        """마지막 assistant 메시지의 텍스트를 반환한다.

        히스토리를 끝에서부터 거슬러 올라가며 처음 만나는 assistant 메시지의
        텍스트를 돌려준다. assistant 응답이 아직 없으면 빈 문자열을 반환한다
        (호출부에서 None 체크 없이 바로 쓸 수 있게 한 안전한 기본값).
        """
        for msg in reversed(self._messages):
            role = str(msg.role)
            if role == "assistant":
                return msg.text_content
        return ""
