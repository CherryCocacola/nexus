"""
모델 디스패처 — Worker 모델 실행을 감싸는 얇은 래퍼 모듈.

■ 이 파일 한눈에 보기 (온보딩용)
  Nexus의 상위 오케스트레이터인 QueryEngine(Tier 1)은 실제 모델 추론을 돌릴 때
  이 ModelDispatcher.route()를 호출한다. route()는 내부적으로 곧장
  query_loop(Tier 2, core/orchestrator/query_loop.py)으로 이벤트를 흘려보내는
  "통과(passthrough)" 역할만 한다. 즉 Dispatcher는 지금 시점에서는 라우팅 판단을
  하지 않는 매우 얇은 계층이며, 4-Tier AsyncGenerator 체인을 우회하지 않고
  그대로 이어 붙이는 접착제(glue)에 가깝다.

■ 왜 "디스패처"라는 이름인데 아무것도 분기하지 않는가 (히스토리)
  v7.0 Phase 9 재설계(B 방식) 이전에는 이 모듈이 TIER_S 환경에서 Scout(경량 보조
  모델)를 먼저 자동으로 돌려 질의를 전처리한 뒤 Worker에 넘기는 "자동 전처리기"였다.
  재설계 이후 Scout는 자동 전처리기가 아니라 "Worker가 필요할 때 스스로 호출하는
  서브에이전트"로 바뀌었고, 그 호출은 AgentTool + AgentDefinition
  (core/orchestrator/agent_definition.py)이 담당한다. 그 결과 Dispatcher에서
  자동 분기 로직이 통째로 빠졌고, 지금은 Worker query_loop으로 직행만 한다.

■ 현재 Dispatcher가 실제로 하는 일
  - 하드웨어 티어 정보(TIER_S/M/L)를 보관해 관측/메트릭에 노출한다.
  - Worker 프로바이더/도구/컨텍스트/시스템 프롬프트/최대 턴 수를 붙들고 있다가
    route() 호출 시 query_loop에 그대로 전달한다.
  - route()는 query_loop이 내보내는 StreamEvent/Message를 그대로 재yield한다.

■ 왜 굳이 제거하지 않고 남겨 두는가 (설계 의도)
  - QueryEngine이 `model_dispatcher`라는 협력 객체를 받는 인터페이스를 유지해야 하고,
    이를 없애면 상위 호출부를 크게 뜯어고쳐야 한다.
  - 향후 티어별 차등 라우팅(예: TIER_L에서 병렬 Worker 실행)을 얹을 확장점이다.
  - scout_provider/scout_tools 인자는 AgentTool이 context.options에서 직접 꺼내
    쓰므로 Dispatcher가 보관할 실익은 사라졌다. 그래도 하위 호환을 위해 생성자에서
    "인자로 받아 두기만" 하고 route()에서는 사용하지 않는다.

■ 하위 호환용 Scout 통계 필드
  stats 프로퍼티는 예전 대시보드/테스트가 참조하던 키(scout_calls 등)를 계속
  노출하되, Dispatcher가 더 이상 Scout를 돌리지 않으므로 값은 항상 0이다.
  살아있는 실제 수치는 AgentTool.get_stats()["scout"]에서 집계되며, /metrics
  엔드포인트도 이번 Phase 이후 그쪽을 참조한다.

■ 주요 구성 요소
  - class ModelDispatcher: 생성자에서 의존성을 붙들고, route()로 query_loop에 위임.
  - 관측 프로퍼티: tier / scout_enabled / worker_provider / worker_tools / stats.

■ 의존 모듈
  - core.orchestrator.query_loop.query_loop (Tier 2 진입점)
  - core.message (Message, StreamEvent), core.model.inference (ModelProvider)
  - core.model.hardware_tier (HardwareTier), core.tools.base (BaseTool, ToolUseContext)

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from core.message import Message, StreamEvent
from core.model.hardware_tier import HardwareTier
from core.model.inference import ModelProvider, StructuredOutputSpec
from core.orchestrator.query_loop import query_loop
from core.tools.base import BaseTool, ToolUseContext

logger = logging.getLogger("nexus.orchestrator.model_dispatcher")


class ModelDispatcher:
    """
    Worker 모델 실행을 감싸는 래퍼 클래스.

    역할:
      QueryEngine(Tier 1)이 붙들고 있는 협력 객체로, 실제 추론이 필요할 때
      route()를 호출받아 query_loop(Tier 2)으로 넘긴다. 생성자에서 Worker
      프로바이더/도구/컨텍스트 등 실행에 필요한 의존성을 미리 붙들어 둔다.

    왜 필요한가:
      QueryEngine과 query_loop 사이의 얇은 접착 계층이다. 지금은 단순
      passthrough지만, 티어별 차등 라우팅을 얹을 수 있는 확장점으로 남겨 둔다.

    핵심 흐름:
      __init__(의존성 보관) → route(messages, ...) 호출 → query_loop 위임 →
      query_loop이 내보내는 StreamEvent/Message를 그대로 재yield.

    히스토리:
      이전 버전의 자동 Scout 전처리 로직은 제거되었다. Scout는 이제 AgentTool을
      통해 Worker가 스스로 호출하는 서브에이전트이며, Dispatcher는 관여하지 않는다.
      route()는 항상 Worker query_loop으로 직행한다 — TIER 구분 없는 passthrough.
      TIER_L에서 병렬 Worker 등 고급 라우팅이 필요해지면 이 위치에 확장한다.
    """

    def __init__(
        self,
        tier: HardwareTier,
        worker_provider: ModelProvider,
        worker_tools: list[BaseTool],
        context: ToolUseContext,
        scout_provider: ModelProvider | None = None,
        scout_tools: list[BaseTool] | None = None,
        system_prompt: str = "",
        max_turns: int = 200,
    ) -> None:
        """
        ModelDispatcher를 초기화한다.

        생성자는 route()가 나중에 쓸 의존성들을 인스턴스 필드에 붙들어 두는 역할만
        한다. 여기서 모델을 호출하거나 네트워크를 타는 일은 전혀 없다.

        Args:
            tier: 하드웨어 티어 (TIER_S, TIER_M, TIER_L). 라우팅 판단에는 쓰지
                않고 관측/메트릭 노출 용도로만 보관한다.
            worker_provider: Worker 모델 프로바이더. route()에서 query_loop의
                model_provider 인자로 그대로 전달된다.
            worker_tools: Worker에 할당할 도구 리스트.
            context: 도구 실행 컨텍스트(ToolUseContext). AgentTool은 이 안의
                options에서 Scout 관련 값을 직접 꺼내 쓴다.
            scout_provider: Scout 프로바이더. AgentTool이 context.options에서
                직접 꺼내 쓰므로 여기서는 보관만 한다(하위 호환용, route에서 미사용).
            scout_tools: Scout 전용 도구. 동일한 이유로 보관만 한다.
            system_prompt: 기본 시스템 프롬프트. 참고로 route()는 인자로 받은
                system_prompt를 우선 쓰므로 이 필드는 보관용 기본값에 가깝다.
            max_turns: Worker query_loop의 최대 턴 수 상한.
        """
        # Worker 관련 의존성 — route()가 실제로 query_loop에 넘겨 사용하는 값들이다.
        self._tier = tier
        self._worker_provider = worker_provider
        self._worker_tools = worker_tools
        self._context = context
        # Scout 관련 — 현재는 붙들어 두기만 한다(하위 호환). 실제 Scout 호출은
        # Worker가 AgentTool을 통해 context.options에서 직접 꺼내 수행하므로
        # Dispatcher는 이 두 필드를 route()에서 참조하지 않는다.
        self._scout_provider = scout_provider
        # scout_tools가 None이면 빈 리스트로 정규화해 이후 len() 등에서 안전하게.
        self._scout_tools = scout_tools or []
        self._system_prompt = system_prompt
        self._max_turns = max_turns

        # Scout 서버가 "연결 가능한 환경"인지 나타내는 관측용 불리언 플래그.
        # provider가 주입됐으면 True. 다만 True라도 실제로 Scout를 부를지는
        # Worker(Qwen)가 AgentTool로 판단하며, Dispatcher는 자동 호출하지 않는다.
        self._scout_available = scout_provider is not None

        logger.info(
            "ModelDispatcher 초기화: tier=%s, scout_available=%s, worker_tools=%d개",
            tier.value,
            self._scout_available,
            len(worker_tools),
        )

    async def route(
        self,
        messages: list[Message],
        system_prompt: str,
        on_turn_complete: Any | None = None,
        model_override: str | None = None,
        temperature: float = 0.7,
        max_tokens_cap: int | None = None,
        enable_thinking: bool = False,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        # 출력 토큰 에스컬레이션 단계(하드코딩 외부화). None이면 query_loop이
        # 모듈 상수로 폴백 → 무회귀. QueryEngine이 config 값을 넘겨준다.
        output_token_escalation: list[int] | None = None,
        # 구조화 출력 스펙(guided decoding). None이면 일반 경로(무회귀). 라우팅
        # 판단과 무관하게 query_loop으로 그대로 통과시킨다(passthrough).
        structured_output: StructuredOutputSpec | None = None,
    ) -> AsyncGenerator[StreamEvent | Message, None]:
        """
        Worker query_loop으로 직행하는 비동기 제너레이터 (passthrough).

        동작:
          내부에서 query_loop(...)을 호출하고, 그것이 yield하는 StreamEvent 또는
          Message를 하나씩 그대로 다시 yield한다. 즉 이 메서드는 이벤트 스트림을
          가공하지 않고 상위(QueryEngine)로 흘려보내는 파이프 역할이다. 4-Tier
          체인 규칙상 Tier를 건너뛰지 않고 바로 아래 Tier(query_loop)만 소비한다.

        히스토리:
          이전에는 TIER_S에서 Scout를 먼저 호출해 전처리했으나, v7.0 Phase 9
          재설계 이후 Scout는 Worker가 AgentTool로 호출하는 서브에이전트가 되었다.
          그래서 Dispatcher는 더 이상 자동 전처리/분기를 하지 않는다.

          v7.0 Part 2.5 (2026-04-21): 샘플링·라우팅 파라미터
          (model_override/temperature/max_tokens_cap/enable_thinking 등)는
          QueryEngine이 쿼리 타입별로 미리 결정해 넘겨주며, route()는 이를 만들지
          않고 query_loop에 그대로 전달만 한다. 라우팅 판단은 QueryEngine 몫이다.

        Args:
            messages: 지금까지의 대화 메시지 리스트(Message).
            system_prompt: 이번 실행에 쓸 시스템 프롬프트. 생성자에 보관된
                self._system_prompt가 아니라 호출자가 넘긴 이 값을 사용한다.
            on_turn_complete: 각 턴 완료 시 호출될 콜백(선택). query_loop에 전달.
            model_override: 특정 모델을 강제 지정할 때(선택). None이면 기본 모델.
            temperature: 샘플링 온도.
            max_tokens_cap: 출력 토큰 상한(선택).
            enable_thinking: thinking 모드 활성화 여부.
            top_p / repetition_penalty / frequency_penalty / presence_penalty:
                샘플링 세부 파라미터. QueryEngine이 정한 값을 그대로 통과시킨다.
            output_token_escalation: 출력 토큰 에스컬레이션 단계 리스트(선택).
                None이면 query_loop이 모듈 상수로 폴백하므로 기존 동작과 동일하다.
            structured_output: 구조화 출력 스펙(선택). None이면 일반 경로. Dispatcher는
                이 값을 만들지 않고 query_loop으로 그대로 통과시킨다.

        Yields:
            StreamEvent | Message: query_loop이 산출하는 스트리밍 이벤트/메시지.
        """
        async for event in query_loop(
            messages=messages,
            system_prompt=system_prompt,
            model_provider=self._worker_provider,
            tools=self._worker_tools,
            context=self._context,
            max_turns=self._max_turns,
            on_turn_complete=on_turn_complete,
            model_override=model_override,
            temperature=temperature,
            max_tokens_cap=max_tokens_cap,
            enable_thinking=enable_thinking,
            # 샘플링 파라미터 passthrough — Dispatcher는 값을 만들지 않고
            # QueryEngine이 라우팅으로 결정한 값을 query_loop로 그대로 넘긴다.
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            # 출력 토큰 에스컬레이션 단계 passthrough (None이면 상수 폴백).
            output_token_escalation=output_token_escalation,
            # 구조화 출력 스펙 passthrough (None이면 일반 경로).
            structured_output=structured_output,
        ):
            yield event

    # ─── 관측 프로퍼티 ───────────────────────────────
    @property
    def tier(self) -> HardwareTier:
        """현재 하드웨어 티어를 반환한다."""
        return self._tier

    @property
    def scout_enabled(self) -> bool:
        """
        Scout 서버가 연결된 환경인지 여부.

        이전 버전의 "Scout 자동 실행" 의미는 사라졌다.
        값이 True여도 Scout를 실제로 호출할지는 Worker가 판단한다.
        """
        return self._scout_available

    @property
    def worker_provider(self) -> ModelProvider:
        """Worker 모델 프로바이더를 반환한다."""
        return self._worker_provider

    @property
    def worker_tools(self) -> list[BaseTool]:
        """Worker 도구 리스트를 반환한다."""
        return self._worker_tools

    @property
    def stats(self) -> dict[str, Any]:
        """
        하위 호환 Scout 통계 — 실제 수치는 항상 0이다.

        Scout 자동 전처리가 제거됐으므로 Dispatcher에는 누적할 호출이 없다.
        Scout 호출 통계는 이제 AgentTool.get_stats()["scout"]에서 집계된다.
        /metrics 엔드포인트는 Phase 5에서 AgentTool.get_stats()를 참조하도록
        변경된다.

        기존 사용처(테스트/대시보드)가 갑자기 깨지지 않도록 키 목록은 유지한다.
        """
        return {
            "tier": self._tier.value,
            "scout_enabled": self._scout_available,
            "scout_calls": 0,
            "scout_fallback_count": 0,
            "scout_fallbacks": 0,  # 하위 호환 alias
            "scout_avg_latency_ms": 0.0,
            "note": "Scout is now invoked by the Worker via AgentTool; "
            "see AgentTool.get_stats() for live numbers.",
        }
