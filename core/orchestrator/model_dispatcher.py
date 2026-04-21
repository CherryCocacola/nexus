"""
모델 디스패처 — Worker 모델 실행 래퍼.

v7.0 Phase 9 재설계 (B 방식):
  이 모듈은 더 이상 Scout 자동 전처리를 수행하지 않는다.
  Scout는 이제 "자동 전처리기"가 아니라 "Worker가 필요할 때 호출하는 서브에이전트"이며,
  AgentTool + AgentDefinition(core/orchestrator/agent_definition.py)으로 처리된다.

현재 Dispatcher의 역할:
  - 티어 정보 보관 (TIER_S/M/L)
  - Worker 도구/프로바이더를 QueryEngine에 전달하기 위한 래퍼
  - route()는 단순히 query_loop을 호출하는 passthrough

왜 Dispatcher를 완전히 제거하지 않는가:
  - QueryEngine이 `model_dispatcher`를 받는 인터페이스는 그대로 유지해야 함
  - 향후 티어별 차등 로직(예: TIER_L 병렬 Worker)을 얹기 위한 확장점
  - scout_provider/scout_tools는 AgentTool이 context.options에서 가져가므로
    Dispatcher가 이를 '보관'할 필요는 사라졌음 — 인자만 수용하고 무시

하위 호환 Scout 통계 필드:
  stats 프로퍼티는 기존 키(scout_calls 등)를 계속 노출하되 값은 항상 0이다.
  실제 Scout 호출 통계는 AgentTool.get_stats()가 담당한다.
  /metrics 엔드포인트는 이번 Phase 이후 AgentTool.get_stats()를 참조한다.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from core.message import Message, StreamEvent
from core.model.hardware_tier import HardwareTier
from core.model.inference import ModelProvider
from core.orchestrator.query_loop import query_loop
from core.tools.base import BaseTool, ToolUseContext

logger = logging.getLogger("nexus.orchestrator.model_dispatcher")


class ModelDispatcher:
    """
    Worker 모델 실행 래퍼.

    이전 버전의 자동 Scout 전처리 로직은 제거되었다.
    Scout는 이제 AgentTool을 통해 Worker가 호출하는 서브에이전트다.

    route()는 항상 Worker query_loop으로 직행한다 — TIER 구분 없이 passthrough.
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

        Args:
            tier: 하드웨어 티어 (TIER_S, TIER_M, TIER_L) — 관측/메트릭 용도
            worker_provider: Worker 모델 프로바이더
            worker_tools: Worker에 할당할 도구 리스트
            context: 도구 실행 컨텍스트
            scout_provider: Scout 프로바이더 (AgentTool이 context.options에서
                직접 꺼내 쓰므로 여기서는 보관만 한다. 하위 호환용)
            scout_tools: Scout 전용 도구 (동일 이유로 보관만)
            system_prompt: 기본 시스템 프롬프트
            max_turns: Worker 최대 턴 수
        """
        self._tier = tier
        self._worker_provider = worker_provider
        self._worker_tools = worker_tools
        self._context = context
        self._scout_provider = scout_provider
        self._scout_tools = scout_tools or []
        self._system_prompt = system_prompt
        self._max_turns = max_turns

        # Scout가 "사용 가능한 환경"인지 관측용 플래그만 남긴다.
        # 실제 호출 여부는 Worker(Qwen)가 AgentTool로 판단한다.
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
    ) -> AsyncGenerator[StreamEvent | Message, None]:
        """
        Worker query_loop으로 직행한다 (passthrough).

        이전에는 TIER_S에서 Scout를 먼저 호출했으나, v7.0 Phase 9 재설계 이후
        Scout는 Worker가 AgentTool로 호출하는 서브에이전트가 되었다.
        Dispatcher는 더 이상 자동 전처리를 하지 않는다.

        v7.0 Part 2.5 (2026-04-21): QueryEngine이 쿼리 타입별로 결정한
        model_override/temperature/max_tokens_cap/enable_thinking을 그대로
        query_loop에 전달한다. Dispatcher 자체는 라우팅에 관여하지 않는다.
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
