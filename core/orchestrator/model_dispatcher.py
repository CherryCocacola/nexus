"""
모델 디스패처 — 하드웨어 티어에 따라 Scout/Worker를 분배한다.

v7.0 핵심 모듈: 4-Tier 체인의 Tier 1(QueryEngine) 내부에서
모델 라우팅을 담당한다.

동작 방식:
  TIER_S: Scout(CPU 4B) → Worker(GPU 31B) 2단계 실행
  TIER_M/L: Worker(GPU) 직행 (v6.1 동일, passthrough)

Scout → Worker 핸드오프 흐름:
  1. Scout가 읽기 전용 도구(Read, Glob, Grep, LS)로 파일 탐색
  2. Scout 결과(텍스트)에서 관련 파일과 계획을 추출
  3. Worker에게 Scout 결과를 시스템 프롬프트로 전달
  4. Worker가 실행 도구(Edit, Write, Bash 등)로 작업 수행
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from core.message import Message, StreamEvent, StreamEventType
from core.model.hardware_tier import HardwareTier, get_tier_config
from core.model.inference import ModelProvider
from core.orchestrator.query_loop import query_loop
from core.tools.base import BaseTool, ToolUseContext

logger = logging.getLogger("nexus.orchestrator.model_dispatcher")

# Scout 전용 시스템 프롬프트
# Scout는 읽기 전용 도구만 사용하여 탐색과 계획만 수행한다.
SCOUT_SYSTEM_PROMPT = (
    "You are Scout, a read-only exploration agent.\n"
    "Your job is to explore files and make a plan. You must NOT modify any files.\n\n"
    "Steps:\n"
    "1. Use your tools (Read, Glob, Grep, LS) to find relevant files\n"
    "2. Read key files to understand the context\n"
    "3. At the end, summarize what you found and suggest a plan\n\n"
    "Keep your responses concise. Focus on facts, not opinions.\n"
    "Respond in the user's language."
)


class ModelDispatcher:
    """
    하드웨어 티어에 따라 Scout/Worker를 분배하는 디스패처.

    TIER_S: Scout(탐색) → Worker(실행) 2단계
    TIER_M/L: Worker 단독 (v6.1 동일, passthrough)
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
            tier: 하드웨어 티어 (TIER_S, TIER_M, TIER_L)
            worker_provider: Worker(31B) 모델 프로바이더 (GPU)
            worker_tools: Worker에 할당할 도구 리스트
            context: 도구 실행 컨텍스트
            scout_provider: Scout(4B) 모델 프로바이더 (CPU, TIER_S 전용)
            scout_tools: Scout에 할당할 읽기 전용 도구 (TIER_S 전용)
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

        tier_cfg = get_tier_config(tier)
        self._scout_enabled = tier_cfg["scout_enabled"] and scout_provider is not None

        # Scout 실행 통계 — Ch 17 SessionMetrics 노출 대상
        # scout_total_latency_ms는 평균 지연 시간 계산용 누계값이다.
        self._scout_calls: int = 0
        self._scout_fallbacks: int = 0
        self._scout_total_latency_ms: float = 0.0

        logger.info(
            "ModelDispatcher 초기화: tier=%s, scout=%s, worker_tools=%d개",
            tier.value,
            "활성" if self._scout_enabled else "비활성",
            len(worker_tools),
        )

    async def route(
        self,
        messages: list[Message],
        system_prompt: str,
        on_turn_complete: Any | None = None,
    ) -> AsyncGenerator[StreamEvent | Message, None]:
        """
        사용자 메시지를 적절한 모델에 라우팅한다.

        TIER_S (Scout 활성):
          1. Scout에게 탐색/계획 요청 (읽기 전용 도구)
          2. Scout 결과를 Worker 시스템 프롬프트에 주입
          3. Worker에게 실행 지시
          → Scout 실패 시 Worker 직행으로 fallback

        TIER_M/L (passthrough):
          Worker에게 직접 전달 (v6.1 동일 경로)
        """
        if self._scout_enabled:
            # Scout 탐색 실행
            scout_result = await self._run_scout(messages)

            if scout_result:
                # Scout 결과를 Worker 시스템 프롬프트에 주입
                enhanced_prompt = (
                    system_prompt
                    + "\n\n--- Scout exploration result ---\n"
                    + scout_result
                    + "\n--- End of Scout result ---\n"
                    + "Based on Scout's findings above, execute the user's request. "
                    + "Do NOT re-read files that Scout already explored."
                )

                # Worker 실행 (Scout 결과 기반)
                async for event in query_loop(
                    messages=messages,
                    system_prompt=enhanced_prompt,
                    model_provider=self._worker_provider,
                    tools=self._worker_tools,
                    context=self._context,
                    max_turns=self._max_turns,
                    on_turn_complete=on_turn_complete,
                ):
                    yield event
                return

            # Scout 실패 → Worker 직행 fallback
            logger.warning("Scout 실패, Worker 직행으로 fallback")
            self._scout_fallbacks += 1

        # Worker 직행 (TIER_M/L 또는 Scout 실패 시)
        async for event in query_loop(
            messages=messages,
            system_prompt=system_prompt,
            model_provider=self._worker_provider,
            tools=self._worker_tools,
            context=self._context,
            max_turns=self._max_turns,
            on_turn_complete=on_turn_complete,
        ):
            yield event

    async def _run_scout(self, messages: list[Message]) -> str | None:
        """
        Scout(4B CPU)에게 탐색/계획을 요청한다.

        Scout는 읽기 전용 도구(Read, Glob, Grep, LS)만 사용하여
        파일을 탐색하고 계획을 수립한다. 최대 3턴으로 제한한다.

        Returns:
            Scout의 탐색 결과 텍스트. 실패 시 None.
        """
        if self._scout_provider is None:
            return None

        self._scout_calls += 1
        start_time = time.monotonic()

        # Scout용 messages 구성 — 원본 messages의 마지막 user 메시지만 전달
        # 이전 히스토리는 TurnState에서 관리되므로 Scout에게는 현재 요청만 전달
        scout_messages: list[Message] = []
        for msg in reversed(messages):
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role == "user":
                scout_messages = [msg]
                break

        if not scout_messages:
            return None

        # Scout query_loop 실행 (max_turns=3, 읽기 전용 도구만)
        scout_text_parts: list[str] = []
        scout_tool_results: list[str] = []

        try:
            async for event in query_loop(
                messages=scout_messages,
                system_prompt=SCOUT_SYSTEM_PROMPT,
                model_provider=self._scout_provider,
                tools=self._scout_tools,
                context=self._context,
                max_turns=3,  # Scout는 최대 3턴
            ):
                if isinstance(event, StreamEvent):
                    event_type = (
                        event.type if isinstance(event.type, str) else event.type.value
                    )
                    if event_type == StreamEventType.TEXT_DELTA.value and event.text:
                        scout_text_parts.append(event.text)
                elif isinstance(event, Message):
                    role = event.role if isinstance(event.role, str) else event.role.value
                    if role == "tool_result":
                        # 도구 결과를 요약으로 수집
                        content = (
                            event.text_content
                            if hasattr(event, "text_content")
                            else str(event.content)
                        )
                        # 도구 결과가 길면 첫 500자만
                        if len(content) > 500:
                            content = content[:500] + "\n... (truncated)"
                        scout_tool_results.append(content)

        except Exception as e:
            logger.warning("Scout 실행 실패: %s", e)
            return None

        elapsed = time.monotonic() - start_time
        # Scout 평균 지연 시간 계산용 누계 (밀리초 단위)
        self._scout_total_latency_ms += elapsed * 1000.0

        # Scout 결과 조합: 도구 결과 + 텍스트 응답
        result_parts: list[str] = []
        if scout_tool_results:
            result_parts.append(
                "Files explored:\n" + "\n---\n".join(scout_tool_results[:5])
            )
        scout_text = "".join(scout_text_parts)
        if scout_text:
            result_parts.append("Scout analysis:\n" + scout_text[:1000])

        result = "\n\n".join(result_parts) if result_parts else None

        if result:
            logger.info(
                "Scout 탐색 완료: %.1f초, 결과 %d자, 도구결과 %d개",
                elapsed,
                len(result),
                len(scout_tool_results),
            )
        else:
            logger.warning("Scout 탐색 결과 없음 (%.1f초)", elapsed)

        return result

    @property
    def tier(self) -> HardwareTier:
        """현재 하드웨어 티어를 반환한다."""
        return self._tier

    @property
    def scout_enabled(self) -> bool:
        """Scout가 활성화되어 있는지 반환한다."""
        return self._scout_enabled

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
        Scout 실행 통계를 반환한다 — Ch 17 SessionMetrics에서 사용한다.

        필드 의미:
          - scout_calls: Scout가 호출된 총 횟수
          - scout_fallback_count: Scout 실패 후 Worker 단독으로 전환된 횟수
          - scout_avg_latency_ms: Scout 1회 호출당 평균 지연 시간(ms)
          - scout_enabled: Scout 활성 여부 (TIER_S + scout_provider 존재)
          - tier: 현재 하드웨어 티어 ("small" | "medium" | "large")

        scout_fallbacks는 구 이름으로 동일 값을 노출한다 (하위 호환).
        """
        avg_latency_ms = (
            self._scout_total_latency_ms / self._scout_calls
            if self._scout_calls > 0
            else 0.0
        )
        return {
            "tier": self._tier.value,
            "scout_enabled": self._scout_enabled,
            "scout_calls": self._scout_calls,
            "scout_fallback_count": self._scout_fallbacks,
            "scout_fallbacks": self._scout_fallbacks,  # 하위 호환 alias
            "scout_avg_latency_ms": round(avg_latency_ms, 2),
        }
