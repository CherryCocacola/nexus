"""
사고 오케스트레이터 — 복잡도 평가 → 전략 선택 → 엔진 실행을 조율한다.

ThinkingOrchestrator는 사고 엔진의 진입점이다.
사용자 메시지를 받아 복잡도를 평가하고, 적절한 전략의 엔진을 실행한 뒤
ThinkingResult를 반환한다.

캐시를 통해 동일 입력의 재처리를 방지한다.

사용 예:
  orchestrator = ThinkingOrchestrator(model_provider)
  result = await orchestrator.think("이 함수를 리팩토링해줘", context=messages)
  print(result.response)       # 최종 응답
  print(result.thinking_text)  # 내부 사고 과정 (DIRECT일 때는 빈 문자열)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from core.message import Message, StreamEventType
from core.model.inference import ModelProvider
from core.thinking.assessor import ComplexityAssessor
from core.thinking.cache import ThinkingCache
from core.thinking.hidden_cot import HiddenCoTEngine
from core.thinking.self_reflection import SelfReflectionEngine
from core.thinking.strategy import ThinkingStrategy, select_strategy

logger = logging.getLogger("nexus.thinking.orchestrator")


@dataclass
class ThinkingResult:
    """
    사고 엔진의 실행 결과.

    Attributes:
        strategy: 적용된 사고 전략
        response: 최종 응답 텍스트
        thinking_text: 내부 사고 과정 (DIRECT일 때는 빈 문자열)
        passes: 실제 실행된 LLM 호출 횟수
        elapsed_seconds: 총 소요 시간(초)
        score: 복잡도 스코어 (0.0~1.0)
    """

    strategy: ThinkingStrategy
    response: str
    thinking_text: str
    passes: int
    elapsed_seconds: float
    score: float


class ThinkingOrchestrator:
    """
    사고 엔진 오케스트레이터.

    책임:
      1. ComplexityAssessor로 복잡도 평가
      2. 스코어에 따라 ThinkingStrategy 선택
      3. 해당 전략의 엔진(HiddenCoTEngine, SelfReflectionEngine 등) 실행
      4. ThinkingCache로 동일 입력 재처리 방지
      5. ThinkingResult 반환

    왜 오케스트레이터 패턴인가:
      전략 선택과 엔진 실행을 분리하면 새 전략 추가가 용이하다.
      MULTI_AGENT 전략은 Phase 5.0b에서 별도 엔진으로 구현할 수 있다.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        cache_max_size: int = 100,
        cache_ttl_seconds: float = 3600.0,
    ) -> None:
        """
        오케스트레이터를 초기화한다.

        Args:
            model_provider: LLM 프로바이더 (ModelProvider ABC 구현체)
            cache_max_size: 사고 캐시 최대 크기
            cache_ttl_seconds: 캐시 TTL(초)
        """
        self._assessor = ComplexityAssessor()
        self._model = model_provider
        self._cache = ThinkingCache(
            max_size=cache_max_size,
            ttl_seconds=cache_ttl_seconds,
        )
        # 전략별 엔진 인스턴스
        self._hidden_cot = HiddenCoTEngine()
        self._self_reflect = SelfReflectionEngine()

    async def think(
        self,
        message: str,
        context: list[Message] | None = None,
    ) -> ThinkingResult:
        """
        메시지를 분석하고 적절한 전략으로 사고한다.

        Args:
            message: 사용자 입력 텍스트
            context: 이전 대화 메시지 목록 (선택)

        Returns:
            ThinkingResult — 최종 응답 + 사고 과정 + 메타데이터

        흐름:
          1. 캐시 확인 → 히트 시 즉시 반환
          2. 복잡도 평가 → 스코어 산출
          3. 전략 선택 → DIRECT / HIDDEN_COT / SELF_REFLECT / MULTI_AGENT
          4. 해당 엔진 실행
          5. 캐시 저장
          6. ThinkingResult 반환
        """
        # ── 1단계: 캐시 확인 ──
        cache_key = self._cache.make_key(message)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info(f"캐시 히트: strategy={cached.strategy.value}, score={cached.score:.2f}")
            return cached

        # ── 2단계: 복잡도 평가 ──
        score = self._assessor.assess(message, context)

        # ── 3단계: 전략 선택 ──
        strategy = select_strategy(score)
        logger.info(f"사고 전략 선택: score={score:.2f} → {strategy.value}")

        # ── 4단계: 엔진 실행 ──
        result = await self._execute_strategy(strategy, message, context, score)

        # ── 5단계: 캐시 저장 ──
        self._cache.put(cache_key, result)

        return result

    async def _execute_strategy(
        self,
        strategy: ThinkingStrategy,
        message: str,
        context: list[Message] | None,
        score: float,
    ) -> ThinkingResult:
        """
        선택된 전략에 맞는 엔진을 실행한다.

        Args:
            strategy: 선택된 사고 전략
            message: 사용자 입력
            context: 대화 컨텍스트
            score: 복잡도 스코어

        Returns:
            ThinkingResult
        """
        if strategy == ThinkingStrategy.DIRECT:
            # DIRECT: 추가 사고 없이 모델에 직접 전달
            return await self._execute_direct(message, context, score)

        elif strategy == ThinkingStrategy.HIDDEN_COT:
            # HIDDEN_COT: 2-pass 분석 후 응답
            result = await self._hidden_cot.run(message, context, self._model)
            result.score = score
            return result

        elif strategy == ThinkingStrategy.SELF_REFLECT:
            # SELF_REFLECT: 3-pass 분석 + 응답 + 검증
            result = await self._self_reflect.run(message, context, self._model)
            result.score = score
            return result

        elif strategy == ThinkingStrategy.MULTI_AGENT:
            # MULTI_AGENT: Phase 5.0b에서 구현 예정
            # 현재는 SELF_REFLECT로 폴백한다
            logger.warning("MULTI_AGENT 전략은 아직 미구현 — SELF_REFLECT로 폴백")
            result = await self._self_reflect.run(message, context, self._model)
            result.score = score
            # 전략 정보는 원래 의도를 기록
            result.strategy = ThinkingStrategy.MULTI_AGENT
            return result

        else:
            # 알 수 없는 전략 — DIRECT로 안전하게 폴백
            logger.error(f"알 수 없는 전략: {strategy} — DIRECT로 폴백")
            return await self._execute_direct(message, context, score)

    async def _execute_direct(
        self,
        message: str,
        context: list[Message] | None,
        score: float,
    ) -> ThinkingResult:
        """
        DIRECT 전략: 추가 사고 없이 모델에 직접 전달하여 응답을 받는다.

        단순한 질문(복잡도 < 0.3)에 적합하다.
        1회의 LLM 호출만 수행한다.
        """
        start_time = time.monotonic()

        # 메시지 구성
        messages: list[Message] = []
        if context:
            messages.extend(context)
        messages.append(Message.user(message))

        # 모델 스트림에서 텍스트 수집
        collected: list[str] = []
        async for event in self._model.stream(
            messages=messages,
            system_prompt="",
            temperature=0.7,
        ):
            if event.type == StreamEventType.TEXT_DELTA and event.text:
                collected.append(event.text)
            elif event.type == StreamEventType.ERROR:
                logger.warning(f"DIRECT 스트림 에러: {event.message}")
                break

        elapsed = time.monotonic() - start_time

        return ThinkingResult(
            strategy=ThinkingStrategy.DIRECT,
            response="".join(collected),
            thinking_text="",  # DIRECT는 별도 사고 과정이 없다
            passes=1,
            elapsed_seconds=round(elapsed, 3),
            score=score,
        )

    @property
    def cache_stats(self) -> dict[str, int]:
        """캐시 통계를 반환한다."""
        return self._cache.stats

    def clear_cache(self) -> None:
        """사고 캐시를 비운다."""
        self._cache.clear()
