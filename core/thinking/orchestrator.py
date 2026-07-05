"""
사고 오케스트레이터 — 복잡도 평가 → 전략 선택 → 엔진 실행을 조율하는 모듈.

[이 파일이 하는 일]
Nexus의 "생각(thinking) 계층" 진입점이다. 사용자 메시지가 들어오면 먼저
그 메시지가 얼마나 복잡한지 점수(score)로 평가한다. 그 점수에 맞춰
"어떻게 생각할지"를 정하는 전략(ThinkingStrategy)을 고른 뒤, 전략별 엔진을
실행해 최종 응답과 내부 사고 과정을 담은 ThinkingResult를 돌려준다.

단순한 질문은 곧바로 모델에 넘겨(DIRECT) 비용을 아끼고, 복잡한 질문일수록
여러 번(2-pass, 3-pass) 곱씹어 품질을 높이는 식으로 자원을 배분한다.
같은 질문이 반복될 때는 캐시로 재처리를 막아 지연과 GPU 비용을 줄인다.

[주요 구성요소]
  - ThinkingResult : 사고 결과를 담는 데이터 컨테이너(전략/응답/사고문/메타).
  - ThinkingOrchestrator : 평가→전략선택→엔진실행→캐시를 총괄하는 오케스트레이터.

[의존하는 모듈]
  - ComplexityAssessor : 메시지 복잡도를 0.0~1.0 점수로 평가.
  - select_strategy    : 점수를 전략(enum)으로 매핑.
  - HiddenCoTEngine / SelfReflectionEngine : 전략별 실제 사고 실행 엔진.
  - ThinkingCache      : 동일 입력 결과를 잠시 저장하는 캐시.
  - ModelProvider      : LLM 추론 스트림을 제공하는 프로바이더(ABC 구현체).

[사용 예]
  orchestrator = ThinkingOrchestrator(model_provider)
  result = await orchestrator.think("이 함수를 리팩토링해줘", context=messages)
  print(result.response)       # 최종 응답
  print(result.thinking_text)  # 내부 사고 과정 (DIRECT일 때는 빈 문자열)

작성자: 이현수 / 작성일: 2026-07-05
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
    사고 엔진 한 번의 실행 결과를 담는 데이터 컨테이너.

    think()의 반환값이자, 각 전략 엔진(HiddenCoT/SelfReflection)이 만들어내는
    표준 결과 타입이다. 최종 사용자에게 보여줄 응답뿐 아니라, 어떤 전략으로
    얼마나 오래·몇 번의 LLM 호출로 답을 만들었는지 같은 관찰용(observability)
    메타데이터도 함께 담아 로깅·모니터링·튜닝에 활용한다.

    참고: @dataclass라서 필드가 가변(mutable)이다. 오케스트레이터는 엔진이
    돌려준 result의 score/strategy 필드를 실행 후 덮어써 최종 값을 확정한다.

    Attributes:
        strategy: 실제로 적용된 사고 전략(enum).
        response: 사용자에게 보여줄 최종 응답 텍스트.
        thinking_text: 내부 사고 과정 텍스트. DIRECT 전략에서는 빈 문자열.
        passes: 이 결과를 만들기 위해 실제 수행한 LLM 호출 횟수.
        elapsed_seconds: 총 소요 시간(초).
        score: 복잡도 스코어(0.0~1.0). 전략 선택의 근거가 된 값.
    """

    strategy: ThinkingStrategy
    response: str
    thinking_text: str
    passes: int
    elapsed_seconds: float
    score: float


class ThinkingOrchestrator:
    """
    사고 계층의 총괄 지휘자(오케스트레이터).

    think()라는 단일 진입점으로 아래 5단계를 순서대로 처리한다:
      1. ComplexityAssessor로 메시지 복잡도를 점수로 평가.
      2. 그 점수를 기준으로 ThinkingStrategy를 선택.
      3. 선택된 전략에 대응하는 엔진(HiddenCoTEngine/SelfReflectionEngine 등)을 실행.
      4. ThinkingCache로 동일 입력의 재처리를 방지(넣기/꺼내기).
      5. 결과를 ThinkingResult로 반환.

    [왜 오케스트레이터 패턴인가]
      "무엇을 할지 정하는 부분(전략 선택)"과 "실제로 하는 부분(엔진 실행)"을
      분리해 두면, 새 전략을 추가할 때 이 클래스의 분기 한 곳만 손대면 된다.
      예컨대 MULTI_AGENT 전략은 Phase 5.0b에서 전용 엔진으로 붙일 수 있고,
      그 전까지는 SELF_REFLECT로 안전하게 폴백하도록 설계돼 있다.

    [상태(인스턴스 필드)]
      - _assessor    : 복잡도 평가기.
      - _model       : LLM 프로바이더(엔진들이 추론에 사용).
      - _cache       : 사고 결과 캐시.
      - _hidden_cot  : HIDDEN_COT 전략 엔진(2-pass).
      - _self_reflect: SELF_REFLECT 전략 엔진(3-pass).
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        cache_max_size: int = 100,
        cache_ttl_seconds: float = 3600.0,
    ) -> None:
        """
        오케스트레이터를 초기화하고 협력 객체들을 한 번에 준비한다.

        엔진과 평가기는 상태가 가벼워 인스턴스당 하나씩 만들어 재사용한다.
        캐시 크기·TTL은 호출부에서 주입받아(하드코딩 회피) 운영 환경에 맞춰
        조절할 수 있게 한다.

        Args:
            model_provider: LLM 프로바이더(ModelProvider ABC 구현체). 실제
                추론 스트림은 이 객체를 통해서만 얻는다.
            cache_max_size: 사고 캐시가 보관할 최대 항목 수.
            cache_ttl_seconds: 캐시 항목의 유효 시간(초). 지나면 만료 처리.
        """
        # 복잡도 평가기: 메시지가 얼마나 어려운지 점수를 매긴다.
        self._assessor = ComplexityAssessor()
        # LLM 프로바이더: 전략 엔진과 DIRECT 실행이 추론에 사용한다.
        self._model = model_provider
        # 동일 입력 재처리 방지용 캐시(크기·TTL은 주입값으로 구성).
        self._cache = ThinkingCache(
            max_size=cache_max_size,
            ttl_seconds=cache_ttl_seconds,
        )
        # 전략별 엔진 인스턴스 — 생성 비용을 아끼려고 미리 만들어 재사용한다.
        self._hidden_cot = HiddenCoTEngine()  # HIDDEN_COT: 2-pass 사고 엔진
        self._self_reflect = SelfReflectionEngine()  # SELF_REFLECT: 3-pass 엔진

    async def think(
        self,
        message: str,
        context: list[Message] | None = None,
    ) -> ThinkingResult:
        """
        메시지를 분석해 적절한 전략으로 "생각"하고 결과를 돌려주는 공개 진입점.

        이 클래스에서 외부(오케스트레이터 상위 계층)가 실제로 호출하는 유일한
        메서드다. 나머지 _execute_* 메서드는 내부 구현 세부다.

        Args:
            message: 사용자 입력 텍스트.
            context: 이전 대화 메시지 목록(선택). None이면 단발 질문으로 처리.

        Returns:
            ThinkingResult — 최종 응답 + 내부 사고 과정 + 관찰용 메타데이터.

        흐름:
          1. 캐시 확인 → 히트 시 곧바로 반환(추가 LLM 호출 없음).
          2. 복잡도 평가 → 0.0~1.0 스코어 산출.
          3. 전략 선택 → DIRECT / HIDDEN_COT / SELF_REFLECT / MULTI_AGENT.
          4. 해당 전략 엔진 실행.
          5. 결과를 캐시에 저장.
          6. ThinkingResult 반환.
        """
        # ── 1단계: 캐시 확인 ──
        # 메시지 텍스트로 캐시 키를 만들고, 유효한 이전 결과가 있으면 그대로
        # 재사용한다. 여기서 반환하면 아래의 평가·추론을 통째로 건너뛴다.
        cache_key = self._cache.make_key(message)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info(f"캐시 히트: strategy={cached.strategy.value}, score={cached.score:.2f}")
            return cached

        # ── 2단계: 복잡도 평가 ──
        # 메시지(그리고 맥락)를 보고 난이도를 0.0~1.0 점수로 환산한다.
        score = self._assessor.assess(message, context)

        # ── 3단계: 전략 선택 ──
        # 점수를 구간별 전략으로 매핑한다(낮으면 DIRECT, 높을수록 다중 pass).
        strategy = select_strategy(score)
        logger.info(f"사고 전략 선택: score={score:.2f} → {strategy.value}")

        # ── 4단계: 엔진 실행 ──
        # 고른 전략에 맞는 엔진을 돌려 실제 응답을 생성한다.
        result = await self._execute_strategy(strategy, message, context, score)

        # ── 5단계: 캐시 저장 ──
        # 같은 질문이 또 오면 1단계에서 바로 반환되도록 결과를 넣어 둔다.
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
        선택된 전략을 실제 엔진 호출로 연결하는 내부 디스패처(분기 라우터).

        전략별로 어떤 엔진을 어떻게 부를지의 매핑을 한 곳에 모아 둔 메서드다.
        각 엔진은 자체적으로 ThinkingResult를 만들어 주지만, 복잡도 score는
        엔진이 모르는 값이므로 여기서 result.score에 채워 넣어 최종 결과를
        일관되게 맞춘다.

        Args:
            strategy: 선택된 사고 전략.
            message: 사용자 입력.
            context: 대화 컨텍스트(선택).
            score: 2단계에서 계산한 복잡도 스코어. 결과에 되채운다.

        Returns:
            ThinkingResult — 전략별 엔진이 생성한 결과(score 보정 완료).
        """
        if strategy == ThinkingStrategy.DIRECT:
            # DIRECT: 추가 사고 단계 없이 모델에 곧장 전달(가장 저렴).
            return await self._execute_direct(message, context, score)

        elif strategy == ThinkingStrategy.HIDDEN_COT:
            # HIDDEN_COT: 숨은 사고(2-pass) — 먼저 분석하고 이어서 응답 생성.
            result = await self._hidden_cot.run(message, context, self._model)
            result.score = score  # 엔진이 모르는 복잡도 점수를 여기서 채운다.
            return result

        elif strategy == ThinkingStrategy.SELF_REFLECT:
            # SELF_REFLECT: 자기검증(3-pass) — 분석 → 응답 → 스스로 검토.
            result = await self._self_reflect.run(message, context, self._model)
            result.score = score
            return result

        elif strategy == ThinkingStrategy.MULTI_AGENT:
            # MULTI_AGENT: 다중 에이전트 전략은 Phase 5.0b에서 구현 예정.
            # 전용 엔진이 아직 없으므로 지금은 SELF_REFLECT로 안전 폴백한다.
            logger.warning("MULTI_AGENT 전략은 아직 미구현 — SELF_REFLECT로 폴백")
            result = await self._self_reflect.run(message, context, self._model)
            result.score = score
            # 실행은 SELF_REFLECT지만, 원래 "의도한" 전략은 기록으로 남긴다.
            result.strategy = ThinkingStrategy.MULTI_AGENT
            return result

        else:
            # 정의되지 않은/예상 밖의 전략값 — DIRECT로 안전하게 폴백(fail-safe).
            logger.error(f"알 수 없는 전략: {strategy} — DIRECT로 폴백")
            return await self._execute_direct(message, context, score)

    async def _execute_direct(
        self,
        message: str,
        context: list[Message] | None,
        score: float,
    ) -> ThinkingResult:
        """
        DIRECT 전략 실행 — 별도 사고 단계 없이 모델에 바로 물어보는 최단 경로.

        복잡도가 낮은 단순 질문(대략 score < 0.3)에 적합하다. 내부 사고 과정을
        만들지 않고 LLM 호출을 1회만 수행하므로 지연·비용이 가장 작다.
        다른 전략 엔진과 달리 이 로직은 오케스트레이터가 직접 들고 있다(엔진이
        따로 없는, 가장 가벼운 경로이기 때문).

        Args:
            message: 사용자 입력.
            context: 이전 대화 메시지(있으면 앞에 붙여 문맥으로 사용).
            score: 복잡도 스코어. 결과 메타데이터에 그대로 기록한다.

        Returns:
            ThinkingResult — thinking_text는 빈 문자열, passes는 1.
        """
        # 소요 시간 측정 시작. 벽시계가 아닌 단조 시계라 시스템 시간 변경에
        # 영향받지 않아 경과 시간 측정에 안전하다.
        start_time = time.monotonic()

        # 모델에 넘길 메시지 목록을 구성한다: (있으면) 이전 맥락 + 이번 질문.
        messages: list[Message] = []
        if context:
            messages.extend(context)  # 대화 이력을 앞쪽에 그대로 이어 붙인다.
        messages.append(Message.user(message))  # 마지막에 이번 사용자 발화 추가.

        # 모델 스트림을 소비하며 텍스트 조각(delta)을 순서대로 모은다.
        # 한 번에 다 받지 않고 스트리밍으로 이어 붙이는 방식이다.
        collected: list[str] = []
        async for event in self._model.stream(
            messages=messages,
            system_prompt="",  # DIRECT는 추가 지시 없이 순수 모델 응답을 받는다.
            temperature=0.7,
        ):
            if event.type == StreamEventType.TEXT_DELTA and event.text:
                # 실제 응답 텍스트 조각 — 버퍼에 누적한다.
                collected.append(event.text)
            elif event.type == StreamEventType.ERROR:
                # 스트림 도중 에러가 나면 경고만 남기고 지금까지 모은 것으로
                # 마무리한다(예외를 던지지 않고 부분 결과라도 반환).
                logger.warning(f"DIRECT 스트림 에러: {event.message}")
                break

        # 총 소요 시간(초) 계산.
        elapsed = time.monotonic() - start_time

        # 모은 텍스트 조각을 하나로 합쳐 최종 결과 객체로 포장해 반환한다.
        return ThinkingResult(
            strategy=ThinkingStrategy.DIRECT,
            response="".join(collected),
            thinking_text="",  # DIRECT는 별도 사고 과정이 없으므로 비운다.
            passes=1,  # LLM 호출은 정확히 1회.
            elapsed_seconds=round(elapsed, 3),
            score=score,
        )

    @property
    def cache_stats(self) -> dict[str, int]:
        """
        사고 캐시의 통계(히트/미스 등)를 읽기 전용으로 노출한다.

        모니터링·디버깅 시 캐시가 실제로 효과를 내는지 확인하는 용도다.
        내부 캐시 객체의 stats를 그대로 위임해 돌려준다.
        """
        return self._cache.stats

    def clear_cache(self) -> None:
        """
        사고 캐시를 전부 비운다.

        모델·프롬프트가 바뀌어 이전 결과를 더 쓰면 안 될 때, 혹은 테스트에서
        캐시 영향을 배제하고 싶을 때 호출한다.
        """
        self._cache.clear()
