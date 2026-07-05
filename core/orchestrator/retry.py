"""
재시도 엔진 — 에러 분류 + 지수 백오프(exponential backoff) 재시도.

[이 파일이 하는 일]
Nexus 4-Tier AsyncGenerator 체인에서 가장 아래(Tier 4)에 해당하는 계층이다.
상위 계층(Tier 3의 query_model_streaming 등)이 실제 모델 호출을 수행하다가
예외를 던지면, 이 계층이 그 예외를 붙잡아 "재시도할 가치가 있는 에러인지"를
판단하고, 재시도할 만하면 잠깐 기다렸다가 같은 연산을 다시 실행한다.
즉, 일시적인 장애(서버 5xx, 네트워크 끊김, GPU OOM 등)를 사용자가 눈치채지
못하는 사이에 자동으로 흡수해 주는 안전망이다.

[핵심 구성 요소]
  - ErrorCategory      : 모델 호출 에러를 9가지 종류로 나눈 열거형(Enum).
  - classify_error()   : 실제 예외 객체를 위 9가지 중 하나로 분류하는 함수.
  - RetryConfig        : "몇 번까지, 얼마나 기다리며" 재시도할지 담은 설정.
  - RetryState         : 지금까지 몇 번 재시도했는지 추적하는 상태 객체.
  - calculate_backoff(): 다음 재시도까지의 대기 시간을 계산하는 함수.
  - with_retry()       : 위 요소들을 엮어 실제 재시도 루프를 도는 AsyncGenerator.

[왜 굳이 9가지로 분류하는가]
에러마다 올바른 대처법이 다르기 때문이다. 무조건 재시도하거나 무조건 포기하면
비효율적이거나 복구 가능한 상황을 놓친다.
  - TRANSIENT(5xx)  : 서버 일시 장애라 그냥 다시 보내면 대개 성공한다.
  - CONTEXT_TOO_LONG: 컨텍스트가 너무 길어서 실패 — 압축한 뒤 재시도해야 한다.
  - OOM             : GPU 메모리 부족 — 토큰 수를 줄여야 근본 해결된다.
  - FATAL           : 코드 버그 등 복구 불가 — 재시도는 시간 낭비, 즉시 중단.
각 카테고리에 맞는 복구 전략을 적용해야 재시도가 실질적으로 효과를 낸다.

Ch.7.1 WithRetry Generator 사양을 구현한 것이다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.message import StreamEvent, StreamEventType

logger = logging.getLogger("nexus.orchestrator.retry")


# ─────────────────────────────────────────────
# 에러 분류 (9가지 카테고리)
# ─────────────────────────────────────────────
class ErrorCategory(str, Enum):
    """
    모델 호출 에러를 9가지 카테고리로 분류하는 열거형.

    str을 상속하므로 각 멤버는 그 자체가 문자열이다. 덕분에 로그 출력이나
    StreamEvent의 error_code처럼 문자열이 필요한 곳에 .value로 바로 쓸 수 있다.
    각 카테고리는 아래 DEFAULT_CATEGORY_MAX_RETRIES에서 서로 다른 최대 재시도
    횟수를 부여받으며, 이것이 카테고리별로 다른 복구 전략의 기본이 된다.
    """

    TRANSIENT = "transient"              # 5xx 서버 에러 — 재시도하면 복구
    RATE_LIMIT = "rate_limit"            # 429 — 대기 후 재시도
    CONTEXT_TOO_LONG = "context_too_long"  # 컨텍스트 초과 — 압축 후 재시도
    OOM = "oom"                          # GPU 메모리 부족 — 토큰 축소
    MODEL_ERROR = "model_error"          # CUDA/vLLM 내부 에러 — 재시도 또는 폴백
    INVALID_OUTPUT = "invalid_output"    # JSON 파싱 실패 — 힌트 추가 후 재시도
    STREAM_STALL = "stream_stall"        # 스트림 무응답 — 재시도
    CONNECTION = "connection"            # 네트워크 연결 실패 — 재시도
    FATAL = "fatal"                      # 복구 불가 — 즉시 중단


# 카테고리별 기본 최대 재시도 횟수.
# 복구 가능성이 높고 비용이 싼 에러(연결/Rate limit)는 넉넉히 재시도하고,
# 압축·토큰 축소 같은 부담이 큰 작업이 필요한 에러(컨텍스트 초과 등)는 적게,
# 복구 불가한 FATAL은 0으로 두어 아예 재시도하지 않도록 한다.
DEFAULT_CATEGORY_MAX_RETRIES: dict[ErrorCategory, int] = {
    ErrorCategory.TRANSIENT: 5,
    ErrorCategory.RATE_LIMIT: 8,
    ErrorCategory.CONTEXT_TOO_LONG: 2,
    ErrorCategory.OOM: 3,
    ErrorCategory.MODEL_ERROR: 3,
    ErrorCategory.INVALID_OUTPUT: 2,
    ErrorCategory.STREAM_STALL: 3,
    ErrorCategory.CONNECTION: 5,
    ErrorCategory.FATAL: 0,  # 재시도 안 함
}


@dataclass(frozen=True)
class ClassifiedError:
    """
    분류가 끝난 에러 한 건을 담는 불변(frozen) 데이터 객체.

    classify_error()가 원본 예외를 분석한 결과를 이 형태로 반환한다.
    frozen=True라서 한번 만들어지면 값을 바꿀 수 없다 — 분류 결과가 이후
    재시도 판단 과정에서 실수로 변조되는 것을 막기 위함이다.

    필드:
      category       : 이 에러가 속한 9가지 카테고리 중 하나.
      original_error : 원본 예외 객체(추후 로깅·디버깅용으로 보존).
      message        : 사람이 읽기 쉬운 한국어 설명 메시지.
      is_retryable   : 재시도할 가치가 있는지 여부(FATAL만 False).
    """

    category: ErrorCategory
    original_error: Exception
    message: str
    is_retryable: bool


def classify_error(error: Exception) -> ClassifiedError:
    """
    임의의 예외 객체를 9가지 카테고리 중 하나로 분류한다.

    분류 근거는 두 가지다.
      1) 예외 클래스 이름(type(error).__name__) — 예: "ConnectError".
      2) 예외를 문자열로 변환한 메시지 내용 — 예: "out of memory".
    이 둘을 소문자로 만든 뒤, 특징적인 키워드가 들어 있는지 순서대로 검사한다.

    [주의] 검사 순서가 매우 중요하다.
    위에서부터 아래로 if 문을 훑으며 처음 걸리는 카테고리를 채택하므로,
    더 구체적이고 우선순위가 높은 패턴을 먼저 배치해야 한다.
    (예: 타임아웃을 서버 에러보다 먼저 잡아야 정확히 분류된다.)
    어떤 패턴에도 걸리지 않으면 마지막에 FATAL(재시도 불가)로 처리한다.

    Args:
        error: 상위 계층에서 발생해 전달된 예외

    Returns:
        분류 결과를 담은 ClassifiedError
    """
    # 이후 모든 검사는 대소문자 구분 없이 하기 위해 미리 소문자로 정규화한다.
    error_str = str(error).lower()
    error_type = type(error).__name__

    # 연결 에러 (httpx.ConnectError 등) — 서버까지 아예 닿지 못한 경우.
    if "connect" in error_type.lower() or "connection" in error_str:
        return ClassifiedError(
            category=ErrorCategory.CONNECTION,
            original_error=error,
            message=f"연결 실패: {error}",
            is_retryable=True,
        )

    # 타임아웃 (httpx.ReadTimeout, asyncio.TimeoutError) — 응답이 제때 오지 않음.
    # 스트림이 멈춘 것과 성격이 비슷하므로 STREAM_STALL로 묶어서 다룬다.
    if "timeout" in error_type.lower() or "timeout" in error_str:
        return ClassifiedError(
            category=ErrorCategory.STREAM_STALL,
            original_error=error,
            message=f"타임아웃: {error}",
            is_retryable=True,
        )

    # StreamWatchdog 타임아웃 — 스트림 감시견이 일정 시간 무응답을 감지한 경우.
    if "watchdog" in error_type.lower() or "stall" in error_str:
        return ClassifiedError(
            category=ErrorCategory.STREAM_STALL,
            original_error=error,
            message=f"스트림 무응답: {error}",
            is_retryable=True,
        )

    # GPU OOM(Out Of Memory) — GPU 메모리가 부족. 토큰을 줄여 재시도해야 한다.
    if "out of memory" in error_str or "oom" in error_str or "cuda" in error_str:
        return ClassifiedError(
            category=ErrorCategory.OOM,
            original_error=error,
            message=f"GPU 메모리 부족: {error}",
            is_retryable=True,
        )

    # 컨텍스트 초과 — 입력이 모델 최대 길이를 넘김. 압축 후 재시도가 정석이다.
    if ("context" in error_str and "long" in error_str) or "prompt is too long" in error_str:
        return ClassifiedError(
            category=ErrorCategory.CONTEXT_TOO_LONG,
            original_error=error,
            message=f"컨텍스트 초과: {error}",
            is_retryable=True,
        )

    # Rate limit (HTTP 429) — 요청이 너무 잦음. 잠시 기다렸다 재시도하면 풀린다.
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return ClassifiedError(
            category=ErrorCategory.RATE_LIMIT,
            original_error=error,
            message=f"요청 제한 초과: {error}",
            is_retryable=True,
        )

    # HTTP 5xx 서버 에러 — "500"~"509" 같은 숫자 문자열이 메시지에 있는지 확인.
    # 서버 측 일시 장애이므로 그대로 재시도(TRANSIENT)하면 대개 회복된다.
    if any(f"5{i}" in error_str for i in range(10)):
        return ClassifiedError(
            category=ErrorCategory.TRANSIENT,
            original_error=error,
            message=f"서버 에러: {error}",
            is_retryable=True,
        )

    # JSON 파싱 실패 — 모델이 형식에 맞지 않는 출력을 냄. 힌트를 주고 재시도한다.
    if "json" in error_str and ("decode" in error_str or "parse" in error_str):
        return ClassifiedError(
            category=ErrorCategory.INVALID_OUTPUT,
            original_error=error,
            message=f"출력 파싱 실패: {error}",
            is_retryable=True,
        )

    # 모델/추론 엔진 내부 에러 — vLLM 등 추론 스택 자체의 문제. 재시도 또는 폴백.
    if "model" in error_str or "vllm" in error_str or "inference" in error_str:
        return ClassifiedError(
            category=ErrorCategory.MODEL_ERROR,
            original_error=error,
            message=f"모델 에러: {error}",
            is_retryable=True,
        )

    # 위 어떤 패턴에도 걸리지 않음 → 원인을 알 수 없거나 코드 버그일 가능성.
    # 재시도해도 같은 결과일 것이므로 FATAL로 분류해 즉시 중단시킨다.
    return ClassifiedError(
        category=ErrorCategory.FATAL,
        original_error=error,
        message=f"복구 불가능한 에러: {error_type}: {error}",
        is_retryable=False,
    )


# ─────────────────────────────────────────────
# 재시도 설정
# ─────────────────────────────────────────────
@dataclass
class RetryConfig:
    """
    재시도를 "어떻게" 할지 정하는 설정 묶음.

    대기 시간은 지수 백오프 공식을 따른다:
        delay = min(base * factor^attempt + jitter, max)
    시도가 거듭될수록(attempt 증가) 대기 시간이 기하급수적으로 늘어나
    서버에 과부하를 주지 않으면서 회복할 시간을 벌어 준다. 다만 max로 상한을
    두어 무한정 길어지지는 않게 한다.

    또한 category_max_retries로 카테고리마다 재시도 한도를 따로 줄 수 있어,
    "연결 에러는 여러 번, 컨텍스트 초과는 두 번만" 같은 세밀한 제어가 가능하다.
    """

    max_retries: int = 5                  # 카테고리 불문 전체 재시도 총량 상한
    base_delay_seconds: float = 0.5       # 백오프의 기준이 되는 최초 지연(초)
    max_delay_seconds: float = 30.0       # 한 번 대기의 상한(초) — 이 값을 넘지 않음
    backoff_factor: float = 2.0           # 시도마다 지연에 곱해지는 배수(2배씩 증가)
    jitter_factor: float = 0.1            # 지터 비율(±10%) — 동시 재시도 몰림 방지
    # 카테고리별 한도. 기본값을 그대로 복사해 넣되, dict()로 새 복사본을 만들어
    # 여러 RetryConfig 인스턴스가 같은 딕셔너리를 공유·변조하지 않도록 한다.
    category_max_retries: dict[ErrorCategory, int] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_MAX_RETRIES)
    )


@dataclass
class RetryState:
    """
    한 번의 with_retry 호출이 진행되는 동안의 재시도 진행 상황을 추적한다.

    설정(RetryConfig)이 "규칙"이라면, 이 객체는 "지금까지의 기록"이다.
    총 몇 번 재시도했는지, 카테고리별로 몇 번씩 했는지, 마지막 에러가 무엇이었는지를
    기억하여, can_retry()가 한도 초과 여부를 판단할 근거로 삼는다.

    필드:
      total_retries    : 카테고리 불문 지금까지의 총 재시도 횟수.
      category_retries : 카테고리별 재시도 횟수 누계(딕셔너리).
      last_error       : 가장 최근에 분류된 에러(디버깅·후처리용).
      start_time       : 재시도 루프 시작 시각. 벽시계가 아닌 단조 시계
                         (monotonic)를 써서 시스템 시간 변경에 영향받지 않는다.
    """

    total_retries: int = 0
    category_retries: dict[ErrorCategory, int] = field(default_factory=dict)
    last_error: ClassifiedError | None = None
    start_time: float = field(default_factory=time.monotonic)

    def can_retry(self, classified: ClassifiedError, config: RetryConfig) -> bool:
        """
        방금 발생한 이 에러를 재시도해도 되는지 최종 판정한다.

        아래 세 관문을 모두 통과해야만 재시도를 허용(True)한다. 하나라도
        걸리면 즉시 False를 반환해 재시도를 포기한다.
          1. 재시도 가능한 카테고리인가 — FATAL이면 여기서 바로 탈락.
          2. 전체 재시도 총량이 아직 한도 미만인가.
          3. 이 카테고리의 재시도 횟수가 아직 카테고리별 한도 미만인가.

        Args:
            classified: 방금 classify_error()로 분류한 에러.
            config: 한도 값들을 담은 재시도 설정.

        Returns:
            재시도 가능하면 True, 아니면 False.
        """
        # 관문 1: FATAL 등 재시도 무의미한 에러는 바로 거부.
        if not classified.is_retryable:
            return False
        # 관문 2: 전체 재시도 총량 상한 초과 여부.
        if self.total_retries >= config.max_retries:
            return False
        # 관문 3: 해당 카테고리만의 한도 초과 여부(설정에 없으면 0으로 간주).
        cat_count = self.category_retries.get(classified.category, 0)
        cat_max = config.category_max_retries.get(classified.category, 0)
        if cat_count >= cat_max:
            return False
        return True

    def record_retry(self, classified: ClassifiedError) -> None:
        """
        실제로 재시도를 한 번 수행하기로 결정했을 때 그 사실을 장부에 기록한다.

        전체 카운터와 해당 카테고리 카운터를 각각 1씩 올리고, 마지막 에러를
        갱신한다. 이렇게 남긴 기록이 다음번 can_retry() 판정의 근거가 된다.
        """
        self.total_retries += 1
        # 해당 카테고리 카운터를 1 증가(첫 등장이면 0에서 시작).
        self.category_retries[classified.category] = (
            self.category_retries.get(classified.category, 0) + 1
        )
        self.last_error = classified


def calculate_backoff(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    다음 재시도까지 몇 초를 기다릴지 계산한다(지수 백오프 + 지터).

    공식: delay = min(base * factor^attempt + jitter, max)
    attempt가 커질수록 base * factor^attempt 부분이 기하급수적으로 늘어난다.

    [지터를 넣는 이유]
    여러 요청이 동시에 실패하면, 지터가 없을 경우 모두 똑같은 시각에 재시도해
    서버를 다시 동시에 때리는 "천둥 무리(thundering herd)" 현상이 생긴다.
    각자 조금씩 다른(±jitter_factor 비율) 무작위 오차를 더해 재시도 시점을
    흩뜨리면 이 몰림 현상을 완화할 수 있다.

    Args:
        attempt: 현재 시도 회차(클수록 대기가 길어짐).
        config: 백오프 계산에 쓸 기준값·배수·상한 등을 담은 설정.

    Returns:
        실제로 대기할 시간(초). max_delay_seconds를 넘지 않는다.
    """
    # 1) 기본 지연에 배수를 회차만큼 거듭제곱해 지수적으로 늘린다.
    delay = config.base_delay_seconds * (config.backoff_factor ** attempt)
    # 2) 지터 추가: (2 * random() - 1)은 -1~+1 사이 값이므로, 결과적으로
    #    delay의 ±jitter_factor 범위에서 무작위로 흔든다.
    #    noqa: S311 — 보안용 난수가 아니라 타이밍 분산용이라 random 모듈로 충분.
    jitter = delay * config.jitter_factor * (2 * random.random() - 1)  # noqa: S311
    delay = delay + jitter
    # 3) 아무리 커져도 설정된 최대 지연을 넘지 않도록 상한을 씌운다.
    return min(delay, config.max_delay_seconds)


# ─────────────────────────────────────────────
# with_retry AsyncGenerator
# ─────────────────────────────────────────────
async def with_retry(
    operation: Callable[..., AsyncGenerator[Any, None]],
    config: RetryConfig | None = None,
    on_retry: Callable[[RetryState, ClassifiedError], None] | None = None,
    **operation_kwargs: Any,
) -> AsyncGenerator[Any, None]:
    """
    임의의 AsyncGenerator 연산을 재시도 로직으로 감싸 주는 4-Tier 체인의 최하단 계층.

    이 함수 자체도 AsyncGenerator다. 내부에서 operation을 실행하며 그 이벤트를
    그대로 위로 흘려보내다가(yield), 도중에 예외가 나면 재시도 여부를 판단한다.

    [동작 흐름]
      1. operation을 실행해 나오는 이벤트를 하나씩 그대로 yield한다.
      2. 끝까지 예외 없이 끝나면 성공 — return으로 정상 종료한다.
      3. 예외가 나면 classify_error()로 분류한다.
      4. can_retry()가 False면 → ERROR 이벤트를 yield하고 종료(전파 대신 이벤트화).
      5. can_retry()가 True면 → 기록을 남기고, 백오프만큼 기다린 뒤 while로 재시도.

    [주의] operation은 "이미 만들어진 제너레이터"가 아니라 "제너레이터를 만드는
    함수"여야 한다. 재시도할 때마다 operation(**operation_kwargs)를 새로 호출해
    깨끗한 새 스트림을 열기 때문이다. 소진된 제너레이터는 재시도할 수 없다.

    Args:
        operation: 재시도 대상 AsyncGenerator를 반환하는 호출 가능 객체.
        config: 재시도 설정. None이면 기본 RetryConfig를 사용한다.
        on_retry: 재시도가 결정될 때마다 불리는 선택적 콜백(로깅·계측 등).
        **operation_kwargs: 매 재시도 시 operation에 그대로 넘길 키워드 인자.

    Yields:
        operation이 내보내는 이벤트들, 그리고 재시도/실패 시의 상태 알림 이벤트
        (SYSTEM_WARNING 또는 ERROR).
    """
    # 설정이 주어지지 않으면 안전한 기본값으로 채운다(호출부 편의).
    if config is None:
        config = RetryConfig()

    # 이번 호출 동안만 유효한 재시도 기록장을 새로 만든다.
    state = RetryState()

    # 성공하거나 재시도 불가로 판정될 때까지 계속 도는 재시도 루프.
    while True:
        try:
            # operation을 새로 열어 나오는 이벤트를 하나도 가공 없이 위로 전달.
            async for event in operation(**operation_kwargs):
                yield event
            # 스트림이 예외 없이 끝까지 소비됨 = 성공. 루프를 빠져나간다.
            return
        except Exception as e:
            # 어떤 예외든 일단 붙잡아 카테고리로 분류한다.
            classified = classify_error(e)
            if not state.can_retry(classified, config):
                # 재시도 한도 초과 또는 FATAL — 더는 시도하지 않는다.
                # 예외를 위로 던지는 대신 ERROR StreamEvent로 바꿔 흘려보내,
                # 상위 계층이 스트림의 일부로 일관되게 처리할 수 있게 한다.
                logger.error(
                    "재시도 불가: %s (카테고리: %s, 시도: %d/%d)",
                    classified.message,
                    classified.category.value,
                    state.total_retries,
                    config.max_retries,
                )
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code=classified.category.value,
                    message=classified.message,
                )
                return

            # 여기까지 왔으면 재시도하기로 결정된 것 — 장부에 한 건 기록한다.
            state.record_retry(classified)
            # 방금 올린 total_retries를 회차로 삼아 이번 대기 시간을 계산한다.
            delay = calculate_backoff(state.total_retries, config)

            logger.warning(
                "재시도 %d/%d: %s (카테고리: %s, 대기: %.1f초)",
                state.total_retries,
                config.max_retries,
                classified.message,
                classified.category.value,
                delay,
            )

            # 콜백이 등록돼 있으면 재시도 사실을 외부에 알린다(계측·모니터링 등).
            if on_retry:
                on_retry(state, classified)

            # 사용자·상위 계층에 "재시도 중"임을 알리는 경고 이벤트를 흘려보낸다.
            # 메시지는 100자까지만 잘라 로그·화면이 지나치게 길어지지 않게 한다.
            yield StreamEvent(
                type=StreamEventType.SYSTEM_WARNING,
                message=(
                    f"[재시도 {state.total_retries}/{config.max_retries}] "
                    f"{classified.category.value}: {classified.message[:100]}"
                ),
            )

            # 계산된 백오프 시간만큼 비동기로 대기한 뒤 while 루프 처음으로 돌아가
            # operation을 다시 실행한다. asyncio.sleep이라 이벤트 루프를 막지 않는다.
            await asyncio.sleep(delay)
