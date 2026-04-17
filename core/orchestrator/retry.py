"""
재시도 엔진 — 에러 분류 + 지수 백오프 재시도.

Ch.7.1 WithRetry Generator를 구현한다.
모든 모델 호출 에러를 9가지 카테고리로 분류하고,
카테고리별 최적 복구 전략을 적용한다.

왜 분류하는가:
  - TRANSIENT(5xx)는 재시도하면 대부분 복구된다
  - CONTEXT_TOO_LONG은 압축 후 재시도해야 한다
  - OOM은 토큰을 줄여야 한다
  - FATAL은 재시도해도 의미 없다
  각 카테고리에 맞는 복구 전략이 달라야 효과적이다.
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
    모델 호출 에러를 9가지 카테고리로 분류한다.
    각 카테고리는 서로 다른 복구 전략을 갖는다.
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


# 카테고리별 기본 최대 재시도 횟수
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
    분류된 에러 정보.
    classify_error()의 반환값.
    """

    category: ErrorCategory
    original_error: Exception
    message: str
    is_retryable: bool


def classify_error(error: Exception) -> ClassifiedError:
    """
    에러를 분류한다.

    에러 타입과 메시지 문자열을 기반으로
    9가지 카테고리 중 하나로 분류한다.
    패턴 매칭 순서가 중요 — 구체적인 것부터 검사한다.

    Args:
        error: 발생한 예외

    Returns:
        분류된 에러 정보
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # 연결 에러 (httpx.ConnectError 등)
    if "connect" in error_type.lower() or "connection" in error_str:
        return ClassifiedError(
            category=ErrorCategory.CONNECTION,
            original_error=error,
            message=f"연결 실패: {error}",
            is_retryable=True,
        )

    # 타임아웃 (httpx.ReadTimeout, asyncio.TimeoutError)
    if "timeout" in error_type.lower() or "timeout" in error_str:
        return ClassifiedError(
            category=ErrorCategory.STREAM_STALL,
            original_error=error,
            message=f"타임아웃: {error}",
            is_retryable=True,
        )

    # StreamWatchdog 타임아웃
    if "watchdog" in error_type.lower() or "stall" in error_str:
        return ClassifiedError(
            category=ErrorCategory.STREAM_STALL,
            original_error=error,
            message=f"스트림 무응답: {error}",
            is_retryable=True,
        )

    # GPU OOM
    if "out of memory" in error_str or "oom" in error_str or "cuda" in error_str:
        return ClassifiedError(
            category=ErrorCategory.OOM,
            original_error=error,
            message=f"GPU 메모리 부족: {error}",
            is_retryable=True,
        )

    # 컨텍스트 초과
    if ("context" in error_str and "long" in error_str) or "prompt is too long" in error_str:
        return ClassifiedError(
            category=ErrorCategory.CONTEXT_TOO_LONG,
            original_error=error,
            message=f"컨텍스트 초과: {error}",
            is_retryable=True,
        )

    # Rate limit (429)
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return ClassifiedError(
            category=ErrorCategory.RATE_LIMIT,
            original_error=error,
            message=f"요청 제한 초과: {error}",
            is_retryable=True,
        )

    # HTTP 5xx 서버 에러
    if any(f"5{i}" in error_str for i in range(10)):
        return ClassifiedError(
            category=ErrorCategory.TRANSIENT,
            original_error=error,
            message=f"서버 에러: {error}",
            is_retryable=True,
        )

    # JSON 파싱 실패
    if "json" in error_str and ("decode" in error_str or "parse" in error_str):
        return ClassifiedError(
            category=ErrorCategory.INVALID_OUTPUT,
            original_error=error,
            message=f"출력 파싱 실패: {error}",
            is_retryable=True,
        )

    # 모델 내부 에러
    if "model" in error_str or "vllm" in error_str or "inference" in error_str:
        return ClassifiedError(
            category=ErrorCategory.MODEL_ERROR,
            original_error=error,
            message=f"모델 에러: {error}",
            is_retryable=True,
        )

    # 분류 불가 → FATAL (재시도 안 함)
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
    재시도 전략 설정.

    지수 백오프: delay = min(base * factor^attempt + jitter, max)
    카테고리별 최대 재시도 횟수를 별도로 설정할 수 있다.
    """

    max_retries: int = 5                  # 전체 최대 재시도
    base_delay_seconds: float = 0.5       # 기본 지연 (초)
    max_delay_seconds: float = 30.0       # 최대 지연 (초)
    backoff_factor: float = 2.0           # 지수 백오프 배수
    jitter_factor: float = 0.1            # 지터 비율 (±10%)
    category_max_retries: dict[ErrorCategory, int] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_MAX_RETRIES)
    )


@dataclass
class RetryState:
    """재시도 상태 추적."""

    total_retries: int = 0
    category_retries: dict[ErrorCategory, int] = field(default_factory=dict)
    last_error: ClassifiedError | None = None
    start_time: float = field(default_factory=time.monotonic)

    def can_retry(self, classified: ClassifiedError, config: RetryConfig) -> bool:
        """
        이 에러에 대해 재시도할 수 있는지 판단한다.

        조건:
          1. 에러가 재시도 가능한 카테고리인가
          2. 전체 재시도 횟수가 한도 이내인가
          3. 해당 카테고리의 재시도 횟수가 한도 이내인가
        """
        if not classified.is_retryable:
            return False
        if self.total_retries >= config.max_retries:
            return False
        cat_count = self.category_retries.get(classified.category, 0)
        cat_max = config.category_max_retries.get(classified.category, 0)
        if cat_count >= cat_max:
            return False
        return True

    def record_retry(self, classified: ClassifiedError) -> None:
        """재시도를 기록한다."""
        self.total_retries += 1
        self.category_retries[classified.category] = (
            self.category_retries.get(classified.category, 0) + 1
        )
        self.last_error = classified


def calculate_backoff(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    지수 백오프 + 지터로 대기 시간을 계산한다.

    공식: delay = min(base * factor^attempt + jitter, max)
    지터는 ±jitter_factor 범위의 랜덤 값.
    """
    delay = config.base_delay_seconds * (config.backoff_factor ** attempt)
    # 지터 추가 (±jitter_factor)
    jitter = delay * config.jitter_factor * (2 * random.random() - 1)  # noqa: S311
    delay = delay + jitter
    # 최대 지연 제한
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
    AsyncGenerator 연산을 재시도 로직으로 감싼다.

    operation이 예외를 발생시키면:
      1. 에러를 분류한다 (classify_error)
      2. 재시도 가능한지 판단한다 (RetryState.can_retry)
      3. 가능하면 지수 백오프 대기 후 재시도
      4. 불가능하면 예외를 그대로 전파

    Args:
        operation: 재시도할 AsyncGenerator 함수
        config: 재시도 설정 (None이면 기본값)
        on_retry: 재시도 시 호출되는 콜백 (로깅 등)
        **operation_kwargs: operation에 전달할 키워드 인자

    Yields:
        operation이 yield하는 이벤트들
    """
    if config is None:
        config = RetryConfig()

    state = RetryState()

    while True:
        try:
            async for event in operation(**operation_kwargs):
                yield event
            # 정상 완료
            return
        except Exception as e:
            classified = classify_error(e)
            if not state.can_retry(classified, config):
                # 재시도 불가 — 에러 이벤트 yield 후 종료
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

            state.record_retry(classified)
            delay = calculate_backoff(state.total_retries, config)

            logger.warning(
                "재시도 %d/%d: %s (카테고리: %s, 대기: %.1f초)",
                state.total_retries,
                config.max_retries,
                classified.message,
                classified.category.value,
                delay,
            )

            if on_retry:
                on_retry(state, classified)

            # 시스템 경고 이벤트
            yield StreamEvent(
                type=StreamEventType.SYSTEM_WARNING,
                message=(
                    f"[재시도 {state.total_retries}/{config.max_retries}] "
                    f"{classified.category.value}: {classified.message[:100]}"
                ),
            )

            await asyncio.sleep(delay)
