"""
core/orchestrator/retry.py 단위 테스트.

에러 분류(classify_error), 재시도 상태(RetryState),
지수 백오프(calculate_backoff), with_retry AsyncGenerator를 검증한다.

테스트 전략:
  - ErrorCategory 9가지 enum 값 존재 확인
  - classify_error는 에러 타입/메시지별 올바른 카테고리 분류 검증
  - RetryState.can_retry는 전체/카테고리별 한도 검사
  - calculate_backoff는 지수 증가 + max_delay 제한
  - with_retry는 정상 완료, 재시도 성공, FATAL 즉시 중단 시나리오
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.message import StreamEvent, StreamEventType
from core.orchestrator.retry import (
    DEFAULT_CATEGORY_MAX_RETRIES,
    ClassifiedError,
    ErrorCategory,
    RetryConfig,
    RetryState,
    calculate_backoff,
    classify_error,
    with_retry,
)


# ─────────────────────────────────────────────
# ErrorCategory enum 테스트
# ─────────────────────────────────────────────
class TestErrorCategory:
    """ErrorCategory 열거형 9가지 값 검증."""

    def test_error_category_has_nine_values(self):
        """ErrorCategory에 9가지 카테고리가 정의되어 있는지 확인한다."""
        assert len(ErrorCategory) == 9

    def test_error_category_values(self):
        """각 카테고리의 값이 올바른지 확인한다."""
        expected = {
            "transient", "rate_limit", "context_too_long", "oom",
            "model_error", "invalid_output", "stream_stall",
            "connection", "fatal",
        }
        actual = {e.value for e in ErrorCategory}
        assert actual == expected

    def test_default_category_max_retries_covers_all(self):
        """DEFAULT_CATEGORY_MAX_RETRIES가 모든 카테고리를 포함하는지 확인한다."""
        for cat in ErrorCategory:
            assert cat in DEFAULT_CATEGORY_MAX_RETRIES

    def test_fatal_max_retries_is_zero(self):
        """FATAL 카테고리의 기본 최대 재시도는 0이어야 한다."""
        assert DEFAULT_CATEGORY_MAX_RETRIES[ErrorCategory.FATAL] == 0


# ─────────────────────────────────────────────
# classify_error 테스트
# ─────────────────────────────────────────────
class TestClassifyError:
    """에러 분류 함수 검증. 에러 타입/메시지 → 카테고리 매핑을 테스트한다."""

    def test_classify_connection_error(self):
        """ConnectionError → CONNECTION 카테고리로 분류되는지 확인한다."""
        error = ConnectionError("Connection refused")
        result = classify_error(error)
        assert result.category == ErrorCategory.CONNECTION
        assert result.is_retryable is True

    def test_classify_timeout_error(self):
        """TimeoutError → STREAM_STALL 카테고리로 분류되는지 확인한다."""
        error = TimeoutError("Read timeout")
        result = classify_error(error)
        assert result.category == ErrorCategory.STREAM_STALL
        assert result.is_retryable is True

    def test_classify_oom_error(self):
        """'out of memory' 메시지 → OOM 카테고리로 분류되는지 확인한다."""
        error = RuntimeError("CUDA out of memory")
        result = classify_error(error)
        assert result.category == ErrorCategory.OOM
        assert result.is_retryable is True

    def test_classify_context_too_long_error(self):
        """'context too long' 메시지 → CONTEXT_TOO_LONG으로 분류되는지 확인한다."""
        error = ValueError("context is too long for this model")
        result = classify_error(error)
        assert result.category == ErrorCategory.CONTEXT_TOO_LONG
        assert result.is_retryable is True

    def test_classify_rate_limit_error(self):
        """'429 rate limit' 메시지 → RATE_LIMIT으로 분류되는지 확인한다."""
        error = RuntimeError("429 rate limit exceeded")
        result = classify_error(error)
        assert result.category == ErrorCategory.RATE_LIMIT
        assert result.is_retryable is True

    def test_classify_transient_500_error(self):
        """'500 Internal Server Error' → TRANSIENT로 분류되는지 확인한다."""
        error = RuntimeError("500 Internal Server Error")
        result = classify_error(error)
        assert result.category == ErrorCategory.TRANSIENT
        assert result.is_retryable is True

    def test_classify_json_decode_error(self):
        """'json decode error' → INVALID_OUTPUT으로 분류되는지 확인한다."""
        error = ValueError("json decode error: unexpected token")
        result = classify_error(error)
        assert result.category == ErrorCategory.INVALID_OUTPUT
        assert result.is_retryable is True

    def test_classify_unknown_error_as_fatal(self):
        """분류 불가능한 에러 → FATAL로 분류되는지 확인한다."""
        error = RuntimeError("something completely unexpected happened")
        result = classify_error(error)
        assert result.category == ErrorCategory.FATAL
        assert result.is_retryable is False

    def test_classified_error_preserves_original(self):
        """ClassifiedError가 원본 예외를 보존하는지 확인한다."""
        original = ConnectionError("test")
        result = classify_error(original)
        assert result.original_error is original
        assert len(result.message) > 0


# ─────────────────────────────────────────────
# RetryState 테스트
# ─────────────────────────────────────────────
class TestRetryState:
    """재시도 상태 추적 테스트."""

    def _make_classified(
        self,
        category: ErrorCategory = ErrorCategory.TRANSIENT,
        retryable: bool = True,
    ) -> ClassifiedError:
        """테스트용 ClassifiedError를 생성하는 헬퍼."""
        return ClassifiedError(
            category=category,
            original_error=RuntimeError("test"),
            message="test error",
            is_retryable=retryable,
        )

    def test_can_retry_within_limits(self):
        """전체/카테고리 한도 이내이면 재시도 가능해야 한다."""
        state = RetryState()
        config = RetryConfig(max_retries=5)
        classified = self._make_classified(ErrorCategory.TRANSIENT)
        assert state.can_retry(classified, config) is True

    def test_can_retry_false_when_max_retries_exceeded(self):
        """전체 재시도 횟수가 max_retries 이상이면 False를 반환해야 한다."""
        state = RetryState(total_retries=5)
        config = RetryConfig(max_retries=5)
        classified = self._make_classified(ErrorCategory.TRANSIENT)
        assert state.can_retry(classified, config) is False

    def test_can_retry_false_when_category_limit_exceeded(self):
        """카테고리별 한도를 초과하면 False를 반환해야 한다."""
        state = RetryState(
            total_retries=2,
            # CONTEXT_TOO_LONG의 기본 카테고리 한도는 2
            category_retries={ErrorCategory.CONTEXT_TOO_LONG: 2},
        )
        config = RetryConfig(max_retries=10)
        classified = self._make_classified(ErrorCategory.CONTEXT_TOO_LONG)
        assert state.can_retry(classified, config) is False

    def test_can_retry_false_for_non_retryable(self):
        """is_retryable=False인 에러는 재시도 불가해야 한다."""
        state = RetryState()
        config = RetryConfig(max_retries=5)
        classified = self._make_classified(ErrorCategory.FATAL, retryable=False)
        assert state.can_retry(classified, config) is False

    def test_record_retry_increments_counters(self):
        """record_retry가 전체/카테고리 카운터를 증가시키는지 확인한다."""
        state = RetryState()
        classified = self._make_classified(ErrorCategory.TRANSIENT)

        state.record_retry(classified)
        assert state.total_retries == 1
        assert state.category_retries[ErrorCategory.TRANSIENT] == 1
        assert state.last_error is classified

        # 두 번째 기록
        state.record_retry(classified)
        assert state.total_retries == 2
        assert state.category_retries[ErrorCategory.TRANSIENT] == 2


# ─────────────────────────────────────────────
# calculate_backoff 테스트
# ─────────────────────────────────────────────
class TestCalculateBackoff:
    """지수 백오프 계산 테스트."""

    def test_backoff_increases_exponentially(self):
        """시도 횟수가 증가하면 대기 시간도 지수적으로 증가해야 한다."""
        # 지터를 0으로 설정하여 결정적으로 테스트
        config = RetryConfig(
            base_delay_seconds=1.0,
            backoff_factor=2.0,
            jitter_factor=0.0,
            max_delay_seconds=1000.0,
        )
        delay_0 = calculate_backoff(0, config)  # 1.0 * 2^0 = 1.0
        delay_1 = calculate_backoff(1, config)  # 1.0 * 2^1 = 2.0
        delay_2 = calculate_backoff(2, config)  # 1.0 * 2^2 = 4.0

        assert delay_0 == pytest.approx(1.0)
        assert delay_1 == pytest.approx(2.0)
        assert delay_2 == pytest.approx(4.0)

    def test_backoff_capped_at_max_delay(self):
        """대기 시간이 max_delay_seconds를 초과하지 않아야 한다."""
        config = RetryConfig(
            base_delay_seconds=1.0,
            backoff_factor=2.0,
            jitter_factor=0.0,
            max_delay_seconds=10.0,
        )
        # attempt=10이면 1.0 * 2^10 = 1024.0이지만 max 10.0으로 제한
        delay = calculate_backoff(10, config)
        assert delay <= config.max_delay_seconds

    def test_backoff_with_jitter_in_range(self):
        """지터가 적용되어도 합리적 범위 내에 있어야 한다."""
        config = RetryConfig(
            base_delay_seconds=1.0,
            backoff_factor=2.0,
            jitter_factor=0.1,
            max_delay_seconds=100.0,
        )
        # 여러 번 실행하여 범위 확인 (attempt=0: base=1.0, jitter ±10%)
        for _ in range(50):
            delay = calculate_backoff(0, config)
            assert 0.8 <= delay <= 1.2  # 1.0 ± 0.1*1.0 범위 내


# ─────────────────────────────────────────────
# with_retry AsyncGenerator 테스트
# ─────────────────────────────────────────────


async def _make_success_stream() -> AsyncGenerator[StreamEvent, None]:
    """성공하는 스트림을 생성한다."""
    yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="hello")
    yield StreamEvent(type=StreamEventType.MESSAGE_STOP)


class TestWithRetry:
    """with_retry AsyncGenerator 래퍼 테스트."""

    async def test_with_retry_success_yields_events(self):
        """정상 완료 시 operation의 이벤트를 그대로 yield해야 한다."""
        events: list[StreamEvent] = []
        async for event in with_retry(_make_success_stream):
            events.append(event)

        assert len(events) == 2
        assert events[0].type == StreamEventType.TEXT_DELTA.value
        assert events[0].text == "hello"
        assert events[1].type == StreamEventType.MESSAGE_STOP.value

    async def test_with_retry_retries_on_transient_error(self):
        """일시적 에러 발생 시 재시도하여 성공해야 한다."""
        call_count = 0

        async def flaky_stream() -> AsyncGenerator[StreamEvent, None]:
            """첫 호출은 실패, 두 번째 호출은 성공하는 스트림."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("500 Internal Server Error")
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="recovered")

        # 재시도 대기 시간을 최소화
        config = RetryConfig(
            max_retries=3,
            base_delay_seconds=0.001,
            max_delay_seconds=0.01,
            jitter_factor=0.0,
        )
        events: list[StreamEvent] = []
        async for event in with_retry(flaky_stream, config=config):
            events.append(event)

        assert call_count == 2
        # SYSTEM_WARNING (재시도 알림) + TEXT_DELTA
        warning_events = [e for e in events if e.type == StreamEventType.SYSTEM_WARNING.value]
        text_events = [e for e in events if e.type == StreamEventType.TEXT_DELTA.value]
        assert len(warning_events) == 1
        assert len(text_events) == 1
        assert text_events[0].text == "recovered"

    async def test_with_retry_fatal_error_stops_immediately(self):
        """FATAL 에러는 재시도 없이 즉시 ERROR 이벤트를 yield하고 종료해야 한다."""
        call_count = 0

        async def fatal_stream() -> AsyncGenerator[StreamEvent, None]:
            """항상 분류 불가능한 에러를 발생시키는 스트림."""
            nonlocal call_count
            call_count += 1
            raise RuntimeError("something completely unexpected happened")
            yield  # AsyncGenerator 타입 힌트를 위한 unreachable yield  # noqa: RET504

        config = RetryConfig(
            max_retries=5,
            base_delay_seconds=0.001,
            jitter_factor=0.0,
        )
        events: list[StreamEvent] = []
        async for event in with_retry(fatal_stream, config=config):
            events.append(event)

        # FATAL은 재시도 0이므로 한 번만 호출
        assert call_count == 1
        # ERROR 이벤트가 yield되어야 한다
        error_events = [e for e in events if e.type == StreamEventType.ERROR.value]
        assert len(error_events) == 1
        assert error_events[0].error_code == ErrorCategory.FATAL.value

    async def test_with_retry_on_retry_callback_called(self):
        """재시도 시 on_retry 콜백이 호출되는지 확인한다."""
        call_count = 0
        callback_calls: list[tuple] = []

        async def flaky_stream() -> AsyncGenerator[StreamEvent, None]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection refused")
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="ok")

        def on_retry_callback(state: RetryState, classified: ClassifiedError) -> None:
            callback_calls.append((state.total_retries, classified.category))

        config = RetryConfig(
            max_retries=3,
            base_delay_seconds=0.001,
            jitter_factor=0.0,
        )
        events = []
        async for event in with_retry(
            flaky_stream, config=config, on_retry=on_retry_callback
        ):
            events.append(event)

        # 콜백이 한 번 호출되어야 한다
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 1  # total_retries
        assert callback_calls[0][1] == ErrorCategory.CONNECTION

    async def test_with_retry_max_retries_exhausted(self):
        """최대 재시도 횟수를 소진하면 ERROR 이벤트로 종료해야 한다."""
        call_count = 0

        async def always_fail_stream() -> AsyncGenerator[StreamEvent, None]:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("500 Internal Server Error")
            yield  # noqa: RET504

        config = RetryConfig(
            max_retries=2,
            base_delay_seconds=0.001,
            max_delay_seconds=0.01,
            jitter_factor=0.0,
        )
        events = []
        async for event in with_retry(always_fail_stream, config=config):
            events.append(event)

        # max_retries=2이므로 총 3번 호출 (최초 1 + 재시도 2), 그 다음은 재시도 불가
        # TRANSIENT 카테고리 max=5이므로 전체 max_retries=2가 먼저 소진
        assert call_count == 3  # 첫 호출 + 2번 재시도
        # 마지막에 ERROR 이벤트
        error_events = [e for e in events if e.type == StreamEventType.ERROR.value]
        assert len(error_events) == 1
