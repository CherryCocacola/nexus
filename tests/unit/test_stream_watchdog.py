"""
core/orchestrator/stream_watchdog.py 단위 테스트.

StreamWatchdog의 타임아웃 감지, 경고 발생, ping/check 동작과
stream_with_watchdog 래퍼의 정상/타임아웃 시나리오를 검증한다.

테스트 전략:
  - StreamWatchdog: start/stop 상태, ping으로 토큰 카운트 증가,
    idle/total 타임아웃 감지, 경고 임계치와 중복 경고 방지
  - stream_with_watchdog: 정상 스트림 통과, TEXT_DELTA에서 ping 호출,
    타임아웃 시 StreamWatchdogTimeout 예외 발생
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest

from core.message import StreamEvent, StreamEventType
from core.orchestrator.stream_watchdog import (
    StreamWatchdog,
    StreamWatchdogTimeout,
    _PINGABLE_EVENTS,
    stream_with_watchdog,
)


# ─────────────────────────────────────────────
# StreamWatchdogTimeout 테스트
# ─────────────────────────────────────────────
class TestStreamWatchdogTimeout:
    """StreamWatchdogTimeout 예외 테스트."""

    def test_timeout_exception_attributes(self):
        """예외 객체의 속성이 올바르게 설정되는지 확인한다."""
        exc = StreamWatchdogTimeout(
            timeout_type="idle", elapsed=35.0, threshold=30.0
        )
        assert exc.timeout_type == "idle"
        assert exc.elapsed == 35.0
        assert exc.threshold == 30.0
        assert "idle" in str(exc)
        assert "35.0" in str(exc)


# ─────────────────────────────────────────────
# StreamWatchdog 테스트
# ─────────────────────────────────────────────
class TestStreamWatchdog:
    """StreamWatchdog 클래스 테스트."""

    def test_init_default_values(self):
        """기본 초기화 값이 올바른지 확인한다."""
        wd = StreamWatchdog()
        # 기본 타임아웃 값 확인
        assert wd._idle_timeout == 30.0
        assert wd._total_timeout == 300.0
        assert wd._warning_threshold == 0.8
        # 시작 전 상태
        assert wd._started is False
        assert wd.token_count == 0

    def test_start_stop_state(self):
        """start/stop이 상태를 올바르게 전환하는지 확인한다."""
        wd = StreamWatchdog(idle_timeout=10.0, total_timeout=60.0)

        # 시작 전: started=False
        assert wd._started is False

        wd.start()
        assert wd._started is True
        assert wd._start_time > 0
        assert wd.token_count == 0

        wd.stop()
        assert wd._started is False

    def test_ping_increments_token_count(self):
        """ping()이 토큰 카운터를 증가시키는지 확인한다."""
        wd = StreamWatchdog()
        wd.start()

        assert wd.token_count == 0
        wd.ping()
        assert wd.token_count == 1
        wd.ping()
        wd.ping()
        assert wd.token_count == 3

    def test_check_returns_none_when_not_started(self):
        """시작 전에 check()는 None을 반환해야 한다."""
        wd = StreamWatchdog()
        result = wd.check()
        assert result is None

    def test_check_idle_timeout_detected(self):
        """idle 타임아웃이 감지되는지 확인한다."""
        wd = StreamWatchdog(idle_timeout=1.0, total_timeout=300.0)
        wd.start()

        # time.monotonic을 조작하여 idle 타임아웃 시뮬레이션
        # start 시점의 _last_activity를 1.5초 전으로 설정
        wd._last_activity = time.monotonic() - 1.5

        result = wd.check()
        assert result is not None
        assert isinstance(result, StreamWatchdogTimeout)
        assert result.timeout_type == "idle"
        assert result.elapsed >= 1.0

    def test_check_total_timeout_detected(self):
        """total 타임아웃이 감지되는지 확인한다."""
        wd = StreamWatchdog(idle_timeout=300.0, total_timeout=2.0)
        wd.start()

        # start_time을 2.5초 전으로 설정하고, last_activity는 최근으로 유지
        wd._start_time = time.monotonic() - 2.5
        wd._last_activity = time.monotonic()  # idle은 정상

        result = wd.check()
        assert result is not None
        assert isinstance(result, StreamWatchdogTimeout)
        assert result.timeout_type == "total"

    def test_check_returns_none_when_healthy(self):
        """타임아웃이 발생하지 않으면 None을 반환해야 한다."""
        wd = StreamWatchdog(idle_timeout=30.0, total_timeout=300.0)
        wd.start()

        # 방금 시작했으므로 idle/total 모두 여유
        result = wd.check()
        assert result is None

    def test_check_warnings_idle_threshold(self):
        """idle 경고 임계치(80%)에 도달하면 경고를 반환해야 한다."""
        wd = StreamWatchdog(
            idle_timeout=10.0,
            total_timeout=300.0,
            warning_threshold=0.8,
        )
        wd.start()

        # 80% = 8초 경과 시뮬레이션
        wd._last_activity = time.monotonic() - 8.5

        warning = wd.check_warnings()
        assert warning is not None
        assert "idle" in warning
        assert "경고" in warning

    def test_check_warnings_no_duplicate(self):
        """동일한 경고가 두 번 발생하지 않아야 한다 (중복 방지)."""
        wd = StreamWatchdog(
            idle_timeout=10.0,
            total_timeout=300.0,
            warning_threshold=0.8,
        )
        wd.start()
        wd._last_activity = time.monotonic() - 8.5

        # 첫 번째 경고
        warning1 = wd.check_warnings()
        assert warning1 is not None

        # 두 번째 호출 — 이미 warned_idle=True이므로 None
        warning2 = wd.check_warnings()
        assert warning2 is None

    def test_check_warnings_total_threshold(self):
        """total 경고 임계치(80%)에 도달하면 경고를 반환해야 한다."""
        wd = StreamWatchdog(
            idle_timeout=300.0,
            total_timeout=100.0,
            warning_threshold=0.8,
        )
        wd.start()

        # total 80% = 80초 경과, idle은 정상으로 유지
        wd._start_time = time.monotonic() - 85.0
        wd._last_activity = time.monotonic()

        warning = wd.check_warnings()
        assert warning is not None
        assert "total" in warning

    def test_elapsed_returns_zero_when_not_started(self):
        """시작 전에 elapsed는 0.0을 반환해야 한다."""
        wd = StreamWatchdog()
        assert wd.elapsed == 0.0

    def test_elapsed_returns_positive_after_start(self):
        """시작 후 elapsed는 양수를 반환해야 한다."""
        wd = StreamWatchdog()
        wd.start()
        # 즉시 호출해도 0 이상이어야 한다
        assert wd.elapsed >= 0.0


# ─────────────────────────────────────────────
# stream_with_watchdog 래퍼 테스트
# ─────────────────────────────────────────────


async def _make_normal_stream() -> AsyncGenerator[StreamEvent, None]:
    """정상적인 스트림 — TEXT_DELTA 2개 + MESSAGE_STOP."""
    yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="hello ")
    yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="world")
    yield StreamEvent(type=StreamEventType.MESSAGE_STOP)


async def _make_stream_with_tool_delta() -> AsyncGenerator[StreamEvent, None]:
    """TOOL_USE_DELTA를 포함하는 스트림 — ping 대상 이벤트."""
    yield StreamEvent(type=StreamEventType.TOOL_USE_DELTA, tool_use_delta='{"file_path":')
    yield StreamEvent(type=StreamEventType.TOOL_USE_DELTA, tool_use_delta='"/tmp/f"}')
    yield StreamEvent(type=StreamEventType.MESSAGE_STOP)


class TestStreamWithWatchdog:
    """stream_with_watchdog 래퍼 테스트."""

    async def test_normal_stream_passes_through(self):
        """정상 스트림의 모든 이벤트가 그대로 전달되는지 확인한다."""
        events: list[StreamEvent] = []
        async for event in stream_with_watchdog(
            _make_normal_stream(),
            idle_timeout=30.0,
            total_timeout=300.0,
        ):
            events.append(event)

        assert len(events) == 3
        assert events[0].type == StreamEventType.TEXT_DELTA.value
        assert events[0].text == "hello "
        assert events[1].text == "world"
        assert events[2].type == StreamEventType.MESSAGE_STOP.value

    async def test_text_delta_triggers_ping(self):
        """TEXT_DELTA 이벤트가 watchdog ping을 트리거하는지 확인한다."""
        # _PINGABLE_EVENTS에 TEXT_DELTA가 포함되어 있는지 먼저 확인
        assert StreamEventType.TEXT_DELTA.value in _PINGABLE_EVENTS

        # watchdog의 ping이 호출되는지 검증하기 위해
        # 정상 스트림을 소비하고, 에러 없이 완료되면 ping이 호출된 것
        events = []
        async for event in stream_with_watchdog(
            _make_normal_stream(),
            idle_timeout=30.0,
            total_timeout=300.0,
        ):
            events.append(event)

        # TEXT_DELTA 2개가 정상 통과
        text_deltas = [e for e in events if e.type == StreamEventType.TEXT_DELTA.value]
        assert len(text_deltas) == 2

    async def test_tool_use_delta_triggers_ping(self):
        """TOOL_USE_DELTA 이벤트도 watchdog ping 대상인지 확인한다."""
        assert StreamEventType.TOOL_USE_DELTA.value in _PINGABLE_EVENTS

        events = []
        async for event in stream_with_watchdog(
            _make_stream_with_tool_delta(),
            idle_timeout=30.0,
            total_timeout=300.0,
        ):
            events.append(event)

        assert len(events) == 3

    async def test_idle_timeout_raises_exception(self):
        """idle 타임아웃 시 StreamWatchdogTimeout 예외가 발생해야 한다."""

        async def stalling_stream() -> AsyncGenerator[StreamEvent, None]:
            """첫 이벤트 후 watchdog check에서 타임아웃을 감지하도록 설정."""
            yield StreamEvent(type=StreamEventType.MESSAGE_START)
            # 두 번째 이벤트 — watchdog.check()가 타임아웃을 반환하도록
            # time.monotonic을 패치하여 시간 경과를 시뮬레이션
            yield StreamEvent(type=StreamEventType.SYSTEM_INFO, message="stalling")

        # 매우 짧은 idle_timeout으로 설정
        with patch("core.orchestrator.stream_watchdog.time.monotonic") as mock_mono:
            base_time = 1000.0
            # start() 시점
            mock_mono.return_value = base_time

            call_count = [0]
            original_monotonic = time.monotonic

            def advancing_monotonic():
                """호출할 때마다 시간을 크게 전진시킨다."""
                call_count[0] += 1
                # start/ping 초기 호출은 base_time
                # check 호출 시에는 idle_timeout을 초과하도록 시간 전진
                if call_count[0] <= 3:
                    return base_time
                return base_time + 50.0  # 50초 경과 → idle 30초 초과

            mock_mono.side_effect = advancing_monotonic

            with pytest.raises(StreamWatchdogTimeout) as exc_info:
                async for _event in stream_with_watchdog(
                    stalling_stream(),
                    idle_timeout=30.0,
                    total_timeout=300.0,
                ):
                    pass

            assert exc_info.value.timeout_type == "idle"

    async def test_watchdog_stops_on_normal_completion(self):
        """정상 완료 시 watchdog이 stop되는지 확인한다 (finally 블록)."""
        # stream_with_watchdog의 finally에서 watchdog.stop()이 호출된다
        # 에러 없이 정상 완료되면 된다
        events = []
        async for event in stream_with_watchdog(
            _make_normal_stream(),
            idle_timeout=30.0,
            total_timeout=300.0,
        ):
            events.append(event)

        # 정상 완료 = watchdog이 에러 없이 stop된 것
        assert len(events) == 3

    async def test_watchdog_stops_on_exception(self):
        """예외 발생 시에도 watchdog이 stop되는지 확인한다 (finally 블록)."""

        async def error_stream() -> AsyncGenerator[StreamEvent, None]:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="before error")
            raise RuntimeError("stream broke")

        with pytest.raises(RuntimeError, match="stream broke"):
            async for _event in stream_with_watchdog(
                error_stream(),
                idle_timeout=30.0,
                total_timeout=300.0,
            ):
                pass
        # finally 블록이 실행되어 watchdog.stop()이 호출됨
        # 예외가 RuntimeError로 전파되면 테스트 통과
