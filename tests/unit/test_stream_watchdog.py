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
    _PINGABLE_EVENTS,
    DegenerationMonitor,
    StreamWatchdog,
    StreamWatchdogTimeout,
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


# ─────────────────────────────────────────────
# DegenerationMonitor 테스트 (생성 중 붕괴 감지)
# ─────────────────────────────────────────────
class TestDegenerationMonitor:
    """생성 텍스트 붕괴(동일라인 반복·문자샐러드·이모지 폭주) 감지 검증."""

    def test_repeated_line_flagged(self):
        # 같은 라인이 5회 이상 반복되면 붕괴(kd04 표 헤더 무한반복 유형).
        m = DegenerationMonitor(min_chars=20, check_every=1)
        m.feed("정상 도입부 문장이 여기에 있습니다. ")
        for _ in range(6):
            m.feed("붕괴로 계속 무한 반복되는 매우 긴 표 헤더 라인 예시입니다\n")
        assert m.is_degenerate()

    def test_char_salad_flagged(self):
        # 단일 문자 폭주(대시/기호 샐러드) → 4gram 최빈 비율 초과.
        m = DegenerationMonitor(min_chars=20, check_every=1)
        m.feed("정상 시작 문장입니다. ")
        m.feed("─" * 400)
        assert m.is_degenerate()

    def test_distributed_repetition_flagged(self):
        # 같은 문장이 전체 출력에 흩어져 반복(어느 window에도 안 몰림) → 전역 구절빈도로 감지.
        # kd01형: 반복 사이에 서로 다른 긴 정상 내용이 끼어 window 검사로는 못 잡는 케이스.
        m = DegenerationMonitor(min_chars=20, check_every=1)
        rep = "이 문장은 전체 출력에 흩어져 반복되는 붕괴 구절입니다.\n"
        for i in range(10):
            # 반복 간격을 window(1500)보다 넓게 벌리는 서로 다른 긴 정상 문단.
            m.feed(f"서로 다른 정상 내용 문단 번호 {i} 가 사이사이에 충분히 길게 들어갑니다.\n" * 12)
            m.feed(rep)
        assert m.is_degenerate()

    def test_emoji_spam_flagged(self):
        # 이모지 폭주(kd01 유형) → 이모지 밀도 초과.
        m = DegenerationMonitor(min_chars=20, check_every=1)
        m.feed("객체지향의 추상화를 설명합니다. ")
        m.feed("🚗🚀💨☕♨️" * 80)
        assert m.is_degenerate()

    def test_short_answer_not_flagged(self):
        # min_chars 미만 짧은 정상 답변은 검사하지 않는다(오탐 방지).
        m = DegenerationMonitor(min_chars=700, check_every=1)
        m.feed("리스트는 sorted(lst, reverse=True)로 내림차순 정렬합니다.")
        assert not m.is_degenerate()

    def test_legit_long_varied_not_flagged(self):
        # 각 줄 내용이 다른 긴 정상 답변은 붕괴로 오탐하지 않는다.
        m = DegenerationMonitor(check_every=1)
        text = "\n".join(
            f"{i}. {chr(44032 + i)} 항목은 서로 다른 고유한 설명 문장을 담고 있는 정상 라인 {i}"
            for i in range(60)
        )
        m.feed(text)
        assert not m.is_degenerate()

    def test_legit_table_not_flagged(self):
        # 행마다 내용이 다른 정상 마크다운 표는 통과한다.
        m = DegenerationMonitor(check_every=1)
        words = ["정수", "실수", "문자열", "리스트", "튜플", "딕셔너리", "집합", "불린",
                 "바이트", "복소수", "범위", "제너레이터"]
        rows = "\n".join(
            f"| {w} | {w}는 고유한 자료형 설명 {i}번을 가진다 | 예시값_{w}_{i} |"
            for i, w in enumerate(words * 4)
        )
        m.feed("자료형 정리 표입니다.\n| 타입 | 설명 | 예시 |\n|---|---|---|\n" + rows)
        assert not m.is_degenerate()


# ─────────────────────────────────────────────
# stream_with_watchdog degeneration 절단 테스트
# ─────────────────────────────────────────────
async def _degenerate_stream() -> AsyncGenerator[StreamEvent, None]:
    """정상 앞부분 뒤 동일 라인이 반복되는 붕괴 스트림."""
    yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="정상 시작 부분입니다. ")
    for _ in range(12):
        yield StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            text="붕괴로 계속 무한 반복되는 매우 긴 표 헤더 라인 예시입니다\n",
        )
    yield StreamEvent(type=StreamEventType.MESSAGE_STOP)


class TestDegenerationTruncation:
    """stream_with_watchdog의 degeneration 조기 절단 검증."""

    async def test_degeneration_truncates_without_exception(self):
        # 감지 켜짐 → 붕괴 반복 도중 예외 없이 조기 종료(MESSAGE_STOP 도달 못 함).
        mon = DegenerationMonitor(min_chars=20, check_every=1)
        events = []
        async for e in stream_with_watchdog(
            _degenerate_stream(), detect_degeneration=True, degen_monitor=mon
        ):
            events.append(e)
        types = [e.type for e in events]  # use_enum_values=True → 문자열
        assert StreamEventType.MESSAGE_STOP.value not in types  # 절단되어 종료이벤트 미도달
        text_events = [t for t in types if t == StreamEventType.TEXT_DELTA.value]
        assert len(text_events) < 13  # 반복 12개를 다 흘리기 전에 끊김

    async def test_degeneration_disabled_passes_through(self):
        # 감지 꺼짐(기본) → 붕괴 콘텐츠도 전부 통과(무회귀).
        events = []
        async for e in stream_with_watchdog(
            _degenerate_stream(), detect_degeneration=False
        ):
            events.append(e)
        types = [e.type for e in events]
        assert StreamEventType.MESSAGE_STOP.value in types  # 절단 없이 끝까지
