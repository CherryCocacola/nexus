# 워치독 idle 판정을 "조용한 동안" 실시간으로 하도록 바꾼 변경의 회귀 테스트.
"""
2026-08-24 발견: 예전 워치독은 이벤트가 **도착했을 때만** 판정했다. 그런데 토큰성
이벤트는 판정 직전에 ping() 이 idle 타이머를 리셋해 버려, 실질적으로
**idle 타임아웃이 발동할 수 없었다.**

프로덕션 로그에 idle 타임아웃이 찍힌 적은 있으나, 그것은 176초 침묵 뒤 도착한
이벤트가 우연히 **비-ping 종류**였기 때문이다. 텍스트가 흘러오는 스트림이 중간에
멈추면 아무도 잡지 못했다.

수정: 이터레이터에서 한 건씩 **대기 예산(next_wait_budget)** 을 걸어 받는다.
예산을 넘기면 그 자리에서 타임아웃이다 — 이벤트가 오기를 기다릴 필요가 없다.

★cold start 는 idle 이 아니다. 첫 토큰까지 60초가 걸리는 것이 이 시스템의 정상
동작이라, 첫 토큰 전에는 idle 을 적용하지 않고 total_timeout 이 감싼다.
"""

from __future__ import annotations

import asyncio

import pytest

from core.message import StreamEvent, StreamEventType
from core.orchestrator.stream_watchdog import (
    StreamWatchdog,
    StreamWatchdogTimeout,
    stream_with_watchdog,
)


async def _drain(stream, idle: float, total: float) -> list:
    events = []
    async for event in stream_with_watchdog(stream, idle_timeout=idle, total_timeout=total):
        events.append(event)
    return events


def _text(t: str) -> StreamEvent:
    return StreamEvent(type=StreamEventType.TEXT_DELTA, text=t)


class TestDetectsStallWhileSilent:
    """★핵심 — 멈춘 '동안' 잡는다. 다음 이벤트를 기다리지 않는다."""

    @pytest.mark.asyncio
    async def test_mid_stream_stall_raises_idle(self) -> None:
        """텍스트가 흐르다 멈추는 경우. 예전 구현은 이걸 절대 못 잡았다."""

        async def stalling():
            yield _text("가")
            yield _text("나")
            await asyncio.sleep(10)  # 멈춤
            yield _text("다")

        with pytest.raises(StreamWatchdogTimeout) as exc:
            await _drain(stalling(), idle=0.5, total=30.0)
        assert exc.value.timeout_type == "idle"

    @pytest.mark.asyncio
    async def test_does_not_wait_for_next_event(self) -> None:
        """10초 멈춘 스트림이라도 idle 한계 직후에 끊겨야 한다."""
        import time

        async def stalling():
            yield _text("가")
            await asyncio.sleep(10)
            yield _text("나")

        started = time.monotonic()
        with pytest.raises(StreamWatchdogTimeout):
            await _drain(stalling(), idle=0.5, total=30.0)
        assert time.monotonic() - started < 3.0, "멈춘 스트림을 끝까지 기다렸다"

    @pytest.mark.asyncio
    async def test_upstream_is_torn_down_on_timeout(self) -> None:
        """버려지는 스트림이 살아남지 않는다 — GPU 가 꼬리를 계속 만들지 않게.

        대기를 wait_for 로 감싸므로, 만료 시 그 대기 태스크가 취소되며
        제너레이터에 CancelledError 가 전달된다. 이어서 _close_quietly 가
        aclose 를 부른다. 어느 경로로 끝나든 **정리 코드가 반드시 돈다**는 것이
        여기서 고정할 계약이다(예전에는 아무도 닫지 않아 그대로 버려졌다).
        """
        torn_down = {"v": False}

        async def stalling():
            try:
                yield _text("가")
                await asyncio.sleep(10)
            except (GeneratorExit, asyncio.CancelledError):
                torn_down["v"] = True
                raise

        with pytest.raises(StreamWatchdogTimeout):
            await _drain(stalling(), idle=0.5, total=30.0)
        assert torn_down["v"] is True


class TestColdStartTolerated:
    """★첫 토큰이 늦는 것은 정상이다 — idle 로 오인하면 안 된다."""

    @pytest.mark.asyncio
    async def test_slow_first_token_passes(self) -> None:
        async def cold():
            await asyncio.sleep(1.0)  # idle 한계보다 훨씬 길다
            yield _text("첫 토큰")
            yield StreamEvent(type=StreamEventType.MESSAGE_STOP)

        events = await _drain(cold(), idle=0.3, total=30.0)
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_cold_start_still_bounded_by_total(self) -> None:
        """무한정 기다리지는 않는다 — total 이 감싼다."""

        async def never():
            await asyncio.sleep(30)
            yield _text("영영 안 옴")

        with pytest.raises(StreamWatchdogTimeout) as exc:
            await _drain(never(), idle=0.3, total=0.8)
        assert exc.value.timeout_type == "total"


class TestNormalStreamsUnaffected:
    """정상 스트림은 그대로 통과해야 한다(무회귀)."""

    @pytest.mark.asyncio
    async def test_steady_stream_passes(self) -> None:
        async def steady():
            for i in range(5):
                await asyncio.sleep(0.02)
                yield _text(str(i))

        assert len(await _drain(steady(), idle=1.0, total=30.0)) == 5

    @pytest.mark.asyncio
    async def test_empty_stream_ends_cleanly(self) -> None:
        async def empty():
            return
            yield  # pragma: no cover

        assert await _drain(empty(), idle=1.0, total=30.0) == []

    @pytest.mark.asyncio
    async def test_total_timeout_still_fires_on_active_stream(self) -> None:
        """토큰이 꾸준히 와도 전체가 너무 길면 끊는다."""

        async def slow_but_steady():
            for i in range(20):
                await asyncio.sleep(0.1)
                yield _text(str(i))

        with pytest.raises(StreamWatchdogTimeout) as exc:
            await _drain(slow_but_steady(), idle=5.0, total=0.5)
        assert exc.value.timeout_type == "total"


class TestWaitBudget:
    """대기 예산 계산 자체의 경계."""

    def test_before_first_token_uses_total_budget(self) -> None:
        wd = StreamWatchdog(idle_timeout=1.0, total_timeout=100.0)
        wd.start()
        # 첫 토큰 전 — idle(1초)이 아니라 total 잔여를 써야 cold start 가 산다.
        assert wd.next_wait_budget() > 50.0

    def test_after_first_token_uses_idle_budget(self) -> None:
        wd = StreamWatchdog(idle_timeout=1.0, total_timeout=100.0)
        wd.start()
        wd.ping()
        assert wd.next_wait_budget() == pytest.approx(1.0, abs=0.1)

    def test_budget_never_negative(self) -> None:
        wd = StreamWatchdog(idle_timeout=1.0, total_timeout=0.0)
        wd.start()
        assert wd.next_wait_budget() == 0.0

    def test_expired_before_first_token_is_total(self) -> None:
        """첫 토큰조차 못 받았으면 idle 이 아니라 total 로 보고한다."""
        wd = StreamWatchdog(idle_timeout=1.0, total_timeout=100.0)
        wd.start()
        assert wd.expired_timeout().timeout_type == "total"

    def test_expired_after_token_is_idle(self) -> None:
        wd = StreamWatchdog(idle_timeout=1.0, total_timeout=100.0)
        wd.start()
        wd.ping()
        assert wd.expired_timeout().timeout_type == "idle"
