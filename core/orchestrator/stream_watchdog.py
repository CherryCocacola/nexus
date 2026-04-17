"""
스트림 감시 — 모델 응답 스트림의 타임아웃을 감지한다.

Ch.7.2 StreamWatchdog를 구현한다.
model_provider.stream()의 SSE 스트림을 감시하여,
GPU 행(hang), vLLM 데드락, 네트워크 끊김 등을 감지한다.

2가지 타임아웃:
  - idle_timeout (30초): 마지막 토큰 수신 후 경과 시간
  - total_timeout (300초): 전체 스트리밍 시간

왜 필요한가:
  - vLLM이 CUDA 에러로 행(hang)에 빠질 수 있다
  - GPU 열 스로틀링으로 응답이 무한 지연될 수 있다
  - 네트워크 끊김 시 httpx가 read_timeout까지 대기한다
  Watchdog이 이를 조기에 감지하여 재시도 로직에 넘긴다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from core.message import StreamEvent, StreamEventType

logger = logging.getLogger("nexus.orchestrator.stream_watchdog")


class StreamWatchdogTimeout(Exception):
    """StreamWatchdog 타임아웃 예외."""

    def __init__(self, timeout_type: str, elapsed: float, threshold: float) -> None:
        self.timeout_type = timeout_type  # "idle" 또는 "total"
        self.elapsed = elapsed
        self.threshold = threshold
        super().__init__(
            f"스트림 {timeout_type} 타임아웃: "
            f"{elapsed:.1f}초 경과 (한계: {threshold:.1f}초)"
        )


class StreamWatchdog:
    """
    스트리밍 응답을 감시하는 워치독.

    TEXT_DELTA, TOOL_USE_DELTA 등 토큰 수신 이벤트에서 ping()을 호출하고,
    주기적으로 check()를 호출하여 타임아웃 여부를 판단한다.
    """

    def __init__(
        self,
        idle_timeout: float = 30.0,
        total_timeout: float = 300.0,
        warning_threshold: float = 0.8,
    ) -> None:
        """
        Args:
            idle_timeout: 마지막 토큰 이후 대기 한계 (초)
            total_timeout: 전체 스트리밍 시간 한계 (초)
            warning_threshold: 경고 발생 비율 (0.8 = 80% 도달 시)
        """
        self._idle_timeout = idle_timeout
        self._total_timeout = total_timeout
        self._warning_threshold = warning_threshold

        self._start_time: float = 0.0
        self._last_activity: float = 0.0
        self._token_count: int = 0
        self._started: bool = False

        # 중복 경고 방지
        self._warned_idle: bool = False
        self._warned_total: bool = False

    def start(self) -> None:
        """감시를 시작한다. 스트림 시작 시 호출."""
        now = time.monotonic()
        self._start_time = now
        self._last_activity = now
        self._token_count = 0
        self._started = True
        self._warned_idle = False
        self._warned_total = False

    def ping(self) -> None:
        """
        토큰 수신 시 호출한다.
        활동 시간을 갱신하고 토큰 카운터를 증가시킨다.
        """
        self._last_activity = time.monotonic()
        self._token_count += 1

    def check(self) -> StreamWatchdogTimeout | None:
        """
        타임아웃을 검사한다.

        Returns:
            타임아웃 발생 시 StreamWatchdogTimeout, 아니면 None.
        """
        if not self._started:
            return None

        now = time.monotonic()

        # idle 타임아웃 검사
        idle_elapsed = now - self._last_activity
        if idle_elapsed >= self._idle_timeout:
            return StreamWatchdogTimeout(
                timeout_type="idle",
                elapsed=idle_elapsed,
                threshold=self._idle_timeout,
            )

        # total 타임아웃 검사
        total_elapsed = now - self._start_time
        if total_elapsed >= self._total_timeout:
            return StreamWatchdogTimeout(
                timeout_type="total",
                elapsed=total_elapsed,
                threshold=self._total_timeout,
            )

        return None

    def check_warnings(self) -> str | None:
        """
        경고 임계치 도달 여부를 검사한다.
        80% 도달 시 경고 문자열을 반환한다.
        """
        if not self._started:
            return None

        now = time.monotonic()

        # idle 경고
        idle_elapsed = now - self._last_activity
        idle_pct = idle_elapsed / self._idle_timeout if self._idle_timeout > 0 else 0
        if idle_pct >= self._warning_threshold and not self._warned_idle:
            self._warned_idle = True
            return (
                f"스트림 idle 경고: {idle_elapsed:.0f}초 무응답 "
                f"(한계: {self._idle_timeout:.0f}초)"
            )

        # total 경고
        total_elapsed = now - self._start_time
        total_pct = total_elapsed / self._total_timeout if self._total_timeout > 0 else 0
        if total_pct >= self._warning_threshold and not self._warned_total:
            self._warned_total = True
            return (
                f"스트림 total 경고: {total_elapsed:.0f}초 경과 "
                f"(한계: {self._total_timeout:.0f}초)"
            )

        return None

    def stop(self) -> None:
        """감시를 종료한다."""
        self._started = False

    @property
    def token_count(self) -> int:
        """수신한 토큰 수를 반환한다."""
        return self._token_count

    @property
    def elapsed(self) -> float:
        """전체 경과 시간(초)을 반환한다."""
        if not self._started:
            return 0.0
        return time.monotonic() - self._start_time


# 토큰 수신으로 간주하는 이벤트 타입
# 이 이벤트가 올 때마다 watchdog.ping()을 호출한다.
_PINGABLE_EVENTS = {
    StreamEventType.TEXT_DELTA.value,
    StreamEventType.TOOL_USE_DELTA.value,
}


async def stream_with_watchdog(
    stream: AsyncGenerator[StreamEvent, None],
    idle_timeout: float = 30.0,
    total_timeout: float = 300.0,
) -> AsyncGenerator[StreamEvent, None]:
    """
    model_provider.stream()을 watchdog으로 감싸는 래퍼.

    사용법:
        async for event in stream_with_watchdog(
            model_provider.stream(messages, ...),
            idle_timeout=30.0,
            total_timeout=300.0,
        ):
            # event 처리

    Args:
        stream: model_provider.stream()이 반환하는 AsyncGenerator
        idle_timeout: idle 타임아웃 (초)
        total_timeout: total 타임아웃 (초)

    Yields:
        원본 StreamEvent (타임아웃 발생 시 StreamWatchdogTimeout 예외)

    Raises:
        StreamWatchdogTimeout: 타임아웃 발생 시
    """
    watchdog = StreamWatchdog(
        idle_timeout=idle_timeout,
        total_timeout=total_timeout,
    )
    watchdog.start()

    try:
        async for event in stream:
            # 토큰 수신 이벤트이면 ping
            event_type = event.type if isinstance(event.type, str) else event.type.value
            if event_type in _PINGABLE_EVENTS:
                watchdog.ping()

            # 경고 체크
            warning = watchdog.check_warnings()
            if warning:
                logger.warning(warning)

            # 타임아웃 체크
            timeout = watchdog.check()
            if timeout:
                raise timeout

            yield event
    finally:
        watchdog.stop()
