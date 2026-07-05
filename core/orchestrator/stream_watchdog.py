"""
스트림 감시(StreamWatchdog) — 모델 응답 스트림의 멈춤/지연을 감지한다.

이 모듈은 사양서 Ch.7.2의 StreamWatchdog를 구현한다.
Machine B(GPU 서버)의 vLLM이 내려보내는 SSE 스트림을 감시하여,
GPU 행(hang), vLLM 데드락, 네트워크 끊김처럼 "응답이 멈춘" 상황을
조기에 잡아내고 상위 재시도 로직으로 넘기는 것이 목적이다.

핵심 개념 — 2가지 타임아웃을 동시에 감시한다:
  - idle_timeout (기본 30초): 마지막 토큰을 받은 뒤 경과한 시간.
    토큰이 계속 흐르면 계속 리셋되므로 "중간에 멈춤"을 잡는다.
  - total_timeout (기본 300초): 스트림을 시작한 뒤의 전체 경과 시간.
    토큰이 느리게라도 계속 오더라도 "너무 오래 끄는" 것을 잡는다.

왜 필요한가 (실제로 겪는 장애 시나리오):
  - vLLM이 CUDA 에러로 행(hang)에 빠져 토큰을 더 못 내려보낼 수 있다.
  - GPU 열 스로틀링(throttling)으로 응답이 사실상 무한 지연될 수 있다.
  - 네트워크가 끊기면 httpx가 read_timeout까지 무의미하게 대기한다.
  이 워치독이 위 상황을 임계치 도달 시점에 예외로 터뜨려,
  사용자를 오래 기다리게 하지 않고 재시도/폴백 경로로 넘긴다.

주요 구성 요소:
  - StreamWatchdogTimeout: 타임아웃을 알리는 예외.
  - StreamWatchdog: start/ping/check/check_warnings/stop 상태 머신.
  - stream_with_watchdog(): 모델 스트림을 감싸 자동 감시하는 async 래퍼.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from core.message import StreamEvent, StreamEventType

logger = logging.getLogger("nexus.orchestrator.stream_watchdog")


class StreamWatchdogTimeout(Exception):
    """
    스트림 감시 중 타임아웃이 발생했음을 알리는 예외.

    stream_with_watchdog()가 감시 중 idle/total 한계를 넘기면 이 예외를
    raise 한다. 상위 계층(재시도 로직 등)은 이를 잡아 재시도/폴백을 결정한다.
    어떤 종류의 타임아웃인지(timeout_type)와 경과/한계 값을 함께 담아,
    로그와 사용자 메시지에서 원인을 바로 파악할 수 있게 한다.
    """

    def __init__(self, timeout_type: str, elapsed: float, threshold: float) -> None:
        # timeout_type: 어느 타임아웃에 걸렸는지 구분 — "idle" 또는 "total".
        self.timeout_type = timeout_type  # "idle" 또는 "total"
        # elapsed: 실제로 경과한 시간(초). 한계를 얼마나 넘겼는지 판단용.
        self.elapsed = elapsed
        # threshold: 걸린 타임아웃의 기준 한계(초). 로그·디버깅에 사용.
        self.threshold = threshold
        # 사람이 읽는 한글 메시지를 생성해 부모 Exception에 전달한다.
        super().__init__(
            f"스트림 {timeout_type} 타임아웃: "
            f"{elapsed:.1f}초 경과 (한계: {threshold:.1f}초)"
        )


class StreamWatchdog:
    """
    스트리밍 응답을 감시하는 워치독(감시 상태 머신).

    사용 흐름은 다음과 같다:
      1) start()  — 스트림을 시작할 때 호출해 타이머를 초기화한다.
      2) ping()   — TEXT_DELTA, TOOL_USE_DELTA 같은 토큰 수신 이벤트마다
                    호출해 "마지막 활동 시각"을 갱신한다(idle 타이머 리셋).
      3) check()  — 매 이벤트마다 호출해 idle/total 타임아웃을 검사한다.
      4) stop()   — 스트림이 끝나면 감시를 종료한다.

    이 클래스 자체는 스레드/이벤트 루프를 돌리지 않는다. 즉 스스로 시간을
    재서 끼어드는 게 아니라, 이벤트가 흐를 때 호출자가 check()를 불러줘야
    타임아웃을 판정한다. 시간 기준은 time.monotonic()이라 시스템 시계
    변경(NTP 보정 등)에 영향받지 않는다.
    """

    def __init__(
        self,
        idle_timeout: float = 30.0,
        total_timeout: float = 300.0,
        warning_threshold: float = 0.8,
    ) -> None:
        """
        워치독의 임계치를 설정한다(아직 감시를 시작하지는 않는다).

        Args:
            idle_timeout: 마지막 토큰 수신 이후 허용하는 무응답 한계(초).
            total_timeout: 스트림 시작부터 허용하는 전체 시간 한계(초).
            warning_threshold: 경고를 띄우는 도달 비율. 0.8이면 각 한계의
                80%에 도달했을 때 미리 경고를 한 번 낸다.
        """
        # 설정값(불변 파라미터) — 감시 기준으로 계속 참조한다.
        self._idle_timeout = idle_timeout
        self._total_timeout = total_timeout
        self._warning_threshold = warning_threshold

        # 감시 상태(런타임 변수) — start()에서 실제 값으로 초기화된다.
        self._start_time: float = 0.0  # 스트림 시작 시각(monotonic 기준).
        self._last_activity: float = 0.0  # 마지막 토큰을 받은 시각.
        self._token_count: int = 0  # 지금까지 받은 토큰(ping) 수.
        self._started: bool = False  # 감시 중인지 여부(플래그).

        # 같은 경고를 매 이벤트마다 반복해서 찍지 않도록 하는 1회성 플래그.
        # 한 번 경고를 내면 True로 바꿔 중복 로그를 막는다.
        self._warned_idle: bool = False
        self._warned_total: bool = False

    def start(self) -> None:
        """
        감시를 시작(초기화)한다. 스트림을 열기 직전에 호출한다.

        모든 타이머와 카운터, 경고 플래그를 현재 시각 기준으로 리셋한다.
        같은 인스턴스를 재사용하더라도 여기서 깨끗하게 초기화되므로
        이전 스트림의 상태가 남지 않는다.
        """
        now = time.monotonic()  # 단조 증가 시계 — 기준 시각으로 한 번만 읽는다.
        self._start_time = now  # total 타임아웃 계산의 기준점.
        self._last_activity = now  # idle 타임아웃 계산의 기준점(처음엔 시작 시각).
        self._token_count = 0  # 토큰 카운터 초기화.
        self._started = True  # 이제부터 check()/check_warnings()가 동작한다.
        self._warned_idle = False  # 경고 1회성 플래그도 리셋.
        self._warned_total = False

    def ping(self) -> None:
        """
        토큰을 하나 받았을 때 호출한다(스트림이 살아있다는 신호).

        마지막 활동 시각을 현재로 갱신해 idle 타이머를 리셋하고,
        수신 토큰 수를 1 증가시킨다. 즉 토큰이 계속 흐르는 동안에는
        idle 타임아웃에 걸리지 않는다.
        """
        self._last_activity = time.monotonic()  # idle 기준 시각 갱신.
        self._token_count += 1  # 통계/디버깅용 토큰 수 누적.

    def check(self) -> StreamWatchdogTimeout | None:
        """
        지금 시점에 타임아웃에 걸렸는지 검사한다.

        idle을 먼저 보고, 아니면 total을 본다. 어느 하나라도 한계에
        도달했으면 그 사실을 담은 예외 객체를 "반환"한다(직접 raise하지
        않음 — 호출자가 raise 여부를 결정한다).

        Returns:
            타임아웃 발생 시 StreamWatchdogTimeout, 아직 정상이면 None.
        """
        # 감시를 시작하지 않았거나 이미 stop 했으면 판정하지 않는다.
        if not self._started:
            return None

        now = time.monotonic()  # 두 검사에서 같은 기준 시각을 쓰도록 한 번만 읽는다.

        # 1) idle 타임아웃: 마지막 토큰 이후 얼마나 조용했는지.
        idle_elapsed = now - self._last_activity
        if idle_elapsed >= self._idle_timeout:
            # 중간에 멈춘 상황 — 어느 종류/얼마나 경과했는지 담아 반환.
            return StreamWatchdogTimeout(
                timeout_type="idle",
                elapsed=idle_elapsed,
                threshold=self._idle_timeout,
            )

        # 2) total 타임아웃: 시작부터 지금까지 전체 경과 시간.
        total_elapsed = now - self._start_time
        if total_elapsed >= self._total_timeout:
            # 토큰은 오지만 전체적으로 너무 오래 끄는 상황 — 반환.
            return StreamWatchdogTimeout(
                timeout_type="total",
                elapsed=total_elapsed,
                threshold=self._total_timeout,
            )

        # 두 한계 모두 여유가 있으면 정상 — None을 반환한다.
        return None

    def check_warnings(self) -> str | None:
        """
        타임아웃 "직전" 경고가 필요한지 검사한다(조기 알림용).

        각 한계의 warning_threshold(기본 80%)에 도달하면 경고 문자열을
        딱 한 번만 돌려준다. 한 번 반환한 뒤에는 _warned_* 플래그로
        중복을 막아, 매 이벤트마다 같은 경고가 쏟아지지 않게 한다.
        이 경고는 예외가 아니라 로그로만 남길 목적의 문자열이다.

        Returns:
            경고가 필요하면 경고 메시지 문자열, 아니면 None.
        """
        # 감시 중이 아니면 경고할 대상이 없다.
        if not self._started:
            return None

        now = time.monotonic()  # idle/total 경고 계산에 공통으로 쓸 현재 시각.

        # idle 경고: 무응답 시간이 한계의 warning_threshold 비율을 넘었는지.
        # 0으로 나누는 것을 피하려고 한계가 0 이하이면 비율을 0으로 둔다.
        idle_elapsed = now - self._last_activity
        idle_pct = idle_elapsed / self._idle_timeout if self._idle_timeout > 0 else 0
        if idle_pct >= self._warning_threshold and not self._warned_idle:
            self._warned_idle = True  # 이 스트림에서 idle 경고는 한 번만.
            return (
                f"스트림 idle 경고: {idle_elapsed:.0f}초 무응답 "
                f"(한계: {self._idle_timeout:.0f}초)"
            )

        # total 경고: 전체 경과 시간이 한계의 warning_threshold 비율을 넘었는지.
        total_elapsed = now - self._start_time
        total_pct = total_elapsed / self._total_timeout if self._total_timeout > 0 else 0
        if total_pct >= self._warning_threshold and not self._warned_total:
            self._warned_total = True  # total 경고도 한 번만.
            return (
                f"스트림 total 경고: {total_elapsed:.0f}초 경과 "
                f"(한계: {self._total_timeout:.0f}초)"
            )

        # 아직 경고 임계치에 못 미쳤거나 이미 경고한 경우 — None.
        return None

    def stop(self) -> None:
        """
        감시를 종료한다. 스트림이 끝나거나 예외로 빠져나갈 때 호출한다.

        _started 플래그만 내리면 이후 check()/check_warnings()는 곧바로
        None을 반환하므로, 종료 후 실수로 재검사되는 것을 막는다.
        """
        self._started = False

    @property
    def token_count(self) -> int:
        """지금까지 ping()으로 집계한 수신 토큰 수를 반환한다(읽기 전용)."""
        return self._token_count

    @property
    def elapsed(self) -> float:
        """
        스트림 시작 이후의 전체 경과 시간(초)을 반환한다(읽기 전용).

        아직 시작 전이거나 이미 stop 한 상태면 0.0을 돌려준다.
        """
        if not self._started:
            return 0.0
        return time.monotonic() - self._start_time


# "토큰 하나를 받았다"로 간주하는 스트림 이벤트 타입의 집합.
# 이 집합에 속한 이벤트가 올 때마다 watchdog.ping()을 호출해 idle 타이머를
# 리셋한다. 즉 실제 생성 진행(텍스트/도구 인자 델타)이 있을 때만 "살아있음"으로
# 본다. TURN_START 같은 제어성 이벤트는 여기 넣지 않는다(진짜 진행이 아니므로).
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
    model_provider.stream()을 StreamWatchdog으로 감싸는 async 래퍼.

    원본 스트림을 그대로 중계(passthrough)하되, 이벤트가 흐를 때마다
    자동으로 ping/경고/타임아웃을 검사한다. 호출자는 워치독의 존재를
    거의 의식하지 않고 이 래퍼로 감싼 async for만 돌리면 된다.

    동작 순서(이벤트 1개마다 반복):
      1) 토큰성 이벤트면 ping()으로 idle 타이머 리셋.
      2) check_warnings()로 임계치 근접 경고를 로그에 남김(1회성).
      3) check()로 실제 타임아웃이면 예외를 raise 해 스트림을 끊음.
      4) 문제없으면 원본 event를 그대로 yield.
    스트림이 끝나거나 예외가 나면 finally에서 반드시 stop()을 부른다.

    사용법:
        async for event in stream_with_watchdog(
            model_provider.stream(messages, ...),
            idle_timeout=30.0,
            total_timeout=300.0,
        ):
            # event 처리

    Args:
        stream: model_provider.stream()이 반환하는 원본 AsyncGenerator.
        idle_timeout: 무응답 한계(초). 마지막 토큰 이후 이만큼 조용하면 중단.
        total_timeout: 전체 시간 한계(초). 시작부터 이만큼 지나면 중단.

    Yields:
        원본 StreamEvent를 순서대로 그대로 흘려보낸다.

    Raises:
        StreamWatchdogTimeout: idle 또는 total 한계에 도달한 경우.
    """
    # 이번 스트림 전용 워치독을 만들고 즉시 감시를 시작한다.
    watchdog = StreamWatchdog(
        idle_timeout=idle_timeout,
        total_timeout=total_timeout,
    )
    watchdog.start()

    try:
        async for event in stream:
            # 이벤트 타입은 str일 수도, Enum일 수도 있어 .value로 정규화한다.
            # (제공자마다 문자열/Enum 표현이 다를 수 있어 방어적으로 처리)
            event_type = event.type if isinstance(event.type, str) else event.type.value
            # 실제 생성 진행을 뜻하는 토큰성 이벤트면 살아있음 신호를 보낸다.
            if event_type in _PINGABLE_EVENTS:
                watchdog.ping()

            # 타임아웃 직전이라면 경고 문자열을 받아 로그로 남긴다(비차단).
            warning = watchdog.check_warnings()
            if warning:
                logger.warning(warning)

            # 실제 타임아웃이면 예외 객체를 받아 raise 해 스트림을 중단시킨다.
            timeout = watchdog.check()
            if timeout:
                raise timeout

            # 여기까지 통과하면 정상 이벤트 — 원본을 그대로 상위로 전달.
            yield event
    finally:
        # 정상 종료·예외·소비자 조기 중단 어느 경우든 감시를 확실히 끈다.
        watchdog.stop()
