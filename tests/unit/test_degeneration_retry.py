# 붕괴 조기 절단 후 샘플링을 바꿔 1회 재생성하는 경로를 고정한다
"""
2026-08-13. 종전에는 붕괴를 감지하면 **자르고 끝**이었다.

    14:33:33 stream_watchdog WARNING: degeneration 감지 — 스트림 조기 절단(8944자 생성 후)
    14:33:33 [structured_output] 응답이 유효한 JSON이 아닙니다 (finish_reason=stop, len=8943)
    14:33:33 POST /v1/chat/completions 200 OK

절단본이 그대로 사용자에게 나갔고, 구조화 출력이라 JSON 까지 깨졌다. 이제 샘플링을
바꿔 한 번 다시 만든다.

★가장 중요한 회귀 방지 — `is_degenerate()` 재호출로 절단을 알아내려 하면 안 된다.
그 함수는 `_last_check_len` 을 갱신하는 부작용이 있어 두 번째 호출이 False 를 돌려준다.
그래서 `was_cut` 이라는 별도 사실 기록을 쓴다. 이 성질을 테스트로 못 박는다.
"""

from __future__ import annotations

import pytest

from core.orchestrator.stream_watchdog import DegenerationMonitor


# ─────────────────────────────────────────────
# ★절단 사실 기록 — 판정 재질의와 다르다
# ─────────────────────────────────────────────
def test_is_degenerate_second_call_returns_false() -> None:
    """이 부작용이 있기 때문에 was_cut 이 필요하다(설계 근거를 고정한다).

    규칙 A(window 내 동일 라인 반복)만 건드리는 입력을 쓴다. 규칙 D(전역 구절 반복)는
    플래그라 몇 번을 물어도 True 라 부작용이 드러나지 않는다 — 그래서 반복 횟수를
    전역 임계(6회 초과) 아래인 6회로 맞춘다.
    """
    monitor = DegenerationMonitor()
    # 최소 검사 길이(700자)를 넘기기 위한, 서로 다른 문장들.
    monitor.feed("".join(f"{i}번째로 서로 다른 내용을 적는 문장입니다. " for i in range(40)))
    # 같은 긴 줄 6회 — 규칙 A(5회 이상)는 걸리고 규칙 D(6회 초과)는 안 걸린다.
    monitor.feed("동일한 표 헤더 라인이 반복됩니다 정말로\n" * 6)

    assert monitor.is_degenerate() is True
    # 같은 상태인데 두 번째 호출은 False — 간헐 검사(check_every)의 부작용이다.
    assert monitor.is_degenerate() is False


def test_was_cut_is_stable_across_calls() -> None:
    """was_cut 은 몇 번을 물어도 같은 답을 준다(부작용 없음)."""
    monitor = DegenerationMonitor()
    assert monitor.was_cut is False

    monitor.mark_cut()

    assert monitor.was_cut is True
    assert monitor.was_cut is True


def test_was_cut_is_false_for_healthy_stream() -> None:
    monitor = DegenerationMonitor()
    monitor.feed("정상적인 답변입니다. 문장마다 내용이 다릅니다.\n" * 5)
    assert monitor.was_cut is False


def test_watchdog_marks_cut_on_degeneration() -> None:
    """워치독이 실제로 표시를 남기는지 — 배선이 끊기면 재생성이 영영 안 돈다."""
    import asyncio

    from core.message import StreamEvent, StreamEventType
    from core.orchestrator.stream_watchdog import stream_with_watchdog

    async def _collapsing():
        for _ in range(80):
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA,
                text="같은 줄이 계속 반복됩니다 정말로 반복됩니다\n",
            )

    async def _run() -> DegenerationMonitor:
        monitor = DegenerationMonitor()
        events = []
        async for ev in stream_with_watchdog(
            _collapsing(),
            idle_timeout=30.0,
            total_timeout=300.0,
            detect_degeneration=True,
            degen_monitor=monitor,
        ):
            events.append(ev)
        # 붕괴로 끊겼으므로 전부(80개)를 받지는 못한다.
        assert len(events) < 80
        return monitor

    monitor = asyncio.run(_run())
    assert monitor.was_cut is True


def test_watchdog_does_not_mark_healthy_stream() -> None:
    """무회귀 — 정상 스트림은 전부 통과하고 표시도 남지 않는다."""
    import asyncio

    from core.message import StreamEvent, StreamEventType
    from core.orchestrator.stream_watchdog import stream_with_watchdog

    async def _healthy():
        for i in range(20):
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=f"{i}번째 문장입니다. ")

    async def _run() -> tuple[DegenerationMonitor, int]:
        monitor = DegenerationMonitor()
        count = 0
        async for _ in stream_with_watchdog(
            _healthy(),
            idle_timeout=30.0,
            total_timeout=300.0,
            detect_degeneration=True,
            degen_monitor=monitor,
        ):
            count += 1
        return monitor, count

    monitor, count = asyncio.run(_run())
    assert count == 20
    assert monitor.was_cut is False


# ─────────────────────────────────────────────
# 재생성 예산 — TurnState 규약
# ─────────────────────────────────────────────
def test_turn_state_starts_with_retry_budget() -> None:
    from core.orchestrator.query_loop import DEGEN_MAX_RETRY, LoopState

    state = LoopState(messages=[])
    assert state.degen_retry_count == 0
    assert state.degen_retry_pending is False
    assert DEGEN_MAX_RETRY >= 1


def test_retry_temperature_is_raised_and_capped() -> None:
    """온도 보정 규약 — 페널티가 아니라 온도를 올리는 이유는 상수 주석 참고."""
    from core.orchestrator.query_loop import (
        DEGEN_RETRY_TEMPERATURE_DELTA,
        DEGEN_RETRY_TEMPERATURE_MAX,
    )

    def raised(base: float) -> float:
        return min(DEGEN_RETRY_TEMPERATURE_MAX, base + DEGEN_RETRY_TEMPERATURE_DELTA)

    # tool_mode 0.3 / knowledge_mode 0.2 — 실제 설정값에서 올라가야 한다.
    assert raised(0.3) > 0.3
    assert raised(0.2) > 0.2
    # 상한을 넘지 않는다 — 사실성이 흔들리면 붕괴를 고치고 정확도를 잃는다.
    assert raised(5.0) == DEGEN_RETRY_TEMPERATURE_MAX


@pytest.mark.parametrize("base", [0.0, 0.2, 0.3, 0.5, 0.9])
def test_retry_temperature_never_decreases(base: float) -> None:
    from core.orchestrator.query_loop import (
        DEGEN_RETRY_TEMPERATURE_DELTA,
        DEGEN_RETRY_TEMPERATURE_MAX,
    )

    raised = min(DEGEN_RETRY_TEMPERATURE_MAX, base + DEGEN_RETRY_TEMPERATURE_DELTA)
    assert raised >= min(base, DEGEN_RETRY_TEMPERATURE_MAX)
