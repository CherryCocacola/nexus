# cancel_all() 이 취소한 도구 Task 의 정리를 기다리는지 검증.
"""
2026-08-24: cancel_all() 은 task.cancel() 로 신호만 보내고 곧바로 참조를 비웠다.
코드 주석도 그 사실을 인정하고 있었다 — "cancel()은 신호만 보내며 즉시 멈추는 것을
보장하지는 않지만, 아래에서 참조를 모두 비우므로 결과는 어차피 수거되지 않는다".

그러면 아직 정리 중인 도구 코루틴(과 그 안의 async generator)이 그대로 GC 로
넘어간다. 파이썬의 asyncgen finalizer 가 그것을 닫으려 들면
`aclose(): asynchronous generator is already running` 이 날 수 있고, 취소된 Task 를
아무도 회수하지 않아 "Task exception was never retrieved" 경고도 남는다.

★단, 그 경고와의 인과는 **증명하지 못했다.** 재현 시도가 여러 번 실패했다.
  이 수정은 "취소했으면 정리를 기다린다"는 위생 자체가 옳아서 넣은 것이다.

상한(_CANCEL_DRAIN_TIMEOUT)이 반드시 필요하다. cancel_all() 은 **에러 복구 경로**에서
불리므로, 취소에 반응하지 않는 도구 하나가 재시도 전체를 멈춰 세우면 안 된다.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.orchestrator.stream_handler import (
    _CANCEL_DRAIN_TIMEOUT,
    StreamingToolExecutor,
)


def _executor() -> StreamingToolExecutor:
    return StreamingToolExecutor(tools=[], context=None)


class TestWaitsForCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_completes_before_return(self) -> None:
        """★취소 신호만 보내고 떠나지 않는다 — 정리가 끝난 뒤 돌아온다."""
        cleaned = {"v": False}

        async def slow_tool():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.1)  # 파일 닫기·HTTP 종료 등
                cleaned["v"] = True
                raise

        executor = _executor()
        executor._running_tasks = [asyncio.create_task(slow_tool())]
        await asyncio.sleep(0.05)
        await executor.cancel_all()

        assert cleaned["v"] is True

    @pytest.mark.asyncio
    async def test_all_tasks_are_done_after_return(self) -> None:
        """살아 있는 Task 를 남긴 채 참조만 버리면 GC 로 넘어간다."""
        tasks = []

        async def tool():
            await asyncio.sleep(10)

        executor = _executor()
        tasks = [asyncio.create_task(tool()) for _ in range(3)]
        executor._running_tasks = list(tasks)
        await asyncio.sleep(0.05)
        await executor.cancel_all()

        assert all(t.done() for t in tasks)


class TestBoundedByTimeout:
    """★복구 경로를 볼모로 잡지 않는다."""

    @pytest.mark.asyncio
    async def test_uncooperative_tool_does_not_block_forever(self) -> None:
        async def stubborn():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                # 취소를 무시하고 상한을 넘겨 붙잡는다(하지만 테스트가 끝나기 전엔
                # 스스로 끝나도록 유한하게 둔다 — 안 그러면 pending Task 가 남는다).
                await asyncio.sleep(_CANCEL_DRAIN_TIMEOUT + 1.0)

        executor = _executor()
        task = asyncio.create_task(stubborn())
        executor._running_tasks = [task]
        await asyncio.sleep(0.05)

        started = time.monotonic()
        await executor.cancel_all()
        elapsed = time.monotonic() - started

        assert elapsed < _CANCEL_DRAIN_TIMEOUT + 2.0
        # 뒷정리 — 이 테스트가 남긴 Task 가 다른 테스트로 새지 않게 회수한다.
        await asyncio.wait([task], timeout=_CANCEL_DRAIN_TIMEOUT + 3.0)

    def test_timeout_is_short_enough_for_recovery(self) -> None:
        """복구 경로에서 사람이 체감할 만큼 길면 안 된다."""
        assert 0 < _CANCEL_DRAIN_TIMEOUT <= 10.0


class TestStateIsReset:
    """기존 계약 무회귀 — 세 버퍼가 모두 비워져야 한다."""

    @pytest.mark.asyncio
    async def test_buffers_cleared(self) -> None:
        executor = _executor()
        executor._deferred_calls.append({"id": "x", "name": "Y", "input": {}})
        executor._completed.append("결과")
        await executor.cancel_all()

        assert executor._running_tasks == []
        assert executor._deferred_calls == []
        assert executor._completed == []

    @pytest.mark.asyncio
    async def test_no_tasks_is_fine(self) -> None:
        """실행 중인 것이 없으면 즉시 돌아온다."""
        started = time.monotonic()
        await _executor().cancel_all()
        assert time.monotonic() - started < 0.5
