"""
임베딩 서버 워밍업 + keep-warm 단위 테스트 (v0.14.8).

진단 결과 e5-large 임베딩 서버(:8002)가 idle 상태에서 첫 호출 시 ~60초의
cold start가 발생하여 KNOWLEDGE_MODE 첫 호출 응답이 매우 느려짐. 부트스트랩
직후 한 번 워밍업 + 주기적 ping으로 cold 상태 회피한다.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from core.bootstrap import (
    _embedding_keepalive,
    _warmup_embedding,
)


@pytest.mark.asyncio
async def test_warmup_calls_embed_once_on_success() -> None:
    """첫 시도에서 성공하면 추가 재시도하지 않는다."""
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    ok = await _warmup_embedding(provider, retries=2)
    assert ok is True
    provider.embed.assert_awaited_once()


@pytest.mark.asyncio
async def test_warmup_retries_on_failure_then_succeeds() -> None:
    """첫 시도 실패 후 재시도에서 성공하면 True (재시도 backoff는 mock으로 0초화)."""
    provider = AsyncMock()
    provider.embed = AsyncMock(
        side_effect=[ConnectionError("cold"), [[0.0] * 8]],
    )

    # bootstrap 모듈 안에서 쓰는 asyncio.sleep만 패치 — 테스트 시간 < 0.1s
    with patch("core.bootstrap.asyncio.sleep", new=AsyncMock(return_value=None)):
        ok = await _warmup_embedding(provider, retries=2)

    assert ok is True
    assert provider.embed.await_count == 2


@pytest.mark.asyncio
async def test_warmup_returns_false_after_all_retries_fail() -> None:
    """retries 모두 실패하면 False — 본류는 영향 받지 않아야 한다."""
    provider = AsyncMock()
    provider.embed = AsyncMock(side_effect=ConnectionError("down"))

    with patch("core.bootstrap.asyncio.sleep", new=AsyncMock(return_value=None)):
        ok = await _warmup_embedding(provider, retries=1)

    assert ok is False
    # retries=1이면 총 시도 횟수 2 (초기 + 재시도 1)
    assert provider.embed.await_count == 2


@pytest.mark.asyncio
async def test_warmup_returns_false_when_provider_returns_empty() -> None:
    """embed가 빈 리스트를 돌려주면 워밍업 실패로 본다."""
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=[])

    with patch("core.bootstrap.asyncio.sleep", new=AsyncMock(return_value=None)):
        ok = await _warmup_embedding(provider, retries=1)

    assert ok is False


@pytest.mark.asyncio
async def test_keepalive_cancellation_is_clean() -> None:
    """asyncio.CancelledError로 task가 깨끗이 종료된다."""
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=[[0.0] * 4])

    # interval=0.05초로 짧게 — task가 한두 번 ping 후 cancel
    task = asyncio.create_task(_embedding_keepalive(provider, interval_sec=0.05))
    await asyncio.sleep(0.15)  # 2~3회 ping이 일어나도록
    assert provider.embed.await_count >= 1

    task.cancel()
    # 종료 확인 — CancelledError가 swallow되어 task가 정상 완료
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert task.done()


@pytest.mark.asyncio
async def test_keepalive_keeps_running_after_failure() -> None:
    """일시 장애가 발생해도 다음 주기에 다시 ping을 시도한다."""
    provider = AsyncMock()
    # 첫 호출 실패, 두 번째 성공
    provider.embed = AsyncMock(
        side_effect=[ConnectionError("transient"), [[0.0] * 4], [[0.0] * 4]],
    )

    task = asyncio.create_task(_embedding_keepalive(provider, interval_sec=0.03))
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # 실패 후에도 계속 호출이 일어났는지
    assert provider.embed.await_count >= 2
