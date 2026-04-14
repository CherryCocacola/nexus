"""
Project Nexus — 테스트 공통 fixture.

모든 테스트에서 공유하는 fixture를 정의한다.
외부 서비스(vLLM, Redis, PostgreSQL)는 mock 처리하여
Machine B 없이도 테스트가 가능하도록 한다.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest


# ─────────────────────────────────────────────
# 이벤트 루프 설정
# ─────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """세션 전체에서 하나의 이벤트 루프를 공유한다."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ─────────────────────────────────────────────
# 임시 디렉토리 (파일 도구 테스트용)
# ─────────────────────────────────────────────
@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """
    격리된 작업 디렉토리를 제공한다.
    Read/Write/Edit 도구 테스트 시 실제 프로젝트 파일을 건드리지 않도록 한다.
    """
    work_dir = tmp_path / "workspace"
    work_dir.mkdir()
    return work_dir


# ─────────────────────────────────────────────
# 설정 fixture
# ─────────────────────────────────────────────
@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """임시 설정 디렉토리를 생성한다."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    return cfg_dir


# ─────────────────────────────────────────────
# Mock: GPU 서버 (vLLM OpenAI 호환 API)
# ─────────────────────────────────────────────
@pytest.fixture
def mock_gpu_client() -> AsyncMock:
    """
    GPU 서버(Machine B)의 httpx.AsyncClient를 mock한다.
    SSE 스트리밍 응답을 시뮬레이션할 수 있다.
    """
    client = AsyncMock()
    client.post = AsyncMock()
    client.get = AsyncMock()
    client.aclose = AsyncMock()
    return client


# ─────────────────────────────────────────────
# Mock: Redis (단기 메모리)
# ─────────────────────────────────────────────
@pytest.fixture
def mock_redis() -> AsyncMock:
    """
    Redis 클라이언트를 mock한다.
    fakeredis를 사용할 수 있으면 대체 가능하다.
    """
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)
    redis.keys = AsyncMock(return_value=[])
    redis.close = AsyncMock()
    return redis


# ─────────────────────────────────────────────
# Mock: PostgreSQL (장기 메모리)
# ─────────────────────────────────────────────
@pytest.fixture
def mock_pg_pool() -> AsyncMock:
    """
    asyncpg 커넥션 풀을 mock한다.
    """
    pool = AsyncMock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    pool.close = AsyncMock()

    # 커넥션 mock
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

    return pool


# ─────────────────────────────────────────────
# Mock: abort 시그널 (취소 테스트용)
# ─────────────────────────────────────────────
@pytest.fixture
def abort_event() -> asyncio.Event:
    """query_loop 취소 시그널을 테스트하기 위한 Event."""
    return asyncio.Event()
