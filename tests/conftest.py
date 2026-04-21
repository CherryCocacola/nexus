"""
Project Nexus — 테스트 공통 fixture.

모든 테스트에서 공유하는 fixture를 정의한다.
외부 서비스(vLLM, Redis, PostgreSQL)는 mock 처리하여
Machine B 없이도 테스트가 가능하도록 한다.

Phase 8.0에서 추가:
  - MockResponse: 하나의 stream() 호출에 대한 mock 응답 정의
  - EnhancedMockModelProvider: query_loop 통합 테스트용
    tool_calls 시뮬레이션이 가능한 ModelProvider 구현
  - basic_tools, tool_use_context fixture
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator as AsyncGeneratorType
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.message import (
    Message,
    StopReason,
    StreamEvent,
    StreamEventType,
    ToolUseBlock,
)
from core.model.inference import ModelConfig, ModelProvider
from core.tools.base import BaseTool, ToolUseContext


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


# ─────────────────────────────────────────────
# Phase 8.0: 통합 테스트용 Mock 모델 및 fixture
# ─────────────────────────────────────────────


@dataclass
class MockResponse:
    """
    하나의 model_provider.stream() 호출에 대한 mock 응답을 정의한다.

    query_loop 통합 테스트에서 모델의 동작을 시나리오별로 제어할 수 있다.
    tool_calls가 있으면 도구 호출 시퀀스를, 없으면 텍스트 응답을 yield한다.
    """

    # 모델이 생성하는 텍스트 응답
    text: str = ""
    # 모델이 요청하는 도구 호출 목록 [{"name": "Read", "input": {...}}]
    tool_calls: list[dict[str, Any]] | None = None
    # 응답 종료 이유 (tool_calls가 있으면 TOOL_USE, 없으면 END_TURN)
    stop_reason: StopReason | None = None

    def __post_init__(self) -> None:
        """stop_reason을 자동으로 결정한다."""
        if self.stop_reason is None:
            if self.tool_calls:
                self.stop_reason = StopReason.TOOL_USE
            else:
                self.stop_reason = StopReason.END_TURN


class EnhancedMockModelProvider(ModelProvider):
    """
    query_loop 통합 테스트용 모델 프로바이더.

    MockResponse 리스트를 받아 각 stream() 호출마다 순서대로 응답한다.
    tool_calls가 있으면 TOOL_USE_START → TOOL_USE_STOP → MESSAGE_STOP(TOOL_USE)
    이벤트 시퀀스를 생성하여 query_loop이 실제 도구를 실행하도록 한다.

    사용 예시:
        provider = EnhancedMockModelProvider(responses=[
            MockResponse(tool_calls=[{"name": "Read", "input": {"file_path": "/tmp/f.txt"}}]),
            MockResponse(text="파일 내용을 확인했습니다."),
        ])
    """

    def __init__(self, responses: list[MockResponse] | None = None) -> None:
        # 응답 시나리오 리스트 (순차적으로 소비)
        self._responses: list[MockResponse] = responses or [MockResponse(text="mock response")]
        # stream() 호출 횟수 추적 (테스트 검증용)
        self._call_count: int = 0

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
        model_override: str | None = None,
        enable_thinking: bool | None = False,
    ) -> AsyncGeneratorType[StreamEvent, None]:
        """
        MockResponse에 따라 StreamEvent를 yield한다.

        query_loop은 TOOL_USE_STOP 이벤트의 tool_use에서 도구 정보를 추출하므로,
        tool_calls가 있을 때 정확한 이벤트 시퀀스를 생성해야 한다.

        v7.0 Part 2.5 (2026-04-21): 쿼리 라우팅이 model_override/enable_thinking을
        전달할 수 있도록 시그니처에 동일 파라미터를 추가. Mock은 값을 무시한다.
        """
        # 라우팅 파라미터 — 테스트 검증 편의를 위해 마지막 값을 기록만 해둔다
        self._last_model_override = model_override
        self._last_enable_thinking = enable_thinking
        # 현재 턴에 해당하는 응답을 선택 (마지막 응답은 반복 사용)
        idx = min(self._call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self._call_count += 1

        # 메시지 시작
        yield StreamEvent(type=StreamEventType.MESSAGE_START, model_id="mock-model")

        # 텍스트가 있으면 TEXT_DELTA yield
        if response.text:
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=response.text)

        # tool_calls가 있으면 각 도구에 대해 TOOL_USE 이벤트 시퀀스를 yield
        # query_loop은 TOOL_USE_STOP의 event.tool_use에서 도구 정보를 추출한다
        if response.tool_calls:
            for tc in response.tool_calls:
                tool_use_block = ToolUseBlock(
                    name=tc["name"],
                    input=tc.get("input", {}),
                )
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_START,
                    tool_use=tool_use_block,
                )
                yield StreamEvent(
                    type=StreamEventType.TOOL_USE_STOP,
                    tool_use=tool_use_block,
                )

        # 메시지 종료 (stop_reason으로 query_loop의 종료/계속을 제어)
        yield StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            stop_reason=response.stop_reason,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """임베딩 mock — 고정된 더미 벡터를 반환한다."""
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def health_check(self) -> bool:
        """헬스 체크 mock — 항상 정상."""
        return True

    async def count_tokens(self, messages: list[Message]) -> int:
        """토큰 수 추정 mock — 메시지당 50토큰으로 간주."""
        return len(messages) * 50

    def get_config(self) -> ModelConfig:
        """모델 설정 mock."""
        return ModelConfig(
            model_id="mock-model",
            max_context_tokens=8192,
            max_output_tokens=4096,
        )


@pytest.fixture
def mock_model_factory() -> Callable[..., EnhancedMockModelProvider]:
    """
    EnhancedMockModelProvider 팩토리 fixture.

    사용 예시:
        provider = mock_model_factory([
            MockResponse(text="안녕하세요"),
        ])
    """

    def _factory(responses: list[MockResponse] | None = None) -> EnhancedMockModelProvider:
        return EnhancedMockModelProvider(responses=responses)

    return _factory


@pytest.fixture
def tool_use_context(workspace: Path) -> ToolUseContext:
    """통합 테스트용 도구 실행 컨텍스트."""
    return ToolUseContext(
        cwd=str(workspace),
        session_id="test-session",
        permission_mode="bypass_permissions",
    )


@pytest.fixture
def basic_tools(workspace: Path) -> list[BaseTool]:
    """
    통합 테스트용 실제 도구 인스턴스 목록.
    ReadTool, WriteTool, EditTool, GrepTool, GlobTool을 생성한다.
    """
    from core.tools.implementations.edit_tool import EditTool
    from core.tools.implementations.glob_tool import GlobTool
    from core.tools.implementations.grep_tool import GrepTool
    from core.tools.implementations.read_tool import ReadTool
    from core.tools.implementations.write_tool import WriteTool

    return [ReadTool(), WriteTool(), EditTool(), GrepTool(), GlobTool()]
