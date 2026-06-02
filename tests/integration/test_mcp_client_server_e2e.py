"""
★ MCP 통합 첫 실연결 e2e — 우리 McpClient ↔ create_mcp_app 서버 실연결.

이 파일이 검증하는 핵심:
  지금까지 client/adapter 테스트는 AsyncMock 으로 원격 서버를 흉내냈을 뿐,
  우리 McpClient 가 실제 MCP 서버(mcp_servers/framework.create_mcp_app)와
  JSON-RPC 와이어 프로토콜 수준에서 정합하는지는 검증된 적이 없었다. 이
  테스트는 둘을 실제로 연결해 전 구간(McpClient → adapter)을 관통한다.

연결 방식 (채택: 방식 a — ASGITransport):
  McpClient 는 생성 시 base_url 로 httpx.AsyncClient 를 "자체 생성" 한다.
  실제 포트 바인딩(방식 b)은 포트 경쟁/종료 레이스로 flaky 해질 수 있고,
  framework.py 주석도 "SSE 불필요, 단일 JSON" 을 명시하므로, 가장
  결정론적인 httpx.ASGITransport 로 McpClient 내부 _client 를 교체한다.
    - base_url 은 127.0.0.1(루프백 — LAN 검증 통과)로 두고,
    - ASGITransport(app=app) 가 실제 네트워크 없이 앱을 인프로세스 호출한다.
  이렇게 하면 McpClient 의 _rpc/_parse_response/에러 정규화 로직이 진짜
  서버 응답에 대해 그대로 동작한다(흉내가 아님).

방식 b(임의 포트 uvicorn) 한 건도 보조로 포함해, 진짜 TCP 소켓 경로도
실연결됨을 입증한다(루프백 127.0.0.1 — 에어갭 LAN 준수).

에어갭 준수: 모든 연결은 127.0.0.1(루프백). 외부 네트워크 호출 없음.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from core.tools.base import ToolResult, ToolUseContext
from core.tools.mcp.adapter import McpToolAdapter
from core.tools.mcp.client import McpClient
from mcp_servers.framework import McpServerTool, create_mcp_app

_API_KEY = "e2e-local-key"


# ─────────────────────────────────────────────
# 가짜 서버 도구들 — 실제 서버 도구처럼 동작
# ─────────────────────────────────────────────
class _GreetTool(McpServerTool):
    """이름을 받아 인사말을 돌려주는 read-only 성격 도구."""

    @property
    def name(self) -> str:
        return "greet"

    @property
    def description(self) -> str:
        return "이름을 받아 한국어 인사말을 반환한다."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        who = arguments.get("name", "익명")
        return {"greeting": f"안녕하세요, {who}님"}


class _FailTool(McpServerTool):
    """항상 RuntimeError 를 던져 원격 오류 정규화를 검증하는 도구."""

    @property
    def name(self) -> str:
        return "fail"

    @property
    def description(self) -> str:
        return "원격 오류를 시뮬레이션한다."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def call(self, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("서버 내부 오류")


# ─────────────────────────────────────────────
# 헬퍼 — 방식 a: McpClient 의 _client 를 ASGITransport 로 교체
# ─────────────────────────────────────────────
def _build_app() -> FastAPI:
    """e2e 용 MCP 서버 앱을 만든다."""
    return create_mcp_app(
        tools=[_GreetTool(), _FailTool()],
        api_key=_API_KEY,
        title="E2E MCP Server",
    )


@contextlib.asynccontextmanager
async def _asgi_connected_client(app: FastAPI):
    """
    실제 McpClient 를 만들되, 내부 httpx 클라이언트만 ASGITransport 기반으로
    교체해 인프로세스로 앱에 연결한다.

    McpClient 의 LAN 검증/요청 구성/에러 정규화 로직은 전부 그대로 사용된다.
    base_url 은 루프백(127.0.0.1)이라 LAN 검증을 통과하며, 실제 전송은
    ASGITransport 가 가로채 앱을 직접 호출한다(외부 네트워크 없음).
    """
    client = McpClient(base_url="http://127.0.0.1", api_key=_API_KEY, timeout=5.0)
    # 자체 생성된 httpx 클라이언트를 ASGI 기반으로 교체한다.
    await client._client.aclose()
    transport = httpx.ASGITransport(app=app)
    client._client = httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1")
    try:
        yield client
    finally:
        await client.aclose()


def _make_context() -> ToolUseContext:
    """어댑터 call() 에 넘길 최소 컨텍스트."""
    return ToolUseContext(cwd=".", session_id="e2e", tool_use_id="tu-e2e")


# ─────────────────────────────────────────────
# 방식 a — ASGITransport 실연결 (주 경로)
# ─────────────────────────────────────────────
class TestMcpClientServerE2EViaAsgi:
    """McpClient ↔ create_mcp_app 실연결 (ASGITransport)."""

    async def test_list_tools_returns_server_tool_definitions(self):
        """client.list_tools() 가 서버가 노출한 도구 정의를 그대로 받아야 한다."""
        async with _asgi_connected_client(_build_app()) as client:
            tools = await client.list_tools()

        by_name = {t["name"]: t for t in tools}
        assert set(by_name) == {"greet", "fail"}
        # client 가 inputSchema 키로 스키마를 정규화해 반환하는지(와이어 정합).
        greet = by_name["greet"]
        assert greet["description"]  # 서버 description 이 비어있지 않게 전달됨
        assert greet["inputSchema"]["required"] == ["name"]

    async def test_call_tool_returns_remote_result(self):
        """client.call_tool() 이 원격 도구 실행 결과(result)를 반환해야 한다."""
        async with _asgi_connected_client(_build_app()) as client:
            result = await client.call_tool("greet", {"name": "철수"})
        # framework 의 _rpc_success 봉투에서 result 만 추출되어야 한다.
        assert result == {"greeting": "안녕하세요, 철수님"}

    async def test_call_tool_remote_runtime_error_normalized_to_connection_error(self):
        """서버 도구의 RuntimeError(JSON-RPC error) 는 client 에서 ConnectionError 로 정규화."""
        async with _asgi_connected_client(_build_app()) as client:
            with pytest.raises(ConnectionError):
                await client.call_tool("fail", {})

    async def test_call_unknown_tool_normalized_to_connection_error(self):
        """존재하지 않는 도구(JSON-RPC -32601) 도 ConnectionError 로 정규화되어야 한다."""
        async with _asgi_connected_client(_build_app()) as client:
            with pytest.raises(ConnectionError):
                await client.call_tool("does_not_exist", {})

    async def test_wrong_api_key_normalized_to_connection_error(self):
        """잘못된 api_key 면 서버가 401 → client 가 HTTPStatusError → ConnectionError."""
        app = _build_app()
        # 일부러 틀린 키로 McpClient 를 구성한다.
        client = McpClient(base_url="http://127.0.0.1", api_key="WRONG", timeout=5.0)
        await client._client.aclose()
        transport = httpx.ASGITransport(app=app)
        client._client = httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1")
        try:
            with pytest.raises(ConnectionError):
                await client.list_tools()
        finally:
            await client.aclose()


# ─────────────────────────────────────────────
# 방식 a (확장) — McpClient → McpToolAdapter 전 구간 정합
# ─────────────────────────────────────────────
class TestMcpClientAdapterE2E:
    """list_tools 결과로 어댑터를 만들고 adapter.call() 까지 관통한다 (전 구간)."""

    async def test_adapter_call_success_returns_tool_result_success(self):
        """실서버 list_tools → McpToolAdapter 생성 → adapter.call() → ToolResult.success."""
        async with _asgi_connected_client(_build_app()) as client:
            tools = await client.list_tools()
            greet_def = next(t for t in tools if t["name"] == "greet")

            adapter = McpToolAdapter(
                server_name="e2e",
                remote_tool_name=greet_def["name"],
                remote_schema=greet_def["inputSchema"],
                client=client,
                description=greet_def["description"],
                is_read_only=True,
            )
            # 어댑터 정체성도 정합해야 한다 (권한 분류 규칙).
            assert adapter.name == "mcp__e2e__greet"

            result = await adapter.call({"name": "영희"}, _make_context())

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert result.data == {"greeting": "안녕하세요, 영희님"}

    async def test_adapter_call_remote_error_isolated_to_tool_result_error(self):
        """원격 오류가 어댑터에서 ToolResult.error 로 격리되어야 한다(본류 미영향)."""
        async with _asgi_connected_client(_build_app()) as client:
            adapter = McpToolAdapter(
                server_name="e2e",
                remote_tool_name="fail",
                remote_schema={"type": "object"},
                client=client,
                description="실패 도구",
                is_read_only=False,
            )
            result = await adapter.call({}, _make_context())

        assert result.is_error is True
        # 어느 서버인지 식별 가능해야 한다.
        assert "e2e" in (result.error_message or "")


# ─────────────────────────────────────────────
# 방식 b — 임의 포트 uvicorn 실 TCP 소켓 연결 (보조 입증)
# ─────────────────────────────────────────────
def _free_port() -> int:
    """OS 가 할당하는 빈 포트 번호를 얻는다 (포트 경쟁 회피)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _BackgroundServer:
    """uvicorn 서버를 백그라운드 task 로 띄우고 종료까지 관리하는 헬퍼."""

    def __init__(self, app: FastAPI, port: int) -> None:
        # 로그를 끄고 루프백에만 바인딩한다(에어갭 LAN 준수).
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> _BackgroundServer:
        self._task = asyncio.create_task(self._server.serve())
        # 서버가 startup 을 마칠 때까지 대기(폴링 — sleep 루프 아님).
        for _ in range(100):
            if self._server.started:
                break
            await asyncio.sleep(0.02)
        else:
            raise RuntimeError("uvicorn 서버가 제때 기동하지 못했습니다.")
        return self

    async def __aexit__(self, *exc: object) -> None:
        # graceful shutdown 신호 후 task 종료를 기다린다.
        self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


class TestMcpClientServerE2EViaRealSocket:
    """진짜 TCP 소켓(127.0.0.1:<port>)으로도 McpClient 가 실연결됨을 입증한다."""

    async def test_real_socket_list_and_call(self):
        """실 포트로 띄운 서버에 McpClient(자체 httpx)로 list_tools/call_tool 한다."""
        port = _free_port()
        app = _build_app()
        async with _BackgroundServer(app, port):
            # 여기서는 McpClient 가 자체 생성한 httpx 클라이언트를 그대로 쓴다(교체 없음).
            client = McpClient(base_url=f"http://127.0.0.1:{port}", api_key=_API_KEY, timeout=5.0)
            try:
                tools = await client.list_tools()
                assert {t["name"] for t in tools} == {"greet", "fail"}

                result = await client.call_tool("greet", {"name": "민수"})
                assert result == {"greeting": "안녕하세요, 민수님"}
            finally:
                await client.aclose()
