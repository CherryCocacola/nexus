"""
mcp_servers/framework.py 의 create_mcp_app 단위 테스트.

검증 대상 (클라이언트 client.py 와 1:1 정합):
  1. tools/list → result.tools 배열, 각 항목에 inputSchema(camelCase) 키 존재.
  2. tools/call 성공 → result 에 도구 반환값.
  3. 인증 누락/오류 → HTTP 401 (fail-closed).
  4. 존재하지 않는 도구 → error.code == -32601.
  5. ValueError 를 던지는 도구 → error.code == -32602 (invalid params).
  6. RuntimeError 를 던지는 도구 → error.code == -32603 (internal error).
  7. GET /health → {"status": "healthy"} (인증 없이).
  8. 알 수 없는 method → error.code == -32601.
  9. 본문이 JSON 이 아님 → error.code == -32700 (parse error).

테스트 방법:
  실제 네트워크 없이 httpx.ASGITransport 로 FastAPI 앱을 인프로세스 호출한다.
  (에어갭 규칙 — 외부 호출 없음, Machine B 불필요.)
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from mcp_servers.framework import (
    JSONRPC_INTERNAL_ERROR,
    JSONRPC_INVALID_PARAMS,
    JSONRPC_METHOD_NOT_FOUND,
    JSONRPC_PARSE_ERROR,
    McpServerTool,
    create_mcp_app,
)

# 테스트 전역 — 인증 키와 ASGI 가상 base_url.
_API_KEY = "test-local-key"
_BASE_URL = "http://testserver"


# ─────────────────────────────────────────────
# 가짜 McpServerTool 들 — 성공/입력오류/실행오류 시나리오
# ─────────────────────────────────────────────
class _EchoTool(McpServerTool):
    """입력을 그대로 돌려주는 정상 도구 (성공 경로 검증)."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "입력 인자를 그대로 반환한다."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        # 받은 인자를 echo 키에 담아 반환 — result 정합 검증용.
        return {"echo": arguments.get("value")}


class _BadInputTool(McpServerTool):
    """항상 ValueError 를 던지는 도구 (-32602 경로 검증)."""

    @property
    def name(self) -> str:
        return "bad_input"

    @property
    def description(self) -> str:
        return "입력 검증 실패를 시뮬레이션한다."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def call(self, arguments: dict[str, Any]) -> Any:
        raise ValueError("입력이 잘못되었습니다.")


class _BoomTool(McpServerTool):
    """항상 RuntimeError 를 던지는 도구 (-32603 경로 검증)."""

    @property
    def name(self) -> str:
        return "boom"

    @property
    def description(self) -> str:
        return "내부 실행 오류를 시뮬레이션한다."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def call(self, arguments: dict[str, Any]) -> Any:
        raise RuntimeError("내부 폭발")


# ─────────────────────────────────────────────
# fixture — 앱 + ASGI 인프로세스 httpx 클라이언트
# ─────────────────────────────────────────────
@pytest.fixture
def app() -> FastAPI:
    """3개 도구를 등록한 MCP 서버 앱."""
    return create_mcp_app(
        tools=[_EchoTool(), _BadInputTool(), _BoomTool()],
        api_key=_API_KEY,
        title="Test MCP Server",
    )


@pytest.fixture
async def client(app: FastAPI):
    """앱을 인프로세스로 호출하는 httpx.AsyncClient (ASGITransport)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=_BASE_URL) as c:
        yield c


def _auth_headers() -> dict[str, str]:
    """유효한 Bearer 인증 헤더."""
    return {"Authorization": f"Bearer {_API_KEY}"}


def _rpc_body(method: str, params: dict[str, Any] | None = None, req_id: int = 1) -> dict[str, Any]:
    """JSON-RPC 2.0 요청 봉투를 만든다."""
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}


# ─────────────────────────────────────────────
# tools/list
# ─────────────────────────────────────────────
class TestToolsList:
    """tools/list 응답 형식을 검증한다."""

    async def test_tools_list_returns_tools_array_with_camel_case_schema(self, client):
        """result.tools 배열 + 각 항목에 inputSchema(camelCase) 키가 있어야 한다."""
        resp = await client.post("/", json=_rpc_body("tools/list"), headers=_auth_headers())
        assert resp.status_code == 200
        body = resp.json()

        # JSON-RPC 봉투 정합.
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == 1
        assert "result" in body

        tools = body["result"]["tools"]
        assert isinstance(tools, list)
        names = {t["name"] for t in tools}
        assert names == {"echo", "bad_input", "boom"}

        # 모든 도구가 name/description/inputSchema(camelCase) 키를 가져야 한다.
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "inputSchema" in t  # snake 가 아니라 camelCase 노출(client.py 정합)
            assert "input_schema" not in t

    async def test_tools_list_input_schema_matches_tool_definition(self, client):
        """노출된 inputSchema 가 도구의 input_schema 정의와 동일해야 한다."""
        resp = await client.post("/", json=_rpc_body("tools/list"), headers=_auth_headers())
        tools = {t["name"]: t for t in resp.json()["result"]["tools"]}
        echo_schema = tools["echo"]["inputSchema"]
        assert echo_schema["required"] == ["value"]
        assert echo_schema["properties"]["value"]["type"] == "string"


# ─────────────────────────────────────────────
# tools/call 성공
# ─────────────────────────────────────────────
class TestToolsCallSuccess:
    """tools/call 성공 경로를 검증한다."""

    async def test_tools_call_success_returns_result(self, client):
        """정상 도구 호출 → result 에 도구 반환값이 담겨야 한다."""
        body = _rpc_body("tools/call", {"name": "echo", "arguments": {"value": "안녕"}})
        resp = await client.post("/", json=body, headers=_auth_headers())
        assert resp.status_code == 200
        payload = resp.json()
        assert "error" not in payload
        assert payload["result"] == {"echo": "안녕"}

    async def test_tools_call_missing_arguments_defaults_to_empty_dict(self, client):
        """arguments 미지정 시 빈 dict 로 호출되어야 한다 (방어적 기본값)."""
        body = _rpc_body("tools/call", {"name": "echo"})
        resp = await client.post("/", json=body, headers=_auth_headers())
        assert resp.status_code == 200
        # value 가 없으므로 echo 는 None.
        assert resp.json()["result"] == {"echo": None}


# ─────────────────────────────────────────────
# 인증
# ─────────────────────────────────────────────
class TestAuthentication:
    """Bearer 인증 fail-closed 동작을 검증한다."""

    async def test_missing_auth_header_returns_401(self, client):
        """Authorization 헤더가 없으면 401 이어야 한다."""
        resp = await client.post("/", json=_rpc_body("tools/list"))
        assert resp.status_code == 401

    async def test_wrong_api_key_returns_401(self, client):
        """잘못된 Bearer 키는 401 이어야 한다."""
        resp = await client.post(
            "/", json=_rpc_body("tools/list"), headers={"Authorization": "Bearer wrong-key"}
        )
        assert resp.status_code == 401

    async def test_malformed_auth_header_returns_401(self, client):
        """'Bearer ' 형식이 아닌 헤더(키만 전송)도 거부되어야 한다."""
        resp = await client.post(
            "/", json=_rpc_body("tools/list"), headers={"Authorization": _API_KEY}
        )
        assert resp.status_code == 401

    async def test_health_endpoint_does_not_require_auth(self, client):
        """/health 는 인증 없이도 호출 가능해야 한다 (LAN 모니터링용)."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


# ─────────────────────────────────────────────
# 에러 코드 매핑
# ─────────────────────────────────────────────
class TestErrorMapping:
    """예외/잘못된 요청이 표준 JSON-RPC 에러 코드로 매핑되는지 검증한다."""

    async def test_unknown_tool_returns_method_not_found(self, client):
        """존재하지 않는 도구 → -32601."""
        body = _rpc_body("tools/call", {"name": "nope", "arguments": {}})
        resp = await client.post("/", json=body, headers=_auth_headers())
        assert resp.status_code == 200
        assert resp.json()["error"]["code"] == JSONRPC_METHOD_NOT_FOUND

    async def test_value_error_maps_to_invalid_params(self, client):
        """도구가 ValueError → -32602 (invalid params)."""
        body = _rpc_body("tools/call", {"name": "bad_input", "arguments": {}})
        resp = await client.post("/", json=body, headers=_auth_headers())
        err = resp.json()["error"]
        assert err["code"] == JSONRPC_INVALID_PARAMS
        # 메시지에 도구가 던진 사유가 실려 모델 자가교정에 도움이 되어야 한다.
        assert "잘못" in err["message"]

    async def test_runtime_error_maps_to_internal_error(self, client):
        """도구가 RuntimeError → -32603 (internal error)."""
        body = _rpc_body("tools/call", {"name": "boom", "arguments": {}})
        resp = await client.post("/", json=body, headers=_auth_headers())
        assert resp.json()["error"]["code"] == JSONRPC_INTERNAL_ERROR

    async def test_unknown_method_returns_method_not_found(self, client):
        """지원하지 않는 method → -32601."""
        resp = await client.post("/", json=_rpc_body("tools/unknown"), headers=_auth_headers())
        assert resp.json()["error"]["code"] == JSONRPC_METHOD_NOT_FOUND

    async def test_non_json_body_returns_parse_error(self, client):
        """본문이 JSON 이 아니면 -32700 (parse error). 인증 전에 파싱하므로 401 아님."""
        resp = await client.post(
            "/",
            content=b"this-is-not-json",
            headers={**_auth_headers(), "Content-Type": "application/json"},
        )
        assert resp.json()["error"]["code"] == JSONRPC_PARSE_ERROR

    async def test_non_dict_arguments_returns_invalid_params(self, client):
        """arguments 가 객체(dict)가 아니면 -32602."""
        body = _rpc_body("tools/call", {"name": "echo", "arguments": ["not", "a", "dict"]})
        resp = await client.post("/", json=body, headers=_auth_headers())
        assert resp.json()["error"]["code"] == JSONRPC_INVALID_PARAMS


# ─────────────────────────────────────────────
# shutdown_hooks (lifespan)
# ─────────────────────────────────────────────
class TestShutdownHooks:
    """lifespan 종료 시 등록된 정리 훅이 await 되는지 검증한다."""

    async def test_shutdown_hooks_are_awaited_on_lifespan_exit(self):
        """앱 종료 시 shutdown_hooks 가 호출되어야 한다 (풀/클라이언트 정리).

        httpx.ASGITransport 는 lifespan 이벤트를 구동하지 않으므로
        (asgi-lifespan 미설치), 앱의 lifespan 컨텍스트를 직접 진입/이탈시켜
        startup → shutdown 경로를 검증한다.
        """
        called: list[str] = []

        async def _hook() -> None:
            called.append("closed")

        app = create_mcp_app(
            tools=[_EchoTool()],
            api_key=_API_KEY,
            title="Shutdown Test",
            shutdown_hooks=[_hook],
        )

        # FastAPI/Starlette 의 lifespan 컨텍스트를 직접 구동한다.
        async with app.router.lifespan_context(app):
            # lifespan startup(yield 이전) 후 — 아직 종료 훅 미호출.
            assert called == []
        # 컨텍스트 이탈 시 lifespan shutdown(yield 이후) → 훅 호출.
        assert called == ["closed"]

    async def test_shutdown_hook_failure_does_not_break_other_hooks(self):
        """한 종료 훅이 예외를 던져도 나머지 훅은 정리되어야 한다 (종료 경로 격리)."""
        called: list[str] = []

        async def _boom_hook() -> None:
            raise RuntimeError("정리 실패")

        async def _ok_hook() -> None:
            called.append("ok")

        app = create_mcp_app(
            tools=[_EchoTool()],
            api_key=_API_KEY,
            title="Shutdown Failure Test",
            shutdown_hooks=[_boom_hook, _ok_hook],
        )

        # boom 훅이 실패해도 lifespan 종료가 예외를 전파하지 않고 ok 훅을 호출해야 한다.
        async with app.router.lifespan_context(app):
            pass
        assert called == ["ok"]
