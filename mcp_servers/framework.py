"""
공통 MCP 서버 프레임워크 — JSON-RPC 2.0 over LAN HTTP.

역할:
  사내 시스템(PostgreSQL/임베딩 RAG/문서 인제스트/진단 등)을 LAN 내부에서
  MCP(Model Context Protocol) 도구로 노출하는 "서버 측" 공통 토대다.
  이미 구현된 클라이언트(core/tools/mcp/client.py)와 정확히 짝을 이룬다.

설계 경계 (매우 중요 — 의존성 방향 P2):
  mcp_servers/ → core/ 는 허용(코어 재사용). 반대로 core/ 는 mcp_servers/ 를
  절대 import 하지 않는다. MCP 서버는 Nexus 오케스트레이터가 아니라 사내
  시스템을 감싸는 "독립 서비스"이기 때문이다.

프로토콜 정합 (client.py 와 1:1):
  - 단일 HTTP POST 엔드포인트("/")에서 JSON-RPC 2.0 디스패치.
  - 요청:  {"jsonrpc":"2.0","id":<n>,"method":"...","params":{...}}
  - tools/list → result = {"tools":[{"name","description","inputSchema"}, ...]}
                 (스키마 키는 camelCase 인 inputSchema 로 노출)
  - tools/call → params = {"name","arguments"} → result = 도구 실행 결과(JSON 직렬화 가능)
  - 성공 응답: {"jsonrpc":"2.0","id":<n>,"result":<값>}
  - 에러 응답: {"jsonrpc":"2.0","id":<n>,"error":{"code","message"}}
    (client 는 error.message 를 ConnectionError 로 정규화한다)
  - 응답 Content-Type: application/json (단일 JSON. SSE 불필요).

인증 (fail-closed):
  Authorization 헤더가 정확히 f"Bearer {api_key}" 가 아니면 거부한다.
  헤더 자체가 없거나 형식이 틀려도 거부 — LAN 내부 최소 인증.

에어갭 준수:
  서버는 LAN 주소에 bind 하며(run.py 에서 host/port 지정), 외부 네트워크/도메인
  호출 코드를 포함하지 않는다. 런타임 패키지 설치 코드도 없다.

anti-pattern #8 (bare except 금지):
  도구 실행 중 발생한 예외는 구체 타입으로 포착해 JSON-RPC error 로 변환한다.
  도구 하나가 실패해도 서버 프로세스는 죽지 않는다.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("nexus.mcp_servers.framework")


# ─────────────────────────────────────────────
# JSON-RPC 2.0 표준 에러 코드
# ─────────────────────────────────────────────
# client.py 는 error.message 만 본다(코드는 정규화에 쓰지 않음). 그래도 표준
# 코드를 정확히 채워 두면 향후 클라이언트 고도화/디버깅에 유리하다.
JSONRPC_PARSE_ERROR = -32700  # 본문이 JSON 으로 파싱되지 않음
JSONRPC_INVALID_REQUEST = -32600  # JSON-RPC 형식 위반(jsonrpc/method 누락 등)
JSONRPC_METHOD_NOT_FOUND = -32601  # 알 수 없는 method / 존재하지 않는 도구
JSONRPC_INVALID_PARAMS = -32602  # params/arguments 검증 실패
JSONRPC_INTERNAL_ERROR = -32603  # 도구 실행 중 내부 예외


# ─────────────────────────────────────────────
# McpServerTool — 서버 측 도구 1개의 추상 계약
# ─────────────────────────────────────────────
class McpServerTool(ABC):
    """
    MCP 서버가 노출하는 도구 1개.

    구현체가 채워야 할 것:
      - name:         도구 이름(클라이언트가 tools/call 의 params.name 으로 지정).
      - description:  도구 설명(모델이 언제 호출할지 판단하는 근거).
      - input_schema: JSON Schema(dict). tools/list 응답에서 camelCase
                      키 inputSchema 로 노출된다.
      - call():       실제 실행. arguments(dict)를 받아 JSON 직렬화 가능한 값을 반환.

    왜 input_schema(snake)인데 노출은 inputSchema(camel)인가:
      Python 내부 규약은 snake_case 가 자연스럽고, MCP 와이어 포맷은 camelCase
      를 요구한다(client.py 참고). 프레임워크가 tools/list 직렬화 시점에 키를
      변환해 주므로, 구현체는 파이썬다운 이름만 신경 쓰면 된다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """도구 이름(고유)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """도구 설명."""

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema(dict). tools/list 에서 inputSchema 로 노출."""

    @abstractmethod
    async def call(self, arguments: dict[str, Any]) -> Any:
        """
        도구를 실행한다.

        Args:
            arguments: 클라이언트가 보낸 입력 인자(tools/call 의 arguments).

        Returns:
            JSON 직렬화 가능한 실행 결과(dict/list/str/숫자 등).

        예외 정책:
            - 입력이 잘못됐으면 ValueError 를 던진다 → 프레임워크가 -32602 로 변환.
            - 그 외 실행 오류는 RuntimeError 등 구체 예외로 던진다 →
              프레임워크가 -32603 으로 변환. 절대 bare except 로 삼키지 않는다.
        """


# ─────────────────────────────────────────────
# JSON-RPC 응답 헬퍼 — 항상 동일한 봉투(envelope)로 직렬화
# ─────────────────────────────────────────────
def _rpc_success(request_id: Any, result: Any) -> dict[str, Any]:
    """성공 응답 봉투를 만든다."""
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """에러 응답 봉투를 만든다(client 는 message 만 사용)."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


# ─────────────────────────────────────────────
# create_mcp_app — 도구 목록 + 인증 키로 FastAPI 앱 1개 생성
# ─────────────────────────────────────────────
def create_mcp_app(
    tools: list[McpServerTool],
    api_key: str,
    title: str,
    shutdown_hooks: list[Callable[[], Awaitable[None]]] | None = None,
) -> FastAPI:
    """
    MCP 서버용 FastAPI 앱을 만든다.

    Args:
        tools:   이 서버가 노출할 McpServerTool 목록.
        api_key: Bearer 인증 키(요청 Authorization 헤더와 비교).
        title:   서버 식별용 제목(health 응답·OpenAPI 문서 제목).
        shutdown_hooks: 앱 종료 시 await 할 정리 콜백 목록(선택).
            db 풀/httpx 클라이언트 close 등 자원 해제에 쓴다. FastAPI 0.115 에서
            on_event("shutdown") 이 deprecated 라 lifespan 으로 일원화했다.

    Returns:
        단일 POST "/" 엔드포인트(JSON-RPC) + GET "/health" 를 가진 FastAPI 앱.

    도구 이름 충돌:
        같은 이름의 도구가 둘 이상이면 마지막 것이 이긴다(dict 특성). 운영
        실수를 빨리 드러내기 위해 경고 로그를 남긴다.
    """
    hooks = shutdown_hooks or []

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        """앱 수명 — 시작 시 별도 작업 없음, 종료 시 등록된 정리 훅을 await."""
        yield
        for hook in hooks:
            try:
                await hook()
            except Exception as e:  # noqa: BLE001 — 종료 경로에서 한 훅 실패가 다른 정리를 막지 않게
                logger.warning("종료 정리 훅 실패: %s: %s", type(e).__name__, e)

    app = FastAPI(title=title, version="1.0.0", lifespan=lifespan)

    # 이름 → 도구 매핑을 미리 만들어 두면 tools/call 디스패치가 O(1).
    tool_map: dict[str, McpServerTool] = {}
    for t in tools:
        if t.name in tool_map:
            logger.warning("MCP 도구 이름 충돌: '%s' (나중 등록이 우선)", t.name)
        tool_map[t.name] = t

    # ── 인증 검사 ─────────────────────────────
    def _is_authorized(request: Request) -> bool:
        """
        Authorization 헤더가 정확히 'Bearer {api_key}' 인지 확인한다.

        fail-closed: 헤더 누락/형식 오류/키 불일치는 모두 미인증으로 본다.
        """
        header = request.headers.get("authorization", "")
        return header == f"Bearer {api_key}"

    # ── JSON-RPC 단일 엔드포인트 ───────────────
    @app.post("/")
    async def jsonrpc_endpoint(request: Request) -> JSONResponse:
        """
        JSON-RPC 2.0 요청 1건을 디스패치한다.

        흐름:
          1) 본문 JSON 파싱 (실패 → -32700)
          2) Bearer 인증 (실패 → 401 + JSON-RPC error)
          3) method 분기 (tools/list, tools/call)
          4) 도구 실행 → result, 예외 → JSON-RPC error 로 변환

        항상 {"jsonrpc":"2.0","id",...} 형태의 단일 JSON 을 반환한다.
        """
        # 1) 본문 파싱 — JSON 이 아니면 파싱 에러(id 는 알 수 없어 None).
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return JSONResponse(
                _rpc_error(None, JSONRPC_PARSE_ERROR, "요청 본문이 올바른 JSON 이 아닙니다."),
                media_type="application/json",
            )

        # JSON-RPC 봉투에서 id/method/params 추출(형식 방어적으로 처리).
        if not isinstance(body, dict):
            return JSONResponse(
                _rpc_error(None, JSONRPC_INVALID_REQUEST, "JSON-RPC 요청은 객체여야 합니다."),
                media_type="application/json",
            )
        request_id = body.get("id")
        method = body.get("method")
        params = body.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        # 2) 인증 — 미인증이면 401 + JSON-RPC error. client 는 HTTP 4xx 도
        #    HTTPStatusError → ConnectionError 로 정규화하므로 어느 쪽이든 차단된다.
        if not _is_authorized(request):
            return JSONResponse(
                _rpc_error(
                    request_id,
                    JSONRPC_INVALID_REQUEST,
                    "인증 실패: 유효한 Bearer 토큰이 필요합니다.",
                ),
                status_code=401,
                media_type="application/json",
            )

        # 3) method 분기
        if method == "tools/list":
            # 각 도구의 input_schema 를 camelCase 키 inputSchema 로 노출한다.
            tool_list = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in tools
            ]
            return JSONResponse(
                _rpc_success(request_id, {"tools": tool_list}),
                media_type="application/json",
            )

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                return JSONResponse(
                    _rpc_error(
                        request_id,
                        JSONRPC_INVALID_PARAMS,
                        "arguments 는 객체(dict)여야 합니다.",
                    ),
                    media_type="application/json",
                )

            tool = tool_map.get(tool_name)
            if tool is None:
                return JSONResponse(
                    _rpc_error(
                        request_id,
                        JSONRPC_METHOD_NOT_FOUND,
                        f"존재하지 않는 도구입니다: {tool_name!r}",
                    ),
                    media_type="application/json",
                )

            # 4) 도구 실행 — 구체 예외만 포착해 JSON-RPC error 로 변환(서버 생존).
            try:
                result = await tool.call(arguments)
            except ValueError as e:
                # 입력 검증 실패 계열 → invalid params
                logger.info("도구 %s 입력 검증 실패: %s", tool_name, e)
                return JSONResponse(
                    _rpc_error(request_id, JSONRPC_INVALID_PARAMS, str(e)),
                    media_type="application/json",
                )
            except (RuntimeError, OSError, KeyError, TypeError) as e:
                # 실행 중 내부 오류 → internal error
                logger.error("도구 %s 실행 오류: %s: %s", tool_name, type(e).__name__, e)
                return JSONResponse(
                    _rpc_error(
                        request_id,
                        JSONRPC_INTERNAL_ERROR,
                        f"{type(e).__name__}: {e}",
                    ),
                    media_type="application/json",
                )

            return JSONResponse(
                _rpc_success(request_id, result),
                media_type="application/json",
            )

        # 알 수 없는 method
        return JSONResponse(
            _rpc_error(
                request_id,
                JSONRPC_METHOD_NOT_FOUND,
                f"지원하지 않는 method 입니다: {method!r}",
            ),
            media_type="application/json",
        )

    # ── 헬스체크 ──────────────────────────────
    @app.get("/health")
    async def health() -> dict[str, str]:
        """서버 가용성 점검 — 인증 없이 호출 가능(LAN 내부 모니터링용)."""
        return {"status": "healthy", "server": title}

    return app
