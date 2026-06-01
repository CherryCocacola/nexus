"""
McpClient — JSON-RPC 2.0 over LAN HTTP/SSE 저수준 클라이언트.

MCP(Model Context Protocol)의 두 핵심 메서드만 PoC 수준으로 구현한다:
  - tools/list  : 원격 서버가 제공하는 도구 목록 + 스키마 조회
  - tools/call  : 원격 도구 1개 실행

설계 경계 (매우 중요):
  MCP JSON-RPC 프로토콜은 이 파일과 adapter.py 내부에만 존재한다.
  query_loop·executor·QueryEngine은 이 클라이언트를 전혀 모른다(P3 계약 유지).

에어갭 준수:
  생성 시 base_url의 hostname이 LAN 대역인지 재검증한다(2단계 검증의 2단계).
  비-LAN이면 ValueError를 던져 연결 자체를 막는다(fail-closed).

HTTP 호출 스타일은 core/model/inference.py(LocalModelProvider)와 동일하게
httpx.AsyncClient + Bearer 인증 헤더를 사용한다.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

# 의존성 방향(P2): core/tools/mcp/ → core/security (보안 헬퍼 재사용)
# LAN/에어갭 판정은 보안 관심사이므로 core/security/network_guard로 이동했다.
# tools → security는 단방향(상위 → 하위) import라 P2 위반이 아니다.
# (이전에는 config의 private _is_lan_hostname을 패키지 경계 넘어 import하던 결함이 있었다.)
from core.security.network_guard import is_lan_hostname

logger = logging.getLogger("nexus.tools.mcp.client")


class McpClient:
    """
    원격 LAN MCP 서버 1개와 통신하는 JSON-RPC 클라이언트.

    JSON-RPC 2.0 요청 형식:
      {"jsonrpc": "2.0", "id": <n>, "method": "...", "params": {...}}

    응답은 두 형태 모두 처리한다:
      1. 단일 JSON-RPC 응답 (Content-Type: application/json) — 우선
      2. SSE 스트림 (text/event-stream) — "data: {...}" 라인을 파싱
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "local-key",
        timeout: float = 5.0,
    ):
        """
        Args:
            base_url: 원격 MCP 서버의 LAN URL. 비-LAN이면 ValueError.
            api_key: LAN 내부 인증 키 (Bearer 헤더로 전송).
            timeout: 기본 연결/요청 타임아웃(초).

        왜 LAN 재검증을 또 하나: McpConfig 검증(1단계)을 통과했더라도,
        코드에서 직접 McpClient를 만드는 경로가 있을 수 있다. 클라이언트
        자체가 비-LAN URL을 거부해야 어떤 경로로도 외부 연결이 불가능하다.
        """
        from urllib.parse import urlparse

        hostname = urlparse(base_url).hostname or ""
        if not is_lan_hostname(hostname):
            # 에어갭 fail-closed — 외부 주소면 클라이언트 생성 자체를 거부
            raise ValueError(
                f"MCP base_url '{base_url}'은(는) LAN 대역이 아닙니다 "
                f"(에어갭 위반). localhost/10.x/172.16~31.x/192.168.x만 허용."
            )

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # JSON-RPC 요청 id 카운터 (인스턴스 단위로 증가)
        self._id_counter: int = 0

        # httpx 비동기 클라이언트 — inference.py와 동일한 스타일
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=timeout,
                read=timeout,
                write=timeout,
                pool=timeout,
            ),
        )

    # ─── 공개 API ───

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        MCP tools/list를 호출해 원격 도구 목록을 조회한다.

        Returns:
            각 도구의 {name, description, inputSchema} dict 리스트.
            서버 응답에 description/inputSchema가 없으면 기본값으로 보정한다.
        """
        result = await self._rpc("tools/list", {})

        # MCP 표준: result.tools 가 도구 배열. 일부 서버는 result 자체가 배열일 수 있어 폴백.
        if isinstance(result, dict):
            raw_tools = result.get("tools", [])
        elif isinstance(result, list):
            raw_tools = result
        else:
            raw_tools = []

        tools: list[dict[str, Any]] = []
        for t in raw_tools:
            if not isinstance(t, dict) or "name" not in t:
                continue  # 이름 없는 항목은 무시 (방어적)
            tools.append(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    # MCP는 inputSchema(카멜케이스) 사용. 없으면 빈 객체 스키마.
                    "inputSchema": t.get("inputSchema") or {"type": "object", "properties": {}},
                }
            )
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,  # noqa: ASYNC109 — 도구별 타임아웃은 호출자 계약(spec 인터페이스)
    ) -> Any:
        """
        MCP tools/call을 호출해 원격 도구 1개를 실행한다.

        Args:
            name: 원격 도구 이름(mcp__ 접두사 없는 원본 이름).
            arguments: 도구 입력 인자.
            timeout: 이 호출에만 적용할 타임아웃(None이면 기본 timeout).

        Returns:
            JSON-RPC result 필드(원격 도구의 실행 결과).
        """
        params = {"name": name, "arguments": arguments or {}}
        return await self._rpc("tools/call", params, timeout=timeout)

    async def aclose(self) -> None:
        """httpx 클라이언트를 정리한다(커넥션 풀 반환)."""
        await self._client.aclose()

    # ─── 내부: JSON-RPC 호출 ───

    async def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,  # noqa: ASYNC109 — 호출별 타임아웃 전달용(httpx에 위임)
    ) -> Any:
        """
        JSON-RPC 2.0 요청 1건을 POST로 전송하고 result를 반환한다.

        에러 정규화 정책(상위 어댑터가 일관되게 처리할 수 있도록):
          - 연결 실패/HTTP 오류/JSON-RPC error → ConnectionError
          - 타임아웃 → TimeoutError (3.11+에서 asyncio.TimeoutError와 동일 객체)
        bare except 금지(anti-pattern #8): 구체 예외만 포착해 정규화한다.
        """
        self._id_counter += 1
        request_id = self._id_counter
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        # Bearer 인증 + SSE도 받을 수 있음을 Accept로 명시
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
        }

        # 이 호출 전용 타임아웃(미지정 시 인스턴스 기본값)
        effective_timeout = timeout if timeout is not None else self.timeout

        try:
            response = await self._client.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=effective_timeout,
            )
            response.raise_for_status()
        except httpx.TimeoutException as e:
            # 타임아웃은 TimeoutError로 정규화 (어댑터가 일관 처리; asyncio.TimeoutError와 동일)
            raise TimeoutError(
                f"MCP {method} 타임아웃({effective_timeout}s): {self.base_url}"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(
                f"MCP {method} HTTP 오류 {e.response.status_code}: {self.base_url}"
            ) from e
        except httpx.HTTPError as e:
            # 연결 거부·DNS 실패 등 그 외 httpx 오류
            raise ConnectionError(f"MCP {method} 연결 실패: {self.base_url} ({e})") from e

        # 응답 본문을 JSON-RPC 응답으로 파싱 (단일 JSON 우선, SSE 폴백)
        message = self._parse_response(response)

        # JSON-RPC error 객체가 있으면 ConnectionError로 정규화
        if isinstance(message, dict) and message.get("error"):
            err = message["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise ConnectionError(f"MCP {method} 원격 오류: {err_msg}")

        if isinstance(message, dict):
            return message.get("result")
        return message

    def _parse_response(self, response: httpx.Response) -> Any:
        """
        HTTP 응답 본문을 JSON-RPC 메시지로 파싱한다.

        두 경로:
          1. 단일 JSON 응답 — 그대로 json.loads
          2. SSE 스트림(text/event-stream) — "data: {...}" 라인 중
             마지막으로 유효한 JSON-RPC 메시지를 채택

        PoC 수준으로 단순화: SSE라도 응답 전체를 받아 라인 단위로 파싱한다
        (진정한 스트리밍 소비는 PoC 범위 밖 — 과설계 금지).
        """
        content_type = response.headers.get("content-type", "")
        text = response.text

        # 단일 JSON 응답 — 가장 흔한 경로
        if "text/event-stream" not in content_type:
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise ConnectionError(
                    f"MCP 응답 JSON 파싱 실패: {e} (본문 앞부분: {text[:200]!r})"
                ) from e

        # SSE 응답 — "data:" 라인을 순회하며 마지막 유효 JSON을 채택
        last_message: Any = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                last_message = json.loads(data_str)
            except json.JSONDecodeError:
                # 부분/비정형 data 라인은 건너뛴다 (방어적)
                continue

        if last_message is None:
            raise ConnectionError(
                f"MCP SSE 응답에서 유효한 data 메시지를 찾지 못함: {text[:200]!r}"
            )
        return last_message
