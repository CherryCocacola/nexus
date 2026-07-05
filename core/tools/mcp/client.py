"""
McpClient — JSON-RPC 2.0 over LAN HTTP/SSE 저수준 클라이언트.

이 파일이 하는 일 (한 줄 요약):
  Nexus가 같은 LAN 안에 있는 "MCP 서버"에게 HTTP로 말을 걸어서,
  그 서버가 제공하는 외부 도구를 목록으로 받아오고(list_tools),
  그 도구를 하나씩 실행시키는(call_tool) 통신 담당 계층이다.

MCP(Model Context Protocol)란?
  LLM이 외부 도구·데이터에 접근하도록 표준화한 프로토콜.
  통신 규약은 JSON-RPC 2.0(요청/응답 JSON 포맷)을 쓴다.
  Nexus는 이 프로토콜 전체가 아니라, 실제로 필요한 두 메서드만
  PoC(개념검증) 수준으로 최소 구현한다:
    - tools/list  : 원격 서버가 제공하는 도구 목록 + 입력 스키마 조회
    - tools/call  : 원격 도구 1개를 인자와 함께 실행

주요 구성:
  - McpClient 클래스   : 원격 MCP 서버 1개당 인스턴스 1개.
  - list_tools()       : tools/list 호출 (공개 API)
  - call_tool()        : tools/call 호출 (공개 API)
  - aclose()           : httpx 커넥션 정리 (공개 API)
  - _rpc()             : JSON-RPC 요청 전송 + 에러 정규화 (내부)
  - _parse_response()  : 단일 JSON / SSE 응답 파싱 (내부)

설계 경계 (매우 중요):
  MCP JSON-RPC 프로토콜의 "날것(raw)"은 이 파일과 adapter.py 안에만 갇혀 있다.
  즉 query_loop·executor·QueryEngine 같은 상위 오케스트레이션 계층은
  이 클라이언트의 존재를 전혀 모르며, 오직 OpenAI tool_calls 표준 계약만
  주고받는다(프로젝트 규칙 P3 유지). 상위가 MCP 세부를 몰라야 나중에
  프로토콜이 바뀌어도 이 두 파일만 고치면 된다.

에어갭(폐쇄망) 준수:
  Nexus는 외부 네트워크 연결이 금지된다. 그래서 클라이언트를 만드는 순간
  base_url의 hostname이 LAN 대역인지 다시 검증한다(2단계 검증의 2단계).
  비-LAN 주소면 ValueError를 던져 연결 시도 자체를 막는다(fail-closed).

HTTP 호출 스타일:
  core/model/inference.py(LocalModelProvider)와 일부러 동일하게
  httpx.AsyncClient + Bearer 인증 헤더 방식을 쓴다. 코드베이스 전체가
  같은 패턴이면 유지보수가 쉽기 때문이다.

작성자: 이현수 / 작성일: 2026-07-05
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

    하나의 인스턴스는 하나의 base_url(=하나의 MCP 서버)에 묶인다.
    여러 서버에 붙어야 하면 서버마다 McpClient를 따로 만든다.

    JSON-RPC 2.0 요청 형식 (서버에 이런 JSON을 POST한다):
      {"jsonrpc": "2.0", "id": <n>, "method": "...", "params": {...}}
      - id     : 요청마다 1씩 증가하는 정수(응답 짝맞춤용).
      - method : "tools/list" 또는 "tools/call".
      - params : 메서드별 인자.

    서버 응답은 두 형태가 올 수 있어 둘 다 처리한다:
      1. 단일 JSON-RPC 응답 (Content-Type: application/json) — 가장 흔함, 우선 처리.
      2. SSE 스트림 (text/event-stream) — "data: {...}" 라인 형태로 오면 파싱.
    어느 쪽이 오든 상위 코드는 동일한 result만 받도록 이 클래스가 흡수한다.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "local-key",
        timeout: float = 5.0,
    ):
        """
        McpClient를 초기화하면서 LAN 재검증과 httpx 클라이언트 준비를 한다.

        Args:
            base_url: 원격 MCP 서버의 LAN URL. 비-LAN이면 ValueError를 던진다.
            api_key: LAN 내부 인증 키. 매 요청의 Bearer 헤더로 전송된다.
                     (에어갭 내부라 값 자체는 형식적이며 기본값 "local-key".)
            timeout: 기본 연결/요청 타임아웃(초). 호출별로 덮어쓸 수 있다.

        왜 LAN 재검증을 또 하나:
          설정 로드 시 McpConfig 검증(1단계)이 이미 한 번 걸러 준다. 하지만
          테스트나 다른 모듈이 코드에서 직접 McpClient(base_url=...)를 만드는
          경로도 존재한다. 클라이언트 스스로도 비-LAN URL을 거부해야, 어떤
          경로로 만들어지든 외부 연결이 원천 차단된다(방어의 이중화).
        """
        # 표준 라이브러리 urlparse로 URL에서 hostname만 뽑아낸다.
        # 함수 안에서 지역 import 하는 이유: 모듈 최상단을 가볍게 유지하고,
        # 이 헬퍼는 __init__에서 한 번만 쓰이므로 필요할 때만 로드한다.
        from urllib.parse import urlparse

        # hostname이 None일 수 있어(잘못된 URL) or ""로 빈 문자열 폴백 처리.
        hostname = urlparse(base_url).hostname or ""
        if not is_lan_hostname(hostname):
            # 에어갭 fail-closed — 외부 주소면 아예 객체 생성 단계에서 거부한다.
            raise ValueError(
                f"MCP base_url '{base_url}'은(는) LAN 대역이 아닙니다 "
                f"(에어갭 위반). localhost/10.x/172.16~31.x/192.168.x만 허용."
            )

        # 뒤에 경로를 붙일 때 "//"가 생기지 않도록 끝의 슬래시는 제거해 저장.
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

        # JSON-RPC 요청 id 카운터. 요청을 보낼 때마다 1씩 올려 고유 id를 부여한다.
        # (인스턴스 단위라 서버별로 독립적으로 증가한다.)
        self._id_counter: int = 0

        # 재사용할 httpx 비동기 클라이언트. inference.py와 같은 스타일로,
        # connect/read/write/pool 네 구간 모두 동일한 기본 타임아웃을 건다.
        # 커넥션 풀을 재사용하므로 요청마다 새로 만드는 것보다 효율적이다.
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
        MCP tools/list를 호출해 원격 서버가 제공하는 도구 목록을 조회한다.

        흐름:
          1. _rpc로 "tools/list" 요청을 보낸다(params는 빈 객체).
          2. 응답에서 도구 배열을 꺼낸다(서버 구현 차이를 흡수).
          3. 각 도구를 {name, description, inputSchema} 형태로 정규화한다.

        Returns:
            각 도구의 {name, description, inputSchema} dict 리스트.
            서버 응답에 description/inputSchema가 빠져 있으면 기본값으로 보정한다.
            (이 리스트는 상위 adapter가 Nexus 내부 도구 형식으로 변환한다.)
        """
        result = await self._rpc("tools/list", {})

        # MCP 표준은 result.tools가 도구 배열이다. 다만 일부 서버는 result
        # 자체를 배열로 주기도 해서, 둘 다 받아들이도록 폴백 처리한다.
        if isinstance(result, dict):
            raw_tools = result.get("tools", [])
        elif isinstance(result, list):
            raw_tools = result
        else:
            raw_tools = []

        # 받아온 raw 도구들을 하나씩 검사·정규화해 깨끗한 리스트로 만든다.
        tools: list[dict[str, Any]] = []
        for t in raw_tools:
            # dict가 아니거나 name이 없는 항목은 도구로 쓸 수 없으니 건너뛴다.
            # (신뢰할 수 없는 원격 응답에 대한 방어적 필터링.)
            if not isinstance(t, dict) or "name" not in t:
                continue  # 이름 없는 항목은 무시 (방어적)
            tools.append(
                {
                    "name": t["name"],
                    # description이 없으면 빈 문자열로 채워 키 존재를 보장한다.
                    "description": t.get("description", ""),
                    # MCP는 카멜케이스 inputSchema를 쓴다. 값이 없거나 falsy면
                    # "빈 객체 스키마"로 보정해 상위 코드가 항상 스키마를 갖게 한다.
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
        MCP tools/call을 호출해 원격 도구 1개를 실행하고 결과를 받아온다.

        Args:
            name: 원격 도구 이름. Nexus 내부에서 쓰는 mcp__ 접두사는 뗀
                  "원본 이름"이어야 한다(접두사 제거는 상위 adapter 책임).
            arguments: 도구에 넘길 입력 인자 dict. None이면 빈 dict로 보낸다.
            timeout: 이 호출에만 적용할 타임아웃(초). None이면 인스턴스 기본값.
                     (도구마다 실행 시간이 달라 호출자가 조정할 수 있게 열어 둠.)

        Returns:
            JSON-RPC 응답의 result 필드 = 원격 도구의 실행 결과(형태는 도구별).
        """
        # tools/call의 params는 MCP 규격상 name + arguments 두 키를 갖는다.
        params = {"name": name, "arguments": arguments or {}}
        return await self._rpc("tools/call", params, timeout=timeout)

    async def aclose(self) -> None:
        """
        내부 httpx 클라이언트를 닫아 커넥션 풀을 반환한다.

        더 이상 이 클라이언트를 쓰지 않을 때 반드시 호출해야 소켓 누수가 없다.
        (상위 adapter의 종료/정리 로직에서 호출된다.)
        """
        await self._client.aclose()

    # ─── 내부: JSON-RPC 호출 ───

    async def _rpc(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,  # noqa: ASYNC109 — 호출별 타임아웃 전달용(httpx에 위임)
    ) -> Any:
        """
        JSON-RPC 2.0 요청 1건을 POST로 전송하고, 응답의 result만 반환한다.

        이 메서드가 list_tools/call_tool의 공통 엔진이다. 요청 조립,
        인증 헤더 부착, HTTP 전송, 예외 정규화, 응답 파싱까지 한곳에 모아
        두어 두 공개 API가 중복 없이 동일한 규칙으로 동작하게 한다.

        Args:
            method: JSON-RPC 메서드 이름("tools/list" 또는 "tools/call").
            params: 메서드 인자 dict.
            timeout: 이 호출에만 적용할 타임아웃(None이면 인스턴스 기본값).

        Returns:
            JSON-RPC 응답의 result 값. 응답이 dict가 아니면 그대로 반환.

        에러 정규화 정책(상위 어댑터가 종류별로 일관되게 처리하도록):
          - 연결 실패/HTTP 오류/JSON-RPC error → ConnectionError
          - 타임아웃 → TimeoutError (파이썬 3.11+에서 asyncio.TimeoutError와 동일 객체)
        bare except 금지(anti-pattern #8): 구체 예외만 포착해 위 두 종류로 변환한다.
        """
        # 요청 id를 1 증가시켜 이번 요청의 고유 번호로 사용한다.
        self._id_counter += 1
        request_id = self._id_counter
        # JSON-RPC 2.0 규격의 요청 본문(payload)을 조립한다.
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        # 헤더 구성:
        #  - Content-Type: 우리가 보내는 본문이 JSON임을 알림.
        #  - Accept: 응답으로 JSON 또는 SSE 둘 다 받을 수 있다고 명시.
        #  - Authorization: LAN 내부 인증용 Bearer 토큰.
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
        }

        # 이번 호출에 실제로 쓸 타임아웃. 인자로 안 주면 인스턴스 기본값 사용.
        effective_timeout = timeout if timeout is not None else self.timeout

        # HTTP 전송 구간. 발생 가능한 예외를 종류별로 잡아 정규화한다.
        try:
            response = await self._client.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=effective_timeout,
            )
            # 4xx/5xx 상태 코드면 여기서 HTTPStatusError를 일으켜 아래에서 처리.
            response.raise_for_status()
        except httpx.TimeoutException as e:
            # 타임아웃은 TimeoutError로 정규화한다. 어댑터가 "느림"을 일관되게
            # 다룰 수 있고, 3.11+에서는 asyncio.TimeoutError와 같은 객체다.
            raise TimeoutError(
                f"MCP {method} 타임아웃({effective_timeout}s): {self.base_url}"
            ) from e
        except httpx.HTTPStatusError as e:
            # 서버가 응답은 했지만 오류 상태 코드(예: 401, 500)를 준 경우.
            raise ConnectionError(
                f"MCP {method} HTTP 오류 {e.response.status_code}: {self.base_url}"
            ) from e
        except httpx.HTTPError as e:
            # 위 두 경우를 뺀 나머지 httpx 오류(연결 거부·DNS 실패 등).
            # httpx.HTTPError는 상위 예외라 마지막에 둬야 앞의 구체 예외가 먼저 잡힌다.
            raise ConnectionError(f"MCP {method} 연결 실패: {self.base_url} ({e})") from e

        # 전송 성공 → 본문을 JSON-RPC 응답 메시지로 파싱(단일 JSON 우선, SSE 폴백).
        message = self._parse_response(response)

        # 응답 안에 JSON-RPC error 객체가 있으면(원격 도구 실행 실패 등)
        # ConnectionError로 정규화해 상위가 성공 응답과 구분하게 한다.
        if isinstance(message, dict) and message.get("error"):
            err = message["error"]
            # error가 표준 형태({message: ...})면 message를, 아니면 통째로 문자열화.
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise ConnectionError(f"MCP {method} 원격 오류: {err_msg}")

        # 정상 응답이면 result만 꺼내 돌려준다. dict가 아닌 특이 응답은 그대로 반환.
        if isinstance(message, dict):
            return message.get("result")
        return message

    def _parse_response(self, response: httpx.Response) -> Any:
        """
        HTTP 응답 본문을 JSON-RPC 메시지(dict 등)로 파싱해 돌려준다.

        서버가 응답을 두 가지 형태로 줄 수 있어 Content-Type으로 분기한다:
          1. 단일 JSON 응답 — 본문 전체를 그대로 json.loads.
          2. SSE 스트림(text/event-stream) — "data: {...}" 라인들 중에서
             마지막으로 유효한 JSON-RPC 메시지를 최종 결과로 채택한다.

        PoC 수준 단순화:
          SSE라도 스트림을 실시간으로 소비하지 않고, 응답 전체를 받은 뒤
          줄 단위로 파싱한다. 진짜 스트리밍 소비는 PoC 범위 밖이라
          일부러 과설계하지 않았다.

        Args:
            response: raise_for_status를 통과한 httpx 응답 객체.

        Returns:
            파싱된 JSON-RPC 메시지(보통 dict). 파싱 실패 시 ConnectionError.
        """
        content_type = response.headers.get("content-type", "")
        text = response.text

        # 경로 1: 단일 JSON 응답 — 가장 흔하다. Content-Type에 SSE 표시가 없으면 여기.
        if "text/event-stream" not in content_type:
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                # JSON이 깨졌으면 원격이 비정상 응답을 준 것 → 연결 오류로 취급.
                # 디버깅용으로 본문 앞 200자를 함께 남긴다.
                raise ConnectionError(
                    f"MCP 응답 JSON 파싱 실패: {e} (본문 앞부분: {text[:200]!r})"
                ) from e

        # 경로 2: SSE 응답 — "data:" 로 시작하는 라인만 훑어 마지막 유효 JSON을 채택.
        # 왜 "마지막"인가: 스트림 중간에 진행 이벤트가 여러 개 와도, 최종 결과는
        # 보통 마지막 data 메시지이기 때문이다.
        last_message: Any = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            # SSE 데이터 라인만 관심 대상. 그 외(빈 줄, event: 등)는 건너뛴다.
            if not line.startswith("data:"):
                continue
            # "data:" 접두어를 떼고 공백 정리해 실제 payload 문자열만 남긴다.
            data_str = line[len("data:") :].strip()
            # 빈 데이터거나 스트림 종료 신호([DONE])면 무시.
            if not data_str or data_str == "[DONE]":
                continue
            try:
                last_message = json.loads(data_str)
            except json.JSONDecodeError:
                # 부분 전송·비정형 data 라인은 조용히 건너뛴다(방어적).
                continue

        # data 라인을 하나도 못 건졌으면 정상 응답으로 볼 수 없다.
        if last_message is None:
            raise ConnectionError(
                f"MCP SSE 응답에서 유효한 data 메시지를 찾지 못함: {text[:200]!r}"
            )
        return last_message
