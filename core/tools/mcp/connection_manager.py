"""
McpConnectionManager — LAN MCP 서버 발견·등록 관리자.

부트스트랩(Phase 2) 시점에 설정된 LAN MCP 서버들에 연결하여:
  1. tools/list로 원격 도구를 발견하고
  2. 각 도구를 McpToolAdapter로 래핑한 뒤
  3. 메인 ToolRegistry에 등록한다.

fail-closed + 실패 격리:
  서버 하나의 연결 실패가 다른 서버나 본류(채팅)에 영향을 주지 않는다.
  실패한 서버는 빈 리스트로 기록하고 WARNING만 남긴다
  (v7.1 _warmup_embedding/RAG 초기화의 fire-and-forget + 격리 패턴과 동일).

의존성 방향(P2): core/tools/mcp/ → core/tools/base.py, core/tools/registry.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.tools.mcp.adapter import McpToolAdapter
from core.tools.mcp.client import McpClient

# 타입 체크 전용 — 런타임 의존을 줄인다
if TYPE_CHECKING:
    from core.config import McpConfig
    from core.tools.registry import ToolRegistry

logger = logging.getLogger("nexus.tools.mcp.connection_manager")


class McpConnectionManager:
    """
    설정된 LAN MCP 서버들에 연결하고 도구를 registry에 등록한다.

    생성한 McpClient들을 보관하여 종료 시 일괄 정리(aclose_all)할 수 있다.
    """

    def __init__(self, mcp_config: McpConfig):
        """
        Args:
            mcp_config: NexusConfig.mcp — 전역 enabled + 서버 목록.
        """
        self._config = mcp_config
        # 등록에 성공한 서버의 McpClient들 — 종료 시 정리 대상
        self._clients: list[McpClient] = []

    async def connect_and_register(self, registry: ToolRegistry) -> dict[str, list[str]]:
        """
        enabled 서버를 순회하며 도구를 발견·등록한다.

        Returns:
            {server_name: [등록된 도구 이름...]}.
            실패한 서버는 빈 리스트(본류 무영향, fail-closed).

        왜 서버별 try/except인가: 한 서버의 연결 실패(다운/타임아웃/스키마
        이상)가 다른 서버 등록까지 막으면 안 된다. 각 서버를 독립적으로
        격리해, 일부만 살아 있어도 그만큼은 도구 풀에 흡수되게 한다.
        """
        registered: dict[str, list[str]] = {}

        for server in self._config.servers:
            # 개별 enabled가 꺼진 서버는 건너뛴다 (config 검증에서 비-LAN도 이미 강등됨)
            if not server.enabled:
                continue

            # ───────────────────────────────────────────────────────────
            # 제품 정책(신뢰 경계는 "등록 단계"에서 통제):
            #   초기 제품 정책상 read-only로 신뢰된 MCP 서버의 도구만 등록한다.
            #   쓰기 가능(trust.read_only=False) 서버는 기본적으로 등록을 건너뛴다.
            #
            #   왜 등록 단계에서 막는가: 직전 강화는 쓰기 도구를 Layer 2에서 ASK로
            #   처리했는데, 그 ASK가 권한 파이프라인을 조기 단락시켜 PLAN 모드의
            #   최종 보정(ASK→DENY)을 무력화했다. 그래서 신뢰 통제를 "권한 판단"이
            #   아니라 "등록 여부"로 옮긴다 — 신뢰 안 되는 쓰기 도구는 아예 도구
            #   풀에 들어오지 못하게 한다(모델이 호출 자체 불가).
            #
            #   명시적 허용 경로: trust.read_only가 False여도 운영자가
            #   allow_write=True로 의도적으로 켰다면 등록을 허용한다(이때 어댑터는
            #   is_read_only=False로 생성되어, 최종 권한은 표준 5계층이 결정).
            # ───────────────────────────────────────────────────────────
            read_only = bool(server.trust.get("read_only", False))
            if not read_only and not server.allow_write:
                logger.warning(
                    "쓰기 가능 MCP 서버 '%s'은 초기 제품 정책상 등록하지 않음 "
                    "(read-only만 허용, 쓰기는 명시적 allow rule 필요)",
                    server.name,
                )
                registered[server.name] = []
                continue

            try:
                # McpClient 생성 단계에서 LAN URL을 다시 검증(2단계 검증).
                # 비-LAN이면 ValueError가 나며 이 서버만 스킵된다.
                client = McpClient(
                    base_url=server.base_url,
                    api_key=server.api_key,
                    timeout=self._config.connect_timeout_sec,
                )

                # tools/list — 원격 도구 발견
                remote_tools = await client.list_tools()

                names: list[str] = []
                # 어댑터의 is_read_only 플래그 결정:
                #   - read_only 서버: True (조회 전용으로 완화)
                #   - allow_write로 명시 등록된 쓰기 서버: False
                #     (쓰기 도구임을 표시 — 최종 권한은 5계층이 결정)
                adapter_read_only = read_only

                for t in remote_tools:
                    adapter = McpToolAdapter(
                        server_name=server.name,
                        remote_tool_name=t["name"],
                        remote_schema=t["inputSchema"],
                        client=client,
                        description=t.get("description", ""),
                        is_read_only=adapter_read_only,
                    )
                    # registry가 이름순 정렬을 보장하므로 등록 순서는 무관
                    registry.register(adapter)
                    names.append(adapter.name)

                # 등록 성공 시에만 client를 보관(종료 시 정리)
                self._clients.append(client)
                registered[server.name] = names
                logger.info("MCP 서버 '%s' 연결 성공: %d개 도구 등록", server.name, len(names))

            except (ConnectionError, TimeoutError, ValueError, OSError) as e:
                # 예상 가능한 운영 실패만 좁혀 포착해 이 서버만 격리한다:
                #   - ConnectionError : McpClient가 정규화한 연결/HTTP/JSON-RPC 오류
                #   - TimeoutError    : 연결/요청 타임아웃 (3.11+ asyncio.TimeoutError 동일)
                #   - ValueError      : 비-LAN URL 등 McpClient 생성 단계 거부(에어갭)
                #   - OSError         : 소켓/DNS 등 저수준 네트워크 오류
                # 이들은 "그 서버만 스킵 + WARNING"으로 처리한다(본류·타 서버 무영향).
                #
                # 왜 broad except를 쓰지 않는가(제품 기준): AttributeError/TypeError
                # 같은 예상치 못한 버그가 조용히 묻히면 결함을 놓친다. 그런 예외는
                # 의도적으로 전파시켜 상위 bootstrap의 try/except(최후 방어선)가
                # 본류를 보호하되 로그로 드러나게 한다.
                logger.warning("MCP 서버 '%s' 연결 실패 (스킵): %s", server.name, e)
                registered[server.name] = []

        return registered

    async def aclose_all(self) -> None:
        """
        보유한 모든 McpClient를 정리한다(커넥션 풀 반환).

        개별 aclose 실패가 다른 client 정리를 막지 않도록 격리한다.
        """
        for client in self._clients:
            try:
                await client.aclose()
            except (OSError, RuntimeError) as e:
                # 정리(close)는 best-effort다. 소켓/이벤트루프 수준의 예상 가능한
                # 오류만 좁혀 무시하고 다음 client 정리를 계속한다(격리 유지).
                #   - OSError    : 소켓 종료 중 저수준 오류
                #   - RuntimeError : 이미 닫힘/이벤트 루프 종료 등
                # 그 외 예상치 못한 예외는 전파시켜 결함이 묻히지 않게 한다(제품 기준).
                logger.warning("MCP 클라이언트 정리 실패 (무시): %s", e)
        self._clients.clear()
