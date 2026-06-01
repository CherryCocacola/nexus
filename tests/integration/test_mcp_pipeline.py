"""
MCP 통합 테스트 — 클라이언트 에어갭 경계 + 연결 관리자 + 5계층 권한 통합.

검증 대상 (read-only MCP 등록 필터 회귀 반영):
  1. McpClient: 비-LAN base_url 생성 시 ValueError (에어갭 fail-closed 2단계).
  2. McpConnectionManager.connect_and_register: enabled 서버의 도구를
     registry에 등록하고, name이 mcp__{server}__{tool} 형식으로 이름순 정렬에 포함.
  3. 서버별 격리(예외 범위 축소): list_tools가 (ConnectionError/TimeoutError/
     ValueError/OSError) 중 하나면 그 서버만 빈 리스트로 격리하고 계속한다.
     그 외 예외(예: TypeError)는 격리하지 않고 전파시켜 결함을 드러낸다.
  4. read-only 등록 필터 (신규 회귀): 신뢰 통제가 "권한 판단"에서 "등록 여부"로
     이동했다.
     - trust.read_only=True 서버 → 등록(어댑터 is_read_only=True).
     - read_only=False + allow_write=False 서버 → 등록 안 됨(빈 리스트,
       list_tools 미호출). fail-closed.
     - read_only=False + allow_write=True 서버 → 등록(어댑터 is_read_only=False).
  5. 5계층 PermissionPipeline 통합 — Layer 2가 항상 ALLOW이므로 모든 MCP 도구가
     Layer 5 모드 보정에 도달한다 (회귀 방지 핵심):
     DEFAULT=ASK / PLAN=DENY / BYPASS=ALLOW.
     read-only 도구든 allow_write로 등록된 쓰기 도구든 동일하게 표준 모드
     정책을 받는다 (직전엔 쓰기 ASK 단락으로 PLAN=DENY가 무력화됐었음 — 복구 확인).

McpClient는 connection_manager 모듈 내 참조를 patch하여 실제 네트워크를 차단한다.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from core.config import McpConfig, McpServerConfig
from core.permission.pipeline import PermissionPipeline
from core.permission.types import (
    PermissionBehavior,
    PermissionContext,
    PermissionMode,
)
from core.tools.base import (
    PermissionBehavior as ToolPermissionBehavior,
)
from core.tools.base import (
    PermissionResult,
    ToolUseContext,
)
from core.tools.mcp.adapter import McpToolAdapter
from core.tools.mcp.client import McpClient
from core.tools.mcp.connection_manager import McpConnectionManager
from core.tools.registry import ToolRegistry


# ─────────────────────────────────────────────
# McpClient — 에어갭 경계 (2단계 검증)
# ─────────────────────────────────────────────
class TestMcpClientAirgapBoundary:
    """McpClient 생성 시점의 LAN URL 재검증을 확인한다."""

    def test_mcp_client_non_lan_url_raises_value_error(self):
        """비-LAN base_url로 McpClient를 만들면 ValueError여야 한다 (에어갭 fail-closed)."""
        with pytest.raises(ValueError):
            McpClient(base_url="https://api.openai.com")

    def test_mcp_client_public_ip_raises_value_error(self):
        """공인 IP(8.8.8.8) base_url도 거부되어야 한다."""
        with pytest.raises(ValueError):
            McpClient(base_url="http://8.8.8.8:9000")

    def test_mcp_client_lan_url_constructs_successfully(self):
        """LAN base_url이면 McpClient 생성이 성공해야 한다."""
        client = McpClient(base_url="http://192.168.10.39:9000")
        # 후행 슬래시 제거 + LAN 통과 확인
        assert client.base_url == "http://192.168.10.39:9000"


# ─────────────────────────────────────────────
# 가짜 McpClient — connect_and_register용
# ─────────────────────────────────────────────
def _make_fake_client_factory(tools_by_server: dict[str, Any]):
    """
    McpClient를 대체할 가짜 팩토리를 만든다.

    base_url의 hostname으로 어느 서버인지 식별할 수 없으므로,
    connection_manager가 McpClient(base_url=..., api_key=..., timeout=...) 형태로
    호출할 때 base_url을 키로 list_tools 동작을 분기한다.

    tools_by_server: {base_url: tools_list_or_exception}
    """

    def factory(base_url: str, api_key: str = "local-key", timeout: float = 5.0):
        spec = tools_by_server[base_url]
        client = AsyncMock()
        if isinstance(spec, Exception):
            # list_tools가 예외를 던지는 서버 (격리 검증용)
            client.list_tools = AsyncMock(side_effect=spec)
        else:
            client.list_tools = AsyncMock(return_value=spec)
        client.aclose = AsyncMock()
        return client

    return factory


# ─────────────────────────────────────────────
# McpConnectionManager — 발견·등록
# ─────────────────────────────────────────────
class TestMcpConnectionManagerRegister:
    """connect_and_register의 등록·정렬·격리 동작을 검증한다."""

    @pytest.mark.asyncio
    async def test_connect_and_register_registers_adapters_with_mcp_names(self):
        """enabled read-only LAN 서버의 도구가 mcp__{server}__{tool} 이름으로 등록되어야 한다.

        read-only 등록 필터 회귀: trust.read_only=True인 서버만 자동 등록되므로
        명시적으로 trust를 설정한다.
        """
        config = McpConfig(
            enabled=True,
            servers=[
                McpServerConfig(
                    name="db",
                    base_url="http://192.168.10.39:9000",
                    enabled=True,
                    trust={"read_only": True},
                ),
            ],
        )
        fake_tools = [
            {"name": "query", "description": "조회", "inputSchema": {"type": "object"}},
            {"name": "count", "description": "건수", "inputSchema": {"type": "object"}},
        ]
        factory = _make_fake_client_factory({"http://192.168.10.39:9000": fake_tools})

        registry = ToolRegistry()
        manager = McpConnectionManager(config)
        # connection_manager 모듈이 참조하는 McpClient를 가짜 팩토리로 교체.
        with patch("core.tools.mcp.connection_manager.McpClient", side_effect=factory):
            registered = await manager.connect_and_register(registry)

        # 반환 dict 검증 — db 서버에 2개 도구.
        assert registered["db"] == ["mcp__db__query", "mcp__db__count"]

        # get_all_tools는 이름순 정렬을 보장 — mcp__ 도구가 포함되고 정렬돼야 한다.
        all_names = [t.name for t in registry.get_all_tools()]
        assert "mcp__db__query" in all_names
        assert "mcp__db__count" in all_names
        assert all_names == sorted(all_names)

    @pytest.mark.asyncio
    async def test_connect_and_register_skips_disabled_server(self):
        """개별 enabled=False 서버는 건너뛰어야 한다 (등록 dict에 키 없음)."""
        config = McpConfig(
            enabled=True,
            servers=[
                McpServerConfig(name="off", base_url="http://10.0.0.5:9000", enabled=False),
            ],
        )
        registry = ToolRegistry()
        manager = McpConnectionManager(config)
        # 비활성 서버는 McpClient를 만들지 않으므로 patch 없이도 안전하지만, 방어적으로 patch.
        with patch("core.tools.mcp.connection_manager.McpClient", side_effect=AssertionError):
            registered = await manager.connect_and_register(registry)

        assert "off" not in registered
        assert registry.tool_count == 0

    @pytest.mark.asyncio
    async def test_connect_and_register_skips_writable_server_without_allow_write(self):
        """read_only=False + allow_write=False 서버는 등록되지 않아야 한다 (fail-closed).

        신뢰 통제가 등록 단계로 이동했다: trust.read_only가 False이고 운영자가
        allow_write도 켜지 않은 쓰기 가능 서버는 도구 풀에 들어오지 못한다.
        registered[name]==[]이고, list_tools조차 호출되지 않아야 한다(연결 자체 안 함).
        """
        config = McpConfig(
            enabled=True,
            servers=[
                # trust 미설정 → read_only=False, allow_write 미설정 → False
                McpServerConfig(name="writer", base_url="http://192.168.1.30:9000", enabled=True),
            ],
        )
        fake_tools = [{"name": "exec", "description": "쓰기", "inputSchema": {"type": "object"}}]
        factory = _make_fake_client_factory({"http://192.168.1.30:9000": fake_tools})

        registry = ToolRegistry()
        manager = McpConnectionManager(config)
        with patch("core.tools.mcp.connection_manager.McpClient", side_effect=factory) as mock_cls:
            registered = await manager.connect_and_register(registry)

        # 등록 안 됨 — 빈 리스트.
        assert registered["writer"] == []
        # registry에 아무 도구도 없어야 한다.
        assert registry.tool_count == 0
        # 연결(McpClient 생성) 자체가 일어나지 않아야 한다 → list_tools 미호출 보장.
        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_and_register_registers_writable_server_with_allow_write(self):
        """read_only=False + allow_write=True 서버는 등록되고 어댑터 is_read_only=False여야 한다.

        명시적 허용 경로: 운영자가 allow_write=True로 쓰기 서버를 의도적으로 켜면
        등록을 허용한다. 이때 어댑터는 is_read_only=False로 생성되어, 최종 권한은
        표준 5계층 파이프라인이 결정한다.
        """
        config = McpConfig(
            enabled=True,
            servers=[
                McpServerConfig(
                    name="writer",
                    base_url="http://192.168.1.31:9000",
                    enabled=True,
                    allow_write=True,
                ),
            ],
        )
        fake_tools = [{"name": "exec", "description": "쓰기", "inputSchema": {"type": "object"}}]
        factory = _make_fake_client_factory({"http://192.168.1.31:9000": fake_tools})

        registry = ToolRegistry()
        manager = McpConnectionManager(config)
        with patch("core.tools.mcp.connection_manager.McpClient", side_effect=factory):
            registered = await manager.connect_and_register(registry)

        # 등록됨.
        assert registered["writer"] == ["mcp__writer__exec"]
        # 등록된 어댑터의 is_read_only는 False여야 한다 (쓰기 도구임을 표시).
        tool = registry.find_tool("mcp__writer__exec")
        assert tool is not None
        assert tool.is_read_only is False

    @pytest.mark.asyncio
    async def test_connect_and_register_read_only_server_adapter_is_read_only(self):
        """trust.read_only=True 서버는 등록되고 어댑터 is_read_only=True(불변)여야 한다."""
        config = McpConfig(
            enabled=True,
            servers=[
                McpServerConfig(
                    name="reader",
                    base_url="http://192.168.1.32:9000",
                    enabled=True,
                    trust={"read_only": True},
                ),
            ],
        )
        fake_tools = [{"name": "query", "description": "조회", "inputSchema": {"type": "object"}}]
        factory = _make_fake_client_factory({"http://192.168.1.32:9000": fake_tools})

        registry = ToolRegistry()
        manager = McpConnectionManager(config)
        with patch("core.tools.mcp.connection_manager.McpClient", side_effect=factory):
            registered = await manager.connect_and_register(registry)

        assert registered["reader"] == ["mcp__reader__query"]
        tool = registry.find_tool("mcp__reader__query")
        assert tool is not None
        assert tool.is_read_only is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "isolated_exc",
        [
            ConnectionError("연결 실패"),
            TimeoutError("타임아웃"),
            ValueError("비-LAN URL"),
            OSError("소켓 오류"),
        ],
    )
    async def test_connect_and_register_isolates_failing_server(self, isolated_exc):
        """예측 가능한 운영 예외(4종)는 그 서버만 격리하고 다른 서버는 정상 등록.

        v7.2에서 except 범위가 (ConnectionError/TimeoutError/ValueError/OSError)로
        축소되었다 — 이 4종은 모두 "해당 서버만 빈 리스트 + 계속" 동작이어야 한다.
        """
        config = McpConfig(
            enabled=True,
            servers=[
                McpServerConfig(
                    name="good",
                    base_url="http://192.168.1.10:9000",
                    enabled=True,
                    trust={"read_only": True},
                ),
                McpServerConfig(
                    name="bad",
                    base_url="http://192.168.1.11:9000",
                    enabled=True,
                    trust={"read_only": True},
                ),
            ],
        )
        good_tools = [{"name": "ping", "description": "", "inputSchema": {"type": "object"}}]
        factory = _make_fake_client_factory(
            {
                "http://192.168.1.10:9000": good_tools,
                "http://192.168.1.11:9000": isolated_exc,
            }
        )

        registry = ToolRegistry()
        manager = McpConnectionManager(config)
        with patch("core.tools.mcp.connection_manager.McpClient", side_effect=factory):
            registered = await manager.connect_and_register(registry)

        # good 서버는 정상 등록, bad 서버는 빈 리스트(격리).
        assert registered["good"] == ["mcp__good__ping"]
        assert registered["bad"] == []
        # bad 서버 도구는 registry에 등록되지 않아야 한다.
        all_names = [t.name for t in registry.get_all_tools()]
        assert all_names == ["mcp__good__ping"]

    @pytest.mark.asyncio
    async def test_connect_and_register_propagates_unexpected_exception(self):
        """격리 대상이 아닌 예외(TypeError)는 격리하지 않고 전파되어야 한다.

        v7.2 예외 범위 축소의 핵심: AttributeError/TypeError 같은 예상치 못한
        버그가 조용히 빈 리스트로 묻히면 결함을 놓친다. 이런 예외는 의도적으로
        connect_and_register 밖으로 전파시켜 상위 bootstrap이 로그로 드러내게 한다.
        """
        config = McpConfig(
            enabled=True,
            servers=[
                McpServerConfig(
                    name="buggy",
                    base_url="http://192.168.1.20:9000",
                    enabled=True,
                    trust={"read_only": True},
                ),
            ],
        )
        factory = _make_fake_client_factory(
            {"http://192.168.1.20:9000": TypeError("예상치 못한 버그")}
        )

        registry = ToolRegistry()
        manager = McpConnectionManager(config)
        with patch("core.tools.mcp.connection_manager.McpClient", side_effect=factory):
            # TypeError는 격리되지 않고 그대로 전파되어야 한다.
            with pytest.raises(TypeError):
                await manager.connect_and_register(registry)


# ─────────────────────────────────────────────
# 5계층 PermissionPipeline 통합 — ToolCategory.MCP
# ─────────────────────────────────────────────
# 실제 호출하는 pipeline API:
#   PermissionPipeline(context=PermissionContext(mode=...))
#   await pipeline.check(tool, tool_input, tool_use_context) -> PermissionDecision
#   decision.behavior ∈ {ALLOW, DENY, ASK}
# ─────────────────────────────────────────────
class _FakeMcpTool(McpToolAdapter):
    """파이프라인 통합용 어댑터 — client는 호출되지 않으므로 더미.

    파이프라인은 도구를 실행하지 않고 분류(_categorize_tool)와 모드 정책만
    적용하므로, call()이 실제로 실행되지 않는다. 그러나 안전하게 더미를 둔다.

    is_read_only는 어댑터 감사 메타에만 영향을 준다(Layer 2는 항상 ALLOW):
      - True(read-only): 등록 단계에서 read-only 필터를 통과한 도구.
      - False(allow_write로 등록된 쓰기): 둘 다 Layer 2 ALLOW → Layer 5 모드 보정 도달.
    """

    def __init__(
        self,
        server_name: str = "db",
        remote_tool_name: str = "query",
        is_read_only: bool = False,
    ):
        super().__init__(
            server_name=server_name,
            remote_tool_name=remote_tool_name,
            remote_schema={"type": "object"},
            client=AsyncMock(),
            description="MCP 통합 테스트용 도구",
            is_read_only=is_read_only,
        )


def _pipeline_for_mode(mode: PermissionMode) -> PermissionPipeline:
    """지정 모드의 PermissionPipeline을 만든다."""
    ctx = PermissionContext(mode=mode, working_directory=".")
    return PermissionPipeline(context=ctx)


def _tool_use_context() -> ToolUseContext:
    """파이프라인 check()에 넘길 ToolUseContext."""
    return ToolUseContext(cwd=".", session_id="s1", tool_use_id="tu-1")


class TestMcpPermissionPipelineTrustedReadOnly:
    """신뢰된 read-only MCP 도구는 Layer 2를 통과해 Layer 5 모드 보정에 도달한다.

    is_read_only=True이면 check_permissions가 ALLOW(None 통과)를 내므로,
    최종 결정은 모드별 정책(MODE_BEHAVIOR_MAP[MCP] + Layer 5 보정)에서 나온다.
    """

    @pytest.mark.asyncio
    async def test_pipeline_default_mode_trusted_mcp_asks(self):
        """DEFAULT 모드 + 신뢰 read-only 도구는 ASK여야 한다 (MODE_BEHAVIOR_MAP[MCP]=ASK)."""
        pipeline = _pipeline_for_mode(PermissionMode.DEFAULT)
        tool = _FakeMcpTool(is_read_only=True)
        decision = await pipeline.check(tool, {"sql": "SELECT 1"}, _tool_use_context())
        assert decision.behavior == PermissionBehavior.ASK

    @pytest.mark.asyncio
    async def test_pipeline_plan_mode_trusted_mcp_denied(self):
        """PLAN 모드 + 신뢰 read-only 도구는 DENY여야 한다 (Layer 5: 쓰기 ASK→DENY).

        Layer 2를 통과해야만 Layer 5의 PLAN 보정(MCP는 write_categories에 포함)에
        도달한다. read-only 신뢰 도구만 이 경로를 탄다.
        """
        pipeline = _pipeline_for_mode(PermissionMode.PLAN)
        tool = _FakeMcpTool(is_read_only=True)
        decision = await pipeline.check(tool, {"sql": "SELECT 1"}, _tool_use_context())
        assert decision.behavior == PermissionBehavior.DENY

    @pytest.mark.asyncio
    async def test_pipeline_bypass_mode_trusted_mcp_allowed(self):
        """BYPASS_PERMISSIONS 모드 + 신뢰 read-only 도구는 ALLOW여야 한다 (Layer 5: ASK→ALLOW)."""
        pipeline = _pipeline_for_mode(PermissionMode.BYPASS_PERMISSIONS)
        tool = _FakeMcpTool(is_read_only=True)
        decision = await pipeline.check(tool, {"sql": "SELECT 1"}, _tool_use_context())
        assert decision.behavior == PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_pipeline_layer2_allow_reaches_mode_policy(self):
        """신뢰 read-only 도구의 Layer 2 ALLOW(None 통과) → Layer 3/5 모드 정책 적용.

        check_permissions가 ALLOW를 반환하면 Layer 2는 통과(None)하고,
        최종 결정은 모드 정책(DEFAULT=ASK)에서 나와야 한다.
        """
        # 어댑터의 check_permissions가 ALLOW임을 직접 확인 (계약 보증).
        tool = _FakeMcpTool(is_read_only=True)
        result: PermissionResult = await tool.check_permissions({}, _tool_use_context())
        assert result.behavior == ToolPermissionBehavior.ALLOW

        # 따라서 DEFAULT 모드 최종 결정은 ASK.
        pipeline = _pipeline_for_mode(PermissionMode.DEFAULT)
        decision = await pipeline.check(tool, {}, _tool_use_context())
        assert decision.behavior == PermissionBehavior.ASK


class TestMcpPermissionPipelineWritableReachesModePolicy:
    """allow_write로 등록된 쓰기 MCP 도구도 Layer 2 ALLOW로 통과해 Layer 5 모드 보정에 도달한다.

    회귀 방지 핵심: 직전 버전은 쓰기 도구(is_read_only=False)를 Layer 2에서 ASK로
    단락시켜 PLAN 모드의 최종 보정(ASK→DENY)을 무력화했다. 현재는 Layer 2가 항상
    ALLOW를 내므로, 쓰기 도구도 read-only 도구와 동일하게 표준 모드 정책을 받는다:
      DEFAULT=ASK / PLAN=DENY / BYPASS=ALLOW.
    """

    @pytest.mark.asyncio
    async def test_pipeline_default_mode_writable_mcp_asks(self):
        """DEFAULT 모드 + 쓰기 도구는 ASK (MODE_BEHAVIOR_MAP[MCP]=ASK)."""
        pipeline = _pipeline_for_mode(PermissionMode.DEFAULT)
        tool = _FakeMcpTool(is_read_only=False)
        decision = await pipeline.check(tool, {"sql": "SELECT 1"}, _tool_use_context())
        assert decision.behavior == PermissionBehavior.ASK

    @pytest.mark.asyncio
    async def test_pipeline_plan_mode_writable_mcp_denied(self):
        """PLAN 모드 + 쓰기 도구는 DENY여야 한다 (Layer 5: 쓰기 ASK→DENY 보정 복구).

        Layer 2 ALLOW 통과 덕분에 Layer 5의 PLAN 보정(MCP는 write_categories에
        포함)에 도달한다. 직전엔 ASK 단락으로 이 DENY가 무력화됐었다 — 복구 확인.
        """
        pipeline = _pipeline_for_mode(PermissionMode.PLAN)
        tool = _FakeMcpTool(is_read_only=False)
        decision = await pipeline.check(tool, {"sql": "SELECT 1"}, _tool_use_context())
        assert decision.behavior == PermissionBehavior.DENY

    @pytest.mark.asyncio
    async def test_pipeline_bypass_mode_writable_mcp_allowed(self):
        """BYPASS 모드 + 쓰기 도구는 ALLOW여야 한다 (Layer 5: ASK→ALLOW 보정).

        Layer 2가 더 이상 단락하지 않으므로 BYPASS 모드 보정이 정상 적용된다.
        allow_write로 의도적으로 켠 도구는 BYPASS에서 자동 허용된다.
        """
        pipeline = _pipeline_for_mode(PermissionMode.BYPASS_PERMISSIONS)
        tool = _FakeMcpTool(is_read_only=False)
        decision = await pipeline.check(tool, {"sql": "SELECT 1"}, _tool_use_context())
        assert decision.behavior == PermissionBehavior.ALLOW


class TestMcpPermissionPipelineClassification:
    """mcp__ 접두사가 ToolCategory.MCP로 분류되는지 감사 로그로 확인한다."""

    @pytest.mark.asyncio
    async def test_pipeline_classifies_mcp_prefix_as_mcp_category(self):
        """감사 로그를 통해 mcp__ 접두사가 ToolCategory.MCP로 분류됨을 확인한다."""
        pipeline = _pipeline_for_mode(PermissionMode.DEFAULT)
        tool = _FakeMcpTool(server_name="diag", remote_tool_name="latency")
        await pipeline.check(tool, {}, _tool_use_context())
        # 가장 최근 감사 로그의 카테고리가 "mcp"여야 한다.
        recent = pipeline.get_recent_audit(1)
        assert recent[0].tool_category == "mcp"
