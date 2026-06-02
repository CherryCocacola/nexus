"""
core/tools/mcp/adapter.py 의 McpToolAdapter 단위 테스트.

검증 대상 (v7.2 보안 강화 반영):
  1. name 규칙: mcp__{server}__{tool} 형식.
  2. input_schema / description / is_read_only 주입값을 그대로 노출.
  3. check_permissions(): read-only/쓰기 무관 항상 ALLOW(회귀) —
       신뢰 통제는 등록 단계(connect_and_register)로 이동했고,
       Layer 2는 최종 판단을 5계층 파이프라인에 위임한다.
     두 경우 모두 details에 감사 메타(mcp_server/remote_tool/is_read_only) 포함.
  4. validate_input(): 원격 input_schema 기반 최소 검증 —
       required 누락 → 한글 에러, 충족/빈 스키마 → None.
  5. call() 성공: client.call_tool 결과를 ToolResult.success로 정규화.
  6. call() 실패: ConnectionError/TimeoutError를 ToolResult.error로 격리.
  7. map_result(): 에러 결과를 <tool_use_error>로 래핑 (BaseTool 기본 동작).

외부 MCP 서버(httpx)는 AsyncMock client로 대체 — 실제 네트워크 호출 없음.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.tools.base import PermissionBehavior, ToolResult, ToolUseContext
from core.tools.mcp.adapter import McpToolAdapter


# ─────────────────────────────────────────────
# 공용 픽스처 — 가짜 McpClient + ToolUseContext
# ─────────────────────────────────────────────
def _make_context() -> ToolUseContext:
    """어댑터 호출에 필요한 최소 ToolUseContext를 만든다."""
    return ToolUseContext(cwd=".", session_id="test-session", tool_use_id="tu-1")


def _make_adapter(
    client: Any,
    *,
    server_name: str = "db",
    remote_tool_name: str = "query",
    remote_schema: dict[str, Any] | None = None,
    description: str = "원격 DB 조회 도구",
    is_read_only: bool = False,
) -> McpToolAdapter:
    """테스트용 McpToolAdapter를 만든다."""
    if remote_schema is None:
        remote_schema = {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        }
    return McpToolAdapter(
        server_name=server_name,
        remote_tool_name=remote_tool_name,
        remote_schema=remote_schema,
        client=client,
        description=description,
        is_read_only=is_read_only,
    )


# ─────────────────────────────────────────────
# Identity / Schema / Behavior — 주입값 노출
# ─────────────────────────────────────────────
class TestMcpAdapterIdentity:
    """어댑터의 정체성/스키마/동작 플래그 노출을 검증한다."""

    def test_name_follows_mcp_server_tool_convention(self):
        """name은 mcp__{server}__{tool} 형식이어야 한다 (권한 분류 규칙 정합)."""
        adapter = _make_adapter(AsyncMock(), server_name="db", remote_tool_name="query")
        assert adapter.name == "mcp__db__query"

    def test_input_schema_exposes_injected_schema_unchanged(self):
        """input_schema는 주입된 remote_schema를 변환 없이 그대로 노출해야 한다."""
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        adapter = _make_adapter(AsyncMock(), remote_schema=schema)
        assert adapter.input_schema == schema

    def test_description_exposes_injected_value(self):
        """description은 주입값을 그대로 노출해야 한다."""
        adapter = _make_adapter(AsyncMock(), description="설명 텍스트")
        assert adapter.description == "설명 텍스트"

    def test_is_read_only_defaults_to_false_failclosed(self):
        """is_read_only 기본값은 fail-closed로 False여야 한다."""
        adapter = _make_adapter(AsyncMock())
        assert adapter.is_read_only is False

    def test_is_read_only_reflects_injected_true(self):
        """is_read_only=True를 주입하면 True를 노출해야 한다 (조회 전용 완화)."""
        adapter = _make_adapter(AsyncMock(), is_read_only=True)
        assert adapter.is_read_only is True

    def test_group_is_namespaced_by_server(self):
        """group은 서버별 네임스페이스(mcp:{server})여야 한다."""
        adapter = _make_adapter(AsyncMock(), server_name="diag")
        assert adapter.group == "mcp:diag"


# ─────────────────────────────────────────────
# check_permissions — 항상 ALLOW로 회귀 (신뢰 통제는 등록 단계로 이동)
# ─────────────────────────────────────────────
class TestMcpAdapterPermissions:
    """어댑터 Layer 2 검사가 read-only/쓰기 무관 항상 ALLOW를 반환하는지 검증한다.

    설계 회귀: 직전 버전은 쓰기 도구를 ASK로 단락시켜 PLAN 모드의 최종 보정
    (ASK→DENY)을 무력화했다. 현재는 Layer 2가 항상 ALLOW를 내고 최종 판단을
    전적으로 5계층 파이프라인에 위임한다 — 신뢰 통제는 등록 단계
    (connect_and_register)에서 read-only 필터로 수행된다.
    """

    @pytest.mark.asyncio
    async def test_check_permissions_read_only_returns_allow(self):
        """read-only 도구(is_read_only=True)는 ALLOW를 반환한다.

        최종 판단(모드별 ALLOW/ASK/DENY)은 5계층 파이프라인에 위임한다.
        """
        adapter = _make_adapter(AsyncMock(), is_read_only=True)
        result = await adapter.check_permissions({"sql": "SELECT 1"}, _make_context())
        assert result.behavior == PermissionBehavior.ALLOW
        # 운영자에게 보여줄 안내 메시지가 존재해야 한다.
        assert result.message

    @pytest.mark.asyncio
    async def test_check_permissions_writable_returns_allow(self):
        """쓰기 가능 도구(is_read_only=False)도 항상 ALLOW를 반환한다 (회귀).

        직전엔 ASK였으나, ASK 조기 단락이 PLAN 모드 보정을 무력화하는 문제로
        제거되었다. Layer 2는 read-only/쓰기 무관하게 ALLOW로 통과시키고,
        모드별 최종 결정(ASK/DENY/ALLOW)은 5계층 파이프라인이 내린다.
        """
        adapter = _make_adapter(AsyncMock(), is_read_only=False)
        result = await adapter.check_permissions({"sql": "SELECT 1"}, _make_context())
        assert result.behavior == PermissionBehavior.ALLOW
        # 안내 메시지가 존재해야 한다 (5계층 위임 취지).
        assert result.message is not None

    @pytest.mark.asyncio
    async def test_check_permissions_details_carries_audit_meta(self):
        """read-only/쓰기 두 경우 모두 details에 감사 추적 메타를 실어야 한다.

        details = {mcp_server, remote_tool, is_read_only} —
        PermissionAuditEntry가 어느 서버의 어떤 read-only 상태에서 내린
        결정인지 추적할 수 있어야 한다. (이전 trusted_read_only → is_read_only 개명)
        """
        # read-only 도구
        ro_adapter = _make_adapter(
            AsyncMock(), server_name="diag", remote_tool_name="latency", is_read_only=True
        )
        ro_result = await ro_adapter.check_permissions({}, _make_context())
        assert ro_result.behavior == PermissionBehavior.ALLOW
        assert ro_result.details == {
            "mcp_server": "diag",
            "remote_tool": "latency",
            "is_read_only": True,
        }

        # 쓰기 가능 도구 — 동일 키 구성, is_read_only=False, 동일하게 ALLOW
        rw_adapter = _make_adapter(
            AsyncMock(), server_name="db", remote_tool_name="exec", is_read_only=False
        )
        rw_result = await rw_adapter.check_permissions({}, _make_context())
        assert rw_result.behavior == PermissionBehavior.ALLOW
        assert rw_result.details == {
            "mcp_server": "db",
            "remote_tool": "exec",
            "is_read_only": False,
        }


# ─────────────────────────────────────────────
# validate_input — 원격 스키마 기반 최소 검증 (v7.2 신규)
# ─────────────────────────────────────────────
class TestMcpAdapterValidateInput:
    """validate_input()의 required/type 최소 검증을 확인한다."""

    def test_validate_input_missing_required_returns_korean_error(self):
        """required 필드 누락 시 한글 에러 문자열을 반환해야 한다 ("필수 입력 누락")."""
        schema = {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        }
        adapter = _make_adapter(AsyncMock(), remote_schema=schema)
        # sql 누락 — 빈 입력
        error = adapter.validate_input({})
        assert error is not None
        assert "필수 입력 누락" in error
        # 누락된 키 이름이 메시지에 드러나야 자가 교정이 쉽다.
        assert "sql" in error

    def test_validate_input_required_satisfied_returns_none(self):
        """required 필드가 모두 충족되면 None(유효)을 반환해야 한다."""
        schema = {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        }
        adapter = _make_adapter(AsyncMock(), remote_schema=schema)
        assert adapter.validate_input({"sql": "SELECT 1"}) is None

    def test_validate_input_empty_schema_returns_none(self):
        """빈 스키마({})는 검증할 제약이 없으므로 None(유효)을 반환해야 한다."""
        adapter = _make_adapter(AsyncMock(), remote_schema={})
        assert adapter.validate_input({"anything": 1}) is None

    def test_validate_input_no_required_key_returns_none(self):
        """type=object지만 required가 없으면 누락 검사 대상이 없어 None이어야 한다."""
        schema = {"type": "object", "properties": {"sql": {"type": "string"}}}
        adapter = _make_adapter(AsyncMock(), remote_schema=schema)
        assert adapter.validate_input({}) is None

    def test_validate_input_multiple_missing_lists_all(self):
        """여러 required가 누락되면 모든 누락 키를 에러 메시지에 나열해야 한다."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        }
        adapter = _make_adapter(AsyncMock(), remote_schema=schema)
        error = adapter.validate_input({})
        assert error is not None
        assert "a" in error
        assert "b" in error


# ─────────────────────────────────────────────
# call() — 성공/실패 정규화
# ─────────────────────────────────────────────
class TestMcpAdapterCall:
    """call()이 원격 결과/예외를 ToolResult로 정규화하는지 검증한다."""

    @pytest.mark.asyncio
    async def test_call_success_returns_tool_result_success(self):
        """client.call_tool이 값을 반환하면 ToolResult.success(is_error=False)여야 한다."""
        client = AsyncMock()
        client.call_tool = AsyncMock(return_value={"rows": [{"id": 1}]})
        adapter = _make_adapter(client, remote_tool_name="query")

        result = await adapter.call({"sql": "SELECT 1"}, _make_context())

        assert isinstance(result, ToolResult)
        assert result.is_error is False
        assert result.data == {"rows": [{"id": 1}]}

    @pytest.mark.asyncio
    async def test_call_forwards_remote_tool_name_and_arguments(self):
        """call()은 원격 도구 이름과 입력 인자를 client.call_tool에 그대로 전달해야 한다."""
        client = AsyncMock()
        client.call_tool = AsyncMock(return_value="ok")
        adapter = _make_adapter(client, remote_tool_name="query")

        await adapter.call({"sql": "SELECT 2"}, _make_context())

        # 첫 번째 위치 인자 = 원격 도구 이름, 두 번째 = 입력 데이터
        call = client.call_tool.await_args
        assert call.args[0] == "query"
        assert call.args[1] == {"sql": "SELECT 2"}

    @pytest.mark.asyncio
    async def test_call_connection_error_returns_tool_result_error(self):
        """client.call_tool이 ConnectionError면 ToolResult.error로 격리되어야 한다."""
        client = AsyncMock()
        client.call_tool = AsyncMock(side_effect=ConnectionError("서버 다운"))
        adapter = _make_adapter(client, server_name="db")

        result = await adapter.call({"sql": "SELECT 1"}, _make_context())

        assert result.is_error is True
        # 에러 메시지에 서버명이 포함되어야 운영자가 어느 MCP 서버인지 식별 가능.
        assert "db" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_call_timeout_error_returns_tool_result_error(self):
        """client.call_tool이 TimeoutError면 ToolResult.error로 격리되어야 한다."""
        client = AsyncMock()
        client.call_tool = AsyncMock(side_effect=TimeoutError("타임아웃"))
        adapter = _make_adapter(client, server_name="diag")

        result = await adapter.call({"sql": "SELECT 1"}, _make_context())

        assert result.is_error is True
        assert "diag" in (result.error_message or "")


# ─────────────────────────────────────────────
# map_result — tool_use_error 래핑 (BaseTool 기본)
# ─────────────────────────────────────────────
class TestMcpAdapterMapResult:
    """에러 결과가 BaseTool.map_result로 tool_use_error 래핑되는지 검증한다."""

    @pytest.mark.asyncio
    async def test_map_result_wraps_error_in_tool_use_error(self):
        """call() 실패 결과를 map_result로 변환하면 <tool_use_error>로 감싸져야 한다."""
        client = AsyncMock()
        client.call_tool = AsyncMock(side_effect=ConnectionError("서버 다운"))
        adapter = _make_adapter(client, server_name="db")

        result = await adapter.call({"sql": "SELECT 1"}, _make_context())
        rendered = adapter.map_result(result)

        assert rendered.startswith("<tool_use_error>")
        assert rendered.endswith("</tool_use_error>")

    @pytest.mark.asyncio
    async def test_map_result_success_returns_plain_string(self):
        """성공 결과는 tool_use_error 래핑 없이 평문 문자열로 변환되어야 한다."""
        client = AsyncMock()
        client.call_tool = AsyncMock(return_value="결과값")
        adapter = _make_adapter(client)

        result = await adapter.call({"sql": "SELECT 1"}, _make_context())
        rendered = adapter.map_result(result)

        assert "tool_use_error" not in rendered
        assert "결과값" in rendered
