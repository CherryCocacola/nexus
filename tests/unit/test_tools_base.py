"""
core/tools/base.py + registry.py 단위 테스트.

BaseTool, ToolResult, ToolRegistry의 동작을 검증한다.
"""

from __future__ import annotations

from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
)
from core.tools.registry import ToolRegistry


# ─── 테스트용 더미 도구 ───
class DummyReadTool(BaseTool):
    """테스트용 읽기 도구."""

    @property
    def name(self) -> str:
        return "DummyRead"

    @property
    def description(self) -> str:
        return "Test read tool"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"path": {"type": "string"}}}

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    @property
    def aliases(self) -> list[str]:
        return ["dummy_read", "DRead"]

    @property
    def group(self) -> str:
        return "test"

    async def check_permissions(self, input_data, context):
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data, context):
        return ToolResult.success(f"read: {input_data.get('path', '')}")


class DummyWriteTool(BaseTool):
    """테스트용 쓰기 도구."""

    @property
    def name(self) -> str:
        return "DummyWrite"

    @property
    def description(self) -> str:
        return "Test write tool"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"content": {"type": "string"}}}

    @property
    def group(self) -> str:
        return "test"

    async def check_permissions(self, input_data, context):
        return PermissionResult(behavior=PermissionBehavior.ASK, message="write?")

    async def call(self, input_data, context):
        return ToolResult.success("written")


class TestToolResult:
    """ToolResult 테스트."""

    def test_success_factory(self):
        """성공 결과 생성을 확인한다."""
        result = ToolResult.success("data", key="val")
        assert result.data == "data"
        assert result.is_error is False
        assert result.metadata["key"] == "val"

    def test_error_factory(self):
        """에러 결과 생성을 확인한다."""
        result = ToolResult.error("실패!")
        assert result.is_error is True
        assert result.error_message == "실패!"


class TestBaseTool:
    """BaseTool 기본 동작 테스트."""

    def test_fail_closed_defaults(self):
        """기본값이 가장 제한적(fail-closed)인지 확인한다."""
        tool = DummyWriteTool()
        assert tool.is_read_only is False
        assert tool.is_concurrency_safe is False
        assert tool.is_destructive is False
        assert tool.requires_confirmation is False
        assert tool.has_side_effects is True

    def test_read_tool_flags(self):
        """읽기 도구의 플래그가 올바른지 확인한다."""
        tool = DummyReadTool()
        assert tool.is_read_only is True
        assert tool.is_concurrency_safe is True
        assert tool.has_side_effects is False

    def test_to_schema(self):
        """to_schema()가 올바른 형식을 반환하는지 확인한다."""
        tool = DummyReadTool()
        schema = tool.to_schema()
        assert schema["name"] == "DummyRead"
        assert "description" in schema
        assert "input_schema" in schema

    def test_repr(self):
        """__repr__이 플래그를 포함하는지 확인한다."""
        tool = DummyReadTool()
        assert "RO" in repr(tool)
        assert "CS" in repr(tool)

    def test_map_result_success(self):
        """성공 결과의 map_result가 데이터를 반환하는지 확인한다."""
        tool = DummyReadTool()
        result = ToolResult.success("hello")
        assert tool.map_result(result) == "hello"

    def test_map_result_error(self):
        """에러 결과의 map_result가 tool_use_error로 래핑되는지 확인한다."""
        tool = DummyReadTool()
        result = ToolResult.error("fail")
        mapped = tool.map_result(result)
        assert "<tool_use_error>" in mapped
        assert "fail" in mapped


class TestToolRegistry:
    """ToolRegistry 테스트."""

    def test_register_and_find(self):
        """도구 등록 및 조회를 확인한다."""
        reg = ToolRegistry()
        reg.register(DummyReadTool())
        assert reg.find_tool("DummyRead") is not None
        assert reg.tool_count == 1

    def test_find_by_alias(self):
        """alias로 도구를 조회할 수 있는지 확인한다."""
        reg = ToolRegistry()
        reg.register(DummyReadTool())
        # 대소문자 무시
        assert reg.find_tool("dummy_read") is not None
        assert reg.find_tool("DRead") is not None

    def test_find_unknown_returns_none(self):
        """존재하지 않는 도구 조회 시 None을 반환하는지 확인한다."""
        reg = ToolRegistry()
        assert reg.find_tool("NonExistent") is None

    def test_get_all_tools_sorted(self):
        """get_all_tools가 이름순으로 정렬되는지 확인한다."""
        reg = ToolRegistry()
        reg.register(DummyWriteTool())
        reg.register(DummyReadTool())
        tools = reg.get_all_tools()
        assert tools[0].name == "DummyRead"
        assert tools[1].name == "DummyWrite"

    def test_deny_pattern_filter(self):
        """deny 패턴이 도구를 제외하는지 확인한다."""
        reg = ToolRegistry()
        reg.register(DummyReadTool())
        reg.register(DummyWriteTool())
        tools = reg.get_tools(deny_patterns=["Dummy*"])
        assert len(tools) == 0

    def test_deny_specific_tool(self):
        """특정 도구만 deny할 수 있는지 확인한다."""
        reg = ToolRegistry()
        reg.register(DummyReadTool())
        reg.register(DummyWriteTool())
        tools = reg.get_tools(deny_patterns=["DummyWrite"])
        assert len(tools) == 1
        assert tools[0].name == "DummyRead"

    def test_assemble_tool_pool(self):
        """assemble_tool_pool이 도구와 스키마를 반환하는지 확인한다."""
        reg = ToolRegistry()
        reg.register(DummyReadTool())
        tools, schemas = reg.assemble_tool_pool()
        assert len(tools) == 1
        assert len(schemas) == 1
        assert schemas[0]["name"] == "DummyRead"

    def test_unregister(self):
        """도구 등록 해제를 확인한다."""
        reg = ToolRegistry()
        reg.register(DummyReadTool())
        removed = reg.unregister("DummyRead")
        assert removed is not None
        assert reg.tool_count == 0
        assert reg.find_tool("dummy_read") is None

    def test_register_many(self):
        """일괄 등록을 확인한다."""
        reg = ToolRegistry()
        reg.register_many([DummyReadTool(), DummyWriteTool()])
        assert reg.tool_count == 2


class TestToolImplementations:
    """실제 도구 구현 import 테스트."""

    def test_all_tools_import(self):
        """8개 핵심 도구가 모두 import 가능한지 확인한다."""
        from core.tools.implementations.bash_tool import BashTool
        from core.tools.implementations.edit_tool import EditTool
        from core.tools.implementations.glob_tool import GlobTool
        from core.tools.implementations.grep_tool import GrepTool
        from core.tools.implementations.ls_tool import LSTool
        from core.tools.implementations.multi_edit_tool import MultiEditTool
        from core.tools.implementations.read_tool import ReadTool
        from core.tools.implementations.write_tool import WriteTool

        tools = [ReadTool(), WriteTool(), EditTool(), MultiEditTool(),
                 BashTool(), GlobTool(), GrepTool(), LSTool()]
        assert len(tools) == 8

    def test_all_tools_have_required_attrs(self):
        """모든 도구가 필수 속성을 가지는지 확인한다."""
        from core.tools.implementations.bash_tool import BashTool
        from core.tools.implementations.edit_tool import EditTool
        from core.tools.implementations.glob_tool import GlobTool
        from core.tools.implementations.grep_tool import GrepTool
        from core.tools.implementations.ls_tool import LSTool
        from core.tools.implementations.read_tool import ReadTool
        from core.tools.implementations.write_tool import WriteTool

        for tool_cls in [ReadTool, WriteTool, EditTool, BashTool, GlobTool, GrepTool, LSTool]:
            tool = tool_cls()
            assert isinstance(tool.name, str)
            assert isinstance(tool.description, str)
            assert isinstance(tool.input_schema, dict)
            assert "type" in tool.input_schema

    def test_read_only_tools_flags(self):
        """읽기 도구가 올바른 플래그를 가지는지 확인한다."""
        from core.tools.implementations.glob_tool import GlobTool
        from core.tools.implementations.grep_tool import GrepTool
        from core.tools.implementations.ls_tool import LSTool
        from core.tools.implementations.read_tool import ReadTool

        for tool_cls in [ReadTool, GlobTool, GrepTool, LSTool]:
            tool = tool_cls()
            assert tool.is_read_only is True, f"{tool.name} should be read_only"
            assert tool.is_concurrency_safe is True, f"{tool.name} should be concurrency_safe"

    def test_registry_with_all_tools(self):
        """8개 도구를 레지스트리에 등록할 수 있는지 확인한다."""
        from core.tools.implementations.bash_tool import BashTool
        from core.tools.implementations.edit_tool import EditTool
        from core.tools.implementations.glob_tool import GlobTool
        from core.tools.implementations.grep_tool import GrepTool
        from core.tools.implementations.ls_tool import LSTool
        from core.tools.implementations.multi_edit_tool import MultiEditTool
        from core.tools.implementations.read_tool import ReadTool
        from core.tools.implementations.write_tool import WriteTool

        reg = ToolRegistry()
        reg.register_many([
            ReadTool(), WriteTool(), EditTool(), MultiEditTool(),
            BashTool(), GlobTool(), GrepTool(), LSTool(),
        ])
        assert reg.tool_count == 8
        _, schemas = reg.assemble_tool_pool()
        # cache-stable: 이름순 정렬
        names = [s["name"] for s in schemas]
        assert names == sorted(names)
