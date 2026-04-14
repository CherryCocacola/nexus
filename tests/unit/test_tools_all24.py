"""
24개 도구 전체 통합 테스트.

모든 도구가 올바르게 import, 등록, 스키마 생성되는지 검증한다.
"""

from __future__ import annotations

from core.tools.registry import ToolRegistry


def _create_all_tools():
    """24개 도구 인스턴스를 모두 생성한다."""
    from core.tools.implementations.agent_tool import AgentTool
    from core.tools.implementations.bash_tool import BashTool
    from core.tools.implementations.docker_tools import DockerBuildTool, DockerRunTool
    from core.tools.implementations.edit_tool import EditTool
    from core.tools.implementations.git_tools import (
        GitBranchTool,
        GitCheckoutTool,
        GitCommitTool,
        GitDiffTool,
        GitLogTool,
        GitStatusTool,
    )
    from core.tools.implementations.glob_tool import GlobTool
    from core.tools.implementations.grep_tool import GrepTool
    from core.tools.implementations.ls_tool import LSTool
    from core.tools.implementations.memory_tools import MemoryReadTool, MemoryWriteTool
    from core.tools.implementations.multi_edit_tool import MultiEditTool
    from core.tools.implementations.notebook_tools import (
        NotebookEditTool,
        NotebookReadTool,
    )
    from core.tools.implementations.read_tool import ReadTool
    from core.tools.implementations.task_tools import TaskTool, TodoReadTool, TodoWriteTool
    from core.tools.implementations.write_tool import WriteTool

    return [
        ReadTool(), WriteTool(), EditTool(), MultiEditTool(),
        BashTool(), GlobTool(), GrepTool(), LSTool(),
        GitLogTool(), GitDiffTool(), GitStatusTool(),
        GitCommitTool(), GitBranchTool(), GitCheckoutTool(),
        NotebookReadTool(), NotebookEditTool(),
        TodoReadTool(), TodoWriteTool(), TaskTool(),
        AgentTool(),
        MemoryReadTool(), MemoryWriteTool(),
        DockerBuildTool(), DockerRunTool(),
    ]


class TestAll24Tools:
    """24개 도구 전체 검증."""

    def test_total_count(self):
        """정확히 24개 도구가 등록되는지 확인한다."""
        reg = ToolRegistry()
        reg.register_many(_create_all_tools())
        assert reg.tool_count == 24

    def test_all_unique_names(self):
        """모든 도구 이름이 고유한지 확인한다."""
        tools = _create_all_tools()
        names = [t.name for t in tools]
        dupes = [n for n in names if names.count(n) > 1]
        assert len(names) == len(set(names)), f"중복 이름: {dupes}"

    def test_cache_stable_sorting(self):
        """assemble_tool_pool이 항상 같은 순서를 반환하는지 확인한다 (cache stability)."""
        reg = ToolRegistry()
        reg.register_many(_create_all_tools())
        _, schemas1 = reg.assemble_tool_pool()
        _, schemas2 = reg.assemble_tool_pool()
        names1 = [s["name"] for s in schemas1]
        names2 = [s["name"] for s in schemas2]
        assert names1 == names2
        assert names1 == sorted(names1)

    def test_all_tools_have_valid_schema(self):
        """모든 도구의 input_schema가 유효한 JSON Schema인지 확인한다."""
        for tool in _create_all_tools():
            schema = tool.input_schema
            assert isinstance(schema, dict), f"{tool.name}: schema가 dict가 아님"
            assert schema.get("type") == "object", f"{tool.name}: type이 object가 아님"
            assert "properties" in schema, f"{tool.name}: properties가 없음"

    def test_tool_groups(self):
        """예상 그룹이 모두 존재하는지 확인한다."""
        tools = _create_all_tools()
        groups = {t.group for t in tools}
        expected = {
            "filesystem", "search", "execution", "git",
            "notebook", "task", "agent", "memory", "docker",
        }
        assert groups == expected

    def test_read_only_tools_count(self):
        """읽기 전용 도구가 예상 개수인지 확인한다."""
        tools = _create_all_tools()
        ro_tools = [t for t in tools if t.is_read_only]
        # Read, Glob, Grep, LS, Git(4), NotebookRead, TodoRead, MemoryRead = 11+
        assert len(ro_tools) >= 11, f"읽기 전용 도구: {[t.name for t in ro_tools]}"

    def test_concurrency_safe_tools(self):
        """동시성 안전 도구가 최소 예상 개수인지 확인한다."""
        tools = _create_all_tools()
        cs_tools = [t for t in tools if t.is_concurrency_safe]
        # 읽기 전용 도구는 대부분 동시성 안전
        assert len(cs_tools) >= 10

    def test_deny_filter_removes_agent(self):
        """deny 패턴으로 Agent 도구를 제거할 수 있는지 확인한다."""
        reg = ToolRegistry()
        reg.register_many(_create_all_tools())
        tools = reg.get_tools(deny_patterns=["Agent"])
        names = [t.name for t in tools]
        assert "Agent" not in names
        assert len(names) == 23

    def test_to_schema_format(self):
        """모든 도구의 to_schema()가 올바른 형식을 반환하는지 확인한다."""
        for tool in _create_all_tools():
            schema = tool.to_schema()
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
            assert isinstance(schema["description"], str)
            assert len(schema["description"]) > 0, f"{tool.name}: description이 비어 있음"
