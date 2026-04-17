"""
core/tools/implementations/agent_tool.py 단위 테스트.

v7.0 Phase 9 확장 기능을 검증한다:
  - subagent_type으로 AgentRegistry 조회
  - model_override 해석 ("scout" → ScoutModelProvider)
  - allowed_tools 기반 도구 필터링
  - 레지스트리/서브에이전트 누락 시 에러 처리
  - 하위 호환: subagent_type 없으면 description 기반 ad-hoc 동작
  - 호출 통계(get_stats) 누적

실제 QueryEngine/모델 호출은 mock 처리하여 설정 해석 로직만 격리 검증한다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.message import StreamEvent, StreamEventType
from core.orchestrator.agent_definition import (
    AgentDefinition,
    AgentRegistry,
    SCOUT_AGENT,
)
from core.tools.base import BaseTool, ToolUseContext
from core.tools.implementations.agent_tool import (
    DISALLOWED_TOOLS_FOR_AGENTS,
    AgentTool,
)


# ─────────────────────────────────────────────
# 공용 fixture
# ─────────────────────────────────────────────
@pytest.fixture(autouse=True)
def reset_stats():
    """각 테스트 전에 AgentTool 클래스 통계를 초기화한다."""
    AgentTool.reset_stats()
    yield
    AgentTool.reset_stats()


def _make_tool(name: str) -> MagicMock:
    """지정된 이름을 가진 BaseTool mock을 만든다."""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    return tool


@pytest.fixture
def all_tools() -> list[MagicMock]:
    """기본 도구 풀 — Scout용 5개 + 쓰기 2개 + Agent(자기 자신)."""
    return [
        _make_tool("Read"),
        _make_tool("Glob"),
        _make_tool("Grep"),
        _make_tool("LS"),
        _make_tool("DocumentProcess"),
        _make_tool("Write"),
        _make_tool("Edit"),
        _make_tool("Agent"),  # DISALLOWED
    ]


@pytest.fixture
def registry_with_scout() -> AgentRegistry:
    """SCOUT_AGENT가 등록된 레지스트리."""
    reg = AgentRegistry()
    reg.register(SCOUT_AGENT)
    return reg


@pytest.fixture
def worker_provider() -> MagicMock:
    """부모 Worker 모델 프로바이더 mock."""
    return MagicMock(name="WorkerProvider")


@pytest.fixture
def scout_provider() -> MagicMock:
    """Scout 모델 프로바이더 mock."""
    return MagicMock(name="ScoutProvider")


@pytest.fixture
def context(
    tmp_path: Path,
    all_tools: list[MagicMock],
    registry_with_scout: AgentRegistry,
    worker_provider: MagicMock,
    scout_provider: MagicMock,
) -> ToolUseContext:
    """options에 registry/providers/도구를 모두 주입한 컨텍스트."""
    return ToolUseContext(
        cwd=str(tmp_path),
        session_id="test-agent",
        permission_mode="bypass_permissions",
        options={
            "agent_registry": registry_with_scout,
            "available_tools": all_tools,
            "model_provider": worker_provider,
            "scout_provider": scout_provider,
        },
    )


# ─────────────────────────────────────────────
# validate_input
# ─────────────────────────────────────────────
class TestValidateInput:
    """input 검증 로직."""

    def test_prompt_required(self):
        err = AgentTool().validate_input({"prompt": ""})
        assert err is not None and "prompt" in err

    def test_either_subagent_type_or_description_required(self):
        err = AgentTool().validate_input({"prompt": "hi"})
        assert err is not None
        assert "subagent_type" in err or "description" in err

    def test_subagent_type_only_is_ok(self):
        err = AgentTool().validate_input(
            {"prompt": "hi", "subagent_type": "scout"}
        )
        assert err is None

    def test_description_only_is_ok_backcompat(self):
        err = AgentTool().validate_input(
            {"prompt": "hi", "description": "helper"}
        )
        assert err is None


# ─────────────────────────────────────────────
# subagent_type 경로 — AgentDefinition 기반
# ─────────────────────────────────────────────
class TestSubagentTypePath:
    """subagent_type='scout' 호출 시 AgentDefinition 기반 설정 해석을 검증한다."""

    async def test_resolves_scout_from_registry(
        self, context: ToolUseContext, scout_provider: MagicMock
    ):
        """
        subagent_type='scout'이면 ScoutModelProvider를 사용하고
        allowed_tools만 서브에이전트에 전달된다.
        """
        tool = AgentTool()
        captured: dict = {}

        async def fake_run(self, prompt, config, parent_context):
            captured["model_provider"] = config.model_provider
            captured["tools"] = config.tools
            captured["system_prompt"] = config.system_prompt
            captured["max_turns"] = config.max_turns
            return "done", 1

        with patch.object(AgentTool, "_run_subagent", fake_run):
            result = await tool.call(
                {"prompt": "find login code", "subagent_type": "scout"},
                context,
            )

        assert not result.is_error
        # ScoutModelProvider가 선택됐다
        assert captured["model_provider"] is scout_provider
        # 도구는 Read/Glob/Grep/LS만 (Write/Edit/Agent 제외)
        tool_names = {t.name for t in captured["tools"]}
        assert tool_names == {"Read", "Glob", "Grep", "LS", "DocumentProcess"}
        # max_turns가 AgentDefinition의 값(5)으로 설정됐다 — Part 2.3 개정
        assert captured["max_turns"] == 5
        # system_prompt가 AgentDefinition의 값을 사용한다 (Scout 프롬프트)
        assert "Scout" in captured["system_prompt"]

    async def test_unknown_subagent_type_returns_error(
        self, context: ToolUseContext
    ):
        """등록되지 않은 subagent_type은 목록 안내와 함께 에러 반환."""
        tool = AgentTool()
        result = await tool.call(
            {"prompt": "hi", "subagent_type": "unknown-agent"},
            context,
        )
        assert result.is_error
        assert "unknown-agent" in (result.error_message or "")
        # 사용 가능한 이름(scout)이 안내에 포함된다
        assert "scout" in (result.error_message or "")

    async def test_missing_registry_returns_error(
        self,
        tmp_path: Path,
        all_tools: list[MagicMock],
        worker_provider: MagicMock,
    ):
        """agent_registry가 context.options에 없으면 에러."""
        ctx = ToolUseContext(
            cwd=str(tmp_path),
            session_id="s",
            permission_mode="bypass_permissions",
            options={
                "available_tools": all_tools,
                "model_provider": worker_provider,
            },
        )
        tool = AgentTool()
        result = await tool.call(
            {"prompt": "hi", "subagent_type": "scout"}, ctx
        )
        assert result.is_error
        assert "agent_registry" in (result.error_message or "")

    async def test_missing_scout_provider_returns_error(
        self,
        tmp_path: Path,
        all_tools: list[MagicMock],
        registry_with_scout: AgentRegistry,
        worker_provider: MagicMock,
    ):
        """Scout를 요청했는데 scout_provider가 없으면 에러 (TIER_M/L 상황)."""
        ctx = ToolUseContext(
            cwd=str(tmp_path),
            session_id="s",
            permission_mode="bypass_permissions",
            options={
                "agent_registry": registry_with_scout,
                "available_tools": all_tools,
                "model_provider": worker_provider,
                # scout_provider 없음
            },
        )
        tool = AgentTool()
        result = await tool.call(
            {"prompt": "hi", "subagent_type": "scout"}, ctx
        )
        assert result.is_error
        assert "scout_provider" in (result.error_message or "")


# ─────────────────────────────────────────────
# description 경로 — 하위 호환
# ─────────────────────────────────────────────
class TestDescriptionPath:
    """subagent_type이 없을 때 description 기반 ad-hoc 동작을 검증한다."""

    async def test_description_uses_parent_model_and_filtered_tools(
        self,
        context: ToolUseContext,
        worker_provider: MagicMock,
    ):
        """description 모드에서는 부모 Worker 재사용 + DISALLOWED 필터만 적용."""
        tool = AgentTool()
        captured: dict = {}

        async def fake_run(self, prompt, config, parent_context):
            captured["model_provider"] = config.model_provider
            captured["tools"] = config.tools
            captured["system_prompt"] = config.system_prompt
            return "done", 1

        with patch.object(AgentTool, "_run_subagent", fake_run):
            result = await tool.call(
                {"prompt": "hi", "description": "helper role"},
                context,
            )

        assert not result.is_error
        # 부모 프로바이더 재사용
        assert captured["model_provider"] is worker_provider
        # DISALLOWED(=Agent)만 제외, 나머지 전부 포함
        tool_names = {t.name for t in captured["tools"]}
        assert "Agent" not in tool_names
        assert tool_names >= {"Read", "Glob", "Grep", "LS", "Write", "Edit"}
        # system_prompt가 description 값
        assert captured["system_prompt"] == "helper role"


# ─────────────────────────────────────────────
# 호출 통계 (Ch 17)
# ─────────────────────────────────────────────
class TestAgentToolStats:
    """AgentTool.get_stats()가 호출 통계를 정확히 집계하는지 검증한다."""

    async def test_stats_increment_per_subagent_type(
        self, context: ToolUseContext
    ):
        """Scout 호출 시 stats['scout'] 카운터가 증가한다."""
        tool = AgentTool()

        async def fake_run(self, prompt, config, parent_context):
            return "ok", 1

        with patch.object(AgentTool, "_run_subagent", fake_run):
            await tool.call(
                {"prompt": "p1", "subagent_type": "scout"}, context
            )
            await tool.call(
                {"prompt": "p2", "subagent_type": "scout"}, context
            )

        stats = AgentTool.get_stats()
        assert "scout" in stats
        assert stats["scout"]["calls"] == 2
        assert stats["scout"]["avg_latency_ms"] >= 0.0

    async def test_stats_separate_for_adhoc(self, context: ToolUseContext):
        """subagent_type이 없는 ad-hoc 호출은 별도 'ad-hoc' 키로 집계된다."""
        tool = AgentTool()

        async def fake_run(self, prompt, config, parent_context):
            return "ok", 1

        with patch.object(AgentTool, "_run_subagent", fake_run):
            await tool.call({"prompt": "p", "description": "d"}, context)

        stats = AgentTool.get_stats()
        assert "ad-hoc" in stats
        assert stats["ad-hoc"]["calls"] == 1

    async def test_stats_recorded_even_on_failure(
        self, context: ToolUseContext
    ):
        """서브에이전트 실행 실패해도 통계는 누적된다 (관찰성 보장)."""
        tool = AgentTool()

        async def failing_run(self, prompt, config, parent_context):
            raise RuntimeError("mocked failure")

        with patch.object(AgentTool, "_run_subagent", failing_run):
            result = await tool.call(
                {"prompt": "p", "subagent_type": "scout"}, context
            )

        assert result.is_error
        stats = AgentTool.get_stats()
        assert stats["scout"]["calls"] == 1

    def test_reset_stats_clears_all(self):
        """reset_stats()가 모든 항목을 비운다."""
        AgentTool._stats["scout"] = {"calls": 5, "total_latency_ms": 150000.0}
        AgentTool.reset_stats()
        assert AgentTool.get_stats() == {}


# ─────────────────────────────────────────────
# DISALLOWED 도구 제외
# ─────────────────────────────────────────────
class TestDisallowedToolsFilter:
    """subagent 경로/ad-hoc 경로 모두 DISALLOWED 도구는 제외되어야 한다."""

    async def test_agent_tool_not_in_subagent_tools(
        self, context: ToolUseContext
    ):
        """Scout가 허용한 도구 목록에 Agent가 들어 있어도 DISALLOWED라 제외된다."""
        # Scout의 allowed_tools는 Agent를 포함하지 않으므로 이중 보호 확인
        tool = AgentTool()
        captured: dict = {}

        async def fake_run(self, prompt, config, parent_context):
            captured["tools"] = config.tools
            return "ok", 1

        with patch.object(AgentTool, "_run_subagent", fake_run):
            await tool.call(
                {"prompt": "p", "subagent_type": "scout"}, context
            )

        tool_names = {t.name for t in captured["tools"]}
        assert "Agent" not in tool_names
        for bad in DISALLOWED_TOOLS_FOR_AGENTS:
            assert bad not in tool_names
