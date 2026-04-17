"""
core/orchestrator/agent_definition.py 단위 테스트.

AgentDefinition 불변성, AgentRegistry 등록/조회 로직,
SCOUT_AGENT 선언이 사양서 v7.0 Ch 14의 요구사항을 충족하는지 검증한다.
"""

from __future__ import annotations

import dataclasses

import pytest

from core.orchestrator.agent_definition import (
    DEFAULT_AGENTS,
    SCOUT_AGENT,
    AgentDefinition,
    AgentRegistry,
    build_default_agent_registry,
)


# ─────────────────────────────────────────────
# AgentDefinition — 불변성/필드
# ─────────────────────────────────────────────
class TestAgentDefinitionImmutability:
    """AgentDefinition이 frozen dataclass로서 불변인지 검증한다."""

    def test_is_frozen_dataclass(self):
        """dataclasses.is_dataclass + frozen 플래그 확인."""
        assert dataclasses.is_dataclass(AgentDefinition)
        # frozen은 _FIELDS가 아닌 __dataclass_params__.frozen에 들어있다
        assert AgentDefinition.__dataclass_params__.frozen is True

    def test_assign_raises_frozen_instance_error(self):
        """name을 변경하려 하면 FrozenInstanceError가 발생한다."""
        agent = AgentDefinition(
            name="x",
            description="d",
            system_prompt="p",
            allowed_tools=("Read",),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            agent.name = "y"  # type: ignore[misc]

    def test_default_max_turns_is_10(self):
        """max_turns의 기본값이 10이다."""
        agent = AgentDefinition(
            name="x",
            description="d",
            system_prompt="p",
            allowed_tools=(),
        )
        assert agent.max_turns == 10

    def test_default_model_override_is_none(self):
        """model_override 기본값은 None — 부모 Worker 모델 재사용을 의미한다."""
        agent = AgentDefinition(
            name="x",
            description="d",
            system_prompt="p",
            allowed_tools=(),
        )
        assert agent.model_override is None


# ─────────────────────────────────────────────
# AgentRegistry
# ─────────────────────────────────────────────
class TestAgentRegistry:
    """AgentRegistry 등록/조회/목록화 로직을 검증한다."""

    def test_register_and_get(self):
        """register 후 get으로 조회 가능하다."""
        registry = AgentRegistry()
        agent = AgentDefinition(
            name="scout",
            description="d",
            system_prompt="p",
            allowed_tools=("Read",),
        )
        registry.register(agent)

        assert registry.get("scout") is agent

    def test_get_missing_returns_none(self):
        """등록되지 않은 이름은 None을 반환한다."""
        registry = AgentRegistry()
        assert registry.get("nonexistent") is None

    def test_register_overwrites(self):
        """같은 name 재등록 시 덮어쓴다."""
        registry = AgentRegistry()
        first = AgentDefinition(name="a", description="first", system_prompt="", allowed_tools=())
        second = AgentDefinition(name="a", description="second", system_prompt="", allowed_tools=())
        registry.register(first)
        registry.register(second)
        assert registry.get("a") is second

    def test_register_many(self):
        """여러 에이전트를 한 번에 등록한다."""
        registry = AgentRegistry()
        agents = [
            AgentDefinition(name=f"a{i}", description="", system_prompt="", allowed_tools=())
            for i in range(3)
        ]
        registry.register_many(agents)
        assert len(registry) == 3

    def test_list_names_sorted(self):
        """list_names가 이름순 정렬된 결과를 반환한다 (캐시 안정성)."""
        registry = AgentRegistry()
        registry.register(AgentDefinition(name="c", description="", system_prompt="", allowed_tools=()))
        registry.register(AgentDefinition(name="a", description="", system_prompt="", allowed_tools=()))
        registry.register(AgentDefinition(name="b", description="", system_prompt="", allowed_tools=()))

        assert registry.list_names() == ["a", "b", "c"]

    def test_list_descriptions(self):
        """list_descriptions가 이름→description 매핑을 반환한다."""
        registry = AgentRegistry()
        registry.register(
            AgentDefinition(
                name="scout",
                description="Read-only explorer",
                system_prompt="",
                allowed_tools=(),
            )
        )
        descs = registry.list_descriptions()
        assert descs == {"scout": "Read-only explorer"}

    def test_contains_operator(self):
        """__contains__가 등록 여부를 반환한다."""
        registry = AgentRegistry()
        registry.register(AgentDefinition(name="x", description="", system_prompt="", allowed_tools=()))
        assert "x" in registry
        assert "y" not in registry

    def test_len(self):
        """__len__이 등록된 에이전트 수를 반환한다."""
        registry = AgentRegistry()
        assert len(registry) == 0
        registry.register(AgentDefinition(name="a", description="", system_prompt="", allowed_tools=()))
        assert len(registry) == 1


# ─────────────────────────────────────────────
# SCOUT_AGENT 선언 검증
# ─────────────────────────────────────────────
class TestScoutAgent:
    """SCOUT_AGENT 선언이 사양서 Part 2.3 / Ch 14의 요구를 충족하는지 검증한다."""

    def test_scout_name(self):
        """이름은 'scout'."""
        assert SCOUT_AGENT.name == "scout"

    def test_scout_allowed_tools_are_read_only(self):
        """
        Scout의 도구는 읽기 전용 5개 (v7.0 Part 2.3 개정, 2026-04-17):
        Read/Glob/Grep/LS + DocumentProcess. 쓰기 도구 없음.
        """
        assert set(SCOUT_AGENT.allowed_tools) == {
            "Read", "Glob", "Grep", "LS", "DocumentProcess"
        }

    def test_scout_model_override_is_scout(self):
        """model_override='scout'으로 ScoutModelProvider 사용을 지시한다."""
        assert SCOUT_AGENT.model_override == "scout"

    def test_scout_max_turns_limited(self):
        """
        Scout는 최대 5턴 — 문서 청크 순차 처리를 위해 기존 3턴에서 증가
        (Part 2.3 개정과 함께 조정).
        """
        assert SCOUT_AGENT.max_turns == 5

    def test_scout_description_warns_about_cost(self):
        """Worker가 비용을 인지하도록 description에 '느림' 정보가 있어야 한다."""
        # 영어 description이므로 'slow' 또는 'CPU' 키워드 확인
        desc_lower = SCOUT_AGENT.description.lower()
        assert "slow" in desc_lower or "cpu" in desc_lower

    def test_scout_description_warns_about_misuse(self):
        """Scout를 간단한 질문에 쓰지 않도록 유도하는 문구가 있어야 한다."""
        # "Do NOT use for simple questions" 류 경고
        desc_lower = SCOUT_AGENT.description.lower()
        assert "do not" in desc_lower or "only when" in desc_lower


# ─────────────────────────────────────────────
# build_default_agent_registry 팩토리
# ─────────────────────────────────────────────
class TestBuildDefaultAgentRegistry:
    """build_default_agent_registry가 기본 에이전트를 모두 등록하는지 검증한다."""

    def test_contains_scout_by_default(self):
        """기본 레지스트리에 scout가 포함된다."""
        registry = build_default_agent_registry()
        assert "scout" in registry
        assert registry.get("scout") is SCOUT_AGENT

    def test_registers_all_default_agents(self):
        """DEFAULT_AGENTS의 모든 항목이 등록된다."""
        registry = build_default_agent_registry()
        for agent in DEFAULT_AGENTS:
            assert registry.get(agent.name) is agent

    def test_multiple_calls_independent_registries(self):
        """팩토리를 여러 번 호출하면 서로 독립된 인스턴스를 반환한다."""
        r1 = build_default_agent_registry()
        r2 = build_default_agent_registry()
        assert r1 is not r2
        # 하지만 내부 AgentDefinition은 공유 가능 (불변 객체라 상관없음)
        assert r1.get("scout") is r2.get("scout")
