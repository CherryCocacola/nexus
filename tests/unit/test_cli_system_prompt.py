# CLI Worker 시스템 프롬프트의 티어 분기 검증 — 프롬프트↔도구 풀 일치가 핵심.
"""
`_build_default_system_prompt()`가 티어별 도구 레지스트리와 일치하는 지침을
내보내는지 검증한다.

배경(회귀 방지): CLI는 티어에 따라 도구 풀이 완전히 다르다.
  - TIER_S   : `_create_cli_tool_registry()`  = 7개  (Agent 있음, Read/Glob 없음)
  - TIER_M/L : `_create_tool_registry()`      = 23개 (Read/Glob 있음, Agent 없음)
그런데 프롬프트에는 티어 분기가 없어 항상 TIER_S 서사("Read/Glob 없음, scout에
위임하라")를 내보냈고, 모델이 그대로 따라 `Agent`를 호출해
`알 수 없는 도구: 'Agent'` 에러가 실제로 발생했다. 이 테스트가 그 재발을 막는다.
"""

from __future__ import annotations

from core.bootstrap import _build_default_system_prompt
from core.model.hardware_tier import HardwareTier


class _FakeAgentRegistry:
    """서브에이전트 레지스트리 최소 대역 — list_descriptions()와 len()만 쓴다."""

    def __init__(self, agents: dict[str, str]):
        self._agents = agents

    def __len__(self) -> int:
        return len(self._agents)

    def list_descriptions(self) -> dict[str, str]:
        return self._agents


# ── TIER_S — Scout 위임 서사(기존 동작 유지) ─────────────────────────────


def test_tier_s_prompt_delegates_to_scout():
    p = _build_default_system_prompt(tier=HardwareTier.TIER_S)
    assert "subagent_type='scout'" in p
    assert "You do NOT have Read/Glob/Grep/LS" in p


def test_tier_s_prompt_injects_registered_subagents():
    """TIER_S는 Agent 도구가 실제로 있으므로 서브에이전트 목록을 안내해도 된다."""
    reg = _FakeAgentRegistry({"scout": "read-only explorer"})
    p = _build_default_system_prompt(reg, tier=HardwareTier.TIER_S)
    assert "Registered sub-agents" in p
    assert "scout: read-only explorer" in p


def test_no_tier_argument_keeps_tier_s_behaviour():
    """하위 호환 — tier 미지정 호출은 기존(TIER_S) 프롬프트를 그대로 낸다."""
    assert _build_default_system_prompt() == _build_default_system_prompt(
        tier=HardwareTier.TIER_S
    )


# ── TIER_M/L — 직접 탐색 서사(이번 수정) ─────────────────────────────────


def test_expanded_prompt_advertises_direct_exploration_tools():
    """23개 풀에 실재하는 탐색 도구를 쓰라고 안내해야 한다."""
    p = _build_default_system_prompt(tier=HardwareTier.TIER_L)
    for tool in ("Read", "Glob", "Grep", "LS"):
        assert tool in p


def test_expanded_prompt_never_instructs_agent_call():
    """회귀 핵심: TIER_M/L 풀에 Agent가 없으므로 호출을 지시하면 안 된다."""
    p = _build_default_system_prompt(tier=HardwareTier.TIER_L)
    assert "subagent_type='scout'" not in p
    assert "Agent(prompt=" not in p


def test_expanded_prompt_explicitly_forbids_agent():
    """단순 미언급이 아니라 명시적으로 금지해야 모델의 사전지식 오호출을 막는다."""
    p = _build_default_system_prompt(tier=HardwareTier.TIER_L)
    assert "do NOT have an `Agent` tool" in p


def test_expanded_prompt_does_not_deny_read_capability():
    """TIER_S의 '너는 Read가 없다'가 새 프롬프트로 새어나오면 안 된다."""
    p = _build_default_system_prompt(tier=HardwareTier.TIER_L)
    assert "You do NOT have Read/Glob/Grep/LS" not in p
    assert "NEVER attempt Read/Glob/Grep/LS" not in p


def test_expanded_prompt_omits_subagent_list():
    """Agent 도구가 없는 티어에서 서브에이전트를 안내하면 오호출을 유도한다."""
    reg = _FakeAgentRegistry({"scout": "read-only explorer"})
    p = _build_default_system_prompt(reg, tier=HardwareTier.TIER_L)
    assert "Registered sub-agents" not in p


def test_tier_m_uses_same_expanded_prompt_as_tier_l():
    """TIER_M도 23개 풀을 쓰므로 TIER_L과 동일한 프롬프트여야 한다."""
    assert _build_default_system_prompt(
        tier=HardwareTier.TIER_M
    ) == _build_default_system_prompt(tier=HardwareTier.TIER_L)


# ── 공통 섹션 — 티어와 무관하게 유지되어야 한다 ──────────────────────────


def test_expanded_prompt_lists_exactly_the_registered_tools():
    """근본 수정: 프롬프트의 도구 목록은 레지스트리에서 유도되어야 한다."""
    p = _build_default_system_prompt(
        tier=HardwareTier.TIER_L, tool_names={"Read", "Bash", "CustomThing"}
    )
    assert "- CustomThing\n" in p       # 레지스트리에 있으면 나온다
    assert "- Write\n" not in p          # 없으면 안 나온다(하드코딩이면 남았을 것)


def test_expanded_prompt_allows_agent_when_actually_registered():
    """Agent가 실제로 등록된 풀이면 금지문이 아니라 사용 안내가 나와야 한다."""
    p = _build_default_system_prompt(
        tier=HardwareTier.TIER_L, tool_names={"Read", "Agent"}
    )
    assert "do NOT have an `Agent` tool" not in p
    assert "Sub-agent delegation" in p


def test_expanded_prompt_forbids_agent_when_absent():
    """Agent가 없는 풀이면 명시적 금지가 유지되어야 한다(현재 CLI 23개 풀)."""
    p = _build_default_system_prompt(
        tier=HardwareTier.TIER_L, tool_names={"Read", "Bash"}
    )
    assert "do NOT have an `Agent` tool" in p


def test_expanded_prompt_omits_exploration_note_without_explorers():
    """탐색 도구가 없는 풀이면 '직접 탐색하라' 안내가 나오면 안 된다."""
    p = _build_default_system_prompt(
        tier=HardwareTier.TIER_L, tool_names={"Bash", "Write"}
    )
    assert "Exploring a codebase" not in p


def test_expanded_prompt_tool_list_is_sorted():
    """등록 순서가 달라져도 프롬프트가 흔들리지 않아야 한다(prompt cache 안정성)."""
    a = _build_default_system_prompt(tier=HardwareTier.TIER_L, tool_names={"Read", "Bash"})
    b = _build_default_system_prompt(tier=HardwareTier.TIER_L, tool_names={"Bash", "Read"})
    assert a == b


def test_common_sections_present_in_both_tiers():
    """대화 규약과 지식베이스 처리 지침은 티어 분기와 무관하게 살아 있어야 한다."""
    for tier in (HardwareTier.TIER_S, HardwareTier.TIER_L):
        p = _build_default_system_prompt(tier=tier)
        assert "Conversational style" in p
        assert "--- Knowledge base ---" in p
        assert "NEVER create a file the user didn't ask for." in p


def test_both_tiers_declare_nova_identity():
    for tier in (HardwareTier.TIER_S, HardwareTier.TIER_L):
        assert "IDINO NOVA" in _build_default_system_prompt(tier=tier)
