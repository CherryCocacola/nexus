"""
서브 에이전트 정의 — Worker가 AgentTool로 호출할 수 있는 에이전트의 명세.

v7.0 Phase 9 재설계: Scout가 "자동 전처리기"에서 "호출 가능한 서브 에이전트"로
승격된다. Worker(Qwen 27B)가 맥락을 보고 스스로 판단하여 Scout를 호출한다.

사양서 Ch 14 Agent & Task System의 AgentDefinition 명세를 구체화한다.

핵심 구성 요소:
  - AgentDefinition: 서브 에이전트의 이름/프롬프트/도구/모델 명세 (frozen)
  - AgentRegistry: 선언된 서브 에이전트를 등록·조회하는 레지스트리
  - SCOUT_AGENT: v7.0 TIER_S에서 사용하는 읽기 전용 탐색 에이전트 선언

설계 원칙:
  1. AgentDefinition은 불변(frozen dataclass) — 런타임에 수정 불가
  2. allowed_tools는 문자열 튜플 — ToolRegistry와 느슨하게 결합
  3. model_override는 이름 기반("scout") — 실제 프로바이더는 AgentTool이 해석
  4. description은 Worker의 "도구 선택 판단"에 쓰이는 힌트 — 명확하게 작성
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# AgentDefinition — 서브 에이전트 명세 (불변)
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class AgentDefinition:
    """
    서브 에이전트의 불변 명세.

    Worker가 AgentTool을 호출할 때 subagent_type으로 이 name을 지정하면,
    AgentTool은 해당 AgentDefinition을 레지스트리에서 찾아 설정을 적용한다.

    Fields:
        name: 에이전트 식별자 (subagent_type 값으로 사용)
        description: Worker가 "언제 이 에이전트를 써야 하는지" 판단하는 힌트.
            모델에게 보이는 문자열이므로 명확한 용도/비용 설명이 중요하다.
        system_prompt: 서브 에이전트의 시스템 프롬프트
        allowed_tools: 사용 가능한 도구 이름 튜플 (ToolRegistry에서 이 이름으로 조회)
        max_turns: 서브 에이전트의 최대 턴 수 (무한 루프 방지)
        model_override: 사용할 모델의 이름 힌트 ("scout" 또는 None).
            None이면 부모(Worker)와 동일한 ModelProvider 사용.
            AgentTool이 context.options에서 실제 Provider를 해석한다.

    왜 frozen인가: 서브 에이전트 정의는 실행 중 수정되면 안 된다.
        권한 모드를 위반한 실행을 막고 레지스트리 일관성을 유지한다.
    """

    name: str
    description: str
    system_prompt: str
    allowed_tools: tuple[str, ...]
    max_turns: int = 10
    model_override: str | None = None


# ─────────────────────────────────────────────
# AgentRegistry — 선언된 서브 에이전트 저장소
# ─────────────────────────────────────────────
class AgentRegistry:
    """
    서브 에이전트 레지스트리.

    Phase 2 부트스트랩 시 기본 에이전트(SCOUT_AGENT 등)를 등록하고,
    AgentTool이 subagent_type으로 조회한다.

    ToolRegistry와 별도로 유지하는 이유:
        - 에이전트는 도구가 아니라 "도구를 쓰는 주체"다
        - 도구는 ToolRegistry, 에이전트는 AgentRegistry로 개념을 분리
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, agent: AgentDefinition) -> None:
        """
        에이전트를 등록한다.

        같은 name이 이미 있으면 덮어쓴다 (테스트/재설정 용이성).
        프로덕션에서는 부트스트랩 중 1회만 등록되는 것이 정상이다.
        """
        self._agents[agent.name] = agent

    def register_many(self, agents: list[AgentDefinition]) -> None:
        """여러 에이전트를 한 번에 등록한다."""
        for agent in agents:
            self.register(agent)

    def get(self, name: str) -> AgentDefinition | None:
        """
        이름으로 에이전트를 조회한다.

        존재하지 않으면 None 반환 (AgentTool이 에러 처리).
        """
        return self._agents.get(name)

    def list_names(self) -> list[str]:
        """등록된 에이전트 이름 목록을 이름순으로 반환한다."""
        return sorted(self._agents.keys())

    def list_descriptions(self) -> dict[str, str]:
        """
        에이전트 이름 → description 매핑을 반환한다.

        Worker의 시스템 프롬프트에 "사용 가능한 서브 에이전트 목록"을
        주입할 때 사용한다.
        """
        return {name: self._agents[name].description for name in self.list_names()}

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)


# ─────────────────────────────────────────────
# 기본 내장 에이전트 선언
# ─────────────────────────────────────────────

# Scout 서브 에이전트 — v7.0 TIER_S에서 사용
# 읽기 전용 파일 탐색만 수행하는 CPU 4B 모델 기반 에이전트.
# Worker가 "전체 프로젝트 구조 파악"이나 "여러 파일 검색"이 필요하다고
# 판단할 때 호출한다. CPU 기반이라 느리므로(~30초) 단순 질문에는 쓰지 않는다.
SCOUT_AGENT: AgentDefinition = AgentDefinition(
    name="scout",
    description=(
        "Read-only file/document explorer running on CPU (Qwen3.5-4B, slow ~15-30s). "
        "Use when the user asks for broad project exploration, multi-file search, "
        "codebase understanding, OR when analyzing an uploaded document "
        "(PDF/DOCX/XLSX). Do NOT use for simple questions, greetings, or "
        "single-line file edits."
    ),
    system_prompt=(
        "You are Scout, a read-only exploration agent.\n"
        "Your job is to EXPLORE and PLAN, NOT to analyze or summarize.\n"
        "Worker(27B) will do the final analysis — you only gather facts.\n\n"
        "## Tool selection (STRICT)\n"
        "Binary document files — ALWAYS DocumentProcess, NEVER Read:\n"
        "  .pdf, .docx, .doc, .xlsx, .xls, .hwp, .pptx, .ppt\n"
        "Text / code files — use Read (or Glob/Grep for searching).\n"
        "Directory — use LS. Name pattern — Glob. Content search — Grep.\n\n"
        "## Workflow\n"
        "1. Use the right tools to find relevant files / read uploaded documents.\n"
        "2. For big documents, call DocumentProcess chunk_index=0, 1, 2, ... as "
        "needed (up to 5 chunks).\n"
        "3. After gathering, STOP calling tools and output the report below.\n\n"
        "## Output format (MANDATORY — markdown sections)\n"
        "Your final message MUST be exactly 4 markdown sections, in this order, "
        "with these exact headers. No prose before the first header. No JSON.\n\n"
        "## relevant_files\n"
        "- path/to/file1\n"
        "- path/to/file2\n"
        "(or '- none' if no files involved)\n\n"
        "## file_summaries\n"
        "- path/to/file1: one-line factual description\n"
        "- path/to/file2: one-line factual description\n"
        "(or '- none')\n\n"
        "## plan\n"
        "Bullet list of the key facts, numbers, headings, or findings that the "
        "Worker needs to answer the user's question. Be specific — include "
        "section titles, item counts, concrete values. Do NOT write a Korean "
        "prose summary; the Worker will do that. Keep this section under 1500 "
        "characters. Use markdown bullets.\n\n"
        "## requires_tools\n"
        "- Edit (if Worker will need to edit a file)\n"
        "- Bash (if Worker will need to run a command)\n"
        "- (none if Worker only needs to answer from facts)\n\n"
        "After the 4 sections, stop. Do not add a summary paragraph — Worker does "
        "that from your sections."
    ),
    allowed_tools=("Read", "Glob", "Grep", "LS", "DocumentProcess"),
    max_turns=5,
    model_override="scout",
)


# Phase 2 부트스트랩에서 기본으로 등록할 에이전트 목록.
# 공공 납품 환경별로 추가 에이전트를 여기에 넣을 수 있다
# (예: code-reviewer, sql-explorer, compliance-auditor).
DEFAULT_AGENTS: tuple[AgentDefinition, ...] = (SCOUT_AGENT,)


def build_default_agent_registry() -> AgentRegistry:
    """
    기본 에이전트가 등록된 AgentRegistry를 생성한다.

    부트스트랩에서 이 함수를 호출하여 registry를 얻고,
    ToolUseContext.options["agent_registry"]로 전달한다.
    """
    registry = AgentRegistry()
    registry.register_many(list(DEFAULT_AGENTS))
    return registry
