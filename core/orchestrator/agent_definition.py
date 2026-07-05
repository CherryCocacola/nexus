"""
서브 에이전트 정의 모듈 — Worker가 AgentTool로 호출할 수 있는 "부하 직원(서브
에이전트)"의 명세를 담는다.

■ 이 파일이 하는 일 (한눈에)
  메인 에이전트인 Worker(Qwen 27B)가 혼자 처리하기 버거운 작업(예: 프로젝트
  전체 구조 파악, 여러 파일 검색, 업로드 문서 분석)을 만나면, 별도의 작은
  에이전트에게 "이 일 좀 대신 해줘"라고 위임할 수 있다. 그 위임 대상이 바로
  서브 에이전트다. 이 파일은 그런 서브 에이전트가 "누구이고 / 어떤 도구를 쓸
  수 있고 / 어떤 모델로 도는지"를 기록하는 명세(AgentDefinition)와, 그 명세들을
  모아 두는 저장소(AgentRegistry)를 정의한다.

■ 배경 (왜 이렇게 바뀌었나)
  v7.0 Phase 9 재설계에서 Scout는 "무조건 먼저 도는 자동 전처리기"에서 "필요할
  때만 호출되는 서브 에이전트"로 승격되었다. 즉 Worker가 맥락을 보고 스스로
  판단하여 Scout를 부른다. 덕분에 단순 질문에는 Scout를 건너뛰어 속도가 빨라진다.

  사양서 Ch 14 Agent & Task System의 AgentDefinition 명세를 코드로 구체화한 것이다.

■ 핵심 구성 요소 (이 파일에 들어 있는 것)
  - AgentDefinition : 서브 에이전트 1명의 이름/프롬프트/도구/모델 명세 (frozen)
  - AgentRegistry   : 선언된 서브 에이전트들을 등록·조회하는 저장소(레지스트리)
  - SCOUT_AGENT     : v7.0 TIER_S에서 쓰는 읽기 전용 탐색 에이전트의 실제 선언
  - build_default_agent_registry() : 기본 에이전트가 채워진 레지스트리 생성 함수

■ 누가 이 모듈을 쓰나 (호출 관계)
  - 부트스트랩(Phase 2)이 build_default_agent_registry()를 호출해 레지스트리를 만들고,
    ToolUseContext.options["agent_registry"]에 실어 아래로 흘려보낸다.
  - AgentTool이 실행될 때 subagent_type 인자로 받은 이름을 레지스트리에서 조회하여,
    해당 AgentDefinition의 프롬프트/도구/모델 설정을 그대로 적용한다.

■ 설계 원칙
  1. AgentDefinition은 불변(frozen dataclass) — 런타임에 수정 불가(안전성 보장)
  2. allowed_tools는 문자열 튜플 — ToolRegistry와 이름으로만 느슨하게 결합
  3. model_override는 이름 기반("scout") — 실제 프로바이더는 AgentTool이 해석
  4. description은 Worker의 "이 에이전트를 언제 쓸지" 판단에 쓰이는 힌트 — 명확히 작성

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# AgentDefinition — 서브 에이전트 명세 (불변)
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class AgentDefinition:
    """
    서브 에이전트 1명의 불변 명세(설정 카드).

    이 객체 하나가 "서브 에이전트 한 명"을 통째로 설명한다. 실행 로직은 담지
    않고, 순수하게 설정값(이름/프롬프트/도구/모델)만 들고 있는 데이터 컨테이너다.

    흐름: Worker가 AgentTool을 호출할 때 subagent_type 인자로 이 name을 지정하면,
    AgentTool은 그 이름으로 AgentRegistry에서 이 AgentDefinition을 찾아, 여기 적힌
    설정(어떤 프롬프트로 / 어떤 도구만 허용하고 / 어떤 모델로 몇 턴까지)을
    그대로 적용해 서브 에이전트를 실제로 구동한다.

    필드 설명:
        name: 에이전트 식별자. AgentTool 호출 시 subagent_type 값으로 쓰인다.
        description: Worker가 "언제 이 에이전트를 써야 하는지" 판단하는 힌트.
            이 문자열은 모델(Worker)에게 그대로 노출되므로, 용도와 비용(느림 등)을
            명확하게 적어야 Worker가 올바르게 선택한다.
        system_prompt: 서브 에이전트가 구동될 때 주입되는 시스템 프롬프트(행동 지침).
        allowed_tools: 이 에이전트가 쓸 수 있는 도구 이름들의 튜플. 실제 도구 객체가
            아니라 "이름"만 담으며, ToolRegistry에서 이 이름으로 도구를 조회한다.
        max_turns: 서브 에이전트가 돌 수 있는 최대 턴 수. 무한 루프를 막는 안전장치.
        model_override: 사용할 모델의 이름 힌트("scout" 또는 None).
            None이면 부모(Worker)와 같은 ModelProvider를 그대로 쓴다.
            값이 있으면 AgentTool이 context.options에서 실제 Provider를 해석해 교체한다.

    왜 frozen(불변)인가:
        서브 에이전트 정의는 한번 선언되면 실행 도중 절대 바뀌면 안 되기 때문이다.
        예컨대 도중에 allowed_tools가 조작되어 권한 모드를 우회하는 실행을 막고,
        여러 곳에서 같은 정의를 공유해도 값이 흔들리지 않아 레지스트리 일관성을
        지킬 수 있다. frozen=True라 필드에 재할당을 시도하면 즉시 예외가 난다.
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
    서브 에이전트 저장소(레지스트리).

    선언된 AgentDefinition들을 name → 정의 형태의 딕셔너리로 담아 두고,
    이름으로 꺼내 쓰는 아주 단순한 조회 창고다. 내부적으로는 dict 하나가 전부다.

    사용 시점:
        - Phase 2 부트스트랩 때 기본 에이전트(SCOUT_AGENT 등)를 여기에 등록하고,
        - 이후 AgentTool이 subagent_type(에이전트 이름)으로 여기서 정의를 조회한다.

    왜 ToolRegistry와 따로 두나:
        - 에이전트는 "도구(Tool)"가 아니라 "도구를 쓰는 주체"이기 때문이다.
        - 그래서 도구는 ToolRegistry가, 에이전트는 AgentRegistry가 관리하도록
          개념을 분리했다. 두 저장소를 섞지 않는 것이 아키텍처 원칙이다.
    """

    def __init__(self) -> None:
        # 이름(name) → AgentDefinition 매핑을 담는 내부 딕셔너리.
        # 외부에서 직접 건드리지 않도록 밑줄(_)을 붙여 비공개로 표시한다.
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, agent: AgentDefinition) -> None:
        """
        에이전트 정의 1개를 레지스트리에 등록한다.

        같은 name이 이미 있으면 조용히 덮어쓴다. 이는 테스트나 재설정을 쉽게
        하기 위한 의도적 동작이다. 다만 프로덕션에서는 부트스트랩 중 딱 1회만
        등록되는 것이 정상이며, 실행 중 재등록이 일어나면 설정 실수일 가능성이 크다.
        """
        # 이름을 키로 사용하므로, 같은 이름이면 기존 값이 새 값으로 교체된다.
        self._agents[agent.name] = agent

    def register_many(self, agents: list[AgentDefinition]) -> None:
        """
        여러 에이전트를 한 번에 등록한다.

        내부적으로 register()를 반복 호출할 뿐이라, 등록 규칙(덮어쓰기 등)은 동일하다.
        부트스트랩에서 DEFAULT_AGENTS 목록을 통째로 넣을 때 주로 쓴다.
        """
        for agent in agents:
            self.register(agent)

    def get(self, name: str) -> AgentDefinition | None:
        """
        이름으로 에이전트 정의를 조회한다.

        찾으면 AgentDefinition을, 없으면 None을 반환한다. 여기서 예외를 던지지 않고
        None을 돌려주는 이유는, 없는 에이전트에 대한 에러 처리를 호출자(AgentTool)가
        맥락에 맞게 담당하도록 책임을 넘기기 위해서다.
        """
        return self._agents.get(name)

    def list_names(self) -> list[str]:
        """
        등록된 에이전트 이름 목록을 이름순(사전순)으로 정렬해 반환한다.

        정렬해서 돌려주는 이유: 출력 순서를 항상 일정하게 만들어, 프롬프트에
        주입될 때 순서가 흔들리지 않도록(재현성/캐시 안정성) 하기 위함이다.
        """
        return sorted(self._agents.keys())

    def list_descriptions(self) -> dict[str, str]:
        """
        에이전트 이름 → description 매핑을 반환한다.

        Worker의 시스템 프롬프트에 "지금 쓸 수 있는 서브 에이전트 목록과 각각의
        용도"를 주입할 때 사용한다. 이름순으로 정렬된 list_names()를 기반으로 만들어
        순서가 항상 일정하다.
        """
        return {name: self._agents[name].description for name in self.list_names()}

    def __contains__(self, name: str) -> bool:
        # `name in registry` 문법을 지원하기 위한 특수 메서드.
        # 해당 이름의 에이전트가 등록돼 있는지 True/False로 알려준다.
        return name in self._agents

    def __len__(self) -> int:
        # `len(registry)` 문법을 지원하기 위한 특수 메서드.
        # 현재 등록된 에이전트 수를 반환한다.
        return len(self._agents)


# ─────────────────────────────────────────────
# 기본 내장 에이전트 선언
# ─────────────────────────────────────────────

# Scout 서브 에이전트 선언 — v7.0 TIER_S에서 사용하는 대표 서브 에이전트.
#
# 성격: 읽기 전용 "탐색·정찰" 에이전트. CPU에서 도는 작은 모델(Qwen3.5-4B)을
#   쓰며, 파일/문서를 뒤져 사실만 그러모아 오는 역할이다. 스스로 결론을 내리지
#   않고, 최종 분석은 Worker(27B)에게 넘긴다(system_prompt에 그 규칙이 박혀 있다).
#
# 언제 호출되나: Worker가 "프로젝트 전체 구조 파악", "여러 파일 검색", 또는
#   "업로드된 문서(PDF/DOCX/XLSX) 분석"이 필요하다고 판단할 때 부른다.
#
# 주의(비용): CPU 기반이라 느리다(대략 15~30초). 그래서 단순 질문·인사·한 줄
#   수정 같은 가벼운 작업에는 쓰지 않는다. 이 비용 특성은 아래 description에도
#   그대로 적어 두어 Worker가 남용하지 않도록 유도한다.
#
# 아래 생성자에 넘기는 문자열들(description, system_prompt)은 모델에게 그대로
# 전달되는 "계약"이므로, 함부로 수정하면 Scout의 출력 형식/판단이 깨질 수 있다.
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
    # 읽기 전용 탐색에 필요한 도구만 화이트리스트로 허용한다. 쓰기/실행 계열
    # 도구(Edit, Write, Bash 등)는 일부러 빼서, 정찰용 에이전트가 파일을
    # 건드리지 못하도록 fail-closed 원칙을 지킨다.
    allowed_tools=("Read", "Glob", "Grep", "LS", "DocumentProcess"),
    # 탐색은 몇 번의 도구 호출이면 끝나므로 턴 상한을 짧게(5) 둬서 무한 루프와
    # CPU 낭비를 막는다.
    max_turns=5,
    # "scout"라는 이름 힌트만 남기고, 실제 어떤 ModelProvider인지는 AgentTool이
    # 런타임에 해석한다(여기서는 프로바이더 객체를 직접 참조하지 않는다).
    model_override="scout",
)


# Phase 2 부트스트랩에서 기본으로 등록할 에이전트 목록(튜플).
# 지금은 Scout 하나뿐이지만, 공공 납품 환경별로 추가 에이전트를 여기에 넣으면
# 자동으로 함께 등록된다(예: code-reviewer, sql-explorer, compliance-auditor).
DEFAULT_AGENTS: tuple[AgentDefinition, ...] = (SCOUT_AGENT,)


def build_default_agent_registry() -> AgentRegistry:
    """
    기본 에이전트들이 미리 등록된 AgentRegistry를 만들어 반환한다.

    이 프로젝트에서 서브 에이전트를 쓰기 위한 "출발점" 함수다. 부트스트랩이
    이 함수를 호출해 완성된 registry를 얻은 뒤, ToolUseContext.options의
    "agent_registry" 키에 실어 아래 계층으로 전달한다. 그러면 AgentTool이 그
    registry에서 subagent_type으로 정의를 조회할 수 있게 된다.

    반환: DEFAULT_AGENTS가 모두 등록된 새 AgentRegistry 인스턴스.
    """
    # 1) 빈 레지스트리를 새로 만든다.
    registry = AgentRegistry()
    # 2) 기본 에이전트 목록(튜플)을 리스트로 바꿔 한꺼번에 등록한다.
    registry.register_many(list(DEFAULT_AGENTS))
    # 3) 준비가 끝난 레지스트리를 호출자에게 돌려준다.
    return registry
