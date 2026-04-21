"""
쿼리 라우팅 분류기 단위 테스트 — v7.0 Part 2.5 (2026-04-21).

QueryEngine의 classify_query() 함수가 입력 특성에 따라 적절히
KNOWLEDGE / TOOL 프로필로 분기하는지 검증한다.

분류 규칙(우선순위):
  1. routing.enabled=False       → 항상 TOOL
  2. len(input) >= threshold     → TOOL (문서/로그 첨부 가정)
  3. tool_keywords 중 하나 포함   → TOOL
  4. 그 외                        → KNOWLEDGE
"""
from __future__ import annotations

import pytest

from core.config import RoutingConfig, RoutingProfile
from core.orchestrator.query_engine import _resolve_profile, classify_query


# ─────────────────────────────────────────────
# Fixture: 테스트용 RoutingConfig
# ─────────────────────────────────────────────
@pytest.fixture
def default_routing() -> RoutingConfig:
    """
    기본 활성화 상태의 RoutingConfig를 반환한다.
    키워드와 임계값은 기본값을 그대로 사용한다.
    """
    return RoutingConfig()


@pytest.fixture
def disabled_routing() -> RoutingConfig:
    """라우팅 전체가 비활성인 상태 — 항상 tool_mode로 수렴해야 한다."""
    return RoutingConfig(enabled=False)


# ─────────────────────────────────────────────
# 1) 일반 지식 QA는 KNOWLEDGE로 분류되어야 한다
# ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "user_input",
    [
        "차라투스트라는 이렇게 말했다 에 대해 설명해줘",
        "니체 철학 요약해줘",
        "경제 공황의 원인이 뭐야?",
        "Python GIL이 뭔가요",
        "Explain the eternal recurrence concept",
        "안녕",
    ],
)
def test_classify_query_general_knowledge_returns_knowledge(
    user_input: str, default_routing: RoutingConfig
) -> None:
    """일반 교양/인사/개념 설명은 KNOWLEDGE 경로로 가야 한다."""
    assert classify_query(user_input, default_routing) == "KNOWLEDGE"


# ─────────────────────────────────────────────
# 2) 프로젝트/파일 키워드가 포함되면 TOOL로 분류
# ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "user_input",
    [
        "이 파일 읽어줘",
        "첨부된 문서 요약",
        "업로드한 로그 분석해줘",
        "이 프로젝트 구조 설명",
        "코드베이스에서 auth 관련 함수 찾아봐",
        "Read the config directory",
        "Edit(file_path='a.py')",
        "Agent(subagent_type='scout') 호출해",
        "DocumentProcess 써서 pdf 처리",
        # 프로젝트 경로 언급은 코드 작업으로 간주
        "core/orchestrator 모듈 구조 분석해",
        "web/app.py 수정해",
        "tests/unit 디렉토리 확인해줘",
        "training/ 폴더 뭐 있어",
        # 대문자 도구명이 평문으로 등장
        "Read 도구로 파일 좀 열어봐",
        "Edit the README",
    ],
)
def test_classify_query_tool_keyword_returns_tool(
    user_input: str, default_routing: RoutingConfig
) -> None:
    """도구/프로젝트 힌트가 있으면 TOOL 경로로 가야 한다."""
    assert classify_query(user_input, default_routing) == "TOOL"


# ─────────────────────────────────────────────
# 3) 긴 입력(문서 첨부 추정)은 TOOL로 분류
# ─────────────────────────────────────────────
def test_classify_query_long_input_over_threshold_returns_tool(
    default_routing: RoutingConfig,
) -> None:
    """
    long_input_threshold 이상 길이의 입력은 문서 첨부로 간주하여 TOOL 경로로.
    """
    long_text = "가" * (default_routing.long_input_threshold + 10)
    assert classify_query(long_text, default_routing) == "TOOL"


def test_classify_query_exactly_threshold_returns_tool(
    default_routing: RoutingConfig,
) -> None:
    """임계값과 정확히 같은 길이도 TOOL 경로(경계 포함)."""
    text = "a" * default_routing.long_input_threshold
    assert classify_query(text, default_routing) == "TOOL"


def test_classify_query_just_below_threshold_without_keyword_returns_knowledge(
    default_routing: RoutingConfig,
) -> None:
    """
    임계값 바로 아래 길이 + 키워드 없음이면 KNOWLEDGE.
    """
    # 키워드 없는 일반 문장을 임계값 직전까지 반복
    text = "니체 철학에 대해 간단히 설명해줘. "
    # 임계값 미만이 되도록 자름
    text = text * 10
    text = text[: default_routing.long_input_threshold - 1]
    assert len(text) < default_routing.long_input_threshold
    assert classify_query(text, default_routing) == "KNOWLEDGE"


# ─────────────────────────────────────────────
# 4) enabled=False일 때는 항상 TOOL
# ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "user_input",
    [
        "안녕",
        "니체 철학 요약",
        "이 파일 읽어줘",
        "",
    ],
)
def test_classify_query_disabled_always_returns_tool(
    user_input: str, disabled_routing: RoutingConfig
) -> None:
    """라우팅 비활성 시 분류 없이 TOOL로 수렴한다 (기존 동작 유지)."""
    assert classify_query(user_input, disabled_routing) == "TOOL"


# ─────────────────────────────────────────────
# 5) 대소문자 무시 — 키워드 매칭
# ─────────────────────────────────────────────
def test_classify_query_tool_keyword_is_case_insensitive(
    default_routing: RoutingConfig,
) -> None:
    """'FILE' 대문자로 써도 'file' 키워드와 매칭되어야 한다."""
    assert classify_query("Please read this FILE", default_routing) == "TOOL"


# ─────────────────────────────────────────────
# 6) _resolve_profile — 분류 결과에 맞는 프로필 반환
# ─────────────────────────────────────────────
def test_resolve_profile_knowledge_returns_knowledge_mode(
    default_routing: RoutingConfig,
) -> None:
    """KNOWLEDGE 분류 → knowledge_mode 프로필."""
    profile = _resolve_profile("KNOWLEDGE", default_routing)
    assert profile is default_routing.knowledge_mode
    # 기본값 정합성 확인
    assert profile.model == "qwen3.5-27b"
    assert profile.temperature == pytest.approx(0.2)


def test_resolve_profile_tool_returns_tool_mode(
    default_routing: RoutingConfig,
) -> None:
    """TOOL 분류 → tool_mode 프로필."""
    profile = _resolve_profile("TOOL", default_routing)
    assert profile is default_routing.tool_mode
    assert profile.model == "nexus-phase3"
    assert profile.temperature == pytest.approx(0.3)


def test_resolve_profile_unknown_class_falls_back_to_tool(
    default_routing: RoutingConfig,
) -> None:
    """알 수 없는 분류 문자열이 와도 tool_mode로 안전하게 폴백(fail-safe)."""
    profile = _resolve_profile("UNKNOWN_XYZ", default_routing)
    assert profile is default_routing.tool_mode


# ─────────────────────────────────────────────
# 7) 커스텀 RoutingConfig — 주입된 프로필/키워드가 반영되어야 함
# ─────────────────────────────────────────────
def test_classify_query_custom_keyword_is_respected() -> None:
    """운영자가 커스텀 키워드를 추가하면 분류기가 반영해야 한다."""
    custom = RoutingConfig(
        enabled=True,
        long_input_threshold=10_000,  # 길이 규칙 무효화
        tool_keywords=["특수키워드XYZ"],
        knowledge_mode=RoutingProfile(model="kmodel"),
        tool_mode=RoutingProfile(model="tmodel"),
    )
    assert classify_query("특수키워드XYZ 포함 질문", custom) == "TOOL"
    assert classify_query("니체 철학 설명", custom) == "KNOWLEDGE"


def test_classify_query_custom_threshold_short_text_stays_knowledge() -> None:
    """임계값을 크게 올리면 짧은 문서성 요청도 KNOWLEDGE로 가야 한다."""
    custom = RoutingConfig(
        enabled=True,
        long_input_threshold=10_000,
        tool_keywords=[],  # 키워드 규칙 무효화
    )
    # 600자짜리 일반 설명 요청 — 기본 임계값(500)이면 TOOL이지만,
    # 커스텀 임계값(10_000)에서는 KNOWLEDGE가 되어야 한다.
    text = "니체 철학을 풀어서 설명해줘. " * 30
    assert len(text) > 500
    assert classify_query(text, custom) == "KNOWLEDGE"
