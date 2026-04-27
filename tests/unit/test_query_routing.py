"""
쿼리 라우팅 분류기 단위 테스트 — v7.0 Part 2.5 (2026-04-21).

QueryEngine의 classify_query() 함수가 입력 특성에 따라 적절히
KNOWLEDGE / TOOL 프로필로 분기하는지 검증한다.

분류 규칙(우선순위, Part 2.5.9 v0.14.6 갱신):
  1. routing.enabled=False                           → 항상 TOOL
  2. len(input) >= long_input_threshold              → TOOL (문서/로그 첨부 가정)
  3. tool_keywords 중 하나 포함                       → TOOL
  4. len(input) <= chat_max_length AND chat_keywords → CHAT (인사·잡담)
  5. 그 외                                            → KNOWLEDGE
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
    ],
)
def test_classify_query_general_knowledge_returns_knowledge(
    user_input: str, default_routing: RoutingConfig
) -> None:
    """일반 교양/개념 설명은 KNOWLEDGE 경로로 가야 한다.

    참고: v0.14.6부터 짧은 인사("안녕")는 CHAT으로 분리됨 (Part 2.5.9).
    """
    assert classify_query(user_input, default_routing) == "KNOWLEDGE"


# ─────────────────────────────────────────────
# 1-b) 짧은 인사·잡담은 CHAT으로 분류 (Part 2.5.9 v0.14.6)
# ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "user_input",
    [
        "안녕",
        "안녕!",
        "안녕하세요",
        "좋은 아침이야",
        "굿모닝",
        "잘 자",
        "고마워",
        "감사합니다",
        "hi",
        "Hello there",
        "good night",
        "thanks!",
        "bye",
    ],
)
def test_classify_query_short_greeting_returns_chat(
    user_input: str, default_routing: RoutingConfig
) -> None:
    """짧은 인사·잡담은 CHAT 경로로 분류되어야 한다.

    CHAT은 PromptAssembler에서 KB RAG 단계를 스킵 — kowiki 100만 청크 환경에서
    "안녕" → 가수 "안녕" 청크 주입 같은 부작용을 차단한다.
    """
    assert classify_query(user_input, default_routing) == "CHAT"


def test_classify_query_long_input_with_greeting_is_not_chat(
    default_routing: RoutingConfig,
) -> None:
    """긴 입력 안에 인사어가 우연히 들어 있어도 CHAT이 되지 않는다.

    chat_max_length 임계가 길이 게이트로 작동 — 30자를 넘으면 KNOWLEDGE 경로.
    """
    long_with_greeting = (
        "안녕, 오늘 알베르 카뮈의 이방인 줄거리를 자세히 알려줘"
    )
    assert len(long_with_greeting) > default_routing.chat_max_length
    assert classify_query(long_with_greeting, default_routing) == "KNOWLEDGE"


def test_classify_query_tool_keyword_beats_chat(
    default_routing: RoutingConfig,
) -> None:
    """TOOL 키워드는 CHAT보다 우선 — '이 파일 안녕히 처리'는 TOOL."""
    assert classify_query("이 파일 안녕히 처리해줘", default_routing) == "TOOL"


def test_classify_query_disabled_chat_via_zero_max_length() -> None:
    """chat_max_length=0이면 CHAT 분류 자체가 비활성 (운영자 스위치)."""
    routing = RoutingConfig(chat_max_length=0)
    assert classify_query("안녕", routing) == "KNOWLEDGE"


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


def test_resolve_profile_chat_returns_chat_mode(
    default_routing: RoutingConfig,
) -> None:
    """CHAT 분류 → chat_mode 프로필 (Part 2.5.9 v0.14.6).

    chat_mode는 KNOWLEDGE와 같은 베이스 모델을 쓰지만 max_tokens가 작고
    PromptAssembler가 KB RAG를 주입하지 않는다.
    """
    profile = _resolve_profile("CHAT", default_routing)
    assert profile is default_routing.chat_mode
    assert profile.model == "qwen3.5-27b"
    # 짧은 응답 유도
    assert profile.max_tokens <= 1024


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


# ─────────────────────────────────────────────
# 8) classifier 팩토리 — 전략 패턴 확장 지점 (2026-04-22 리팩토링 4)
# ─────────────────────────────────────────────
# RoutingConfig.classifier_type 문자열로 어떤 QueryClassifier 구현체가
# 선택되는지, 알 수 없는 값이 와도 안전하게 heuristic으로 폴백하는지를
# 검증한다. 새 분류기를 추가할 때 이 테스트가 계약으로 작동한다.
def test_build_classifier_default_returns_heuristic() -> None:
    """classifier_type 미지정 시 HeuristicClassifier 인스턴스를 반환."""
    from core.orchestrator.routing import HeuristicClassifier, build_classifier

    cfg = RoutingConfig()  # classifier_type="heuristic" (기본값)
    clf = build_classifier(cfg)
    assert isinstance(clf, HeuristicClassifier)


def test_build_classifier_explicit_heuristic() -> None:
    """classifier_type="heuristic" 명시도 같은 결과를 낸다."""
    from core.orchestrator.routing import HeuristicClassifier, build_classifier

    cfg = RoutingConfig(classifier_type="heuristic")
    clf = build_classifier(cfg)
    assert isinstance(clf, HeuristicClassifier)


def test_build_classifier_unknown_type_falls_back_to_heuristic() -> None:
    """오타/미구현 타입이 들어와도 서버가 죽지 않고 heuristic으로 폴백해야 한다."""
    from core.orchestrator.routing import HeuristicClassifier, build_classifier

    cfg = RoutingConfig(classifier_type="nonexistent_classifier_xyz")
    clf = build_classifier(cfg)
    assert isinstance(clf, HeuristicClassifier)


def test_routing_resolver_uses_classifier_type_from_config() -> None:
    """RoutingResolver가 주입된 config.classifier_type을 존중해야 한다."""
    from core.orchestrator.routing import HeuristicClassifier, RoutingResolver

    cfg = RoutingConfig(classifier_type="heuristic")
    resolver = RoutingResolver(cfg)
    assert isinstance(resolver._classifier, HeuristicClassifier)


# ─────────────────────────────────────────────
# 9) 구조화된 매칭 (2026-04-22 리팩토링 6) — word boundary / regex
# ─────────────────────────────────────────────
# substring 매칭은 "file"이 "filename"에 오탐하는 한계가 있다.
# word-boundary 매칭과 정규식 매칭으로 이를 해소한다.
def test_word_pattern_matches_whole_word() -> None:
    """word-boundary 매칭은 단어 경계에서만 TOOL로 분류해야 한다."""
    cfg = RoutingConfig(
        enabled=True,
        long_input_threshold=10_000,
        tool_keywords=[],
        tool_word_patterns=["file"],
    )
    # 단어 경계에 걸림 → TOOL
    assert classify_query("Please open this file for me", cfg) == "TOOL"
    # 더 긴 단어의 부분 → KNOWLEDGE (오탐 방지)
    assert classify_query("What is a filename", cfg) == "KNOWLEDGE"


def test_word_pattern_is_case_insensitive() -> None:
    """word-boundary 매칭도 대소문자 무시해야 한다."""
    cfg = RoutingConfig(
        enabled=True,
        long_input_threshold=10_000,
        tool_keywords=[],
        tool_word_patterns=["read"],
    )
    assert classify_query("Please READ the doc", cfg) == "TOOL"


def test_regex_pattern_matches_complex_case() -> None:
    """regex 매칭은 함수 호출 등 복잡 패턴을 구분한다."""
    cfg = RoutingConfig(
        enabled=True,
        long_input_threshold=10_000,
        tool_keywords=[],
        tool_regex_patterns=[r"Read\s*\("],
    )
    # 괄호 앞 공백 유무 상관없이 모두 잡힘
    assert classify_query("call Read(path)", cfg) == "TOOL"
    assert classify_query("call Read (path)", cfg) == "TOOL"
    # 괄호 없는 경우는 매칭 안 됨
    assert classify_query("please read the docs", cfg) == "KNOWLEDGE"


def test_invalid_regex_is_skipped_not_raising() -> None:
    """유효하지 않은 regex는 경고 로그 후 무시되고 서버가 죽지 않아야 한다."""
    cfg = RoutingConfig(
        enabled=True,
        long_input_threshold=10_000,
        tool_keywords=[],
        tool_regex_patterns=["[invalid(regex"],  # 컴파일 실패 예상
    )
    # 예외 없이 KNOWLEDGE로 분류 (인사어가 아니므로 CHAT 분기에도 안 걸림)
    assert classify_query("what is python GIL", cfg) == "KNOWLEDGE"


def test_all_three_matching_kinds_coexist() -> None:
    """substring/word/regex가 모두 정의되면 각자 독립적으로 작동한다."""
    cfg = RoutingConfig(
        enabled=True,
        long_input_threshold=10_000,
        tool_keywords=["첨부"],
        tool_word_patterns=["file"],
        tool_regex_patterns=[r"Read\s*\("],
    )
    assert classify_query("문서 첨부했어요", cfg) == "TOOL"  # substring
    assert classify_query("open the file please", cfg) == "TOOL"  # word
    assert classify_query("call Read(x)", cfg) == "TOOL"  # regex
    # 인사어가 아닌 일반 지식 질의 → KNOWLEDGE
    assert classify_query("니체 철학 요약해줘", cfg) == "KNOWLEDGE"


# ─────────────────────────────────────────────
# 10) RoutingResolver — CHAT 결과 (Part 2.5.9 v0.14.6)
# ─────────────────────────────────────────────
def test_resolver_chat_decision_uses_chat_mode_and_no_kb() -> None:
    """CHAT 분류 시 RoutingDecision이 chat_mode 모델을 쓰고 KB 미주입."""
    from core.orchestrator.routing import RoutingResolver

    cfg = RoutingConfig()
    resolver = RoutingResolver(cfg)
    decision = resolver.resolve("안녕")

    assert decision.query_class == "CHAT"
    assert decision.model_override == cfg.chat_mode.model
    assert decision.max_tokens_cap == cfg.chat_mode.max_tokens
    assert decision.allowed_knowledge_sources is None
    assert decision.inject_knowledge_rag is False


def test_resolver_knowledge_decision_keeps_inject_kb_true() -> None:
    """KNOWLEDGE 분류는 inject_knowledge_rag=True (기존 동작 유지)."""
    from core.orchestrator.routing import RoutingResolver

    cfg = RoutingConfig()
    resolver = RoutingResolver(cfg)
    decision = resolver.resolve("니체 철학 요약해줘")

    assert decision.query_class == "KNOWLEDGE"
    assert decision.inject_knowledge_rag is True


def test_resolver_chat_with_tenant_override_applies_lora() -> None:
    """CHAT도 KNOWLEDGE처럼 tenant.model_override를 적용한다 (Part 2.5.9)."""
    from core.config import TenantConfig
    from core.orchestrator.routing import RoutingResolver

    cfg = RoutingConfig()
    resolver = RoutingResolver(cfg)
    tenant = TenantConfig(
        id="school-a",
        name="A학교",
        model_override="nexus-school-a",
    )
    decision = resolver.resolve("안녕", tenant=tenant)

    assert decision.query_class == "CHAT"
    assert decision.model_override == "nexus-school-a"
    # CHAT에서도 KB는 안 주입
    assert decision.inject_knowledge_rag is False
