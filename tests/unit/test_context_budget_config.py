"""ContextBudgetConfig 무회귀 검증 — 하드코딩 컨텍스트 예산 외부화(B200 티어 Phase 1).

무엇을 지키나:
    컨텍스트 예산(turn state·RAG·도구결과·문서청크·출력토큰 에스컬레이션)을 코드
    하드코딩에서 YAML 설정으로 외부화했다. 이때 **기본값이 외부화 이전 하드코딩 값과
    정확히 같아야** 5090(TIER_S)·기존 테스트 동작이 1도 바뀌지 않는다(무회귀).

왜 이 테스트가 중요한가:
    누군가 ContextBudgetConfig 기본값을 실수로 바꾸면(예: B200 상향값을 기본값에
    잘못 박으면) 5090 운영이 조용히 회귀한다. 이 테스트가 그 변경을 즉시 잡는다.
    B200 상향은 기본값이 아니라 별도 config(nexus_config.b200.yaml)에서 오버라이드한다.
"""

from __future__ import annotations

from core.config import ContextBudgetConfig, NexusConfig

# 외부화 이전에 코드에 박혀 있던 현행 값 — 이 스냅샷과 기본값이 달라지면 회귀.
# (출처: prompt_assembler.py / context_manager.py / document_tool.py / query_loop.py)
EXPECTED_DEFAULTS = {
    "turn_state_tokens": 1000,  # prompt_assembler — 이전 턴 상태 주입 예산
    "project_rag_tokens": 1500,  # prompt_assembler — 프로젝트 RAG 주입 예산
    "knowledge_rag_tokens": 1000,  # prompt_assembler — 지식(kowiki) RAG 주입 예산
    "tool_result_budget": 2048,  # context_manager — 도구 결과 보존 토큰 예산
    "preserve_recent_turns": 3,  # context_manager — 압축 시 보존할 최근 턴 수
    "preserve_recent_tool_results": 2,  # context_manager — 보존할 최근 도구 결과 수
    "document_chunk_size": 2500,  # document_tool CHUNK_SIZE — 문서 분할 청크 크기
    "output_token_escalation": [4096, 8192, 16384],  # query_loop — 출력 토큰 증가 단계
}


def test_context_budget_defaults_match_legacy_hardcoded() -> None:
    """기본값이 외부화 이전 하드코딩 값과 정확히 일치해야 한다(무회귀 핵심)."""
    budget = ContextBudgetConfig()
    for field, expected in EXPECTED_DEFAULTS.items():
        actual = getattr(budget, field)
        assert actual == expected, f"{field}: 기본값 {actual!r} != 현행값 {expected!r}"


def test_nexus_config_has_context_budgets_default() -> None:
    """NexusConfig가 기본 ContextBudgetConfig 인스턴스를 자동으로 가진다."""
    cfg = NexusConfig()
    assert isinstance(cfg.context_budgets, ContextBudgetConfig)
    # 대표 필드 하나로 기본값 연결 확인
    assert cfg.context_budgets.turn_state_tokens == 1000


def test_context_budget_yaml_override_applies() -> None:
    """설정으로 예산을 오버라이드하면 반영된다(B200 상향 경로)."""
    # B200처럼 일부 예산만 상향한 상황을 흉내낸다.
    cfg = NexusConfig(
        context_budgets={"turn_state_tokens": 4000, "document_chunk_size": 24000}
    )
    assert cfg.context_budgets.turn_state_tokens == 4000
    assert cfg.context_budgets.document_chunk_size == 24000
    # 지정하지 않은 필드는 현행 기본값을 그대로 유지해야 한다(부분 오버라이드).
    assert cfg.context_budgets.project_rag_tokens == 1500
    assert cfg.context_budgets.output_token_escalation == [4096, 8192, 16384]
