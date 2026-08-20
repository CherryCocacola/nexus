# 코딩 질의로 판정되면 질의 클래스도 TOOL 로 맞춰지는지 검증한다.
"""
코더 판정과 질의 클래스 정합 테스트 (2026-08-20).

무엇을 막는가:
    "이 버그를 고쳐줘" 는 coder_keywords 에 걸려 코딩 모델로 가는데, 분류기는
    이를 KNOWLEDGE 로 봤다. 그 결과 코딩 모델에 **사내 문서 RAG 가 주입**되고
    출력이 4096 토큰으로 묶였다(실측). 코딩 모델에게 무관한 문서를 밀어 넣는
    조합이라 방해만 된다.

경계:
    호출자가 클래스를 명시(forced_class)했으면 그 의도가 우선이다 — 08-16 에
    "요청 단위로 클래스를 고정한다"고 정한 계약을 깨지 않는다.
"""

from __future__ import annotations

from core.config import RoutingConfig
from core.orchestrator.routing import RoutingResolver


def _resolver(**kwargs) -> RoutingResolver:
    return RoutingResolver(RoutingConfig(enabled=True, **kwargs))


def test_coder_query_is_not_left_as_knowledge() -> None:
    """코딩 질의면 KNOWLEDGE 로 남지 않는다 — RAG 주입·토큰 제한을 피한다."""
    r = _resolver(coder_enabled=True)
    d = r.resolve("이 버그를 고쳐줘")

    assert d.use_coder is True, "이 문장은 코더 키워드에 걸려야 한다(전제)"
    assert d.query_class == "TOOL"
    assert d.inject_knowledge_rag is False


def test_coder_disabled_keeps_original_class() -> None:
    """코더가 꺼져 있으면 분류를 건드리지 않는다(무회귀)."""
    r = _resolver(coder_enabled=False)
    d = r.resolve("이 버그를 고쳐줘")

    assert d.use_coder is False
    assert d.query_class == "KNOWLEDGE"


def test_forced_class_wins_over_alignment() -> None:
    """호출자가 클래스를 명시하면 보정하지 않는다(08-16 요청 단위 고정 계약)."""
    r = _resolver(coder_enabled=True)
    d = r.resolve("이 버그를 고쳐줘", forced_class="KNOWLEDGE")

    assert d.query_class == "KNOWLEDGE"


def test_plain_knowledge_query_is_untouched() -> None:
    """코딩과 무관한 지식 질의는 그대로 KNOWLEDGE 다."""
    r = _resolver(coder_enabled=True)
    d = r.resolve("광합성 원리를 설명해줘")

    assert d.use_coder is False
    assert d.query_class == "KNOWLEDGE"
