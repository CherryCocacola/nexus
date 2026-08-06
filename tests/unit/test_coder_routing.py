# 코딩 서브모델 라우팅 검증 — 기본 비활성(무회귀)과 활성 시 판정.
"""
코드 작업 질의를 코딩 전용 모델로 보내는 라우팅 계약을 고정한다 (2026-08-05).

[왜 기본이 꺼져 있나]
  실측상 코딩 특화 모델이 항상 더 낫지 않았다. 원샷 UI 생성은 primary(A.X-4.0)가
  완결성이 높았고 Devstral은 레이아웃이 붕괴했다. 반면 코드 정리·리팩터링은
  코딩 모델이 빨랐다. 그래서 자동 전환은 운영자가 명시적으로 켤 때만 동작한다.

[반드시 지켜야 할 것]
  - routing.coder_enabled 기본 False → use_coder는 항상 False(기존 동작 불변)
  - coder_provider가 없으면 켜져 있어도 기본 프로바이더를 쓴다(fail-safe)
"""

from __future__ import annotations

from core.config import RoutingConfig
from core.orchestrator.routing import RoutingResolver


class TestCoderDetection:
    def test_disabled_by_default(self) -> None:
        """기본 설정에서는 코딩 키워드가 있어도 전환하지 않는다(무회귀)."""
        resolver = RoutingResolver(RoutingConfig())
        decision = resolver.resolve("이 함수를 리팩터링해줘")
        assert decision.use_coder is False

    def test_enabled_detects_keyword(self) -> None:
        """켜면 코딩 키워드가 포함된 질의를 코딩 질의로 판정한다."""
        resolver = RoutingResolver(RoutingConfig(coder_enabled=True))
        assert resolver.resolve("이 함수를 리팩터링해줘").use_coder is True
        assert resolver.resolve("스택트레이스 좀 봐줘").use_coder is True

    def test_enabled_ignores_non_coding_query(self) -> None:
        """켜져 있어도 일반 대화는 전환하지 않는다(한국어 품질 보호)."""
        resolver = RoutingResolver(RoutingConfig(coder_enabled=True))
        assert resolver.resolve("오늘 날씨 어때?").use_coder is False
        assert resolver.resolve("안녕하세요").use_coder is False

    def test_keyword_match_is_case_insensitive(self) -> None:
        """영문 키워드는 대소문자를 가리지 않는다."""
        resolver = RoutingResolver(RoutingConfig(coder_enabled=True))
        assert resolver.resolve("Please REFACTOR this module").use_coder is True

    def test_custom_keywords_respected(self) -> None:
        """운영자가 키워드를 바꾸면 그 목록을 따른다."""
        resolver = RoutingResolver(
            RoutingConfig(coder_enabled=True, coder_keywords=["파이썬"])
        )
        assert resolver.resolve("파이썬 코드 짜줘").use_coder is True
        assert resolver.resolve("리팩터링해줘").use_coder is False  # 목록에 없음

    def test_routing_disabled_yields_no_coder(self) -> None:
        """라우팅 자체가 꺼져 있으면 코딩 전환도 없다(비상 스위치 일관성)."""
        resolver = RoutingResolver(RoutingConfig(enabled=False, coder_enabled=True))
        assert resolver.resolve("리팩터링해줘").use_coder is False


class TestConfigDefaults:
    def test_coder_url_empty_by_default(self) -> None:
        """coder_url 기본값은 빈 문자열 — 프로바이더를 만들지 않아 기존 동작 유지."""
        from core.config import GPUServerConfig

        assert GPUServerConfig().coder_url == ""
        assert GPUServerConfig().coder_model == "devstral-small"

    def test_coder_keywords_have_defaults(self) -> None:
        """키워드 기본값이 비어 있지 않아야 켜자마자 동작한다."""
        assert len(RoutingConfig().coder_keywords) > 0

    def test_test_writing_keywords_excluded(self) -> None:
        """테스트 작성 질의는 코딩 모델로 보내지 않는다.

        2026-08-07 실행 채점 비교: 디버깅·리팩터링은 A.X와 Devstral이 각각 6/6로
        동률이었지만, **테스트 작성만 Devstral이 더 나빴다**(0~1/6 vs 1~2/6).
        rep penalty를 운영값으로 올린 뒤에도 폭주가 1/12 남았다. 그래서 이 범주는
        전환 대상에서 뺐다 — 켜더라도 primary(A.X)가 받는다.
        """
        resolver = RoutingResolver(RoutingConfig(coder_enabled=True))

        assert resolver.resolve("이 함수 테스트 코드 짜줘").use_coder is False
        assert resolver.resolve("write a unit test for this").use_coder is False
        # 동률이 확인된 범주는 그대로 전환된다.
        assert resolver.resolve("이 함수 디버깅해줘").use_coder is True
        assert resolver.resolve("리팩터링 부탁해").use_coder is True
