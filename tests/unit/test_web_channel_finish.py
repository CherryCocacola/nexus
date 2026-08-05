# 웹 채널 해석 + finish_reason 매핑 검증 — 외부 소비자 히스토리 격리·잘림 신호.
"""
2026-08-05 실측 문제 2건에 대한 회귀 방지 테스트.

문제 ①: /v1/chat·/v1/chat/stream 이 channel="web" 하드코딩이라, API 키로 붙은
        외부 소비자(VSCode 플러그인)의 대화가 웹 사용자 히스토리에 그대로 섞였다.
        → X-Client-Channel 헤더로 격리하되, 화이트리스트로만 허용(경로 순회 차단).
문제 ②: OpenAI 응답의 finish_reason이 항상 "stop"이라, 토큰 한도로 잘린 응답을
        클라이언트가 완결로 오해했다(잘린 JSON 파싱 실패).
        → StopReason.MAX_TOKENS 는 "length"로 정직하게 매핑.

정책 고정:
  - web/app → 같은 "web"(브라우저·앱 사용자는 이력 공유)
  - cli/api → 각자 독립 채널
  - 미지정·미지값 → "web"(무회귀)
"""

from __future__ import annotations

import pytest

from core.message import StopReason
from web.app import _map_finish_reason, _resolve_channel


class TestResolveChannel:
    """X-Client-Channel 헤더 → 저장 채널 변환 규칙을 고정한다."""

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("web", "web"),
            ("app", "web"),  # 앱은 웹과 이력을 공유한다(요구사항)
            ("APP", "web"),  # 대소문자 무관
            (" cli ", "cli"),  # 공백 허용
            ("cli", "cli"),
            ("api", "api"),
        ],
    )
    def test_known_channels(self, header: str, expected: str) -> None:
        """화이트리스트 값은 정확히 대응 채널로 변환된다."""
        assert _resolve_channel(header) == expected

    @pytest.mark.parametrize("header", [None, "", "   "])
    def test_missing_defaults_to_web(self, header: str | None) -> None:
        """헤더가 없으면 종전대로 web — 기존 웹 UI 무회귀."""
        assert _resolve_channel(header) == "web"

    @pytest.mark.parametrize(
        "header",
        ["../etc", "web/../..", "..", "/absolute", "web;rm -rf", "unknown", "WEB2"],
    )
    def test_unknown_or_traversal_falls_back_to_web(self, header: str) -> None:
        """모르는 값·경로 순회 시도는 전부 web으로 떨어진다(fail-closed).

        channel은 디렉토리명과 Redis 키 네임스페이스가 되므로, 임의 문자열이
        통과하면 저장 경로를 벗어날 수 있다. 화이트리스트 외에는 절대 통과 금지.
        """
        assert _resolve_channel(header) == "web"

    def test_non_string_input_is_safe(self) -> None:
        """문자열이 아닌 값(FastAPI Header 객체 등)도 web으로 안전 처리한다.

        핸들러를 라우팅 없이 직접 호출하는 테스트에서는 기본값인 Header 객체가
        그대로 전달된다. 이때 문자열 메서드를 호출하면 AttributeError로 죽으므로
        타입을 먼저 확인한다(실측 회귀 방지).
        """
        from fastapi import Header

        assert _resolve_channel(Header(default=None)) == "web"
        assert _resolve_channel(123) == "web"  # type: ignore[arg-type]
        assert _resolve_channel(object()) == "web"  # type: ignore[arg-type]

    def test_only_whitelisted_values_can_be_returned(self) -> None:
        """어떤 입력이 와도 반환값은 3종 중 하나뿐임을 보장한다."""
        candidates = ["web", "app", "cli", "api", "../x", None, "", "zzz", "APP"]
        for c in candidates:
            assert _resolve_channel(c) in {"web", "cli", "api"}


class TestMapFinishReason:
    """StopReason → OpenAI finish_reason 매핑을 고정한다."""

    def test_max_tokens_maps_to_length(self) -> None:
        """토큰 한도 도달은 length — 클라이언트가 잘림을 인지해야 한다."""
        assert _map_finish_reason(StopReason.MAX_TOKENS) == "length"

    def test_max_tokens_string_value_also_maps(self) -> None:
        """문자열 값으로 와도 동일하게 판정한다(방어적)."""
        assert _map_finish_reason("max_tokens") == "length"

    @pytest.mark.parametrize(
        "reason",
        [StopReason.END_TURN, StopReason.STOP_SEQUENCE, StopReason.TOOL_USE],
    )
    def test_other_reasons_map_to_stop(self, reason: StopReason) -> None:
        """그 외 종료 이유는 정상 종료(stop)로 본다."""
        assert _map_finish_reason(reason) == "stop"

    def test_none_maps_to_stop(self) -> None:
        """종료 이유를 못 받았으면 종전 동작(stop)을 유지한다(무회귀)."""
        assert _map_finish_reason(None) == "stop"
