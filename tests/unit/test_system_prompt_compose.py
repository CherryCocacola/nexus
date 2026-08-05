# 시스템 프롬프트 단일 조립기 검증 — 멱등·순서·빈 섹션 처리.
"""
compose_system_prompt()의 계약을 고정한다 (2026-08-05).

[왜 중요한가]
  기존에는 세 진입점이 각자 문자열을 덧붙여(`base + "[사용자 지시]" + x`) 순서와
  형식이 제각각이었고, 엔진을 재사용하면 지시가 중복될 위험이 있었다. 이 함수는
  **항상 base에서 전체를 다시 만든다** — 같은 입력이면 몇 번 호출해도 결과가 같다.

[순서를 고정하는 이유]
  base가 항상 맨 앞이어야 vLLM prefix cache가 적중한다. 자주 바뀌는 스타일·지시가
  앞에 오면 캐시가 매번 깨진다.
"""

from __future__ import annotations

from core.system_prompt.compose import (
    PROJECT_HEADER,
    SESSION_HEADER,
    STYLE_HEADER,
    USER_HEADER,
    compose_system_prompt,
)


class TestComposition:
    def test_base_only_returns_base(self) -> None:
        """섹션이 없으면 base를 그대로 돌려준다."""
        assert compose_system_prompt("BASE") == "BASE"

    def test_sections_appear_in_fixed_order(self) -> None:
        """스타일 → 사용자 → 프로젝트 → 세션 순서를 지킨다(base가 맨 앞)."""
        out = compose_system_prompt(
            "BASE",
            style="S",
            user_instruction="U",
            project_instruction="P",
            session_instruction="X",
        )
        positions = [
            out.index("BASE"),
            out.index(STYLE_HEADER),
            out.index(USER_HEADER),
            out.index(PROJECT_HEADER),
            out.index(SESSION_HEADER),
        ]
        assert positions == sorted(positions)

    def test_empty_sections_are_omitted(self) -> None:
        """빈 값·공백만 있는 값은 헤더까지 통째로 생략한다."""
        out = compose_system_prompt("BASE", user_instruction="U", project_instruction="   ")
        assert USER_HEADER in out
        assert PROJECT_HEADER not in out
        assert STYLE_HEADER not in out

    def test_is_idempotent_from_same_base(self) -> None:
        """같은 base·같은 입력이면 몇 번 호출해도 결과가 동일하다(중복 누적 없음)."""
        first = compose_system_prompt("BASE", user_instruction="U")
        second = compose_system_prompt("BASE", user_instruction="U")
        assert first == second
        assert first.count(USER_HEADER) == 1

    def test_body_is_stripped(self) -> None:
        """섹션 본문의 앞뒤 공백은 정리해 형식을 일정하게 유지한다."""
        out = compose_system_prompt("BASE", user_instruction="\n\n  U  \n\n")
        assert f"{USER_HEADER}\nU" in out

    def test_empty_base_still_composes(self) -> None:
        """base가 비어 있어도(테스트·폴백 경로) 섹션만으로 조립된다."""
        out = compose_system_prompt("", user_instruction="U")
        assert out.startswith(USER_HEADER)

    def test_sections_separated_by_blank_line(self) -> None:
        """섹션 사이는 빈 줄로 구분해 모델이 경계를 인식하게 한다."""
        out = compose_system_prompt("BASE", user_instruction="U")
        assert out == f"BASE\n\n{USER_HEADER}\nU"
