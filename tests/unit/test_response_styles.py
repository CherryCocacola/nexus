# 응답 스타일 프리셋(W1) 검증 — 무회귀 기본값·우선순위·fail-soft.
"""
core/system_prompt/styles.py 계약을 고정한다 (2026-08-05).

[반드시 지켜야 할 것]
  - 기본(normal)은 빈 문자열을 돌려준다 → compose가 섹션을 생략 → **기존 프롬프트와
    완전히 동일**(무회귀). 스타일 기능을 켜도 아무 설정을 안 하면 달라지는 게 없다.
  - 사용자가 직접 쓴 custom 문구가 프리셋보다 우선한다.
  - 파일이 없거나 깨져도 예외를 내지 않는다(스타일은 부가 기능 — 대화를 막으면 안 된다).
"""

from __future__ import annotations

from pathlib import Path

from core.system_prompt.compose import STYLE_HEADER, compose_system_prompt
from core.system_prompt.styles import (
    default_style_id,
    list_styles,
    resolve_style_prompt,
)

REPO_YAML = Path(__file__).resolve().parents[2] / "config" / "response_styles.yaml"


class TestPresets:
    def test_repo_yaml_loads(self) -> None:
        """리포의 실제 프리셋 파일이 정상 로딩된다(오타·문법 오류 회귀 방지)."""
        styles = list_styles(REPO_YAML)
        ids = {s["id"] for s in styles}
        assert {"normal", "concise", "explanatory", "formal"} <= ids
        for s in styles:
            assert s["label"] and isinstance(s["label"], str)

    def test_default_is_normal(self) -> None:
        """기본 스타일은 normal이다."""
        assert default_style_id(REPO_YAML) == "normal"

    def test_normal_yields_empty_prompt(self) -> None:
        """normal은 빈 문구 — 섹션이 생략되어 기존 동작과 같아진다(무회귀 핵심)."""
        assert resolve_style_prompt("normal", path=REPO_YAML) == ""

    def test_presets_have_content(self) -> None:
        """normal 외 프리셋은 실제 지시 문구를 갖는다."""
        for style_id in ("concise", "explanatory", "formal"):
            assert resolve_style_prompt(style_id, path=REPO_YAML).strip()


class TestResolution:
    def test_custom_overrides_preset(self) -> None:
        """사용자가 직접 쓴 문구가 프리셋보다 우선한다."""
        out = resolve_style_prompt("concise", custom_text="무조건 존댓말", path=REPO_YAML)
        assert out == "무조건 존댓말"

    def test_unknown_style_is_ignored(self) -> None:
        """모르는 id는 스타일 없음으로 처리한다(잘못된 값이 대화를 막지 않게)."""
        assert resolve_style_prompt("does-not-exist", path=REPO_YAML) == ""

    def test_none_and_blank_are_safe(self) -> None:
        """None·공백도 안전하게 빈 문자열."""
        assert resolve_style_prompt(None, path=REPO_YAML) == ""
        assert resolve_style_prompt("   ", path=REPO_YAML) == ""

    def test_missing_file_is_fail_soft(self, tmp_path) -> None:
        """파일이 없어도 예외 없이 빈 결과를 돌려준다."""
        missing = tmp_path / "none.yaml"
        assert resolve_style_prompt("concise", path=missing) == ""
        assert list_styles(missing) == []
        assert default_style_id(missing) == "normal"

    def test_broken_yaml_is_fail_soft(self, tmp_path) -> None:
        """YAML이 깨져 있어도 예외를 내지 않는다."""
        broken = tmp_path / "broken.yaml"
        broken.write_text("styles: [unclosed", encoding="utf-8")
        assert resolve_style_prompt("concise", path=broken) == ""


class TestCompositionIntegration:
    def test_style_section_added_after_base(self) -> None:
        """스타일은 base 뒤에 붙는다(prefix cache 보존)."""
        style = resolve_style_prompt("concise", path=REPO_YAML)
        out = compose_system_prompt("BASE", style=style)
        assert out.startswith("BASE")
        assert STYLE_HEADER in out

    def test_normal_style_keeps_prompt_identical(self) -> None:
        """normal이면 base와 완전히 동일한 프롬프트가 나온다(무회귀 증명)."""
        style = resolve_style_prompt("normal", path=REPO_YAML)
        assert compose_system_prompt("BASE", style=style) == "BASE"
