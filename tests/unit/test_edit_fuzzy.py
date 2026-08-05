# Edit/MultiEdit 공백 정규화 폴백 매칭 검증 — 정확 매칭 실패 시 안전한 복구.
"""
배경(실측): A.X-4.0이 Edit의 old_string 들여쓰기·공백을 정확히 재현하지 못해
5연속 실패 후 작업을 포기하는 사례가 관찰됐다(2026-08-04 홈페이지 e2e).
보완: ①공백 정규화 유일 매치 시 폴백 적용 ②실패 시 근접 원문 힌트+Write 전환 팁.

고정하는 계약:
  - 정확 매칭은 종전과 100% 동일(무회귀).
  - 폴백은 정규화 매치가 "정확히 1곳"일 때만. 2곳 이상/0곳이면 실패(fail-closed).
  - 실패 메시지에 근접 부분 힌트와 "Read→Write 재작성" 팁 포함.
"""

from __future__ import annotations

import pytest

from core.tools.base import ToolUseContext
from core.tools.implementations.edit_tool import (
    EditTool,
    closest_match_hint,
    find_whitespace_fuzzy_span,
)
from core.tools.implementations.multi_edit_tool import MultiEditTool

SAMPLE = """<html>
<body>
  <div class="hero">
    <h1>IDINO NOVA</h1>
  </div>
  <script src="app.js"></script>
</body>
</html>
"""


def _ctx() -> ToolUseContext:
    return ToolUseContext(cwd=".")


# ─── 헬퍼 단위 ───


def test_fuzzy_span_finds_indent_mismatch() -> None:
    """들여쓰기가 달라도 내용이 같으면 유일 구간의 원본 오프셋을 찾는다."""
    old = '<div class="hero">\n<h1>IDINO NOVA</h1>\n</div>'  # 들여쓰기 없음
    span = find_whitespace_fuzzy_span(SAMPLE, old)
    assert span is not None
    start, end = span
    assert SAMPLE[start:end] == '  <div class="hero">\n    <h1>IDINO NOVA</h1>\n  </div>'


def test_fuzzy_span_ambiguous_returns_none() -> None:
    """정규화 매치가 2곳이면 모호 — None(fail-closed)."""
    content = "a\n  x\nb\n    x\nc\n"
    assert find_whitespace_fuzzy_span(content, "x") is None


def test_fuzzy_span_whitespace_only_returns_none() -> None:
    """공백뿐인 old_string은 폴백 대상이 아니다(오매칭 위험)."""
    assert find_whitespace_fuzzy_span(SAMPLE, "   \n  ") is None


def test_closest_hint_returns_nearby_original() -> None:
    """오타가 있어도 가장 비슷한 줄 주변의 '원문'을 힌트로 돌려준다."""
    hint = closest_match_hint(SAMPLE, '<h1>IDINO NOVAA</h1>')
    assert "<h1>IDINO NOVA</h1>" in hint


# ─── EditTool 통합 ───


@pytest.mark.asyncio
async def test_edit_exact_match_unchanged(tmp_path) -> None:
    """정확 매칭 경로는 종전과 동일하게 동작한다(무회귀)."""
    f = tmp_path / "page.html"
    f.write_text(SAMPLE, encoding="utf-8")

    result = await EditTool().call(
        {
            "file_path": str(f),
            "old_string": "<h1>IDINO NOVA</h1>",
            "new_string": "<h1>NOVA</h1>",
        },
        _ctx(),
    )

    assert not result.is_error
    assert "<h1>NOVA</h1>" in f.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_edit_fuzzy_fallback_applies(tmp_path) -> None:
    """들여쓰기 불일치 old_string도 유일 매치면 폴백으로 교체된다."""
    f = tmp_path / "page.html"
    f.write_text(SAMPLE, encoding="utf-8")

    result = await EditTool().call(
        {
            "file_path": str(f),
            # 들여쓰기를 전부 뺀 잘못된 old_string(모델 실수 재현)
            "old_string": '<div class="hero">\n<h1>IDINO NOVA</h1>\n</div>',
            "new_string": '  <div class="hero">\n    <h1>NOVA</h1>\n  </div>',
        },
        _ctx(),
    )

    assert not result.is_error
    assert "공백 정규화 매칭" in result.data
    text = f.read_text(encoding="utf-8")
    assert "<h1>NOVA</h1>" in text
    assert "IDINO NOVA</h1>" not in text


@pytest.mark.asyncio
async def test_edit_not_found_error_has_hint_and_tip(tmp_path) -> None:
    """폴백까지 실패하면 근접 원문 힌트 + Write 재작성 팁을 담아 실패한다."""
    f = tmp_path / "page.html"
    f.write_text(SAMPLE, encoding="utf-8")

    result = await EditTool().call(
        {
            "file_path": str(f),
            "old_string": '<h1>IDINO NOVAAA</h1>',  # 내용 자체가 틀림 → 폴백도 실패
            "new_string": "<h1>X</h1>",
        },
        _ctx(),
    )

    assert result.is_error
    assert "가장 비슷한 부분" in result.error_message
    assert "<h1>IDINO NOVA</h1>" in result.error_message  # 원문 힌트
    assert "Write로 파일 전체를 재작성" in result.error_message
    # 파일은 변경되지 않아야 한다
    assert f.read_text(encoding="utf-8") == SAMPLE


# ─── MultiEditTool 통합 ───


@pytest.mark.asyncio
async def test_multiedit_fuzzy_fallback_applies(tmp_path) -> None:
    """MultiEdit도 동일 폴백을 공유한다 — 들여쓰기 불일치 편집이 성공한다."""
    f = tmp_path / "page.html"
    f.write_text(SAMPLE, encoding="utf-8")

    result = await MultiEditTool().call(
        {
            "edits": [
                {
                    "file_path": str(f),
                    "old_string": '<div class="hero">\n<h1>IDINO NOVA</h1>\n</div>',
                    "new_string": "<section>NOVA</section>",
                }
            ]
        },
        _ctx(),
    )

    assert not result.is_error
    assert "공백 정규화 매칭" in str(result.data)
    assert "<section>NOVA</section>" in f.read_text(encoding="utf-8")
