# 쓰기 전 구문 검사 — 깨진 파일이 조용히 저장되지 않는지 검증한다.
"""
`syntax_validator` 와 Write·Edit 배선의 계약을 고정한다 (2026-08-07).

[왜 생겼나 — 실측]
    모델이 리팩터링 결과를 Write 로 저장했는데 내용이 손상돼 있었다.

        c oupon=coupon)    passed = Falseelse:    b ut got '{result}'

    `SyntaxError` 가 나는 파일인데 도구는 **"파일을 작성했습니다"라고 성공을
    보고했다.** 모델도 사용자도 몰랐다. 조용히 깨진 파일이 남는 것이 핵심 문제다.

    원인은 못 찾았다(샘플링 A/B 9회에서 0회 재현 → 산발적). 막을 수 없으면 잡아야 한다.

[이 테스트가 지키는 것]
    1. 깨진 내용은 **디스크에 닿기 전에** 거부된다(기존 파일 파괴 방지).
    2. 정상 파일과 검사 대상이 아닌 확장자는 막지 않는다(과잉 차단 방지).
    3. Write 가 거부되면 모델이 Edit 로 갈 텐데, Edit 에도 같은 검사가 있어야 한다.
"""

from __future__ import annotations

import pytest

from core.tools.base import ToolUseContext
from core.tools.implementations.edit_tool import EditTool
from core.tools.implementations.write_tool import WriteTool
from core.tools.validation.syntax_validator import syntax_error

CTX = ToolUseContext(cwd=".")

# 실측된 손상 내용 그대로 — 단어 중간 공백, 줄바꿈 소실.
CORRUPTED = 'def f(coupon):\n    c oupon=coupon)\n    passed = Falseelse:\n'


# ─────────────────────────────────────────────
# 검사기 자체
# ─────────────────────────────────────────────


def test_detects_the_observed_corruption():
    """★실측된 손상을 잡는다."""
    assert syntax_error("solution.py", CORRUPTED) is not None


def test_reports_line_number():
    """줄 번호를 알려 준다 — 모델이 어디를 고쳐야 하는지 알아야 한다."""
    detail = syntax_error("a.py", "def f():\n    return (1\n")
    assert detail and "line" in detail


def test_valid_python_passes():
    assert syntax_error("a.py", "def f():\n    return 1\n") is None


def test_valid_json_passes_and_broken_json_fails():
    assert syntax_error("a.json", '{"a": 1}') is None
    assert syntax_error("a.json", '{"a": 1,}') is not None


def test_empty_json_is_allowed():
    """빈 파일은 의도일 수 있다(나중에 채울 자리)."""
    assert syntax_error("a.json", "   ") is None


def test_unchecked_suffixes_are_never_blocked():
    """검사할 수 없는 확장자는 막지 않는다 — 추측 검사는 정상 파일을 막는다."""
    for name in ("a.md", "a.txt", "a.yaml", "a.sh", "noext"):
        assert syntax_error(name, "이건 파이썬이 아니다 ( { [") is None


# ─────────────────────────────────────────────
# Write 배선 — 디스크에 닿기 전에 막는가
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_rejects_broken_python(tmp_path):
    target = tmp_path / "solution.py"
    result = await WriteTool().call(
        {"file_path": str(target), "content": CORRUPTED}, CTX
    )

    assert result.is_error
    assert "저장하지 않았습니다" in result.error_message
    assert not target.exists(), "거부했는데 파일이 만들어졌다"


@pytest.mark.asyncio
async def test_write_does_not_destroy_existing_file(tmp_path):
    """★가장 중요한 성질 — 멀쩡한 기존 파일이 깨진 내용으로 덮이지 않는다."""
    target = tmp_path / "good.py"
    original = "def good():\n    return 42\n"
    target.write_text(original, encoding="utf-8")

    result = await WriteTool().call(
        {"file_path": str(target), "content": CORRUPTED}, CTX
    )

    assert result.is_error
    assert target.read_text(encoding="utf-8") == original, "원본이 손상됐다"


@pytest.mark.asyncio
async def test_write_still_saves_valid_python(tmp_path):
    target = tmp_path / "ok.py"
    result = await WriteTool().call(
        {"file_path": str(target), "content": "x = 1\n"}, CTX
    )

    assert not result.is_error
    assert target.read_text(encoding="utf-8") == "x = 1\n"


@pytest.mark.asyncio
async def test_write_still_saves_non_code_files(tmp_path):
    """마크다운 등은 종전대로 저장된다(과잉 차단 방지)."""
    target = tmp_path / "note.md"
    result = await WriteTool().call(
        {"file_path": str(target), "content": "# 제목 ( { ["}, CTX
    )

    assert not result.is_error
    assert target.exists()


# ─────────────────────────────────────────────
# Edit 배선 — Write 가 막히면 모델이 이쪽으로 온다
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_edit_rejects_change_that_breaks_syntax(tmp_path):
    """Edit 로도 파일을 깨뜨릴 수 없다 — 여기가 뚫리면 Write 차단이 무의미하다."""
    target = tmp_path / "m.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")

    result = await EditTool().call(
        {
            "file_path": str(target),
            "old_string": "return 1",
            "new_string": "return (1",  # 괄호가 닫히지 않는다
        },
        CTX,
    )

    assert result.is_error
    assert "저장하지 않았습니다" in result.error_message
    assert target.read_text(encoding="utf-8") == "def f():\n    return 1\n"


@pytest.mark.asyncio
async def test_edit_still_applies_valid_change(tmp_path):
    target = tmp_path / "m.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")

    result = await EditTool().call(
        {"file_path": str(target), "old_string": "return 1", "new_string": "return 2"},
        CTX,
    )

    assert not result.is_error
    assert "return 2" in target.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# 통짜 쓰기를 덜 하도록 유도 (2026-08-07)
# ─────────────────────────────────────────────
#
# 구문 검사는 .py/.json 손상만 잡는다. .md·.txt·.sh 는 파서가 없어 못 잡으므로,
# 애초에 파일 전문을 Write 인자로 넘기는 일 자체를 줄이는 편이 낫다.
# 길이 상한 같은 마찰은 넣지 않았다 — 손상이 9회 중 0회 재현이라 근거가 부족하다.


def test_write_description_steers_to_edit():
    """도구 설명이 '기존 파일은 Edit' 를 가리켜야 한다(모델이 보는 문구)."""
    description = WriteTool().description

    assert "prefer Edit" in description
    assert "existing file" in description


def test_cli_prompt_prefers_edit_over_write():
    """CLI 프롬프트에도 같은 방향이 있어야 한다."""
    from core.bootstrap import _build_expanded_system_prompt

    prompt = _build_expanded_system_prompt()

    assert "Prefer Edit over Write" in prompt
    # 기존 폴백 규칙(Edit 2회 실패 시 Write)과 모순되지 않아야 한다.
    assert "use Write to rewrite it" in prompt
