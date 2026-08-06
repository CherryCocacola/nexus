# 답변 숫자와 근거 자료를 대조하는 검증기의 탐지·오탐 억제 동작 검증.
"""
core.verification.number_citation 단위 테스트.

무엇을 지키나:
    - 문서 원문에 없는데 '자릿수만 다른 비슷한 값'이 원문에 있으면 경고한다
      (실측된 실패: 150,000,000 → 1,500,000,000 / 15,000,000,000).
    - 오탐을 억제한다. 아래는 경고하지 않는다.
        · 원문에 그대로 있는 값(정상 인용)
        · 원문에 비슷한 값조차 없는 값(모델이 직접 계산한 합계 등)
        · 근거 자료가 아예 없는 경우(도구를 쓰지 않은 턴)
    - 경고 문구는 답변 본문을 고치지 않고 덧붙일 수 있는 형태여야 한다.

왜 '비슷한 값이 있을 때만' 경고하나:
    쉼표 표기 숫자는 대개 문서 인용이지만, 모델이 계산한 합계일 수도 있다.
    "원문에 없음" 하나만으로 경고하면 계산값마다 잘못된 경고가 붙는다. 그래서
    "원문에 자릿수만 다른 값이 있다"는 두 번째 조건을 함께 건다.
"""

from __future__ import annotations

from core.verification.number_citation import (
    build_number_warning,
    find_uncited_numbers,
)

# 실제 입찰공고문에서 가져온 형태의 근거 자료.
SOURCE = [
    "[문서: bid.hwp] 2. 사업금액: 금 150,000,000원(부가세 포함)\n"
    "견적 기준: 20,000명 이상, 5개월 이용 기준"
]


# ─────────────────────────────────────────────
# 탐지
# ─────────────────────────────────────────────


def test_detects_ten_times_error():
    """10배 오독(150,000,000 → 1,500,000,000)을 잡는다."""
    found = find_uncited_numbers("사업금액은 1,500,000,000원입니다.", SOURCE)

    assert [f.value for f in found] == ["1,500,000,000"]
    assert found[0].similar == ("150,000,000",)


def test_detects_hundred_times_error():
    """100배 오독도 같은 방식으로 잡힌다."""
    found = find_uncited_numbers("사업금액은 15,000,000,000원입니다.", SOURCE)

    assert [f.value for f in found] == ["15,000,000,000"]


def test_reports_each_value_once():
    """같은 오답이 여러 번 나와도 한 번만 보고한다."""
    answer = "금액은 1,500,000,000원이며, 1,500,000,000원을 기준으로 산정합니다."

    assert len(find_uncited_numbers(answer, SOURCE)) == 1


# ─────────────────────────────────────────────
# 오탐 억제
# ─────────────────────────────────────────────


def test_correct_quote_is_not_flagged():
    """원문에 그대로 있는 값은 경고하지 않는다."""
    assert find_uncited_numbers("사업금액은 150,000,000원입니다.", SOURCE) == []


def test_computed_value_is_not_flagged():
    """원문에 비슷한 값조차 없는 숫자(모델 계산값)는 경고하지 않는다."""
    assert find_uncited_numbers("월 환산 시 3,750,000원입니다.", SOURCE) == []


def test_other_quoted_number_is_not_flagged():
    """원문의 다른 인용값(20,000명)도 정상 통과한다."""
    assert find_uncited_numbers("20,000명 이상 기준입니다.", SOURCE) == []


def test_no_sources_means_no_warning():
    """도구를 쓰지 않아 근거 자료가 없으면 대조 자체를 하지 않는다."""
    assert find_uncited_numbers("금액은 1,500,000,000원입니다.", []) == []
    assert find_uncited_numbers("금액은 1,500,000,000원입니다.", [""]) == []


def test_empty_answer_is_safe():
    """빈 답변에도 예외 없이 빈 결과."""
    assert find_uncited_numbers("", SOURCE) == []


def test_plain_numbers_without_separator_are_ignored():
    """쉼표 없는 숫자는 대상이 아니다(연도·조항 번호 등 오탐 방지)."""
    assert find_uncited_numbers("제12조 및 2026년 기준입니다.", SOURCE) == []


# ─────────────────────────────────────────────
# 경고 문구
# ─────────────────────────────────────────────


def test_warning_contains_both_values():
    """경고에 '답변의 값'과 '문서의 값'이 모두 드러난다."""
    text = build_number_warning(find_uncited_numbers("금액은 1,500,000,000원입니다.", SOURCE))

    assert "1,500,000,000" in text
    assert "150,000,000" in text
    assert "숫자 확인 필요" in text


def test_warning_is_empty_when_nothing_found():
    """붙일 것이 없으면 빈 문자열 — 본문에 아무 것도 덧붙지 않는다."""
    assert build_number_warning([]) == ""


def test_warning_caps_the_list():
    """항목이 많아도 목록이 무한정 길어지지 않는다."""
    source = ["\n".join(f"{i}00,000,000" for i in range(1, 10))]
    answer = " ".join(f"{i}0,000,000,000" for i in range(1, 10))

    text = build_number_warning(find_uncited_numbers(answer, source))

    assert text.count("→ 문서에 있는 값") <= 5
    assert "그 외" in text
