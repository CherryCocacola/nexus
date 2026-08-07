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


# ─────────────────────────────────────────────
# 규칙② 전사 오염 — 자릿수는 같은데 몇 글자만 다른 값 (2026-08-07)
# ─────────────────────────────────────────────
#
# 실측: Calculate 도구가 `506,628` 을 정확히 돌려줬는데 모델이 답변에 `500,662` 라고
# 옮겨 적었다(2/2 재현). 유효 숫자가 달라 규칙①(자릿수만 틀린 값)에 걸리지 않는다.
# 도구가 정답을 준 턴에는 서버에 정답이 있으므로, 이 형태를 잡을 수 있어야 한다.


def test_transcription_corruption_is_caught():
    """★실측 재현 — 도구 결과 506,628 을 답변이 500,662 로 옮겼다."""
    from core.verification.number_citation import find_uncited_numbers

    found = find_uncited_numbers(
        "18,764 곱하기 27은 500,662입니다.", ["18764 * 27 = 506,628"]
    )

    assert [f.value for f in found] == ["500,662"]
    assert found[0].similar == ("506,628",)


def test_different_digit_length_does_not_trigger_rule2():
    """자릿수가 다르면 규칙②는 발동하지 않는다 — 오탐 억제의 핵심 조건.

    모델이 실제로 계산한 파생값(부가세·1/10 등)은 대개 자릿수가 다르다.
    """
    from core.verification.number_citation import _digits, _significant

    # 1,500,000(7자리)과 15,000,000(8자리) — 길이가 달라 규칙② 대상이 아니다.
    assert len(_digits("1,500,000")) != len(_digits("15,000,000"))
    # (이 조합은 규칙①이 잡는다 — 유효 숫자가 둘 다 "15" 이므로.)
    assert _significant("1,500,000") == _significant("15,000,000")


def test_far_numbers_of_same_length_are_not_flagged():
    """자릿수가 같아도 충분히 다르면 경고하지 않는다(무관한 값 오탐 방지)."""
    from core.verification.number_citation import find_uncited_numbers

    assert find_uncited_numbers("직원 123,456명", ["매출 987,654원"]) == []
    assert find_uncited_numbers("약 1,235,000원", ["정확히 1,234,567원"]) == []


def test_edit_distance_helper_bounds():
    """편집 거리 계산기가 상한을 지키는지 — 넘으면 계산을 이어가지 않는다."""
    from core.verification.number_citation import _edit_distance_at_most

    assert _edit_distance_at_most("506628", "500662", 2) is True  # 실측 케이스
    assert _edit_distance_at_most("506628", "506628", 2) is True  # 동일
    assert _edit_distance_at_most("123456", "987654", 2) is False  # 완전히 다름
    assert _edit_distance_at_most("1234567", "123456", 0) is False  # 길이 차 > 한계
