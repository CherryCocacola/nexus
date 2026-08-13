# 리터럴 인용 검증 — 도구 결과 원문과 답변 표기의 어긋남 대조를 고정한다
"""
2026-08-13. S3/S4 실측에서 나온 두 오독 형태를 각각 고정한다.

    도구 결과 `OMEGA77` → 답변 `OMEGA777`   (자릿수 증식)
    도구 결과 `OMEGA77` → 답변 `오메가77`     (한글 음차)

비전 모델은 3/3 정확히 읽었다. 오독은 **주 모델이 도구 결과를 옮겨 적는 단계**에서
났다. 그래서 이 검증기는 답변이 아니라 '답변 vs 도구 결과'를 본다.

오탐 억제가 이 모듈의 절반이라, 경고하면 안 되는 경우를 함께 고정한다.
"""

from __future__ import annotations

import pytest

from core.verification.literal_citation import (
    build_literal_warning,
    find_misquoted_literals,
)


# ─────────────────────────────────────────────
# 실측 재현 — 잡아야 하는 것
# ─────────────────────────────────────────────
def test_digit_inflation_is_detected() -> None:
    """S3 실측. 자릿수가 하나 늘어난 형태."""
    items = find_misquoted_literals(
        "이미지 우하단에는 OMEGA777 이 적혀 있습니다.",
        ["우하단 텍스트: OMEGA77, 좌상단 텍스트: ALPHA42"],
    )

    assert len(items) == 1
    assert items[0].source == "OMEGA77"
    assert items[0].answer == "OMEGA777"
    assert items[0].kind == "near_miss"


def test_hangul_transliteration_is_detected() -> None:
    """S4 실측. 배포 후 재현했을 때 실제로 나온 형태."""
    items = find_misquoted_literals(
        "우하단 글자는 오메가77 입니다.",
        ["우하단 텍스트: OMEGA77"],
    )

    assert len(items) == 1
    assert items[0].source == "OMEGA77"
    assert items[0].answer == "오메가77"
    assert items[0].kind == "transliteration"


def test_warning_shows_the_source_literal_verbatim() -> None:
    """★이 모듈의 목적. 원문을 그대로 보여 줘야 사람이 판단할 수 있다."""
    items = find_misquoted_literals("값은 NOVA7392 입니다.", ["화면 문자열: NOVA7391"])
    text = build_literal_warning(items)

    assert "`NOVA7391`" in text
    assert "`NOVA7392`" in text
    assert "표기 확인 필요" in text


# ─────────────────────────────────────────────
# 오탐 억제 — 경고하면 안 되는 것
# ─────────────────────────────────────────────
def test_exact_match_is_not_flagged() -> None:
    items = find_misquoted_literals("우하단 글자는 OMEGA77 입니다.", ["텍스트: OMEGA77"])
    assert items == []


def test_case_difference_alone_is_not_flagged() -> None:
    """대소문자만 다른 것은 노이즈 대비 얻는 게 없다."""
    items = find_misquoted_literals("값은 omega77 입니다.", ["텍스트: OMEGA77"])
    assert items == []


def test_no_tool_results_means_no_check() -> None:
    """도구를 안 쓴 턴은 대조할 원문이 없다 — [[number_citation]] 과 같은 규칙."""
    assert find_misquoted_literals("OMEGA777 이라고 합니다.", []) == []
    assert find_misquoted_literals("OMEGA777 이라고 합니다.", [""]) == []


def test_pure_words_are_not_literals() -> None:
    """숫자가 없는 낱말은 대상이 아니다 — 일반 문장에서 오탐이 폭발한다."""
    items = find_misquoted_literals(
        "koje 서비스를 확인했습니다.", ["파일: koje.py 와 kojestats.py 를 읽었습니다"]
    )
    assert items == []


def test_pure_numbers_are_left_to_number_citation() -> None:
    """숫자만 있는 값은 [[number_citation]] 담당 — 두 검증기가 겹치지 않게 한다."""
    items = find_misquoted_literals("금액은 1,500,000,000원", ["사업금액 150,000,000원"])
    assert items == []


def test_two_distinct_literals_both_present_are_not_flagged() -> None:
    """도구 결과에 둘 다 있으면 서로 다른 정상 식별자다(근접해도 경고 금지)."""
    items = find_misquoted_literals(
        "ALPHA42 와 ALPHA43 을 비교했습니다.",
        ["항목: ALPHA42, ALPHA43"],
    )
    assert items == []


@pytest.mark.parametrize(
    ("source", "answer"),
    [
        ("base64", "base32"),  # 길이 6 — 최소 길이에서 걸러짐
        ("int32", "int64"),  # 길이 5
        ("float32", "float64"),  # 길이 7 — 이름으로 걸러짐
        ("sha256", "sha512"),  # 길이 6
    ],
)
def test_common_technical_tokens_are_not_flagged(source: str, answer: str) -> None:
    """코딩 답변에서 자리만 바꿔 쓰는 것이 정상인 토큰들 — 경고하면 노이즈가 된다."""
    items = find_misquoted_literals(
        f"{answer} 로 처리합니다.", [f"현재 구현은 {source} 를 씁니다"]
    )
    assert items == []


def test_unrelated_identifier_is_not_flagged() -> None:
    """편집 거리가 멀면 서로 무관한 값이다."""
    items = find_misquoted_literals("결과는 ZULU9988 입니다.", ["텍스트: OMEGA77"])
    assert items == []


def test_hangul_without_matching_digit_tail_is_not_flagged() -> None:
    """숫자 꼬리가 달라야 음차로 보지 않는다."""
    items = find_misquoted_literals("항목99 를 확인했습니다.", ["텍스트: OMEGA77"])
    assert items == []


# ─────────────────────────────────────────────
# 경고 문구
# ─────────────────────────────────────────────
def test_empty_items_produce_no_warning() -> None:
    assert build_literal_warning([]) == ""


def test_warning_is_capped() -> None:
    """경고가 길어져 본문을 가리지 않게 한다."""
    sources = [" ".join(f"CODE{i:02d}0" for i in range(10))]
    answer = " ".join(f"CODE{i:02d}00" for i in range(10))
    text = build_literal_warning(find_misquoted_literals(answer, sources))

    assert text.count("→ 도구 결과 원문") <= 5
    assert "그 외" in text


# ─────────────────────────────────────────────
# 진입점 통합 — CLI·웹 6경로가 같은 것을 쓴다
# ─────────────────────────────────────────────
def test_post_check_includes_literal_warning() -> None:
    """[[post_check]] 에 배선되지 않으면 어느 표면에서도 동작하지 않는다."""
    from core.message import Message
    from core.verification.post_check import build_answer_warnings

    messages = [Message.tool_result(tool_use_id="t1", content="화면 문자열: OMEGA77")]
    warning = build_answer_warnings("우하단은 오메가77 입니다.", messages)

    assert "OMEGA77" in warning


@pytest.mark.parametrize("answer", ["", "   "])
def test_empty_answer_is_safe(answer: str) -> None:
    assert find_misquoted_literals(answer, ["OMEGA77"]) == []
