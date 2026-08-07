# "실행해서 확인했다"는 주장이 실제 도구 실행으로 뒷받침되는지 검증한다.
"""
`execution_claim` 의 계약을 고정한다 (2026-08-07).

[왜 생겼나 — 실측]
    코딩 과제에서 모델이 도구를 **하나도 쓰지 않은 채** 이렇게 답했다.

        "### 실행 결과 확인 방법
         터미널에서 다음 명령어로 실행하여 확인합니다: python solution.py
         ### 예상 출력 결과
         모든 테스트가 통과했습니다."

    실행한 적이 없는데 "통과했습니다"라고 적었다. 같은 날 시스템 프롬프트에
    "실제로 실행하고 출력한 것을 보고하라"를 넣었는데도 나온 답이다.
    지시로 막히지 않으므로 코드로 대조한다.

[오탐이 더 무서운 검사다]
    안내문("다음 명령어로 실행하세요")이나 도구를 실제로 쓴 정상 답변에 경고가 붙으면
    사용자가 경고를 무시하게 되고, 그러면 진짜 경고도 함께 묻힌다.
    아래 절반이 그 오탐을 막는 테스트다.
"""

from __future__ import annotations

from core.verification.execution_claim import (
    build_execution_warning,
    find_execution_claims,
)

# R2 실측 응답의 핵심 부분.
OBSERVED = (
    "### 실행 결과 확인 방법\n"
    "터미널에서 다음 명령어로 실행하여 확인합니다:\n"
    "python solution.py\n"
    "### 예상 출력 결과\n"
    "모든 테스트가 통과했습니다."
)


# ─────────────────────────────────────────────
# 검출
# ─────────────────────────────────────────────


def test_detects_the_observed_false_claim():
    """★실측된 거짓 주장을 잡는다."""
    claims = find_execution_claims(OBSERVED)

    assert claims, "실측 사례를 못 잡았다"
    assert any("통과했습니다" in c for c in claims)


def test_detects_common_korean_completion_forms():
    for text in (
        "코드를 실행해 보았고 정상 동작함을 확인했습니다.",
        "테스트가 모두 통과했습니다.",
        "검증을 완료했습니다.",
        "실행 결과 정상이었습니다.",
    ):
        assert find_execution_claims(text), f"놓침: {text}"


def test_detects_english_forms():
    for text in ("I ran the tests and they pass.", "All tests passed."):
        assert find_execution_claims(text), f"놓침: {text}"


# ─────────────────────────────────────────────
# ★오탐 방지 — 여기가 더 중요하다
# ─────────────────────────────────────────────


def test_instructional_text_is_not_a_claim():
    """'실행하세요' 같은 안내는 주장이 아니다 — 완료형만 잡는다."""
    for text in (
        "터미널에서 `python solution.py` 로 실행하세요.",
        "아래 명령을 실행하면 결과를 볼 수 있습니다.",
        "테스트를 실행할 때는 pytest 를 쓰면 됩니다.",
        "이 코드를 실행하기 전에 의존성을 설치해야 합니다.",
    ):
        assert not find_execution_claims(text), f"안내문에 경고가 붙는다: {text}"


def test_no_warning_when_tools_were_used():
    """★도구를 실제로 썼으면 경고하지 않는다 — 정상 답변에 붙으면 안 된다."""
    claims = find_execution_claims("테스트가 모두 통과했습니다.")

    assert claims  # 주장 자체는 있다
    assert build_execution_warning(claims, tool_result_count=1) == ""
    assert build_execution_warning(claims, tool_result_count=3) == ""


def test_warning_when_no_tools_at_all():
    claims = find_execution_claims(OBSERVED)
    warning = build_execution_warning(claims, tool_result_count=0)

    assert "실행 확인 필요" in warning
    assert "도구를 실행한 기록이 없습니다" in warning
    assert "통과했습니다" in warning  # 어느 문장이 문제인지 보여 준다


def test_no_claims_means_no_warning():
    """평범한 답변에는 아무 것도 붙지 않는다."""
    assert find_execution_claims("파이썬 리스트와 튜플의 차이를 설명하면…") == []
    assert build_execution_warning([], 0) == ""


def test_empty_answer_is_safe():
    assert find_execution_claims("") == []


# ─────────────────────────────────────────────
# 표시
# ─────────────────────────────────────────────


def test_warning_caps_quote_count_and_length():
    """경고가 본문을 가리지 않도록 개수와 길이를 제한한다."""
    many = [f"테스트{i}가 통과했습니다. " + "긴내용" * 40 for i in range(6)]
    warning = build_execution_warning(many, tool_result_count=0)

    assert "그 외 3건" in warning
    assert all(len(line) < 120 for line in warning.splitlines())


def test_code_blocks_are_not_scanned():
    """★코드 안의 문자열은 주장이 아니다.

    실서버 첫 시험에서 경고가 `print("모든 테스트가 통과되었습니다.")` 를 인용했다.
    모델이 **작성한 코드**이지 모델의 주장이 아니다. 코드까지 세면 경고가 잡음으로
    채워지고, 그러면 진짜 경고도 함께 무시된다.
    """
    fenced = (
        "아래 코드를 참고하세요.\n\n"
        "```python\n"
        'print("모든 테스트가 통과되었습니다.")\n'
        "```\n\n"
        "필요하면 직접 돌려 보세요."
    )
    assert find_execution_claims(fenced) == []

    inline = "출력은 `테스트가 통과했습니다` 형태가 됩니다."
    assert find_execution_claims(inline) == []


def test_claim_outside_code_block_still_caught():
    """코드는 걷어내되, 코드 밖의 진짜 주장은 그대로 잡는다."""
    mixed = (
        "```python\n"
        "def add(a, b):\n    return a + b\n"
        "```\n\n"
        "테스트가 모두 통과했습니다."
    )
    claims = find_execution_claims(mixed)

    assert len(claims) == 1
    assert "통과했습니다" in claims[0]
