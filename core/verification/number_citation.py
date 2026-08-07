# 답변에 인용된 숫자가 근거 자료(도구 결과)에 실제로 있는지 대조하는 검증기.
"""
숫자 인용 검증 — 모델이 문서에서 옮겨 적은 숫자가 원문에 실제로 있는지 확인한다.

■ 왜 필요한가 (2026-08-06 실측)
  실제 입찰공고문(사업금액 `150,000,000원`)을 분석시키면 모델이 금액을
  `1,500,000,000원`(10배) 또는 `15,000,000,000원`(100배)으로 잘못 옮겼다.
  프롬프트에 "원문 그대로 복사하라 / 억·만으로 환산하지 말라"를 넣자 **단위 환산과
  날짜 재서술은 사라졌지만 자릿수 오류는 남았다** — 질문 형태·실행마다 정확도가
  0/5 ~ 5/5 로 요동쳤다. 지시로 통제되지 않는 오류이므로 코드로 대조한다.

  계약금액을 10배로 잘못 읽는 것은 입찰·계약 문서에서 치명적이다. 그래서 조용히
  넘기지 않고 **눈에 보이는 경고**를 붙인다.

■ 무엇을 검사하나 (오탐을 줄이는 설계)
  자릿수 구분 쉼표가 있는 숫자(`150,000,000`)만 본다. 이런 표기는 거의 항상
  "문서에서 옮겨 적은 값"이지 모델이 계산한 값이 아니기 때문이다.
  그 숫자가 근거 자료에 **문자 그대로 없고**, 동시에 근거 자료에 **자릿수만 다른
  비슷한 값이 있을 때만** 경고한다. 두 조건을 함께 걸어야
    - 모델이 직접 계산한 합계(원문에 없는 게 당연함)
    - 사용자가 질문에 쓴 숫자
  같은 정상 사례를 경고하지 않는다.

■ 무엇을 하지 않나
  값을 **고치지 않는다.** 어떤 값이 맞는지 단정할 수 없는 경우가 있고(모델이 실제로
  계산했을 수도 있다), 조용한 자동 교정은 틀렸을 때 발견조차 안 된다. 사람이
  판단할 수 있도록 사실만 보여 준다.

작성자: 이현수 / 작성일: 2026-08-06
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 자릿수 구분 쉼표가 있는 숫자만 대상으로 한다(1,000 이상). 소수점은 다루지 않는다.
#   예: 150,000,000 / 1,500 / 12,345,678
_GROUPED_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+")

# 경고를 붙일 최대 개수 — 답변 끝에 목록이 길게 붙어 본문을 가리지 않게 한다.
_MAX_WARNINGS = 5

# '전사 오염' 판정에 허용하는 편집 거리(2026-08-07 추가).
#   2 로 둔 근거: 실측된 오염 `506,628` → `500,662` 가 거리 2 다(삽입 1 + 삭제 1).
#   3 이상으로 넓히면 서로 무관한 값끼리 걸려 오탐이 는다. 자릿수가 같아야 한다는
#   조건과 함께 걸어 "베낀 값이 어긋난 경우"만 좁게 잡는다.
_NEAR_MISS_EDITS = 2


@dataclass(frozen=True)
class UncitedNumber:
    """근거 자료에서 확인되지 않은 숫자 하나.

    필드:
      value    — 답변에 적힌 표기(예: "1,500,000,000").
      similar  — 근거 자료에 있는, 자릿수만 다른 비슷한 값들(예: ["150,000,000"]).
                 비어 있지 않을 때만 경고 대상이 된다(위 docstring 근거).
    """

    value: str
    similar: tuple[str, ...]


def _digits(value: str) -> str:
    """쉼표를 뗀 숫자 문자열."""
    return value.replace(",", "")


def _edit_distance_at_most(a: str, b: str, limit: int) -> bool:
    """편집 거리가 limit 이하인지 판정한다(그 이상은 값을 계산하지 않고 False).

    자릿수가 같은데 몇 글자만 다른 값을 찾기 위한 것이다. 숫자 문자열이라
    길이가 짧아(보통 15자 이하) 단순 DP 로 충분하다.
    """
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,  # 삭제
                cur[j - 1] + 1,  # 삽입
                prev[j - 1] + (ca != cb),  # 치환
            )
        # 이 행의 최솟값이 이미 limit 을 넘으면 더 볼 필요가 없다.
        if min(cur) > limit:
            return False
        prev = cur
    return prev[-1] <= limit


def _significant(value: str) -> str:
    """뒤쪽 0을 떼어 '유효 숫자 부분'만 남긴다.

    150,000,000 과 1,500,000,000 은 뒤 0의 개수만 다르고 앞부분은 둘 다 "15" 다.
    이 성질로 "자릿수만 틀린 값"을 찾아낸다. 전부 0이면 "0" 을 돌려준다.
    """
    stripped = _digits(value).rstrip("0")
    return stripped or "0"


def find_uncited_numbers(answer: str, sources: list[str]) -> list[UncitedNumber]:
    """답변의 숫자 중 근거 자료에 없는 것들을 찾는다.

    매개변수:
      answer  — 모델이 낸 최종 답변 텍스트.
      sources — 근거 자료 텍스트들(이 턴의 도구 결과 내용).

    반환:
      경고 대상 목록. 근거 자료가 비었으면(도구를 안 썼으면) 항상 빈 목록이다 —
      대조할 원문이 없는데 경고하는 것은 무의미하기 때문이다.
    """
    if not answer or not sources:
        return []

    source_text = "\n".join(s for s in sources if s)
    if not source_text:
        return []

    # 근거 자료에 있는 숫자들을 두 가지로 색인해 둔다.
    #   ① 유효 숫자 부분 — 자릿수(scale)만 틀린 값을 찾는다.
    #   ② 자릿수 길이     — 같은 길이인데 몇 글자만 다른 값(전사 오염)을 찾는다.
    source_numbers = _GROUPED_NUMBER.findall(source_text)
    by_significant: dict[str, list[str]] = {}
    by_length: dict[int, list[str]] = {}
    for number in source_numbers:
        by_significant.setdefault(_significant(number), []).append(number)
        by_length.setdefault(len(_digits(number)), []).append(number)

    found: list[UncitedNumber] = []
    seen: set[str] = set()
    for number in _GROUPED_NUMBER.findall(answer):
        if number in seen:
            continue
        seen.add(number)
        # 원문에 그대로 있으면 정상이다.
        if number in source_text:
            continue

        # 규칙 ① 자릿수만 다른 값 — 150,000,000 → 1,500,000,000 형태(10·100배 오독).
        similar = [s for s in by_significant.get(_significant(number), []) if s != number]

        # 규칙 ② 전사 오염 — 자릿수는 같은데 몇 글자만 다른 값.
        #   실측(2026-08-07): Calculate 도구가 `506,628` 을 정확히 돌려줬는데 모델이
        #   답변에 `500,662` 라고 옮겨 적었다(2/2 재현). 유효 숫자가 달라 규칙 ①에
        #   걸리지 않는다. 자릿수가 같을 것을 요구해 오탐을 억제한다 —
        #   모델이 실제로 계산한 파생값(부가세·합계 등)은 대개 자릿수가 다르다.
        if not similar:
            digits = _digits(number)
            similar = [
                s
                for s in by_length.get(len(digits), [])
                if s != number and _edit_distance_at_most(digits, _digits(s), _NEAR_MISS_EDITS)
            ]

        if not similar:
            continue
        # 중복 제거하되 원문 등장 순서를 지킨다.
        unique_similar = list(dict.fromkeys(similar))
        found.append(UncitedNumber(value=number, similar=tuple(unique_similar)))

    return found


def build_number_warning(items: list[UncitedNumber]) -> str:
    """경고 문구를 만든다. 붙일 것이 없으면 빈 문자열.

    답변 본문을 고치지 않고 끝에 덧붙이는 용도라, 본문과 구분되도록 구분선을 넣고
    사실만 짧게 적는다.
    """
    if not items:
        return ""

    lines = [
        "",
        "---",
        "⚠️ **숫자 확인 필요** — 아래 값은 문서 원문에서 그대로 찾지 못했습니다.",
    ]
    for item in items[:_MAX_WARNINGS]:
        similar = ", ".join(item.similar)
        lines.append(f"- 답변의 `{item.value}` → 문서에 있는 값: `{similar}`")
    if len(items) > _MAX_WARNINGS:
        lines.append(f"- 그 외 {len(items) - _MAX_WARNINGS}건")
    lines.append("문서 원문을 기준으로 확인해 주세요.")
    return "\n".join(lines)
