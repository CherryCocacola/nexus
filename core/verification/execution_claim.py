# "실행해서 확인했다"는 주장이 실제 도구 실행으로 뒷받침되는지 대조하는 검증기.
"""
실행 주장 검증 — 관측 없이 성공을 단정하는 답변을 잡는다.

■ 왜 필요한가 (2026-08-07 실측)
  코딩 과제에서 모델이 도구를 **하나도 쓰지 않은 채** 이렇게 답했다.

      "### 실행 결과 확인 방법
       터미널에서 다음 명령어로 실행하여 확인합니다: python solution.py
       ### 예상 출력 결과
       모든 테스트가 통과했습니다."

  실행한 적이 없는데 "통과했습니다"라고 적었다. 예측을 관측인 양 쓴 것이다.

  같은 날 시스템 프롬프트에 "실행할 수 있는 코드를 썼으면 실제로 실행하고 출력한
  것을 보고하라"를 넣었는데도 이 답이 나왔다. **지시로는 막히지 않는다**는 것이
  확인됐으므로 코드로 대조한다(숫자 인용 검증과 같은 판단이다).

■ 무엇을 검사하나 (오탐을 줄이는 설계)
  "실행/테스트를 **마쳤다**"는 완료형 주장이 있는데 **이번 턴에 도구 결과가 하나도
  없을 때**만 경고한다. 두 조건을 함께 걸어야 다음이 걸리지 않는다.

    · "다음 명령어로 실행하세요" 같은 안내문(완료형이 아니다)
    · 도구를 실제로 써서 결과를 보고한 정상 답변(도구 결과가 있다)

  도구를 **썼는데** 그 주장이 맞는지까지는 판단하지 않는다. 어떤 도구 결과가 어떤
  문장을 뒷받침하는지 기계적으로 잇는 것은 오탐이 크다. 여기서는 "아무 것도 실행하지
  않았는데 실행했다고 말한 경우"라는 가장 분명한 사례만 잡는다.

■ 무엇을 하지 않나
  답변을 고치지 않는다. 사용자가 문서·로그를 붙여넣어 그것을 근거로 말한 정상적인
  경우도 있으므로, 단정하지 않고 "확인이 필요하다"고만 알린다.

작성자: 이현수 / 작성일: 2026-08-07
"""

from __future__ import annotations

import re

# 실행·검증을 **마쳤다**고 단정하는 표현들.
#   완료형만 넣는다. "실행하세요"·"실행하면" 같은 안내·가정형은 제외한다.
#   영어는 과거형/현재완료만 본다.
_CLAIM_PATTERNS = (
    r"실행(?:해|하여|해서)?\s*(?:보았|봤|했)\w*",
    r"실행\s*결과\s*[,:]?\s*(?:정상|성공|통과)",
    r"(?:테스트|테스트가|모두|전부|모든\s*테스트가)\s*통과(?:했|하였|됐|되었)\w*",
    r"통과(?:했|하였|됐|되었)습니다",
    r"정상(?:적으로)?\s*(?:동작|작동)(?:함|하는\s*것)?을?\s*확인(?:했|하였)\w*",
    r"검증(?:을)?\s*(?:완료|마쳤|했)\w*",
    r"\bI\s+ran\b",
    r"\bran\s+the\s+(?:tests?|script|code)\b",
    r"\b(?:all\s+)?tests?\s+(?:passed|pass)\b",
)
_CLAIM_RE = re.compile("|".join(_CLAIM_PATTERNS), re.IGNORECASE)

# 코드 블록은 검사 대상이 아니다.
#   실서버 첫 시험에서 경고가 `print("모든 테스트가 통과되었습니다.")` 를 인용했다.
#   그건 모델의 주장이 아니라 모델이 **작성한 코드**다. 코드 안의 문자열까지 주장으로
#   세면 경고가 잡음으로 채워지고, 그러면 진짜 경고도 함께 무시된다.
_FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")

# 경고에 보여줄 최대 인용 개수.
_MAX_QUOTES = 3
# 인용 문장이 길면 잘라 낸다.
_QUOTE_CHARS = 60


def find_execution_claims(answer: str) -> list[str]:
    """답변에서 '실행·검증을 마쳤다'는 완료형 주장 문장을 찾는다.

    Returns:
        주장이 담긴 문장들(원문 그대로, 중복 제거). 없으면 빈 목록.
    """
    if not answer:
        return []

    # 코드 블록·인라인 코드를 먼저 걷어낸다(위 상수 주석의 실측 근거 참조).
    prose = _INLINE_CODE.sub(" ", _FENCED_BLOCK.sub("\n", answer))

    found: list[str] = []
    seen: set[str] = set()
    # 문장 단위로 끊어 어느 문장이 문제인지 그대로 보여 준다.
    for raw in re.split(r"(?<=[.!?。])\s+|\n+", prose):
        sentence = raw.strip()
        if not sentence or sentence in seen:
            continue
        if _CLAIM_RE.search(sentence):
            seen.add(sentence)
            found.append(sentence)
    return found


def build_execution_warning(claims: list[str], tool_result_count: int) -> str:
    """경고 문구를 만든다. 붙일 것이 없으면 빈 문자열.

    Args:
        claims:            find_execution_claims 의 결과.
        tool_result_count: 이번 턴의 도구 결과 개수. 1개 이상이면 경고하지 않는다.
    """
    if not claims or tool_result_count > 0:
        return ""

    lines = [
        "",
        "---",
        "⚠️ **실행 확인 필요** — 이번 답변에서는 도구를 실행한 기록이 없습니다.",
    ]
    for sentence in claims[:_MAX_QUOTES]:
        quote = sentence if len(sentence) <= _QUOTE_CHARS else sentence[:_QUOTE_CHARS] + "…"
        lines.append(f"- {quote}")
    if len(claims) > _MAX_QUOTES:
        lines.append(f"- 그 외 {len(claims) - _MAX_QUOTES}건")
    lines.append(
        "실제로 실행해 확인한 것이 아니라면 예상 결과일 수 있습니다. 직접 확인해 주세요."
    )
    return "\n".join(lines)
