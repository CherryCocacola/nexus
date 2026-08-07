# 파일을 저장하기 전에 그 내용이 문법에 맞는지 확인하는 검사기.
"""
쓰기 전 구문 검사 — 깨진 파일이 조용히 저장되는 것을 막는다.

[왜 필요한가 — 2026-08-07 실측]
    모델이 리팩터링 결과를 Write 로 저장했는데 내용이 손상돼 있었다.

        c oupon=coupon)    passed = Falseelse:    b ut got '{result}'

    단어 중간에 공백이 끼고 줄바꿈이 탭으로 바뀌어 `SyntaxError` 가 났다. 그런데
    도구는 "파일을 작성했습니다"라고 **성공을 보고했고**, 모델도 사용자도 몰랐다.
    깨진 파일이 조용히 남는 것이 문제의 핵심이다.

    원인은 찾지 못했다. 샘플링 파라미터를 의심해 A/B(rep·freq 3조건 × 3회)를
    돌렸으나 **9회 중 0회 재현**이라 기각했다 — 산발적이다.
    막을 수 없는 손상은 **잡아야** 한다. 파싱 한 번이면 확실히 잡힌다.

[언제 검사하나 — 쓰기 "전"]
    쓴 뒤에 검사하면 기존의 멀쩡한 파일을 이미 덮어쓴 뒤다. 그건 손상이 아니라
    데이터 손실이다. 그래서 디스크에 닿기 전에 판정한다.

[무엇을 검사하나]
    내장 파서로 **정확히** 판정할 수 있는 것만 본다(.py / .json).
    그 외 확장자는 검사하지 않는다 — 어설픈 추측 검사는 정상 파일을 막는다.
    (fail-open 이 맞는 드문 자리다. 확신 없는 차단은 도구를 못 쓰게 만든다.)

작성자: 이현수 / 작성일: 2026-08-07
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

# 검사 가능한 확장자. 늘릴 때는 "표준 라이브러리 파서로 정확히 판정 가능한가"를
# 기준으로 삼는다. 정규식 기반 추측은 넣지 않는다.
CHECKED_SUFFIXES = (".py", ".json")


def syntax_error(file_path: str, content: str) -> str | None:
    """저장할 내용이 확장자에 맞는 문법인지 본다.

    Args:
        file_path: 저장 대상 경로(확장자로 검사 종류를 정한다).
        content:   저장하려는 전체 내용.

    Returns:
        문제가 있으면 사람이 읽을 한 줄 설명(줄 번호 포함), 없으면 None.
    """
    suffix = Path(file_path).suffix.lower()

    if suffix == ".py":
        try:
            ast.parse(content)
        except SyntaxError as e:
            return f"{e.msg} (line {e.lineno})"
        except ValueError as e:
            # NUL 바이트가 섞인 경우 등 — SyntaxError 가 아니라 ValueError 로 온다.
            return str(e)
        return None

    if suffix == ".json":
        # 빈 파일은 의도일 수 있으므로 통과시킨다(예: 나중에 채울 자리).
        if not content.strip():
            return None
        try:
            json.loads(content)
        except ValueError as e:
            return str(e)
        return None

    return None


def rejection_message(detail: str) -> str:
    """거부 사유를 모델이 바로 행동할 수 있는 형태로 만든다.

    "무엇이 잘못됐는지 + 파일은 안전한지 + 다음에 무엇을 하면 되는지" 세 가지를
    담는다. 셋 중 하나라도 빠지면 모델이 같은 실수를 반복하거나, 파일이 이미
    깨진 줄 알고 엉뚱한 복구를 시도한다.
    """
    return (
        f"구문 오류가 있어 저장하지 않았습니다: {detail}\n"
        "파일은 그대로 두었습니다. 내용을 고쳐 다시 저장하세요. "
        "긴 파일을 통째로 쓰다 내용이 깨졌다면, 나눠 쓰거나 필요한 부분만 "
        "바꾸는 편이 안전합니다."
    )
