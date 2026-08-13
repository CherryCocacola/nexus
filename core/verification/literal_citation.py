# 답변에 옮겨 적은 식별자·코드가 도구 결과 원문과 같은지 대조하는 검증기.
"""
리터럴 인용 검증 — 모델이 도구 결과에서 옮겨 적은 **문자+숫자 식별자**를 대조한다.

■ 왜 필요한가 (2026-08-12 S3/S4 실측)
  이미지에 `OMEGA77` 을 그려 넣고 AnalyzeImage 로 읽혔다. 비전 모델(터널 18004)에
  직접 물으면 640x400·2000x1400 모두 전체/크롭 **3/3 정확**했다. 그런데 대화로
  돌리면 답이 `OMEGA777`(자릿수 증식) 또는 `오메가77`(한글 음차)이 나왔다.

  즉 오독이 난 곳은 **비전이 아니라 주 모델이 도구 결과를 옮겨 적는 단계**다.
  S4 에서 region 크롭을 붙였지만 그건 이 오독을 고치지 못한다 — 원인 단계가 다르다.

  같은 계열의 결함을 이 리포는 이미 세 번 겪었다.
    · 문서 금액 자릿수 오독        → [[number_citation]] 로 대조
    · 긴 다운로드 URL 재현 붕괴    → 모델이 적지 않게 하고 서버가 tool_result 에서 추출
    · 긴 업로드 파일명 재현 취약   → 짧은 ASCII 이름으로 강제
  공통 처방은 하나다 — **재생성 단계를 줄이거나, 원문을 서버가 그대로 보여 준다.**
  여기서는 후자를 한다.

■ 무엇을 검사하나 (오탐을 줄이는 설계)
  도구 결과에 있는 "주목할 리터럴"만 본다 — 글자 3개 이상 + 숫자 2개 이상을 모두 가진
  5자 이상 토큰(`OMEGA77`, `NOVA7391`, `ALPHA42`). 이런 표기는 거의 항상 화면·문서에서
  읽어 온 값이지 모델이 지어낸 말이 아니다. 순수 낱말(`koje`)이나 순수 숫자는
  각각 오탐이 크고 [[number_citation]] 이 이미 담당하므로 제외한다.

  그 리터럴이 답변에 **대소문자 무시하고도 없을 때만**, 아래 둘 중 하나를 찾는다.
    A) 같은 문자 계열의 근접값 — 편집 거리 2 이하 (`OMEGA77` → `OMEGA777`)
    B) 한글 음차              — 같은 숫자 꼬리를 단 한글 토큰 (`OMEGA77` → `오메가77`)
  둘 다 답변 쪽 후보가 도구 결과에 없을 것을 함께 요구한다. 도구 결과에 둘 다 있으면
  서로 다른 정상 리터럴이기 때문이다.

■ 무엇을 하지 않나
  답변을 고치지 않는다. [[number_citation]] 과 같은 이유다 — 조용한 자동 교정은
  틀렸을 때 발견조차 되지 않는다. **원문 리터럴을 그대로 보여 주고** 사람이 판단하게 한다.

작성자: 이현수 / 작성일: 2026-08-13
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 검사 대상 토큰 — 영문자·숫자·밑줄·하이픈으로 이어진 5자 이상 덩어리.
# 경계에 한글이 오면 별도 규칙(B)에서 다루므로 여기서는 ASCII 계열만 본다.
_ASCII_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{3,}")

# 한글이 섞이고 숫자 꼬리가 붙은 토큰 — 음차 후보(`오메가77`).
_HANGUL_TAIL_TOKEN = re.compile(r"[가-힣]{2,}\d{2,}")

# 리터럴로 인정할 최소 조건.
_MIN_LETTERS = 3  # 글자 수
_MIN_DIGITS = 2  # 숫자 개수
# 최소 길이 7 — 흔한 기술 토큰을 통째로 제외하기 위한 값이다.
#   제외됨: base64(6) sha256(6) int32(5) utf16(5) x86_64(6)
#   포함됨: OMEGA77(7) ALPHA42(7) NOVA7391(8) CODE0001(8)
# 코딩 답변에는 `base64`↔`base32` 처럼 **정상적으로 자리만 다른** 기술 토큰이 흔해서,
# 길이를 낮추면 경고가 노이즈가 된다. 실측된 오독은 전부 7자 이상이었다.
_MIN_TOKEN_LEN = 7

# 길이 7 이상이면서도 자리만 바꿔 쓰는 것이 정상인 기술 토큰들.
#   `float32` ↔ `float64` 는 편집 거리 2 라 규칙 A 에 걸리지만 오독이 아니다.
#   길이로는 못 거르므로 이름으로 뺀다(대소문자 무시). 추측이 아니라 실제 충돌만 적는다.
_TECH_TOKENS = frozenset(
    {
        "float32",
        "float64",
        "uint32",
        "uint64",
        "int32",
        "int64",
        "utf16le",
        "sha3512",
        "base32",
        "base64",
        "sha1024",
    }
)

# 근접값 판정에 허용하는 편집 거리.
#   2 로 둔 근거: 실측 오독 `OMEGA77` → `OMEGA777` 이 거리 1 이다. 여유를 하나 두되
#   3 이상으로 넓히면 서로 무관한 식별자끼리 걸린다([[number_citation]] 과 같은 기준).
_NEAR_MISS_EDITS = 2

# 답변 끝에 붙일 경고 최대 개수 — 본문을 가리지 않게 한다.
_MAX_WARNINGS = 5


@dataclass(frozen=True)
class MisquotedLiteral:
    """도구 결과와 어긋나 보이는 리터럴 하나.

    필드:
      source — 도구 결과에 있는 **원문 그대로의** 표기(예: "OMEGA77").
      answer — 답변에 적힌 표기(예: "OMEGA777" 또는 "오메가77").
      kind   — "near_miss"(같은 문자 계열) 또는 "transliteration"(한글 음차).
    """

    source: str
    answer: str
    kind: str


def _letters(token: str) -> int:
    return sum(1 for ch in token if ch.isascii() and ch.isalpha())


def _digits(token: str) -> int:
    return sum(1 for ch in token if ch.isdigit())


def _is_notable(token: str) -> bool:
    """글자와 숫자를 함께 가진 충분히 긴 식별자만 검사 대상으로 삼는다."""
    if len(token) < _MIN_TOKEN_LEN:
        return False
    if token.lower() in _TECH_TOKENS:
        return False
    return _letters(token) >= _MIN_LETTERS and _digits(token) >= _MIN_DIGITS


def _digit_tail(token: str) -> str:
    """토큰 끝의 숫자 꼬리(`OMEGA77` → `77`). 끝이 숫자가 아니면 빈 문자열."""
    match = re.search(r"\d+$", token)
    return match.group(0) if match else ""


def _edit_distance_at_most(a: str, b: str, limit: int) -> bool:
    """편집 거리가 limit 이하인지 판정한다(그 이상은 값을 끝까지 계산하지 않는다).

    식별자는 짧아(보통 20자 이하) 단순 DP 로 충분하다.
    [[number_citation]] 의 같은 이름 함수와 동일한 알고리즘이지만, 그쪽은 숫자
    문자열 전용이고 이쪽은 임의 토큰을 받는다. 한 곳으로 합치면 어느 한쪽의
    입력 가정(숫자만/토큰)이 다른 쪽에 새므로 각자 둔다.
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
        if min(cur) > limit:
            return False
        prev = cur
    return prev[-1] <= limit


def find_misquoted_literals(answer: str, sources: list[str]) -> list[MisquotedLiteral]:
    """도구 결과의 리터럴 중 답변에서 어긋나게 옮겨진 것을 찾는다.

    매개변수:
      answer  — 모델이 낸 최종 답변 텍스트.
      sources — 근거 자료(이 턴의 도구 결과 본문들).

    반환:
      경고 대상 목록. 도구를 안 쓴 턴이면 대조할 원문이 없으므로 항상 빈 목록이다.
    """
    if not answer or not sources:
        return []

    source_text = "\n".join(s for s in sources if s)
    if not source_text:
        return []

    # ★비교는 반드시 **토큰 단위**로 한다. 부분문자열로 보면 `OMEGA77` 이
    # `OMEGA777` 안에 들어 있어 "답변에 이미 있다"로 오판하고, 정작 잡아야 할
    # 자릿수 증식을 통째로 놓친다(테스트로 고정).
    source_token_list = _ASCII_TOKEN.findall(source_text)
    source_tokens = set(source_token_list)
    source_hangul = set(_HANGUL_TAIL_TOKEN.findall(source_text))

    # 도구 결과 쪽 리터럴 — 원문 등장 순서를 지킨다(경고 순서가 실행마다 흔들리면
    # 같은 입력에 다른 출력이 나와 재현이 어렵다). set 으로 만들지 않는 이유다.
    source_literals = list(dict.fromkeys(t for t in source_token_list if _is_notable(t)))
    if not source_literals:
        return []

    # 답변 쪽 후보들. 도구 결과에 그대로 있는 토큰은 정상이므로 후보에서 뺀다.
    answer_ascii = [
        t
        for t in dict.fromkeys(_ASCII_TOKEN.findall(answer))
        if _is_notable(t) and t not in source_tokens
    ]
    answer_hangul = [
        t for t in dict.fromkeys(_HANGUL_TAIL_TOKEN.findall(answer)) if t not in source_hangul
    ]

    answer_tokens_lower = {t.lower() for t in _ASCII_TOKEN.findall(answer)}
    found: list[MisquotedLiteral] = []
    for literal in source_literals:
        # 대소문자만 다른 것은 경고하지 않는다 — 노이즈에 비해 얻는 게 없다.
        if literal.lower() in answer_tokens_lower:
            continue

        # 규칙 A) 같은 문자 계열의 근접값 — 자릿수 증식·글자 하나 어긋남.
        near = next(
            (
                t
                for t in answer_ascii
                if t != literal and _edit_distance_at_most(literal, t, _NEAR_MISS_EDITS)
            ),
            None,
        )
        if near is not None:
            found.append(MisquotedLiteral(source=literal, answer=near, kind="near_miss"))
            continue

        # 규칙 B) 한글 음차 — 숫자 꼬리가 같은 한글 토큰.
        tail = _digit_tail(literal)
        if not tail:
            continue
        translit = next((t for t in answer_hangul if _digit_tail(t) == tail), None)
        if translit is not None:
            found.append(
                MisquotedLiteral(source=literal, answer=translit, kind="transliteration")
            )

    return found


def build_literal_warning(items: list[MisquotedLiteral]) -> str:
    """경고 문구를 만든다. 붙일 것이 없으면 빈 문자열.

    본문을 고치지 않고 끝에 덧붙이므로, 구분선을 넣고 **원문 리터럴을 그대로** 적는다.
    """
    if not items:
        return ""

    lines = [
        "",
        "---",
        "⚠️ **표기 확인 필요** — 아래는 도구 결과 원문과 다르게 적혔습니다.",
    ]
    for item in items[:_MAX_WARNINGS]:
        lines.append(f"- 답변의 `{item.answer}` → 도구 결과 원문: `{item.source}`")
    if len(items) > _MAX_WARNINGS:
        lines.append(f"- 그 외 {len(items) - _MAX_WARNINGS}건")
    lines.append("도구 결과 원문을 기준으로 확인해 주세요.")
    return "\n".join(lines)
