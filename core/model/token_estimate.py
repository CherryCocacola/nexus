# 토크나이저를 못 부를 때 쓰는 보수적 입력 토큰 추정기(문자 종류별 가중합).
"""
입력 토큰 추정 — **정확히 세는 경로가 실패했을 때만** 쓰는 폴백이다.

■ 왜 이 모듈이 따로 있나 (2026-08-14)
  종전 추정식은 `총글자수 // 3`(0.3333 토큰/글자) 하나였다. A.X-4.0 토크나이저로
  실측해 보니 문자 종류에 따라 **100배 넘게** 차이가 난다.

    문자 종류        토큰/글자   글자/토큰
    한글(공백없음)     0.7778      1.29
    한글(문서체)       0.3963      2.52
    한글(대화체)       0.2286      4.37
    영문 소문자        0.2000      5.00
    코드               0.3026      3.30
    숫자               1.0000      1.00   ← 숫자는 한 글자가 한 토큰
    한자               1.3333      0.75   ← 한 글자가 한 토큰보다 비싸다
    기호               0.6843      1.46
    공백               0.0078    127.66
    개행               0.0625     16.00

  단일 계수(1/3)로는 어느 쪽도 못 맞춘다. 실제로 문서체 한국어에서 15% 과소추정해
  컨텍스트 예산을 보수적으로 잡을 수밖에 없었다.

■ 그런데 이것도 근본 해법이 아니다
  같은 한글인데도 0.23 ~ 0.78 로 3배 차이가 난다. BPE 병합은 문자 종류가 아니라
  **실제 낱말**에 달려 있어서, 문자만 세는 방식으로는 원리상 못 맞춘다.
  그래서 이 모듈은 폴백이고, 정답은 vLLM `/tokenize` 로 **세는** 것이다
  (`LocalModelProvider.count_prompt_tokens`). 여기서는 **과소추정만 안 하면 된다.**

■ 그래서 일부러 넉넉하게 잡는다
  과소추정하면 컨텍스트를 넘겨 vLLM 이 요청을 거부한다(긴급 압축 재시도 → 지연 + 내용 손실).
  과대추정하면 필요보다 일찍 자른다(답변 품질만 조금 손해). 후자가 낫다.
  가중치는 위 실측의 **나쁜 쪽**에 맞춰 두었고, 테스트가 "어떤 표본에서도 실제보다
  적게 추정하지 않는다"를 고정한다.

작성자: 이현수 / 작성일: 2026-08-14
"""

from __future__ import annotations

# ── 문자 종류별 토큰 가중치 (실측의 나쁜 쪽) ───────────────────────────
# 한글은 0.2286(대화체) ~ 0.7778(공백없음) 로 3배 차이가 난다. 최악(공백없음)을
# 덮어야 과소추정이 안 나므로 상단에 맞춘다. 대신 대화체는 2.75배 부풀지만,
# 이 경로는 폴백이라 부풀어도 손해가 작다(위 docstring 참고).
_W_HANGUL = 0.80
# 한자는 한 글자가 한 토큰보다 비싸다(실측 1.3333). 여유를 조금 얹는다.
_W_CJK = 1.40
# 숫자는 한 글자가 한 토큰(실측 1.0). 숫자 옆 공백까지 토큰을 먹어(실측 11토큰/
# "1234567890 ") 1.0 으로는 모자란다.
_W_DIGIT = 1.15
# 영문 낱말은 0.20 이지만 코드(0.3026, 기호·들여쓰기 포함)를 덮어야 해서 더 높인다.
_W_ASCII_ALPHA = 0.36
# 기호·구두점. 실측 0.6843 + 여유.
_W_SYMBOL = 0.75
# 공백 자체는 거의 공짜지만(0.0078) 코드 들여쓰기·숫자 사이에서는 토큰을 먹는다.
_W_SPACE = 0.05
# 개행 실측 0.0625 + 여유.
_W_NEWLINE = 0.10
# 그 밖(이모지·기타 문자). 모르면 가장 비싼 쪽(한자)으로 친다.
_W_OTHER = 1.40

# 채팅 템플릿이 메시지마다 붙이는 역할 표시·특수 토큰의 몫.
# 실측: 사용자 메시지 1개짜리 요청에서 본문 2토큰 → 템플릿 적용 7토큰(차이 5).
_TEMPLATE_TOKENS_PER_MESSAGE = 8


def _is_hangul(ch: str) -> bool:
    """완성형 한글 음절 + 자모."""
    code = ord(ch)
    return 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F


def _is_cjk(ch: str) -> bool:
    """한자(CJK 통합 한자) + 일본어 가나."""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK 통합 한자
        or 0x3400 <= code <= 0x4DBF  # 확장 A
        or 0x3040 <= code <= 0x30FF  # 히라가나·가타카나
    )


def estimate_tokens(text: str) -> int:
    """문자 종류별 가중합으로 토큰 수를 **넉넉하게** 추정한다.

    반환값은 올림한 정수다. 빈 문자열이면 0.
    """
    if not text:
        return 0

    total = 0.0
    for ch in text:
        if ch == "\n":
            total += _W_NEWLINE
        elif ch.isspace():
            total += _W_SPACE
        elif ch.isdigit():
            total += _W_DIGIT
        elif _is_hangul(ch):
            total += _W_HANGUL
        elif _is_cjk(ch):
            total += _W_CJK
        elif ch.isascii():
            # 영문자·기호 구분. isalpha() 로 갈라 기호를 비싸게 친다.
            total += _W_ASCII_ALPHA if ch.isalpha() else _W_SYMBOL
        else:
            total += _W_OTHER

    return int(total) + 1


def estimate_prompt_tokens(
    message_texts: list[str],
    system_prompt: str = "",
    tool_schema_texts: list[str] | None = None,
) -> int:
    """프롬프트 전체(메시지 + 시스템 프롬프트 + 도구 스키마)를 추정한다.

    메시지 개수만큼 채팅 템플릿 오버헤드를 더한다 — 메시지가 많으면 무시할 수 없다
    (도구 루프는 한 턴에 10개 넘는 메시지를 보낸다).
    """
    total = estimate_tokens(system_prompt)
    for text in message_texts:
        total += estimate_tokens(text) + _TEMPLATE_TOKENS_PER_MESSAGE
    for schema in tool_schema_texts or []:
        total += estimate_tokens(schema)
    return total
