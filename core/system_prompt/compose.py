# 시스템 프롬프트 조립 — base에서 매번 "재조립"하는 단일 진입점.
"""
시스템 프롬프트 단일 조립기 (2026-08-05).

[왜 필요한가 — 부채 청산]
  지금까지 웹은 세 지점에서 각자 `engine.system_prompt + "\\n\\n[사용자 지시]\\n" + x`
  처럼 **문자열을 덧붙여** 왔다. 이 방식은 두 가지 문제가 있다.
    1) 누적 위험: 같은 엔진에 두 번 덧붙이면 지시가 중복된다. 지금은 "요청마다
       엔진을 새로 조립한다"는 전제 덕분에 우연히 안전하지만, 그 전제가 깨지는
       순간(엔진 재사용·재시도) 조용히 중복된다.
    2) 순서·형식이 호출 지점마다 달라진다. 응답 스타일(W1)처럼 섹션이 하나 더
       늘면 세 곳을 모두 고쳐야 하고, 한 곳만 빠뜨려도 표면 간 동작이 어긋난다.

  그래서 "덧붙이기"를 금지하고, **항상 base에서 전체를 다시 만든다**(멱등).
  같은 입력이면 몇 번을 호출해도 결과가 동일하다.

[섹션 순서 — prompt cache를 고려한 배치]
    base → [응답 스타일] → [사용자 지시] → [프로젝트 지시] → [세션 지시]
  base를 맨 앞에 고정하는 이유: vLLM prefix cache는 앞부분이 같아야 적중한다.
  자주 바뀌는 값(스타일·지시)을 앞에 두면 캐시가 매번 깨진다.

작성자: 이현수 / 작성일: 2026-08-05
"""

from __future__ import annotations

# 섹션 제목 — 조립 결과를 파싱하거나 로그로 확인할 때 기준이 되므로 상수로 고정한다.
STYLE_HEADER = "[응답 스타일]"
USER_HEADER = "[사용자 지시]"
PROJECT_HEADER = "[프로젝트 지시]"
SESSION_HEADER = "[세션 지시]"


def compose_system_prompt(
    base: str,
    *,
    style: str | None = None,
    user_instruction: str | None = None,
    project_instruction: str | None = None,
    session_instruction: str | None = None,
) -> str:
    """base 프롬프트에 지시 섹션들을 붙여 최종 시스템 프롬프트를 만든다.

    [멱등 보장] 이 함수는 base만 읽고 전체를 새로 만든다. 이전 호출 결과를
    입력으로 다시 넣지 않는 한(=호출부가 base를 보관하는 한) 중복이 생기지 않는다.

    Args:
        base: 기본 시스템 프롬프트(도구 목록·행동 규약 등). 이 값이 항상 맨 앞에
            온다 — prefix cache 적중을 위해서다.
        style: 응답 스타일 지시(W1). 예) "간결하게, 불릿 위주로 답한다."
        user_instruction: 사용자(테넌트) 커스텀 인스트럭션.
        project_instruction: 프로젝트 전용 인스트럭션.
        session_instruction: 이번 요청에만 적용할 지시(OpenAI system 메시지 등).

    Returns:
        조립된 시스템 프롬프트. 빈 섹션은 아예 포함하지 않는다(공백만 있는 값도
        무시). 모든 섹션이 비면 base를 그대로 돌려준다.
    """
    parts: list[str] = [base.rstrip()] if base and base.strip() else []
    for header, body in (
        (STYLE_HEADER, style),
        (USER_HEADER, user_instruction),
        (PROJECT_HEADER, project_instruction),
        (SESSION_HEADER, session_instruction),
    ):
        if body and body.strip():
            parts.append(f"{header}\n{body.strip()}")
    return "\n\n".join(parts)
