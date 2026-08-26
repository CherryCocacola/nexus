# 답변이 완성된 뒤 붙이는 사후 검증들을 한자리에서 실행하는 진입점.
"""
사후 검증 통합 — 답변 하나에 걸어야 할 대조를 한 번에 수행한다.

■ 왜 이 모듈이 필요한가 (2026-08-07)
  숫자 인용 검증과 실행 주장 검증을 만들었는데 둘 다 `web/app.py` 안에만 배선돼
  있었다. 그런데 **정작 문제가 관측된 곳은 CLI 였다.**

    · 테스트 기대값 암산 오류(5415 → 5115)      — CLI
    · "예상 출력 결과: 모든 테스트가 통과했습니다" — CLI
    · Write 로 저장한 파일 손상                  — CLI

  보호가 필요한 표면에 보호가 없던 셈이다. CLI 가 `web/` 을 import 하는 것은 의존성
  방향 위반이므로(cli·web → core), 공용 로직을 core 로 내리고 양쪽이 같은 것을 쓴다.

■ 왜 하나로 묶었나
  호출부마다 "숫자 검증 + 실행 검증"을 각각 부르면, 새 검증기를 추가할 때 모든
  호출부(웹 4경로 + CLI 2경로)를 다시 손대야 한다. 실제로 실행 주장 검증을 넣을 때
  웹 4곳을 전부 고쳤다. 진입점을 하나로 두면 다음부터는 여기만 고치면 된다.

■ 무엇을 하지 않나
  답변을 고치지 않는다. 어느 값이 맞는지 단정할 수 없는 경우가 있고, 조용한 자동
  교정은 틀렸을 때 발견조차 되지 않는다. 사실만 덧붙여 사람이 판단하게 한다.

작성자: 이현수 / 작성일: 2026-08-07
"""

from __future__ import annotations

import logging

logger = logging.getLogger("nexus.verification")


def collect_tool_result_texts(messages: list) -> list[str]:
    """이번 턴의 도구 결과 본문만 모은다(검증기들이 쓸 '근거 자료').

    도구를 쓰지 않은 턴이면 빈 목록이 된다. 그 사실 자체가 두 검증기에서 서로 다르게
    쓰인다 — 숫자 검증은 대조할 원문이 없으니 아무 것도 하지 않고, 실행 주장 검증은
    "실행했다면서 도구 기록이 없다"는 신호로 삼는다.
    """
    out: list[str] = []
    for msg in messages:
        role = msg.role if isinstance(msg.role, str) else msg.role.value
        if role == "tool_result" and isinstance(msg.content, str):
            out.append(msg.content)
    return out


def build_answer_warnings(answer: str, messages: list) -> str:
    """답변에 덧붙일 경고 문구를 만든다. 붙일 것이 없으면 빈 문자열.

    Args:
        answer:   모델이 낸 최종 답변 텍스트.
        messages: 이번 턴 구간의 메시지들(engine._messages 의 해당 슬라이스).

    검증이 실패해도 답변은 그대로 나가야 하므로 통째로 fail-soft 로 감싼다.
    진단 장치가 응답을 막으면 본말이 전도된다.
    """
    try:
        from core.verification.execution_claim import (
            build_execution_warning,
            find_execution_claims,
        )
        from core.verification.file_claim import (
            build_file_claim_warning,
            collect_successful_write_tools,
            find_file_claims,
        )
        from core.verification.literal_citation import (
            build_literal_warning,
            find_misquoted_literals,
        )
        from core.verification.number_citation import (
            build_number_warning,
            find_uncited_numbers,
        )

        sources = collect_tool_result_texts(messages)
        warning = build_number_warning(find_uncited_numbers(answer, sources))
        # 리터럴(식별자·코드) 대조 — 숫자 검증과 대상이 겹치지 않는다. 숫자 쪽은
        # 자릿수 구분 쉼표가 있는 값만, 이쪽은 글자+숫자가 섞인 식별자만 본다
        # (2026-08-13, `OMEGA77` → `오메가77` 실측).
        warning += build_literal_warning(find_misquoted_literals(answer, sources))
        warning += build_execution_warning(find_execution_claims(answer), len(sources))
        # 파일 작성 주장 대조(2026-08-23 추가) — 위 실행 주장 검증과 **별개**다.
        #   실행 주장은 도구 결과 '개수'만 보므로, Read 만 하고 "파일에 작성했습니다"
        #   라고 답한 실측 사고를 못 잡았다(개수가 0 이 아니라 억제됨).
        #   이쪽은 **성공한 쓰기 도구**만 세어 그 구멍을 막는다. 기존 검증기의
        #   시그니처·동작은 건드리지 않는다(무회귀).
        warning += build_file_claim_warning(
            find_file_claims(answer), collect_successful_write_tools(messages)
        )
        return warning
    except Exception as e:  # noqa: BLE001 — 검증 실패가 응답을 막지 않게 한다
        logger.warning("사후 검증 실패(무시): %s", e)
        return ""


def build_structured_warnings(answer: str, messages: list) -> list[str]:
    """응답의 `warnings` 배열에 실을 검증 결과를 만든다(답변 본문은 건드리지 않는다).

    ■ 왜 본문이 아니라 배열인가 (2026-08-25)
      위 `build_answer_warnings` 는 경고를 답변 **본문 뒤에 덧붙인다.** 사람이 읽는
      화면에는 그게 맞지만, 클라이언트가 프로그램으로 판정해야 하는 신호에는 맞지
      않는다. 본문에 섞이면 클라이언트는 문자열을 뒤져야 하고, 사용자 화면도
      더러워진다.

      플러그인 팀이 요청한 것도 정확히 이것이다 — "HTTP 200, finish_reason=stop,
      warnings 빈 배열이라 정상 응답과 구분할 수 없다. warnings 에 한 줄만 실어
      주셔도 충분하다."

    ■ 지금 싣는 것
      적용 주장 대조 하나뿐이다. 클라이언트가 도구를 실행하는 표면에서 모델이
      읽기 도구의 성공을 쓰기 성공으로 오독해 "적용되었습니다"라고 답하는 실측
      사고를 잡는다(2026-08-25, 1,760 세션 중 43건).

    Args:
        answer:   모델이 낸 최종 답변 텍스트.
        messages: 이번 요청 구간의 메시지들.

    Returns:
        경고 문자열 목록. 없으면 빈 목록. 검증 실패도 빈 목록(fail-soft).
    """
    try:
        from core.verification.apply_claim import (
            build_apply_claim_warning,
            collect_client_tool_results,
            find_apply_claims,
            is_change_proposal,
        )

        # 변경안 제출(final_proposal)은 완료 선언이 아니다. 클라이언트가 사용자
        # 승인 뒤 로컬에서 적용하므로 정상 흐름에서도 서버는 쓰기 성공을 못 본다.
        # 이 분기가 없으면 정상 제안이 전부 오탐이 된다(실측 222건 중 153건).
        if is_change_proposal(answer):
            return []

        warning = build_apply_claim_warning(
            find_apply_claims(answer), collect_client_tool_results(messages)
        )
        return [warning] if warning else []
    except Exception as e:  # noqa: BLE001 — 진단이 응답을 막지 않게 한다
        logger.warning("구조화 경고 생성 실패(무시): %s", e)
        return []
