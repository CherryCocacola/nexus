"""
종료 판단기(StopResolver) — 쿼리 루프의 "이번 턴에서 멈출까, 계속할까"를 결정한다.

이 파일은 무엇을 하나:
  에이전트가 한 턴(모델 1회 응답)을 끝낼 때마다, 그 결과를 보고
  "다음 턴으로 넘어가야 하는지" 또는 "여기서 대화를 마무리해야 하는지"를
  판정한다. 판정에 필요한 재료는 세 가지다.
    - 이번 턴에 모델이 도구(tool)를 호출했는가?
    - 모델이 알려준 종료 이유(stop_reason)는 무엇인가?
    - 응답 텍스트가 중간에 잘린 것처럼 보이는가?

주요 구성:
  - StopResolver           : 판단 로직을 담은 클래스 (아래 3개 메서드 제공)
      · should_continue()      → 도구 호출 유무로 계속/후보 판정
      · resolve_stop_reason()  → 종료 이유를 사람이 읽을 문자열로 변환
      · is_truncated()         → 응답이 잘렸는지(max_tokens 복구용) 판정
  - _seems_truncated()     : 텍스트가 잘렸는지 휴리스틱으로 검사하는 헬퍼

어디서 호출되나 / 의존 관계:
  Claude Code의 query.ts에서 종료 판단 로직만 떼어내 옮긴 모듈이다.
  query_loop(Ch.5.8)이 매 턴 끝(Phase 3)에서 이 클래스를 호출한다.
    1. 도구 호출이 있으면 → 다음 턴 계속
    2. 도구 호출이 없으면 → 종료 후보 → 세부 판단(잘림/복구/hook 등)
  종료 이유 상수는 core.message.StopReason(enum)에 의존한다.

왜 별도 모듈로 분리했나:
  query_loop은 이미 400줄이 넘는다. 종료 판단 로직을 이 파일로 떼어내면
  단위 테스트가 쉬워지고, 앞으로 Hook 기반 종료 제어(Transition 5)를
  query_loop 본체를 건드리지 않고 독립적으로 확장할 수 있다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.message import StopReason

if TYPE_CHECKING:
    # 순환 import 방지 — LoopState는 query_loop에 정의돼 있고, 그 query_loop이
    # 다시 이 모듈을 쓰기 때문에 런타임에 서로 import하면 순환이 생긴다.
    # 타입 힌트용으로만 필요하므로 TYPE_CHECKING 블록 안에서만 참조한다.
    from core.orchestrator.query_loop import LoopState

logger = logging.getLogger("nexus.orchestrator.stop_resolver")


class StopResolver:
    """
    쿼리 루프의 종료/계속 판단을 담당하는 클래스.

    query_loop의 Phase 3(턴 마무리 단계) 끝에서 호출된다.
    도구 호출 유무, 모델이 준 종료 이유, 응답 잘림 여부 등을 종합해
    "루프를 한 턴 더 돌릴지" 또는 "여기서 멈출지"를 결정한다.

    상태(state)를 내부에 들고 있지 않은 순수 판단기라서,
    인스턴스 하나를 만들어 계속 재사용해도 안전하다.
    """

    def should_continue(
        self,
        state: LoopState,
        tool_use_blocks: list[dict],
    ) -> bool:
        """
        이번 턴 결과를 보고 "다음 턴을 계속해야 하는지" 1차 판정한다.

        규칙은 단순하다. 모델이 도구를 호출했다면 그 도구의 실행 결과를
        모델에게 다시 돌려줘야 하므로 반드시 다음 턴이 필요하다.
        도구 호출이 없으면 대화가 끝났을 "후보"이므로 False를 돌려주고,
        최종 종료 여부는 query_loop이 추가 판단(잘림 복구·hook 등)으로 정한다.

        Args:
            state: 현재 루프 상태 (턴 수, 에러 카운터 등). 지금 규칙에서는
                직접 쓰지 않지만, 향후 턴 상한 등 조건 확장을 위해 받아 둔다.
            tool_use_blocks: 이번 턴에서 모델이 호출한 도구 목록.

        Returns:
            True면 "다음 턴 계속", False면 "종료 후보".
        """
        # 도구 호출이 하나라도 있으면 실행 결과를 모델에 되돌려야 하므로 무조건 계속.
        if tool_use_blocks:
            return True

        # 도구 호출이 없으면 종료 후보다. 여기서 곧장 끝내지 않고 False만 반환해,
        # query_loop이 추가 종료 판단(max_tokens 복구, hook 등)을 이어서 수행한다.
        return False

    def resolve_stop_reason(
        self,
        stop_reason: StopReason | None,
        text: str,
    ) -> str:
        """
        모델이 준 종료 이유(enum)를 사람이 읽기 좋은 문자열로 해석한다.

        로그나 상위 UI에서 "왜 턴이 끝났는지"를 표시할 때 쓴다.
        특히 max_tokens로 끝난 경우는 응답이 정상 완료된 것인지,
        아니면 길이 한도에 걸려 중간에 잘린 것인지까지 구분해 준다.

        Args:
            stop_reason: 모델이 반환한 종료 이유(StopReason enum). None이면
                이유를 알 수 없는 상황이다.
            text: assistant 응답 텍스트. 잘림 여부 판단에 함께 쓰인다.

        Returns:
            종료 이유를 설명하는 문자열
            (예: "completed", "tool_use", "truncated", "max_tokens_reached").
        """
        # 종료 이유가 아예 없으면 판단 불가 — "unknown"으로 표시한다.
        if stop_reason is None:
            return "unknown"

        # StopReason이 enum이면 .value를, 이미 문자열이면 그대로 문자열화해서 비교한다.
        # (상위에서 어떤 형태로 넘겨줘도 안전하게 다루기 위한 방어 코드)
        reason_value = stop_reason.value if hasattr(stop_reason, "value") else str(stop_reason)

        # 모델이 스스로 턴을 정상적으로 마쳤다 → 대화 완료.
        if reason_value == StopReason.END_TURN.value:
            return "completed"

        # 도구를 호출하려고 멈춘 것 → 도구 실행 후 다음 턴으로 이어진다.
        if reason_value == StopReason.TOOL_USE.value:
            return "tool_use"

        if reason_value == StopReason.MAX_TOKENS.value:
            # 길이 한도에 걸려 멈췄다. 이 경우는 두 갈래로 나뉜다.
            # 텍스트가 잘린 것처럼 보이면 "truncated", 아니면 "max_tokens_reached".
            if _seems_truncated(text):
                return "truncated"
            return "max_tokens_reached"

        # 지정한 정지 문자열(stop sequence)에 도달해 멈춘 경우.
        if reason_value == StopReason.STOP_SEQUENCE.value:
            return "stop_sequence"

        # 위 어디에도 해당하지 않는 값 — 원본 값을 붙여 디버깅에 도움을 준다.
        return f"unknown({reason_value})"

    def is_truncated(self, stop_reason: StopReason | None, text: str) -> bool:
        """
        응답이 길이 한도에 걸려 "중간에 잘렸는지"를 판정한다.

        조건은 두 가지를 모두 만족해야 한다.
          (1) 종료 이유가 max_tokens 이고,
          (2) 텍스트가 잘린 것처럼 보인다(_seems_truncated 휴리스틱).
        query_loop의 Transition 3/4(max_output 복구)에서 이 결과를 보고,
        잘린 응답을 이어서 받아오는 복구 로직을 돌릴지 결정한다.

        Args:
            stop_reason: 모델이 반환한 종료 이유. None이면 잘림 아님으로 본다.
            text: assistant 응답 텍스트.

        Returns:
            응답이 잘렸으면 True, 아니면 False.
        """
        # 종료 이유가 없으면 잘림 판단의 전제가 성립하지 않으므로 False.
        if stop_reason is None:
            return False

        # enum이면 .value, 문자열이면 그대로 — resolve_stop_reason과 동일한 방어 처리.
        reason_value = stop_reason.value if hasattr(stop_reason, "value") else str(stop_reason)

        # max_tokens로 끝난 게 아니라면(정상 종료 등) 애초에 잘림 대상이 아니다.
        if reason_value != StopReason.MAX_TOKENS.value:
            return False

        # max_tokens로 끝났다면, 텍스트 모양을 보고 실제 잘림 여부를 최종 판단한다.
        return _seems_truncated(text)


def _seems_truncated(text: str) -> bool:
    """
    응답 텍스트가 중간에 잘린 것처럼 보이는지 휴리스틱으로 검사한다.

    Claude Code의 truncation detection 로직에 대응한다. 모델이 정확히
    "잘렸다"고 알려주진 않으므로, 텍스트의 마지막 모양을 보고 추정한다.
    코드 블록이 열린 채 끝났거나, 긴 문장이 마침표 없이 끊겼으면 잘린 것으로 본다.

    판단 순서(위에서부터 차례로 검사):
      1. 텍스트가 없거나 너무 짧으면(50자 미만) → 잘리지 않음으로 간주.
      2. 문장 종료 문자(. ! ? 。 ！ ？ 개행 등)로 끝나면 → 정상 종료.
      3. 코드 블록 표시(```)가 홀수 개면 → 열린 채로 끊긴 것 → 잘림.
      4. 200자를 넘는 긴 텍스트가 문장 종료 없이 끝나면 → 잘림.
      5. 위 어디에도 안 걸리면 → 잘리지 않음.

    Args:
        text: 검사할 assistant 응답 텍스트.

    Returns:
        잘린 것으로 보이면 True, 아니면 False.
    """
    # 빈 텍스트나 너무 짧은 텍스트는 잘렸다고 보기 어렵다 → 잘림 아님으로 간주.
    if not text or len(text) < 50:
        return False

    # 뒤쪽 공백/개행을 떼고 마지막 "의미 있는" 글자를 확인한다.
    stripped = text.rstrip()
    if not stripped:
        return False

    last_char = stripped[-1]

    # 마침표·물음표·느낌표(한글 문장부호 포함)나 개행/코드펜스로 끝나면 정상 종료로 판단.
    if last_char in ".!?。！？\n```":
        return False

    # 코드 블록 여닫이(```)의 개수가 홀수면 블록이 닫히지 않은 채 끝난 것 → 잘림.
    if text.count("```") % 2 == 1:
        return True

    # 200자가 넘는 긴 텍스트인데 문장 종료 문자 없이 끝났다면 문장이 끊긴 것 → 잘림.
    if len(text) > 200 and last_char not in ".!?。！？\n":
        return True

    # 위 어느 조건에도 걸리지 않으면 정상 종료로 본다.
    return False
