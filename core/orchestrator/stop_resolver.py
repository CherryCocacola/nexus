"""
종료 판단기 — 쿼리 루프의 종료/계속 결정을 담당한다.

Claude Code의 query.ts에서 종료 판단 로직만 추출한 모듈이다.
query_loop(Ch.5.8)에서 매 턴 끝에 호출하여:
  1. 도구 호출이 있으면 → 다음 턴 계속
  2. 도구 호출이 없으면 → 종료 후보 → 세부 판단

왜 별도 모듈인가:
  query_loop은 이미 400줄 이상이다. 종료 판단 로직을 분리하면
  테스트가 쉬워지고, 향후 Hook 기반 종료 제어(Transition 5)를
  독립적으로 확장할 수 있다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.message import StopReason

if TYPE_CHECKING:
    # 순환 import 방지 — 타입 검사 시에만 LoopState를 참조한다
    from core.orchestrator.query_loop import LoopState

logger = logging.getLogger("nexus.orchestrator.stop_resolver")


class StopResolver:
    """
    쿼리 루프의 종료/계속 판단을 담당한다.

    query_loop의 Phase 3 끝에서 호출된다.
    도구 호출 유무, 종료 이유, 잘림 여부 등을 종합하여
    루프를 계속할지 종료할지 결정한다.
    """

    def should_continue(
        self,
        state: LoopState,
        tool_use_blocks: list[dict],
    ) -> bool:
        """
        도구 호출이 있으면 다음 턴을 계속해야 하는지 판단한다.

        Args:
            state: 현재 루프 상태 (턴 수, 에러 카운터 등)
            tool_use_blocks: 이번 턴에서 모델이 호출한 도구 목록

        Returns:
            True면 다음 턴 계속, False면 종료 후보
        """
        # 도구 호출이 있으면 항상 계속한다
        if tool_use_blocks:
            return True

        # 도구 호출이 없으면 종료 후보 — False를 반환하여
        # query_loop이 추가 종료 판단(max_tokens 복구, hook 등)을 수행한다
        return False

    def resolve_stop_reason(
        self,
        stop_reason: StopReason | None,
        text: str,
    ) -> str:
        """
        종료 이유를 사람이 읽을 수 있는 문자열로 해석한다.

        Args:
            stop_reason: 모델이 반환한 종료 이유 (StopReason enum)
            text: assistant 응답 텍스트

        Returns:
            종료 이유를 설명하는 문자열
        """
        if stop_reason is None:
            return "unknown"

        # StopReason 값을 문자열로 변환 (enum 또는 이미 문자열일 수 있음)
        reason_value = stop_reason.value if hasattr(stop_reason, "value") else str(stop_reason)

        if reason_value == StopReason.END_TURN.value:
            return "completed"

        if reason_value == StopReason.TOOL_USE.value:
            return "tool_use"

        if reason_value == StopReason.MAX_TOKENS.value:
            # 응답이 잘린 것인지 판단
            if _seems_truncated(text):
                return "truncated"
            return "max_tokens_reached"

        if reason_value == StopReason.STOP_SEQUENCE.value:
            return "stop_sequence"

        return f"unknown({reason_value})"

    def is_truncated(self, stop_reason: StopReason | None, text: str) -> bool:
        """
        응답이 중간에 잘렸는지 판단한다.

        max_tokens로 종료되었고, 텍스트가 잘린 것처럼 보이면 True.
        query_loop의 Transition 3/4 (max_output 복구)에서 사용한다.
        """
        if stop_reason is None:
            return False

        reason_value = stop_reason.value if hasattr(stop_reason, "value") else str(stop_reason)

        if reason_value != StopReason.MAX_TOKENS.value:
            return False

        return _seems_truncated(text)


def _seems_truncated(text: str) -> bool:
    """
    응답이 중간에 잘린 것처럼 보이는지 휴리스틱으로 검사한다.

    Claude Code의 truncation detection 로직에 대응한다.
    코드 블록이 열려있거나, 문장이 마무리되지 않았으면 잘린 것으로 판단한다.

    판단 기준:
      1. 텍스트가 너무 짧으면 → 잘리지 않음
      2. 문장 종료 문자(., !, ?, 등)로 끝나면 → 잘리지 않음
      3. 코드 블록(```)이 홀수 개면 → 잘림 (열린 채 종료)
      4. 200자 넘는 텍스트가 문장 종료 없이 끝나면 → 잘림
    """
    # 빈 텍스트나 너무 짧은 텍스트는 잘리지 않은 것으로 간주
    if not text or len(text) < 50:
        return False

    stripped = text.rstrip()
    if not stripped:
        return False

    last_char = stripped[-1]

    # 문장 종료 문자로 끝나면 정상 종료로 판단
    if last_char in ".!?。！？\n```":
        return False

    # 코드 블록이 홀수 개면 열린 채로 잘린 것
    if text.count("```") % 2 == 1:
        return True

    # 200자 넘는 긴 텍스트가 문장 종료 없이 끝나면 잘린 것
    if len(text) > 200 and last_char not in ".!?。！？\n":
        return True

    return False
