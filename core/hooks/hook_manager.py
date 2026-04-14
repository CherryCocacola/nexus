"""
훅 매니저 — HookManager.

Ch.10.4 사양서 기반. 도구 실행 전후에 훅을 실행하여
추가 검증, 로깅, 변환 등을 수행한다.

훅 실행 규칙:
  - 등록된 순서대로 실행한다 (순차)
  - BLOCK이 나오면 즉시 중단하고 나머지 훅을 건너뛴다
  - APPROVE가 나오면 즉시 승인하고 나머지 훅을 건너뛴다
  - CONTINUE는 다음 훅으로 넘긴다
  - 모든 훅이 CONTINUE면 최종 결과도 CONTINUE

핵심 원칙: 훅 실행 중 에러가 발생하면 안전하게 CONTINUE를 반환한다.
(훅 에러 때문에 도구 실행이 차단되면 안 된다 — 훅은 부가 기능)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("nexus.hooks")


# ─────────────────────────────────────────────
# 훅 이벤트 타입
# ─────────────────────────────────────────────
class HookEvent(str, Enum):
    """훅이 실행되는 시점."""

    PRE_TOOL_USE = "pre_tool_use"  # 도구 실행 전
    POST_TOOL_USE = "post_tool_use"  # 도구 실행 후
    STOP = "stop"  # 쿼리 루프 중단 시
    NOTIFICATION = "notification"  # 알림 이벤트


# ─────────────────────────────────────────────
# 훅 결정
# ─────────────────────────────────────────────
class HookDecision(str, Enum):
    """훅의 결정 결과."""

    APPROVE = "approve"  # 즉시 승인 (나머지 훅 건너뜀)
    BLOCK = "block"  # 즉시 차단 (나머지 훅 건너뜀)
    CONTINUE = "continue"  # 다음 훅으로 넘김 (기본값)


# ─────────────────────────────────────────────
# 훅 입력 — 훅 핸들러에 전달되는 데이터
# ─────────────────────────────────────────────
class HookInput(BaseModel):
    """
    훅 핸들러에 전달되는 입력 데이터.
    이벤트 타입에 따라 다른 필드가 채워진다.
    """

    event: HookEvent  # 어떤 이벤트인지
    tool_name: str | None = None  # 실행하려는 도구 이름
    tool_input: dict | None = None  # 도구 입력 데이터
    tool_result: str | None = None  # 도구 실행 결과 (POST_TOOL_USE에서)
    metadata: dict[str, Any] = Field(default_factory=dict)  # 추가 메타데이터


# ─────────────────────────────────────────────
# 훅 결과 — 훅 핸들러가 반환하는 데이터
# ─────────────────────────────────────────────
class HookResult(BaseModel):
    """
    훅 핸들러의 반환 결과.
    decision이 CONTINUE면 다음 훅으로 넘어간다.
    """

    decision: HookDecision = HookDecision.CONTINUE  # 기본: 다음으로 넘김
    block_reason: str = ""  # BLOCK일 때 차단 이유
    updated_input: dict | None = None  # 입력을 수정한 경우 (PRE_TOOL_USE)
    message: str = ""  # 추가 메시지 (로깅/알림용)


# 훅 핸들러 타입 — 비동기 함수로 HookInput을 받아 HookResult를 반환
HookHandler = Callable[[HookInput], Coroutine[Any, Any, HookResult]]


# ─────────────────────────────────────────────
# HookManager — 훅 등록 및 실행 관리
# ─────────────────────────────────────────────
class HookManager:
    """
    훅을 등록하고 실행하는 매니저.

    이벤트별로 훅 핸들러를 등록하고,
    해당 이벤트 발생 시 등록된 순서대로 실행한다.
    """

    def __init__(self) -> None:
        """HookManager를 초기화한다."""
        # 이벤트별 핸들러 목록 (등록 순서 유지)
        self._hooks: dict[HookEvent, list[HookHandler]] = defaultdict(list)

    def register(self, event: HookEvent, handler: HookHandler) -> None:
        """
        훅 핸들러를 등록한다.

        같은 이벤트에 여러 핸들러를 등록할 수 있다.
        등록 순서대로 실행된다.

        Args:
            event: 훅이 실행될 이벤트
            handler: 비동기 훅 핸들러 함수
        """
        self._hooks[event].append(handler)
        logger.debug("훅 등록: event=%s, handler=%s", event.value, handler.__name__)

    def unregister(self, event: HookEvent, handler: HookHandler) -> bool:
        """
        훅 핸들러를 제거한다.

        Args:
            event: 훅 이벤트
            handler: 제거할 핸들러

        Returns:
            제거 성공 여부
        """
        handlers = self._hooks.get(event, [])
        try:
            handlers.remove(handler)
            logger.debug("훅 제거: event=%s, handler=%s", event.value, handler.__name__)
            return True
        except ValueError:
            return False

    async def run(self, event: HookEvent, hook_input: HookInput) -> HookResult:
        """
        등록된 훅을 순서대로 실행한다.

        실행 규칙:
        - BLOCK이면 즉시 중단하고 반환
        - APPROVE면 즉시 승인하고 반환
        - CONTINUE면 다음 훅으로
        - 모든 훅이 CONTINUE면 최종 CONTINUE 반환
        - 핸들러 에러 시 해당 훅만 건너뛰고 계속 진행

        Args:
            event: 실행할 이벤트 타입
            hook_input: 훅 입력 데이터

        Returns:
            최종 훅 결과
        """
        handlers = self._hooks.get(event, [])

        if not handlers:
            # 등록된 훅이 없으면 CONTINUE
            return HookResult(decision=HookDecision.CONTINUE)

        for handler in handlers:
            try:
                result = await handler(hook_input)

                # BLOCK — 즉시 중단
                if result.decision == HookDecision.BLOCK:
                    logger.info(
                        "훅이 차단: handler=%s, reason=%s",
                        handler.__name__,
                        result.block_reason,
                    )
                    return result

                # APPROVE — 즉시 승인
                if result.decision == HookDecision.APPROVE:
                    logger.debug(
                        "훅이 승인: handler=%s, message=%s",
                        handler.__name__,
                        result.message,
                    )
                    return result

                # CONTINUE — 다음 훅으로
                # updated_input이 있으면 다음 훅에 전달
                if result.updated_input is not None:
                    hook_input = hook_input.model_copy(update={"tool_input": result.updated_input})

            except Exception as e:
                # 훅 에러는 로그만 남기고 계속 진행 (fail-open for hooks)
                # 왜: 훅은 부가 기능이므로 훅 에러가 도구 실행을 막으면 안 된다
                logger.error(
                    "훅 실행 에러: handler=%s, error=%s",
                    handler.__name__,
                    str(e),
                    exc_info=True,
                )
                continue

        # 모든 훅이 CONTINUE — 최종 CONTINUE 반환
        return HookResult(decision=HookDecision.CONTINUE)

    def get_registered_events(self) -> list[HookEvent]:
        """핸들러가 등록된 이벤트 목록을 반환한다."""
        return [event for event, handlers in self._hooks.items() if handlers]

    def get_handler_count(self, event: HookEvent) -> int:
        """특정 이벤트에 등록된 핸들러 수를 반환한다."""
        return len(self._hooks.get(event, []))

    def clear(self) -> None:
        """모든 훅을 제거한다 (테스트용)."""
        self._hooks.clear()
