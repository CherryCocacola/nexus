"""
훅(Hook) 매니저 — HookManager.

[이 파일이 하는 일]
Nexus는 도구(Tool)를 실행할 때 그 "전/후" 특정 시점에 사용자가 끼워 넣은
작은 함수(=훅)를 실행할 수 있다. 이 파일은 그런 훅들을 이벤트별로 모아 두고,
정해진 순서와 규칙에 따라 실행해 주는 관리자(HookManager)를 제공한다.
훅은 도구 실행을 막거나(BLOCK), 강제로 승인하거나(APPROVE), 입력을 살짝
바꾸는(updated_input) 용도로 쓴다. 예: 위험한 명령 추가 차단, 감사 로깅,
알림 발송, 입력 정규화 등. (사양서 Ch.10.4 기반)

[핵심 구성요소]
  - HookEvent   : 훅이 실행되는 "시점"을 나타내는 열거형(4종).
  - HookDecision: 훅 하나가 내리는 "결정"(승인/차단/통과).
  - HookInput   : 훅 핸들러에 넘겨주는 입력 데이터(도구 이름/입력/결과 등).
  - HookResult  : 훅 핸들러가 돌려주는 결과(결정 + 차단사유 + 수정입력 등).
  - HookHandler : 훅 핸들러 함수의 타입 별칭(HookInput→HookResult 비동기 함수).
  - HookManager : 훅을 register/unregister 하고 run()으로 순차 실행하는 본체.

[훅 실행 규칙 — run()이 지키는 순서]
  - 같은 이벤트에 등록된 훅들을 "등록된 순서대로" 하나씩 순차 실행한다.
  - 어떤 훅이 BLOCK을 반환하면 즉시 멈추고 그 결과를 반환(뒤 훅은 실행 안 함).
  - 어떤 훅이 APPROVE를 반환하면 즉시 승인으로 확정하고 반환(뒤 훅 건너뜀).
  - CONTINUE면 그 훅은 통과로 보고 다음 훅으로 넘어간다.
  - 모든 훅이 CONTINUE면, 최종 결과도 CONTINUE(=아무도 막지 않음)로 반환한다.

[가장 중요한 안전 원칙 — fail-open]
훅 핸들러 실행 도중 예외가 나면, 그 훅만 로그로 남기고 "건너뛴" 뒤 계속 진행한다.
즉 훅 에러가 도구 실행 자체를 막지 않는다. 훅은 어디까지나 부가 기능이므로,
훅이 터졌다고 본래 작업을 멈추면 안 된다는 판단이다. (권한 시스템의 fail-closed
와는 반대 방향이라는 점에 유의 — 훅은 "열림" 쪽으로 안전하게 무너진다.)

작성자: 이현수 / 작성일: 2026-07-05
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
    """
    훅이 실행되는 "시점"을 나타내는 열거형.

    str을 상속하므로 값 자체가 문자열("pre_tool_use" 등)이라 YAML/JSON 설정이나
    로그에 그대로 실어도 사람이 읽기 좋다. 오케스트레이터/도구 실행기가 각 시점에
    도달했을 때 이 값으로 HookManager.run(event=...)을 호출한다.
    """

    PRE_TOOL_USE = "pre_tool_use"  # 도구 실행 "직전" — 검증/차단/입력수정에 사용
    POST_TOOL_USE = "post_tool_use"  # 도구 실행 "직후" — 결과 로깅/후처리에 사용
    STOP = "stop"  # 쿼리 루프가 중단(정지)되는 시점
    NOTIFICATION = "notification"  # 일반 알림 이벤트(사용자 통지 등)


# ─────────────────────────────────────────────
# 훅 결정
# ─────────────────────────────────────────────
class HookDecision(str, Enum):
    """
    훅 하나가 내리는 "결정" 결과.

    run()은 이 값을 보고 흐름을 정한다: APPROVE/BLOCK은 즉시 확정(단락 평가)이고,
    CONTINUE는 판단을 다음 훅에게 넘긴다는 뜻이다. 아무 결정도 명시하지 않은 훅은
    HookResult 기본값인 CONTINUE로 동작한다.
    """

    APPROVE = "approve"  # 즉시 승인 — 이후 훅을 건너뛰고 이 결과로 확정
    BLOCK = "block"  # 즉시 차단 — 이후 훅을 건너뛰고 이 결과로 확정
    CONTINUE = "continue"  # 판단 보류 — 다음 훅으로 넘김 (기본값)


# ─────────────────────────────────────────────
# 훅 입력 — 훅 핸들러에 전달되는 데이터
# ─────────────────────────────────────────────
class HookInput(BaseModel):
    """
    훅 핸들러에 전달되는 입력 데이터(Pydantic v2 모델).

    어떤 이벤트냐에 따라 실제로 채워지는 필드가 달라진다. 예를 들어
    PRE_TOOL_USE에서는 tool_name/tool_input이 의미를 갖고, POST_TOOL_USE에서는
    거기에 더해 tool_result가 채워진다. 채워지지 않는 필드는 None으로 둔다.
    필요한 부가 정보는 metadata에 자유롭게 실어 보낼 수 있다.
    """

    event: HookEvent  # 지금 어떤 시점에서 호출된 훅인지(필수)
    tool_name: str | None = None  # 실행하려는(또는 실행한) 도구 이름
    tool_input: dict | None = None  # 도구에 넘어가는 입력 인자
    tool_result: str | None = None  # 도구 실행 결과 문자열 (POST_TOOL_USE에서만 채움)
    metadata: dict[str, Any] = Field(default_factory=dict)  # 그 밖의 부가 정보


# ─────────────────────────────────────────────
# 훅 결과 — 훅 핸들러가 반환하는 데이터
# ─────────────────────────────────────────────
class HookResult(BaseModel):
    """
    훅 핸들러가 돌려주는 결과(Pydantic v2 모델).

    decision이 이 결과의 핵심이며, 기본값은 CONTINUE라서 훅이 아무 것도 하지 않고
    빈 HookResult()만 돌려줘도 "그냥 통과"로 해석된다.
      - BLOCK을 줄 때는 block_reason에 사람이 읽을 수 있는 차단 사유를 담는다.
      - PRE_TOOL_USE에서 입력을 손보고 싶으면 updated_input에 새 입력 dict를 담는다.
        (run()이 이걸 받아 다음 훅에게 전달되는 tool_input을 갱신한다.)
      - message는 승인/로깅/알림에 쓰는 부가 설명 문자열이다.
    """

    decision: HookDecision = HookDecision.CONTINUE  # 기본: 다음 훅으로 넘김
    block_reason: str = ""  # BLOCK일 때의 차단 사유(사용자 안내용)
    updated_input: dict | None = None  # 입력을 수정했을 때만 채움 (PRE_TOOL_USE)
    message: str = ""  # 로깅/알림용 부가 메시지


# 훅 핸들러 타입 별칭.
# 훅 핸들러는 "HookInput 하나를 받아 HookResult를 돌려주는 비동기 함수(async def)"다.
# Coroutine[..., HookResult]로 표기한 이유는 async 함수를 호출하면 코루틴이 나오고,
# run()이 그것을 await 하기 때문이다.
HookHandler = Callable[[HookInput], Coroutine[Any, Any, HookResult]]


# ─────────────────────────────────────────────
# HookManager — 훅 등록 및 실행 관리
# ─────────────────────────────────────────────
class HookManager:
    """
    훅을 등록하고 실행하는 매니저(이 파일의 본체).

    내부에 "이벤트 → 핸들러 목록" 사전을 두고, register()로 훅을 쌓고 run()으로
    특정 이벤트의 훅들을 순서대로 돌린다. 오케스트레이터/도구 실행기 쪽에서 이
    인스턴스를 하나 들고 다니며 각 시점마다 run()을 호출하는 식으로 쓴다.

    사용 흐름(예):
      manager = HookManager()
      manager.register(HookEvent.PRE_TOOL_USE, my_guard)   # 훅 등록
      result = await manager.run(HookEvent.PRE_TOOL_USE, hook_input)  # 실행
      if result.decision == HookDecision.BLOCK: ...          # 결과에 따라 분기
    """

    def __init__(self) -> None:
        """HookManager를 초기화한다."""
        # 이벤트별 핸들러 목록. defaultdict(list)라서 처음 보는 이벤트를 조회/추가해도
        # 자동으로 빈 리스트가 만들어진다. list이므로 "등록 순서"가 그대로 보존된다.
        self._hooks: dict[HookEvent, list[HookHandler]] = defaultdict(list)

    def register(self, event: HookEvent, handler: HookHandler) -> None:
        """
        특정 이벤트에 훅 핸들러를 하나 등록한다.

        같은 이벤트에 여러 핸들러를 등록할 수 있으며, 나중에 run()에서 "등록한
        순서대로" 실행된다. 중복 등록 방지는 하지 않으므로, 같은 핸들러를 두 번
        register하면 두 번 실행된다는 점에 유의한다.

        Args:
            event: 이 훅을 실행할 시점(HookEvent)
            handler: HookInput을 받아 HookResult를 반환하는 비동기 함수
        """
        # 해당 이벤트의 목록 끝에 붙여 등록 순서를 유지한다.
        self._hooks[event].append(handler)
        logger.debug("훅 등록: event=%s, handler=%s", event.value, handler.__name__)

    def unregister(self, event: HookEvent, handler: HookHandler) -> bool:
        """
        이미 등록된 훅 핸들러를 제거한다.

        해당 이벤트에서 주어진 핸들러를 찾아 목록에서 뺀다. 없으면 아무 일도
        일어나지 않고 False를 돌려준다(예외로 터뜨리지 않음).

        Args:
            event: 훅 이벤트
            handler: 제거할 핸들러(등록 때와 동일한 함수 객체여야 함)

        Returns:
            실제로 제거했으면 True, 목록에 없어서 못 뺐으면 False
        """
        # get()으로 조회 — 등록된 적 없는 이벤트여도 KeyError 없이 빈 리스트를 받는다.
        handlers = self._hooks.get(event, [])
        try:
            # list.remove()는 대상이 없으면 ValueError를 던지므로 아래에서 잡는다.
            handlers.remove(handler)
            logger.debug("훅 제거: event=%s, handler=%s", event.value, handler.__name__)
            return True
        except ValueError:
            # 목록에 없던 핸들러 — 실패로 간주하고 False 반환
            return False

    async def run(self, event: HookEvent, hook_input: HookInput) -> HookResult:
        """
        해당 이벤트에 등록된 훅들을 순서대로 실행하고 최종 결정을 반환한다.

        이 메서드가 HookManager의 심장이다. 등록 순서대로 훅을 하나씩 await 하면서
        결정에 따라 흐름을 제어한다(단락 평가).

        실행 규칙:
        - BLOCK이면 즉시 멈추고 그 결과를 반환(뒤 훅은 실행하지 않음)
        - APPROVE면 즉시 승인으로 확정하고 그 결과를 반환(뒤 훅 건너뜀)
        - CONTINUE면 통과로 보고 다음 훅으로 진행
        - 모든 훅이 CONTINUE면 최종적으로 CONTINUE 결과를 새로 만들어 반환
        - 핸들러가 예외를 던지면 그 훅만 건너뛰고 계속 진행(fail-open, 위 참고)

        Args:
            event: 실행할 이벤트 타입
            hook_input: 훅들에게 넘길 입력 데이터

        Returns:
            최종 훅 결과(HookResult). 아무도 막지 않았으면 CONTINUE.
        """
        # 이 이벤트에 등록된 핸들러 목록을 가져온다(없으면 빈 리스트).
        handlers = self._hooks.get(event, [])

        if not handlers:
            # 등록된 훅이 하나도 없으면 그냥 통과(CONTINUE)로 빠르게 반환한다.
            return HookResult(decision=HookDecision.CONTINUE)

        # 등록 순서대로 한 개씩 실행한다.
        for handler in handlers:
            try:
                # 훅 핸들러는 비동기 함수이므로 await로 결과를 기다린다.
                result = await handler(hook_input)

                # BLOCK — 여기서 즉시 멈추고 차단 결과를 그대로 위로 올린다.
                if result.decision == HookDecision.BLOCK:
                    logger.info(
                        "훅이 차단: handler=%s, reason=%s",
                        handler.__name__,
                        result.block_reason,
                    )
                    return result

                # APPROVE — 여기서 즉시 승인으로 확정하고 반환한다.
                if result.decision == HookDecision.APPROVE:
                    logger.debug(
                        "훅이 승인: handler=%s, message=%s",
                        handler.__name__,
                        result.message,
                    )
                    return result

                # CONTINUE — 다음 훅으로 넘어간다.
                # 단, 이 훅이 입력을 수정했다면(updated_input) 그 수정본을 다음 훅에게
                # 전달해야 한다. HookInput은 Pydantic 모델이라 직접 바꾸지 않고,
                # model_copy로 tool_input만 갈아끼운 "새 입력"을 만들어 이어 쓴다.
                if result.updated_input is not None:
                    hook_input = hook_input.model_copy(update={"tool_input": result.updated_input})

            except Exception as e:
                # 훅 에러는 로그만 남기고 계속 진행한다 (훅에 한해 fail-open).
                # 왜: 훅은 부가 기능이므로 훅에서 난 에러가 도구 실행을 막으면 안 된다.
                logger.error(
                    "훅 실행 에러: handler=%s, error=%s",
                    handler.__name__,
                    str(e),
                    exc_info=True,
                )
                # 이 훅만 포기하고 다음 훅으로. (전체 실행을 멈추지 않는다.)
                continue

        # 반복문을 다 돌 때까지 BLOCK/APPROVE가 없었다는 뜻 = 모두 CONTINUE.
        # 그래서 최종 결과도 CONTINUE로 새로 만들어 반환한다.
        return HookResult(decision=HookDecision.CONTINUE)

    def get_registered_events(self) -> list[HookEvent]:
        """
        핸들러가 "실제로 하나 이상 등록된" 이벤트만 골라 목록으로 반환한다.

        defaultdict 특성상 조회만 해도 빈 리스트가 생길 수 있으므로, 빈 목록인
        이벤트는 제외하고 실제 훅이 있는 이벤트만 돌려준다.
        """
        return [event for event, handlers in self._hooks.items() if handlers]

    def get_handler_count(self, event: HookEvent) -> int:
        """특정 이벤트에 등록된 핸들러 개수를 반환한다(없으면 0)."""
        return len(self._hooks.get(event, []))

    def clear(self) -> None:
        """등록된 모든 훅을 싹 비운다 (주로 테스트에서 상태 초기화용)."""
        self._hooks.clear()
