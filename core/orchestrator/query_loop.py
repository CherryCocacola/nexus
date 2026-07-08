"""
쿼리 루프 — while(True) 에이전트 턴 루프.

Claude Code의 query.ts (1,729줄)를 Python으로 완전 재구현한다.
4-Tier AsyncGenerator 체인의 Tier 2에 해당한다.

핵심 구조: while(True) 에이전트 턴 루프
모델이 도구 사용을 멈출 때까지 (또는 에러/제한에 도달할 때까지) 반복한다.

한 번의 반복(iteration) = 4 Phase:
  Phase 1: Pre-API (컨텍스트 압축, max_tokens 결정, 도구 스키마 준비)
  Phase 2: API Call (model_provider.stream() + 이벤트 수집 + 스트리밍 도구 실행)
  Phase 3: Post-API (사용량 추적, 에러 복구, 종료 판단)
  Phase 4: Tool Execution (StreamingToolExecutor drain → 다음 턴)

7가지 Continue Transition:
  1. collapse_drain_retry: 컨텍스트 초과 → 긴급 압축 후 재시도
  2. reactive_compact_retry: prompt-too-long → 압축 후 재시도
  3. max_output_tokens_escalate: 출력 토큰 증가 후 재시도
  4. max_output_tokens_recovery: 멀티턴 이어쓰기 복구
  5. stop_hook_blocking: Hook이 종료를 차단 → 강제 다음 턴
  6. token_budget_continuation: 토큰 예산 부족 → 계속
  7. next_turn: 도구 실행 후 정상 다음 턴

────────────────────────────────────────────
이 파일에 들어 있는 것 (온보딩용 지도):
  - ContinueReason (Enum): 루프가 "왜 다음 턴으로 가는가"를 나타내는 7가지 이유.
  - LoopState (dataclass): 턴 카운트·토큰 누적·각종 재시도 카운터를 담은 루프 상태.
  - 상수들(MAX_TURNS 등): 무한 루프 방지·재시도 한도·출력 토큰 에스컬레이션 단계.
  - _shrink_text / _truncate_input_for_budget: 입력이 컨텍스트를 초과할 때
    메시지 content를 "예산(budget)" 안으로 줄이는 순수 헬퍼(원본 비훼손·페어링 보존).
  - _RecoveryOutcome / _error_event_to_recovery_text / _try_recover_from_model_error:
    모델 에러(컨텍스트 초과·prompt-too-long·GPU OOM)를 판정해 압축/축소로 복구하는 헬퍼.
    예외 경로와 ERROR 이벤트 경로 두 곳에서 재사용한다(감사 Critical #8 대응).
  - query_loop (핵심 함수): 위 요소를 엮은 while(True) 턴 루프. 이 파일의 심장부.

호출·의존 관계 (4-Tier 체인 안에서의 위치):
  - 상위(Tier 1): QueryEngine.submit_message()가 query_loop을 호출한다.
  - 하위(Tier 3): query_loop이 model_provider.stream()을 호출한다(Phase 2).
  - 협력 모듈: StopResolver(종료 판정), StreamingToolExecutor(도구 병렬 실행),
    ContextManager(압축, 선택), HookManager(종료 훅, 선택), stream_watchdog(무응답 감지).
  - 데이터 계약: 내부는 항상 OpenAI tool_calls 형식만 다룬다(XML은 보지 않음).

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.message import (
    Message,
    Role,
    StopReason,
    StreamEvent,
    StreamEventType,
    TokenUsage,
)
from core.model.inference import ModelProvider
from core.orchestrator.stop_resolver import StopResolver, _seems_truncated
from core.orchestrator.stream_handler import StreamingToolExecutor
from core.tools.base import BaseTool, ToolUseContext

logger = logging.getLogger("nexus.orchestrator.query_loop")


# ─────────────────────────────────────────────
# Continue Transitions (7가지 계속 전환 이유)
# ─────────────────────────────────────────────
class ContinueReason(str, Enum):
    """
    query_loop의 7가지 계속 전환 이유.

    각 값은 루프가 왜 다음 턴으로 진행하는지를 나타낸다.
    로깅과 디버깅에서 현재 상태를 추적하는 데 사용한다.
    """

    NEXT_TURN = "next_turn"  # 정상: 도구 실행 후 다음 턴
    COLLAPSE_DRAIN_RETRY = "collapse_drain_retry"  # 컨텍스트 긴급 압축 후 재시도
    REACTIVE_COMPACT_RETRY = "reactive_compact_retry"  # prompt-too-long 압축 후 재시도
    MAX_OUTPUT_TOKENS_ESCALATE = "max_output_tokens_escalate"  # 출력 토큰 증가 후 재시도
    MAX_OUTPUT_TOKENS_RECOVERY = "max_output_tokens_recovery"  # 멀티턴 이어쓰기 복구
    STOP_HOOK_BLOCKING = "stop_hook_blocking"  # Hook이 종료 차단
    TOKEN_BUDGET_CONTINUATION = "token_budget_continuation"  # noqa: S105 — 토큰 예산 계속


# ─────────────────────────────────────────────
# Loop State (루프의 명시적 상태)
# ─────────────────────────────────────────────
@dataclass
class LoopState:
    """
    query_loop의 명시적 상태.

    Claude Code의 query.ts 내부 state 객체에 대응한다.
    각 필드는 특정 Continue Transition에 연결되어 있다.
    """

    # 기본 상태
    messages: list[Message]
    turn_count: int = 0
    continue_reason: ContinueReason = ContinueReason.NEXT_TURN

    # 토큰 추적
    cumulative_usage: TokenUsage = field(default_factory=TokenUsage)

    # max_output_tokens 복구 (Transition 3, 4)
    max_output_tokens_override: int | None = None
    max_output_recovery_count: int = 0

    # 에러 복구 카운터
    tool_parse_retry_count: int = 0
    model_error_count: int = 0
    compact_retry_count: int = 0
    collapse_drain_count: int = 0

    # 마지막 종료 이유
    last_stop_reason: StopReason | None = None

    # 시간 추적
    start_time: float = field(default_factory=time.monotonic)

    @property
    def elapsed_seconds(self) -> float:
        """루프 시작 이후 경과 시간(초)."""
        return time.monotonic() - self.start_time


# ─────────────────────────────────────────────
# 상수 (재시도 제한, 토큰 에스컬레이션 단계)
# ─────────────────────────────────────────────
MAX_TURNS = 200  # 최대 턴 수 (무한 루프 방지)
MAX_OUTPUT_RECOVERY = 3  # max_output 복구 최대 시도 횟수
MAX_TOOL_PARSE_RETRY = 2  # 도구 JSON 파싱 재시도 횟수
MAX_MODEL_ERROR_RETRY = 3  # 모델 에러 재시도 횟수
MAX_COMPACT_RETRY = 2  # prompt-too-long 압축 재시도 횟수
MAX_COLLAPSE_DRAIN = 1  # 긴급 압축 최대 횟수
# 반복 실패 도구 호출 가드 — 모델이 같은 도구를 같은 입력으로 계속 호출하는데
# 매번 실패하는(예: 깨진 Bash 명령을 못 고치고 반복하는) 상황을 조기에 끊는다.
# MAX_TURNS(200)는 이런 짧은 반복 루프를 잡기엔 너무 커서 별도 임계를 둔다.
REPEATED_TOOL_FAILURE_WARN = 3  # 이 횟수 연속 실패 시 "반복 중단" 피드백 1회 주입
REPEATED_TOOL_FAILURE_ABORT = 5  # 이 횟수 연속 실패 시 턴 루프 강제 종료(백스톱)

# 출력 토큰 에스컬레이션 단계
# max_tokens를 점진적으로 증가시킨다 (4K → 8K → 16K)
OUTPUT_TOKEN_ESCALATION = [4096, 8192, 16384]


def _tool_call_signature(name: str, tool_input: Any) -> str:
    """도구 호출을 '이름 + 정규화된 입력'으로 서명한다.

    같은 명령의 반복을 식별하기 위한 키다. 입력 dict를 키 정렬 JSON으로
    직렬화해, 키 순서만 다른 동일 입력도 같은 서명이 되게 한다. 직렬화가
    불가능한 입력은 str()로 폴백한다(서명은 완벽할 필요 없이 안정적이면 된다).
    """
    try:
        norm = json.dumps(tool_input, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        norm = str(tool_input)
    return f"{name}::{norm}"


def _update_tool_failure_streak(
    tool_use_blocks: list[dict[str, Any]],
    id_to_error: dict[str, bool],
    streak: dict[str, int],
) -> int:
    """이번 턴 도구 호출들의 '연속 실패 턴 수'를 갱신하고 최댓값을 돌려준다.

    카운터의 의미는 "그 서명이 연속으로 실패한 '턴' 수"다. 이를 지키기 위해:
    - 같은 서명이 한 턴에 병렬로 여러 번 호출돼도(list[dict] 병렬 호출) 카운터는
      턴당 최대 +1만 오른다(턴 내 중복이 ABORT를 앞당기지 못하게).
    - 실패한 서명: +1. 성공(또는 결과 없음)한 서명: 리셋(더는 '반복 실패' 아님).
    - 이번 턴에 아예 등장하지 않은 과거 서명: '연속'이 끊긴 것이므로 제거한다.
      그래야 모델이 다른 명령으로 넘어가면 낡은 실패 카운트가 판정을 오염시키지
      않는다(안 그러면 max()가 과거 서명에 영영 지배당함).

    반환값은 '이번 턴에 등장한' 서명들의 최고 연속 실패 횟수 — 호출부가 이 값으로
    경고 주입(WARN)/강제 종료(ABORT)를 판단한다. streak dict는 query_loop
    1회 호출 동안만 유지되는 지역 상태라 세션/서브에이전트 간 격리된다.
    """
    # 1) 이번 턴 서명별 실패 여부를 OR로 집계(병렬 중복 호출을 턴당 1회로 합침).
    errored_this_turn: dict[str, bool] = {}
    for tu in tool_use_blocks:
        sig = _tool_call_signature(tu.get("name", ""), tu.get("input"))
        errored = id_to_error.get(tu.get("id", ""), False)
        errored_this_turn[sig] = errored_this_turn.get(sig, False) or errored

    # 2) 이번 턴에 등장하지 않은 과거 서명은 연속이 끊긴 것 → 제거.
    for stale_sig in [s for s in streak if s not in errored_this_turn]:
        streak.pop(stale_sig, None)

    # 3) 이번 턴 서명별로 +1(실패) 또는 리셋(성공).
    for sig, errored in errored_this_turn.items():
        if errored:
            streak[sig] = streak.get(sig, 0) + 1
        else:
            streak.pop(sig, None)

    # 4) 남은 건 이번 턴에 실패한 서명뿐 → 그 최댓값이 최고 연속 실패 턴 수.
    return max(streak.values(), default=0)


# ─────────────────────────────────────────────
# Phase 1 입력 예산 truncation 헬퍼
# ─────────────────────────────────────────────
#
# 배경(왜 이 헬퍼가 필요한가):
#   예전 Phase 1 코드는 입력이 너무 길 때 "마지막 메시지 하나"만 잘랐다.
#   그러나 실제 오버플로우는 대개 '여러 tool_result가 누적'되며 발생하는데,
#   이 경우 마지막 메시지 하나는 초과분(excess)보다 작아서
#   `if len(content) > excess + 200` 조건이 False → 아무것도 못 자르고
#   그대로 모델에 전송 → 모델이 예산 초과로 거부 → 하드에러가 났다.
#   (증거 로그: "입력 truncate: 26524 → 26524 토큰" — 전혀 안 줄어듦)
#
#   그래서 접근을 '예산(budget) 방식'으로 바꾼다. 고정 오버헤드(도구 스키마 +
#   시스템 프롬프트)는 못 줄이므로, "메시지 content가 통틀어 써도 되는 글자 예산"을
#   역산한 뒤, 입력이 어디에 몰려 있든 그 예산 이하로 확실히 낮춘다.


def _shrink_text(text: str, max_chars: int) -> str:
    """
    평문 문자열 하나를 max_chars 글자 이하로 줄인다.

    왜 앞+뒤를 남기고 가운데를 생략하는가:
      - tool_result/사용자 입력은 보통 앞부분(무엇에 대한 결과인지)과
        뒷부분(결론/요약)이 가장 중요하고, 가운데 본문이 길이를 폭발시킨다.
        그래서 head(앞)와 tail(뒤)을 남기고 가운데만 생략 표시로 대체한다.
      - 반환 길이는 반드시 max_chars 이하임을 마지막에 강제 clamp로 보장한다
        (예산 초과를 절대 만들지 않기 위해).
    """
    if len(text) <= max_chars:
        return text
    # 몇 글자를 생략하는지 안내 문구(marker). 길이는 대략적이어도 무방하다.
    omitted = len(text) - max_chars
    marker = f"\n…[중략: 약 {omitted}자 생략]…\n"
    # head/tail에 실제로 나눠 쓸 예산 = 전체 예산에서 marker 길이를 뺀 값
    body_budget = max_chars - len(marker)
    if body_budget <= 0:
        # 예산이 marker보다도 작은 극단적 경우: 표식만 남기고 강제로 자른다.
        return marker.strip()[:max_chars]
    # 앞부분을 더 많이(2/3) 남긴다 — 맥락 파악에 앞부분이 더 유용하기 때문.
    head_budget = (body_budget * 2) // 3
    tail_budget = body_budget - head_budget
    head = text[:head_budget]
    tail = text[len(text) - tail_budget :] if tail_budget > 0 else ""
    result = head + marker + tail
    # 안전장치: 반올림/marker 길이 오차로 혹시라도 넘치면 강제로 자른다.
    if len(result) > max_chars:
        result = result[:max_chars]
    return result


def _truncate_input_for_budget(
    api_messages: list[Message],
    msg_char_budget: int,
) -> tuple[list[Message], int, int]:
    """
    입력 메시지들의 content 총 글자수를 msg_char_budget 이하로 낮춘 '새 리스트'를 만든다.

    핵심 원칙(정확성 제약):
      1. 원본(state.messages)을 절대 훼손하지 않는다.
         줄여야 하는 메시지는 in-place로 고치지 않고, 팩토리(Message.user/
         tool_result/system)로 '새 Message'를 만들어 새 리스트에 담는다.
         변경이 필요 없는 메시지는 원본 객체를 그대로 참조로 재사용한다.
      2. tool_use ↔ tool_result 페어링을 절대 깨지 않는다.
         - 메시지를 '제거'하지 않는다 (제거하면 짝이 깨진다). 오직 content 문자열만 줄인다.
         - assistant의 구조화 content(list — tool_use 블록 포함)는 건드리지 않는다.
           평문 str content(user/tool_result/system)만 줄인다.
         - tool_result는 팩토리로 재생성할 때 tool_use_id/is_error를 그대로 넘겨
           짝 tool_use와의 연결을 유지한다.
      3. 오래된 것부터 줄이고 최근 메시지는 최대한 온전히 남긴다.
         최신 평문 메시지부터 예산을 채워주고, 예산이 소진되면 오래된 메시지가
         먼저 짧아지도록 배분한다.

    Args:
        api_messages: 이번 API 호출에 쓸 메시지 리스트 (state.messages 또는 그 파생)
        msg_char_budget: 메시지 content가 통틀어 쓸 수 있는 글자 예산

    Returns:
        (새 메시지 리스트, 원래 메시지 총 글자수, 줄인 뒤 메시지 총 글자수)
    """
    if msg_char_budget < 0:
        msg_char_budget = 0

    # 줄일 수 있는(평문 str content) 메시지의 인덱스 집합.
    # content가 list(assistant tool_use 등)인 메시지는 건드리지 않는다 → 페어링 보존.
    shrinkable_idx = [
        i for i, m in enumerate(api_messages) if isinstance(m.content, str)
    ]
    shrinkable_set = set(shrinkable_idx)

    # 줄일 수 없는 메시지들이 이미 차지한 고정 글자수.
    fixed_chars = sum(
        len(str(m.content))
        for i, m in enumerate(api_messages)
        if i not in shrinkable_set
    )

    # 평문 메시지들이 '합쳐서' 써도 되는 글자 예산.
    # 고정분이 예산을 이미 초과하면 평문은 최소로 줄일 수밖에 없다(available=0).
    available = max(0, msg_char_budget - fixed_chars)

    # ── 최신 우선 배분 ──
    # 최신 평문 메시지부터 필요한 만큼 예산을 채워주고, 남는 예산이 없으면
    # 오래된 메시지는 0(=최소 표식)까지 줄어든다. 이렇게 하면 가장 최근 턴이
    # 최대한 온전히 남는다.
    alloc: dict[int, int] = {}
    remaining = available
    for i in reversed(shrinkable_idx):  # 최신 → 오래된 순
        cur = len(str(api_messages[i].content))
        give = min(cur, remaining)
        alloc[i] = give
        remaining -= give

    # 새 리스트 구성 — 원본은 그대로 두고, 줄일 메시지만 팩토리로 새로 만든다.
    new_messages: list[Message] = []
    before_chars = 0
    after_chars = 0
    for i, m in enumerate(api_messages):
        cur = len(str(m.content))
        before_chars += cur
        # 배분된 예산보다 길면 줄인다. (평문 메시지만 alloc에 존재)
        if i in alloc and cur > alloc[i]:
            shrunk = _shrink_text(str(m.content), alloc[i])
            if m.role == Role.USER:
                nm = Message.user(shrunk)
            elif m.role == Role.TOOL_RESULT:
                # tool_use_id/is_error를 보존해 짝 tool_use와의 연결을 유지한다.
                nm = Message.tool_result(
                    tool_use_id=m.tool_use_id or "",
                    content=shrunk,
                    is_error=bool(m.is_error),
                )
            elif m.role == Role.SYSTEM:
                nm = Message.system(shrunk)
            else:
                # 예상 못한 역할은 안전하게 원본 유지(페어링 훼손 방지).
                nm = m
            new_messages.append(nm)
            after_chars += len(str(nm.content))
        else:
            # 변경 없는 메시지는 원본 참조를 그대로 재사용한다.
            new_messages.append(m)
            after_chars += cur

    return new_messages, before_chars, after_chars


# ─────────────────────────────────────────────
# 모델 에러 복구 헬퍼 (감사 Critical #8 대응)
# ─────────────────────────────────────────────
#
# 배경(왜 이 헬퍼가 필요한가):
#   Tier 3(core/model/inference.py stream())는 컨텍스트 초과·HTTP 오류를
#   예외로 raise하지 않고 ERROR StreamEvent로 "yield"해서 내려보낸다.
#   그런데 기존 query_loop의 진짜 복구 로직(긴급 압축/반응적 압축/OOM 축소)은
#   `except Exception` 블록 안에만 있어서, raise가 나지 않는 이 오류들에 대해서는
#   영구 미도달이었다(= 컨텍스트 오버플로 복구 도달불가 결함).
#
#   그래서 세 가지 복구 판정을 이 헬퍼로 추출해, 두 경로에서 재사용한다:
#     (1) 기존 `except Exception` 블록  — 모델 provider가 예외를 raise한 경우
#     (2) Phase 2의 ERROR 이벤트 소비 경로 — provider가 ERROR 이벤트를 yield한 경우
#
#   헬퍼는 async generator로 만들지 않는다(4-Tier yield 흐름을 헬퍼가 가로채면
#   체인 구조가 흐트러지기 때문). 대신 "복구가 발동했는지"와 "호출부가 yield해야 할
#   이벤트 목록(예: OOM 경고)"을 담은 결과 객체를 반환하고, 실제 yield는 호출부가 한다.


@dataclass
class _RecoveryOutcome:
    """
    _try_recover_from_model_error()의 반환값.

    recovered: 세 가지 복구(collapse_drain/reactive_compact/OOM) 중 하나가
        실제로 발동했는지 여부. True면 호출부는 다음 턴으로 재시도(continue)한다.
    events: 호출부가 순서대로 yield해야 하는 StreamEvent 목록.
        (OOM 복구는 SYSTEM_WARNING을 사용자에게 보여줘야 하는데, 4-Tier yield
        흐름을 유지하려고 헬퍼가 직접 yield하지 않고 여기에 담아 되돌려준다.)
    """

    recovered: bool
    events: list[StreamEvent] = field(default_factory=list)


def _error_event_to_recovery_text(event: StreamEvent) -> str | None:
    """
    Tier 3가 yield한 ERROR StreamEvent를, 복구 판정 헬퍼가 이해하는
    "에러 텍스트"로 변환한다. 복구 대상이 아니면 None을 반환한다.

    왜 변환이 필요한가:
      - `CONTEXT_OVERFLOW`(입력 과다)는 message가 한글 안내문이라 영어 패턴
        매칭이 안 된다. 이 오류는 "입력이 너무 길다" = 긴급 압축(collapse_drain)
        경로로 처리해야 하므로, 헬퍼가 collapse_drain으로 인식하도록
        "context too long" 계열 텍스트를 합성해 돌려준다.
      - `HTTP_4xx`는 message에 vLLM 본문이 담겨 있다(‘maximum context length’/
        ‘prompt is too long’/‘out of memory’ 등). 그 본문을 그대로 넘겨
        헬퍼의 세 패턴이 판정하게 한다.
      - 그 외(CONNECT_ERROR/READ_TIMEOUT/UNKNOWN/HTTP_5xx 등)는 압축으로
        해결되지 않는 오류이므로 None을 반환한다 → 호출부는 현행 동작을 유지한다.
    """
    code = event.error_code or ""
    # 입력 과다(CONTEXT_OVERFLOW) → 긴급 압축(collapse_drain) 경로로 매핑.
    # "context"와 "long"을 모두 포함시켜 헬퍼의 collapse_drain 조건에 걸리게 한다.
    if code == "CONTEXT_OVERFLOW":
        return "context too long"
    # 4xx 응답만 복구 후보 — 본문(message)을 헬퍼의 패턴 매칭에 넘긴다.
    # (패턴에 안 걸리면 헬퍼가 recovered=False를 반환 → 호출부는 현행 동작 유지)
    if code.startswith("HTTP_4"):
        return event.message or ""
    return None


async def _try_recover_from_model_error(
    error_text: str,
    state: LoopState,
    context_manager: Any | None,
    streaming_executor: Any,
) -> _RecoveryOutcome:
    """
    모델 에러 텍스트를 보고 세 가지 복구(긴급 압축/반응적 압축/OOM 축소) 중
    적용 가능한 것을 수행한다. 기존 `except Exception` 블록(:447/:462/:480)의
    복구 로직을 그대로 옮긴 것으로, 동작은 비트 단위로 동일하다(무회귀).

    각 복구는 (a) 압축/축소 수행, (b) state.continue_reason 설정,
    (c) streaming_executor.cancel_all()을 하고 recovered=True로 반환한다.
    어떤 패턴에도 걸리지 않거나 재시도 예산이 소진되면 recovered=False.

    주의(무회귀 핵심): 원본과 동일하게 세 조건을 순차 `if`로 검사한다.
      - 조건은 매칭됐지만 카운터 상한에 도달한 경우, 원본은 그 자리에서 종료하지
        않고 다음 조건으로 "fall-through"했다. 여기서도 return 하지 않고 다음
        조건 검사로 넘어가 동일한 fall-through를 재현한다.
      - collapse_drain/reactive_compact는 카운터를 `if 카운터 < 상한` 안에서만
        증가시키고, OOM은 원본처럼 매칭 시 먼저 +1 한 뒤 상한을 검사한다
        (상한 초과 시에도 카운터는 이미 증가된 상태로 fall-through).
    """
    text = error_text.lower()
    events: list[StreamEvent] = []

    # ─── Transition 1: collapse_drain_retry ───
    # "context too long" 류 → 긴급 압축 후 재시도
    if "context" in text and "long" in text:
        if state.collapse_drain_count < MAX_COLLAPSE_DRAIN:
            state.collapse_drain_count += 1
            logger.warning(
                f"컨텍스트 초과, 긴급 압축 수행 "
                f"({state.collapse_drain_count}/{MAX_COLLAPSE_DRAIN})"
            )
            if context_manager is not None:
                state.messages = await context_manager.emergency_compact(state.messages)
            state.continue_reason = ContinueReason.COLLAPSE_DRAIN_RETRY
            await streaming_executor.cancel_all()
            return _RecoveryOutcome(recovered=True, events=events)

    # ─── Transition 2: reactive_compact_retry ───
    # "prompt is too long" → 압축 후 재시도
    if "prompt is too long" in text:
        if state.compact_retry_count < MAX_COMPACT_RETRY:
            state.compact_retry_count += 1
            logger.warning(
                f"Prompt 초과, 반응적 압축 수행 "
                f"({state.compact_retry_count}/{MAX_COMPACT_RETRY})"
            )
            if context_manager is not None:
                state.messages = await context_manager.auto_compact_if_needed(
                    state.messages, force=True
                )
            state.continue_reason = ContinueReason.REACTIVE_COMPACT_RETRY
            await streaming_executor.cancel_all()
            return _RecoveryOutcome(recovered=True, events=events)

    # ─── GPU OOM → 컨텍스트 30% 감소 후 재시도 ───
    # 왜 0.7배: vLLM이 메모리를 다 못 잡으면 입력을 줄이는 것 외엔 방법이 없으므로,
    # 다음 시도에서 컨텍스트 상한을 70%로 낮춰 메모리를 확보한다.
    if "out of memory" in text:
        state.model_error_count += 1
        if state.model_error_count <= MAX_MODEL_ERROR_RETRY:
            if context_manager is not None:
                context_manager.max_tokens = int(context_manager.max_tokens * 0.7)
            events.append(
                StreamEvent(
                    type=StreamEventType.SYSTEM_WARNING,
                    message=(
                        f"[GPU OOM] 컨텍스트 축소 후 재시도 "
                        f"({state.model_error_count}/{MAX_MODEL_ERROR_RETRY})"
                    ),
                )
            )
            state.continue_reason = ContinueReason.REACTIVE_COMPACT_RETRY
            await streaming_executor.cancel_all()
            return _RecoveryOutcome(recovered=True, events=events)

    # 어떤 복구도 발동하지 못함 (미대상 패턴 또는 예산 소진)
    return _RecoveryOutcome(recovered=False, events=events)


# ─────────────────────────────────────────────
# Query Loop (핵심 함수)
# ─────────────────────────────────────────────
async def query_loop(
    messages: list[Message],
    system_prompt: str,
    model_provider: ModelProvider,
    tools: list[BaseTool],
    context: ToolUseContext,
    context_manager: Any | None = None,
    max_turns: int = MAX_TURNS,
    hook_manager: Any | None = None,
    on_turn_complete: Any | None = None,
    model_override: str | None = None,
    temperature: float = 0.7,
    max_tokens_cap: int | None = None,
    enable_thinking: bool = False,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    # 출력 토큰 에스컬레이션 단계(하드코딩 외부화, 2026-07-03).
    # None이면 모듈 상수 OUTPUT_TOKEN_ESCALATION으로 폴백 → 기존 호출부(테스트
    # 포함)는 [4096,8192,16384] 그대로 사용해 동작 불변(무회귀). config 값은
    # bootstrap→QueryEngine→(dispatcher.route/폴백 query_loop) 경로로 주입된다.
    output_token_escalation: list[int] | None = None,
) -> AsyncGenerator[StreamEvent | Message, None]:
    """
    핵심 에이전트 턴 루프.

    Claude Code의 query.ts:241-1729를 완전 재구현한다.
    모델이 도구 사용을 멈출 때까지 반복하고,
    에러 발생 시 7가지 Continue Transition으로 자동 복구한다.

    이 함수는 4-Tier 체인의 Tier 2이다:
      Tier 1: QueryEngine.submit_message() → 이 함수를 호출
      Tier 2: query_loop() ← 여기
      Tier 3: model_provider.stream() ← Phase 2에서 호출
      Tier 4: httpx 클라이언트 (model_provider 내부)

    Args:
        messages: 대화 히스토리 (mutated — 턴마다 메시지가 추가됨)
        system_prompt: 시스템 프롬프트
        model_provider: LLM 프로바이더 (Tier 3 진입점)
        tools: 사용 가능한 도구 리스트
        context: 도구 실행 컨텍스트
        context_manager: 컨텍스트 압축 관리자 (Ch.6, 선택)
        max_turns: 최대 턴 수 (기본: 200)
        hook_manager: 훅 매니저 (Ch.10, 선택) — 도구 실행 전후/종료 시 훅 실행
        model_override: v7.0 Part 2.5 — 호출 시점에 LoRA 어댑터를 덮어쓸 때 사용.
            KNOWLEDGE_MODE에서 "qwen3.5-27b"(LoRA OFF)로 라우팅하기 위한 경로.
        temperature: 샘플링 온도. 라우팅 프로필에서 결정한 값을 전달받는다.
        max_tokens_cap: 출력 상한 (선택). 지정되면 기존 동적 max_tokens 계산에서
            base_max_tokens로 사용되어, 가벼운 답변(예: KNOWLEDGE_MODE 2048)이
            필요할 때 과도한 토큰 소비를 막는다.
        enable_thinking: Qwen3.5 chat_template_kwargs 인자 (기본 False).
        top_p: nucleus 샘플링 임계 (기본 1.0=비활성). 라우팅 프로필에서 결정한 값을
            그대로 Tier 3(model_provider.stream)로 전달한다(passthrough).
        repetition_penalty: 반복 토큰 페널티 (기본 1.0=비활성). 동일 문장 무한 반복
            (degeneration)을 억제하기 위해 추가됨.
        frequency_penalty: 빈도 페널티 (기본 0.0=비활성).
        presence_penalty: 등장 페널티 (기본 0.0=비활성).

    Yields:
        StreamEvent: 스트리밍 이벤트 (UI 업데이트용)
        Message: assistant/tool_result 메시지 (대화 히스토리용)
    """
    # 루프 상태 초기화
    # state: 턴 카운트·토큰 누적·각종 재시도 카운터를 한 객체에 모아 관리한다.
    # stop_resolver: 매 턴 끝에서 "도구 호출이 남았는지"를 판정해 계속/종료를 결정한다.
    state = LoopState(messages=messages)
    stop_resolver = StopResolver()

    # 출력 토큰 에스컬레이션 단계 확정 — 주입값 우선, 미주입 시 모듈 상수 폴백.
    # (하드코딩 외부화 무회귀: config 미주입 경로/기존 테스트는 상수 그대로 사용)
    escalation_steps = output_token_escalation or OUTPUT_TOKEN_ESCALATION

    # 반복 실패 도구 호출 추적 — 서명(도구+입력)별 연속 실패 턴 수.
    # query_loop 1회 호출 동안만 유지되는 지역 상태다(서브에이전트/세션 간 격리).
    tool_failure_streak: dict[str, int] = {}
    # WARN 피드백을 이미 준 서명 집합 — 같은 실패에 매 턴 중복 경고를 막는다.
    warned_sigs: set[str] = set()

    # ── while(True) 에이전트 턴 루프 (Tier 2의 심장부) ──
    # 모델이 도구 사용을 멈추거나(정상 종료), 7가지 Continue Transition 중 하나로
    # 복구가 끝날 때까지 턴을 반복한다. max_turns는 무한 루프를 막는 안전장치다.
    while state.turn_count < max_turns:
        state.turn_count += 1

        logger.info(
            f"=== 턴 {state.turn_count} "
            f"(이유: {state.continue_reason.value}, "
            f"메시지: {len(state.messages)}개) ==="
        )

        # 턴 시작 이벤트 — UI에서 프로그레스 바 등에 사용
        yield StreamEvent(type=StreamEventType.STREAM_REQUEST_START)

        # ═══════════════════════════════════════
        # Phase 1: Pre-API 준비
        # ═══════════════════════════════════════
        # 모델에 보낼 메시지를 준비한다.
        # 컨텍스트 압축, 토큰 제한 결정, 도구 스키마 정렬 등.

        # 1a. 컨텍스트 압축
        # ContextManager가 있으면 apply_all()로 전처리하고
        # auto_compact_if_needed()로 필요 시 압축한다
        api_messages = state.messages
        if context_manager is not None:
            try:
                api_messages = context_manager.apply_all(state.messages)
                api_messages = await context_manager.auto_compact_if_needed(api_messages)
            except Exception as e:
                logger.error(f"컨텍스트 압축 실패: {e}")
                # 압축 실패 시 원본 사용

        # 1b. 도구 스키마 준비 (이름순 정렬 — prompt cache 안정성)
        tool_schemas = [t.to_schema() for t in tools]

        # 1c. max_output_tokens 동적 결정
        # 도구 스키마 + 시스템 프롬프트 + 메시지가 차지하는 토큰을 추정하고,
        # max_model_len에서 남는 공간을 출력에 할당한다.
        # 왜 동적인가: 도구 수와 대화 길이에 따라 입력 토큰이 달라지므로
        # 고정 max_tokens는 컨텍스트 초과 에러를 유발한다.
        #
        # v7.0 Part 2.5: max_tokens_cap이 주어지면 그것을 base로 사용한다.
        # 예) KNOWLEDGE_MODE는 2048로 cap하여 지식 답변에 과도한 토큰을 쓰지 않음.
        model_cfg = model_provider.get_config()
        base_max_tokens = max_tokens_cap or model_cfg.max_output_tokens

        # Transition 3/4(에스컬레이션·이어쓰기 복구)에서 override 값이 설정되면
        # 동적 계산을 건너뛰고 그 값을 그대로 쓴다. 그 외에는 매 턴 새로 계산한다.
        if state.max_output_tokens_override:
            max_tokens = state.max_output_tokens_override
        else:
            # 입력 토큰 추정: 시스템 프롬프트 + 도구 스키마 + 메시지
            # 토큰 추정은 문자수/3 (보수적) — 한글/특수문자가 많으면 토큰이 더 많다
            import json as _json

            # ensure_ascii=False로 직렬화해 한글이 \uXXXX로 부풀려지지 않게 한다
            # (그래야 실제 전송될 토큰 양에 가까운 문자수를 얻는다).
            tool_chars = sum(len(_json.dumps(s, ensure_ascii=False)) for s in tool_schemas)
            msg_chars = sum(len(str(m.content)) for m in api_messages)
            prompt_chars = len(system_prompt)
            total_chars = tool_chars + msg_chars + prompt_chars
            estimated_input = total_chars // 3  # 보수적 추정 (영어 /4, 한글 /2 → 평균 /3)

            max_context = model_cfg.max_context_tokens

            # 입력이 컨텍스트의 85%를 초과하면 메시지를 truncate한다
            # 왜 85%: 최소 출력 512토큰 + 버퍼 확보
            input_limit = int(max_context * 0.85)
            if estimated_input > input_limit and len(api_messages) >= 1:
                # ── 예산(budget) 방식 truncation ──
                # 왜 마지막 메시지 하나만 자르면 안 되는가:
                #   오버플로우가 '여러 tool_result 누적'에서 오면 마지막 메시지
                #   하나는 초과분보다 작아 아무것도 못 자르고 그대로 전송 → 하드에러.
                # 그래서 "메시지 content가 통틀어 써도 되는 글자 예산"을 역산해,
                # 입력이 어디에 몰려 있든 그 예산 이하로 확실히 낮춘다.
                #
                # estimated_input = (tool_chars + msg_chars + prompt_chars) // 3 이므로,
                # 목표 estimated_input <= input_limit 을 만족시키려면
                #   tool_chars + msg_chars + prompt_chars <= input_limit * 3
                # 이어야 한다. 고정 오버헤드(tool_chars=도구 스키마,
                # prompt_chars=시스템 프롬프트)는 줄일 수 없으므로, 메시지가 쓸 수 있는
                # 글자 예산만 역산한다:
                #   msg_char_budget = input_limit*3 - (tool_chars + prompt_chars)
                msg_char_budget = input_limit * 3 - (tool_chars + prompt_chars)

                # 헬퍼가 원본을 훼손하지 않고 '이번 호출용 사본 리스트'를 돌려준다.
                # (줄일 메시지는 팩토리로 새로 만들고, 나머지는 원본 참조 재사용)
                api_messages, before_msg_chars, after_msg_chars = (
                    _truncate_input_for_budget(api_messages, msg_char_budget)
                )

                # 자른 뒤 입력 토큰을 '실제로' 다시 추정한다.
                # (예전의 오해를 주는 26524 → 26524 로그를 진짜 before→after로 교정)
                before_input = (tool_chars + before_msg_chars + prompt_chars) // 3
                msg_chars = after_msg_chars
                total_chars = tool_chars + msg_chars + prompt_chars
                estimated_input = total_chars // 3

                logger.info(
                    "입력 truncate: %d → %d 토큰 "
                    "(컨텍스트 %d의 85%%=%d, 메시지 글자 %d → %d)",
                    before_input, estimated_input, max_context, input_limit,
                    before_msg_chars, after_msg_chars,
                )

            # 최대 컨텍스트에서 입력을 빼고 200 토큰 버퍼를 둔다
            # (추정 오차를 흡수하는 안전 마진).
            dynamic_max = max_context - estimated_input - 200
            # 최종 max_tokens는 세 값의 균형으로 정한다:
            #   - 최소 512: 너무 작아 답이 끊기지 않도록 하한
            #   - base_max_tokens: 모드/설정이 정한 상한(예: KNOWLEDGE_MODE 2048)
            #   - dynamic_max: 컨텍스트에 실제로 남은 공간
            # 즉 "남은 공간 안에서, 설정 상한을 넘지 않되, 최소 512은 보장"한다.
            max_tokens = max(512, min(base_max_tokens, dynamic_max))

        # ═══════════════════════════════════════
        # Phase 2: API Call (모델 스트리밍)
        # ═══════════════════════════════════════
        # model_provider.stream()을 호출하고 이벤트를 수집/yield한다.
        # 동시에 StreamingToolExecutor로 도구를 미리 실행한다.

        # 이번 턴 동안 스트림에서 수집할 정보들. 매 턴 새로 초기화한다.
        assistant_text_parts: list[str] = []  # TEXT_DELTA 누적
        tool_use_blocks: list[dict[str, Any]] = []  # 완성된 tool_use 블록
        turn_usage = TokenUsage()  # 이번 턴의 토큰 사용량
        stop_reason: StopReason | None = None  # 모델 종료 이유
        model_error: str | None = None  # 모델 에러 메시지
        # ERROR 이벤트 기반 복구가 발동했는지 표시하는 플래그(감사 Critical #8).
        # Tier 3가 예외 대신 ERROR StreamEvent로 컨텍스트 초과를 내려보낸 경우,
        # Phase 2 안에서 헬퍼로 복구한 뒤 이 플래그를 세워, except/Phase 3를 건너뛰고
        # 곧장 다음 턴으로 재시도(continue)하게 한다.
        error_event_recovered = False

        # StreamingToolExecutor 생성 — 스트리밍 중 도구를 병렬 실행
        # 모델이 응답을 다 만들 때까지 기다리지 않고, tool_use가 완성되는 즉시
        # 실행을 시작해 지연(latency)을 줄인다. 매 턴 새 인스턴스를 쓴다.
        streaming_executor = StreamingToolExecutor(
            tools=tools,
            context=context,
        )

        try:
            # Tier 3 호출: model_provider.stream()
            # StreamWatchdog (Ch.7.2)로 감싸서 스트림 무응답을 감지한다.
            # idle 30초, total 300초 초과 시 StreamWatchdogTimeout 발생 → 재시도
            from core.orchestrator.stream_watchdog import stream_with_watchdog

            # v7.0 Part 2.5: 라우팅에서 결정된 model_override/temperature/
            # enable_thinking을 Tier 3로 전달한다. None/기본값인 경우 프로바이더의
            # 기본 설정(config.primary_model, temp=0.7, thinking=False)이 적용된다.
            _raw_stream = model_provider.stream(
                messages=api_messages,
                system_prompt=system_prompt,
                tools=tool_schemas if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
                model_override=model_override,
                enable_thinking=enable_thinking,
                # 샘플링 파라미터를 Tier 3로 전달 — 누락 시 degeneration(무한 반복)
                # 결함이 발생하므로 항상 함께 흘려보낸다.
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            async for event in stream_with_watchdog(
                _raw_stream,
                idle_timeout=30.0,
                total_timeout=300.0,
            ):
                # 이벤트를 UI로 전파 (먼저 상위 Tier로 흘려보낸 뒤 내부 처리)
                yield event

                # 이벤트 처리 — type이 enum이거나 문자열일 수 있음
                # 프로바이더 구현에 따라 둘 중 무엇이 와도 동작하도록 .value로 정규화한다.
                event_type = event.type if isinstance(event.type, str) else event.type.value

                if event_type == StreamEventType.TEXT_DELTA.value:
                    # 텍스트 조각 누적
                    if event.text:
                        assistant_text_parts.append(event.text)

                elif event_type == StreamEventType.TOOL_USE_STOP.value:
                    # 도구 호출 완성 — tool_use_blocks에 추가하고
                    # StreamingToolExecutor에도 전달하여 병렬 실행 시작
                    if event.tool_use:
                        tu_dict = {
                            "id": event.tool_use.id,
                            "name": event.tool_use.name,
                            "input": event.tool_use.input,
                        }
                        tool_use_blocks.append(tu_dict)
                        streaming_executor.add_tool(tu_dict)

                elif event_type == StreamEventType.MESSAGE_STOP.value:
                    # 모델 응답 종료
                    if event.stop_reason:
                        stop_reason = (
                            event.stop_reason
                            if isinstance(event.stop_reason, StopReason)
                            else StopReason(event.stop_reason)
                        )
                    if event.usage:
                        turn_usage = event.usage

                elif event_type == StreamEventType.USAGE_UPDATE.value:
                    # 사용량 업데이트
                    if event.usage:
                        turn_usage = event.usage

                elif event_type == StreamEventType.ERROR.value:
                    # 모델 에러 (스트리밍 중)
                    model_error = event.message

                    # ─── 감사 Critical #8: ERROR 이벤트 경로 복구 배선 ───
                    # Tier 3(inference)는 컨텍스트 초과/HTTP 오류를 raise하지 않고
                    # ERROR 이벤트로 내려보내므로, 여기서도 except 블록과 동일한
                    # 복구(긴급 압축/반응적 압축/OOM)를 시도해야 한다. 예전에는
                    # CONTEXT_OVERFLOW를 무조건 즉시 종료해 복구가 도달불가였다.
                    recovery_text = _error_event_to_recovery_text(event)
                    if recovery_text is not None:
                        outcome = await _try_recover_from_model_error(
                            recovery_text,
                            state,
                            context_manager,
                            streaming_executor,
                        )
                        if outcome.recovered:
                            # 헬퍼가 돌려준 이벤트(예: OOM 경고)를 호출부가 yield —
                            # 4-Tier yield 흐름을 유지하기 위해 여기서 흘려보낸다.
                            for _ev in outcome.events:
                                yield _ev
                            # 복구 성공 → 스트림 소비 중단하고 다음 턴 재시도.
                            # (cancel_all은 헬퍼가 이미 호출함)
                            error_event_recovered = True
                            break

                    # 복구 대상이 아니거나(현행 유지 오류) 재시도 예산이 소진된 경우:
                    # 컨텍스트 초과는 기존과 동일하게 즉시 종료(abort)한다.
                    if event.error_code == "CONTEXT_OVERFLOW":
                        await streaming_executor.cancel_all()
                        return

                # 스트리밍 중 완료된 도구 결과를 소비
                # add_tool()로 미리 돌려둔 도구 중 끝난 것이 있으면 즉시 결과를
                # 흘려보내, 모델 응답과 도구 실행을 겹쳐 전체 시간을 단축한다.
                for completed in streaming_executor.get_completed():
                    yield completed

        # ── 스트림 도중 발생한 예외를 종류별로 분기 처리 ──
        # 일부는 복구(압축/재시도) 후 continue로 다음 턴을, 일부는 return으로 종료한다.
        except Exception as e:
            error_name = type(e).__name__

            # ─── StreamWatchdog 타임아웃 → 재시도 ───
            # GPU 행(hang), vLLM 데드락, 네트워크 끊김을 감지한다.
            # STREAM_STALL로 분류되어 재시도 대상이 된다.
            from core.orchestrator.stream_watchdog import StreamWatchdogTimeout

            if isinstance(e, StreamWatchdogTimeout):
                state.model_error_count += 1
                if state.model_error_count <= MAX_MODEL_ERROR_RETRY:
                    logger.warning(
                        "스트림 타임아웃 (%s), 재시도 %d/%d",
                        e.timeout_type,
                        state.model_error_count,
                        MAX_MODEL_ERROR_RETRY,
                    )
                    yield StreamEvent(
                        type=StreamEventType.SYSTEM_WARNING,
                        message=(
                            f"[스트림 {e.timeout_type} 타임아웃] "
                            f"재시도 {state.model_error_count}/{MAX_MODEL_ERROR_RETRY}"
                        ),
                    )
                    state.continue_reason = ContinueReason.NEXT_TURN
                    await streaming_executor.cancel_all()
                    continue

            # ─── Transition 1/2 + OOM: 복구 헬퍼로 위임 ───
            # 예외 메시지(str(e))를 그대로 헬퍼에 넘긴다. 헬퍼는 예전에 이 자리에
            # 인라인으로 있던 세 복구(긴급 압축/반응적 압축/OOM)를 비트 단위로 동일하게
            # 수행한다(무회귀). recovered=True면 헬퍼가 돌려준 이벤트를 yield하고
            # 다음 턴으로 재시도(continue)한다.
            outcome = await _try_recover_from_model_error(
                str(e), state, context_manager, streaming_executor
            )
            if outcome.recovered:
                for _ev in outcome.events:
                    yield _ev
                continue

            # 복구 불가능한 에러
            logger.error(f"복구 불가능한 모델 에러: {error_name}: {e}")
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error_code=error_name,
                message=str(e),
            )
            await streaming_executor.cancel_all()
            return

        # ─── 감사 Critical #8: ERROR 이벤트 경로 복구 후 다음 턴 재시도 ───
        # Phase 2에서 provider가 raise 대신 ERROR 이벤트로 컨텍스트 초과를 내려보내
        # 헬퍼로 복구한 경우, except 블록을 거치지 않으므로 여기서 continue한다.
        # (예외 경로는 except 안에서 이미 continue/return으로 처리됨)
        if error_event_recovered:
            continue

        # ═══════════════════════════════════════
        # Phase 3: Post-API 처리
        # ═══════════════════════════════════════
        # 사용량 추적, 에러 복구, 종료 판단을 수행한다.

        # 사용량 누적
        state.cumulative_usage = state.cumulative_usage + turn_usage

        yield StreamEvent(
            type=StreamEventType.USAGE_UPDATE,
            usage=state.cumulative_usage,
        )

        # 모델 에러 처리 (도구 호출 JSON 파싱 실패 등)
        # 조건이 "에러 O + 도구 블록 X"인 이유: 도구가 하나라도 정상 파싱됐다면
        # 그 도구를 실행해 진행할 수 있으므로 재시도 대상에서 제외한다. 도구를
        # 전혀 못 건진 경우에만 같은 입력으로 모델을 다시 불러본다.
        if model_error and not tool_use_blocks:
            state.tool_parse_retry_count += 1
            if state.tool_parse_retry_count <= MAX_TOOL_PARSE_RETRY:
                # 재시도 — system 메시지를 추가하지 않는다 (토큰 증가 방지)
                state.continue_reason = ContinueReason.NEXT_TURN
                yield StreamEvent(
                    type=StreamEventType.SYSTEM_WARNING,
                    message=(
                        f"[도구 파싱 재시도 {state.tool_parse_retry_count}/{MAX_TOOL_PARSE_RETRY}]"
                    ),
                )
                await streaming_executor.cancel_all()
                continue
            else:
                # 재시도 모두 실패 — 사용자에게 에러 메시지 전달 후 종료
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code="CONTEXT_OVERFLOW",
                    message=(
                        "입력 내용이 너무 길어 분석할 수 없습니다. "
                        "더 짧은 내용으로 다시 시도해 주세요."
                    ),
                )
                return

        # assistant 메시지 기록
        # 스트림으로 조각조각 받은 텍스트를 하나로 합치고, 이번 턴에 모델이 요청한
        # tool_use 블록을 함께 묶어 하나의 assistant 메시지로 만든다.
        # 대화 히스토리(state.messages)에 추가해야 다음 턴에서 모델이 맥락을 본다.
        assistant_text = "".join(assistant_text_parts)
        assistant_msg = Message.assistant(
            text=assistant_text,
            tool_uses=tool_use_blocks if tool_use_blocks else None,
        )
        state.messages.append(assistant_msg)
        yield assistant_msg

        state.last_stop_reason = stop_reason

        # TurnState 생성 — v7.0 상태 외부화
        # 매 턴 끝에 핵심 정보를 추출하여 콜백으로 전달한다.
        # QueryEngine이 이 상태를 TurnStateStore에 저장한다.
        if on_turn_complete is not None:
            from core.orchestrator.turn_state import extract_turn_state

            # 도구 결과 요약 수집
            # 각 tool_result를 앞 100자만 잘라 요약 리스트로 모은다(상태 저장 비용 절감).
            # role은 enum/문자열 양쪽 모두 올 수 있어 .value로 정규화해 비교한다.
            _tool_result_summaries: list[str] = []
            for msg in state.messages:
                role = msg.role if isinstance(msg.role, str) else msg.role.value
                if role == "tool_result":
                    content = msg.text_content if hasattr(msg, "text_content") else str(msg.content)
                    _tool_result_summaries.append(content[:100])

            # 사용자 요청 추출 (messages에서 마지막 user 메시지)
            # 뒤에서부터 훑어 가장 최근 user 발화를 찾는다 — 이번 턴이 응답하려는
            # 실제 요청이 무엇인지 TurnState에 남기기 위함이다.
            _user_req = ""
            for msg in reversed(state.messages):
                role = msg.role if isinstance(msg.role, str) else msg.role.value
                if role == "user":
                    _user_req = (
                        msg.text_content if hasattr(msg, "text_content") else str(msg.content)
                    )
                    break

            turn_state = extract_turn_state(
                turn_number=state.turn_count,
                user_request=_user_req,
                assistant_text=assistant_text,
                tool_use_blocks=tool_use_blocks,
                tool_results=_tool_result_summaries[-5:],  # 최근 5개만
            )
            on_turn_complete(turn_state)

        # 턴 종료 이벤트
        yield StreamEvent(type=StreamEventType.STREAM_REQUEST_END)

        # ─── 종료 판단 ───
        # StopResolver로 도구 호출 유무를 확인
        # should_continue가 True면(=실행할 도구가 남음) 아래 종료 블록을 건너뛰고
        # 곧장 Phase 4(도구 실행) → 다음 턴으로 간다. False면 종료 후보로 보고,
        # 그 전에 Transition 3~6(이어쓰기 복구·Hook 차단 등)을 차례로 검사한다.
        if not stop_resolver.should_continue(state, tool_use_blocks):
            # 도구 호출이 없음 → 종료 후보

            # ─── Transition 3: max_output_tokens_escalate ───
            # max_tokens로 종료되었고 응답이 잘린 것 같으면
            # 출력 토큰 한도를 증가시켜 재시도
            if stop_reason == StopReason.MAX_TOKENS:
                if _seems_truncated(assistant_text):
                    state.max_output_recovery_count += 1
                    if state.max_output_recovery_count == 1:
                        # 첫 번째 시도: 토큰 한도 증가
                        # 현재 max_tokens가 에스컬레이션 단계(4K/8K/16K) 중 하나면
                        # 그 다음 단계로, 아니면 0번(4K)부터 시작한다.
                        current_idx = (
                            escalation_steps.index(max_tokens)
                            if max_tokens in escalation_steps
                            else 0
                        )
                        # 마지막 단계(16K)를 넘지 않도록 min으로 상한을 건다.
                        next_idx = min(
                            current_idx + 1,
                            len(escalation_steps) - 1,
                        )
                        # override를 세팅하면 다음 턴 Phase 1에서 동적 계산 대신
                        # 이 값을 그대로 max_tokens로 사용한다.
                        state.max_output_tokens_override = escalation_steps[next_idx]
                        state.continue_reason = ContinueReason.MAX_OUTPUT_TOKENS_ESCALATE
                        logger.info(
                            f"출력 토큰 에스컬레이션: "
                            f"{max_tokens} → {state.max_output_tokens_override}"
                        )
                        await streaming_executor.cancel_all()
                        continue

                    # ─── Transition 4: max_output_tokens_recovery ───
                    # 에스컬레이션 후에도 잘리면 "이어서 써주세요" 메시지로 복구
                    elif state.max_output_recovery_count <= MAX_OUTPUT_RECOVERY:
                        state.messages.append(
                            Message.user(
                                "응답이 중간에 잘렸습니다. 중단된 부분부터 이어서 작성해주세요."
                            )
                        )
                        state.continue_reason = ContinueReason.MAX_OUTPUT_TOKENS_RECOVERY
                        logger.info(
                            f"멀티턴 이어쓰기 복구 "
                            f"({state.max_output_recovery_count}/{MAX_OUTPUT_RECOVERY})"
                        )
                        await streaming_executor.cancel_all()
                        continue

            # ─── Transition 5: stop_hook_blocking ───
            # HookManager가 있으면 STOP 이벤트를 실행하여
            # 종료를 차단할 수 있는지 확인한다
            if hook_manager is not None:
                try:
                    from core.hooks.hook_manager import HookDecision, HookEvent, HookInput

                    hook_input = HookInput(
                        event=HookEvent.STOP,
                        metadata={
                            "stop_reason": str(stop_reason) if stop_reason else None,
                            "turn_count": state.turn_count,
                            "assistant_text": assistant_text[:200],
                        },
                    )
                    hook_result = await hook_manager.run(HookEvent.STOP, hook_input)
                    if hook_result.decision == HookDecision.BLOCK:
                        # Hook이 종료를 차단 → 강제 다음 턴
                        state.continue_reason = ContinueReason.STOP_HOOK_BLOCKING
                        logger.info(
                            "Hook이 종료를 차단: %s", hook_result.block_reason
                        )
                        await streaming_executor.cancel_all()
                        continue
                except Exception as e:
                    # Hook 실행 실패 시 정상 종료 진행 (fail-open)
                    logger.warning("STOP Hook 실행 실패: %s", e)

            # ─── Transition 6: token_budget_continuation ───
            # 로컬 모델은 비용이 0이므로 이 전환은 거의 발생하지 않음
            # 토큰 예산 기반 계속은 향후 필요 시 구현
            # TODO(nexus): 토큰 예산 계속 — 필요 시 구현

            # 정상 종료
            logger.info(
                f"쿼리 루프 종료: {state.turn_count}턴 "
                f"(이유: {stop_reason}, 경과: {state.elapsed_seconds:.1f}초)"
            )
            return

        # ═══════════════════════════════════════
        # Phase 4: Tool Execution (도구 실행)
        # ═══════════════════════════════════════
        # StreamingToolExecutor의 drain_remaining()으로
        # 모든 도구 실행을 완료하고 결과를 yield/기록한다.

        # 스트리밍 중 미처 끝나지 않은 도구를 모두 완료시키고 결과를 흘려보낸다.
        # 도구 결과(tool_result Message)는 다음 턴에서 모델이 읽도록 히스토리에 넣는다.
        # 이번 턴 도구 결과의 에러 여부를 tool_use_id로 수집(반복 실패 가드용).
        this_turn_errors: dict[str, bool] = {}
        async for event in streaming_executor.drain_remaining():
            yield event
            # Message 이벤트(tool_result)는 대화 히스토리에 추가
            if isinstance(event, Message):
                state.messages.append(event)
                if getattr(event, "tool_use_id", None):
                    this_turn_errors[event.tool_use_id] = bool(
                        getattr(event, "is_error", False)
                    )

        # ─── 반복 실패 도구 호출 가드 (무한 재시도 루프 방지) ───
        # 같은 도구를 같은 입력으로 계속 호출하며 매번 실패하면(예: 깨진 Bash
        # 명령을 못 고치고 반복) 시간만 태운다. 연속 실패가 WARN 임계에 닿으면
        # 모델에 "반복하지 말라"는 피드백을 1회 주입하고, ABORT를 넘으면 강제
        # 종료한다. 성공한 호출은 헬퍼가 서명별로 카운터를 리셋한다.
        worst_streak = _update_tool_failure_streak(
            tool_use_blocks, this_turn_errors, tool_failure_streak
        )
        # 리셋/제거된 서명은 경고 이력에서도 지워, 같은 명령이 나중에 재발하면
        # 다시 한 번 경고할 수 있게 한다(살아있는 streak 서명과 동기화).
        warned_sigs.intersection_update(tool_failure_streak.keys())
        if worst_streak >= REPEATED_TOOL_FAILURE_ABORT:
            logger.warning(
                "동일 도구 호출이 %d턴 연속 실패 — 턴 루프 강제 종료", worst_streak
            )
            yield StreamEvent(
                type=StreamEventType.SYSTEM_WARNING,
                message=(
                    f"[반복 실패 중단] 동일한 도구 호출이 {worst_streak}턴 연속 "
                    "실패하여 중단합니다. 다른 방법으로 다시 시도해 주세요."
                ),
            )
            return
        # WARN 임계를 '처음' 넘은 서명에만 1회 피드백을 주입한다(warned_sigs로
        # 중복 주입 차단 — 같은 실패에 매 턴 경고가 반복되지 않게). user 역할로
        # 넣어 모델이 다음 턴에 확실히 읽고 반복을 멈추도록 유도한다.
        newly_warned = [
            sig
            for sig, cnt in tool_failure_streak.items()
            if cnt >= REPEATED_TOOL_FAILURE_WARN and sig not in warned_sigs
        ]
        if newly_warned:
            warned_sigs.update(newly_warned)
            yield StreamEvent(
                type=StreamEventType.SYSTEM_WARNING,
                message=f"[반복 실패 경고] 동일 도구 호출 {worst_streak}턴 연속 실패",
            )
            state.messages.append(
                Message.user(
                    "[시스템] 방금 같은 도구 호출이 여러 번 연속 실패했습니다. "
                    "똑같은 명령을 그대로 반복하지 마세요. 명령이나 입력을 바꾸거나, "
                    "도구 없이 지금까지의 정보로 답변을 완성하세요."
                )
            )

        # ─── Transition 7: next_turn ───
        # 도구 실행 완료 → 정상적으로 다음 턴 진행
        # 이번 턴이 정상적으로 한 바퀴를 마쳤으므로, 에러 상황에서만 의미 있는
        # 복구/재시도 카운터를 0으로 되돌린다. 그래야 다음에 같은 에러가 나도
        # 누적치가 아니라 처음부터 다시 재시도 한도를 쓸 수 있다.
        state.continue_reason = ContinueReason.NEXT_TURN
        state.max_output_recovery_count = 0  # 복구 카운터 리셋
        state.tool_parse_retry_count = 0  # 파싱 재시도 카운터 리셋

    # max_turns 도달 — 무한 루프 방지
    logger.warning(f"쿼리 루프 최대 턴 수 도달 ({max_turns})")
    yield StreamEvent(
        type=StreamEventType.SYSTEM_WARNING,
        message=f"[경고] 최대 턴 수({max_turns})에 도달했습니다. 중단합니다.",
    )
