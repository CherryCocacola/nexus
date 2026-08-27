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
    STREAM_DISCARD_RETRY,
    STREAM_TRUNCATED,
    Message,
    Role,
    StopReason,
    StreamEvent,
    StreamEventType,
    TokenUsage,
)
from core.model.inference import ModelProvider, StructuredOutputSpec
from core.orchestrator.self_consistency import resolve_consensus
from core.orchestrator.stop_resolver import StopResolver, _seems_truncated
from core.orchestrator.stream_handler import StreamingToolExecutor
from core.tools.base import BaseTool, ToolUseContext

logger = logging.getLogger("nexus.orchestrator.query_loop")


def _chunk_text(text: str, size: int) -> list[str]:
    """텍스트를 size 글자 단위로 잘라 리스트로 돌려준다(SC 승자 의사-스트림용).

    합의로 확정한 승자 텍스트를 한 번에 던지지 않고 조각내어 TEXT_DELTA로
    연속 yield하면, 상위 소비자(웹 SSE/CLI Rich)가 일반 턴과 구분 없이 자연스럽게
    렌더링한다(설계 §6.2). size 이하이거나 빈 문자열이면 통째로 담는다.
    """
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


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
    # ★도구별 파싱 실패 **누적** 횟수 (2026-08-23). 위 tool_parse_retry_count 와
    #   달리 정상 턴에도 리셋하지 않는다.
    #   왜 따로 필요한가(실측): [파싱 실패 턴 → 성공 턴 → 파싱 실패 턴 …] 교대
    #   패턴에서, Transition 7(정상 턴)이 tool_parse_retry_count 를 0 으로 되돌려
    #   카운터가 1↔0 을 오가며 한도에 영영 도달하지 못했다. 그 결과 같은 문서를
    #   11회 다시 쓰며 30턴·10분을 공전했다. 누적 카운터가 그 사각을 메운다.
    tool_parse_fail_total: dict[str, int] = field(default_factory=dict)
    # STOP 훅이 이 세션에서 종료를 막은 횟수. 상한 없이 두면 모델이 계속 같은
    # 주장을 반복할 때 무한 루프가 된다(builtin_hooks.MAX_FILE_CLAIM_BLOCKS 참조).
    stop_hook_block_count: int = 0
    # 도구별 넛지를 이미 보냈는가(도구당 1회만 — 매 턴 잔소리하면 컨텍스트만 먹는다).
    parse_nudge_sent: set[str] = field(default_factory=set)
    # guided 재시도용 — 도구 인자 JSON 파싱이 실패한 도구 이름. 다음 턴 stream()에
    # tool_choice로 강제 전달돼 vLLM guided decoding으로 유효 JSON을 받게 한다.
    # 한 턴만 유효(one-shot): stream() 호출 직후 None으로 리셋한다.
    force_tool_choice: str | None = None
    compact_retry_count: int = 0
    collapse_drain_count: int = 0
    # degeneration 재생성 예산(2026-08-13). 붕괴로 스트림이 조기 절단됐을 때 샘플링을
    # 바꿔 한 번만 다시 만든다. 붕괴 없이 스트림이 끝나면 0으로 되돌려, "붕괴 1건당
    # 1회"가 되게 한다(누적 소진으로 이후 붕괴가 무방비가 되는 것을 막는다).
    degen_retry_count: int = 0
    # one-shot 플래그 — 다음 stream() 호출에만 보정 샘플링을 적용하고 즉시 내린다.
    # (force_tool_choice 와 같은 규약. 안 내리면 이후 모든 턴의 온도가 올라간다.)
    degen_retry_pending: bool = False

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
# 도구별 파싱 실패 **누적** 임계 (2026-08-23). 연속이 아니라 누적으로 세는 이유는
# LoopState.tool_parse_fail_total 주석 참조 — 교대 패턴이 연속 카운터를 빠져나간다.
PARSE_FAIL_NUDGE_AT = 3   # 이 횟수에 도달하면 행동 지시를 한 번 주입한다
PARSE_FAIL_ABORT_AT = 6   # 이 횟수에 도달하면 정직한 에러로 종료한다(무한 공전 차단)
MAX_MODEL_ERROR_RETRY = 3  # 모델 에러 재시도 횟수
MAX_COMPACT_RETRY = 2  # prompt-too-long 압축 재시도 횟수
# 생성 중 degeneration(동일라인 반복·문자샐러드·이모지 폭주) 감지 시 스트림 조기 절단.
# 페널티 완화로 런어웨이는 제거됐으나 자체종료 내부 붕괴(~17%)가 잔존해(2026-07-20
# 품질 114건 실증) 스트리밍 워치독에 붕괴 감지·절단을 켠다. 임계값은 stream_watchdog의
# DegenerationMonitor 기본값(정상 표/목록/코드는 통과하도록 넉넉히 설정).
DEGEN_GUARD_ENABLED = True  # False로 두면 종전 동작(감지 없음)과 100% 동일(무회귀)

# ── 붕괴 감지 후 재생성 (2026-08-13) ─────────────────────────────
# 종전에는 붕괴를 감지하면 **자르고 끝**이었다. 실측(2026-08-13 14:33): 8,944자를
# 만들고 잘린 뒤 그 상태로 사용자에게 나갔고, 구조화 출력 요청이라 JSON 이 깨져
# finish_reason 까지 오염됐다. 잘린 쓰레기를 그대로 주느니 한 번 다시 만든다.
#
# 왜 하필 temperature 를 올리나 — 이 리포의 실측이 두 페널티를 모두 배제한다.
#   · repetition_penalty 1.15 → 종결어미·조사까지 억눌러 EOS 붕괴(2026-07-20 사고)
#   · frequency_penalty 0.3   → 등장횟수 선형비례라 긴 생성일수록 억압이 누적(붕괴 주범,
#                               그래서 knowledge_mode 에서 0.3→0.1 로 내렸다)
# 즉 붕괴는 긴 생성에서 나는데 두 페널티는 긴 생성에서 되레 붕괴를 키운다. 반면
# temperature 상향은 반복 끌개(attractor)에서 빠져나오게 하면서 누적 억압이 없다.
# ★이 값은 아직 실측되지 않았다. 재생성 자체의 효과와 함께 측정 대상이다.
DEGEN_MAX_RETRY = 1  # 붕괴 1건당 재생성 횟수. 2 이상은 지연이 배로 늘어 사용자가 먼저 떠난다
DEGEN_RETRY_TEMPERATURE_DELTA = 0.2  # 재생성 시 온도 가산
DEGEN_RETRY_TEMPERATURE_MAX = 0.9  # 가산 후 상한(그 이상은 사실성이 흔들린다)
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


def _successful_write_tools_since_user(messages: list[Any]) -> list[str]:
    """마지막 사용자 메시지 이후 **성공한** 쓰기 도구 이름을 모은다 (STOP 훅 판정용).

    왜 "마지막 사용자 메시지 이후"인가: 모델이 세 턴 전에 파일을 쓰고 지금은 그
    작업을 요약하는 중일 수 있다. 턴 하나만 보면 그런 정상 케이스를 오탐한다.
    한 요청 단위로 봐야 "이번 요청에서 파일을 만들었나"에 답할 수 있다.

    왜 "성공"인가: 권한 거부나 에러로 실패한 Write 를 실행으로 치면, 정작 잡아야 할
    케이스(호출은 했는데 실패했고 그런데도 "작성했습니다"라고 하는 경우)가 빠진다.

    Args:
        messages: 대화 메시지 전체(state.messages).

    Returns:
        성공한 쓰기 도구 이름 목록. 없으면 빈 목록.
    """
    # 마지막 user 메시지 위치를 찾는다. 없으면 전체를 본다.
    start = 0
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].role
        role_value = role if isinstance(role, str) else getattr(role, "value", "")
        if role_value == "user":
            start = idx
            break

    try:
        from core.verification.file_claim import collect_successful_write_tools

        return sorted(collect_successful_write_tools(messages[start:]))
    except Exception:  # noqa: BLE001 — 판정 실패로 루프를 막지 않는다(fail-open)
        logger.debug("쓰기 도구 수집 실패", exc_info=True)
        return []


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


# 이 글자 수 아래면 어떤 문자 구성이라도 컨텍스트를 위협하지 못하므로 세지 않는다.
# 가장 비싼 문자(한자 1.3333 토큰/글자)로 쳐도 5300 토큰 남짓이라, 어떤 티어의
# 컨텍스트에서도 truncate·출력 축소 판단에 영향을 주지 않는다. 짧은 대화가 대부분인
# 표면(웹 채팅)에서 불필요한 왕복을 없애기 위한 것이다.
EXACT_COUNT_MIN_CHARS = 4000


async def _measure_input_tokens(
    model_provider: Any,
    api_messages: list[Message],
    system_prompt: str,
    tool_schemas: list[dict[str, Any]],
    tool_schema_texts: list[str],
    total_chars: int,
) -> tuple[int, bool]:
    """이번 요청의 입력 토큰 수를 구한다 — 되도록 **세고**, 안 되면 추정한다 (2026-08-14).

    왜 두 경로인가:
      정확한 값은 vLLM `/tokenize` 만 알 수 있다(채팅 템플릿·도구 스키마 주입까지
      반영). 하지만 그 서버가 없거나(테스트·다른 프로바이더) 실패할 수 있으므로
      문자 종류별 폴백 추정을 함께 둔다. 폴백은 **과소추정만 안 하면** 되도록
      넉넉하게 잡혀 있다(core/model/token_estimate.py 참고).

    Returns:
        (토큰 수, 실측 여부). 실측 여부는 로그·판단 근거 표시에만 쓴다.
    """
    counter = getattr(model_provider, "count_prompt_tokens", None)
    if counter is not None and total_chars >= EXACT_COUNT_MIN_CHARS:
        try:
            exact = await counter(api_messages, system_prompt, tool_schemas or None)
        except Exception as e:  # noqa: BLE001 — 세기 실패가 턴을 막지 않게 한다
            logger.debug("[tokenize] 예외(폴백 추정 사용): %s", e)
            exact = None
        if isinstance(exact, int) and exact > 0:
            return exact, True

    from core.model.token_estimate import estimate_prompt_tokens

    estimated = estimate_prompt_tokens(
        [str(m.content) for m in api_messages],
        system_prompt=system_prompt,
        tool_schema_texts=tool_schema_texts,
    )
    return estimated, False


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
                # 결과를 채택했으므로 경계를 되돌린다 — 남겨 두면 다음 턴에
                # 짧아진 리스트에 옛 인덱스가 다시 적용돼 한 번 더 잘린다.
                context_manager.mark_result_adopted()
                # 긴급 압축이 실제로 줄였으면 CONTEXT_COMPACT를 UI에 흘려보낸다.
                _compact_ev = _compaction_event(context_manager)
                if _compact_ev is not None:
                    events.append(_compact_ev)
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
                # 위 emergency_compact 와 같은 이유로 경계를 되돌린다.
                context_manager.mark_result_adopted()
                # 반응적 압축이 실제로 줄였으면 CONTEXT_COMPACT를 UI에 흘려보낸다.
                _compact_ev = _compaction_event(context_manager)
                if _compact_ev is not None:
                    events.append(_compact_ev)
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


def _compaction_event(context_manager: Any | None) -> StreamEvent | None:
    """
    context_manager가 직전 호출에서 "실제로" 대화를 줄였으면, 그 요약 문구를 담은
    CONTEXT_COMPACT StreamEvent를 1회 만들어 돌려준다(없으면 None).

    [왜 별도 헬퍼인가]
      압축 호출 지점이 세 곳(매 턴 apply_all/auto_compact, 긴급 압축, 반응적 압축)이라
      "실제 압축 여부를 꺼내(take) 이벤트로 변환"하는 로직을 한곳에 모아 중복을 없앤다.

    [무회귀·방어]
      context_manager는 타입이 Any(모의객체·구버전 가능)이므로 take_last_compaction
      훅이 없으면 조용히 None을 반환한다. no-op 통과(압축 없음)에서도 None을 반환해
      매 턴 표시가 뜨지 않도록 한다(anti-patterns #3: StreamEvent는 새 인스턴스 생성).
    """
    if context_manager is None:
        return None
    take = getattr(context_manager, "take_last_compaction", None)
    if take is None:
        return None
    phrase = take()
    if not phrase:
        return None
    return StreamEvent(type=StreamEventType.CONTEXT_COMPACT, message=phrase)


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
    request_id: str | None = None,
    session_id: str = "",
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
    # 구조화 출력(guided decoding) 스펙. None이면 일반 에이전트 턴(무회귀).
    # 지정되면 도구를 노출하지 않고(1차 방어) Tier 3로 그대로 전달한다.
    structured_output: StructuredOutputSpec | None = None,
    # 자기일관성(Self-Consistency) 파라미터 (Point 4.3). sc_n=1(기본)이면 SC 로직을
    # 통째로 우회하므로 기존 동작이 1비트도 바뀌지 않는다(무회귀). sc_n>1이면
    # Tier 3에 n=sc_n을 전달해 표본 N개를 받고, 합의(다수결/임베딩 클러스터)로 승자를
    # 확정한 뒤 승자만 의사-스트림한다. structured_output이 지정되면 SC는 비활성이다
    # (guided decoding과 상호 배타 — 구조화 응답에 다중표본 합의는 부적합).
    sc_n: int = 1,
    sc_min_agreement: int = 2,
    sc_short_answer_max_chars: int = 80,
    sc_similarity_threshold: float = 0.90,
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
        structured_output: 구조화 출력(guided decoding) 스펙 (기본 None). 지정되면
            (1) 이번 루프에서 도구 스키마를 모델에 노출하지 않고(1차 방어 —
            Tier 3의 tools 상호배타 ValueError를 애초에 유발하지 않음),
            (2) model_provider.stream()에 그대로 전달해 응답 JSON 문법을 강제한다.
            AgentHub 단발 호출·내부 추출 파이프라인 같은 "구조화 응답 전용" 경로용.

    Yields:
        StreamEvent: 스트리밍 이벤트 (UI 업데이트용)
        Message: assistant/tool_result 메시지 (대화 히스토리용)
    """
    # 루프 상태 초기화
    # state: 턴 카운트·토큰 누적·각종 재시도 카운터를 한 객체에 모아 관리한다.
    # stop_resolver: 매 턴 끝에서 "도구 호출이 남았는지"를 판정해 계속/종료를 결정한다.
    state = LoopState(messages=messages)

    # GPU OOM 복구가 줄여 놓은 컨텍스트 상한을 요청 경계에서 되돌린다(2026-08-26).
    # 축소(max_tokens *= 0.7)에는 복원 코드가 없어, CLI 처럼 관리자가 프로세스
    # 수명 내내 사는 표면에서는 OOM 이 날 때마다 0.7ⁿ 로 누적 축소됐다. 축소의
    # 목적은 "이번 요청을 통과시키는 것"이므로 다음 요청은 설정값에서 시작한다.
    # hasattr 로 방어하지 않는다 — 메서드가 없으면 복원이 조용히 건너뛰어져 고치려던
    # 버그가 그대로 남는다. 없으면 시끄럽게 실패하는 편이 낫다.
    if context_manager is not None:
        context_manager.restore_max_tokens()
    stop_resolver = StopResolver()

    # ── 자기일관성(SC) 활성 여부 (Point 4.3) ──────────────────────────────
    # sc_n>1이고 구조화 출력이 아닐 때만 SC 경로를 켠다. structured_output과는
    # 상호 배타(둘 다 응답 본문 문법을 다투므로). sc_active=False면 아래 SC 분기는
    # 전부 죽은 코드가 되어 기존 단일 경로와 완전히 동일하게 동작한다(무회귀).
    sc_active = structured_output is None and sc_n > 1

    # 출력 토큰 에스컬레이션 단계 확정 — 주입값 우선, 미주입 시 모듈 상수 폴백.
    # (하드코딩 외부화 무회귀: config 미주입 경로/기존 테스트는 상수 그대로 사용)
    escalation_steps = output_token_escalation or OUTPUT_TOKEN_ESCALATION

    # 클라이언트가 실행하는 도구 이름 집합(2026-08-08).
    #   OpenAI `tools` 규약으로 호출자가 자기 도구를 선언한 경우, 그 실행 주체는
    #   서버가 아니라 호출자다(예: VSCode 플러그인이 개발자 PC 의 파일을 읽는다).
    #   서버가 대신 실행하면 ①엉뚱한 파일시스템을 만지고 ②테넌트 간 유출 경로가 된다.
    #   여기서 이름을 미리 뽑아 두고, 아래에서 실행 대기열 진입을 막는다.
    client_tool_names = {
        t.name for t in tools if getattr(t, "is_client_executed", False)
    }

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

            # 이번 턴 압축이 "실제로" 대화를 줄였으면 UI에 CONTEXT_COMPACT를 1회
            # 흘려보낸다("🗜️ 대화 압축 중 · <요약>"). 임계치 미달 no-op 통과에서는
            # _compaction_event가 None을 반환해 아무것도 표시하지 않는다(매 턴 방지).
            _compact_ev = _compaction_event(context_manager)
            if _compact_ev is not None:
                yield _compact_ev

        # 1b. 도구 스키마 준비 (이름순 정렬 — prompt cache 안정성)
        # 구조화 출력 모드에서는 도구를 아예 노출하지 않는다(1차 방어).
        # 왜: response_format(guided decoding)과 tools는 vLLM에서 상호 배타이며,
        # Tier 3(stream)가 둘 다 오면 ValueError를 던진다. 여기서 도구 스키마를
        # 비우면 그 예외를 애초에 유발하지 않고, 도구 스키마가 프롬프트 앞부분에
        # 들어가지 않아 구조화 응답 전용 라인이 깔끔하게 유지된다.
        tool_schemas = [] if structured_output is not None else [t.to_schema() for t in tools]

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
            # 입력 토큰 계산: 시스템 프롬프트 + 도구 스키마 + 메시지
            #
            # ★2026-08-14 — 추정에서 '세기'로 바꿨다.
            #   종전에는 `총글자수 // 3` 하나로 추정했다. A.X-4.0 토크나이저 실측 결과
            #   문자 종류에 따라 0.0078(공백) ~ 1.3333(한자) 토큰/글자로 100배 넘게
            #   벌어진다. 문서체 한국어를 15% 과소추정해 컨텍스트 예산을 크게 남겨
            #   둘 수밖에 없었다.
            #   이제 vLLM `/tokenize` 로 실제 프롬프트를 센다(채팅 템플릿·도구 스키마
            #   포함). 세기가 실패하면 문자 종류별 폴백 추정으로 넘어간다.
            import json as _json

            # ensure_ascii=False로 직렬화해 한글이 \uXXXX로 부풀려지지 않게 한다
            # (그래야 실제 전송될 토큰 양에 가까운 문자수를 얻는다).
            tool_schema_texts = [_json.dumps(s, ensure_ascii=False) for s in tool_schemas]
            tool_chars = sum(len(t) for t in tool_schema_texts)
            msg_chars = sum(len(str(m.content)) for m in api_messages)
            prompt_chars = len(system_prompt)
            total_chars = tool_chars + msg_chars + prompt_chars

            estimated_input, input_is_exact = await _measure_input_tokens(
                model_provider=model_provider,
                api_messages=api_messages,
                system_prompt=system_prompt,
                tool_schemas=tool_schemas,
                tool_schema_texts=tool_schema_texts,
                total_chars=total_chars,
            )

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
                # 자를 글자 예산은 "이번 입력의 실제 글자당 토큰"에서 역산한다.
                # 종전에는 1/3 을 가정해 `input_limit * 3` 으로 역산했는데, 그 상수가
                # 틀린 것이 이번 수정의 출발점이다. 이제는 방금 센(또는 추정한) 값에서
                #   글자당 토큰 = estimated_input / total_chars
                # 를 얻어 쓰므로, 한글 문서든 코드든 그 입력에 맞는 비율이 적용된다.
                tokens_per_char = estimated_input / max(1, total_chars)
                budget_chars = int(input_limit / max(tokens_per_char, 1e-6))
                msg_char_budget = budget_chars - (tool_chars + prompt_chars)

                # 헬퍼가 원본을 훼손하지 않고 '이번 호출용 사본 리스트'를 돌려준다.
                # (줄일 메시지는 팩토리로 새로 만들고, 나머지는 원본 참조 재사용)
                before_input = estimated_input
                api_messages, before_msg_chars, after_msg_chars = (
                    _truncate_input_for_budget(api_messages, msg_char_budget)
                )

                # 자른 뒤 입력 토큰을 **다시 센다.** 글자당 토큰은 자르는 위치에 따라
                # 달라지므로(앞부분이 한글, 뒷부분이 코드일 수 있다) 재계산이 필요하다.
                msg_chars = after_msg_chars
                total_chars = tool_chars + msg_chars + prompt_chars
                estimated_input, input_is_exact = await _measure_input_tokens(
                    model_provider=model_provider,
                    api_messages=api_messages,
                    system_prompt=system_prompt,
                    tool_schemas=tool_schemas,
                    tool_schema_texts=tool_schema_texts,
                    total_chars=total_chars,
                )

                logger.info(
                    "입력 truncate: %d → %d 토큰 (%s) "
                    "(컨텍스트 %d의 85%%=%d, 메시지 글자 %d → %d)",
                    before_input, estimated_input,
                    "실측" if input_is_exact else "추정",
                    max_context, input_limit,
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
            if state.force_tool_choice:
                # guided 재시도 턴: 긴 문서(DocumentExport content 등)를 스키마대로
                # 완결된 JSON으로 다시 쓰게 해야 하므로, 설정 상한(base_max_tokens,
                # 예: tool_mode 8192)에 막혀 절단되지 않도록 base cap을 풀고 잔여
                # 윈도우(dynamic_max) 전체를 출력에 할당한다. dynamic_max는 이미
                # max_context - 입력 - 200으로 윈도우 불변식을 지키므로 안전하다.
                max_tokens = max(512, dynamic_max)
            else:
                max_tokens = max(512, min(base_max_tokens, dynamic_max))

        # ═══════════════════════════════════════
        # Phase 2: API Call (모델 스트리밍)
        # ═══════════════════════════════════════
        # model_provider.stream()을 호출하고 이벤트를 수집/yield한다.
        # 동시에 StreamingToolExecutor로 도구를 미리 실행한다.

        # 이번 턴 동안 스트림에서 수집할 정보들. 매 턴 새로 초기화한다.
        assistant_text_parts: list[str] = []  # TEXT_DELTA 누적
        tool_use_blocks: list[dict[str, Any]] = []  # 완성된 tool_use 블록
        # 클라이언트가 실행할 도구 호출 — 서버는 모아만 두고 실행하지 않는다.
        client_tool_calls: list[dict[str, Any]] = []
        # 인자 JSON 파싱이 실패한 도구 이름들(parse_error=True). 이 도구는 빈 인자로
        # 실행하지 않고(오답 방지), 스트림 종료 후 guided 재시도 판정에 쓴다.
        parse_failed_tools: list[str] = []
        turn_usage = TokenUsage()  # 이번 턴의 토큰 사용량
        stop_reason: StopReason | None = None  # 모델 종료 이유
        model_error: str | None = None  # 모델 에러 메시지
        # SC 표본 후보 버퍼(표본 index → 완성 텍스트). sc_active일 때만 채워진다.
        # Tier 3가 SC_CANDIDATE 이벤트로 올려보낸 후보들을 여기 모아, 스트림 종료 후
        # 다수결/임베딩 클러스터 합의를 낸다(§3, §6.2).
        sc_candidates: dict[int, str] = {}
        # ERROR 이벤트 기반 복구가 발동했는지 표시하는 플래그(감사 Critical #8).
        # Tier 3가 예외 대신 ERROR StreamEvent로 컨텍스트 초과를 내려보낸 경우,
        # Phase 2 안에서 헬퍼로 복구한 뒤 이 플래그를 세워, except/Phase 3를 건너뛰고
        # 곧장 다음 턴으로 재시도(continue)하게 한다.
        error_event_recovered = False

        # ── SC 활동 표시 (Point 4.3, §6.2) ──
        # SC 모드는 전 표본을 버퍼링한 뒤 합의를 내므로, 합의 전까지 출력이 없다
        # (무출력 구간). 이때 사용자가 hang으로 오인하지 않도록 SYSTEM_INFO를 즉시
        # 흘려보내 "사실 교차 검증 중"임을 알린다. 신규 이벤트 타입 불필요(재사용).
        if sc_active:
            yield StreamEvent(
                type=StreamEventType.SYSTEM_INFO,
                message=f"사실 교차 검증 중 ({sc_n}표본)",
            )

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
            from core.orchestrator import prompt_dump
            from core.orchestrator.stream_watchdog import (
                DegenerationMonitor,
                stream_with_watchdog,
            )

            # 붕괴 감지기를 **우리가 만들어 넘긴다**. 워치독은 붕괴를 감지하면 예외가
            # 아니라 정상 종료로 스트림을 끊으므로(재시도 경로에서 붕괴가 재발하는 것을
            # 막기 위한 의도된 설계), 밖에서는 "짧게 끝났다"와 구분할 방법이 없다.
            # 같은 인스턴스를 들고 있으면 스트림이 끝난 뒤 물어볼 수 있다.
            degen_monitor = DegenerationMonitor() if DEGEN_GUARD_ENABLED else None

            # 붕괴 재생성 시도라면 이번 호출에만 온도를 올린다(상수 주석의 근거 참고).
            effective_temperature = temperature
            if state.degen_retry_pending:
                effective_temperature = min(
                    DEGEN_RETRY_TEMPERATURE_MAX,
                    temperature + DEGEN_RETRY_TEMPERATURE_DELTA,
                )

            # v7.0 Part 2.5: 라우팅에서 결정된 model_override/temperature/
            # enable_thinking을 Tier 3로 전달한다. None/기본값인 경우 프로바이더의
            # 기본 설정(config.primary_model, temp=0.7, thinking=False)이 적용된다.
            # ── 프롬프트 덤프 (기본 비활성) ─────────────────────────────
            # 여기가 진짜 최종본이다 — 프롬프트 조립·컨텍스트 압축이 모두 끝난 뒤,
            # 모델로 나가기 직전. 무상태 엔드포인트라 이 값은 어디에도 남지 않아,
            # "모델이 무엇을 봤는가"를 나중에 확인할 방법이 없었다.
            # 사내 문서 RAG 본문이 그대로 들어가므로 환경변수를 준 경우에만 켜진다.
            if prompt_dump.is_enabled():
                prompt_dump.dump_prompt(
                    request_id=request_id,
                    session_id=session_id,
                    turn=state.turn_count,
                    system_prompt=system_prompt,
                    messages=api_messages,
                    tool_names=[t.get("name", "?") for t in tool_schemas],
                    routing={
                        "model_override": model_override,
                        "max_tokens": max_tokens,
                        "structured_output": structured_output is not None,
                    },
                    sampling={
                        "temperature": effective_temperature,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                        "frequency_penalty": frequency_penalty,
                        "presence_penalty": presence_penalty,
                    },
                )

            _raw_stream = model_provider.stream(
                messages=api_messages,
                system_prompt=system_prompt,
                # tool_schemas는 구조화 출력 모드에서 []로 비워지므로 tools=None이 된다
                # (Tier 3의 tools 상호배타 ValueError를 애초에 유발하지 않음).
                tools=tool_schemas if tool_schemas else None,
                temperature=effective_temperature,
                max_tokens=max_tokens,
                model_override=model_override,
                enable_thinking=enable_thinking,
                # 샘플링 파라미터를 Tier 3로 전달 — 누락 시 degeneration(무한 반복)
                # 결함이 발생하므로 항상 함께 흘려보낸다.
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                # 구조화 출력 스펙을 Tier 3로 전달(passthrough) — None이면 일반 생성.
                structured_output=structured_output,
                # SC 표본 수 — sc_active일 때만 n>1, 아니면 1(기존 단일 경로, 무회귀).
                n=sc_n if sc_active else 1,
                # guided 재시도: 이전 턴에서 인자 파싱 실패가 감지된 도구가 있으면
                # 그 도구로 tool_choice를 강제해 vLLM이 유효 JSON을 내게 한다.
                force_tool_choice=state.force_tool_choice,
            )
            # one-shot 리셋 — 이 턴에만 강제하고, 다음 턴은 다시 auto로 돌아간다.
            # (재시도 응답도 실패하면 Phase 3에서 다시 세팅된다.)
            state.force_tool_choice = None
            # 온도 보정도 one-shot 이다 — 안 내리면 이후 모든 턴이 올라간 채로 돈다.
            state.degen_retry_pending = False
            async for event in stream_with_watchdog(
                _raw_stream,
                idle_timeout=30.0,
                total_timeout=300.0,
                detect_degeneration=DEGEN_GUARD_ENABLED,
                degen_monitor=degen_monitor,
            ):
                # 이벤트 처리 — type이 enum이거나 문자열일 수 있음
                # 프로바이더 구현에 따라 둘 중 무엇이 와도 동작하도록 .value로 정규화한다.
                event_type = event.type if isinstance(event.type, str) else event.type.value

                # ── SC 후보 버퍼링 (Point 4.3, §6.2) ──
                # SC 모드의 표본 후보(SC_CANDIDATE)는 UI로 흘리지 않고 모아둔다.
                # 합의 후 승자만 TEXT_DELTA로 의사-스트림하기 때문이다. 원문 후보 N개가
                # 그대로 노출되면 혼란스러우므로 Tier 2에서 확실히 가로챈다(yield 안 함).
                if sc_active and event_type == StreamEventType.SC_CANDIDATE.value:
                    if event.sample_index is not None:
                        sc_candidates[event.sample_index] = event.text or ""
                    continue

                # 이벤트를 UI로 전파 (먼저 상위 Tier로 흘려보낸 뒤 내부 처리)
                yield event

                if event_type == StreamEventType.TEXT_DELTA.value:
                    # 텍스트 조각 누적
                    if event.text:
                        assistant_text_parts.append(event.text)

                elif event_type == StreamEventType.TOOL_USE_STOP.value:
                    # 도구 호출 완성 — tool_use_blocks에 추가하고
                    # StreamingToolExecutor에도 전달하여 병렬 실행 시작
                    if event.tool_use:
                        # 인자 JSON 파싱이 실패한 도구는 실행하지 않는다. 빈 인자로
                        # 실행하면 스키마 검증 실패(예: "content는 비어 있을 수 없습니다")
                        # → 모델이 본문을 채팅에 덤프하는 UX 붕괴로 이어진다. 대신 이름만
                        # 기록해 스트림 종료 후 guided decoding 재시도로 정상 JSON을 받는다.
                        if getattr(event.tool_use, "parse_error", False):
                            parse_failed_tools.append(event.tool_use.name)
                        else:
                            tu_dict = {
                                "id": event.tool_use.id,
                                "name": event.tool_use.name,
                                "input": event.tool_use.input,
                            }
                            tool_use_blocks.append(tu_dict)
                            # 클라이언트가 실행하는 도구는 서버가 돌리지 않는다.
                            #   실행 주체가 호출자(예: VSCode 플러그인)이기 때문이다.
                            #   대기열에 넣지 않고 따로 모아 두었다가 이번 턴을 끝낸다.
                            if event.tool_use.name in client_tool_names:
                                client_tool_calls.append(tu_dict)
                            else:
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

            # ── 붕괴 조기 절단 → 샘플링을 바꿔 1회 재생성 (2026-08-13) ──
            # 워치독이 붕괴로 스트림을 끊었으면 여기 남은 텍스트는 쓸 수 없다.
            # 예전에는 그대로 사용자에게 나갔다(2026-08-13 14:33 실측: 8,944자 절단본).
            #
            # continue 로 턴을 다시 도는 것은 스트림 타임아웃 재시도와 같은 경로다.
            # 이 시점에는 assistant 메시지를 아직 대화에 넣지 않았으므로, 다음 턴이
            # **같은 입력으로 다시 생성**한다. 턴 머리에서 누적 버퍼가 새로 초기화되니
            # 절단본은 서버 쪽에서 사라진다.
            #
            # 이미 흘려보낸 TEXT_DELTA 는 되돌릴 수 없다(스트리밍 소비자는 절단본을
            # 봤다). 그래서 무슨 일이 일어났는지 SYSTEM_WARNING 으로 알린다. 반면
            # **비스트리밍 소비자(코딩 API 등)는 아직 아무것도 못 봤으므로 완전히
            # 깨끗하게 교체된다** — 오늘 문제가 관측된 표면이 정확히 그쪽이다.
            if degen_monitor is not None and degen_monitor.was_cut:
                if state.degen_retry_count < DEGEN_MAX_RETRY:
                    state.degen_retry_count += 1
                    state.degen_retry_pending = True
                    logger.warning(
                        "degeneration 재생성 %d/%d (절단 %d자, 온도 %.2f→%.2f)",
                        state.degen_retry_count,
                        DEGEN_MAX_RETRY,
                        degen_monitor.length,
                        temperature,
                        min(
                            DEGEN_RETRY_TEMPERATURE_MAX,
                            temperature + DEGEN_RETRY_TEMPERATURE_DELTA,
                        ),
                    )
                    yield StreamEvent(
                        type=StreamEventType.SYSTEM_WARNING,
                        # 소비자가 텍스트가 아니라 코드로 판별하게 한다 — 파이프
                        # 모드는 이 신호를 받으면 여태 모은 버퍼를 버려야 한다.
                        error_code=STREAM_DISCARD_RETRY,
                        message=(
                            "[생성 붕괴 감지] 앞의 출력은 버리고 다시 생성합니다 "
                            f"({state.degen_retry_count}/{DEGEN_MAX_RETRY})"
                        ),
                    )
                    state.continue_reason = ContinueReason.NEXT_TURN
                    # 절단본이 부른 도구는 취소한다 — 쓰레기 출력에서 나온 호출이다.
                    await streaming_executor.cancel_all()
                    continue
                # 예산 소진 — 재생성해도 또 붕괴했다. 절단본을 그대로 쓰되 기록은 남긴다.
                logger.warning(
                    "degeneration 재생성 예산 소진 — 절단본을 그대로 사용한다(%d자)",
                    degen_monitor.length,
                )
                # ★2026-08-23: 여기가 유일하게 신호가 없던 자리였다. 절단본을 그대로
                #   내보내면서 아무에게도 알리지 않아, 비대화형 `nexus ask` 가
                #   **잘린 답변을 exit 0 으로** 내보냈다(실측). 소비자가 성공으로
                #   오인하는 것을 막으려면 이 사실이 스트림에 실려야 한다.
                yield StreamEvent(
                    type=StreamEventType.SYSTEM_WARNING,
                    error_code=STREAM_TRUNCATED,
                    message=(
                        f"[생성 붕괴] 재생성 {DEGEN_MAX_RETRY}회로도 복구하지 못해 "
                        f"잘린 출력을 그대로 사용합니다({degen_monitor.length}자). "
                        "답변이 불완전할 수 있습니다."
                    ),
                )
            elif state.degen_retry_count:
                # 붕괴 없이 끝났다 → 예산을 되돌려 다음 붕괴도 1회 재생성을 받는다.
                state.degen_retry_count = 0

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
                        # 붕괴 재생성과 같은 구조다 — 이미 흘린 부분 출력은 버려지고
                        # 처음부터 다시 생성된다. 소비자도 같게 처리해야 한다.
                        error_code=STREAM_DISCARD_RETRY,
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

        # ─── 도구 인자 파싱 실패 → guided decoding 재시도 (Option 1) ───
        # A.X-4.0는 tool_choice="auto" + hermes 파서에서 긴 인자 JSON을 자유형식으로
        # 내다 깨뜨린다(미이스케이프 따옴표/개행, 닫는 "·}·필수필드 누락). 빈 인자로
        # 실행하면 스키마 검증 실패(예: "content는 비어 있을 수 없습니다") → 모델이
        # 본문을 채팅에 덤프하는 UX 붕괴로 이어진다. 그래서 그 도구로 tool_choice를
        # 'named'로 강제해 다시 요청하면, vLLM이 그 도구의 parameters 스키마로 guided
        # decoding을 적용해 유효 JSON을 보장한다(B200 실서버 확증). 정상 파싱된 도구가
        # 하나라도 있으면(tool_use_blocks 비어있지 않음) 그걸로 진행하므로 건너뛴다.
        # ── 도구별 파싱 실패 누적 가드 (2026-08-23) ──
        # 아래 `if parse_failed_tools and not tool_use_blocks:` 분기와 별개다.
        # 그 분기는 **이번 턴에 건진 도구가 하나도 없을 때만** 돌고, 재시도 카운터도
        # 정상 턴마다 리셋된다. 그래서 [실패 → 성공 → 실패 …] 교대 패턴이 통째로
        # 빠져나갔다(실측: 같은 문서를 11회 다시 쓰며 30턴 공전).
        # 여기서는 성공 여부와 무관하게 **누적**으로 세어 그 사각을 막는다.
        for _failed_name in dict.fromkeys(parse_failed_tools):
            state.tool_parse_fail_total[_failed_name] = (
                state.tool_parse_fail_total.get(_failed_name, 0) + 1
            )
            _total = state.tool_parse_fail_total[_failed_name]

            if _total >= PARSE_FAIL_ABORT_AT:
                logger.warning(
                    "도구 인자 파싱 실패 누적 %d회(%s) — 턴 루프 강제 종료",
                    _total,
                    _failed_name,
                )
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code="TOOL_ARGS_REPEATEDLY_UNPARSEABLE",
                    message=(
                        f"{_failed_name} 도구의 인자를 {_total}회 연속으로 생성하지 "
                        "못했습니다. 요청 범위를 나눠(예: 섹션별로) 다시 시도해 주세요."
                    ),
                )
                await streaming_executor.cancel_all()
                return

            # 넛지는 도구당 1회만. 매 턴 잔소리하면 컨텍스트만 먹고 효과는 없다.
            if _total >= PARSE_FAIL_NUDGE_AT and _failed_name not in state.parse_nudge_sent:
                state.parse_nudge_sent.add(_failed_name)
                logger.warning(
                    "도구 인자 파싱 실패 누적 %d회(%s) — 분할 작성 넛지 주입",
                    _total,
                    _failed_name,
                )
                yield StreamEvent(
                    type=StreamEventType.SYSTEM_WARNING,
                    message=(
                        f"[인자 생성 반복 실패] {_failed_name} {_total}회 — "
                        "요청을 나눠 진행하도록 안내합니다"
                    ),
                )
                # user 역할로 넣어야 모델이 다음 턴에 확실히 읽는다(위 반복 실패
                # 경고와 같은 방식). 내용은 **행동 지시**여야 한다 — "실패했다"만
                # 알리면 모델이 같은 크기로 또 시도한다.
                state.messages.append(
                    Message.user(
                        f"[시스템] {_failed_name} 도구의 인자 JSON 이 {_total}회 "
                        "연속으로 깨졌습니다. 한 번에 넘기는 내용이 너무 깁니다. "
                        "같은 크기로 다시 시도하지 마세요. 문서를 여러 조각으로 "
                        "나눠 짧게 여러 번 쓰거나, 이미 쓴 파일이 있으면 그대로 두고 "
                        "다음 단계로 넘어가세요."
                    )
                )

        if parse_failed_tools and not tool_use_blocks:
            # 중복 제거하며 순서 보존
            unique_failed = list(dict.fromkeys(parse_failed_tools))
            # 복수 도구가 동시에 파싱 실패하면 named tool_choice로 하나만 강제할 수
            # 없어 나머지 호출이 소실된다 → guided 재시도 대신 정직한 에러(가짜 완료 금지).
            if len(unique_failed) == 1 and state.tool_parse_retry_count < MAX_TOOL_PARSE_RETRY:
                state.tool_parse_retry_count += 1
                state.force_tool_choice = unique_failed[0]  # 다음 턴 stream()에 강제
                state.continue_reason = ContinueReason.NEXT_TURN
                yield StreamEvent(
                    type=StreamEventType.SYSTEM_WARNING,
                    message=(
                        f"[도구 인자 재생성 {state.tool_parse_retry_count}/"
                        f"{MAX_TOOL_PARSE_RETRY}] {unique_failed[0]}"
                    ),
                )
                await streaming_executor.cancel_all()
                continue
            else:
                # guided 재시도로도 유효 JSON을 못 받았거나(대개 진짜 length 절단)
                # 복수 도구 동시 실패. 빈 인자 실행/인라인 덤프 대신 정직한 에러로 종료.
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code="TOOL_ARGS_UNPARSEABLE",
                    message=(
                        "요청하신 산출물이 너무 길어 한 번에 생성하지 못했습니다. "
                        "범위를 나눠(예: 섹션별로) 다시 요청해 주세요."
                    ),
                )
                await streaming_executor.cancel_all()
                return

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
                # 재시도 모두 실패 — 사용자에게 에러 메시지 전달 후 종료.
                # ★사유를 뭉뚱그리지 않는다(2026-08-20). 예전에는 어떤 모델 오류든
                #   "입력이 너무 길다"로 보고해, 실제 원인이 HTTP 404(모델 이름
                #   불일치)인데도 사용자가 입력만 줄이게 만들었다. 실측으로 확인된
                #   오진이라, 컨텍스트 초과로 단정할 수 있을 때만 그렇게 말한다.
                detail = (model_error or "").strip()
                is_context = "context" in detail.lower() or "너무 길" in detail
                if is_context:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error_code="CONTEXT_OVERFLOW",
                        message=(
                            "입력 내용이 너무 길어 분석할 수 없습니다. "
                            "더 짧은 내용으로 다시 시도해 주세요."
                        ),
                    )
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error_code="MODEL_ERROR",
                        message=(
                            "모델 응답을 처리하지 못했습니다"
                            f"({MAX_TOOL_PARSE_RETRY}회 재시도 실패). "
                            f"원인: {detail[:300] or '알 수 없음'}"
                        ),
                    )
                return

        # ─── 자기일관성(SC) 합의 + 승자 의사-스트림 (Point 4.3, §3·§6.2) ───
        # SC 모드에서 표본 후보가 모였고, tool_calls 폴백도 아니고, 모델 에러도 아니면
        # 합의를 낸다. 표본에 tool_calls가 섞이면(tool_use_blocks 있음) Tier 3가 이미
        # 단일 스트림으로 폴백했으므로 여기선 SC를 건너뛰고 일반 도구 실행 턴으로 간다.
        if sc_active and sc_candidates and not tool_use_blocks and not model_error:
            # 표본을 index 순서대로 정렬해 후보 리스트를 만든다.
            candidates = [sc_candidates[i] for i in sorted(sc_candidates)]
            embeddings: list[list[float]] | None = None
            # 서술형(긴 답)이 하나라도 있으면 임베딩 클러스터 폴백을 위해 임베딩을 만든다.
            # (짧은 사실형만 있으면 임베딩 없이 순수 다수결 — 임베딩 서버 왕복 절약)
            if any(len(c) > sc_short_answer_max_chars for c in candidates):
                try:
                    embeddings = await model_provider.embed(candidates)
                except Exception as e:
                    # 임베딩 서버 실패는 치명적이지 않다 — 다수결로 폴백(정직히 로그).
                    logger.warning("SC 임베딩 실패 — 다수결로 폴백: %s", e)
                    embeddings = None
            # 순수 함수 합의 모듈 호출(모델 호출 없음 — 체인 밖 헬퍼).
            result = resolve_consensus(
                candidates,
                min_agreement=sc_min_agreement,
                short_answer_max_chars=sc_short_answer_max_chars,
                embeddings=embeddings,
                similarity_threshold=sc_similarity_threshold,
            )
            if not result.consensus_reached:
                # 합의 실패(3표가 전부 다름 등) — 후보 0번 채택 + 관측 로그(§3.4).
                # 재샘플링(N 추가)은 하지 않는다(토큰 예산 원칙).
                logger.warning(
                    "SC 합의 실패 (%d/%d표) — 첫 표본 채택: %.80s",
                    result.agreement, result.total, result.winner,
                )
                yield StreamEvent(
                    type=StreamEventType.SYSTEM_WARNING,
                    message=(
                        f"[사실 교차검증] {result.total}표본 중 합의 실패 "
                        f"({result.agreement}표) — 첫 표본을 사용합니다."
                    ),
                )
            else:
                logger.info(
                    "SC 합의 성공 (%s, %d/%d표)",
                    result.method, result.agreement, result.total,
                )
            # 승자를 40자씩 잘라 TEXT_DELTA로 의사-스트림(일반 턴과 동일 렌더링, §6.2).
            for _chunk in _chunk_text(result.winner, 40):
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=_chunk)
            # assistant 메시지 본문을 승자로 확정한다(아래 join이 이 값을 집는다).
            assistant_text_parts = [result.winner]

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

        # ─── 클라이언트 실행 도구 → 여기서 턴을 끝낸다 ───
        # 서버가 실행할 것이 없으므로 다음 턴으로 넘어갈 근거가 없다. 호출자가
        # tool_calls 를 받아 자기 쪽에서 실행하고, 결과를 다음 요청에 실어 보낸다.
        # (OpenAI `tools` 규약의 본래 동작 — 루프의 주인은 클라이언트다.)
        if client_tool_calls:
            # 정상 종료 경로와 같이 continue_reason 은 건드리지 않고 그대로 빠져나간다
            # (ContinueReason 에는 '완료' 값이 없다 — 종료는 return 으로 표현한다).
            await streaming_executor.cancel_all()
            state.last_stop_reason = StopReason.TOOL_USE
            logger.info(
                "클라이언트 실행 도구 %d건 — 서버 실행 없이 턴 종료: %s",
                len(client_tool_calls),
                [c["name"] for c in client_tool_calls],
            )
            return

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
                            # ★전문을 넘긴다(2026-08-23). 200자로 자르면 주장이 대개
                            #   답변 끝에 오기 때문에 판정에 필요한 문장이 잘려 나간다.
                            "assistant_text": assistant_text,
                            # 마지막 사용자 메시지 이후 **성공한** 쓰기 도구들.
                            #   훅이 "썼다는데 안 썼다"를 판정할 유일한 근거다.
                            "write_tools": _successful_write_tools_since_user(
                                state.messages
                            ),
                            "block_count": state.stop_hook_block_count,
                        },
                    )
                    hook_result = await hook_manager.run(HookEvent.STOP, hook_input)
                    if hook_result.decision == HookDecision.BLOCK:
                        # Hook이 종료를 차단 → 강제 다음 턴
                        state.stop_hook_block_count += 1
                        state.continue_reason = ContinueReason.STOP_HOOK_BLOCKING
                        logger.info(
                            "Hook이 종료를 차단(%d회): %s",
                            state.stop_hook_block_count,
                            hook_result.block_reason,
                        )
                        # ★차단 사유를 대화에 넣어야 모델이 무엇을 해야 하는지 안다.
                        #   이게 없으면 모델은 왜 턴이 이어지는지 모른 채 같은 답을
                        #   반복한다(차단이 루프로 바뀌는 지점).
                        if hook_result.block_reason:
                            state.messages.append(
                                Message.user(
                                    f"[시스템] {hook_result.block_reason} "
                                    "말로 끝내지 말고 지금 실제로 도구를 호출해 "
                                    "파일을 만드세요. 이미 만들었다면 Read 나 LS 로 "
                                    "존재를 확인한 뒤 그 결과를 근거로 답하세요."
                                )
                            )
                        yield StreamEvent(
                            type=StreamEventType.SYSTEM_WARNING,
                            message=f"[종료 차단] {hook_result.block_reason}",
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
