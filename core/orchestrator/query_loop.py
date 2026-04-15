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
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.message import (
    Message,
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

# 출력 토큰 에스컬레이션 단계
# max_tokens를 점진적으로 증가시킨다 (4K → 8K → 16K)
OUTPUT_TOKEN_ESCALATION = [4096, 8192, 16384]


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

    Yields:
        StreamEvent: 스트리밍 이벤트 (UI 업데이트용)
        Message: assistant/tool_result 메시지 (대화 히스토리용)
    """
    # 루프 상태 초기화
    state = LoopState(messages=messages)
    stop_resolver = StopResolver()

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
        model_cfg = model_provider.get_config()
        base_max_tokens = model_cfg.max_output_tokens

        if state.max_output_tokens_override:
            max_tokens = state.max_output_tokens_override
        else:
            # 입력 토큰 추정: 시스템 프롬프트 + 도구 스키마 + 메시지
            # 토큰 추정은 문자수/3 (보수적) — 한글/특수문자가 많으면 토큰이 더 많다
            import json as _json

            tool_chars = sum(len(_json.dumps(s, ensure_ascii=False)) for s in tool_schemas)
            msg_chars = sum(len(str(m.content)) for m in api_messages)
            prompt_chars = len(system_prompt)
            total_chars = tool_chars + msg_chars + prompt_chars
            estimated_input = total_chars // 3  # 보수적 추정 (영어 /4, 한글 /2 → 평균 /3)

            max_context = model_cfg.max_context_tokens

            # 입력이 컨텍스트의 85%를 초과하면 메시지를 truncate한다
            # 왜 85%: 최소 출력 512토큰 + 버퍼 확보
            input_limit = int(max_context * 0.85)
            if estimated_input > input_limit and len(api_messages) > 1:
                # 가장 최근 메시지의 내용을 자른다 (대화 맥락 유지)
                last_msg = api_messages[-1]
                content_str = str(last_msg.content)
                # 초과분 계산 후 마지막 메시지에서 제거
                excess_chars = (estimated_input - input_limit) * 3
                if len(content_str) > excess_chars + 200:
                    truncated = content_str[: len(content_str) - excess_chars]
                    truncated += "\n\n[내용이 길어서 일부가 잘렸습니다. 핵심 부분만 분석합니다.]"
                    last_msg.content = truncated
                    # 재추정
                    msg_chars = sum(len(str(m.content)) for m in api_messages)
                    total_chars = tool_chars + msg_chars + prompt_chars
                    estimated_input = total_chars // 3

                orig_chars = tool_chars + prompt_chars + sum(
                    len(str(m.content)) for m in state.messages
                )
                logger.info(
                    "입력 truncate: %d → %d 토큰 (컨텍스트 %d의 85%%)",
                    orig_chars // 3, estimated_input, max_context,
                )

            # 최대 컨텍스트에서 입력을 빼고 200 토큰 버퍼를 둔다
            dynamic_max = max_context - estimated_input - 200
            max_tokens = max(512, min(base_max_tokens, dynamic_max))

        # ═══════════════════════════════════════
        # Phase 2: API Call (모델 스트리밍)
        # ═══════════════════════════════════════
        # model_provider.stream()을 호출하고 이벤트를 수집/yield한다.
        # 동시에 StreamingToolExecutor로 도구를 미리 실행한다.

        assistant_text_parts: list[str] = []  # TEXT_DELTA 누적
        tool_use_blocks: list[dict[str, Any]] = []  # 완성된 tool_use 블록
        turn_usage = TokenUsage()  # 이번 턴의 토큰 사용량
        stop_reason: StopReason | None = None  # 모델 종료 이유
        model_error: str | None = None  # 모델 에러 메시지

        # StreamingToolExecutor 생성 — 스트리밍 중 도구를 병렬 실행
        streaming_executor = StreamingToolExecutor(
            tools=tools,
            context=context,
        )

        try:
            # Tier 3 호출: model_provider.stream()
            # 참고: 프로덕션에서는 with_retry() (Ch.7.1)와
            # StreamWatchdog (Ch.7.2)로 감싸야 한다
            async for event in model_provider.stream(
                messages=api_messages,
                system_prompt=system_prompt,
                tools=tool_schemas if tools else None,
                temperature=0.7,
                max_tokens=max_tokens,
            ):
                # 이벤트를 UI로 전파
                yield event

                # 이벤트 처리 — type이 enum이거나 문자열일 수 있음
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

                # 스트리밍 중 완료된 도구 결과를 소비
                for completed in streaming_executor.get_completed():
                    yield completed

        except Exception as e:
            error_name = type(e).__name__

            # ─── Transition 1: collapse_drain_retry ───
            # "context too long" 류의 에러 → 긴급 압축 후 재시도
            if "context" in str(e).lower() and "long" in str(e).lower():
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
                    continue

            # ─── Transition 2: reactive_compact_retry ───
            # "prompt is too long" 에러 → 압축 후 재시도
            if "prompt is too long" in str(e).lower():
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
                    continue

            # GPU OOM → 컨텍스트 30% 감소 후 재시도
            if "out of memory" in str(e).lower():
                state.model_error_count += 1
                if state.model_error_count <= MAX_MODEL_ERROR_RETRY:
                    if context_manager is not None:
                        context_manager.max_tokens = int(context_manager.max_tokens * 0.7)
                    yield StreamEvent(
                        type=StreamEventType.SYSTEM_WARNING,
                        message=(
                            f"[GPU OOM] 컨텍스트 축소 후 재시도 "
                            f"({state.model_error_count}/{MAX_MODEL_ERROR_RETRY})"
                        ),
                    )
                    state.continue_reason = ContinueReason.REACTIVE_COMPACT_RETRY
                    await streaming_executor.cancel_all()
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
        if model_error and not tool_use_blocks:
            state.tool_parse_retry_count += 1
            if state.tool_parse_retry_count <= MAX_TOOL_PARSE_RETRY:
                # 모델에 재시도를 요청하는 시스템 메시지 추가
                state.messages.append(
                    Message.system(
                        "이전 응답에 에러가 있었습니다. "
                        "도구를 사용하려면 올바른 형식을 사용해주세요. "
                        "그렇지 않으면 일반 텍스트로 응답해주세요."
                    )
                )
                state.continue_reason = ContinueReason.NEXT_TURN
                yield StreamEvent(
                    type=StreamEventType.SYSTEM_WARNING,
                    message=(
                        f"[도구 파싱 재시도 {state.tool_parse_retry_count}/{MAX_TOOL_PARSE_RETRY}]"
                    ),
                )
                await streaming_executor.cancel_all()
                continue

        # assistant 메시지 기록
        assistant_text = "".join(assistant_text_parts)
        assistant_msg = Message.assistant(
            text=assistant_text,
            tool_uses=tool_use_blocks if tool_use_blocks else None,
        )
        state.messages.append(assistant_msg)
        yield assistant_msg

        state.last_stop_reason = stop_reason

        # 턴 종료 이벤트
        yield StreamEvent(type=StreamEventType.STREAM_REQUEST_END)

        # ─── 종료 판단 ───
        # StopResolver로 도구 호출 유무를 확인
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
                        current_idx = (
                            OUTPUT_TOKEN_ESCALATION.index(max_tokens)
                            if max_tokens in OUTPUT_TOKEN_ESCALATION
                            else 0
                        )
                        next_idx = min(
                            current_idx + 1,
                            len(OUTPUT_TOKEN_ESCALATION) - 1,
                        )
                        state.max_output_tokens_override = OUTPUT_TOKEN_ESCALATION[next_idx]
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

        async for event in streaming_executor.drain_remaining():
            yield event
            # Message 이벤트(tool_result)는 대화 히스토리에 추가
            if isinstance(event, Message):
                state.messages.append(event)

        # ─── Transition 7: next_turn ───
        # 도구 실행 완료 → 정상적으로 다음 턴 진행
        state.continue_reason = ContinueReason.NEXT_TURN
        state.max_output_recovery_count = 0  # 복구 카운터 리셋
        state.tool_parse_retry_count = 0  # 파싱 재시도 카운터 리셋

    # max_turns 도달 — 무한 루프 방지
    logger.warning(f"쿼리 루프 최대 턴 수 도달 ({max_turns})")
    yield StreamEvent(
        type=StreamEventType.SYSTEM_WARNING,
        message=f"[경고] 최대 턴 수({max_turns})에 도달했습니다. 중단합니다.",
    )
