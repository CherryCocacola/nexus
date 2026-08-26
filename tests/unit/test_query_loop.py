"""
core/orchestrator/query_loop.py — 컨텍스트 오버플로 복구 도달불가 결함(감사 Critical #8)
회귀 방지 단위 테스트.

핵심 배경:
  Tier 3(core/model/inference.py stream())는 컨텍스트 초과·HTTP 오류를 예외로
  raise하지 않고 ERROR StreamEvent로 "yield"해서 내려보낸다. 예전 query_loop은
  이 ERROR 이벤트에 대해 복구 없이 즉시 종료(abort)했고, 진짜 복구 로직은
  `except Exception` 블록 안에만 있어 raise가 없는 이 경로에서는 영구 미도달이었다.

  이 테스트는 두 경로 모두에서 동일한 복구가 걸리는지 검증한다:
    (a) ERROR 이벤트(CONTEXT_OVERFLOW) → 긴급 압축(emergency_compact) + 다음 턴 재시도
    (b) ERROR 이벤트(HTTP_400 "prompt is too long") → 반응적 압축(auto_compact force=True)
    (c) 복구 예산 소진 → 결국 abort(다음 턴으로 넘어가지 않음)
    (d) raise된 예외 경로(기존)도 여전히 동일 복구 — 무회귀
    (e) OOM ERROR 이벤트 → SYSTEM_WARNING yield + context_manager.max_tokens 축소
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from core.message import (
    Message,
    StopReason,
    StreamEvent,
    StreamEventType,
)
from core.model.inference import ModelConfig, ModelProvider
from core.orchestrator.query_loop import MAX_COLLAPSE_DRAIN, query_loop
from core.tools.base import ToolUseContext


# ─────────────────────────────────────────────
# 테스트용 스크립트형 ModelProvider
# ─────────────────────────────────────────────
class ScriptedProvider(ModelProvider):
    """
    stream() 호출 순번마다 스크립트대로 동작하는 mock 프로바이더.

    각 action(dict)은 셋 중 하나:
      {"error_event": (error_code, message)} → ERROR StreamEvent yield 후 종료
          (실제 inference가 raise하지 않고 ERROR를 yield하는 상황을 재현)
      {"raise": Exception(...)}              → 예외를 raise (기존 except 경로 재현)
      {"text": "..."}                        → 정상 텍스트 응답(END_TURN)
      {"tool_parse_error": "ToolName"}       → 인자 파싱 실패 도구 호출 재현
          (parse_error=True인 TOOL_USE_STOP + stop_reason=TOOL_USE)

    마지막 action은 리스트를 벗어난 호출에서 반복 사용된다.
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = script
        self.call_count = 0
        # 마지막 stream() 호출에서 받은 tools/structured_output을 기록해 두어
        # Tier2→Tier3 전달(passthrough) 및 도구 비노출 분기를 테스트에서 검증한다.
        self.last_structured_output: Any = None
        self.last_tools: list[dict[str, Any]] | None = None
        self.last_n: int = 1
        # 매 호출에서 받은 force_tool_choice를 순서대로 기록(guided 재시도 검증용).
        self.force_tool_choice_history: list[str | None] = []

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
        model_override: str | None = None,
        enable_thinking: bool | None = False,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        structured_output: Any = None,
        n: int = 1,
        force_tool_choice: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        # 전파 검증용 기록 — 실제 동작은 흉내내지 않는다.
        self.last_structured_output = structured_output
        self.last_tools = tools
        self.last_n = n
        self.last_force_tool_choice = force_tool_choice
        self.force_tool_choice_history.append(force_tool_choice)
        idx = min(self.call_count, len(self._script) - 1)
        action = self._script[idx]
        self.call_count += 1

        yield StreamEvent(type=StreamEventType.MESSAGE_START, model_id="mock")

        if "tool_parse_error" in action:
            # 인자 JSON 파싱 실패 도구 호출 재현: parse_error=True인 TOOL_USE_STOP.
            # (inference._finalize_tool_calls가 strict=False로도 실패 시 내는 형태)
            from core.message import ToolUseBlock

            name = action["tool_parse_error"]
            yield StreamEvent(
                type=StreamEventType.TOOL_USE_START,
                tool_use=ToolUseBlock(id="call_x", name=name, input={}),
            )
            yield StreamEvent(
                type=StreamEventType.TOOL_USE_STOP,
                tool_use=ToolUseBlock(
                    id="call_x", name=name, input={}, parse_error=True
                ),
            )
            yield StreamEvent(
                type=StreamEventType.MESSAGE_STOP,
                stop_reason=StopReason.TOOL_USE,
            )
            return

        if "raise" in action:
            # inference가 예외를 그대로 던지는 상황(기존 except 경로) 재현
            raise action["raise"]

        if "error_event" in action:
            code, msg = action["error_event"]
            # inference처럼 ERROR StreamEvent를 yield하고 스트림을 종료한다
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error_code=code,
                message=msg,
            )
            return

        # 정상 텍스트 응답
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=action.get("text", "ok"))
        yield StreamEvent(
            type=StreamEventType.MESSAGE_STOP,
            stop_reason=StopReason.END_TURN,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def health_check(self) -> bool:
        return True

    async def count_tokens(self, messages: list[Message]) -> int:
        return len(messages) * 10

    def get_config(self) -> ModelConfig:
        return ModelConfig(
            model_id="mock",
            max_context_tokens=8192,
            max_output_tokens=4096,
        )


# ─────────────────────────────────────────────
# 복구 호출 여부를 감시(spy)하는 ContextManager 대역
# ─────────────────────────────────────────────
class SpyContextManager:
    """
    query_loop이 압축 API를 언제/어떻게 호출하는지 기록만 하는 가짜 컨텍스트 매니저.

    - apply_all / auto_compact_if_needed(무force)는 Phase 1에서 매 턴 호출된다.
    - 복구 시에는 emergency_compact(긴급) 또는 auto_compact_if_needed(force=True)가
      추가로 호출된다 → 이 호출을 spy로 잡아 어떤 복구가 발동했는지 검증한다.
    """

    def __init__(self) -> None:
        self.max_tokens = 4000
        self.apply_all_calls = 0
        # auto_compact_if_needed에 전달된 force 값들을 순서대로 기록
        self.auto_compact_forces: list[bool] = []
        self.emergency_compact_calls = 0
        # 결과를 채택한 호출부가 경계를 되돌렸는지 기록한다(2026-08-26).
        self.adopted_calls = 0

    def apply_all(self, messages: list[Message]) -> list[Message]:
        self.apply_all_calls += 1
        return messages

    async def auto_compact_if_needed(
        self, messages: list[Message], force: bool = False
    ) -> list[Message]:
        self.auto_compact_forces.append(force)
        return messages

    async def emergency_compact(self, messages: list[Message]) -> list[Message]:
        self.emergency_compact_calls += 1
        return messages

    def mark_result_adopted(self) -> None:
        """압축 결과로 리스트를 교체한 호출부가 부른다.

        실물과 같은 시그니처를 갖춰야 한다 — 이 더블에 메서드가 없으면 복구 경로가
        AttributeError 로 죽는데, 그건 코드 결함이 아니라 더블이 뒤처진 것이다.
        호출 여부 자체가 검증 대상이므로 횟수를 센다.
        """
        self.adopted_calls += 1


# ─────────────────────────────────────────────
# 공통 실행 헬퍼
# ─────────────────────────────────────────────
async def _run_loop(
    provider: ScriptedProvider,
    cm: SpyContextManager,
    ctx: ToolUseContext,
) -> list[StreamEvent | Message]:
    """query_loop을 끝까지 소비하고 이벤트를 모아 반환한다."""
    events: list[StreamEvent | Message] = []
    async for event in query_loop(
        messages=[Message.user("안녕")],
        system_prompt="테스트 시스템 프롬프트",
        model_provider=provider,
        tools=[],
        context=ctx,
        context_manager=cm,
    ):
        events.append(event)
    return events


def _texts(events: list[StreamEvent | Message]) -> str:
    """이벤트에서 TEXT_DELTA를 모두 이어붙인다."""
    parts: list[str] = []
    for e in events:
        if isinstance(e, StreamEvent) and e.type == StreamEventType.TEXT_DELTA.value:
            parts.append(e.text or "")
    return "".join(parts)


# ─────────────────────────────────────────────
# (a) CONTEXT_OVERFLOW ERROR 이벤트 → emergency_compact + 다음 턴 재시도
# ─────────────────────────────────────────────
async def test_error_event_context_overflow_triggers_emergency_compact_and_retries(
    tool_use_context: ToolUseContext,
) -> None:
    """
    Tier 3가 CONTEXT_OVERFLOW ERROR 이벤트를 yield하면, query_loop은 즉시 abort하지
    않고 emergency_compact(긴급 압축)를 호출한 뒤 '다음 턴'으로 재시도해야 한다.
    """
    provider = ScriptedProvider(
        [
            {"error_event": ("CONTEXT_OVERFLOW", "입력이 너무 길어 응답을 생성할 수 없습니다.")},
            {"text": "압축 후 정상 응답입니다."},
        ]
    )
    cm = SpyContextManager()

    events = await _run_loop(provider, cm, tool_use_context)

    # 긴급 압축이 정확히 1회 발동 (= collapse_drain 경로)
    assert cm.emergency_compact_calls == 1
    # 즉시 abort가 아니라 다음 턴으로 재시도 → 모델이 2번 호출됨
    assert provider.call_count == 2
    # 재시도 턴의 정상 응답이 사용자에게 전달됨
    assert "압축 후 정상 응답입니다." in _texts(events)


# ─────────────────────────────────────────────
# (b) HTTP_400 "prompt is too long" → reactive_compact(force=True)
# ─────────────────────────────────────────────
async def test_error_event_http400_prompt_too_long_triggers_reactive_compact(
    tool_use_context: ToolUseContext,
) -> None:
    """
    HTTP_400 응답 본문에 'prompt is too long'이 담겨 오면, 반응적 압축
    (auto_compact_if_needed(force=True))을 호출하고 다음 턴으로 재시도한다.
    """
    provider = ScriptedProvider(
        [
            {"error_event": ("HTTP_400", "vLLM 서버 에러: 400 - prompt is too long: 9000 tokens")},
            {"text": "압축 후 정상 응답."},
        ]
    )
    cm = SpyContextManager()

    events = await _run_loop(provider, cm, tool_use_context)

    # force=True 호출이 최소 1번 있어야 한다(반응적 압축 발동 증거)
    assert True in cm.auto_compact_forces
    # 긴급 압축은 이 경로에서 호출되지 않는다
    assert cm.emergency_compact_calls == 0
    assert provider.call_count == 2
    assert "압축 후 정상 응답." in _texts(events)


# ─────────────────────────────────────────────
# (c) 복구 예산 소진 → 결국 abort (다음 턴으로 넘어가지 않음)
# ─────────────────────────────────────────────
async def test_error_event_context_overflow_budget_exhausted_aborts(
    tool_use_context: ToolUseContext,
) -> None:
    """
    CONTEXT_OVERFLOW가 MAX_COLLAPSE_DRAIN 한도를 넘어 반복되면, 더는 재시도하지 않고
    기존 동작대로 abort한다(3번째 턴의 정상 응답에 도달하지 못한다).
    """
    provider = ScriptedProvider(
        [
            {"error_event": ("CONTEXT_OVERFLOW", "너무 김 1")},
            {"error_event": ("CONTEXT_OVERFLOW", "너무 김 2")},
            {"text": "여기까지 오면 안 됨"},
        ]
    )
    cm = SpyContextManager()

    events = await _run_loop(provider, cm, tool_use_context)

    # 긴급 압축은 한도(MAX_COLLAPSE_DRAIN)까지만 발동
    assert cm.emergency_compact_calls == MAX_COLLAPSE_DRAIN
    # 2번째 CONTEXT_OVERFLOW에서 abort → 3번째 턴(정상 응답)에는 도달하지 못함
    assert provider.call_count == 2
    assert "여기까지 오면 안 됨" not in _texts(events)


# ─────────────────────────────────────────────
# (d) 무회귀: raise된 예외 경로도 동일하게 복구
# ─────────────────────────────────────────────
async def test_raised_exception_context_long_still_recovers(
    tool_use_context: ToolUseContext,
) -> None:
    """
    (무회귀) 모델이 예외를 raise하는 기존 경로에서도 'context ... long' 오류는
    emergency_compact로 복구되고 다음 턴으로 재시도돼야 한다(동작 불변).
    """
    provider = ScriptedProvider(
        [
            {"raise": RuntimeError("This model's context is too long to process")},
            {"text": "예외 복구 후 정상 응답."},
        ]
    )
    cm = SpyContextManager()

    events = await _run_loop(provider, cm, tool_use_context)

    assert cm.emergency_compact_calls == 1
    assert provider.call_count == 2
    assert "예외 복구 후 정상 응답." in _texts(events)


async def test_raised_exception_prompt_too_long_still_recovers(
    tool_use_context: ToolUseContext,
) -> None:
    """
    (무회귀) 예외로 던져진 'prompt is too long'도 반응적 압축(force=True)으로
    복구된다 — ERROR 이벤트 경로와 동일한 결과.
    """
    provider = ScriptedProvider(
        [
            {"raise": RuntimeError("Bad request: prompt is too long")},
            {"text": "예외 반응 복구 응답."},
        ]
    )
    cm = SpyContextManager()

    events = await _run_loop(provider, cm, tool_use_context)

    assert True in cm.auto_compact_forces
    assert cm.emergency_compact_calls == 0
    assert provider.call_count == 2
    assert "예외 반응 복구 응답." in _texts(events)


# ─────────────────────────────────────────────
# (e) OOM ERROR 이벤트 → SYSTEM_WARNING yield + max_tokens 축소
# ─────────────────────────────────────────────
async def test_error_event_oom_yields_warning_and_reduces_max_tokens(
    tool_use_context: ToolUseContext,
) -> None:
    """
    HTTP_400 본문에 'out of memory'가 담겨 오면, OOM 복구가 발동해
    (1) SYSTEM_WARNING을 사용자에게 yield하고 (2) context_manager.max_tokens를
    0.7배로 축소한 뒤 다음 턴으로 재시도한다.
    """
    provider = ScriptedProvider(
        [
            {"error_event": ("HTTP_400", "vLLM 서버 에러: 400 - CUDA out of memory")},
            {"text": "OOM 축소 후 정상 응답."},
        ]
    )
    cm = SpyContextManager()
    original_max = cm.max_tokens

    events = await _run_loop(provider, cm, tool_use_context)

    # max_tokens가 0.7배로 줄었는지
    assert cm.max_tokens == int(original_max * 0.7)
    # SYSTEM_WARNING 이벤트가 사용자에게 전달됐는지 (OOM 안내 문구 포함)
    warnings = [
        e
        for e in events
        if isinstance(e, StreamEvent)
        and e.type == StreamEventType.SYSTEM_WARNING.value
        and e.message
        and "GPU OOM" in e.message
    ]
    assert len(warnings) == 1
    assert provider.call_count == 2
    assert "OOM 축소 후 정상 응답." in _texts(events)


# ─────────────────────────────────────────────
# (f) 무회귀: 복구 대상이 아닌 ERROR(CONNECT_ERROR)는 복구 시도 안 함
# ─────────────────────────────────────────────
async def test_error_event_connect_error_does_not_trigger_recovery(
    tool_use_context: ToolUseContext,
) -> None:
    """
    (무회귀) CONNECT_ERROR 같은 압축 불가 오류는 복구를 시도하지 않는다.
    emergency_compact / 반응적 압축(force)이 발동하지 않아야 한다.
    """
    provider = ScriptedProvider(
        [
            {"error_event": ("CONNECT_ERROR", "GPU 서버 연결 실패")},
        ]
    )
    cm = SpyContextManager()

    await _run_loop(provider, cm, tool_use_context)

    assert cm.emergency_compact_calls == 0
    assert True not in cm.auto_compact_forces


# ─────────────────────────────────────────────
# (g) 도구 인자 파싱 실패 → guided decoding 재시도(named tool_choice)
# ─────────────────────────────────────────────
def _warnings(events: list[StreamEvent | Message]) -> list[str]:
    """SYSTEM_WARNING 이벤트의 메시지만 모은다."""
    return [
        e.message or ""
        for e in events
        if isinstance(e, StreamEvent)
        and e.type == StreamEventType.SYSTEM_WARNING.value
    ]


async def test_tool_args_parse_error_retries_with_forced_tool_choice(
    tool_use_context: ToolUseContext,
) -> None:
    """
    인자 파싱 실패(parse_error) 도구 호출이 오면, 다음 턴에서 그 도구로
    tool_choice를 강제(force_tool_choice)해 재시도한 뒤 정상 종료한다.
    """
    provider = ScriptedProvider(
        [
            {"tool_parse_error": "DocumentExport"},  # 턴0: 인자 파싱 실패
            {"text": "완료"},  # 턴1(재시도): 정상 응답 → 종료
        ]
    )
    cm = SpyContextManager()

    events = await _run_loop(provider, cm, tool_use_context)

    # 턴0은 auto(None), 턴1(재시도)은 DocumentExport로 강제되어야 한다
    assert provider.force_tool_choice_history == [None, "DocumentExport"]
    # 재시도 안내(SYSTEM_WARNING)가 나왔는지
    assert any("도구 인자 재생성" in w and "DocumentExport" in w for w in _warnings(events))
    # 빈 인자 도구 실행/인라인 덤프 없이 정상 종료
    assert "완료" in _texts(events)
    assert not any(
        isinstance(e, StreamEvent)
        and e.type == StreamEventType.ERROR.value
        and e.error_code == "TOOL_ARGS_UNPARSEABLE"
        for e in events
    )


async def test_tool_args_parse_error_exhausted_yields_honest_error(
    tool_use_context: ToolUseContext,
) -> None:
    """
    guided 재시도(MAX_TOOL_PARSE_RETRY회)로도 계속 파싱 실패하면, 빈 인자 실행
    대신 정직한 TOOL_ARGS_UNPARSEABLE 에러를 내고 종료한다(가짜 완료 금지).
    """
    provider = ScriptedProvider(
        [
            {"tool_parse_error": "DocumentExport"},  # 마지막 action → 매 턴 반복
        ]
    )
    cm = SpyContextManager()

    events = await _run_loop(provider, cm, tool_use_context)

    # 초기 1회 + guided 재시도 MAX_TOOL_PARSE_RETRY회 = 총 3회 호출
    from core.orchestrator.query_loop import MAX_TOOL_PARSE_RETRY

    assert len(provider.force_tool_choice_history) == MAX_TOOL_PARSE_RETRY + 1
    assert provider.force_tool_choice_history[0] is None
    assert all(
        f == "DocumentExport" for f in provider.force_tool_choice_history[1:]
    )
    # 정직한 에러로 종료
    assert any(
        isinstance(e, StreamEvent)
        and e.type == StreamEventType.ERROR.value
        and e.error_code == "TOOL_ARGS_UNPARSEABLE"
        for e in events
    )
