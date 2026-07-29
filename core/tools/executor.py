"""
도구 실행 파이프라인 — 하나의 tool_use 블록을 안전하게 실행하는 13단계 라이프사이클.

[이 파일이 하는 일]
LLM(모델)이 "이 도구를 이런 입력으로 실행해줘"라고 요청한 tool_use 블록 하나를 받아서,
검증 → 권한 확인 → 실제 실행 → 결과 포맷팅까지 순서대로 처리한다. 각 단계에서 문제가
생기면 즉시 <tool_use_error> 형태의 오류 결과를 내보내고 파이프라인을 중단한다(fail-closed).
Claude Code의 toolExecution.ts(337~1600행)를 파이썬으로 재구현한 모듈이다.

[핵심 함수 / 헬퍼]
  - run_tool_use()            : 진입점. 13단계 전체를 순서대로 도는 AsyncGenerator.
  - _validate_json_schema()   : 입력을 도구의 JSON Schema로 검증(3단계).
  - _speculative_bash_security(): Bash 명령어의 위험 패턴을 사전 차단(5단계, 비동기 병렬).
  - _save_large_result()      : 너무 큰 결과를 디스크에 저장하고 앞뒤만 남김(11단계).

[13단계 개요]
  1. 도구 찾기 (정확한 이름 → alias → 레지스트리 폴백)
  2. abort 확인 (사용자 취소 시그널)
  3. JSON Schema 검증 (입력 형식)
  4. 도메인 검증 (도구별 커스텀 규칙)
  5. 보안 분류 (Bash 전용, 비동기로 병렬 실행)
  6. 입력 backfill (Hook용 관찰 가능 입력) — 현재 간소화
  7. PreToolUse 훅 — 현재 간소화(Phase 4에서 전체 구현)
  8. 권한 해결 (5계층 파이프라인 관측 + 도구 자체 check_permissions)
  9. 도구 실행 (tool.call(), 타임아웃 포함)
  10. 결과 직렬화 (map_result로 문자열 변환)
  11. 대형 결과 디스크 저장 (max_result_size 초과 시)
  12. PostToolUse 훅 — 현재 간소화
  13. 결과 yield (tool_result 메시지로 반환)

[의존 관계]
  상위 orchestrator(query_loop)가 이 함수를 호출한다. 이 모듈은 core.message(StreamEvent/
  Message)와 core.tools.base(BaseTool/ToolUseContext)에만 의존한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from core.message import Message, StreamEvent, StreamEventType, ToolResultBlock
from core.tools.base import (
    BaseTool,
    ToolUseContext,
)

# 모듈 전용 로거. 프로젝트 규칙상 "nexus.{모듈경로}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.tools.executor")

# ASK 판정인데 확인 핸들러(ask_handler)가 주입되지 않아 통과시킨 도구 이름을
# 기록해 둔다. 웹·비대화형 경로에서는 대화형 프롬프트가 불가능해 통과가 정상이지만,
# 배선 누락을 조용히 넘기지 않도록 도구별로 딱 1회만 경고 로그를 남기기 위한 집합.
# (매 Bash 호출마다 경고하면 웹 로그가 도배되므로 프로세스 수명 동안 도구당 1회.)
_ASK_PASSTHROUGH_WARNED: set[str] = set()


async def _emit_tool_result(
    tool_use_id: str,
    content: str,
    is_error: bool = False,
) -> AsyncGenerator[StreamEvent | Message, None]:
    """
    하나의 도구 결과를 "표시용 StreamEvent"와 "히스토리용 Message" 두 형태로 함께 내보낸다.

    왜 두 개를 yield하는가 (핵심):
      - StreamEvent(TOOL_RESULT): UI(웹/CLI)가 "접힌 활동라인 + 접힌 요약(펼치면 원문)"
        을 실시간으로 그리기 위한 표시용 프레임이다. 결과 본문은 tool_result 필드
        (ToolResultBlock)에 담는다. 이 이벤트는 4-Tier 체인의 정상 이벤트 흐름을 타고
        상위(query_loop → QueryEngine → web)로 그대로 전파된다. (체인 우회 아님)
      - Message(tool_result): 다음 턴에서 "모델"이 읽어야 하는 대화 히스토리용 결과다.
        query_loop은 흐르는 값 중 Message만 골라 state.messages에 누적하고,
        StreamEvent는 표시용이라 히스토리에 넣지 않는다(중복 저장 방지).

    두 값은 같은 tool_use_id/content/is_error를 공유하므로, 사용자가 보는 UI 요약과
    모델이 보는 히스토리가 항상 일치한다. 기존 계약(Message.tool_result)은 그대로 두고,
    표시용 StreamEvent만 "추가"하는 것이라 하위 호환이 유지된다.

    Args:
        tool_use_id: 이 결과가 어떤 도구 호출(ToolUseBlock.id)에 대한 것인지.
        content: 도구 결과 문자열(성공 결과 또는 <tool_use_error> 오류 메시지).
        is_error: 실패 여부. UI는 이 값으로 에러 색상/아이콘을, 모델은 오류 인지를 한다.

    Yields:
        StreamEvent(TOOL_RESULT): 표시용 프레임(먼저).
        Message(tool_result): 히스토리용 결과(다음).
    """
    # 1) 표시용 이벤트 — UI가 접힌 요약/펼침 원문을 그릴 수 있도록 먼저 흘려보낸다.
    yield StreamEvent(
        type=StreamEventType.TOOL_RESULT,
        tool_result=ToolResultBlock(
            tool_use_id=tool_use_id,
            content=content,
            is_error=is_error,
        ),
    )
    # 2) 히스토리용 메시지 — 모델이 다음 턴에 읽도록 query_loop이 이 값만 누적한다.
    yield Message.tool_result(tool_use_id, content, is_error=is_error)

# 대형 도구 결과를 파일로 떨어뜨릴 디렉토리.
# 사용자 홈 아래 ~/.nexus/tool_results 에 저장한다(11단계에서 사용).
LARGE_RESULT_DIR = Path.home() / ".nexus" / "tool_results"


async def run_tool_use(
    tool_use: dict[str, Any],
    tools: list[BaseTool],
    context: ToolUseContext,
    tool_registry: Any | None = None,
) -> AsyncGenerator[StreamEvent | Message, None]:
    """
    13단계 도구 실행 파이프라인 — 이 모듈의 유일한 공개 진입점.

    모델이 요청한 tool_use 블록 하나를 받아 검증 → 권한 → 실행 → 결과까지 처리한다.
    각 단계에서 실패하면 오류 tool_result를 yield하고 즉시 return으로 중단한다.
    이 함수는 코루틴이 아니라 AsyncGenerator이므로, 호출측은 반드시 `async for`로
    소비해야 이벤트/결과가 흘러나온다.

    Args:
        tool_use: 실행할 도구 요청. {"id": "...", "name": "...", "input": {...}} 형태.
        tools: 이번 턴에 사용 가능한 도구 리스트(모델에게 노출된 것들).
        context: 실행 컨텍스트(cwd, 세션, abort 시그널, 권한 모드, options 등).
        tool_registry: 도구 레지스트리. tools에서 못 찾았을 때 alias 폴백용(선택).

    Yields:
        StreamEvent: 실행 진행 상황 알림(예: 진행 라벨 표시).
        Message: 최종 tool_result 메시지(성공 또는 <tool_use_error> 오류).
    """
    # tool_use 딕셔너리에서 이름/입력/식별자를 꺼낸다.
    # id가 없으면 객체 메모리 주소 기반으로 임시 id를 만들어 충돌을 피한다.
    tool_name: str = tool_use.get("name", "")
    tool_input: dict[str, Any] = tool_use.get("input", {})
    tool_use_id: str = tool_use.get("id", f"toolu_{id(tool_use):x}")

    # 전체 실행 시간을 재기 위한 시작 시각. 벽시계가 아닌 monotonic(단조 증가) 시계를
    # 써서 시스템 시간이 바뀌어도 경과 시간이 음수가 되지 않도록 한다.
    start_time = time.monotonic()

    # ═══ Step 1: 도구 찾기 ═══
    # 모델이 준 이름을 실제 BaseTool 인스턴스로 해석한다. 3단계 폴백을 순서대로 시도한다.
    tool: BaseTool | None = None

    # 1a. 정확한 이름 매칭 — 가장 흔한 경우. 이름이 딱 맞는 첫 도구를 고른다.
    tool = next((t for t in tools if t.name == tool_name), None)

    # 1b. alias 매칭 — 정식 이름은 아니지만 별칭(aliases)에 걸리는 경우.
    #     대소문자 무시로 비교한다(모델이 대소문자를 틀릴 수 있으므로).
    if tool is None:
        for t in tools:
            if tool_name.lower() in [a.lower() for a in t.aliases]:
                tool = t
                logger.info(f"도구 '{tool_name}' alias로 '{t.name}'에 매칭됨")
                break

    # 1c. ToolRegistry 폴백 — tools 리스트에 없어도 레지스트리 전체에서 한 번 더 찾는다.
    #     레지스트리가 주입되지 않았으면(None) 이 단계는 건너뛴다.
    if tool is None and tool_registry is not None:
        tool = tool_registry.find_tool(tool_name)

    # 세 방법 모두 실패하면 "알 수 없는 도구" 오류를 내고 중단한다(사용 가능 목록을 함께 안내).
    if tool is None:
        async for _ev in _emit_tool_result(
            tool_use_id,
            f"<tool_use_error>알 수 없는 도구: '{tool_name}'. "
            f"사용 가능: {', '.join(t.name for t in tools)}</tool_use_error>",
            is_error=True,
        ):
            yield _ev
        return

    # ═══ Step 2: abort 확인 ═══
    # 실행 직전에 사용자가 취소(Ctrl+C 등)했는지 확인한다. abort_signal은 보통
    # asyncio.Event이며, is_set()이 True면 이미 취소 요청이 들어온 상태다.
    # 시그널이 없거나 is_set 속성이 없으면(오리 타이핑) 이 검사를 안전하게 건너뛴다.
    if context.abort_signal and hasattr(context.abort_signal, "is_set"):
        if context.abort_signal.is_set():
            async for _ev in _emit_tool_result(
                tool_use_id,
                "<tool_use_error>사용자에 의해 실행이 취소되었습니다</tool_use_error>",
                is_error=True,
            ):
                yield _ev
            return

    # ═══ Step 3: JSON Schema 검증 ═══
    # 입력이 도구의 input_schema(형식 명세)에 맞는지 검사한다. 타입/필수 필드 등
    # 형식 수준의 오류를 여기서 먼저 걸러 실제 실행에 잘못된 입력이 들어가지 않게 한다.
    schema_error = _validate_json_schema(tool, tool_input)
    if schema_error:
        async for _ev in _emit_tool_result(
            tool_use_id,
            f"<tool_use_error>InputValidationError: {schema_error}</tool_use_error>",
            is_error=True,
        ):
            yield _ev
        return

    # ═══ Step 4: 도메인 검증 ═══
    # 형식(Schema)만으로는 못 잡는 도구별 비즈니스 규칙을 검사한다. 예: "start_line은
    # end_line보다 작아야 한다" 같은 규칙. 각 BaseTool이 validate_input()으로 구현한다.
    domain_error = tool.validate_input(tool_input)
    if domain_error:
        async for _ev in _emit_tool_result(
            tool_use_id,
            f"<tool_use_error>DomainValidationError: {domain_error}</tool_use_error>",
            is_error=True,
        ):
            yield _ev
        return

    # ═══ Step 5: 보안 분류 (Bash 전용, 비동기) ═══
    # Bash 도구만 명령어 위험 패턴 검사가 필요하다. 이 검사를 지금 바로 기다리지 않고
    # asyncio.create_task로 "백그라운드에서 미리 시작"시켜 둔다. 그러면 아래 권한
    # 확인(8단계)과 병렬로 돌아 전체 지연이 줄어든다. 실제 결과 확인은 9단계 직전에서 한다.
    security_task: asyncio.Task | None = None
    if tool.name == "Bash":
        command = tool_input.get("command", "")
        security_task = asyncio.create_task(_speculative_bash_security(command))

    # ═══ Step 6-8a: 권한 파이프라인 관측 (P1 shadow — 감사 Critical #1~3, 2026-07-03) ═══
    # ★무회귀 핵심★: 이 블록은 "관측 전용"이다. 5계층 파이프라인 판정을 계산해
    # 감사 로그(AuditLogger)에 기록만 하고, 실행 경로(무엇을 차단할지)는 조금도
    # 바꾸지 않는다. 실제 차단은 바로 아래 "현행 로직"(도구 check_permissions의 DENY
    # 만 차단)이 그대로 담당한다.
    #
    # 동작 조건:
    #   - permission_enforcement.enabled=False 또는 파이프라인 미주입(None)이면 이
    #     블록을 통째로 건너뛴다(파이프라인 호출조차 없음). → executor 동작 100% 현행 동일.
    #   - enabled=True + 파이프라인 있음이면 pipeline.check()로 판정을 계산해
    #     AuditLogger에 남긴다(shadow). 이 판정으로는 절대 차단하지 않는다.
    #     (실제 차단 전환은 후속 단계에서 mode="enforce"로 수행한다.)
    _pe = context.options.get("permission_enforcement") or {}
    _pipeline = context.options.get("permission_pipeline")
    _audit_logger = context.options.get("audit_logger")
    # 강제 방식 — "shadow"(기록만) 또는 "enforce"(deny면 실제 차단). 기본 shadow.
    _pe_mode = _pe.get("mode", "shadow")
    if _pe.get("enabled") and _pipeline is not None:
        try:
            # 5계층 파이프라인 판정. shadow에서는 관측용, enforce에서는 실제 게이트다.
            # 이 호출은 tool.check_permissions()를 Layer 2로 한 번 더 수행하지만,
            # check_permissions는 순수 검증이라 부수효과가 없다. Layer 4 Hook는
            # 현재 미배선(None)이다.
            decision = await _pipeline.check(tool, tool_input, context)
            # 파이프라인이 방금 내부 감사 로그에 남긴 엔트리를 JSONL로도 영구 기록한다.
            # (엔트리에는 거친 레이어·사유·모드가 담겨 있어 그대로 재사용이 안전하다.)
            # enforce에서 차단할 때도 "차단됐다는 사실"이 감사 로그에 남도록, 차단
            # 처리보다 먼저 기록한다.
            if _audit_logger is not None:
                recent = _pipeline.get_recent_audit(1)
                if recent:
                    await _audit_logger.async_log_decision(recent[-1])

            _decision_type = getattr(decision, "type", "?")
            # ── enforce 모드: deny 판정이면 여기서 실제로 차단한다 ──
            # ASK/allow는 아직 통과시킨다(ASK 강제는 P3 범위). shadow는 항상 통과.
            if _pe_mode == "enforce" and _decision_type == "deny":
                reason = getattr(decision, "message", "") or "권한 정책 위반"
                logger.info(
                    "[permission enforce] 도구=%s 차단: %s", tool.name, reason
                )
                async for _ev in _emit_tool_result(
                    tool_use_id,
                    f"<tool_use_error>권한 거부: {reason}</tool_use_error>",
                    is_error=True,
                ):
                    yield _ev
                return

            logger.debug(
                "[permission %s] 도구=%s 판정=%s",
                _pe_mode,
                tool.name,
                _decision_type,
            )
        except Exception as e:
            # 관측/강제 파이프라인 자체 실패가 도구 실행을 막으면 안 된다(fail-safe).
            # bare except가 아니라 Exception을 명시해 잡고(anti #8), 결함이 묻히지
            # 않도록 경고로 남긴 뒤 실행을 계속한다. 아래 기존 게이트(도구 자체
            # check_permissions의 DENY)가 최종 안전망으로 여전히 동작한다.
            logger.warning("[permission %s] 파이프라인 처리 실패(무시): %s", _pe_mode, e)

    # ═══ Step 6-8: Hook + 권한 (간소화 — Phase 4에서 전체 구현) ═══
    # 위 shadow/enforce 파이프라인과 별개로, 여기가 현재 "실제로 차단을 담당하는"
    # 최종 안전망이다. 각 도구가 스스로 구현한 check_permissions를 실행해 판정을 받고,
    # 결과가 deny면 실행을 막는다. PreToolUse 훅 등 나머지는 Phase 4에서 붙일 예정.
    try:
        perm_result = await tool.check_permissions(tool_input, context)
        # behavior가 "deny"면 권한 거부 → 오류 결과를 내고 중단.
        if perm_result.behavior.value == "deny":
            async for _ev in _emit_tool_result(
                tool_use_id,
                f"<tool_use_error>권한 거부: {perm_result.message}</tool_use_error>",
                is_error=True,
            ):
                yield _ev
            return
        # behavior가 "ask"면 사용자 확인이 필요하다(예: Bash는 항상 ASK). 대화형
        # 확인 핸들러(ask_handler)가 options에 주입돼 있으면(CLI REPL) 사용자에게 물어
        # 거부 시 차단한다. 핸들러가 없으면(웹·비대화형 nexus ask·기존 테스트) 대화형
        # 프롬프트가 불가능하므로 종전대로 통과시킨다(무회귀 조건). — B-1
        if perm_result.behavior.value == "ask":
            ask_handler = context.options.get("ask_handler")
            if ask_handler is not None:
                # 핸들러는 (tool_name, message) -> bool 코루틴. True=허용, False=거부.
                approved = await ask_handler(tool.name, perm_result.message or "")
                if not approved:
                    async for _ev in _emit_tool_result(
                        tool_use_id,
                        "<tool_use_error>사용자가 도구 실행을 거부했습니다.</tool_use_error>",
                        is_error=True,
                    ):
                        yield _ev
                    return
            elif tool.name not in _ASK_PASSTHROUGH_WARNED:
                # 핸들러 부재 — 통과시키되, 배선 누락을 도구별 1회만 경고로 남긴다.
                _ASK_PASSTHROUGH_WARNED.add(tool.name)
                logger.warning(
                    "[permission ask] 도구=%s ASK 확인 핸들러 미배선 — 통과"
                    "(웹/비대화형). CLI 확인 프롬프트 없음.",
                    tool.name,
                )
    except Exception as e:
        # 권한 확인 자체가 예외로 실패하면 fail-closed 원칙에 따라 실행을 막는다.
        # (안전 판단을 못 했으니 통과시키지 않는다.)
        logger.error(f"권한 확인 실패: {e}")
        async for _ev in _emit_tool_result(
            tool_use_id,
            f"<tool_use_error>권한 확인 에러: {e}</tool_use_error>",
            is_error=True,
        ):
            yield _ev
        return

    # Step 5 결과 확인 (비동기 보안 검사)
    # 5단계에서 백그라운드로 시작해 둔 Bash 보안 검사 결과를 이제 회수한다.
    # 위험 패턴이 걸리면(safe=False) 실행을 차단한다. Bash가 아니면 task가 None이라 건너뛴다.
    if security_task is not None:
        try:
            sec_result = await security_task
            if not sec_result["safe"]:
                async for _ev in _emit_tool_result(
                    tool_use_id,
                    f"<tool_use_error>보안 검사 실패: {sec_result['reason']}</tool_use_error>",
                    is_error=True,
                ):
                    yield _ev
                return
        except Exception as e:
            # 보안 검사 로직 자체의 오류는 도구 차단이 아니라 경고만 남기고 통과시킨다.
            # (검사기 버그로 정상 명령까지 막지 않기 위한 선택. 실제 위험 차단은 위 판정에서.)
            logger.warning(f"보안 검사 에러 (허용): {e}")

    # ═══ Step 9: 도구 실행 ═══
    # 실제 실행 직전, 사용자에게 "무엇을 하는 중인지" 진행 라벨을 먼저 알려준다.
    # (예: "파일 읽는 중: config.yaml") — UI 피드백용 SYSTEM_INFO 이벤트.
    yield StreamEvent(
        type=StreamEventType.SYSTEM_INFO,
        message=tool.get_progress_label(tool_input),
    )

    # 도구 call()에 넘길 실행 컨텍스트를 새로 구성한다. 기존 context를 거의 그대로
    # 복사하되, 이 실행 고유의 tool_use_id를 넣어 도구가 자기 호출을 식별할 수 있게 한다.
    exec_context = ToolUseContext(
        cwd=context.cwd,
        session_id=context.session_id,
        agent_id=context.agent_id,
        tool_use_id=tool_use_id,
        read_file_timestamps=context.read_file_timestamps,
        abort_signal=context.abort_signal,
        permission_mode=context.permission_mode,
        options=context.options,
    )

    # 도구를 실행하되 tool.timeout_seconds 안에 끝나지 않으면 강제로 취소한다.
    # asyncio.wait_for가 시간 초과 시 TimeoutError를 던져 무한 대기를 막아준다.
    try:
        result = await asyncio.wait_for(
            tool.call(tool_input, exec_context),
            timeout=tool.timeout_seconds,
        )
    except TimeoutError:
        elapsed = time.monotonic() - start_time
        async for _ev in _emit_tool_result(
            tool_use_id,
            f"<tool_use_error>도구 '{tool.name}' 타임아웃: "
            f"{tool.timeout_seconds}초 (경과: {elapsed:.1f}초)</tool_use_error>",
            is_error=True,
        ):
            yield _ev
        return
    except Exception as e:
        # 도구 내부에서 난 예외를 삼키지 않고 로그(스택트레이스 포함)로 남긴 뒤,
        # 모델이 이해할 수 있게 예외 타입/메시지를 tool_use_error로 감싸 돌려준다.
        logger.error(f"도구 실행 에러: {tool.name}: {e}", exc_info=True)
        async for _ev in _emit_tool_result(
            tool_use_id,
            f"<tool_use_error>{type(e).__name__}: {e}</tool_use_error>",
            is_error=True,
        ):
            yield _ev
        return

    # ═══ Step 10: 결과 직렬화 ═══
    # 도구가 돌려준 구조화된 결과(ToolResult)를 모델에게 보낼 문자열로 변환한다.
    # 변환 규칙은 각 도구의 map_result()가 정의한다.
    content = tool.map_result(result)

    # ═══ Step 11: 대형 결과 디스크 저장 ═══
    # 결과 문자열이 도구가 허용한 최대 크기를 넘으면, 컨텍스트를 폭파시키지 않도록
    # 전체는 파일로 저장하고 모델에는 앞(head)+뒤(tail) 일부만 보여준다.
    if len(content) > tool.max_result_size:
        # 전체 원본을 디스크에 저장하고 그 경로를 안내에 포함시킨다.
        saved_path = await _save_large_result(content, tool_use_id, tool.name)
        # 앞 1/3, 뒤 1/3만 남기고 가운데는 잘라낸다(생략 표시로 대체).
        head = content[: tool.max_result_size // 3]
        tail = content[-(tool.max_result_size // 3) :]
        content = (
            f"{head}\n\n"
            f"... ({len(content):,}자 전체, "
            f"전체 결과: {saved_path}) ...\n\n"
            f"{tail}"
        )

    # ═══ Step 12-13: 결과 yield ═══
    # 전체 소요 시간을 계산해 로그로 남기고(성능 추적용), 최종 tool_result를 내보낸다.
    # is_error는 도구가 판단한 성공/실패 값을 그대로 전달한다(정상 결과도 실패일 수 있음).
    elapsed = time.monotonic() - start_time
    logger.info(f"도구 '{tool.name}' 완료: {elapsed:.2f}초")

    # 표시용 TOOL_RESULT StreamEvent + 히스토리용 tool_result Message를 함께 내보낸다.
    # (같은 content/is_error를 공유 → UI 요약과 모델 히스토리가 일치)
    async for _ev in _emit_tool_result(
        tool_use_id,
        content,
        is_error=result.is_error,
    ):
        yield _ev


# ─────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────
def _validate_json_schema(tool: BaseTool, input_data: dict[str, Any]) -> str | None:
    """
    입력을 도구의 JSON Schema로 검증한다(3단계 헬퍼).

    반환값 규칙:
      - 문제 없음 → None (호출측은 None이면 통과로 본다).
      - 형식 오류 → 사람이 읽을 수 있는 오류 메시지 문자열.

    jsonschema 라이브러리는 지연 import한다. 라이브러리가 없는 환경(에어갭 최소 배포
    등)에서는 검증을 건너뛰고 None을 반환해 파이프라인이 멈추지 않게 한다.
    """
    try:
        import jsonschema

        # 실제 검증. 스키마 위반 시 예외를 던진다.
        jsonschema.validate(instance=input_data, schema=tool.input_schema)
        return None
    except ImportError:
        # jsonschema가 설치되지 않았으면 검증을 건너뛴다(오류 아님).
        return None
    except Exception as e:
        # 검증 실패. jsonschema 예외는 .message에 상세 사유가 있으니 있으면 그걸 쓴다.
        if hasattr(e, "message"):
            return e.message
        return str(e)


async def _speculative_bash_security(command: str) -> dict[str, Any]:
    """
    Bash 명령어를 정규식으로 훑어 위험 패턴을 사전 차단한다(5단계 헬퍼).

    "speculative(투기적)"라는 이름은, 권한 확인(8단계)이 끝나기 전에 미리 병렬로
    돌려두기 때문이다. 그만큼 전체 파이프라인 지연이 줄어든다.

    반환: {"safe": bool, "reason": str, ("pattern": str)}
      - 위험 패턴 발견 → safe=False + 사유/패턴.
      - 이상 없음      → safe=True + 빈 사유.

    에어갭(폐쇄망) 정책상 curl/wget/ssh 같은 네트워크 명령도 위험으로 본다.
    """
    # (정규식, 사람이 읽을 사유) 쌍의 목록. 하나라도 걸리면 즉시 위험으로 판정한다.
    dangerous_patterns = [
        (r"\brm\s+(-rf?|--recursive)\s+/", "루트 재귀 삭제"),
        (r"\bmkfs\b", "파일시스템 포맷"),
        (r"\bdd\s+.*of=/dev/", "디바이스 직접 쓰기"),
        (r"\bcurl\b|\bwget\b|\bssh\b|\bscp\b", "네트워크 접근 (에어갭 위반)"),
        (r"\bnc\b.*-[lL]", "Netcat 리스너"),
        (r"\bsudo\b", "권한 상승"),
        (r"\bchmod\s+[0-7]*7[0-7]*\s", "전체 쓰기 권한"),
        (r"\bkill\s+-9\s+1\b", "init 프로세스 종료"),
        (r"\bkillall\b", "전체 프로세스 종료"),
    ]

    # 명령어를 각 패턴과 대조한다. 대소문자는 무시(re.IGNORECASE)한다.
    for pattern, reason in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return {"safe": False, "reason": reason, "pattern": pattern}

    # 어떤 위험 패턴에도 걸리지 않았으면 안전으로 판정한다.
    return {"safe": True, "reason": ""}


async def _save_large_result(
    content: str, tool_use_id: str, tool_name: str
) -> str:
    """
    너무 큰 도구 결과 전체를 파일로 저장하고 그 경로를 반환한다(11단계 헬퍼).

    반환된 경로는 모델에게 보여줄 축약본 안내에 "전체 결과: {경로}" 형태로 들어간다.
    파일 쓰기는 블로킹 I/O라 asyncio.to_thread로 별도 스레드에서 수행해
    이벤트 루프(다른 비동기 작업)를 막지 않는다.
    """
    # 저장 디렉토리가 없으면 만든다(이미 있어도 오류 안 남, exist_ok=True).
    LARGE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    # 도구 이름을 파일명에 안전하게 쓰도록 정리한다.
    # 영숫자와 -, _ 만 남기고 나머지 문자는 모두 _ 로 치환한다(경로 안전).
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_name)
    # 도구이름 + 호출 id 조합으로 고유 파일명을 만든다.
    filename = f"{safe_name}_{tool_use_id}.txt"
    path = LARGE_RESULT_DIR / filename

    # 실제 파일 쓰기 동작. UTF-8로 저장해 한글/유니코드가 깨지지 않게 한다.
    def _write():
        path.write_text(content, encoding="utf-8")

    # 블로킹 쓰기를 워커 스레드로 넘겨 실행하고 완료를 기다린다.
    await asyncio.to_thread(_write)
    return str(path)
