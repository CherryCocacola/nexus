"""
도구 실행 파이프라인 — 13단계 실행 라이프사이클.

Claude Code의 toolExecution.ts (337~1600행)를 재구현한다.
하나의 tool_use 블록을 받아서 검증 → 권한 → 실행 → 결과 반환을 수행한다.

13단계:
  1. 도구 찾기 (alias 폴백)
  2. abort 확인
  3. JSON Schema 검증
  4. 도메인 검증 (도구별 커스텀)
  5. 보안 분류 (Bash 전용, 비동기)
  6. 입력 backfill (Hook용 관찰 가능 입력)
  7. PreToolUse 훅
  8. 권한 해결
  9. 도구 실행 (tool.call())
  10. 결과 직렬화 (map_result)
  11. 대형 결과 디스크 저장
  12. PostToolUse 훅
  13. 결과 yield
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from core.message import Message, StreamEvent, StreamEventType
from core.tools.base import (
    BaseTool,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.executor")

# 대형 결과 저장 디렉토리
LARGE_RESULT_DIR = Path.home() / ".nexus" / "tool_results"


async def run_tool_use(
    tool_use: dict[str, Any],
    tools: list[BaseTool],
    context: ToolUseContext,
    tool_registry: Any | None = None,
) -> AsyncGenerator[StreamEvent | Message, None]:
    """
    13단계 도구 실행 파이프라인.

    Args:
        tool_use: {"id": "...", "name": "...", "input": {...}}
        tools: 사용 가능한 도구 리스트
        context: 실행 컨텍스트
        tool_registry: 도구 레지스트리 (alias 조회용, 선택)

    Yields:
        StreamEvent: 진행 이벤트
        Message: tool_result 메시지
    """
    tool_name: str = tool_use.get("name", "")
    tool_input: dict[str, Any] = tool_use.get("input", {})
    tool_use_id: str = tool_use.get("id", f"toolu_{id(tool_use):x}")

    start_time = time.monotonic()

    # ═══ Step 1: 도구 찾기 ═══
    tool: BaseTool | None = None

    # 1a. 정확한 이름 매칭
    tool = next((t for t in tools if t.name == tool_name), None)

    # 1b. alias 매칭
    if tool is None:
        for t in tools:
            if tool_name.lower() in [a.lower() for a in t.aliases]:
                tool = t
                logger.info(f"도구 '{tool_name}' alias로 '{t.name}'에 매칭됨")
                break

    # 1c. ToolRegistry 폴백
    if tool is None and tool_registry is not None:
        tool = tool_registry.find_tool(tool_name)

    if tool is None:
        yield Message.tool_result(
            tool_use_id,
            f"<tool_use_error>알 수 없는 도구: '{tool_name}'. "
            f"사용 가능: {', '.join(t.name for t in tools)}</tool_use_error>",
            is_error=True,
        )
        return

    # ═══ Step 2: abort 확인 ═══
    if context.abort_signal and hasattr(context.abort_signal, "is_set"):
        if context.abort_signal.is_set():
            yield Message.tool_result(
                tool_use_id,
                "<tool_use_error>사용자에 의해 실행이 취소되었습니다</tool_use_error>",
                is_error=True,
            )
            return

    # ═══ Step 3: JSON Schema 검증 ═══
    schema_error = _validate_json_schema(tool, tool_input)
    if schema_error:
        yield Message.tool_result(
            tool_use_id,
            f"<tool_use_error>InputValidationError: {schema_error}</tool_use_error>",
            is_error=True,
        )
        return

    # ═══ Step 4: 도메인 검증 ═══
    domain_error = tool.validate_input(tool_input)
    if domain_error:
        yield Message.tool_result(
            tool_use_id,
            f"<tool_use_error>DomainValidationError: {domain_error}</tool_use_error>",
            is_error=True,
        )
        return

    # ═══ Step 5: 보안 분류 (Bash 전용, 비동기) ═══
    security_task: asyncio.Task | None = None
    if tool.name == "Bash":
        command = tool_input.get("command", "")
        security_task = asyncio.create_task(_speculative_bash_security(command))

    # ═══ Step 6-8: Hook + 권한 (간소화 — Phase 4에서 전체 구현) ═══
    # 현재는 도구 자체의 check_permissions만 실행
    try:
        perm_result = await tool.check_permissions(tool_input, context)
        if perm_result.behavior.value == "deny":
            yield Message.tool_result(
                tool_use_id,
                f"<tool_use_error>권한 거부: {perm_result.message}</tool_use_error>",
                is_error=True,
            )
            return
    except Exception as e:
        logger.error(f"권한 확인 실패: {e}")
        yield Message.tool_result(
            tool_use_id,
            f"<tool_use_error>권한 확인 에러: {e}</tool_use_error>",
            is_error=True,
        )
        return

    # Step 5 결과 확인 (비동기 보안 검사)
    if security_task is not None:
        try:
            sec_result = await security_task
            if not sec_result["safe"]:
                yield Message.tool_result(
                    tool_use_id,
                    f"<tool_use_error>보안 검사 실패: {sec_result['reason']}</tool_use_error>",
                    is_error=True,
                )
                return
        except Exception as e:
            logger.warning(f"보안 검사 에러 (허용): {e}")

    # ═══ Step 9: 도구 실행 ═══
    yield StreamEvent(
        type=StreamEventType.SYSTEM_INFO,
        message=tool.get_progress_label(tool_input),
    )

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

    try:
        result = await asyncio.wait_for(
            tool.call(tool_input, exec_context),
            timeout=tool.timeout_seconds,
        )
    except TimeoutError:
        elapsed = time.monotonic() - start_time
        yield Message.tool_result(
            tool_use_id,
            f"<tool_use_error>도구 '{tool.name}' 타임아웃: "
            f"{tool.timeout_seconds}초 (경과: {elapsed:.1f}초)</tool_use_error>",
            is_error=True,
        )
        return
    except Exception as e:
        logger.error(f"도구 실행 에러: {tool.name}: {e}", exc_info=True)
        yield Message.tool_result(
            tool_use_id,
            f"<tool_use_error>{type(e).__name__}: {e}</tool_use_error>",
            is_error=True,
        )
        return

    # ═══ Step 10: 결과 직렬화 ═══
    content = tool.map_result(result)

    # ═══ Step 11: 대형 결과 디스크 저장 ═══
    if len(content) > tool.max_result_size:
        saved_path = await _save_large_result(content, tool_use_id, tool.name)
        head = content[: tool.max_result_size // 3]
        tail = content[-(tool.max_result_size // 3) :]
        content = (
            f"{head}\n\n"
            f"... ({len(content):,}자 전체, "
            f"전체 결과: {saved_path}) ...\n\n"
            f"{tail}"
        )

    # ═══ Step 12-13: 결과 yield ═══
    elapsed = time.monotonic() - start_time
    logger.info(f"도구 '{tool.name}' 완료: {elapsed:.2f}초")

    yield Message.tool_result(
        tool_use_id,
        content,
        is_error=result.is_error,
    )


# ─────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────
def _validate_json_schema(tool: BaseTool, input_data: dict[str, Any]) -> str | None:
    """JSON Schema 기반 입력 검증."""
    try:
        import jsonschema

        jsonschema.validate(instance=input_data, schema=tool.input_schema)
        return None
    except ImportError:
        # jsonschema가 없으면 검증 건너뛰기
        return None
    except Exception as e:
        if hasattr(e, "message"):
            return e.message
        return str(e)


async def _speculative_bash_security(command: str) -> dict[str, Any]:
    """
    Bash 명령어의 보안 분류.
    위험 패턴을 감지하여 실행 전에 차단한다.
    Step 8(권한)과 병렬로 실행하여 전체 파이프라인 지연을 줄인다.
    """
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

    for pattern, reason in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return {"safe": False, "reason": reason, "pattern": pattern}

    return {"safe": True, "reason": ""}


async def _save_large_result(
    content: str, tool_use_id: str, tool_name: str
) -> str:
    """대형 도구 결과를 디스크에 저장한다."""
    LARGE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_name)
    filename = f"{safe_name}_{tool_use_id}.txt"
    path = LARGE_RESULT_DIR / filename

    def _write():
        path.write_text(content, encoding="utf-8")

    await asyncio.to_thread(_write)
    return str(path)
