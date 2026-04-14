"""
Bash 도구 — 셸 명령어 실행.

subprocess로 bash 명령어를 실행하고 exit_code, stdout, stderr를 반환한다.
위험한 명령어일 수 있으므로 requires_confirmation=True로 항상 사용자 확인을 요구한다.
에어갭 환경이므로 외부 네트워크 호출(curl, wget 등)은 권한 파이프라인에서 차단된다.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.bash")

# stdout/stderr 최대 캡처 크기 (초과 시 뒷부분만 보존)
_MAX_OUTPUT_SIZE = 50_000


class BashTool(BaseTool):
    """
    셸 명령어 실행 도구.
    지정한 명령어를 bash에서 실행하고 결과를 반환한다.
    항상 사용자 확인이 필요하다 (requires_confirmation=True).
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "Bash"

    @property
    def description(self) -> str:
        return (
            "셸 명령어를 실행합니다. "
            "exit_code, stdout, stderr를 반환합니다. "
            "timeout(초)으로 실행 시간을 제한할 수 있습니다."
        )

    @property
    def group(self) -> str:
        return "execution"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "실행할 셸 명령어",
                },
                "timeout": {
                    "type": "integer",
                    "description": "실행 타임아웃 (초, 기본 120)",
                    "default": 120,
                    "minimum": 1,
                    "maximum": 600,
                },
                "description": {
                    "type": "string",
                    "description": "명령어에 대한 간단한 설명 (UI 표시용)",
                },
            },
            "required": ["command"],
        }

    # ═══ 3. Behavior Flags ═══
    # 가장 위험한 도구 — 항상 확인 필요

    @property
    def requires_confirmation(self) -> bool:
        return True

    @property
    def is_destructive(self) -> bool:
        return True

    # ═══ 4. Limits ═══

    @property
    def timeout_seconds(self) -> float:
        """기본 타임아웃 120초. 입력의 timeout 필드로 오버라이드 가능."""
        return 120.0

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """command가 비어 있는지 검증한다."""
        command = input_data.get("command", "")
        if not command or not command.strip():
            return "command는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """Bash 도구는 항상 사용자에게 확인을 요청한다."""
        command = input_data.get("command", "")
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Run: {command}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        셸 명령어를 실행하고 결과를 반환한다.

        처리 순서:
          1. 작업 디렉토리 결정 (context.cwd)
          2. subprocess.run으로 명령어 실행 (타임아웃 적용)
          3. stdout/stderr 캡처 (너무 길면 뒷부분만 보존)
          4. exit_code와 함께 결과 반환
        """
        command = input_data["command"]
        timeout = input_data.get("timeout", 120)

        # 작업 디렉토리: context.cwd 사용
        cwd = context.cwd

        logger.info("Bash: %s (cwd=%s, timeout=%ds)", command, cwd, timeout)

        try:
            # asyncio에서 블로킹 subprocess를 실행하기 위해 to_thread 사용
            result = await asyncio.to_thread(_run_command, command, cwd, timeout)
        except subprocess.TimeoutExpired:
            return ToolResult.error(
                f"명령어가 {timeout}초 타임아웃을 초과했습니다.",
                command=command,
                timeout=timeout,
            )
        except FileNotFoundError:
            return ToolResult.error(
                f"작업 디렉토리를 찾을 수 없습니다: {cwd}",
                command=command,
            )
        except OSError as e:
            return ToolResult.error(
                f"명령어 실행에 실패했습니다: {e}",
                command=command,
            )

        # 결과 포맷팅
        exit_code = result.returncode
        stdout = _truncate_output(result.stdout or "")
        stderr = _truncate_output(result.stderr or "")

        # 출력 조합
        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        parts.append(f"Exit code: {exit_code}")

        output_text = "\n".join(parts)

        logger.debug(
            "Bash exit=%d, stdout=%d chars, stderr=%d chars", exit_code, len(stdout), len(stderr)
        )

        if exit_code != 0:
            return ToolResult.success(
                output_text,
                exit_code=exit_code,
                command=command,
            )

        return ToolResult.success(
            output_text,
            exit_code=exit_code,
            command=command,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        desc = input_data.get("description", "")
        if desc:
            return desc
        command = input_data.get("command", "")
        # 명령어가 길면 앞부분만 표시
        if len(command) > 60:
            return command[:57] + "..."
        return command

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("command", "")


# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def _run_command(command: str, cwd: str, timeout: int) -> subprocess.CompletedProcess:
    """
    subprocess.run으로 명령어를 실행한다.
    블로킹 함수이므로 asyncio.to_thread에서 호출해야 한다.
    """
    # 셸 환경 구성: 현재 환경 변수를 상속하되 인터랙티브 프롬프트 비활성화
    env = os.environ.copy()
    env["TERM"] = "dumb"  # 색상 코드 비활성화

    return subprocess.run(  # noqa: S602 — Bash 도구는 shell=True가 의도된 동작
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _truncate_output(output: str) -> str:
    """
    출력이 너무 길면 뒷부분만 보존한다.
    앞부분은 '... (truncated)' 메시지로 대체한다.
    """
    if len(output) <= _MAX_OUTPUT_SIZE:
        return output
    # 뒷부분을 보존 (최근 출력이 더 중요)
    removed = len(output) - _MAX_OUTPUT_SIZE
    return f"... (앞부분 생략, {removed}자 제거)\n{output[-_MAX_OUTPUT_SIZE:]}"
