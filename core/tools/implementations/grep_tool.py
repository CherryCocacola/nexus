"""
Grep 도구 — 파일 내용 검색.

정규표현식 패턴으로 파일 내용을 검색한다.
ripgrep(rg)을 우선 사용하고, 없으면 grep으로 폴백한다.
3가지 출력 모드: content(매칭 라인), files_with_matches(파일 경로), count(매칭 수).
읽기 전용이므로 is_read_only=True, is_concurrency_safe=True로 완화한다.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.grep")

# 기본 결과 제한
_DEFAULT_HEAD_LIMIT = 250
_MAX_OUTPUT_SIZE = 100_000  # 최대 출력 크기 (문자 수)


class GrepTool(BaseTool):
    """
    파일 내용 검색 도구.
    ripgrep(rg) 우선, grep 폴백으로 정규표현식 패턴을 검색한다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "Grep"

    @property
    def description(self) -> str:
        return (
            "정규표현식 패턴으로 파일 내용을 검색합니다. "
            "output_mode로 출력 형식을 선택할 수 있습니다: "
            "content, files_with_matches, count."
        )

    @property
    def group(self) -> str:
        return "search"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "검색할 정규표현식 패턴",
                },
                "path": {
                    "type": "string",
                    "description": "검색할 파일 또는 디렉토리 (기본: 현재 작업 디렉토리)",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "출력 모드 (기본: files_with_matches)",
                    "default": "files_with_matches",
                },
                "glob": {
                    "type": "string",
                    "description": "파일 필터 glob 패턴 (예: *.py, *.{ts,tsx})",
                },
                "head_limit": {
                    "type": "integer",
                    "description": "결과 최대 줄 수 (기본 250)",
                    "default": 250,
                    "minimum": 1,
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "대소문자 무시 검색 (기본 false)",
                    "default": False,
                },
            },
            "required": ["pattern"],
        }

    # ═══ 3. Behavior Flags ═══
    # 읽기 전용 — 안전하게 완화

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """pattern이 비어 있는지 검증한다."""
        pattern = input_data.get("pattern", "")
        if not pattern or not pattern.strip():
            return "pattern은 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """읽기 도구이므로 항상 허용한다."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        정규표현식 패턴으로 파일 내용을 검색한다.

        처리 순서:
          1. ripgrep(rg) 사용 가능 여부 확인
          2. 검색 명령어 구성
          3. subprocess로 실행
          4. head_limit에 맞게 결과 자르기
          5. 결과 반환
        """
        pattern = input_data["pattern"]
        search_path = input_data.get("path", context.cwd)
        output_mode = input_data.get("output_mode", "files_with_matches")
        glob_filter = input_data.get("glob")
        head_limit = input_data.get("head_limit", _DEFAULT_HEAD_LIMIT)
        case_insensitive = input_data.get("case_insensitive", False)

        # ripgrep이 있는지 확인
        rg_path = shutil.which("rg")
        use_rg = rg_path is not None

        if use_rg:
            cmd = _build_rg_command(
                pattern,
                search_path,
                output_mode,
                glob_filter,
                head_limit,
                case_insensitive,
            )
        else:
            cmd = _build_grep_command(
                pattern,
                search_path,
                output_mode,
                glob_filter,
                case_insensitive,
            )

        logger.debug("Grep command: %s", " ".join(cmd))

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=context.cwd,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error("검색이 60초 타임아웃을 초과했습니다.")
        except FileNotFoundError:
            return ToolResult.error(
                "검색 도구를 찾을 수 없습니다. rg 또는 grep이 설치되어 있는지 확인하세요."
            )
        except OSError as e:
            return ToolResult.error(f"검색 실행 실패: {e}")

        # exit code: 0=결과 있음, 1=결과 없음, 2+=에러
        if result.returncode >= 2:
            stderr = result.stderr.strip()
            return ToolResult.error(f"검색 오류: {stderr or '알 수 없는 오류'}")

        output = result.stdout or ""

        # 결과가 없는 경우
        if not output.strip():
            return ToolResult.success(
                f"패턴 '{pattern}'에 일치하는 결과가 없습니다.",
                total=0,
            )

        # head_limit 적용 (rg가 아닌 경우)
        lines = output.splitlines()
        total_lines = len(lines)
        if not use_rg and total_lines > head_limit:
            lines = lines[:head_limit]
            output = "\n".join(lines)
            output += f"\n\n... (총 {total_lines}줄 중 {head_limit}줄만 표시)"

        # 출력 크기 제한
        if len(output) > _MAX_OUTPUT_SIZE:
            output = output[:_MAX_OUTPUT_SIZE] + "\n... (출력 크기 초과로 잘림)"

        return ToolResult.success(
            output,
            total=total_lines,
            tool_used="rg" if use_rg else "grep",
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Searching for '{input_data.get('pattern', '...')}'"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("pattern", "")


# ─────────────────────────────────────────────
# ripgrep 명령어 구성
# ─────────────────────────────────────────────
def _build_rg_command(
    pattern: str,
    search_path: str,
    output_mode: str,
    glob_filter: str | None,
    head_limit: int,
    case_insensitive: bool,
) -> list[str]:
    """ripgrep(rg) 명령어를 구성한다."""
    cmd = ["rg"]

    # 출력 모드에 따른 옵션
    if output_mode == "files_with_matches":
        cmd.append("--files-with-matches")
    elif output_mode == "count":
        cmd.append("--count")
    else:
        # content 모드: 라인 번호 표시
        cmd.append("--line-number")

    # 대소문자 무시
    if case_insensitive:
        cmd.append("--ignore-case")

    # glob 필터
    if glob_filter:
        cmd.extend(["--glob", glob_filter])

    # 결과 수 제한 (content 모드에서만 유효)
    if output_mode != "count":
        cmd.extend(["--max-count", str(head_limit)])

    # 패턴과 검색 경로
    cmd.append(pattern)
    cmd.append(search_path)

    return cmd


# ─────────────────────────────────────────────
# grep 폴백 명령어 구성
# ─────────────────────────────────────────────
def _build_grep_command(
    pattern: str,
    search_path: str,
    output_mode: str,
    glob_filter: str | None,
    case_insensitive: bool,
) -> list[str]:
    """grep 폴백 명령어를 구성한다. rg가 없을 때 사용."""
    cmd = ["grep", "--recursive"]

    # 출력 모드
    if output_mode == "files_with_matches":
        cmd.append("--files-with-matches")
    elif output_mode == "count":
        cmd.append("--count")
    else:
        cmd.append("--line-number")

    # 대소문자 무시
    if case_insensitive:
        cmd.append("--ignore-case")

    # glob 필터 (grep의 --include 옵션)
    if glob_filter:
        cmd.extend(["--include", glob_filter])

    # 바이너리 파일 무시
    cmd.append("--binary-files=without-match")

    # 패턴과 검색 경로
    cmd.extend(["--regexp", pattern, search_path])

    return cmd
