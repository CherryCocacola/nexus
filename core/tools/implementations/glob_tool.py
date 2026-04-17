"""
Glob 도구 — 파일 패턴 검색.

glob 패턴(예: **/*.py)으로 파일을 검색하고, 수정 시간순(최신 먼저)으로 정렬하여 반환한다.
최대 250개 결과로 제한한다.
읽기 전용이므로 is_read_only=True, is_concurrency_safe=True로 완화한다.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.glob")

# 검색 결과 최대 개수
_MAX_RESULTS = 250


class GlobTool(BaseTool):
    """
    파일 패턴 검색 도구.
    glob 패턴으로 파일을 찾아 수정 시간순(최신 먼저)으로 반환한다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "Glob"

    @property
    def description(self) -> str:
        return "Find files matching a glob pattern."

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
                    "description": "Glob pattern (e.g. **/*.py)",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory path",
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
        glob 패턴으로 파일을 검색한다.

        처리 순서:
          1. 검색 디렉토리 결정 (path 또는 context.cwd)
          2. glob.glob으로 재귀 검색
          3. 파일만 필터링 (디렉토리 제외)
          4. 수정 시간순 정렬 (최신 먼저)
          5. 최대 250개로 제한하여 반환
        """
        pattern = input_data["pattern"]
        search_dir = input_data.get("path", context.cwd)

        # 검색 디렉토리 유효성 확인
        search_path = Path(search_dir)
        if not search_path.exists():
            return ToolResult.error(f"디렉토리를 찾을 수 없습니다: {search_dir}")
        if not search_path.is_dir():
            return ToolResult.error(f"디렉토리가 아닙니다: {search_dir}")

        # glob 검색: 검색 디렉토리를 기준으로 패턴 적용
        full_pattern = os.path.join(search_dir, pattern)
        try:
            matches = glob.glob(full_pattern, recursive=True)
        except Exception as e:
            return ToolResult.error(f"glob 검색 실패: {e}")

        # 파일만 필터링 (디렉토리 제외)
        files = [m for m in matches if os.path.isfile(m)]

        # 수정 시간순 정렬 (최신 먼저)
        # 파일이 삭제된 경우를 대비하여 안전하게 처리
        def _safe_mtime(filepath: str) -> float:
            try:
                return os.path.getmtime(filepath)
            except OSError:
                return 0.0

        files.sort(key=_safe_mtime, reverse=True)

        total_found = len(files)
        # 최대 개수 제한
        files = files[:_MAX_RESULTS]

        # 결과 포맷팅
        if not files:
            return ToolResult.success(
                f"패턴 '{pattern}'에 일치하는 파일이 없습니다.",
                total=0,
            )

        # 각 파일 경로를 줄바꿈으로 구분하여 반환
        result_lines = "\n".join(files)
        if total_found > _MAX_RESULTS:
            result_lines += f"\n\n... (총 {total_found}개 중 {_MAX_RESULTS}개만 표시)"

        logger.debug("Glob '%s' in '%s': %d files found", pattern, search_dir, total_found)

        return ToolResult.success(
            result_lines,
            total=total_found,
            shown=len(files),
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Searching {input_data.get('pattern', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("pattern", "")
