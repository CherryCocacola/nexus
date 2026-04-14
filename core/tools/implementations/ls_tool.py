"""
LS 도구 — 디렉토리 목록 표시.

지정한 디렉토리의 파일/폴더 목록을 표시한다.
각 항목의 크기, 수정 시간, 유형(파일/디렉토리)을 포함한다.
읽기 전용이므로 is_read_only=True, is_concurrency_safe=True로 완화한다.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.ls")


class LSTool(BaseTool):
    """
    디렉토리 목록 표시 도구.
    파일/폴더의 이름, 크기, 수정 시간을 표시한다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "LS"

    @property
    def description(self) -> str:
        return "디렉토리의 파일/폴더 목록을 표시합니다. 크기와 수정 시간을 포함합니다."

    @property
    def group(self) -> str:
        return "filesystem"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "표시할 디렉토리의 절대 경로",
                },
            },
            "required": ["path"],
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
        """path가 비어 있는지 검증한다."""
        path = input_data.get("path", "")
        if not path or not path.strip():
            return "path는 비어 있을 수 없습니다."
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
        디렉토리 목록을 표시한다.

        처리 순서:
          1. 디렉토리 존재 여부 확인
          2. os.scandir로 항목 읽기
          3. 각 항목의 타입, 크기, 수정 시간 수집
          4. 이름순 정렬 (디렉토리 먼저)
          5. 포맷팅하여 반환
        """
        dir_path = input_data["path"]
        path = Path(dir_path)

        # 1단계: 디렉토리 존재 확인
        if not path.exists():
            return ToolResult.error(f"경로를 찾을 수 없습니다: {dir_path}")
        if not path.is_dir():
            return ToolResult.error(f"디렉토리가 아닙니다: {dir_path}")

        # 2단계: 디렉토리 항목 읽기
        try:
            entries = list(os.scandir(dir_path))
        except PermissionError:
            return ToolResult.error(f"디렉토리 접근 권한이 없습니다: {dir_path}")
        except OSError as e:
            return ToolResult.error(f"디렉토리를 읽을 수 없습니다: {e}")

        if not entries:
            return ToolResult.success(
                f"빈 디렉토리입니다: {dir_path}",
                total=0,
            )

        # 3단계: 각 항목 정보 수집
        items: list[_DirEntry] = []
        for entry in entries:
            try:
                st = entry.stat()
                is_dir = entry.is_dir()
                size = st.st_size if not is_dir else 0
                mtime = st.st_mtime
                items.append(
                    _DirEntry(
                        name=entry.name,
                        is_dir=is_dir,
                        size=size,
                        mtime=mtime,
                    )
                )
            except OSError:
                # 심볼릭 링크 깨짐 등의 경우
                items.append(
                    _DirEntry(
                        name=entry.name,
                        is_dir=False,
                        size=0,
                        mtime=0.0,
                        error=True,
                    )
                )

        # 4단계: 정렬 — 디렉토리 먼저, 이름순
        items.sort(key=lambda e: (not e.is_dir, e.name.lower()))

        # 5단계: 포맷팅
        lines: list[str] = []
        dir_count = 0
        file_count = 0

        for item in items:
            if item.error:
                lines.append(f"  ?  {item.name} (접근 불가)")
                continue

            if item.is_dir:
                # 디렉토리: 이름 뒤에 / 표시
                mtime_str = _format_mtime(item.mtime)
                lines.append(f"  d  {mtime_str}  {'':>10}  {item.name}/")
                dir_count += 1
            else:
                # 파일: 크기와 수정 시간 표시
                mtime_str = _format_mtime(item.mtime)
                size_str = _format_size(item.size)
                lines.append(f"  -  {mtime_str}  {size_str:>10}  {item.name}")
                file_count += 1

        # 헤더와 요약 추가
        header = f"디렉토리: {dir_path}"
        summary = f"합계: {dir_count}개 디렉토리, {file_count}개 파일"
        result_text = f"{header}\n\n" + "\n".join(lines) + f"\n\n{summary}"

        logger.debug("LS %s: %d dirs, %d files", dir_path, dir_count, file_count)

        return ToolResult.success(
            result_text,
            total=dir_count + file_count,
            directories=dir_count,
            files=file_count,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Listing {input_data.get('path', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("path", "")


# ─────────────────────────────────────────────
# 내부 데이터 클래스
# ─────────────────────────────────────────────
class _DirEntry:
    """디렉토리 항목 정보를 담는 간단한 데이터 클래스."""

    __slots__ = ("name", "is_dir", "size", "mtime", "error")

    def __init__(
        self,
        name: str,
        is_dir: bool,
        size: int,
        mtime: float,
        error: bool = False,
    ) -> None:
        self.name = name
        self.is_dir = is_dir
        self.size = size
        self.mtime = mtime
        self.error = error


# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def _format_size(size: int) -> str:
    """바이트 크기를 사람이 읽기 쉬운 형식으로 변환한다."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            if unit == "B":
                return f"{size}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _format_mtime(mtime: float) -> str:
    """수정 시간을 YYYY-MM-DD HH:MM 형식으로 변환한다."""
    if mtime == 0.0:
        return "                "  # 16자리 공백 (정렬용)
    dt = datetime.fromtimestamp(mtime, tz=UTC)
    return dt.strftime("%Y-%m-%d %H:%M")
