"""
Read 도구 — 파일 읽기.

로컬 파일 시스템에서 파일을 읽어 cat -n 형식(라인 번호 포함)으로 반환한다.
바이너리 파일을 감지하고, 한국어 인코딩(euc-kr, cp949) 폴백을 지원한다.
읽기 전용 도구이므로 is_read_only=True, is_concurrency_safe=True로 완화한다.
"""

from __future__ import annotations

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

logger = logging.getLogger("nexus.tools.read")

# 바이너리 감지에 사용할 NULL 바이트 비율 임계값
_BINARY_CHECK_SIZE = 8192
_NULL_THRESHOLD = 0.01  # 1% 이상 NULL 바이트가 있으면 바이너리로 판정

# 인코딩 폴백 순서: utf-8 → euc-kr → cp949 → latin-1(항상 성공)
_ENCODING_FALLBACKS = ("utf-8", "euc-kr", "cp949", "latin-1")


class ReadTool(BaseTool):
    """
    파일 읽기 도구.
    지정한 파일을 읽어서 라인 번호와 함께 반환한다.
    offset과 limit으로 읽을 범위를 조절할 수 있다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "Read"

    @property
    def description(self) -> str:
        return "Read file contents with optional line range."

    @property
    def group(self) -> str:
        return "filesystem"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute file path",
                },
                "offset": {
                    "type": "integer",
                    "description": "Start line number (0-based)",
                    "default": 0,
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read",
                    "default": 2000,
                    "minimum": 1,
                },
            },
            "required": ["file_path"],
        }

    # ═══ 3. Behavior Flags ═══
    # 읽기 전용이므로 안전하게 완화

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """file_path가 비어 있는지 검증한다."""
        file_path = input_data.get("file_path", "")
        if not file_path or not file_path.strip():
            return "file_path는 비어 있을 수 없습니다."
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
        파일을 읽어서 cat -n 형식으로 반환한다.

        처리 순서:
          1. 파일 존재 여부 확인
          2. 바이너리 파일인지 검사
          3. 인코딩 폴백으로 텍스트 디코딩
          4. offset/limit에 맞게 라인 슬라이싱
          5. 라인 번호를 붙여서 반환
        """
        file_path = input_data["file_path"]
        offset = input_data.get("offset", 0)
        limit = input_data.get("limit", 2000)

        path = Path(file_path)

        # 1단계: 파일 존재 여부 확인
        if not path.exists():
            return ToolResult.error(f"파일을 찾을 수 없습니다: {file_path}")

        if not path.is_file():
            return ToolResult.error(f"파일이 아닙니다 (디렉토리일 수 있습니다): {file_path}")

        # 2단계: 바이너리 파일인지 검사
        try:
            raw_head = path.read_bytes()[:_BINARY_CHECK_SIZE]
        except OSError as e:
            return ToolResult.error(f"파일을 읽을 수 없습니다: {e}")

        if _is_binary(raw_head):
            # 바이너리 파일은 크기 정보만 반환
            size = path.stat().st_size
            return ToolResult.success(
                f"바이너리 파일입니다 ({_format_size(size)}). 텍스트로 표시할 수 없습니다.",
                binary=True,
                file_path=file_path,
                size=size,
            )

        # 3단계: 인코딩 폴백으로 전체 파일 읽기
        raw_bytes = path.read_bytes()
        text = _decode_with_fallback(raw_bytes)

        # 4단계: 라인 슬라이싱
        lines = text.splitlines()
        total_lines = len(lines)
        sliced = lines[offset : offset + limit]

        # 5단계: cat -n 형식 (라인 번호 + 탭 + 내용)
        numbered_lines = []
        for i, line in enumerate(sliced, start=offset + 1):
            numbered_lines.append(f"{i}\t{line}")

        result_text = "\n".join(numbered_lines)

        # 범위를 벗어난 경우 안내 메시지 추가
        if offset + limit < total_lines:
            result_text += (
                f"\n\n... (총 {total_lines}줄 중 {offset + 1}~{offset + len(sliced)}줄 표시)"
            )

        # read_file_timestamps에 읽은 시간 기록 (Write/Edit 도구의 충돌 감지용)
        context.read_file_timestamps[file_path] = os.path.getmtime(file_path)

        logger.debug(
            "Read %s: offset=%d, limit=%d, total=%d",
            file_path,
            offset,
            limit,
            total_lines,
        )

        return ToolResult.success(
            result_text,
            file_path=file_path,
            total_lines=total_lines,
            lines_shown=len(sliced),
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Reading {input_data.get('file_path', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("file_path", "")


# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def _is_binary(data: bytes) -> bool:
    """바이너리 데이터인지 판별한다. NULL 바이트 비율로 판단."""
    if not data:
        return False
    null_count = data.count(b"\x00")
    return (null_count / len(data)) > _NULL_THRESHOLD


def _decode_with_fallback(data: bytes) -> str:
    """
    여러 인코딩을 순서대로 시도하여 텍스트로 디코딩한다.
    한국어 환경을 위해 euc-kr, cp949도 시도한다.
    latin-1은 항상 성공하므로 최종 폴백이다.
    """
    for encoding in _ENCODING_FALLBACKS:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    # latin-1은 항상 성공하므로 여기에 도달하지 않지만, 안전장치
    return data.decode("latin-1")


def _format_size(size: int) -> str:
    """바이트 크기를 사람이 읽기 쉬운 형식으로 변환한다."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}{unit}"
        size /= 1024
    return f"{size:.1f}TB"
