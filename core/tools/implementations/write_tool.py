"""
Write 도구 — 파일 쓰기.

파일에 내용을 원자적으로 쓴다. tempfile에 먼저 쓴 후 os.replace로 교체하여
쓰기 도중 충돌/정전 시에도 파일이 손상되지 않도록 보장한다.
부모 디렉토리가 없으면 자동으로 생성한다.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.write")


class WriteTool(BaseTool):
    """
    파일 쓰기 도구.
    지정한 경로에 내용을 원자적으로 쓴다.
    기존 파일이 있으면 덮어쓴다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "Write"

    @property
    def description(self) -> str:
        return "Create or overwrite a file."

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
                "content": {
                    "type": "string",
                    "description": "File content to write",
                },
            },
            "required": ["file_path", "content"],
        }

    # ═══ 3. Behavior Flags ═══
    # 쓰기 도구 — fail-closed 기본값 유지, is_destructive만 True로 설정

    @property
    def is_destructive(self) -> bool:
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
        """쓰기 도구이므로 사용자에게 확인을 요청한다."""
        file_path = input_data.get("file_path", "")
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Write to: {file_path}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        파일을 원자적으로 쓴다.

        처리 순서:
          1. 부모 디렉토리 생성 (없으면)
          2. 같은 디렉토리에 임시 파일 생성 후 내용 쓰기
          3. os.replace로 원자적 교체 (같은 파일시스템이어야 원자적)
          4. 실패 시 임시 파일 정리
        """
        file_path = input_data["file_path"]
        content = input_data["content"]

        path = Path(file_path)

        # 1단계: 부모 디렉토리 자동 생성
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return ToolResult.error(f"디렉토리를 생성할 수 없습니다: {e}")

        # 2단계: 임시 파일에 먼저 쓰기 (원자적 쓰기 보장)
        tmp_fd = None
        tmp_path = None
        try:
            # 같은 디렉토리에 임시 파일 생성 (os.replace 원자성 보장)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            # 파일 디스크립터로 직접 쓰기
            os.write(tmp_fd, content.encode("utf-8"))
            os.close(tmp_fd)
            tmp_fd = None  # close 완료 표시

            # 3단계: 원자적 교체
            os.replace(tmp_path, file_path)
            tmp_path = None  # replace 완료 표시

        except OSError as e:
            return ToolResult.error(f"파일 쓰기에 실패했습니다: {e}")
        finally:
            # 4단계: 실패 시 정리
            if tmp_fd is not None:
                os.close(tmp_fd)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # 결과 통계 계산
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        byte_count = len(content.encode("utf-8"))

        logger.info("Write %s: %d lines, %d bytes", file_path, line_count, byte_count)

        return ToolResult.success(
            f"파일을 작성했습니다: {file_path} ({line_count}줄, {byte_count}바이트)",
            file_path=file_path,
            lines=line_count,
            bytes=byte_count,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Writing {input_data.get('file_path', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("file_path", "")
