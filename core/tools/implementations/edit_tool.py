"""
Edit 도구 — 파일 내용 치환.

파일 내에서 old_string을 찾아 new_string으로 교체한다.
replace_all=False(기본)일 때 old_string이 파일 내에 정확히 1회만 존재해야 한다.
원자적 쓰기(tempfile + os.replace)로 안전하게 교체한다.
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

logger = logging.getLogger("nexus.tools.edit")


class EditTool(BaseTool):
    """
    파일 편집 도구.
    old_string을 찾아 new_string으로 교체한다.
    기본적으로 old_string은 파일 내에서 유일해야 한다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "Edit"

    @property
    def description(self) -> str:
        return "Replace exact string in a file."

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
                "old_string": {
                    "type": "string",
                    "description": "Exact string to find",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement string",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "True면 모든 발생을 교체. 기본 false (유일성 검증)",
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    # ═══ 3. Behavior Flags ═══
    # 쓰기 도구 — fail-closed 기본값 유지

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """필수 필드와 old_string != new_string 검증."""
        file_path = input_data.get("file_path", "")
        if not file_path or not file_path.strip():
            return "file_path는 비어 있을 수 없습니다."

        old_string = input_data.get("old_string", "")
        new_string = input_data.get("new_string", "")
        if old_string == new_string:
            return "old_string과 new_string이 동일합니다. 변경할 내용이 없습니다."

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
            message=f"Edit: {file_path}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        파일에서 old_string을 new_string으로 교체한다.

        처리 순서:
          1. 파일 존재 및 읽기 가능 확인
          2. old_string 존재 횟수 검증
          3. replace_all=False면 유일성 검증 (1회만 존재해야 함)
          4. 문자열 치환
          5. 원자적 쓰기 (tempfile + os.replace)
        """
        file_path = input_data["file_path"]
        old_string = input_data["old_string"]
        new_string = input_data["new_string"]
        replace_all = input_data.get("replace_all", False)

        path = Path(file_path)

        # 1단계: 파일 존재 확인
        if not path.exists():
            return ToolResult.error(f"파일을 찾을 수 없습니다: {file_path}")
        if not path.is_file():
            return ToolResult.error(f"파일이 아닙니다: {file_path}")

        # 파일 내용 읽기
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.error(f"파일을 UTF-8로 읽을 수 없습니다: {file_path}")
        except OSError as e:
            return ToolResult.error(f"파일을 읽을 수 없습니다: {e}")

        # 2단계: old_string 존재 횟수 확인
        count = content.count(old_string)
        if count == 0:
            return ToolResult.error(
                "old_string을 파일에서 찾을 수 없습니다. "
                "정확한 문자열(공백, 들여쓰기 포함)을 확인해주세요."
            )

        # 3단계: 유일성 검증 (replace_all=False일 때)
        if not replace_all and count > 1:
            return ToolResult.error(
                f"old_string이 파일에 {count}회 존재합니다. "
                f"더 많은 주변 컨텍스트를 포함하여 유일하게 만들거나, "
                f"replace_all=true를 사용하세요."
            )

        # 4단계: 문자열 치환
        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            # 첫 번째(유일한) 발생만 교체
            new_content = content.replace(old_string, new_string, 1)

        # 5단계: 원자적 쓰기
        try:
            _atomic_write(path, new_content)
        except OSError as e:
            return ToolResult.error(f"파일 쓰기에 실패했습니다: {e}")

        logger.info("Edit %s: %d replacements", file_path, count if replace_all else 1)

        replacements = count if replace_all else 1
        return ToolResult.success(
            f"편집 완료: {file_path} ({replacements}건 교체)",
            file_path=file_path,
            replacements=replacements,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Editing {input_data.get('file_path', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("file_path", "")


# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def _atomic_write(path: Path, content: str) -> None:
    """
    원자적 파일 쓰기.
    같은 디렉토리에 임시 파일을 만들고 os.replace로 교체한다.
    """
    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.write(tmp_fd, content.encode("utf-8"))
        os.close(tmp_fd)
        tmp_fd = None

        os.replace(tmp_path, str(path))
        tmp_path = None
    finally:
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
