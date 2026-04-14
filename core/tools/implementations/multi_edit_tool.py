"""
MultiEdit 도구 — 여러 파일 일괄 편집.

여러 파일에 대한 편집(old_string → new_string)을 한 번에 수행한다.
각 편집은 EditTool과 동일한 로직(유일성 검증, 원자적 쓰기)을 따른다.
하나라도 실패하면 해당 편집의 에러를 보고하되, 나머지 편집은 계속 진행한다.
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

logger = logging.getLogger("nexus.tools.multi_edit")


class MultiEditTool(BaseTool):
    """
    다중 파일 편집 도구.
    여러 파일에 대한 편집을 배열로 받아 순차적으로 적용한다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "MultiEdit"

    @property
    def description(self) -> str:
        return (
            "여러 파일에 대한 편집을 한 번에 수행합니다. "
            "각 편집은 old_string → new_string 치환입니다."
        )

    @property
    def group(self) -> str:
        return "filesystem"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "편집 목록. 각 항목은 하나의 파일 편집을 정의한다.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "편집할 파일의 절대 경로",
                            },
                            "old_string": {
                                "type": "string",
                                "description": "교체할 기존 문자열",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "교체 후 문자열",
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "True면 모든 발생을 교체 (기본 false)",
                                "default": False,
                            },
                        },
                        "required": ["file_path", "old_string", "new_string"],
                    },
                },
            },
            "required": ["edits"],
        }

    # ═══ 3. Behavior Flags ═══
    # 쓰기 도구 — fail-closed 기본값 유지

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """edits 배열이 비어 있지 않은지 검증한다."""
        edits = input_data.get("edits", [])
        if not edits:
            return "edits 배열이 비어 있습니다. 최소 1개의 편집이 필요합니다."
        # 각 편집의 필수 필드 존재 확인
        for i, edit in enumerate(edits):
            if not edit.get("file_path"):
                return f"edits[{i}]: file_path가 비어 있습니다."
            if edit.get("old_string") == edit.get("new_string"):
                return f"edits[{i}]: old_string과 new_string이 동일합니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """쓰기 도구이므로 사용자에게 확인을 요청한다."""
        edits = input_data.get("edits", [])
        # 편집 대상 파일 목록을 표시
        files = list({e.get("file_path", "") for e in edits})
        file_list = ", ".join(files[:5])
        if len(files) > 5:
            file_list += f" 외 {len(files) - 5}개"
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"MultiEdit: {file_list}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        여러 편집을 순차적으로 적용한다.

        처리 순서:
          1. 각 편집에 대해 파일 읽기
          2. old_string 검증 (존재, 유일성)
          3. 문자열 치환 + 원자적 쓰기
          4. 성공/실패 집계하여 결과 반환
        """
        edits = input_data["edits"]
        results: list[str] = []
        success_count = 0
        error_count = 0

        for i, edit in enumerate(edits):
            file_path = edit["file_path"]
            old_string = edit["old_string"]
            new_string = edit["new_string"]
            replace_all = edit.get("replace_all", False)

            path = Path(file_path)

            # 파일 존재 확인
            if not path.exists() or not path.is_file():
                results.append(f"[{i}] 실패 — 파일 없음: {file_path}")
                error_count += 1
                continue

            # 파일 읽기
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                results.append(f"[{i}] 실패 — 읽기 오류: {file_path}: {e}")
                error_count += 1
                continue

            # old_string 존재 및 유일성 확인
            count = content.count(old_string)
            if count == 0:
                results.append(f"[{i}] 실패 — old_string을 찾을 수 없음: {file_path}")
                error_count += 1
                continue

            if not replace_all and count > 1:
                results.append(
                    f"[{i}] 실패 — old_string이 {count}회 존재 (유일하지 않음): {file_path}"
                )
                error_count += 1
                continue

            # 문자열 치환
            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                new_content = content.replace(old_string, new_string, 1)

            # 원자적 쓰기
            try:
                _atomic_write(path, new_content)
            except OSError as e:
                results.append(f"[{i}] 실패 — 쓰기 오류: {file_path}: {e}")
                error_count += 1
                continue

            replacements = count if replace_all else 1
            results.append(f"[{i}] 성공 — {file_path} ({replacements}건 교체)")
            success_count += 1

        # 최종 결과 조합
        summary = f"MultiEdit 완료: {success_count}건 성공, {error_count}건 실패"
        detail = "\n".join(results)
        result_text = f"{summary}\n\n{detail}"

        logger.info("MultiEdit: %d success, %d error", success_count, error_count)

        if error_count > 0 and success_count == 0:
            return ToolResult.error(result_text)

        return ToolResult.success(
            result_text,
            success_count=success_count,
            error_count=error_count,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        edits = input_data.get("edits", [])
        return f"MultiEdit: {len(edits)}개 편집 적용 중..."


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
