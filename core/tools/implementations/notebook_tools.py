"""
Notebook 도구 모음 — Jupyter Notebook (.ipynb) 파일 조작.

2개의 노트북 도구를 제공한다:
  - NotebookRead: .ipynb 파일을 파싱하여 셀 내용을 읽기 쉽게 표시 (읽기 전용)
  - NotebookEdit: .ipynb 파일의 특정 셀을 수정

Jupyter Notebook은 JSON 형식이므로 json 모듈로 파싱한다.
에어갭 환경이므로 Jupyter 서버 연결 없이 파일 직접 조작만 수행한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.notebook")


def _resolve_notebook_path(file_path: str, cwd: str) -> Path:
    """
    노트북 파일의 절대 경로를 계산한다.
    상대 경로이면 cwd를 기준으로 해석한다.
    """
    p = Path(file_path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


def _format_cell(index: int, cell: dict[str, Any]) -> str:
    """
    하나의 노트북 셀을 사람이 읽기 쉬운 형식으로 변환한다.

    Args:
        index: 셀 번호 (0부터 시작)
        cell: 노트북 JSON의 cell 딕셔너리

    Returns:
        포맷된 셀 문자열
    """
    cell_type = cell.get("cell_type", "unknown")
    # source는 문자열 리스트 또는 단일 문자열
    source = cell.get("source", [])
    if isinstance(source, list):
        source_text = "".join(source)
    else:
        source_text = str(source)

    # 셀 헤더: 번호와 타입 표시
    header = f"--- Cell {index} [{cell_type}] ---"

    # 출력이 있으면 함께 표시 (code 셀만)
    output_text = ""
    if cell_type == "code":
        outputs = cell.get("outputs", [])
        if outputs:
            output_parts = []
            for out in outputs:
                # text 출력 (stream, execute_result, display_data)
                if "text" in out:
                    text = out["text"]
                    if isinstance(text, list):
                        text = "".join(text)
                    output_parts.append(text)
                elif "data" in out:
                    # MIME 타입별 출력 — text/plain을 우선 표시
                    data = out["data"]
                    if "text/plain" in data:
                        plain = data["text/plain"]
                        if isinstance(plain, list):
                            plain = "".join(plain)
                        output_parts.append(plain)
                    else:
                        # 다른 MIME 타입은 키만 표시
                        mime_types = ", ".join(data.keys())
                        output_parts.append(f"[Output: {mime_types}]")
                elif "traceback" in out:
                    # 에러 트레이스백
                    tb = "\n".join(out["traceback"])
                    output_parts.append(f"[Error]\n{tb}")

            if output_parts:
                output_text = "\n[Output]\n" + "\n".join(output_parts)

    return f"{header}\n{source_text}{output_text}"


# ─────────────────────────────────────────────
# NotebookReadTool — 노트북 파일 읽기
# ─────────────────────────────────────────────
class NotebookReadTool(BaseTool):
    """
    Jupyter Notebook (.ipynb) 파일을 읽어 셀 내용을 표시하는 도구.
    JSON 구조를 파싱하여 각 셀의 타입, 소스 코드, 출력을 보여준다.
    읽기 전용이므로 병렬 실행이 안전하다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "NotebookRead"

    @property
    def description(self) -> str:
        return (
            "Jupyter Notebook (.ipynb) 파일을 읽습니다. "
            "각 셀의 타입(code/markdown), 소스 코드, 출력을 표시합니다."
        )

    @property
    def group(self) -> str:
        return "notebook"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": ".ipynb 파일 경로",
                },
                "cell_index": {
                    "type": "integer",
                    "description": "특정 셀만 읽기 (0부터 시작). 생략 시 전체 셀 표시",
                    "minimum": 0,
                },
            },
            "required": ["file_path"],
        }

    # ═══ 3. Behavior Flags ═══

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """파일 경로가 .ipynb로 끝나는지 검증한다."""
        file_path = input_data.get("file_path", "")
        if not file_path:
            return "file_path는 비어 있을 수 없습니다."
        if not file_path.endswith(".ipynb"):
            return "file_path는 .ipynb 파일이어야 합니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """읽기 전용이므로 항상 허용한다."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        노트북 파일을 읽고 셀 내용을 포맷하여 반환한다.

        처리 순서:
          1. 파일 경로 해석 (절대/상대)
          2. JSON 파싱
          3. 셀 목록 추출
          4. 각 셀을 읽기 쉬운 형식으로 변환
          5. 결과 반환
        """
        file_path = input_data["file_path"]
        cell_index = input_data.get("cell_index")

        # 파일 경로 해석
        resolved = _resolve_notebook_path(file_path, context.cwd)
        if not resolved.exists():
            return ToolResult.error(f"파일을 찾을 수 없습니다: {resolved}")
        if not resolved.is_file():
            return ToolResult.error(f"파일이 아닙니다: {resolved}")

        # JSON 파싱
        try:
            content = resolved.read_text(encoding="utf-8")
            notebook = json.loads(content)
        except json.JSONDecodeError as e:
            return ToolResult.error(f"노트북 JSON 파싱 실패: {e}")
        except OSError as e:
            return ToolResult.error(f"파일 읽기 실패: {e}")

        # 셀 목록 추출
        cells = notebook.get("cells", [])
        if not cells:
            return ToolResult.success("노트북에 셀이 없습니다.", cell_count=0)

        # 특정 셀만 읽기
        if cell_index is not None:
            if cell_index < 0 or cell_index >= len(cells):
                return ToolResult.error(
                    f"셀 인덱스 {cell_index}이(가) 범위를 벗어났습니다. "
                    f"총 셀 수: {len(cells)} (0~{len(cells) - 1})"
                )
            formatted = _format_cell(cell_index, cells[cell_index])
            return ToolResult.success(formatted, cell_count=1, total_cells=len(cells))

        # 전체 셀 표시
        formatted_cells = [_format_cell(i, cell) for i, cell in enumerate(cells)]
        result_text = "\n\n".join(formatted_cells)

        # 노트북 메타데이터 헤더 추가
        kernel_info = notebook.get("metadata", {}).get("kernelspec", {})
        kernel_name = kernel_info.get("display_name", "unknown")
        header = f"Notebook: {resolved.name} | Kernel: {kernel_name} | Cells: {len(cells)}\n"
        header += "=" * 60

        logger.debug("NotebookRead: %s, %d cells", resolved, len(cells))
        return ToolResult.success(
            f"{header}\n\n{result_text}",
            cell_count=len(cells),
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Reading {input_data.get('file_path', 'notebook')}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("file_path", "")


# ─────────────────────────────────────────────
# NotebookEditTool — 노트북 셀 수정
# ─────────────────────────────────────────────
class NotebookEditTool(BaseTool):
    """
    Jupyter Notebook (.ipynb) 파일의 특정 셀을 수정하는 도구.
    셀 인덱스와 새 소스 코드를 받아 해당 셀을 업데이트한다.
    셀 삽입/삭제도 지원한다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "NotebookEdit"

    @property
    def description(self) -> str:
        return (
            "Jupyter Notebook (.ipynb)의 특정 셀을 수정합니다. "
            "셀 내용 변경, 새 셀 삽입, 셀 삭제를 지원합니다."
        )

    @property
    def group(self) -> str:
        return "notebook"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": ".ipynb 파일 경로",
                },
                "cell_index": {
                    "type": "integer",
                    "description": "수정할 셀 번호 (0부터 시작)",
                    "minimum": 0,
                },
                "new_source": {
                    "type": "string",
                    "description": "새 셀 소스 코드/텍스트",
                },
                "cell_type": {
                    "type": "string",
                    "description": "셀 타입 (code 또는 markdown). 삽입 시 사용",
                    "enum": ["code", "markdown"],
                },
                "operation": {
                    "type": "string",
                    "description": "수행할 작업: edit(수정), insert(삽입), delete(삭제)",
                    "enum": ["edit", "insert", "delete"],
                    "default": "edit",
                },
            },
            "required": ["file_path", "cell_index"],
        }

    # ═══ 3. Behavior Flags ═══
    # 쓰기 도구 — 기본 fail-closed 유지

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """입력 유효성을 검증한다."""
        file_path = input_data.get("file_path", "")
        if not file_path:
            return "file_path는 비어 있을 수 없습니다."
        if not file_path.endswith(".ipynb"):
            return "file_path는 .ipynb 파일이어야 합니다."

        operation = input_data.get("operation", "edit")
        # edit과 insert에는 new_source가 필요하다
        if operation in ("edit", "insert"):
            if not input_data.get("new_source") and input_data.get("new_source") != "":
                return f"'{operation}' 작업에는 new_source가 필요합니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """쓰기 도구이므로 사용자에게 확인을 요청한다."""
        file_path = input_data.get("file_path", "")
        operation = input_data.get("operation", "edit")
        cell_index = input_data.get("cell_index", 0)
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Notebook {operation}: cell {cell_index} in {file_path}",
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        노트북 셀을 수정/삽입/삭제한다.

        처리 순서:
          1. 파일 읽기 및 JSON 파싱
          2. 셀 인덱스 유효성 검증
          3. 지정된 작업 수행 (edit/insert/delete)
          4. JSON으로 직렬화하여 파일에 저장
        """
        file_path = input_data["file_path"]
        cell_index = input_data["cell_index"]
        new_source = input_data.get("new_source", "")
        cell_type = input_data.get("cell_type", "code")
        operation = input_data.get("operation", "edit")

        # 파일 경로 해석
        resolved = _resolve_notebook_path(file_path, context.cwd)
        if not resolved.exists():
            return ToolResult.error(f"파일을 찾을 수 없습니다: {resolved}")

        # JSON 파싱
        try:
            content = resolved.read_text(encoding="utf-8")
            notebook = json.loads(content)
        except json.JSONDecodeError as e:
            return ToolResult.error(f"노트북 JSON 파싱 실패: {e}")
        except OSError as e:
            return ToolResult.error(f"파일 읽기 실패: {e}")

        cells = notebook.get("cells", [])

        # 작업별 처리
        if operation == "edit":
            # 기존 셀의 소스 코드 수정
            if cell_index < 0 or cell_index >= len(cells):
                return ToolResult.error(
                    f"셀 인덱스 {cell_index}이(가) 범위를 벗어났습니다. 총 셀 수: {len(cells)}"
                )
            # source를 줄 단위 리스트로 변환 (Jupyter 표준 형식)
            cells[cell_index]["source"] = _source_to_lines(new_source)
            msg = f"셀 {cell_index}의 내용을 수정했습니다."

        elif operation == "insert":
            # 새 셀을 지정 위치에 삽입
            if cell_index < 0 or cell_index > len(cells):
                return ToolResult.error(
                    f"삽입 위치 {cell_index}이(가) 범위를 벗어났습니다. 허용 범위: 0~{len(cells)}"
                )
            new_cell = _create_empty_cell(cell_type, new_source)
            cells.insert(cell_index, new_cell)
            msg = f"셀 {cell_index} 위치에 새 {cell_type} 셀을 삽입했습니다."

        elif operation == "delete":
            # 셀 삭제
            if cell_index < 0 or cell_index >= len(cells):
                return ToolResult.error(
                    f"셀 인덱스 {cell_index}이(가) 범위를 벗어났습니다. 총 셀 수: {len(cells)}"
                )
            deleted = cells.pop(cell_index)
            deleted_type = deleted.get("cell_type", "unknown")
            msg = f"셀 {cell_index} ({deleted_type})을 삭제했습니다."

        else:
            return ToolResult.error(f"알 수 없는 작업: {operation}")

        # 수정된 노트북을 파일에 저장
        notebook["cells"] = cells
        try:
            resolved.write_text(
                json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            return ToolResult.error(f"파일 저장 실패: {e}")

        logger.info("NotebookEdit: %s cell %d in %s", operation, cell_index, resolved)
        return ToolResult.success(msg, cell_count=len(cells))

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        operation = input_data.get("operation", "edit")
        return f"Notebook {operation}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return f"{input_data.get('file_path', '')} cell {input_data.get('cell_index', '')}"


# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def _source_to_lines(source: str) -> list[str]:
    """
    소스 문자열을 Jupyter 표준 형식(줄 단위 리스트)으로 변환한다.
    각 줄(마지막 제외)에 개행 문자를 포함한다.
    """
    if not source:
        return []
    lines = source.split("\n")
    # 마지막 줄을 제외한 모든 줄에 \n 추가
    result = [line + "\n" for line in lines[:-1]]
    # 마지막 줄은 개행 없이 추가 (빈 문자열이 아닌 경우)
    if lines[-1]:
        result.append(lines[-1])
    return result


def _create_empty_cell(cell_type: str, source: str = "") -> dict[str, Any]:
    """
    새 노트북 셀을 생성한다.

    Args:
        cell_type: 셀 타입 ("code" 또는 "markdown")
        source: 셀 소스 코드/텍스트

    Returns:
        Jupyter 노트북 셀 딕셔너리
    """
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": _source_to_lines(source),
    }
    if cell_type == "code":
        # code 셀에는 실행 관련 필드가 필요하다
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell
