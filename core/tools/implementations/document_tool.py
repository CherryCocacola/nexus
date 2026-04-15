"""
DocumentProcess 도구 — 문서 파일 파싱 및 텍스트 추출.

사양서 Ch.13.6에 정의된 에어갭 전용 도구.
PDF, DOCX, XLSX 파일을 읽어 텍스트 내용을 반환한다.

지원 형식:
  - .pdf  → pypdf로 페이지별 텍스트 추출
  - .docx → python-docx로 단락 텍스트 추출
  - .xlsx → openpyxl로 시트별 셀 데이터 추출
  - .txt, .csv, .md 등 → 일반 텍스트 읽기

왜 별도 도구인가: Read 도구는 텍스트 파일만 읽을 수 있다.
바이너리 형식(PDF, DOCX, XLSX)은 전용 파서가 필요하다.
"""

from __future__ import annotations

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

logger = logging.getLogger("nexus.tools.document")

# 텍스트 추출 결과의 최대 문자 수 (컨텍스트 초과 방지)
# RTX 5090 (8192 ctx): tool_result는 ~3,000 토큰(~6,000자) 이내여야
# 도구 스키마 + 시스템 + 메시지 + 출력 토큰이 컨텍스트에 들어간다.
MAX_EXTRACT_CHARS = 6000


class DocumentProcessTool(BaseTool):
    """
    문서 파일을 파싱하여 텍스트 내용을 추출하는 도구.
    PDF, DOCX, XLSX 등 바이너리 문서 형식을 지원한다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "DocumentProcess"

    @property
    def description(self) -> str:
        return (
            "PDF, DOCX, XLSX 파일을 파싱하여 텍스트를 추출합니다. "
            "file_path에 파일 경로를 지정하세요."
        )

    @property
    def aliases(self) -> list[str]:
        return ["DocProcess", "ParseDocument"]

    @property
    def group(self) -> str:
        return "file"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "파싱할 문서 파일 경로",
                },
                "pages": {
                    "type": "string",
                    "description": "PDF 페이지 범위 (예: '1-5'). PDF에만 적용.",
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

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """읽기 전용이므로 항상 허용한다."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> ToolResult:
        """
        문서 파일을 파싱하여 텍스트를 추출한다.

        처리 순서:
          1. 파일 존재 여부 확인
          2. 확장자별 파서 선택 (pdf/docx/xlsx/txt)
          3. 텍스트 추출 (최대 MAX_EXTRACT_CHARS 제한)
          4. 결과 반환
        """
        file_path = input_data.get("file_path", "")
        if not file_path:
            return ToolResult.error("file_path가 필요합니다.")

        path = Path(file_path)
        if not path.exists():
            return ToolResult.error(f"파일을 찾을 수 없습니다: {file_path}")

        ext = path.suffix.lower()

        try:
            if ext == ".pdf":
                text = self._parse_pdf(path, input_data.get("pages"))
            elif ext in (".docx", ".doc"):
                text = self._parse_docx(path)
            elif ext in (".xlsx", ".xls"):
                text = self._parse_xlsx(path)
            else:
                # 일반 텍스트 파일 폴백
                text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("문서 파싱 실패: %s — %s", file_path, e)
            return ToolResult.error(f"문서 파싱 실패: {type(e).__name__}: {e}")

        # 컨텍스트 초과 방지를 위해 텍스트 길이 제한
        if len(text) > MAX_EXTRACT_CHARS:
            text = (
                text[:MAX_EXTRACT_CHARS]
                + f"\n\n... [전체 {len(text)}자 중 {MAX_EXTRACT_CHARS}자까지 표시]"
            )

        return ToolResult.success(
            text,
            file=str(path),
            format=ext,
            chars=len(text),
        )

    # ─── 파서 구현 ───

    @staticmethod
    def _parse_pdf(path: Path, pages: str | None) -> str:
        """PDF 파일에서 텍스트를 추출한다."""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)

        # 페이지 범위 파싱
        if pages:
            parts = pages.split("-")
            start = max(int(parts[0]) - 1, 0)
            end = min(int(parts[-1]), total_pages)
        else:
            start = 0
            end = min(total_pages, 20)  # 기본: 최대 20페이지

        texts = []
        for i in range(start, end):
            page_text = reader.pages[i].extract_text() or ""
            if page_text.strip():
                texts.append(f"[페이지 {i + 1}]\n{page_text}")

        if not texts:
            return f"[PDF {total_pages}페이지, 텍스트 추출 불가 (스캔 이미지일 수 있음)]"

        return "\n\n".join(texts)

    @staticmethod
    def _parse_docx(path: Path) -> str:
        """DOCX 파일에서 텍스트를 추출한다."""
        from docx import Document

        doc = Document(str(path))
        texts = []

        for para in doc.paragraphs:
            if para.text.strip():
                texts.append(para.text)

        # 표(table) 내용도 추출
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    texts.append(" | ".join(cells))

        if not texts:
            return "[DOCX 파일에서 텍스트를 추출할 수 없습니다]"

        return "\n".join(texts)

    @staticmethod
    def _parse_xlsx(path: Path) -> str:
        """XLSX 파일에서 텍스트를 추출한다."""
        from openpyxl import load_workbook

        wb = load_workbook(str(path), read_only=True, data_only=True)
        texts = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            texts.append(f"[시트: {sheet_name}]")

            row_count = 0
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    texts.append(" | ".join(cells))
                    row_count += 1
                    if row_count >= 200:  # 시트당 최대 200행
                        texts.append(f"... [{sheet_name} 시트 {row_count}+ 행]")
                        break

        wb.close()

        if len(texts) <= len(wb.sheetnames):
            return "[XLSX 파일에서 데이터를 추출할 수 없습니다]"

        return "\n".join(texts)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Parsing document..."
