"""
DocumentExport 도구 — 텍스트/마크다운을 docx/pptx/hwpx/md/txt 파일로 "생성"한다.

DocumentProcess(읽기)의 반대 방향 도구다. 모델이 만든 요약·보고서 등을
사용자가 내려받을 수 있는 실제 문서 파일로 만들어, 웹에서는 /v1/download 로
다운로드하게 한다.

안전 설계(왜 임의 경로 쓰기가 아닌가):
  Write 도구는 모델이 지정한 임의 경로에 쓰지만, 이 도구는 파일 "내용"만 받고
  저장 위치는 **샌드박스 exports 디렉토리로 고정**한다. 파일명도 정화 + uuid로
  고유화하므로, 경로 순회나 기존 파일 덮어쓰기가 구조적으로 불가능하다.
  그래서 check_permissions 를 ALLOW 로 둔다(경로를 모델이 못 정함).

에어갭: 모든 생성은 로컬 순수 파이썬 라이브러리로만 수행(외부 네트워크 0).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)
from core.tools.implementations.document_export_renderers import RENDERERS

logger = logging.getLogger("nexus.tools.document_export")


def resolve_exports_dir(configured: str | None = None) -> Path:
    """
    생성물 저장 디렉토리를 결정하고(없으면 만들고) 반환한다.

    우선순위: 설정값(configured, 예: config.document_export.exports_dir)
             → 비어 있으면 시스템 임시폴더/nexus_exports.
    업로드 라우트(/v1/upload)가 {tempdir}/nexus_uploads 를 쓰는 것과 같은 관례라,
    개발(Windows)·배포(Linux) 어디서든 경로를 하드코딩하지 않는다.

    도구(쓰기)와 다운로드 라우트(읽기)가 이 함수를 공유해 같은 위치를 가리킨다.
    """
    base = (configured or "").strip() or os.path.join(tempfile.gettempdir(), "nexus_exports")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename(name: str, fmt: str) -> str:
    """
    사용자/모델이 준 파일명을 안전하게 정화하고 uuid로 고유화한다.

    - 경로 구분자·제어문자·위험문자를 제거(한글·영숫자·-_. 공백만 허용).
    - 공백은 밑줄로, 길이는 60자로 제한.
    - 뒤에 uuid 8자리를 붙여 충돌·덮어쓰기를 방지한다.
    결과 예: "회의록-a1b2c3d4.docx"
    """
    stem = Path(name or "").stem
    stem = re.sub(r"[^\w.\- ]", "", stem, flags=re.UNICODE).strip()
    stem = re.sub(r"\s+", "_", stem)[:60] or "document"
    return f"{stem}-{uuid.uuid4().hex[:8]}.{fmt}"


class DocumentExportTool(BaseTool):
    """
    텍스트/마크다운을 문서 파일로 생성하는 도구.
    지원 포맷: docx, pptx, hwpx, md, txt.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "DocumentExport"

    @property
    def description(self) -> str:
        return (
            "Generate a downloadable document file (docx, pptx, hwpx, md, txt) "
            "from markdown/plain text content. Use when the user asks to export, "
            "save, or download an answer as a file."
        )

    @property
    def aliases(self) -> list[str]:
        return ["GenerateDocument", "SaveAs", "ExportDocument"]

    @property
    def group(self) -> str:
        return "file"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "Document body as markdown/plain text. "
                        "'#','##','###' become headings, '- ' become bullets, "
                        "'**bold**' becomes bold."
                    ),
                },
                "format": {
                    "type": "string",
                    "enum": ["docx", "pptx", "hwpx", "md", "txt"],
                    "description": "Output file format.",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional base filename (without extension).",
                },
                "title": {
                    "type": "string",
                    "description": "Optional document title (rendered as top heading).",
                },
            },
            "required": ["content", "format"],
        }

    # ═══ 3. Behavior Flags (fail-closed 유지) ═══
    # 쓰기 도구지만 저장 위치가 샌드박스로 고정이라 is_destructive=False.
    # is_concurrency_safe 는 기본값(False)을 유지 — 안전측.

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """content 존재 + format 지원 여부를 사전 검증한다."""
        content = input_data.get("content", "")
        if not content or not str(content).strip():
            return "content는 비어 있을 수 없습니다."
        fmt = str(input_data.get("format", "")).lower().strip()
        if fmt not in RENDERERS:
            return f"지원하지 않는 format입니다: '{fmt}'. 지원: {', '.join(RENDERERS)}"
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        저장 위치가 샌드박스 exports 디렉토리로 고정이고 파일명도 정화·고유화되므로,
        임의 경로 쓰기 위험이 없다 → 항상 허용한다.
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        문서 파일을 생성하고 다운로드 URL을 반환한다.

        처리 순서:
          1. format 정규화 및 지원 여부 확인
          2. exports 디렉토리 확보(설정 → tempdir 폴백)
          3. 파일명 정화 + uuid 고유화
          4. 포맷별 렌더러 호출(라이브러리는 렌더러 내부 lazy import)
          5. 성공 시 다운로드 URL + 메타데이터 반환
        """
        content = input_data["content"]
        fmt = str(input_data["format"]).lower().strip()
        title = str(input_data.get("title", "") or "")

        # exports_dir 은 웹이 base_options 로 주입(config 값)한다. CLI 등 미주입 시
        # None → resolve_exports_dir 가 tempdir 폴백으로 처리(무회귀).
        exports_dir = resolve_exports_dir(context.options.get("exports_dir"))
        filename = _safe_filename(input_data.get("filename", ""), fmt)
        out_path = exports_dir / filename

        try:
            RENDERERS[fmt](content, out_path, title)
        except ImportError as e:
            # 해당 포맷 생성 라이브러리 미설치 — 에어갭 번들 누락 등.
            return ToolResult.error(f"'{fmt}' 생성 라이브러리가 설치되어 있지 않습니다: {e}")
        except Exception as e:  # noqa: BLE001 — 렌더 실패를 tool_use_error로 래핑
            logger.exception("문서 생성 실패: format=%s", fmt)
            return ToolResult.error(f"문서 생성에 실패했습니다({fmt}): {e}")

        size = out_path.stat().st_size
        download_url = f"/v1/download/{filename}"
        logger.info("DocumentExport %s (%d bytes) → %s", filename, size, out_path)

        # 모델이 사용자에게 그대로 제시할 수 있도록 마크다운 링크를 결과에 포함한다.
        return ToolResult.success(
            f"문서를 생성했습니다: {filename} ({size:,} bytes).\n"
            f"사용자에게 아래 다운로드 링크를 그대로 제시하세요:\n"
            f"[{filename} 다운로드]({download_url})",
            download_url=download_url,
            filename=filename,
            format=fmt,
            bytes=size,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Generating {input_data.get('format', 'document')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return str(input_data.get("format", ""))
