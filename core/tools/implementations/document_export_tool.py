"""
DocumentExport 도구 — 텍스트/마크다운을 docx/pptx/hwpx/md/txt 파일로 "생성"한다.

[이 파일이 하는 일]
  모델(LLM)이 만들어 낸 요약·보고서·회의록 같은 결과 텍스트를, 사용자가 실제로
  내려받을 수 있는 문서 "파일"로 변환한다. 파일을 만든 뒤에는 다운로드 URL을
  돌려주고, 웹에서는 그 URL(/v1/download/...)로 파일을 받아간다.
  DocumentProcess(파일 "읽기") 도구의 정반대 방향, 즉 파일 "쓰기(생성)" 도구다.

[구성 요소]
  - resolve_exports_dir(): 생성물을 저장할 폴더를 결정(없으면 생성)한다.
  - _safe_filename():      모델/사용자가 준 파일명을 안전하게 정화 + uuid 고유화한다.
  - DocumentExportTool:    BaseTool을 상속한 실제 도구 클래스(검증→권한→생성→결과).
  - RENDERERS:             포맷별 실제 파일 생성 함수 표(document_export_renderers).

[안전 설계 — 왜 임의 경로 쓰기가 아닌가]
  Write 도구는 모델이 지정한 임의 경로에 쓰지만, 이 도구는 파일 "내용"만 받고
  저장 위치는 **샌드박스 exports 디렉토리로 고정**한다. 파일명도 정화 + uuid로
  고유화하므로, 경로 순회(../..)나 기존 파일 덮어쓰기가 구조적으로 불가능하다.
  그래서 check_permissions 를 ALLOW 로 둔다(저장 경로를 모델이 정할 수 없으므로 안전).

[에어갭]
  모든 문서 생성은 로컬에 사전 설치된 순수 파이썬 라이브러리로만 수행한다.
  외부 네트워크 호출은 전혀 없다(폐쇄망 준수).

작성자: 이현수 / 작성일: 2026-07-05
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

# 이 모듈 전용 로거. 규칙에 따라 "nexus.{module}" 네임스페이스를 사용한다.
# (파일 생성 성공/실패는 이 로거로 남겨 운영 중 추적할 수 있게 한다.)
logger = logging.getLogger("nexus.tools.document_export")


def resolve_exports_dir(configured: str | None = None) -> Path:
    """
    생성물(문서 파일)을 저장할 디렉토리를 결정하고, 없으면 만들어서 반환한다.

    [왜 이 함수가 따로 필요한가]
      파일을 "쓰는" 쪽(이 도구)과 "읽어서 내려주는" 쪽(웹의 /v1/download 라우트)이
      반드시 같은 폴더를 가리켜야 한다. 그래서 경로 결정 로직을 이 함수 하나로
      모아 두 곳이 공유한다. 경로를 여기저기 하드코딩하면 어긋나기 쉽기 때문이다.

    [경로 결정 우선순위]
      1) configured 설정값이 있으면 그대로 사용
         (예: config.document_export.exports_dir 를 웹이 주입).
      2) 비어 있으면 시스템 임시폴더 아래 nexus_exports 폴더로 폴백.
      이는 업로드 라우트(/v1/upload)가 {tempdir}/nexus_uploads 를 쓰는 것과 같은
      관례라, 개발(Windows)·배포(Linux) 어디서든 경로를 하드코딩하지 않아도 된다.

    매개변수:
      configured — 설정에서 온 저장 경로 문자열. None/빈문자열이면 폴백을 쓴다.
    반환:
      실제로 존재가 보장된(mkdir 완료된) 저장 디렉토리의 Path.
    """
    # 설정값이 있으면 앞뒤 공백을 제거해 쓰고, 없으면 임시폴더/nexus_exports 로 결정.
    base = (configured or "").strip() or os.path.join(tempfile.gettempdir(), "nexus_exports")
    path = Path(base)
    # 상위 폴더까지 한 번에 생성. 이미 있으면 조용히 넘어간다(exist_ok=True).
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename(name: str, fmt: str) -> str:
    """
    사용자/모델이 준 파일명을 안전하게 정화(sanitize)하고 uuid로 고유화한다.

    [왜 필요한가]
      모델이 준 파일명을 그대로 저장 경로에 쓰면 "../../etc/passwd" 같은 경로 순회나
      기존 파일 덮어쓰기가 가능해진다. 여기서 위험 요소를 모두 제거하고, 뒤에
      임의의 uuid 조각을 붙여 항상 새롭고 안전한 파일명이 나오도록 보장한다.

    [처리 단계]
      1) 확장자를 뗀 이름(stem)만 취한다(포맷은 인자 fmt로 따로 붙임).
      2) 한글·영숫자·(-_. )와 공백을 제외한 모든 문자를 제거
         → 경로 구분자(슬래시·역슬래시), 제어문자, 위험문자 차단.
      3) 연속 공백을 밑줄로 바꾸고 60자로 길이 제한. 비면 "document"로 대체.
      4) 뒤에 uuid 8자리를 붙여 이름 충돌·덮어쓰기를 원천 차단.

    매개변수:
      name — 원본 파일명(확장자 유무 무관, None 가능).
      fmt  — 붙일 확장자(docx/pptx/hwpx/md/txt 등).
    반환:
      안전하게 정화된 최종 파일명. 예: "회의록-a1b2c3d4.docx"
    """
    # 1) 확장자 제거 후 순수 이름(stem)만 추출. None 대비로 or "" 를 둔다.
    stem = Path(name or "").stem
    # 2) 허용 문자(단어문자/./-/공백)만 남기고 나머지는 삭제 → 경로·제어문자 차단.
    stem = re.sub(r"[^\w.\- ]", "", stem, flags=re.UNICODE).strip()
    # 3) 공백을 밑줄로 치환하고 60자로 자른다. 결과가 비면 기본값 "document".
    stem = re.sub(r"\s+", "_", stem)[:60] or "document"
    # 4) uuid 8자리 + 확장자를 붙여 항상 고유한 파일명을 만든다.
    return f"{stem}-{uuid.uuid4().hex[:8]}.{fmt}"


class DocumentExportTool(BaseTool):
    """
    텍스트/마크다운을 문서 파일로 생성하는 도구(BaseTool 구현체).

    [수명주기(BaseTool 계약)]
      validate_input()  → 입력을 미리 검증(content 존재, format 지원 여부).
      check_permissions() → 실행 허가 판단(이 도구는 항상 ALLOW, 위 안전설계 참조).
      call()            → 실제로 파일을 생성하고 다운로드 URL을 반환.
      그 외 get_*()      → 진행 표시 등 UI 힌트.

    지원 포맷: docx, pptx, hwpx, md, txt (실제 생성은 RENDERERS 표에 위임).
    """

    # ═══ 1. Identity(도구 식별 정보) ═══
    # 레지스트리가 도구를 이름/별칭/그룹으로 찾을 때 쓰는 메타데이터다.

    @property
    def name(self) -> str:
        # 모델이 tool_calls에서 호출할 때 쓰는 고유 이름(레지스트리 키).
        return "DocumentExport"

    @property
    def description(self) -> str:
        # 모델에게 "이 도구를 언제 써야 하는지" 알려주는 설명(프롬프트에 노출됨).
        return (
            "Generate a downloadable document file (docx, pptx, hwpx, md, txt) "
            "from markdown/plain text content. Use when the user asks to export, "
            "save, or download an answer as a file."
        )

    @property
    def aliases(self) -> list[str]:
        # 모델이 다른 이름으로 부를 때도 이 도구로 연결되도록 하는 별칭 목록.
        return ["GenerateDocument", "SaveAs", "ExportDocument"]

    @property
    def group(self) -> str:
        # 도구 분류(파일 계열). 권한/UI에서 그룹 단위로 다룰 때 쓴다.
        return "file"

    # ═══ 2. Schema(입력 스키마) ═══
    # 모델이 넘겨야 하는 인자의 JSON Schema. content·format이 필수다.

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

    # ═══ 3. Behavior Flags (fail-closed 기본값 유지) ═══
    # BaseTool의 behavior flag는 가장 제한적인 값이 기본값이다. 이 도구는 그 기본값을
    # 그대로 쓰므로 여기서 별도로 완화(override)하지 않는다.
    # - is_destructive: 파일을 "쓰지만" 저장 위치가 샌드박스로 고정이고 파일명이
    #   고유화되어 기존 파일을 훼손하지 않으므로, 파괴적 도구로 취급할 필요가 없다.
    # - is_concurrency_safe: 기본값(False)을 유지 — 병렬 실행하지 않는 안전측 선택.

    # ═══ 5. Lifecycle(수명주기 메서드) ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실제 실행 전에 입력을 사전 검증한다.

        무엇을 검사하나:
          - content 가 비어 있지 않은지(빈 문서 생성 방지).
          - format 이 우리가 실제로 렌더링할 수 있는 값인지(RENDERERS에 존재).

        반환:
          문제가 없으면 None, 문제가 있으면 사용자에게 보여줄 오류 메시지(str).
        """
        content = input_data.get("content", "")
        if not content or not str(content).strip():
            return "content는 비어 있을 수 없습니다."
        # format을 소문자·공백제거로 정규화한 뒤, 렌더러 표에 있는지 확인한다.
        fmt = str(input_data.get("format", "")).lower().strip()
        if fmt not in RENDERERS:
            return f"지원하지 않는 format입니다: '{fmt}'. 지원: {', '.join(RENDERERS)}"
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        이 도구의 실행 허가 여부를 판단한다.

        [왜 항상 ALLOW인가]
          일반 Write 도구는 모델이 경로를 정하므로 권한 검사가 필요하지만, 이 도구는
          저장 위치가 샌드박스 exports 디렉토리로 고정되고 파일명도 정화·uuid 고유화된다.
          즉 모델이 저장 경로를 조작할 방법이 구조적으로 없으므로 임의 경로 쓰기 위험이
          없다. 따라서 별도 사용자 확인 없이 항상 허용(ALLOW)한다.

        반환:
          behavior=ALLOW 인 PermissionResult.
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        실제로 문서 파일을 생성하고, 사용자에게 줄 다운로드 URL을 반환한다.
        이 도구의 핵심 메서드다.

        처리 순서:
          1. content/format/title 입력값을 꺼내고 format을 정규화한다.
          2. exports 디렉토리를 확보한다(설정값 우선, 없으면 tempdir 폴백).
          3. 파일명을 정화 + uuid로 고유화해 최종 저장 경로를 만든다.
          4. 포맷별 렌더러를 호출해 파일을 만든다
             (무거운 라이브러리는 각 렌더러 내부에서 lazy import).
          5. 성공하면 다운로드 링크 안내문 + 메타데이터를 담아 반환한다.

        매개변수:
          input_data — content(필수), format(필수), filename(선택), title(선택).
          context    — 실행 컨텍스트. options["exports_dir"]로 저장 폴더를 주입받는다.
        반환:
          성공 시 ToolResult.success(안내문 + download_url 등 메타데이터),
          실패 시 ToolResult.error(사유 메시지).
        """
        # 1) 입력값 추출. format은 소문자·공백제거로 정규화, title은 None이면 빈 문자열로.
        content = input_data["content"]
        fmt = str(input_data["format"]).lower().strip()
        title = str(input_data.get("title", "") or "")

        # 2) 저장 폴더 확보. exports_dir 은 웹이 base_options 로 주입(config 값)한다.
        #    CLI 등에서 미주입 시 None → resolve_exports_dir 가 tempdir 폴백으로
        #    처리하므로 어떤 진입점에서도 동작이 깨지지 않는다(무회귀).
        exports_dir = resolve_exports_dir(context.options.get("exports_dir"))
        # 3) 파일명 정화 + uuid 고유화 후, 고정된 exports 폴더 아래 최종 경로를 만든다.
        filename = _safe_filename(input_data.get("filename", ""), fmt)
        out_path = exports_dir / filename

        # 4) 포맷별 렌더러를 호출해 실제 파일을 만든다. 렌더링 실패는 두 갈래로 처리:
        try:
            RENDERERS[fmt](content, out_path, title)
        except ImportError as e:
            # (a) 해당 포맷 생성 라이브러리 미설치 — 에어갭 번들 누락 등.
            #     원인을 명확히 구분해 안내한다(설치 문제 vs 렌더 로직 문제).
            return ToolResult.error(f"'{fmt}' 생성 라이브러리가 설치되어 있지 않습니다: {e}")
        except Exception as e:  # noqa: BLE001 — 렌더 실패를 tool_use_error로 래핑
            # (b) 그 밖의 모든 렌더 실패. anti-pattern(bare except) 방지를 위해
            #     구체 로그를 남기고, 사용자에게는 사유를 담은 오류 결과로 래핑한다.
            logger.exception("문서 생성 실패: format=%s", fmt)
            return ToolResult.error(f"문서 생성에 실패했습니다({fmt}): {e}")

        # 5) 생성된 파일 크기를 읽고, 웹 다운로드 라우트가 인식하는 URL을 만든다.
        size = out_path.stat().st_size
        download_url = f"/v1/download/{filename}"
        logger.info("DocumentExport %s (%d bytes) → %s", filename, size, out_path)

        # 모델이 사용자에게 그대로 제시할 수 있도록 마크다운 링크를 결과 본문에 포함한다.
        # (download_url 등 구조화 메타데이터도 함께 넘겨, 웹이 신뢰할 값은 모델 텍스트가
        #  아니라 이 메타데이터에서 뽑아 쓰도록 한다.)
        return ToolResult.success(
            f"문서를 생성했습니다: {filename} ({size:,} bytes).\n"
            f"사용자에게 아래 다운로드 링크를 그대로 제시하세요:\n"
            f"[{filename} 다운로드]({download_url})",
            download_url=download_url,
            filename=filename,
            format=fmt,
            bytes=size,
        )

    # ═══ 7. UI Hints(진행 표시용 힌트) ═══
    # CLI/웹이 도구 실행 상황을 사용자에게 보여줄 때 쓰는 짧은 문구를 제공한다.

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 표시 문구. 예: "Generating docx".
        return f"Generating {input_data.get('format', 'document')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출을 한 줄로 요약할 때 쓰는 값(여기서는 대상 포맷).
        return str(input_data.get("format", ""))
