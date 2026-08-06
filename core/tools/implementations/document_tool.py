"""
DocumentProcess 도구 — 문서 파일(PDF/DOCX/XLSX/PPTX/HWPX/HWP)을 파싱해 텍스트를
뽑아내고, 그 텍스트를 작은 "청크(chunk)" 단위로 잘라서 돌려주는 읽기 전용 도구.

이 파일이 하는 일 (한눈에):
  - 에어갭(폐쇄망) 환경에서 로컬 문서를 LLM이 읽을 수 있는 평문 텍스트로 변환한다.
  - PDF/DOCX/XLSX처럼 바이너리 형식인 문서도 외부 API 없이 로컬 라이브러리로 파싱한다.
  - PPTX/HWPX/HWP는 인제스트(ingest) 레이어에 이미 있는 파서를 그대로 빌려 쓴다
    (같은 파싱 로직을 두 벌 유지하지 않기 위함 — 아래 _get_ingest_registry 참조).
  - 문서가 크면 한 번에 다 넘기지 않고 여러 조각(청크)으로 나눠, 호출할 때마다
    한 조각씩 반환한다. (사양서 Ch.13.6에 정의된 도구)

구성 요소:
  - DocumentProcessTool : BaseTool을 상속한 실제 도구 클래스. 아래 5개 축을 구현한다.
      * Identity   — name / description / aliases / group (도구 식별 정보)
      * Schema     — input_schema (LLM에게 노출되는 입력 파라미터 정의)
      * Behavior   — is_read_only / is_concurrency_safe (읽기 전용·병렬 안전)
      * Lifecycle  — check_permissions() → call() (권한 확인 후 실제 실행)
      * UI Hints   — get_progress_label() (진행 표시 문구)
  - CHUNK_SIZE      : 청크 하나의 최대 글자 수(모듈 상수, 폴백 기본값).
  - _document_cache : 같은 파일을 청크별로 반복 파싱하지 않도록 저장하는 모듈 캐시.

청크 분할 전략 (call() 흐름):
  1. 문서 전체 텍스트를 한 번 추출한다(_extract_text).
  2. chunk_size(기본 CHUNK_SIZE, 현행 2500자) 단위로 잘라 리스트로 만든다(_split_chunks).
  3. 첫 호출: 청크 1 + 문서 개요(전체 글자 수·청크 수)를 함께 반환한다.
  4. 이후 호출: chunk_index 파라미터로 원하는 다음 청크를 이어서 반환한다.

왜 굳이 청크로 나누는가:
  RTX 5090(컨텍스트 8192 토큰) 기준으로 tool_result가 ~3,000자를 넘으면
  도구 스키마 + 시스템 프롬프트 + 대화 메시지와 합쳐졌을 때 컨텍스트 한도를 넘긴다.
  작은 청크로 쪼개면 문서 크기와 상관없이 어떤 문서든 나눠서 분석할 수 있다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

if TYPE_CHECKING:  # 타입 힌트 전용 import — 런타임에는 실행되지 않는다(지연 로딩 유지).
    from core.ingest.parser_base import ParserRegistry
    from core.ingest.types import DocumentNode, DocumentTree

logger = logging.getLogger("nexus.tools.document")

# 청크 하나의 최대 글자 수 — RTX 5090(컨텍스트 8192 토큰)을 기준으로 역산한 값이다.
# 계산 근거: 도구 스키마(~2,500) + 시스템 프롬프트(~72) + 대화 메시지(~300)
#           + 모델 출력(~1,500) = 약 4,372 토큰이 이미 소비된다.
# 따라서 tool_result가 쓸 수 있는 여유는 8192 - 4372 = 약 3,820 토큰 ≈ 약 3,000자.
# 안전 마진까지 감안해 청크를 2500자로 잡는다. (설정 주입 시 config 값으로 대체됨)
CHUNK_SIZE = 2500

# 파싱 결과 캐시 — 키는 파일의 절대경로, 값은 청크로 분할된 텍스트 리스트.
# 같은 문서를 chunk_index만 바꿔 여러 번 호출할 때 매번 다시 파싱하지 않기 위한
# 모듈 수준(프로세스 공유) 캐시다. (프로세스가 살아있는 동안 유지)
_document_cache: dict[str, list[str]] = {}

# ─────────────────────────────────────────────
# 인제스트 파서 연결 (PPTX / HWPX / HWP)
# ─────────────────────────────────────────────
# 왜 여기서 core.ingest 를 빌려 쓰나:
#   PPTX·HWPX·HWP 파싱은 core/ingest/parsers/ 에 이미 구현·테스트돼 있다. 같은 로직을
#   이 파일에 한 벌 더 만들면 두 곳이 따로 굳어 버린다(버그 수정이 한쪽에만 반영됨).
#   그래서 "지식 적재용 파서"를 그대로 재사용하고, 이 도구는 결과 트리를 평문으로
#   펼치는 일만 한다.
# 무엇을 등록하지 '않는가'도 중요하다:
#   Docling(레이아웃)·OCR(Tesseract/Paddle) 파서는 무겁고 GPU를 쓰므로 채팅 응답
#   경로에 올리지 않는다. PDF는 기존 pypdf 경로를 그대로 유지한다(무회귀).
_INGEST_EXTS = (".pptx", ".hwpx", ".hwp")

# 텍스트를 못 뽑았을 때 사용자에게 돌려줄 안내. 왜 형식별로 다른가:
# "추출 실패"만 알려 주면 사용자는 같은 파일을 계속 다시 올린다. 무엇을 하면
# 되는지(다른 형식으로 저장)를 알려 줘야 실제로 해결된다.
#
# .hwp(구포맷) 주석 — 2026-08-06 실측으로 확인한 사실:
#   변환 경로인 LibreOffice 의 한글 필터(libhwplo.so)는 HWP **V2.00/V2.10/V3.00**
#   (한글 97 이하)만 인식한다. 한글 2002 이후가 쓰는 **HWP 5.0**(OLE 복합문서)은
#   지원하지 않는다(필터 바이너리에 HWP5 스트림명 BodyText/DocInfo/FileHeader 참조가
#   전혀 없음). 즉 LibreOffice 를 설치해도 요즘 .hwp 파일은 열리지 않는다.
#   반면 .hwpx(신포맷)는 python-hwpx 로 완전히 지원되므로, 한/글에서 HWPX 또는
#   PDF 로 저장해 올리는 것이 현재 가장 확실한 우회다.
_EXTRACT_FAILED_HINTS = {
    ".hwp": (
        "[.hwp(한글 구포맷) 파일의 텍스트를 추출하지 못했습니다. "
        "한/글에서 이 문서를 열어 '다른 이름으로 저장'에서 "
        "**HWPX** 또는 **PDF** 형식으로 저장한 뒤 다시 올려 주세요. "
        "두 형식은 완전히 지원됩니다]"
    ),
}
_EXTRACT_FAILED_DEFAULT = (
    "[{ext} 파일에서 텍스트를 추출하지 못했습니다. "
    "빈 문서이거나, 그림·표 이미지로만 이루어진 문서일 수 있습니다. "
    "PDF 로 저장해 다시 올리면 인식되는 경우가 있습니다]"
)

# 레지스트리는 한 번만 만들어 재사용한다(파서 객체 생성 비용 절감). None = 아직 미생성.
_ingest_registry: ParserRegistry | None = None


def _get_ingest_registry() -> ParserRegistry:
    """PPTX/HWPX/HWP 파서만 담은 인제스트 레지스트리를 (한 번만) 만들어 돌려준다.

    파서별로 따로 import 하는 이유: 특정 라이브러리(python-pptx 등)가 설치되지
    않은 환경에서도 나머지 포맷은 계속 동작해야 하기 때문이다. import 에 실패한
    파서는 등록하지 않고 경고만 남긴다 → 그 확장자는 "미지원"으로 안내된다.
    """
    global _ingest_registry
    if _ingest_registry is not None:
        return _ingest_registry

    from core.ingest.parser_base import ParserRegistry

    registry = ParserRegistry()

    # (모듈 경로, 클래스 이름) 목록. 하나가 실패해도 나머지는 등록된다.
    candidates = (
        ("core.ingest.parsers.pptx", "PptxParser"),
        ("core.ingest.parsers.hwpx", "HwpxParser"),
        ("core.ingest.parsers.hwp_libreoffice", "HwpViaLibreOfficeParser"),
    )
    for module_path, class_name in candidates:
        try:
            module = __import__(module_path, fromlist=[class_name])
            registry.register(getattr(module, class_name)())
        except Exception as e:  # noqa: BLE001 — 어떤 라이브러리든 없으면 그 포맷만 포기
            # bare except 가 아니라 사유를 남기고 넘어간다(부분 가용성 우선).
            logger.warning("문서 파서 등록 실패: %s (%s: %s)", class_name, type(e).__name__, e)

    _ingest_registry = registry
    return registry


def _tree_has_text(tree: DocumentTree) -> bool:
    """트리에 실제 본문 글자가 하나라도 있는지 확인한다(경고는 세지 않는다).

    왜 필요한가: 인제스트 파서는 fail-soft라 변환에 실패해도 예외 대신 '빈 트리 +
    경고'를 돌려준다. 그래서 "내용이 있느냐"를 _flatten_tree 결과로 판정하면
    경고 문구 때문에 항상 참이 되어, 정작 사용자에게는 알맹이 없이 내부 경고만
    전달된다. 이 함수는 노드의 text 만 보고 판정한다.
    """

    def walk(nodes: tuple[DocumentNode, ...]) -> bool:
        for node in nodes:
            if (node.text or "").strip():
                return True
            if node.children and walk(node.children):
                return True
        return False

    return walk(tree.nodes)


def _flatten_tree(tree: DocumentTree) -> str:
    """구조 트리(DocumentTree)를 LLM이 읽을 평문 한 덩어리로 펼친다.

    규칙:
      - 노드를 order 순으로 훑되, 자식(children)까지 재귀로 따라간다.
      - 페이지/슬라이드 번호가 바뀌면 "[슬라이드 N]" 머리표를 한 줄 넣는다.
        (PDF 파서가 "[페이지 N]"을 넣는 것과 같은 맥락 표시 방식)
      - 텍스트가 빈 컨테이너 노드(SLIDE 등)는 머리표 판단에만 쓰고 본문에서는 건너뛴다.
      - 파서가 남긴 fail-soft 경고(warnings)는 맨 끝에 붙여, 일부만 읽힌 경우를
        모델이 알 수 있게 한다(조용한 누락 방지).
    """
    lines: list[str] = []
    last_page: int | None = None

    def walk(nodes: tuple[DocumentNode, ...]) -> None:
        nonlocal last_page
        for node in sorted(nodes, key=lambda n: n.order):
            # 페이지(PPTX는 슬라이드 번호)가 바뀌는 지점마다 머리표를 넣는다.
            if node.page is not None and node.page != last_page:
                lines.append(f"\n[슬라이드/페이지 {node.page}]")
                last_page = node.page
            text = (node.text or "").strip()
            if text:
                lines.append(text)
            if node.children:
                walk(node.children)

    walk(tree.nodes)

    body = "\n".join(lines).strip()
    if tree.warnings:
        # 경고는 본문과 구분해 맨 끝에 모아 붙인다.
        warn_text = "\n".join(f"- {w}" for w in tree.warnings)
        body = f"{body}\n\n[파싱 경고]\n{warn_text}" if body else f"[파싱 경고]\n{warn_text}"
    return body


class DocumentProcessTool(BaseTool):
    """
    문서 파일을 파싱해 텍스트를 청크 단위로 꺼내주는 읽기 전용 도구.

    지원 형식: PDF(.pdf), Word(.docx/.doc), Excel(.xlsx/.xls),
    PowerPoint(.pptx), 한글(.hwpx/.hwp).
    그 외 확장자는 평문 텍스트로 간주해 읽되, 바이너리면 명확한 오류를 돌려준다.

    동작 방식:
      - chunk_index를 주지 않으면(=0) 문서 개요 + 첫 청크를 반환한다.
      - chunk_index를 지정하면 해당 순번의 청크만 이어서 반환한다.
      - LLM은 반환된 안내 문구를 보고 다음 청크를 스스로 요청한다.

    BaseTool을 상속하며, 아래에서 name/schema/behavior/lifecycle을 순서대로 구현한다.
    """

    # ═══ 1. Identity(식별) — 레지스트리 등록과 LLM 도구 호출에 쓰이는 이름표 ═══

    @property
    def name(self) -> str:
        """도구의 정식 이름. LLM은 이 이름으로 도구를 호출한다."""
        return "DocumentProcess"

    @property
    def description(self) -> str:
        """LLM에게 노출되는 한 줄 설명. 어떤 파일을 다루는 도구인지 알린다."""
        return "Parse PDF, DOCX, XLSX, PPTX, HWPX, or HWP files."

    @property
    def aliases(self) -> list[str]:
        """정식 이름 대신 써도 되는 별칭들. 레지스트리에서 함께 등록된다."""
        return ["DocProcess", "ParseDocument"]

    @property
    def group(self) -> str:
        """도구 분류 그룹. 파일을 다루는 도구이므로 'file' 그룹에 속한다."""
        return "file"

    # ═══ 2. Schema(입력 스키마) — LLM에게 노출되는 파라미터 정의(JSON Schema) ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        """
        LLM이 이 도구를 호출할 때 넘길 수 있는 입력 파라미터를 정의한다.

        - file_path   : (필수) 읽을 문서의 절대경로.
        - chunk_index : (선택) 0부터 시작하는 청크 순번. 큰 문서를 이어 읽을 때 사용.
        - pages       : (선택) PDF에서만 의미 있는 페이지 범위 문자열(예: '1-5').
        """
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to document",
                },
                "chunk_index": {
                    "type": "integer",
                    "description": "Chunk number (0-based, for large docs)",
                },
                "pages": {
                    "type": "string",
                    "description": "PDF page range (e.g. '1-5')",
                },
            },
            "required": ["file_path"],
        }

    # ═══ 3. Behavior Flags(동작 플래그) — 권한·병렬 실행 판단에 쓰이는 특성 ═══
    # BaseTool의 기본값은 fail-closed(가장 제한적)이지만, 이 도구는 파일을 읽기만
    # 하고 아무 것도 바꾸지 않으므로 아래 두 플래그를 명시적으로 완화한다.

    @property
    def is_read_only(self) -> bool:
        """파일을 읽기만 하고 수정/삭제하지 않으므로 읽기 전용(True)."""
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        """부작용이 없어 여러 호출을 동시에 돌려도 안전하므로 병렬 안전(True)."""
        return True

    # ═══ 5. Lifecycle(생명주기) — 권한 확인 → 실제 실행 순서로 호출된다 ═══

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        이 도구를 실행해도 되는지 판단한다(권한 파이프라인 Layer 2에 해당).

        파일을 읽기만 하는 안전한 도구라 별도 확인 없이 항상 ALLOW를 돌려준다.
        (쓰기/삭제 도구였다면 여기서 경로 검증이나 사용자 확인을 요구했을 것이다.)
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> ToolResult:
        """
        문서를 파싱해 요청된 청크 하나를 텍스트로 반환하는 실제 실행 진입점.

        매개변수:
          - input_data : LLM이 넘긴 입력. file_path(필수), chunk_index/pages(선택).
          - context    : 실행 컨텍스트. options 안에 설정에서 주입된 값들이 들어 있다.
        반환:
          - ToolResult : 성공 시 청크 텍스트+개요, 실패 시 error 메시지.

        처리 순서:
          1. file_path 유효성과 파일 존재 여부를 확인한다.
          2. 캐시에 있으면 재사용, 없으면 파싱 후 청크로 분할해 캐시에 저장한다.
          3. chunk_index 범위를 검사하고 해당 청크를 꺼낸다.
          4. 문서 개요(전체 글자 수·청크 수·현재 위치)와 다음 청크 안내를 덧붙여 반환한다.
        """
        # 1) 입력 검증 — file_path가 비어 있으면 진행할 수 없다.
        file_path = input_data.get("file_path", "")
        if not file_path:
            return ToolResult.error("file_path가 필요합니다.")

        # 실제 파일이 존재하는지 확인한다. 없으면 명확한 오류를 돌려준다.
        path = Path(file_path)
        if not path.exists():
            return ToolResult.error(f"파일을 찾을 수 없습니다: {file_path}")

        # 2) 청크/통짜 예산 확정 — 하드코딩 외부화(2026-07-03). bootstrap·web/app이
        # config의 context_budgets 값을 ToolUseContext.options에 주입한다.
        # 값이 없으면(테스트/경량 경로) 모듈 상수 CHUNK_SIZE(현행 2500)로 폴백한다.
        # 이렇게 하면 설정 없이 호출하던 기존 코드도 동작이 변하지 않는다(무회귀).
        chunk_size = context.options.get("document_chunk_size") or CHUNK_SIZE
        # 통짜 반환 상한(글자). 문서 전체가 이 값 이하이면 청크로 쪼개지 않고 1회
        # 호출로 전문을 돌려준다 → 모델이 청크마다 재호출·진행 안내를 반복하던
        # 문제를 없앤다. 0(기본, 미주입)이면 비활성 → 항상 기존 청크 동작(무회귀).
        singleshot_chars = context.options.get("document_singleshot_chars") or 0

        # 캐시 키는 정규화 절대경로 + 파일 상태(mtime·크기) + 분할 파라미터로 삼는다.
        # 왜 상태·파라미터까지 넣나: 같은 경로에 다른 파일이 재업로드되거나(mtime/크기
        # 변경) config 청크값이 바뀌면 낡은 분할 결과를 재사용하면 안 되기 때문이다.
        stat = path.stat()
        cache_key = (
            f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
            f"|{singleshot_chars}|{chunk_size}"
        )
        if cache_key not in _document_cache:
            try:
                # 확장자에 맞는 파서로 문서 전체 텍스트를 추출한다.
                full_text = await self._extract_text(path, input_data.get("pages"))
            except Exception as e:
                # 파서 라이브러리 오류·손상된 파일 등은 여기서 잡아 오류로 감싼다.
                # (bare except 금지 규칙에 따라 예외 타입·메시지를 그대로 노출한다.)
                logger.error("문서 파싱 실패: %s — %s", file_path, e)
                return ToolResult.error(f"문서 파싱 실패: {type(e).__name__}: {e}")

            # 통짜 상한 이하이면 전문을 1청크로, 초과하면 chunk_size로 최소 분할한다.
            # (_split_chunks도 chunk_size 이하는 [text]로 돌려주므로, singleshot은
            #  "chunk_size보다 크지만 창에는 들어오는" 중간 크기 문서까지 통짜로 넓힌다.)
            if singleshot_chars > 0 and len(full_text) <= singleshot_chars:
                _document_cache[cache_key] = [full_text]
            else:
                _document_cache[cache_key] = self._split_chunks(full_text, chunk_size)

        # 3) 캐시에서 청크 리스트를 꺼내 전체 규모(청크 수·글자 수)를 계산한다.
        chunks = _document_cache[cache_key]
        total_chunks = len(chunks)
        total_chars = sum(len(c) for c in chunks)

        # 요청된 청크 순번(미지정 시 0=첫 청크). 범위를 벗어나면 오류로 안내한다.
        chunk_index = input_data.get("chunk_index", 0)
        if chunk_index < 0 or chunk_index >= total_chunks:
            return ToolResult.error(
                f"chunk_index {chunk_index}는 범위 밖입니다 (0~{total_chunks - 1})"
            )

        # 4) 결과 조립 — 본문 청크 앞뒤에 개요 헤더와 다음 청크 안내를 붙인다.
        chunk_text = chunks[chunk_index]
        # 헤더: 파일명·전체 규모·현재 위치를 한 줄로 요약해 LLM이 맥락을 잡게 한다.
        header = (
            f"[문서: {path.name} | "
            f"전체 {total_chars}자, {total_chunks}개 청크 | "
            f"현재: 청크 {chunk_index + 1}/{total_chunks}]"
        )

        # 푸터: 다음 청크가 남아 있으면 그걸 읽는 호출 예시를 보여주되, 청크마다
        # 진행 안내 문장을 반복하지 말라고 명시한다(반복 내레이션 억제 — 프롬프트
        # 규칙과 이중 방어). 마지막/유일 청크면 재호출 금지를 분명히 해 무한 반복을 막는다.
        if chunk_index < total_chunks - 1:
            footer = (
                f"\n\n[청크 {chunk_index + 1}/{total_chunks}. 남은 청크가 있습니다 — "
                f"진행 안내 문장 없이 즉시 "
                f'DocumentProcess(file_path="{file_path}", chunk_index={chunk_index + 1}) '
                f"를 호출해 이어 읽고, 모든 청크를 읽은 뒤 한 번에 분석하세요]"
            )
        elif total_chunks == 1:
            footer = (
                f"\n\n[문서 전체({total_chars}자)를 한 번에 반환했습니다. "
                f"이 문서에 대해 DocumentProcess를 다시 호출하지 마세요]"
            )
        else:
            footer = "\n\n[마지막 청크입니다. 문서 전체를 읽었습니다. 이제 한 번에 분석하세요]"

        # 헤더 + 본문 + 푸터를 합쳐 최종 텍스트를 만든다.
        result_text = f"{header}\n\n{chunk_text}{footer}"

        # 텍스트 외에 메타데이터(파일 경로·현재 청크·전체 청크/글자 수)도 함께 실어
        # 보낸다. 상위 계층이 진행률 표시 등에 쓸 수 있다.
        return ToolResult.success(
            result_text,
            file=str(path),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            total_chars=total_chars,
        )

    # ─── 텍스트 추출 (확장자별 분기) ───

    async def _extract_text(self, path: Path, pages: str | None = None) -> str:
        """
        파일 확장자를 보고 알맞은 파서를 골라 문서 전체 텍스트를 뽑아낸다.

        - .pdf              → _parse_pdf (pages 범위 지정 가능)
        - .docx / .doc      → _parse_docx
        - .xlsx / .xls      → _parse_xlsx
        - .pptx/.hwpx/.hwp  → _parse_via_ingest (core/ingest 파서 재사용)
        - 그 외             → _read_text_file (평문으로 간주하되 바이너리는 거부)

        async 인 이유: 인제스트 파서의 계약이 async parse() 이고, HWP 변환처럼
        수 초가 걸리는 작업을 이벤트 루프를 막지 않고 처리해야 하기 때문이다.
        (호출부는 call() 하나뿐이며 이미 async 다.)
        """
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._parse_pdf(path, pages)
        elif ext in (".docx", ".doc"):
            return self._parse_docx(path)
        elif ext in (".xlsx", ".xls"):
            return self._parse_xlsx(path)
        elif ext in _INGEST_EXTS:
            return await self._parse_via_ingest(path)
        else:
            return self._read_text_file(path)

    @staticmethod
    async def _parse_via_ingest(path: Path) -> str:
        """PPTX/HWPX/HWP를 인제스트 파서로 파싱해 평문으로 펼친다.

        왜 별도 스레드(asyncio.to_thread)에서 돌리나:
          인제스트 파서의 parse() 는 async 로 선언돼 있지만 내부는 파일 파싱·
          LibreOffice 변환 같은 '블로킹' 작업이다. 웹 서버의 이벤트 루프에서 그대로
          await 하면 그동안 다른 요청의 스트리밍까지 멈춘다. 그래서 워커 스레드에서
          독립 이벤트 루프(asyncio.run)로 돌려 루프 정지를 막는다.

        파서를 못 찾은 경우(라이브러리 미설치·매직바이트 불일치)는 예외를 던져
        call() 의 공통 오류 처리("문서 파싱 실패: ...")로 흘려보낸다.
        """
        parser = _get_ingest_registry().get_for_path(path)
        if parser is None:
            raise ValueError(
                f"{path.suffix.lower()} 형식을 처리할 파서가 없습니다. "
                "파일이 손상됐거나 해당 포맷 라이브러리가 설치되지 않았습니다."
            )

        tree = await asyncio.to_thread(lambda: asyncio.run(parser.parse(path)))

        # 본문이 한 글자도 없으면 사용자가 바로 조치할 수 있는 안내를 돌려준다.
        # 판정은 '경고를 뺀 본문' 기준이다 — 파서가 fail-soft 경고를 남기면
        # _flatten_tree 결과는 경고 때문에 비어 있지 않게 되므로, 그걸로 판정하면
        # "soffice 미설치" 같은 내부 경고만 사용자에게 노출되고 만다.
        if not _tree_has_text(tree):
            hint = _EXTRACT_FAILED_HINTS.get(
                path.suffix.lower(), _EXTRACT_FAILED_DEFAULT
            ).format(ext=path.suffix.lower())
            if tree.warnings:
                # 진단 정보는 안내 뒤에 덧붙인다(원인 추적은 가능하게 두되 주가 아니게).
                diag = "\n".join(f"- {w}" for w in tree.warnings)
                hint = f"{hint}\n\n[진단]\n{diag}"
            return hint
        return _flatten_tree(tree)

    @staticmethod
    def _read_text_file(path: Path) -> str:
        """알려진 문서 포맷이 아닌 파일을 평문 텍스트로 읽는다.

        두 가지를 막는다:
          1) 바이너리를 평문으로 읽어 깨진 문자열이 모델에 들어가는 것 —
             앞부분에 NUL 바이트가 있으면 바이너리로 보고 명확한 오류를 낸다.
             (예전에는 .hwp/.pptx가 여기로 흘러들어 쓰레기 텍스트가 됐다.)
          2) 한글 CP949(euc-kr) 텍스트가 물음표로 뭉개지는 것 — UTF-8 해독이
             실패하면 CP949를 한 번 더 시도한 뒤에야 대체 문자로 넘어간다.
        """
        raw = path.read_bytes()

        # 1) BOM 우선 판별 — UTF-16/UTF-32 텍스트는 본문 곳곳에 NUL 바이트가 있어
        #    아래 바이너리 스니핑에 그대로 걸린다(Windows 메모장의 '유니코드' 저장이
        #    대표적). BOM이 있으면 인코딩의 확실한 증거이므로 먼저 해독한다.
        #    UTF-32LE BOM(FF FE 00 00)은 UTF-16LE BOM(FF FE)으로 시작하므로
        #    반드시 더 긴 것부터 검사해야 한다(순서 중요).
        for bom, encoding in (
            (b"\xef\xbb\xbf", "utf-8-sig"),
            (b"\xff\xfe\x00\x00", "utf-32-le"),
            (b"\x00\x00\xfe\xff", "utf-32-be"),
            (b"\xff\xfe", "utf-16"),
            (b"\xfe\xff", "utf-16"),
        ):
            if raw.startswith(bom):
                return DocumentProcessTool._normalize_newlines(
                    raw.decode(encoding, errors="replace")
                )

        # 2) 바이너리 판별 — 텍스트 파일에는 NUL 바이트가 사실상 나오지 않는다.
        if b"\x00" in raw[:4096]:
            raise ValueError(
                f"지원하지 않는 파일 형식입니다({path.suffix.lower() or '확장자 없음'}). "
                "지원 형식: PDF, DOCX, XLSX, PPTX, HWPX, HWP, 그리고 평문 텍스트."
            )

        # 3) 인코딩 추정 — UTF-8 → CP949 → (최후) 대체 문자.
        for encoding in ("utf-8", "cp949"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")

        return DocumentProcessTool._normalize_newlines(text)

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        """개행을 LF로 통일한다(CRLF/CR → LF).

        바이트로 읽으면 파이썬 텍스트 모드의 universal newlines 변환이 걸리지
        않아 Windows 파일에 \\r 이 그대로 남는다. 그만큼 글자 수가 늘어 청크
        분할 경계가 어긋나므로 여기서 맞춰 준다(무회귀 보정).
        """
        return text.replace("\r\n", "\n").replace("\r", "\n")

    # ─── 청크 분할 ───

    @staticmethod
    def _split_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
        """
        긴 텍스트를 chunk_size(글자 수) 이하의 여러 조각으로 나눈다.

        핵심 아이디어: 아무 데서나 자르지 않고 줄(\n) 경계에서만 끊는다. 그래야
        문장이나 단락 중간이 잘려 문맥이 깨지는 일을 최대한 줄일 수 있다.

        chunk_size 기본값은 모듈 상수 CHUNK_SIZE(현행 2500)라서, 인자 없이 부르던
        기존 코드/테스트는 동작이 그대로다(무회귀). 실제 호출부(call)는 config에서
        받은 값을 넘겨 상황에 맞는 크기로 자른다.

        반환: 청크 문자열들의 리스트(항상 최소 1개).
        """
        # 짧은 문서는 나눌 필요가 없으니 통째로 한 청크로 돌려준다.
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        paragraphs = text.split("\n")  # 줄 단위로 쪼갠 뒤 다시 묶어 나간다.
        current_chunk: list[str] = []  # 지금 채우고 있는 청크의 줄 모음
        current_len = 0  # 현재 청크에 쌓인 글자 수(개행 포함)

        for para in paragraphs:
            para_len = len(para) + 1  # +1은 join할 때 다시 붙는 개행 한 글자 몫
            # 이 줄을 더하면 한도를 넘고, 이미 담은 내용이 있다면 → 현재 청크를 확정하고
            # 새 청크를 시작한다. (담은 게 없으면 한 줄이 한도보다 길어도 그냥 넣는다.)
            if current_len + para_len > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(para)
            current_len += para_len

        # 마지막까지 남아 있던 줄들도 잊지 말고 청크로 확정한다.
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # 만약(이론상) 아무 청크도 안 만들어졌다면 원문을 그대로 돌려 빈 결과를 막는다.
        return chunks if chunks else [text]

    # ─── 파서 구현 ───

    @staticmethod
    def _parse_pdf(path: Path, pages: str | None) -> str:
        """
        PDF에서 페이지별 텍스트를 추출해 하나의 문자열로 합친다.

        pages 인자가 있으면 그 범위만 읽는다(예: '1-5' → 1~5페이지, '3' → 3페이지).
        내부는 0-기반 인덱스라 시작값에서 1을 빼고, 범위는 전체 페이지 수로 잘라낸다.
        텍스트가 하나도 안 잡히면(스캔 이미지 PDF 등) 그 사실을 알리는 문구를 돌려준다.
        """
        # pypdf는 무거운 의존성이라 함수 안에서 지연 import한다(모듈 로딩 비용 절감).
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)

        # 읽을 페이지 범위 결정 — 지정이 있으면 파싱, 없으면 전체.
        if pages:
            parts = pages.split("-")
            start = max(int(parts[0]) - 1, 0)  # 1-기반 입력을 0-기반으로, 음수 방지
            end = min(int(parts[-1]), total_pages)  # 실제 페이지 수를 넘지 않게 제한
        else:
            start = 0
            end = total_pages

        # 페이지마다 텍스트를 뽑아 "[페이지 N]" 머리표와 함께 모은다(빈 페이지는 건너뜀).
        texts = []
        for i in range(start, end):
            page_text = reader.pages[i].extract_text() or ""
            if page_text.strip():
                texts.append(f"[페이지 {i + 1}]\n{page_text}")

        # 한 글자도 못 뽑았다면 스캔본 PDF일 가능성이 크므로 그 점을 알린다.
        if not texts:
            return f"[PDF {total_pages}페이지, 텍스트 추출 불가 (스캔 이미지일 수 있음)]"

        return "\n\n".join(texts)

    @staticmethod
    def _parse_docx(path: Path) -> str:
        """
        Word(.docx) 문서에서 본문 단락과 표(table) 내용을 텍스트로 추출한다.

        - 문단(paragraph)은 비어 있지 않은 것만 순서대로 담는다.
        - 표는 행마다 셀들을 ' | '로 이어 붙여 한 줄로 만든다.
        추출된 내용이 전혀 없으면 그 사실을 알리는 문구를 돌려준다.
        """
        # python-docx도 지연 import — 에어갭 환경에 오프라인 설치된 라이브러리다.
        from docx import Document

        doc = Document(str(path))
        texts = []

        # 본문 단락 수집(빈 줄 제외).
        for para in doc.paragraphs:
            if para.text.strip():
                texts.append(para.text)

        # 표 내용 수집 — 각 행의 비어있지 않은 셀들을 파이프로 연결한다.
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
        """
        Excel(.xlsx) 통합문서에서 모든 시트의 셀 값을 텍스트로 펼친다.

        - read_only + data_only 모드로 연다: 메모리를 아끼고, 수식이 아니라 계산된
          값을 읽기 위함이다.
        - 시트마다 "[시트: 이름]" 머리표를 붙이고, 각 행의 셀을 ' | '로 이어 붙인다.
        - 시트당 최대 500행까지만 담고, 그 이상은 "... [N+ 행]"으로 요약해 폭주를 막는다.
        """
        # openpyxl 지연 import.
        from openpyxl import load_workbook

        wb = load_workbook(str(path), read_only=True, data_only=True)
        texts = []

        # 시트를 순서대로 돌며 내용을 수집한다.
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            texts.append(f"[시트: {sheet_name}]")

            row_count = 0
            for row in ws.iter_rows(values_only=True):
                # None 셀은 빈 문자열로 바꿔 자리를 유지한다.
                cells = [str(c) if c is not None else "" for c in row]
                # 완전히 빈 행은 건너뛰고, 내용이 있는 행만 한 줄로 합쳐 담는다.
                if any(c.strip() for c in cells):
                    texts.append(" | ".join(cells))
                    row_count += 1
                    # 시트가 너무 크면 500행에서 끊고 남은 분량이 있음을 표시한다.
                    if row_count >= 500:
                        texts.append(f"... [{sheet_name} 시트 {row_count}+ 행]")
                        break

        # read_only 모드로 연 파일 핸들을 반드시 닫아 자원을 해제한다.
        wb.close()
        return "\n".join(texts)

    # ═══ 7. UI Hints(UI 힌트) — 실행 중 사용자에게 보여줄 진행 문구 ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        """도구 실행 중 CLI/웹 UI에 표시할 짧은 진행 상태 문구를 돌려준다."""
        return "Parsing document..."
