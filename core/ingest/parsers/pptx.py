"""
PPTX 파서 — python-pptx(MIT)로 PowerPoint 파일을 구조 트리로 변환한다.

왜 PPTX 가 1차 구현 대상인가 (v7.3 Part 3.1):
  PPTX 는 슬라이드/도형/텍스트박스/표/그림을 "네이티브 객체"로 제공한다.
  즉 레이아웃 모델이나 OCR 없이도 구조를 그대로 읽어올 수 있어 가장 쉽다.
  "충돌 없는 임베딩"의 핵심인 읽기 순서 복원도, 각 도형의 (top,left) 좌표를
  정렬하는 것만으로 해결된다(좌측 텍스트박스와 우측 텍스트박스가 뒤섞이지
  않게 함).

처리 개요:
  파일 → 슬라이드 순회 → 슬라이드마다 SLIDE 노드 생성
    └ 슬라이드 안 도형을 (top,left) 정렬 → 읽기 순서대로 자식 노드 생성
        - 텍스트박스/플레이스홀더 → TEXTBOX 또는 PARAGRAPH/LIST_ITEM
        - 표 → TABLE (행/열 보존)
        - 그림 → FIGURE_CAPTION (대체텍스트/캡션이 있으면)
        - 그룹(GroupShape) → 재귀 순회

fail-soft (anti-pattern #8):
  도형 하나를 읽다 실패해도 전체를 중단하지 않는다. 실패 사유를 warnings 에
  모으고 나머지를 계속 처리한다. bare except 는 쓰지 않고 구체 예외만 잡는다.

의존성 방향 (P2): core.ingest.types / parser_base 만 의존. core/rag·model 무관.
에어갭: python-pptx 는 로컬 파일만 읽는다(외부 네트워크 없음).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.exc import PackageNotFoundError

from core.ingest.parser_base import DocumentParser
from core.ingest.types import DocumentNode, DocumentTree, ElementType

logger = logging.getLogger("nexus.ingest.parsers.pptx")

# PPTX(OOXML)는 ZIP 컨테이너다. 매직바이트는 "PK\x03\x04"(ZIP 시그니처).
# can_parse() 에서 확장자뿐 아니라 이 시그니처도 확인해 fail-closed 한다.
_ZIP_MAGIC = b"PK\x03\x04"

# 좌표가 없는 도형(placeholder 등)을 정렬에서 맨 뒤로 보내기 위한 큰 값.
# v7.3 Part 3.3 의 _ordered_shapes 발췌와 동일한 의도.
_NO_COORD = 10**18


class PptxParser(DocumentParser):
    """
    PowerPoint(.pptx) 파일을 구조 트리로 파싱하는 파서.

    네이티브 구조를 그대로 읽으므로 GPU 가 필요 없다(requires_gpu=False).
    """

    # ─── 정체성/플래그 ───

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        # .pptx 만 처리. .ppt(구포맷)는 대상 아님.
        return (".pptx",)

    @property
    def requires_gpu(self) -> bool:
        # 네이티브 구조 파싱 — GPU 레이아웃/OCR 모델 불필요.
        return False

    # ─── 처리 가능 여부 (fail-closed) ───

    def can_parse(self, path: Path) -> bool:
        """
        확장자(.pptx) + 매직바이트(ZIP 시그니처)로 처리 가능 여부를 판단한다.

        fail-closed: 파일을 읽다 문제가 생기거나 시그니처가 다르면 False.
        """
        if path.suffix.lower() != ".pptx":
            return False
        try:
            with path.open("rb") as f:
                head = f.read(4)
        except OSError as e:
            # 파일을 열 수 없음 — 처리 불가로 간주(fail-closed).
            logger.debug("can_parse 파일 열기 실패: %s (%s)", path, e)
            return False
        return head == _ZIP_MAGIC

    # ─── 핵심: parse ───

    async def parse(self, path: Path) -> DocumentTree:
        """
        .pptx 를 슬라이드 단위 구조 트리로 변환한다.

        반환: DocumentTree(title, source_path, nodes=SLIDE 노드들, doc_format="pptx",
        warnings=fail-soft 경고).

        치명적 실패(파일 손상 등)는 빈 트리 + 경고로 표현한다(파이프라인 중단 방지).
        """
        warnings: list[str] = []

        # 1) 파일 로드 — 손상/형식 오류는 부분 결과(빈 트리)+경고로 처리.
        try:
            presentation = Presentation(str(path))
        except (PackageNotFoundError, OSError, ValueError, KeyError) as e:
            # python-pptx 는 손상/비정상 파일에서 PackageNotFoundError 를 던진다.
            # 이 예외는 PythonPptxError→Exception 계열이라 OSError/ValueError/KeyError
            # 어디에도 속하지 않으므로, 반드시 명시적으로 포착해야 fail-soft(빈 트리+
            # 경고)가 보장된다. bare except 금지 — 구체 예외만 포착.
            msg = f"PPTX 로드 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            return DocumentTree(
                title=path.stem,
                source_path=str(path),
                nodes=(),
                doc_format="pptx",
                warnings=(msg,),
            )

        # 2) 슬라이드 순회 → SLIDE 노드 생성.
        slide_nodes: list[DocumentNode] = []
        for slide_index, slide in enumerate(presentation.slides, start=1):
            slide_node = self._parse_slide(slide, slide_index, warnings)
            slide_nodes.append(slide_node)

        # 3) 문서 제목: 첫 슬라이드 제목 placeholder → 없으면 파일명(stem).
        title = self._document_title(slide_nodes) or path.stem

        return DocumentTree(
            title=title,
            source_path=str(path),
            nodes=tuple(slide_nodes),
            doc_format="pptx",
            warnings=tuple(warnings),
        )

    # ─────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────

    def _parse_slide(self, slide, slide_index: int, warnings: list[str]) -> DocumentNode:
        """
        슬라이드 1장을 SLIDE 노드(+자식 노드들)로 변환한다.

        단계:
          1. 슬라이드 제목 placeholder 에서 heading(슬라이드 제목)을 추출.
             이 제목이 슬라이드 내 모든 자식 노드의 heading_path 가 된다
             (문맥 연결 — 같은 슬라이드 내용은 같은 제목 맥락을 공유).
          2. 도형을 (top,left) 좌표로 정렬해 읽기 순서를 복원.
          3. 각 도형을 종류별 노드로 변환(그룹은 재귀).
        """
        # 1) 슬라이드 제목 — heading_path 의 한 단계가 된다.
        slide_title = self._slide_title(slide)
        # heading_path 는 "슬라이드 N" 또는 실제 제목을 사용한다.
        # 제목이 있으면 그것을, 없으면 "슬라이드 N" 라벨로 맥락을 표시.
        heading = slide_title or f"슬라이드 {slide_index}"
        heading_path: tuple[str, ...] = (heading,)

        # 2)+3) 읽기 순서대로 도형을 자식 노드로 변환.
        children = self._shapes_to_nodes(
            shapes=self._ordered_shapes(slide.shapes),
            heading_path=heading_path,
            page=slide_index,
            warnings=warnings,
            skip_title=slide_title,  # 제목 placeholder 텍스트 중복 방지
        )

        return DocumentNode(
            element_type=ElementType.SLIDE,
            text=heading if slide_title else "",  # 제목이 있으면 SLIDE 노드 텍스트로
            heading_path=(),  # SLIDE 자신은 경로의 루트
            page=slide_index,
            order=slide_index,
            children=tuple(children),
        )

    @staticmethod
    def _ordered_shapes(shapes) -> list:
        """
        도형을 (top, left) 좌표순으로 정렬해 읽기 순서를 복원한다.

        v7.3 Part 3.3 발췌와 동일한 의도:
          좌표가 없는 도형(레이아웃 placeholder 등)은 매우 큰 값으로 처리해
          맨 뒤로 보낸다. 이렇게 하면 좌측 텍스트박스와 우측 텍스트박스가
          읽기 순서 없이 뒤섞이는 "충돌"을 막는다.

        sorted 는 안정 정렬이므로 좌표가 같은 도형은 원래 순서를 유지한다.
        """

        def sort_key(shape):
            # getattr 로 안전 접근 — 일부 도형 타입은 top/left 가 None 일 수 있다.
            top = getattr(shape, "top", None)
            left = getattr(shape, "left", None)
            return (
                top if top is not None else _NO_COORD,
                left if left is not None else _NO_COORD,
            )

        return sorted(shapes, key=sort_key)

    def _shapes_to_nodes(
        self,
        shapes: list,
        heading_path: tuple[str, ...],
        page: int,
        warnings: list[str],
        skip_title: str | None,
    ) -> list[DocumentNode]:
        """
        정렬된 도형 목록을 자식 노드 목록으로 변환한다(읽기 순서 유지).

        도형 종류별 매핑:
          - GROUP        → 재귀 순회(내부 도형도 좌표 정렬)
          - TABLE        → TABLE 노드 (행/열 보존)
          - PICTURE      → FIGURE_CAPTION (대체텍스트가 있을 때만)
          - 그 외 텍스트  → TEXTBOX/PARAGRAPH/LIST_ITEM

        order 는 정렬된 순서대로 0,1,2... 를 부여해 읽기 순서를 보존한다.
        """
        nodes: list[DocumentNode] = []
        order = 0

        for shape in shapes:
            # 도형 타입 판별 — shape_type 접근 자체가 실패할 수 있어 보호한다.
            try:
                shape_type = shape.shape_type
            except (AttributeError, ValueError) as e:
                warnings.append(
                    f"슬라이드 {page}: 도형 타입 판별 실패 — 건너뜀 ({type(e).__name__}: {e})"
                )
                continue

            # (a) 그룹 도형 — 내부 도형을 재귀적으로 좌표 정렬해 순회.
            if shape_type == MSO_SHAPE_TYPE.GROUP:
                try:
                    inner = self._ordered_shapes(shape.shapes)
                except (AttributeError, ValueError) as e:
                    warnings.append(
                        f"슬라이드 {page}: 그룹 내부 순회 실패 — 건너뜀 ({type(e).__name__}: {e})"
                    )
                    continue
                child_nodes = self._shapes_to_nodes(
                    inner, heading_path, page, warnings, skip_title=None
                )
                # 재귀 결과의 order 는 부모 흐름에 맞춰 재부여한다.
                for cn in child_nodes:
                    nodes.append(cn.model_copy(update={"order": order}))
                    order += 1
                continue

            # (b) 표 도형 — 행/열을 보존한 TABLE 노드 1개.
            if shape_type == MSO_SHAPE_TYPE.TABLE:
                table_node = self._table_node(shape, heading_path, page, order, warnings)
                if table_node is not None:
                    nodes.append(table_node)
                    order += 1
                continue

            # (c) 그림 도형 — 대체텍스트/캡션이 있으면 FIGURE_CAPTION.
            if shape_type == MSO_SHAPE_TYPE.PICTURE:
                caption = self._picture_caption(shape)
                if caption:
                    nodes.append(
                        DocumentNode(
                            element_type=ElementType.FIGURE_CAPTION,
                            text=caption,
                            heading_path=heading_path,
                            page=page,
                            order=order,
                        )
                    )
                    order += 1
                continue

            # (d) 텍스트가 있는 그 외 도형(텍스트박스/플레이스홀더/오토셰이프).
            text_nodes = self._text_shape_nodes(shape, heading_path, page, order, skip_title)
            for tn in text_nodes:
                nodes.append(tn)
                order += 1

        return nodes

    def _table_node(
        self,
        shape,
        heading_path: tuple[str, ...],
        page: int,
        order: int,
        warnings: list[str],
    ) -> DocumentNode | None:
        """
        표 도형을 TABLE 노드로 변환한다(행/열 구조 보존).

        직렬화 형식: 각 행을 " | " 로 구분된 셀로, 행은 줄바꿈으로 연결한다.
        예) "이름 | 부서\n홍길동 | 보안팀". 이렇게 하면 행/열 의미가 보존되어
        검색·표시에 활용할 수 있다(v7.3 Part 1.1 — 표 셀이 본문과 섞이지 않게).

        표 접근 실패는 fail-soft: 경고를 남기고 None 반환(노드 생략).
        """
        try:
            table = shape.table
            rows_text: list[str] = []
            for row in table.rows:
                # 각 셀의 텍스트를 좌→우 순서로 모아 " | " 로 연결.
                cells = [cell.text.strip() for cell in row.cells]
                rows_text.append(" | ".join(cells))
        except (AttributeError, ValueError) as e:
            warnings.append(f"슬라이드 {page}: 표 파싱 실패 — 건너뜀 ({type(e).__name__}: {e})")
            return None

        content = "\n".join(rt for rt in rows_text if rt.strip())
        if not content:
            return None

        return DocumentNode(
            element_type=ElementType.TABLE,
            text=content,
            heading_path=heading_path,
            page=page,
            order=order,
        )

    @staticmethod
    def _picture_caption(shape) -> str:
        """
        그림 도형에서 대체텍스트(alt text) 또는 캡션을 추출한다.

        python-pptx 는 도형 이름/대체텍스트를 노출한다. 대체텍스트가 가장
        의미 있는 캡션이므로 우선 사용하고, 없으면 빈 문자열(노드 생략).

        대체텍스트는 XML 의 cNvPr 요소 descr 속성에 들어있다. 안전 접근 후
        없으면 빈 문자열을 돌려준다.
        """
        # 1) python-pptx 공개 API 로 접근 가능한 텍스트프레임(드물게 캡션 포함).
        try:
            descr = shape._element._nvXxPr.cNvPr.get("descr", "")
        except (AttributeError, KeyError, ValueError):
            descr = ""
        return (descr or "").strip()

    def _text_shape_nodes(
        self,
        shape,
        heading_path: tuple[str, ...],
        page: int,
        order: int,
        skip_title: str | None,
    ) -> list[DocumentNode]:
        """
        텍스트를 가진 도형(텍스트박스/플레이스홀더 등)을 노드 목록으로 변환한다.

        규칙:
          - 도형에 텍스트프레임이 없으면 빈 목록.
          - 제목 placeholder 와 동일한 텍스트(skip_title)는 중복이므로 제외
            (이미 슬라이드 제목/heading 으로 반영됨).
          - 문단(paragraph)별로 노드를 만들되, 글머리표(level>0)는 LIST_ITEM,
            그 외는 PARAGRAPH 로 분류한다.
          - 도형 전체는 TEXTBOX 의미를 가지나, 검색 정밀도를 위해 문단 단위로
            쪼개 노드를 만든다(빈 문단은 생략).

        텍스트 접근 실패는 빈 목록 반환(fail-soft — 호출부는 계속 진행).
        """
        try:
            if not shape.has_text_frame:
                return []
            text_frame = shape.text_frame
        except (AttributeError, ValueError):
            return []

        nodes: list[DocumentNode] = []
        local_order = order
        for para in text_frame.paragraphs:
            # 문단 안 run 들의 텍스트를 합친다.
            try:
                para_text = "".join(run.text for run in para.runs).strip()
            except (AttributeError, ValueError):
                continue
            if not para_text:
                continue
            # 제목과 동일하면 중복 — 제외.
            if skip_title and para_text == skip_title.strip():
                continue

            # 들여쓰기 level 이 0보다 크면 글머리표 항목으로 간주.
            level = getattr(para, "level", 0) or 0
            element_type = ElementType.LIST_ITEM if level > 0 else ElementType.PARAGRAPH

            nodes.append(
                DocumentNode(
                    element_type=element_type,
                    text=para_text,
                    heading_path=heading_path,
                    page=page,
                    order=local_order,
                )
            )
            local_order += 1

        return nodes

    @staticmethod
    def _slide_title(slide) -> str | None:
        """
        슬라이드 제목 placeholder 에서 제목 텍스트를 추출한다.

        python-pptx 의 slide.shapes.title 은 제목 placeholder 가 있으면
        그 도형을, 없으면 None 을 반환한다. 안전하게 접근한다.
        """
        try:
            title_shape = slide.shapes.title
        except (AttributeError, ValueError):
            return None
        if title_shape is None:
            return None
        try:
            text = (title_shape.text or "").strip()
        except (AttributeError, ValueError):
            return None
        return text or None

    @staticmethod
    def _document_title(slide_nodes: list[DocumentNode]) -> str | None:
        """
        문서 제목 후보를 첫 슬라이드의 텍스트(제목)에서 가져온다.

        _parse_slide 에서 제목이 있으면 SLIDE 노드의 text 에 담아 두었으므로
        첫 SLIDE 노드의 text 를 사용한다. 없으면 None(호출부가 파일명으로 폴백).
        """
        for node in slide_nodes:
            if node.element_type == ElementType.SLIDE and node.text.strip():
                return node.text.strip()
        return None
