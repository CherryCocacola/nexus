"""
Docling 고품질 PDF 파서 — 레이아웃 인식으로 PDF 를 구조 트리로 변환한다.

왜 Docling(고품질)인가 (v7.3 로드맵 단계 6 — PDF 고품질):
  pdfplumber(경량)는 PDF 안의 "텍스트 글자"를 좌표로 긁어 폰트 크기 휴리스틱으로
  제목/본문을 추정한다. 반면 Docling 은 레이아웃 인식 모델(딥러닝)로 페이지를
  분석해 "이 블록은 제목/본문/표/그림" 인지 분류하고, 사람이 읽는 순서(reading
  order)까지 복원한다. 다단(多段) 편집, 복잡한 표, 머리말/꼬리말이 섞인 실제
  문서에서 경량 파서보다 훨씬 정확한 구조를 얻는다.

  대신 레이아웃 모델을 돌려야 하므로 GPU(A100/H200, ~8GB)가 권장된다(CPU 도
  동작하나 느리다). 따라서 requires_gpu=True 로 표시하고, 티어 분기(v7.3 Part 4)
  에서 GPU 가 있는 호스트에서만 우선 사용되도록 한다(GPU 없으면 pdfplumber 폴백).

Docling 실제 API (런타임 확인 결과):
  from docling.document_converter import DocumentConverter
  converter = DocumentConverter()
  result = converter.convert(path)          # ConversionResult
  doc = result.document                      # DoclingDocument
  for item, _level in doc.iterate_items():   # 읽기 순서대로 아이템 순회
      # item 종류:
      #   TitleItem          → 문서 제목         → SECTION
      #   SectionHeaderItem  → 섹션/소제목(level)→ SECTION(level 1)/SUBHEADING
      #   TextItem(label)    → 본문/캡션/리스트 등 → label 로 세분
      #   ListItem           → 목록 항목         → LIST_ITEM
      #   TableItem          → 표               → TABLE (행/열 보존)
      #   PictureItem        → 그림(캡션 보유)   → FIGURE_CAPTION
      item.text                                # 텍스트 아이템의 본문(str)
      item.prov[0].page_no                     # 출처 페이지 번호(1부터)
      item.label                               # DocItemLabel (text/title/...)
      table.data.table_cells                   # 표 셀 목록(offset idx 보유)
      table.data.num_rows / num_cols           # 표 크기
      picture.caption_text(doc)                # 그림 캡션 텍스트(str)

  예외 타입(docling.exceptions): BaseError 를 루트로
    ConversionError / OperationNotAllowed / SecurityError.
  변환 실패(손상/암호/지원 안 됨)는 이 계열을 던지므로 except 에 BaseError 를
  포함해 fail-soft 한다(과거 pptx/pdf_plumber 에서 라이브러리 고유 예외 누락으로
  fail-soft 가 깨진 전례가 있어, 실제 예외 타입을 확인해 명시 포착한다).

모델 다운로드(에어갭 주의):
  Docling 은 첫 실행 시 레이아웃/표 인식 모델 가중치를 인터넷에서 내려받을 수
  있다(개발 환경은 인터넷 OK). 배포(에어갭) 시에는 모델을 오프라인 번들로
  사전 배치해야 하며(HF 캐시/DOCLING_ARTIFACTS_PATH 등), 런타임 다운로드에
  의존하지 않는다. 이 파서 코드는 import / 호출만 하고 설치·다운로드 코드를
  포함하지 않는다(에어갭 규칙, anti-pattern #10).

fail-soft (anti-pattern #8):
  변환 실패는 BaseError/OSError/ValueError 등 구체 예외로 포착해 빈 트리 +
  warning 으로 표현하고 예외를 전파하지 않는다. 개별 아이템(표/그림) 추출
  실패도 그 아이템만 건너뛰고 나머지를 계속 처리한다. bare except 금지.

의존성 방향 (P2): core.ingest.types / parser_base 만 의존. core/rag·model 무관.
에어갭: convert() 는 로컬 파일만 읽는다(모델 가중치는 사전 배치 전제).
"""

from __future__ import annotations

import logging
from pathlib import Path

# Docling 본체 — 문서 변환기와 예외 계열.
# (모델은 import 만 한다 — 런타임 pip install / 다운로드 코드는 넣지 않는다.)
from docling.document_converter import DocumentConverter
from docling.exceptions import BaseError as DoclingError
from docling_core.types.doc.document import (
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.labels import DocItemLabel

from core.ingest.parser_base import DocumentParser
from core.ingest.types import DocumentNode, DocumentTree, ElementType

logger = logging.getLogger("nexus.ingest.parsers.docling_layout")

# PDF 파일의 매직바이트 — 파일 선두는 항상 "%PDF" 로 시작한다(PDF 규격).
# can_parse() 에서 확장자뿐 아니라 이 시그니처도 확인해 fail-closed 한다.
_PDF_MAGIC = b"%PDF"

# 그림 캡션 노드로 인정할 최소 길이(글자 수). 캡션이 비었거나 한두 글자뿐이면
# 검색 가치가 없어 노드를 만들지 않는다(노이즈 방지).
_MIN_CAPTION_CHARS = 2


class DoclingParser(DocumentParser):
    """
    PDF(.pdf) 파일을 Docling 레이아웃 모델로 구조 트리로 파싱하는 고품질 파서.

    제목/본문/표/그림을 레이아웃 인식으로 분류하고 읽기 순서를 복원한다.
    레이아웃 모델 추론에 GPU 가 권장되므로 requires_gpu=True 다(GPU 없는
    호스트에서는 레지스트리 우선순위상 pdfplumber 경량 파서로 폴백된다).
    """

    def __init__(self) -> None:
        """
        파서 인스턴스를 만든다.

        왜 변환기를 지연 생성(lazy)하는가:
          DocumentConverter() 생성 시 레이아웃 모델 로딩/다운로드가 트리거될 수
          있다(무겁다). 레지스트리에 등록만 하고 실제 파싱이 한 번도 일어나지
          않는 호스트에서까지 모델을 적재하면 낭비이므로, 첫 parse() 호출 때
          한 번만 만들고 이후 재사용한다(_converter 캐시).
        """
        self._converter: DocumentConverter | None = None

    # ─── 정체성/플래그 ───

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        # .pdf 만 처리. (pdfplumber 와 같은 확장자 — priority 로 우선순위를 가른다.)
        return (".pdf",)

    @property
    def requires_gpu(self) -> bool:
        # 레이아웃/표 인식 딥러닝 모델 추론 — GPU(~8GB) 권장.
        # 티어 분기(v7.3 Part 4)가 GPU 호스트에서만 이 파서를 우선시키는 데 쓴다.
        return True

    # ─── 처리 가능 여부 (fail-closed) ───

    def can_parse(self, path: Path) -> bool:
        """
        확장자(.pdf) + 매직바이트("%PDF")로 처리 가능 여부를 판단한다.

        fail-closed: 파일을 읽다 문제가 생기거나 시그니처가 다르면 False.
        (pdfplumber 파서의 can_parse 와 동일 규칙 — 같은 확장자를 공유하므로
        시그니처 검증 기준을 일치시킨다.)
        """
        if path.suffix.lower() != ".pdf":
            return False
        try:
            with path.open("rb") as f:
                head = f.read(4)
        except OSError as e:
            # 파일을 열 수 없음 — 처리 불가로 간주(fail-closed).
            logger.debug("can_parse 파일 열기 실패: %s (%s)", path, e)
            return False
        return head == _PDF_MAGIC

    # ─── 핵심: parse ───

    async def parse(self, path: Path) -> DocumentTree:
        """
        .pdf 를 Docling 레이아웃 인식으로 구조 트리로 변환한다.

        반환: DocumentTree(title, source_path, nodes=레이아웃 노드들,
        doc_format="pdf", warnings=fail-soft 경고).

        - 최상위 노드는 읽기 순서대로 나열된 내용 노드
          (SECTION/SUBHEADING/PARAGRAPH/TABLE/FIGURE_CAPTION/LIST_ITEM)다.
          PDF 는 페이지가 곧 물리 단위이므로 페이지 컨테이너 노드는 두지 않고,
          각 노드의 page 필드에 출처 페이지 번호를 담는다(pdfplumber 와 동일).
        - 치명적 실패(손상/암호/지원 안 됨/모델 부재)는 빈 트리 + 경고로
          표현한다(파이프라인 중단 방지 — fail-soft).
        """
        warnings: list[str] = []

        # 1) 변환기 준비(지연 생성). 모델 로딩 자체가 실패할 수 있으므로 포착한다.
        try:
            converter = self._ensure_converter()
        except (DoclingError, OSError, ValueError, RuntimeError, ImportError) as e:
            # 모델/런타임 준비 실패(에어갭에서 모델 미배치 등) — fail-soft.
            msg = f"Docling 변환기 초기화 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            return self._empty_tree(path, (msg,))

        # 2) 변환 — Docling 이 레이아웃 모델로 PDF 를 분석한다(무거운 단계).
        #    convert() 는 손상/암호/지원 안 됨에서 docling BaseError 계열을,
        #    파일 입출력 문제에서 OSError 를, 내부 구현에 따라 ValueError/
        #    RuntimeError 를 던질 수 있다. 모두 포착해 빈 트리+경고로 흡수한다.
        try:
            result = converter.convert(str(path))
            document = result.document
        except (DoclingError, OSError, ValueError, RuntimeError) as e:
            msg = f"Docling 변환 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            return self._empty_tree(path, (msg,))

        # 3) 변환 결과(DoclingDocument)를 노드 목록으로 옮긴다.
        nodes = self._nodes_from_document(document, warnings)

        return DocumentTree(
            title=path.stem,
            source_path=str(path),
            nodes=tuple(nodes),
            doc_format="pdf",
            warnings=tuple(warnings),
        )

    # ─────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────

    def _ensure_converter(self) -> DocumentConverter:
        """
        DocumentConverter 를 한 번만 만들어 캐시하고 재사용한다(지연 생성).

        첫 호출 시 레이아웃 모델 적재가 일어날 수 있어 비용이 크므로, 인스턴스당
        한 번만 생성한다. 기본 설정(DocumentConverter())을 쓴다 — 파이프라인
        세부 튜닝은 과설계 금지 원칙에 따라 하지 않는다(필요 시 어댑터 슬롯에서).
        """
        if self._converter is None:
            self._converter = DocumentConverter()
        return self._converter

    def _nodes_from_document(self, document, warnings: list[str]) -> list[DocumentNode]:
        """
        DoclingDocument 를 읽기 순서(iterate_items)대로 노드 목록으로 변환한다.

        - iterate_items() 는 (item, level) 튜플을 읽기 순서대로 yield 한다.
          level 은 트리 깊이라 여기선 쓰지 않고, 아이템의 label/타입으로 분류한다.
        - heading_path(소제목 경로)는 순회하며 만난 제목/소제목을 누적해 유지한다.
          SECTION 을 만나면 경로를 그 제목으로 새로 시작하고, SUBHEADING 은
          그 아래 한 단계로 덧붙인다(완벽한 계층 복원이 아니라 "맥락 연결"이 목표).
        - 개별 아이템 추출 실패는 그 아이템만 건너뛰고 경고를 남긴다(fail-soft).
        """
        nodes: list[DocumentNode] = []
        order = 0
        # 현재까지의 소제목 경로(루트→현재). 청커가 청크 metadata 로 전파한다.
        heading_path: tuple[str, ...] = ()

        try:
            items = document.iterate_items()
        except (DoclingError, ValueError, RuntimeError, AttributeError) as e:
            # 문서 순회 자체가 깨지는 드문 경우 — 빈 목록 + 경고.
            warnings.append(f"문서 아이템 순회 실패: {type(e).__name__}: {e}")
            return []

        for item, _level in items:
            try:
                made_node, heading_path = self._item_to_node(item, document, order, heading_path)
            except (DoclingError, ValueError, RuntimeError, AttributeError, IndexError) as e:
                # 한 아이템 추출 실패가 전체를 막지 않게 그 아이템만 건너뛴다.
                warnings.append(f"아이템 추출 실패 — 건너뜀 ({type(e).__name__}: {e})")
                continue

            if made_node is not None:
                nodes.append(made_node)
                order += 1

        return nodes

    def _item_to_node(
        self,
        item,
        document,
        order: int,
        heading_path: tuple[str, ...],
    ) -> tuple[DocumentNode | None, tuple[str, ...]]:
        """
        Docling 아이템 1개를 DocumentNode 로 변환하고, 갱신된 heading_path 를
        함께 돌려준다.

        분류 규칙(레이아웃 라벨 기반):
          - TitleItem / label==title          → SECTION  (heading_path 새로 시작)
          - SectionHeaderItem / label==section_header
              · level<=1 → SECTION (경로 새로 시작)
              · level>=2 → SUBHEADING (경로 한 단계 덧붙임)
          - ListItem / label==list_item        → LIST_ITEM
          - label==caption                      → FIGURE_CAPTION
          - TableItem                           → TABLE (행/열 보존)
          - PictureItem                         → 그림 캡션 → FIGURE_CAPTION
          - 그 외 TextItem(text/paragraph 등)   → PARAGRAPH

        반환:
          (노드 또는 None, 갱신된 heading_path)
          내용이 빈 아이템(빈 표/빈 텍스트/머리말꼬리말 등)은 None 을 돌려
          노드를 생략한다.
        """
        # 1) 표 — 행/열 구조를 보존한 TABLE 노드.
        if isinstance(item, TableItem):
            node = self._table_node(item, document, order, heading_path)
            return node, heading_path

        # 2) 그림 — 캡션 텍스트만 FIGURE_CAPTION 노드로 만든다(이미지 자체는 적재 대상 아님).
        if isinstance(item, PictureItem):
            node = self._picture_node(item, document, order, heading_path)
            return node, heading_path

        # 3) 텍스트 계열(제목/소제목/본문/리스트/캡션) — 공통적으로 .text 를 가진다.
        #    TextItem 및 그 하위(TitleItem/SectionHeaderItem/ListItem)가 여기 해당.
        if isinstance(item, TextItem):
            return self._text_node(item, order, heading_path)

        # 4) 그 외 타입(그룹/폼/키밸류 등)은 이번 범위 밖 — 노드 생략(과설계 금지).
        return None, heading_path

    def _text_node(
        self,
        item: TextItem,
        order: int,
        heading_path: tuple[str, ...],
    ) -> tuple[DocumentNode | None, tuple[str, ...]]:
        """
        텍스트 계열 아이템(제목/소제목/본문/리스트/캡션)을 노드로 변환한다.

        제목/소제목이면 heading_path 를 갱신해 함께 반환한다(이후 본문 노드들이
        이 경로를 물려받아 "어느 맥락의 텍스트인지"를 검색 시 알 수 있게 한다).
        """
        text = (item.text or "").strip()
        if not text:
            # 빈 텍스트(레이아웃상 빈 블록)는 노드를 만들지 않는다.
            return None, heading_path

        label = item.label
        page = self._page_of(item)

        # ── 제목(SECTION): heading_path 를 이 제목으로 새로 시작 ──
        if isinstance(item, TitleItem) or label == DocItemLabel.TITLE:
            new_path = (text,)
            return (
                DocumentNode(
                    element_type=ElementType.SECTION,
                    text=text,
                    heading_path=new_path,
                    page=page,
                    order=order,
                ),
                new_path,
            )

        # ── 섹션 헤더: level 로 SECTION/SUBHEADING 구분 ──
        if isinstance(item, SectionHeaderItem) or label == DocItemLabel.SECTION_HEADER:
            # SectionHeaderItem 만 level 을 가진다(기본 1). 안전하게 getattr.
            level = int(getattr(item, "level", 1) or 1)
            if level <= 1:
                # 최상위 섹션 — 경로를 새로 시작한다.
                new_path = (text,)
                element_type = ElementType.SECTION
            else:
                # 하위 소제목 — 기존 경로(있으면 첫 칸)에 한 단계 덧붙인다.
                # 완전한 계층 복원이 아니라 2단계(섹션>소제목) 맥락만 보존한다.
                root = heading_path[:1] if heading_path else ()
                new_path = (*root, text)
                element_type = ElementType.SUBHEADING
            return (
                DocumentNode(
                    element_type=element_type,
                    text=text,
                    heading_path=new_path,
                    page=page,
                    order=order,
                ),
                new_path,
            )

        # ── 리스트 항목 ──
        if isinstance(item, ListItem) or label == DocItemLabel.LIST_ITEM:
            element_type = ElementType.LIST_ITEM
        # ── 그림/표 캡션 텍스트가 독립 아이템으로 온 경우 ──
        elif label == DocItemLabel.CAPTION:
            element_type = ElementType.FIGURE_CAPTION
        # ── 머리말/꼬리말/페이지번호 등은 본문 노이즈라 생략 ──
        elif label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
            return None, heading_path
        # ── 그 외(text/paragraph/reference/footnote 등)는 본문 단락 ──
        else:
            element_type = ElementType.PARAGRAPH

        return (
            DocumentNode(
                element_type=element_type,
                text=text,
                heading_path=heading_path,
                page=page,
                order=order,
            ),
            heading_path,
        )

    def _table_node(
        self,
        table: TableItem,
        document,
        order: int,
        heading_path: tuple[str, ...],
    ) -> DocumentNode | None:
        """
        Docling TableItem 을 TABLE 노드로 변환한다(행/열 구조 보존).

        견고한 그리드 복원:
          table.data.table_cells 는 각 셀의 (start_row/col_offset_idx) 를 들고
          있다. num_rows × num_cols 빈 격자를 만든 뒤, 각 셀 텍스트를 시작
          오프셋 위치에 채운다(병합 셀은 시작 칸에만 둔다 — 과설계 금지).
          직렬화 형식은 pptx/pdf_plumber/hwpx 파서와 동일하게 맞춘다:
          각 행을 " | " 로 잇고, 행은 줄바꿈으로 연결한다.

        실패/빈 표는 None 반환(노드 생략, fail-soft).
        """
        data = getattr(table, "data", None)
        if data is None:
            return None

        num_rows = int(getattr(data, "num_rows", 0) or 0)
        num_cols = int(getattr(data, "num_cols", 0) or 0)
        cells = list(getattr(data, "table_cells", None) or [])
        if num_rows <= 0 or num_cols <= 0 or not cells:
            return None

        # 빈 격자 생성 후 셀 텍스트를 시작 오프셋 위치에 채운다.
        grid: list[list[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        for cell in cells:
            r = int(getattr(cell, "start_row_offset_idx", 0) or 0)
            c = int(getattr(cell, "start_col_offset_idx", 0) or 0)
            # 인덱스가 격자 범위를 벗어나면 그 셀만 건너뛴다(견고).
            if 0 <= r < num_rows and 0 <= c < num_cols:
                grid[r][c] = (getattr(cell, "text", "") or "").strip()

        rows_text = [" | ".join(row) for row in grid]
        content = "\n".join(rt for rt in rows_text if rt.strip(" |"))
        if not content.strip():
            return None

        return DocumentNode(
            element_type=ElementType.TABLE,
            text=content,
            heading_path=heading_path,
            page=self._page_of(table),
            order=order,
        )

    def _picture_node(
        self,
        picture: PictureItem,
        document,
        order: int,
        heading_path: tuple[str, ...],
    ) -> DocumentNode | None:
        """
        Docling PictureItem 의 캡션을 FIGURE_CAPTION 노드로 변환한다.

        이미지 픽셀 자체는 적재 대상이 아니므로(임베딩은 텍스트), 캡션 텍스트만
        추출한다. 캡션이 없거나 너무 짧으면 노드를 만들지 않는다(노이즈 방지).
        caption_text(doc) 는 그림에 연결된 캡션 아이템들의 텍스트를 합쳐준다.
        """
        try:
            caption = (picture.caption_text(document) or "").strip()
        except (DoclingError, ValueError, RuntimeError, AttributeError):
            # 캡션 접근 실패는 그림을 통째로 생략(상위에서 별도 경고 불필요).
            return None

        if len(caption) < _MIN_CAPTION_CHARS:
            return None

        return DocumentNode(
            element_type=ElementType.FIGURE_CAPTION,
            text=caption,
            heading_path=heading_path,
            page=self._page_of(picture),
            order=order,
        )

    @staticmethod
    def _page_of(item) -> int | None:
        """
        아이템의 출처 페이지 번호(1부터)를 prov 에서 꺼낸다.

        Docling 아이템은 prov(provenance) 목록에 출처 정보를 담고, 그 첫 항목의
        page_no 가 페이지 번호다. prov 가 비었거나 접근이 깨지면 None 을 돌려
        page 미상으로 둔다(노드 생성은 계속 — fail-soft).
        """
        try:
            prov = getattr(item, "prov", None) or []
            if prov:
                page_no = getattr(prov[0], "page_no", None)
                return int(page_no) if page_no is not None else None
        except (ValueError, TypeError, IndexError, AttributeError):
            return None
        return None

    @staticmethod
    def _empty_tree(path: Path, warnings: tuple[str, ...]) -> DocumentTree:
        """치명적 실패 시 돌려줄 빈 트리(파일명을 제목으로, 경고만 담는다)."""
        return DocumentTree(
            title=path.stem,
            source_path=str(path),
            nodes=(),
            doc_format="pdf",
            warnings=warnings,
        )
