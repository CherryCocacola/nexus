"""
core/ingest/parsers/docling_layout.py — DoclingParser 단위 테스트.

검증 대상 (후속 구현 C):
  - 정체성/플래그: requires_gpu=True, supported_extensions=(".pdf",).
  - can_parse: .pdf 확장자 + "%PDF" 매직바이트 둘 다 만족해야 True(fail-closed).
  - parse: DocumentConverter.convert 를 mock 해, iterate_items 가 내보내는
    Docling 아이템(Title/SectionHeader level1·2/ListItem/TextItem/TableItem/
    PictureItem)을 SECTION/SUBHEADING/LIST_ITEM/PARAGRAPH/TABLE/FIGURE_CAPTION
    으로 분류하는지, heading_path 누적, page 매핑, 표 격자 복원(" | ")까지.
  - fail-soft: convert 가 docling BaseError / OSError 를 던지면 빈 트리 + warning
    (예외 비전파). 개별 아이템 예외는 그 아이템만 skip + warning.

docling 모델 다운로드/추론 우회 (핵심):
  DoclingParser.parse 는 self._ensure_converter() 로 DocumentConverter 를 만든 뒤
  converter.convert(path) 로 무거운 레이아웃 모델 추론을 돌린다. 테스트는
  DoclingParser._ensure_converter 를 가짜 변환기로 monkeypatch 하여 실제
  DocumentConverter 생성(모델 로딩/다운로드)과 convert() 추론을 전부 건너뛴다.
  변환 결과(ConversionResult)는 .document.iterate_items() 만 흉내 내면 충분하므로
  SimpleNamespace + 실제 docling_core 아이템 인스턴스로 구성한다(에어갭/오프라인).

  Docling 아이템 클래스는 실제 docling_core 타입을 그대로 만들어 isinstance
  분기를 정확히 태운다(가짜 클래스로는 isinstance(item, TableItem) 등이 깨진다).
  page 출처(prov)는 실제 ProvenanceItem 으로, 그림 캡션은 PictureItem.caption_text
  를 monkeypatch 하여 주입한다.
"""

from __future__ import annotations

from types import SimpleNamespace

# docling 실제 아이템/예외 타입 — isinstance 분기를 정확히 태우기 위해 실 타입을 쓴다.
from docling.exceptions import ConversionError
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.document import (
    ListItem,
    PictureItem,
    ProvenanceItem,
    SectionHeaderItem,
    TableCell,
    TableData,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.labels import DocItemLabel

from core.ingest.parsers.docling_layout import DoclingParser
from core.ingest.types import ElementType


# ─────────────────────────────────────────────
# Docling 아이템/변환결과 빌더 — 실제 타입으로 구성
# ─────────────────────────────────────────────
def _prov(page: int) -> ProvenanceItem:
    """page_no 만 의미 있는 최소 ProvenanceItem(출처). bbox/charspan 은 더미."""
    bbox = BoundingBox(l=0.0, t=0.0, r=1.0, b=1.0, coord_origin=CoordOrigin.TOPLEFT)
    return ProvenanceItem(page_no=page, bbox=bbox, charspan=(0, 1))


def _title(text: str, page: int | None = None, ref: str = "#/texts/0") -> TitleItem:
    """문서 제목 아이템(TitleItem) — SECTION 으로 분류된다."""
    item = TitleItem(self_ref=ref, orig=text, text=text)
    if page is not None:
        item.prov.append(_prov(page))
    return item


def _section_header(
    text: str, level: int, page: int | None = None, ref: str = "#/texts/0"
) -> SectionHeaderItem:
    """섹션 헤더(level<=1→SECTION, level>=2→SUBHEADING)."""
    item = SectionHeaderItem(self_ref=ref, orig=text, text=text, level=level)
    if page is not None:
        item.prov.append(_prov(page))
    return item


def _list_item(text: str, page: int | None = None, ref: str = "#/texts/0") -> ListItem:
    """목록 항목(ListItem) — LIST_ITEM 으로 분류된다."""
    item = ListItem(self_ref=ref, orig=text, text=text)
    if page is not None:
        item.prov.append(_prov(page))
    return item


def _text_item(
    text: str,
    label: DocItemLabel = DocItemLabel.TEXT,
    page: int | None = None,
    ref: str = "#/texts/0",
) -> TextItem:
    """본문/캡션/머리꼬리말 등 일반 TextItem. label 로 세분 분류된다."""
    item = TextItem(self_ref=ref, label=label, orig=text, text=text)
    if page is not None:
        item.prov.append(_prov(page))
    return item


def _table(rows: list[list[str]], page: int | None = None, ref: str = "#/tables/0") -> TableItem:
    """행/열 텍스트로 TableItem 을 만든다(시작 오프셋에 셀 텍스트 배치)."""
    num_rows = len(rows)
    num_cols = max((len(r) for r in rows), default=0)
    cells: list[TableCell] = []
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cells.append(
                TableCell(
                    text=val,
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + 1,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + 1,
                )
            )
    data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=cells)
    item = TableItem(self_ref=ref, data=data)
    if page is not None:
        item.prov.append(_prov(page))
    return item


def _picture(page: int | None = None, ref: str = "#/pictures/0") -> PictureItem:
    """PictureItem(캡션은 caption_text monkeypatch 로 주입)."""
    item = PictureItem(self_ref=ref)
    if page is not None:
        item.prov.append(_prov(page))
    return item


class _FakeDocument:
    """iterate_items() 만 흉내 내는 가짜 DoclingDocument."""

    def __init__(self, items: list) -> None:
        # parse 는 (item, level) 튜플을 기대한다. level 은 분류에 쓰지 않으므로 0 고정.
        self._items = [(it, 0) for it in items]

    def iterate_items(self):
        """등록한 아이템들을 (item, level) 로 읽기 순서대로 yield 한다."""
        yield from self._items


def _install_fake_converter(monkeypatch, items: list | None = None, *, raise_exc=None):
    """
    DoclingParser._ensure_converter 를 가짜 변환기로 교체한다.

    - items: convert 결과 문서가 iterate_items 로 내보낼 아이템 목록.
    - raise_exc: 주면 convert() 가 그 예외를 던진다(fail-soft 검증용).

    이로써 실제 DocumentConverter 생성(모델 로딩)과 convert() 추론을 모두 건너뛴다.
    """

    class _FakeConverter:
        def convert(self, source):  # noqa: D401 — Docling convert 흉내
            if raise_exc is not None:
                raise raise_exc
            # 실제 convert 는 ConversionResult(.document) 를 돌려준다.
            return SimpleNamespace(document=_FakeDocument(items or []))

    monkeypatch.setattr(DoclingParser, "_ensure_converter", lambda self: _FakeConverter())


# ─────────────────────────────────────────────
# 정체성/플래그
# ─────────────────────────────────────────────
class TestDoclingParserIdentity:
    """파서의 정체성 플래그를 검증한다."""

    def test_requires_gpu_is_true(self):
        """레이아웃 모델 추론 — requires_gpu=True 여야 한다(티어 분기 근거)."""
        assert DoclingParser().requires_gpu is True

    def test_supported_extensions_is_pdf_only(self):
        """.pdf 확장자만 지원한다(pdfplumber 와 같은 확장자, priority 로 우선순위 가름)."""
        assert DoclingParser().supported_extensions == (".pdf",)


# ─────────────────────────────────────────────
# can_parse — 확장자 + 매직바이트(fail-closed)
# ─────────────────────────────────────────────
class TestDoclingCanParse:
    """can_parse 의 fail-closed 동작을 검증한다."""

    def test_can_parse_real_pdf_magic_returns_true(self, tmp_path):
        """.pdf 확장자 + '%PDF' 매직바이트면 True."""
        p = tmp_path / "doc.pdf"
        p.write_bytes(b"%PDF-1.7\n...rest...")
        assert DoclingParser().can_parse(p) is True

    def test_can_parse_wrong_extension_returns_false(self, tmp_path):
        """.txt 등 비-pdf 확장자는 매직바이트와 무관하게 False."""
        p = tmp_path / "doc.txt"
        p.write_bytes(b"%PDF-1.7")
        assert DoclingParser().can_parse(p) is False

    def test_can_parse_pdf_ext_wrong_magic_returns_false(self, tmp_path):
        """확장자는 .pdf 이지만 매직바이트가 다르면 False(가짜 pdf 거부, fail-closed)."""
        p = tmp_path / "fake.pdf"
        p.write_bytes(b"NOT-A-PDF")
        assert DoclingParser().can_parse(p) is False

    def test_can_parse_missing_file_returns_false(self, tmp_path):
        """파일이 없어 열 수 없으면 False(OSError → fail-closed)."""
        p = tmp_path / "missing.pdf"
        assert DoclingParser().can_parse(p) is False


# ─────────────────────────────────────────────
# parse — 아이템 분류 + heading_path + page
# ─────────────────────────────────────────────
class TestDoclingParseClassification:
    """convert mock 으로 각 아이템 타입의 노드 분류를 검증한다."""

    async def test_parse_title_classified_as_section(self, tmp_path, monkeypatch):
        """TitleItem → SECTION, heading_path 는 그 제목으로 시작한다."""
        _install_fake_converter(monkeypatch, [_title("문서 제목", page=1)])
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.7")

        tree = await DoclingParser().parse(path)

        assert len(tree.nodes) == 1
        node = tree.nodes[0]
        assert node.element_type == ElementType.SECTION
        assert node.text == "문서 제목"
        assert node.heading_path == ("문서 제목",)
        assert node.page == 1
        assert tree.doc_format == "pdf"

    async def test_parse_section_header_level1_is_section(self, tmp_path, monkeypatch):
        """SectionHeaderItem(level=1) → SECTION (경로 새로 시작)."""
        _install_fake_converter(monkeypatch, [_section_header("1장", level=1, page=2)])
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        node = tree.nodes[0]
        assert node.element_type == ElementType.SECTION
        assert node.heading_path == ("1장",)
        assert node.page == 2

    async def test_parse_section_header_level2_is_subheading(self, tmp_path, monkeypatch):
        """
        level1 섹션 다음의 level2 헤더 → SUBHEADING.
        heading_path 는 (섹션, 소제목) 2단계로 누적된다.
        """
        items = [
            _section_header("1장 보안", level=1, ref="#/texts/0"),
            _section_header("1.1 권한", level=2, ref="#/texts/1"),
        ]
        _install_fake_converter(monkeypatch, items)
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        assert tree.nodes[0].element_type == ElementType.SECTION
        sub = tree.nodes[1]
        assert sub.element_type == ElementType.SUBHEADING
        # 섹션 경로 위에 소제목을 한 단계 덧붙인다.
        assert sub.heading_path == ("1장 보안", "1.1 권한")

    async def test_parse_text_item_is_paragraph_inherits_heading_path(self, tmp_path, monkeypatch):
        """일반 TextItem(label=text) → PARAGRAPH, 앞선 제목의 heading_path 를 물려받는다."""
        items = [
            _title("제목A", ref="#/texts/0"),
            _text_item("본문 단락입니다.", label=DocItemLabel.TEXT, ref="#/texts/1"),
        ]
        _install_fake_converter(monkeypatch, items)
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        para = tree.nodes[1]
        assert para.element_type == ElementType.PARAGRAPH
        assert para.text == "본문 단락입니다."
        # 제목 노드의 heading_path 를 물려받는다.
        assert para.heading_path == ("제목A",)

    async def test_parse_list_item_classified_as_list_item(self, tmp_path, monkeypatch):
        """ListItem → LIST_ITEM."""
        _install_fake_converter(monkeypatch, [_list_item("첫째 항목")])
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        assert tree.nodes[0].element_type == ElementType.LIST_ITEM
        assert tree.nodes[0].text == "첫째 항목"

    async def test_parse_caption_text_item_is_figure_caption(self, tmp_path, monkeypatch):
        """label=caption 인 TextItem → FIGURE_CAPTION."""
        _install_fake_converter(
            monkeypatch, [_text_item("그림 1. 구조도", label=DocItemLabel.CAPTION)]
        )
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        assert tree.nodes[0].element_type == ElementType.FIGURE_CAPTION

    async def test_parse_page_header_footer_skipped(self, tmp_path, monkeypatch):
        """머리말/꼬리말 라벨은 본문 노이즈라 노드를 만들지 않는다."""
        items = [
            _text_item("머리말", label=DocItemLabel.PAGE_HEADER, ref="#/texts/0"),
            _text_item("실제 본문", label=DocItemLabel.TEXT, ref="#/texts/1"),
            _text_item("꼬리말", label=DocItemLabel.PAGE_FOOTER, ref="#/texts/2"),
        ]
        _install_fake_converter(monkeypatch, items)
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        # 본문 1개만 남는다.
        assert len(tree.nodes) == 1
        assert tree.nodes[0].text == "실제 본문"

    async def test_parse_empty_text_item_produces_no_node(self, tmp_path, monkeypatch):
        """빈 텍스트 아이템은 노드를 만들지 않는다(노이즈 방지)."""
        _install_fake_converter(monkeypatch, [_text_item("   ", label=DocItemLabel.TEXT)])
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        assert tree.nodes == ()


# ─────────────────────────────────────────────
# parse — 표 격자 복원
# ─────────────────────────────────────────────
class TestDoclingParseTable:
    """TableItem 의 행/열 격자 복원(" | " 직렬화)을 검증한다."""

    async def test_parse_table_grid_serialized_with_pipe(self, tmp_path, monkeypatch):
        """표는 각 행을 ' | ' 로 잇고 행을 줄바꿈으로 연결한 TABLE 노드가 된다."""
        table = _table([["Name", "Dept"], ["Kim", "R&D"]], page=3)
        _install_fake_converter(monkeypatch, [table])
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        node = tree.nodes[0]
        assert node.element_type == ElementType.TABLE
        assert node.text == "Name | Dept\nKim | R&D"
        assert node.page == 3

    async def test_parse_empty_table_produces_no_node(self, tmp_path, monkeypatch):
        """행/열이 0이거나 셀이 없는 표는 노드를 만들지 않는다(fail-soft)."""
        empty = TableItem(
            self_ref="#/tables/0", data=TableData(num_rows=0, num_cols=0, table_cells=[])
        )
        _install_fake_converter(monkeypatch, [empty])
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        assert tree.nodes == ()


# ─────────────────────────────────────────────
# parse — 그림 캡션
# ─────────────────────────────────────────────
class TestDoclingParsePicture:
    """PictureItem 의 캡션 추출(FIGURE_CAPTION)을 검증한다."""

    async def test_parse_picture_caption_becomes_figure_caption(self, tmp_path, monkeypatch):
        """그림의 caption_text 가 FIGURE_CAPTION 노드로 추출된다."""
        # PictureItem.caption_text(doc) 를 monkeypatch 로 고정 캡션 반환하게 한다.
        monkeypatch.setattr(
            PictureItem, "caption_text", lambda self, doc: "그림 2. 흐름도", raising=True
        )
        _install_fake_converter(monkeypatch, [_picture(page=4)])
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        node = tree.nodes[0]
        assert node.element_type == ElementType.FIGURE_CAPTION
        assert node.text == "그림 2. 흐름도"
        assert node.page == 4

    async def test_parse_picture_short_caption_skipped(self, tmp_path, monkeypatch):
        """캡션이 너무 짧으면(1글자) 노드를 만들지 않는다(노이즈 방지)."""
        monkeypatch.setattr(PictureItem, "caption_text", lambda self, doc: "x", raising=True)
        _install_fake_converter(monkeypatch, [_picture(page=4)])
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        assert tree.nodes == ()


# ─────────────────────────────────────────────
# fail-soft — convert/아이템 예외
# ─────────────────────────────────────────────
class TestDoclingFailSoft:
    """변환/아이템 추출 실패를 빈 트리/skip + warning 으로 흡수하는지 검증한다."""

    async def test_parse_convert_docling_error_returns_empty_tree_with_warning(
        self, tmp_path, monkeypatch
    ):
        """convert 가 docling BaseError 를 던지면 빈 트리 + 경고(예외 비전파)."""
        _install_fake_converter(monkeypatch, raise_exc=ConversionError("손상된 PDF"))
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF")

        # 예외가 전파되지 않아야 한다.
        tree = await DoclingParser().parse(path)

        assert tree.nodes == ()
        assert tree.warnings  # 경고가 1개 이상 채워진다.
        assert any("변환 실패" in w for w in tree.warnings)

    async def test_parse_convert_os_error_returns_empty_tree_with_warning(
        self, tmp_path, monkeypatch
    ):
        """convert 가 OSError 를 던져도 빈 트리 + 경고(예외 비전파)."""
        _install_fake_converter(monkeypatch, raise_exc=OSError("디스크 오류"))
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        assert tree.nodes == ()
        assert any("변환 실패" in w for w in tree.warnings)

    async def test_parse_single_bad_item_skipped_others_kept(self, tmp_path, monkeypatch):
        """
        개별 아이템 추출이 예외를 던지면 그 아이템만 건너뛰고(경고),
        나머지 정상 아이템은 노드로 보존된다(fail-soft).
        """

        # 정상 TextItem 을 만들되, prov 를 "참조(bool 평가) 시 RuntimeError 를 던지는"
        # 객체로 교체한다. _page_of 는 prov 를 `prov or []` 로 평가하므로 __bool__ 에서
        # 예외가 나는데, _page_of 가 잡는 예외 집합(ValueError/TypeError/IndexError/
        # AttributeError)에는 RuntimeError 가 없어 예외가 _item_to_node 까지 전파된다.
        # _nodes_from_document 의 except(RuntimeError 포함)가 이 아이템만 건너뛰고
        # 경고를 남기는지 검증한다(개별 아이템 fail-soft).
        class _BoomProv:
            def __bool__(self):
                raise RuntimeError("prov 접근 손상")

        bad = _text_item("불량 아이템", label=DocItemLabel.TEXT, ref="#/texts/1")
        bad.prov = _BoomProv()

        items = [
            _text_item("정상 본문 1", label=DocItemLabel.TEXT, ref="#/texts/0"),
            bad,  # .text 접근 시 ValueError → 그 아이템만 skip + 경고
            _text_item("정상 본문 2", label=DocItemLabel.TEXT, ref="#/texts/2"),
        ]
        _install_fake_converter(monkeypatch, items)
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")

        tree = await DoclingParser().parse(path)

        # 정상 본문 2개는 보존되고, 불량 아이템은 노드를 만들지 않는다.
        texts = [n.text for n in tree.nodes]
        assert texts == ["정상 본문 1", "정상 본문 2"]
        # 건너뛴 아이템에 대한 경고가 남는다.
        assert any("건너뜀" in w for w in tree.warnings)
