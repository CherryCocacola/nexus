"""
core.ingest.parsers.pdf_plumber — PdfPlumberParser 단위 테스트 (v7.3 단계 5 — PDF 경량).

핵심 검증 의도:
  1. can_parse() — fail-closed: 확장자(.pdf) + 매직바이트("%PDF")까지 봐야 True.
  2. 표 추출 — extract_tables() 결과가 TABLE 노드로 변환되고 " | "/줄바꿈 보존.
  3. 페이지 번호 — 노드의 page 필드가 1부터의 페이지 번호.
  4. 읽기 순서 — (top, x0) 정렬로 라인 order 가 단조 증가.
  5. 헤딩 휴리스틱 — 본문 중앙값×1.25 이상 & 80자 이하 라인 → SUBHEADING.
  6. fail-soft — 손상/잘못된 PDF 의 동작(아래 주의 참고).

fixture 전략(reportlab 부재 우회):
  지시에는 reportlab 또는 user_mig/*.pdf 사용이 허용되나, reportlab 은 이 환경에
  설치되어 있지 않다. 대신 항상 설치되어 있는 matplotlib 의 PdfPages 백엔드로
  tmp_path 에 텍스트/표가 든 PDF 를 런타임 생성한다(바이너리 미커밋, 운영 무관).
  글자 크기를 크게/작게 조절해 헤딩 휴리스틱까지 결정론적으로 검증할 수 있다.

손상 PDF 처리(fail-soft):
  pdfplumber.open() 은 내부 pdfminer 오류를 PdfminerException(Exception 직속,
  PDFSyntaxError/ValueError 아님)으로 감싸 올린다. 파서가 이를 명시적으로
  포착하므로 '헤더는 %PDF 지만 본문이 깨진' PDF 도 예외 전파 없이
  fail-soft(빈 트리 + warning)로 환원된다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

from core.ingest.parsers.pdf_plumber import PdfPlumberParser
from core.ingest.types import ElementType

matplotlib.use("Agg")  # 화면 없는(headless) 백엔드 — 파일 렌더만.
import matplotlib.pyplot as plt  # noqa: E402 — use() 이후 import 해야 백엔드 고정
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


# ─────────────────────────────────────────────
# fixture 생성 헬퍼 (matplotlib PdfPages)
# ─────────────────────────────────────────────
def _make_text_pdf(tmp_path: Path) -> Path:
    """큰 제목 1줄 + 작은 본문 2줄을 가진 PDF 를 만든다(헤딩 휴리스틱 검증용).

    제목 폰트(28)가 본문 폰트(11)의 2배 이상이라, 본문 중앙값×1.25 기준을
    확실히 넘겨 SUBHEADING 으로 분류되어야 한다. 본문 2줄은 PARAGRAPH."""
    path = tmp_path / "text.pdf"
    with PdfPages(str(path)) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 세로
        fig.text(0.1, 0.90, "Security Policy", fontsize=28)
        fig.text(0.1, 0.80, "Access control follows least privilege.", fontsize=11)
        fig.text(0.1, 0.74, "Second body line for paragraph test.", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)
    return path


def _make_table_pdf(tmp_path: Path) -> Path:
    """2x2 표 1개를 가진 PDF 를 만든다(표 추출 검증용)."""
    path = tmp_path / "table.pdf"
    with PdfPages(str(path)) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.table(
            cellText=[["Name", "Dept"], ["Hong", "Security"]],
            loc="center",
        )
        pdf.savefig(fig)
        plt.close(fig)
    return path


def _make_two_page_pdf(tmp_path: Path) -> Path:
    """2페이지 PDF 를 만든다(페이지 번호/순서 검증용)."""
    path = tmp_path / "two_page.pdf"
    with PdfPages(str(path)) as pdf:
        for label in ("First page body.", "Second page body."):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.1, 0.85, label, fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
    return path


# ─────────────────────────────────────────────
# can_parse — fail-closed
# ─────────────────────────────────────────────
def test_can_parse_valid_pdf_returns_true(tmp_path: Path) -> None:
    """정상 .pdf("%PDF" 시그니처 포함)는 True."""
    path = _make_text_pdf(tmp_path)
    assert PdfPlumberParser().can_parse(path) is True


def test_can_parse_wrong_extension_returns_false(tmp_path: Path) -> None:
    """확장자가 .pdf 가 아니면 내용과 무관하게 False(확장자 우선 거부)."""
    valid = _make_text_pdf(tmp_path)
    renamed = tmp_path / "doc.txt"
    renamed.write_bytes(valid.read_bytes())
    assert PdfPlumberParser().can_parse(renamed) is False


def test_can_parse_non_pdf_magic_returns_false(tmp_path: Path) -> None:
    """확장자는 .pdf 지만 매직바이트가 '%PDF' 가 아니면 False(fail-closed)."""
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"NOT-A-PDF-AT-ALL")
    assert PdfPlumberParser().can_parse(fake) is False


def test_can_parse_missing_file_returns_false(tmp_path: Path) -> None:
    """파일이 없으면 OSError 를 삼키고 False."""
    assert PdfPlumberParser().can_parse(tmp_path / "nope.pdf") is False


def test_supported_extensions_and_no_gpu() -> None:
    """정체성 플래그 — .pdf 만 지원, GPU 불필요."""
    parser = PdfPlumberParser()
    assert parser.supported_extensions == (".pdf",)
    assert parser.requires_gpu is False


# ─────────────────────────────────────────────
# 텍스트 + 헤딩 휴리스틱 + 읽기 순서
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_text_classifies_heading_and_paragraph(tmp_path: Path) -> None:
    """큰 글자 라인은 SUBHEADING, 작은 본문 라인은 PARAGRAPH 로 분류돼야 한다."""
    path = _make_text_pdf(tmp_path)
    tree = await PdfPlumberParser().parse(path)

    assert tree.doc_format == "pdf"
    assert tree.title == "text"  # 파일명(stem)

    texts = {n.text: n.element_type for n in tree.nodes}
    # 제목(큰 글자)은 SUBHEADING.
    assert texts.get("Security Policy") == ElementType.SUBHEADING
    # 본문(작은 글자)은 PARAGRAPH.
    body = [n for n in tree.nodes if "least privilege" in n.text]
    assert body and body[0].element_type == ElementType.PARAGRAPH


@pytest.mark.asyncio
async def test_parse_reading_order_is_monotonic(tmp_path: Path) -> None:
    """라인 노드의 order 가 (top, x0) 읽기 순서대로 단조 증가해야 한다."""
    path = _make_text_pdf(tmp_path)
    tree = await PdfPlumberParser().parse(path)

    orders = [n.order for n in tree.nodes]
    assert orders == sorted(orders)
    # 제목이 본문보다 위(top 작음)에 있으므로 먼저 와야 한다.
    texts_in_order = [n.text for n in tree.nodes]
    title_idx = texts_in_order.index("Security Policy")
    body_idx = next(i for i, t in enumerate(texts_in_order) if "least privilege" in t)
    assert title_idx < body_idx


@pytest.mark.asyncio
async def test_parse_all_nodes_have_page_number(tmp_path: Path) -> None:
    """모든 노드의 page 필드가 1 이상의 페이지 번호여야 한다."""
    path = _make_text_pdf(tmp_path)
    tree = await PdfPlumberParser().parse(path)
    assert tree.nodes
    assert all(n.page == 1 for n in tree.nodes)


@pytest.mark.asyncio
async def test_parse_two_pages_have_distinct_page_numbers(tmp_path: Path) -> None:
    """2페이지 PDF 는 각 페이지의 노드가 page=1, page=2 로 구분돼야 한다."""
    path = _make_two_page_pdf(tmp_path)
    tree = await PdfPlumberParser().parse(path)

    pages = {n.page for n in tree.nodes}
    assert pages == {1, 2}
    # order 는 페이지를 넘어가도 계속 증가한다(문서 전체 읽기 순서).
    orders = [n.order for n in tree.nodes]
    assert orders == sorted(orders)


# ─────────────────────────────────────────────
# 표 추출 — TABLE 노드 + 행/열 보존
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_table_preserves_rows_and_columns(tmp_path: Path) -> None:
    """표가 TABLE 노드로 변환되고 셀 텍스트가 ' | '/줄바꿈으로 보존되는지."""
    path = _make_table_pdf(tmp_path)
    tree = await PdfPlumberParser().parse(path)

    table_nodes = [n for n in tree.nodes if n.element_type == ElementType.TABLE]
    assert len(table_nodes) >= 1
    content = table_nodes[0].text
    # 열은 " | ", 행은 줄바꿈.
    assert "Name | Dept" in content
    assert "Hong | Security" in content
    assert "\n" in content


# ─────────────────────────────────────────────
# fail-soft / 손상 PDF
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_missing_file_raises_on_open(tmp_path: Path) -> None:
    """존재하지 않는 파일은 pdfplumber.open 단계에서 OSError → 파서가 fail-soft 처리.

    OSError 는 parse() 의 except 절에 포함되므로, 빈 트리 + warning 으로 환원된다
    (예외 비전파)."""
    missing = tmp_path / "nope.pdf"
    tree = await PdfPlumberParser().parse(missing)
    assert tree.nodes == ()
    assert len(tree.warnings) >= 1
    assert "PDF 로드 실패" in tree.warnings[0]


@pytest.mark.asyncio
async def test_parse_corrupted_pdf_returns_warning_not_raise(tmp_path: Path) -> None:
    """'헤더는 %PDF 지만 본문이 깨진' PDF 는 fail-soft(빈 트리 + warning) 처리된다.

    pdfplumber.open 은 내부 pdfminer 오류를 PdfminerException(Exception 직속,
    ValueError/PDFSyntaxError 아님)으로 감싸 올린다. 파서가 이를 명시적으로
    포착하므로 예외가 전파되지 않고 빈 트리 + 경고로 환원된다(fail-soft 계약)."""
    broken = tmp_path / "broken.pdf"
    # %PDF 시그니처는 있으나 /Root 객체가 없는 잘못된 PDF 본문.
    broken.write_bytes(b"%PDF-1.4\nthis is not a valid pdf body at all\n")

    # can_parse 는 매직바이트만 보므로 True 다(여기서 걸러지지 않음).
    assert PdfPlumberParser().can_parse(broken) is True

    # 예외 전파 없이 빈 트리 + warning 으로 환원된다.
    tree = await PdfPlumberParser().parse(broken)
    assert tree.nodes == ()
    assert len(tree.warnings) >= 1
