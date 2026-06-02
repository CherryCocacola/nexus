"""
core.ingest.parsers.pptx — PptxParser 단위 테스트 (v7.3 문서 인제스트).

핵심 검증 의도:
  1. can_parse() — fail-closed: 확장자(.pptx) + 매직바이트(ZIP)까지 봐야 True.
  2. 읽기 순서 복원 — 좌(left 작음)/우(left 큼) 텍스트박스를 일부러 역순으로
     추가해도 parse 결과 order 가 (top,left) 정렬대로(좌→우) 나와야 한다.
     이것이 "충돌 없는 임베딩"의 전제(읽기 순서 보존).
  3. 표(TABLE) — 행/열 텍스트가 " | " 형태로 보존되는지.
  4. 슬라이드 제목 → heading_path / 문서 title 반영.
  5. fail-soft — 빈 슬라이드/제목 없는 슬라이드에도 예외 대신 부분 트리 반환.

fixture 전략:
  바이너리 .pptx 를 커밋하지 않기 위해, python-pptx 로 tmp_path 에 런타임 생성한다.
  (python-pptx 설치 확인됨 — pptx 1.0.2.)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pptx import Presentation
from pptx.util import Emu

from core.ingest.parsers.pptx import PptxParser
from core.ingest.types import DocumentNode, ElementType


# ─────────────────────────────────────────────
# 헬퍼 — tmp_path 에 .pptx 를 코드로 생성한다.
# ─────────────────────────────────────────────
def _flatten_leaf_nodes(nodes: tuple[DocumentNode, ...]) -> list[DocumentNode]:
    """트리에서 텍스트를 가진 리프 노드만 읽기 순서대로 평탄화한다(검증 보조)."""
    out: list[DocumentNode] = []
    for n in nodes:
        if n.children:
            out.extend(_flatten_leaf_nodes(n.children))
        elif n.text.strip():
            out.append(n)
    return out


def _add_textbox(slide, *, left_emu: int, top_emu: int, text: str) -> None:
    """좌표(EMU)를 지정해 텍스트박스를 추가한다 — 읽기 순서 테스트용."""
    box = slide.shapes.add_textbox(Emu(left_emu), Emu(top_emu), Emu(2_000_000), Emu(500_000))
    box.text_frame.text = text


def _make_reading_order_pptx(tmp_path: Path) -> Path:
    """
    한 슬라이드에 좌/우 텍스트박스를 '우 먼저, 좌 나중' 역순으로 추가한다.

    같은 top, 다른 left 이므로 파서가 (top,left) 정렬을 하면 좌→우 순서로
    복원되어야 한다(추가 순서 ≠ 읽기 순서임을 일부러 만든다).
    """
    prs = Presentation()
    # 제목 placeholder 가 없는 '빈' 레이아웃을 골라 좌표 정렬만 검증한다.
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    same_top = 1_000_000
    # 일부러 오른쪽(left 큼)을 먼저 추가한다.
    _add_textbox(slide, left_emu=5_000_000, top_emu=same_top, text="오른쪽")
    _add_textbox(slide, left_emu=500_000, top_emu=same_top, text="왼쪽")
    path = tmp_path / "reading_order.pptx"
    prs.save(str(path))
    return path


def _make_titled_pptx(tmp_path: Path) -> Path:
    """제목 placeholder 가 있는 슬라이드 + 본문 1개를 만든다."""
    prs = Presentation()
    # 레이아웃 1 = Title and Content.
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "보안 정책"
    # 본문 placeholder 에 문단 추가.
    body = slide.placeholders[1]
    body.text_frame.text = "접근 통제는 최소 권한 원칙을 따른다."
    path = tmp_path / "titled.pptx"
    prs.save(str(path))
    return path


def _make_table_pptx(tmp_path: Path) -> Path:
    """표(2행 2열) 1개를 가진 슬라이드를 만든다."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rows, cols = 2, 2
    table_shape = slide.shapes.add_table(
        rows, cols, Emu(500_000), Emu(500_000), Emu(4_000_000), Emu(1_000_000)
    )
    table = table_shape.table
    table.cell(0, 0).text = "이름"
    table.cell(0, 1).text = "부서"
    table.cell(1, 0).text = "홍길동"
    table.cell(1, 1).text = "보안팀"
    path = tmp_path / "table.pptx"
    prs.save(str(path))
    return path


def _make_empty_slide_pptx(tmp_path: Path) -> Path:
    """텍스트가 전혀 없는 빈 슬라이드만 가진 파일(fail-soft 검증용)."""
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])  # blank
    path = tmp_path / "empty.pptx"
    prs.save(str(path))
    return path


# ─────────────────────────────────────────────
# can_parse — fail-closed
# ─────────────────────────────────────────────
def test_can_parse_valid_pptx_returns_true(tmp_path: Path) -> None:
    """정상 .pptx(ZIP 시그니처 포함)는 True."""
    path = _make_empty_slide_pptx(tmp_path)
    assert PptxParser().can_parse(path) is True


def test_can_parse_wrong_extension_returns_false(tmp_path: Path) -> None:
    """확장자가 .pptx 가 아니면 내용과 무관하게 False(확장자 우선 거부)."""
    valid = _make_empty_slide_pptx(tmp_path)
    renamed = tmp_path / "doc.txt"
    renamed.write_bytes(valid.read_bytes())  # 내용은 ZIP 이지만 확장자가 다름
    assert PptxParser().can_parse(renamed) is False


def test_can_parse_non_zip_magic_returns_false(tmp_path: Path) -> None:
    """확장자는 .pptx 지만 매직바이트가 ZIP 이 아니면 False(fail-closed)."""
    fake = tmp_path / "fake.pptx"
    fake.write_bytes(b"NOT_A_ZIP_FILE_AT_ALL")
    assert PptxParser().can_parse(fake) is False


def test_can_parse_missing_file_returns_false(tmp_path: Path) -> None:
    """파일이 없으면 OSError 를 삼키고 False(처리 불가로 간주)."""
    missing = tmp_path / "nope.pptx"
    assert PptxParser().can_parse(missing) is False


def test_supported_extensions_and_no_gpu() -> None:
    """정체성 플래그 — .pptx 만 지원, GPU 불필요."""
    parser = PptxParser()
    assert parser.supported_extensions == (".pptx",)
    assert parser.requires_gpu is False


# ─────────────────────────────────────────────
# 읽기 순서 복원 (충돌 방지의 전제)
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_reading_order_left_to_right_by_coords(tmp_path: Path) -> None:
    """
    '오른쪽'을 먼저 추가했어도, (top,left) 정렬 결과 '왼쪽'이 먼저 와야 한다.

    이는 좌측/우측 텍스트박스가 추가 순서대로 뒤섞이지 않고 읽기 순서로
    복원됨을 보장한다(v7.3 Part 3.3).
    """
    path = _make_reading_order_pptx(tmp_path)
    tree = await PptxParser().parse(path)

    leaves = _flatten_leaf_nodes(tree.nodes)
    texts = [n.text for n in leaves]
    assert texts == ["왼쪽", "오른쪽"]
    # order 도 읽기 순서(0,1)대로 부여되어야 한다.
    assert leaves[0].order == 0
    assert leaves[1].order == 1


# ─────────────────────────────────────────────
# 표 — TABLE 노드 + 행/열 보존
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_table_preserves_rows_and_columns(tmp_path: Path) -> None:
    """표 도형이 TABLE 노드로 변환되고, 셀 텍스트가 ' | '/줄바꿈으로 보존되는지."""
    path = _make_table_pptx(tmp_path)
    tree = await PptxParser().parse(path)

    leaves = _flatten_leaf_nodes(tree.nodes)
    table_nodes = [n for n in leaves if n.element_type == ElementType.TABLE]
    assert len(table_nodes) == 1

    content = table_nodes[0].text
    # 행은 줄바꿈, 열은 " | " 로 구분된다.
    assert "이름 | 부서" in content
    assert "홍길동 | 보안팀" in content
    # 행 구분(줄바꿈)이 들어가 행/열 구조가 보존됐는지.
    assert "\n" in content


# ─────────────────────────────────────────────
# 제목 → heading_path / 문서 title
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_slide_title_becomes_heading_path(tmp_path: Path) -> None:
    """슬라이드 제목이 본문 노드의 heading_path 로 전파되는지(문맥 연결)."""
    path = _make_titled_pptx(tmp_path)
    tree = await PptxParser().parse(path)

    # 문서 제목은 첫 슬라이드 제목에서 나온다.
    assert tree.title == "보안 정책"
    assert tree.doc_format == "pptx"

    leaves = _flatten_leaf_nodes(tree.nodes)
    body_nodes = [n for n in leaves if "최소 권한" in n.text]
    assert len(body_nodes) == 1
    # 본문 노드의 heading_path 에 슬라이드 제목이 들어 있어야 한다.
    assert body_nodes[0].heading_path == ("보안 정책",)


@pytest.mark.asyncio
async def test_parse_title_not_duplicated_in_body(tmp_path: Path) -> None:
    """
    제목 placeholder 텍스트가 본문 노드로 중복 생성되지 않아야 한다.

    제목은 heading_path 로 이미 반영되므로 본문 리프에 같은 텍스트가 다시
    나오면 중복이다(skip_title 동작 검증).
    """
    path = _make_titled_pptx(tmp_path)
    tree = await PptxParser().parse(path)
    leaves = _flatten_leaf_nodes(tree.nodes)
    # "보안 정책"(제목)과 똑같은 텍스트를 가진 리프가 없어야 한다.
    assert all(n.text.strip() != "보안 정책" for n in leaves)


# ─────────────────────────────────────────────
# fail-soft — 빈/제목없는 슬라이드
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_empty_slide_fail_soft_no_exception(tmp_path: Path) -> None:
    """
    텍스트 없는 빈 슬라이드도 예외 없이 부분 트리를 반환해야 한다(fail-soft).

    제목이 없으면 문서 title 은 파일명(stem)으로 폴백한다.
    """
    path = _make_empty_slide_pptx(tmp_path)
    tree = await PptxParser().parse(path)

    # 슬라이드 노드 자체는 존재하지만 텍스트 리프는 없다.
    assert len(tree.nodes) == 1
    assert tree.nodes[0].element_type == ElementType.SLIDE
    assert _flatten_leaf_nodes(tree.nodes) == []
    # 제목 placeholder 가 없으니 파일명(stem)으로 폴백.
    assert tree.title == "empty"


@pytest.mark.asyncio
async def test_parse_corrupted_file_returns_warning_not_raise(tmp_path: Path) -> None:
    """
    손상된(ZIP 이 아닌) .pptx 를 parse 하면 예외 대신 빈 트리 + warnings 를 반환.

    (can_parse 는 이를 걸러내지만, parse 가 직접 호출돼도 fail-soft 해야 한다.)

    PptxParser 가 PackageNotFoundError 를 포착하도록 수정되어 fail-soft 가 보장된다.
    """
    broken = tmp_path / "broken.pptx"
    broken.write_bytes(b"PK\x03\x04corrupted-not-a-real-zip")
    tree = await PptxParser().parse(broken)

    # 예외가 아니라 경고로 표현된다.
    assert tree.nodes == ()
    assert len(tree.warnings) >= 1
    assert "PPTX 로드 실패" in tree.warnings[0]
    # 제목은 파일명(stem)으로 폴백.
    assert tree.title == "broken"
