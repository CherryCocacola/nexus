"""
core.ingest.parsers.hwpx — HwpxParser 단위 테스트 (v7.3 단계 7 — HWPX).

핵심 검증 의도:
  1. can_parse() — fail-closed: 확장자(.hwpx) + 매직바이트(ZIP "PK\\x03\\x04")까지 봐야 True.
  2. python-hwpx 경로 — HwpxDocument.new() 로 만든 .hwpx 를 섹션/문단/표 노드로 변환.
     - page = 섹션 번호(1부터), 표는 " | "/줄바꿈 보존, 헤딩 휴리스틱(짧고 종결부호 없음).
  3. 폴백 경로 — python-hwpx 가 못 여는 'section*.xml 만 든 ZIP' 은 defusedxml 로
     본문을 긁어 PARAGRAPH 노드로 살린다(구조는 잃되 본문 보존).
  4. fail-soft — 섹션 파일이 없는 ZIP, 잘못된 파일도 예외 없이 빈/부분 트리 + warning.

fixture 전략:
  python-hwpx(hwpx 2.9.0) 설치 확인됨. HwpxDocument.new() 로 tmp_path 에 .hwpx 를
  런타임 생성한다(바이너리 미커밋). 폴백 경로는 zipfile 로 직접 'section*.xml 만
  든 가짜 ZIP' 을 만들어 검증한다(실 한글 파일/운영 무관).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from hwpx.document import HwpxDocument

from core.ingest.parsers.hwpx import HwpxParser
from core.ingest.types import ElementType


# ─────────────────────────────────────────────
# fixture 생성 헬퍼
# ─────────────────────────────────────────────
def _make_hwpx(tmp_path: Path) -> Path:
    """문단 2개 + 2x2 표 1개를 가진 .hwpx 를 HwpxDocument.new() 로 생성한다.

    - "성리학 개요": 짧고 종결부호 없음 → SUBHEADING 휴리스틱.
    - "이기이원론은 ... 구분한다.": 종결부호(.)로 끝남 → PARAGRAPH.
    - 표: 헤더행/데이터행을 " | " 로 직렬화한 TABLE 노드.
    """
    doc = HwpxDocument.new()
    doc.add_paragraph("성리학 개요")
    doc.add_paragraph("이기이원론은 이와 기를 구분한다.")
    table = doc.add_table(2, 2)
    table.cell(0, 0).text = "이름"
    table.cell(0, 1).text = "부서"
    table.cell(1, 0).text = "홍길동"
    table.cell(1, 1).text = "보안팀"
    path = tmp_path / "doc.hwpx"
    doc.save_to_path(str(path))
    return path


def _make_fallback_zip(tmp_path: Path) -> Path:
    """python-hwpx 가 열 수 없는(필수 mimetype 없는) ZIP 에 section0.xml 만 넣는다.

    이렇게 하면 HwpxDocument.open 이 실패(HwpxStructureError)해 zipfile 폴백 경로로
    넘어가고, defusedxml 로 section0.xml 의 텍스트를 PARAGRAPH 노드로 긁어낸다."""
    path = tmp_path / "fallback.hwpx"
    section_xml = (
        "<?xml version='1.0'?><root><p>폴백 본문 첫째</p><p>폴백 본문 둘째</p></root>"
    ).encode()
    with zipfile.ZipFile(str(path), "w") as zf:
        zf.writestr("Contents/section0.xml", section_xml)
    return path


def _make_zip_without_sections(tmp_path: Path) -> Path:
    """section*.xml 이 전혀 없는 ZIP — 폴백조차 본문을 못 찾는 경우(fail-soft)."""
    path = tmp_path / "nosections.hwpx"
    with zipfile.ZipFile(str(path), "w") as zf:
        zf.writestr("other.txt", b"x")
    return path


# ─────────────────────────────────────────────
# can_parse — fail-closed
# ─────────────────────────────────────────────
def test_can_parse_valid_hwpx_returns_true(tmp_path: Path) -> None:
    """정상 .hwpx(ZIP 시그니처 포함)는 True."""
    path = _make_hwpx(tmp_path)
    assert HwpxParser().can_parse(path) is True


def test_can_parse_wrong_extension_returns_false(tmp_path: Path) -> None:
    """확장자가 .hwpx 가 아니면 내용과 무관하게 False(확장자 우선 거부)."""
    valid = _make_hwpx(tmp_path)
    renamed = tmp_path / "doc.zip"
    renamed.write_bytes(valid.read_bytes())
    assert HwpxParser().can_parse(renamed) is False


def test_can_parse_non_zip_magic_returns_false(tmp_path: Path) -> None:
    """확장자는 .hwpx 지만 매직바이트가 ZIP 이 아니면 False(fail-closed)."""
    fake = tmp_path / "fake.hwpx"
    fake.write_bytes(b"NOTZIP")
    assert HwpxParser().can_parse(fake) is False


def test_can_parse_missing_file_returns_false(tmp_path: Path) -> None:
    """파일이 없으면 OSError 를 삼키고 False."""
    assert HwpxParser().can_parse(tmp_path / "nope.hwpx") is False


def test_supported_extensions_and_no_gpu() -> None:
    """정체성 플래그 — .hwpx 만 지원, GPU 불필요."""
    parser = HwpxParser()
    assert parser.supported_extensions == (".hwpx",)
    assert parser.requires_gpu is False


# ─────────────────────────────────────────────
# python-hwpx 경로 — 섹션/문단/표
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_paragraphs_and_table(tmp_path: Path) -> None:
    """문단/표가 노드로 변환되고 page=섹션번호(1), 표 셀이 ' | ' 로 보존되는지."""
    path = _make_hwpx(tmp_path)
    tree = await HwpxParser().parse(path)

    assert tree.doc_format == "hwpx"
    assert tree.title == "doc"
    assert tree.nodes  # 본문 노드가 하나 이상

    # 모든 노드 page 는 섹션번호(여기선 단일 섹션이므로 1).
    assert all(n.page == 1 for n in tree.nodes)

    # 표 노드 — 행/열 보존.
    tables = [n for n in tree.nodes if n.element_type == ElementType.TABLE]
    assert len(tables) == 1
    content = tables[0].text
    assert "이름 | 부서" in content
    assert "홍길동 | 보안팀" in content
    assert "\n" in content


@pytest.mark.asyncio
async def test_parse_heading_heuristic(tmp_path: Path) -> None:
    """짧고 종결부호 없는 문단은 SUBHEADING, 종결부호로 끝나는 문단은 PARAGRAPH."""
    path = _make_hwpx(tmp_path)
    tree = await HwpxParser().parse(path)

    by_text = {n.text: n.element_type for n in tree.nodes}
    # "성리학 개요" — 짧고 종결부호 없음 → SUBHEADING.
    assert by_text.get("성리학 개요") == ElementType.SUBHEADING
    # 마침표로 끝나는 본문 → PARAGRAPH.
    assert by_text.get("이기이원론은 이와 기를 구분한다.") == ElementType.PARAGRAPH


@pytest.mark.asyncio
async def test_parse_order_is_monotonic(tmp_path: Path) -> None:
    """노드 order 가 읽기 순서대로 단조 증가해야 한다."""
    path = _make_hwpx(tmp_path)
    tree = await HwpxParser().parse(path)
    orders = [n.order for n in tree.nodes]
    assert orders == sorted(orders)


# ─────────────────────────────────────────────
# 폴백 경로 (zipfile + defusedxml)
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_fallback_extracts_text_from_section_xml(tmp_path: Path) -> None:
    """python-hwpx 가 못 여는 ZIP 도 폴백이 section0.xml 본문을 PARAGRAPH 로 살린다.

    필수 mimetype 이 없어 HwpxDocument.open 이 실패 → zipfile 폴백 → defusedxml 로
    텍스트 노드를 긁어 PARAGRAPH 로 만든다. 구조(표/제목)는 잃지만 본문은 보존된다."""
    path = _make_fallback_zip(tmp_path)
    tree = await HwpxParser().parse(path)

    texts = [n.text for n in tree.nodes]
    assert "폴백 본문 첫째" in texts
    assert "폴백 본문 둘째" in texts
    # 폴백은 구조를 모르므로 전부 PARAGRAPH.
    assert all(n.element_type == ElementType.PARAGRAPH for n in tree.nodes)
    # 폴백을 탔다는 사실이 warnings 에 기록돼야 한다(fail-soft 추적성).
    assert tree.warnings
    assert tree.doc_format == "hwpx"


@pytest.mark.asyncio
async def test_parse_zip_without_sections_fail_soft(tmp_path: Path) -> None:
    """section*.xml 이 없는 ZIP 은 예외 없이 빈 트리 + warning 을 반환해야 한다."""
    path = _make_zip_without_sections(tmp_path)
    tree = await HwpxParser().parse(path)
    assert tree.nodes == ()
    assert tree.warnings  # "section*.xml 을 찾지 못함" 등 사유 기록


@pytest.mark.asyncio
async def test_parse_non_zip_file_fail_soft(tmp_path: Path) -> None:
    """ZIP 조차 아닌 .hwpx 를 parse 해도 예외 대신 빈 트리 + warning(이중 폴백 실패)."""
    bad = tmp_path / "bad.hwpx"
    bad.write_bytes(b"NOT-A-ZIP-AT-ALL")
    tree = await HwpxParser().parse(bad)
    assert tree.nodes == ()
    assert tree.warnings
    assert tree.title == "bad"  # 파일명(stem) 폴백
