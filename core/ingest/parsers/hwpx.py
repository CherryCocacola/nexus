"""
HWPX 파서 — python-hwpx(OWPML)로 한글(HWPX) 파일을 구조 트리로 변환한다.

왜 HWPX 인가 (v7.3 로드맵 단계 7 — HWPX):
  HWPX 는 한컴오피스 한글의 개방형 포맷으로, 내부가 ZIP 컨테이너 + OWPML(XML)
  구조다. 즉 "섹션 → 문단 → (텍스트|표|도형)" 의 논리 구조를 네이티브로 들고
  있어, PPTX 처럼 레이아웃/OCR 없이도 구조를 그대로 읽을 수 있다. 국내 공공/
  기업 문서가 대부분 HWPX 라 인제스트 가치가 크다.

python-hwpx 실제 API (런타임 확인 결과 — 2.9.0 기준):
  from hwpx.document import HwpxDocument
  doc = HwpxDocument.open(path)            # 파일 열기
  doc.sections                              # list[HwpxOxmlSection] — 구역
  section.paragraphs                        # list[HwpxOxmlParagraph]
  paragraph.text                            # str — 문단 텍스트(run 들 결합)
  paragraph.tables                          # list[HwpxOxmlTable] — 문단 내 표
  table.row_count / table.column_count      # int
  table.cell(r, c).text                     # str — 셀 텍스트(가장 견고한 접근)
  paragraph.style_id_ref / para_pr_id_ref   # 스타일/문단속성 id(헤딩 추정 단서)

  참고: table.iter_grid() 는 일부 표에서 예외를 던질 수 있어, 행/열을
  row_count/column_count + cell(r,c) 로 직접 순회하는 방식을 채택한다(견고).

처리 개요:
  파일 → 섹션 순회 → 섹션마다 그 안의 문단을 읽기 순서대로 노드로 변환
    └ 문단에 표(paragraph.tables)가 있으면 → TABLE 노드(행/열 보존)
    └ 문단 텍스트가 있으면 → PARAGRAPH (제목 스타일이면 SUBHEADING 휴리스틱)

헤딩 휴리스틱(과설계 금지):
  python-hwpx 가 스타일 "이름"을 문단 단위로 신뢰성 있게 노출하지 않으므로,
  스타일 id 만으로 제목을 단정하지 않는다. 대신 "짧고, 직후에 본문이 이어지는"
  보수적 신호가 있을 때만 SUBHEADING 으로 본다. 애매하면 PARAGRAPH(과탐 방지).

폴백:
  python-hwpx 로 여는 데 실패하면 zipfile+xml.etree 로 Contents/section*.xml 의
  텍스트를 직접 긁어 PARAGRAPH 노드로 만든다(구조는 잃지만 본문은 살린다).

fail-soft (anti-pattern #8):
  섹션/문단/표 하나를 읽다 실패해도 전체를 중단하지 않는다. 실패 사유를
  warnings 에 모으고 나머지를 계속 처리한다. bare except 금지 — 구체 예외만.

의존성 방향 (P2): core.ingest.types / parser_base 만 의존. core/rag·model 무관.
에어갭: python-hwpx / zipfile 모두 로컬 파일만 읽는다(외부 네트워크 없음).
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

# 폴백 XML 파싱은 defusedxml 로 한다 — HWPX 는 사용자가 준 파일일 수 있어,
# 표준 xml.etree 는 XML 폭탄(billion laughs)/외부 엔티티 공격에 취약하다.
# defusedxml.ElementTree.fromstring 은 동일 API 에 방어가 더해진 안전판이다.
# (심볼만 직접 import — 모듈 별칭은 ruff N81x 네이밍 규칙과 충돌하기 때문.)
from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import ParseError as XmlParseError
from defusedxml.ElementTree import fromstring as xml_fromstring
from hwpx.document import HwpxDocument

from core.ingest.parser_base import DocumentParser
from core.ingest.types import DocumentNode, DocumentTree, ElementType

logger = logging.getLogger("nexus.ingest.parsers.hwpx")

# HWPX 는 ZIP 컨테이너다. 매직바이트는 "PK\x03\x04"(ZIP 로컬 파일 헤더 시그니처).
# PPTX 와 동일한 시그니처이나, can_parse() 는 확장자(.hwpx)로 1차 구분하므로 충돌
# 없다(확장자가 .hwpx 인 ZIP 만 이 파서가 가져간다 — 레지스트리가 확장자로 라우팅).
_ZIP_MAGIC = b"PK\x03\x04"

# 헤딩(SUBHEADING) 휴리스틱 — 제목으로 인정할 최대 길이(글자 수).
# 제목은 보통 짧다. 길면 본문 문단일 가능성이 높아 PARAGRAPH 로 둔다(과탐 방지).
_HEADING_MAX_CHARS = 40


class HwpxParser(DocumentParser):
    """
    한글(.hwpx) 파일을 python-hwpx 로 구조 트리로 파싱하는 파서.

    OWPML(XML) 네이티브 구조를 읽으므로 GPU 가 필요 없다(requires_gpu=False).
    """

    # ─── 정체성/플래그 ───

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        # .hwpx 만 처리. 구포맷 .hwp(바이너리)는 대상 아님(별도 파서 필요).
        return (".hwpx",)

    @property
    def requires_gpu(self) -> bool:
        # 네이티브 OWPML 구조 파싱 — GPU 레이아웃/OCR 불필요.
        return False

    # ─── 처리 가능 여부 (fail-closed) ───

    def can_parse(self, path: Path) -> bool:
        """
        확장자(.hwpx) + 매직바이트(ZIP 시그니처)로 처리 가능 여부를 판단한다.

        fail-closed: 파일을 읽다 문제가 생기거나 시그니처가 다르면 False.
        """
        if path.suffix.lower() != ".hwpx":
            return False
        try:
            with path.open("rb") as f:
                head = f.read(4)
        except OSError as e:
            logger.debug("can_parse 파일 열기 실패: %s (%s)", path, e)
            return False
        return head == _ZIP_MAGIC

    # ─── 핵심: parse ───

    async def parse(self, path: Path) -> DocumentTree:
        """
        .hwpx 를 섹션·문단 단위 구조 트리로 변환한다.

        반환: DocumentTree(title, source_path, nodes=내용 노드들, doc_format="hwpx",
        warnings=fail-soft 경고).

        - 최상위 노드는 문단/표에서 만든 내용 노드(PARAGRAPH/SUBHEADING/TABLE)다.
          섹션은 page 필드(1부터 증가하는 섹션 번호)로 구분한다.
        - python-hwpx 로 열기 실패 시 zipfile 폴백으로 본문만이라도 살린다.
        """
        warnings: list[str] = []

        # 1) python-hwpx 우선 시도. HwpxDocument.open 은 손상/비HWPX 파일에서
        #    다양한 예외(PackageNotFoundError, KeyError, OSError, ValueError 등)를
        #    던질 수 있다. 이 경우 폴백(zipfile)으로 넘어간다.
        try:
            document = HwpxDocument.open(str(path))
        except Exception as e:  # noqa: BLE001 — 라이브러리 예외 표면이 넓어 폴백 트리거용으로만 포괄
            msg = f"python-hwpx 로드 실패 — zipfile 폴백 시도: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)
            return self._fallback_parse(path, warnings)

        # 2) 섹션 → 문단 순회로 노드 구성.
        nodes = self._nodes_from_document(document, warnings)

        # 본문을 하나도 못 읽었으면 폴백을 한 번 더 시도(라이브러리가 열기는
        # 했으나 구조 추출이 비어버린 드문 경우 — 본문 보존 우선).
        if not nodes:
            warnings.append("python-hwpx 가 본문 노드를 만들지 못함 — zipfile 폴백 시도")
            return self._fallback_parse(path, warnings)

        return DocumentTree(
            title=path.stem,
            source_path=str(path),
            nodes=tuple(nodes),
            doc_format="hwpx",
            warnings=tuple(warnings),
        )

    # ─────────────────────────────────────────
    # 내부: python-hwpx 경로
    # ─────────────────────────────────────────

    def _nodes_from_document(self, document, warnings: list[str]) -> list[DocumentNode]:
        """
        HwpxDocument 의 섹션→문단을 읽기 순서대로 노드 목록으로 변환한다.

        - 섹션 번호(1부터)를 page 필드에 담아 구역을 구분한다.
        - order 는 문서 전체에 걸쳐 단조 증가시켜 청커가 읽기 순서를 알 수 있게 한다.
        - 섹션 목록 접근이 깨지면(드묾) 빈 목록을 돌려 폴백으로 넘긴다.
        """
        try:
            sections = list(document.sections)
        except Exception as e:  # noqa: BLE001 — 섹션 접근 실패는 폴백 신호로만 사용
            warnings.append(f"섹션 목록 접근 실패: {type(e).__name__}: {e}")
            return []

        nodes: list[DocumentNode] = []
        order = 0

        for section_index, section in enumerate(sections, start=1):
            # 섹션의 문단 목록 접근 — 실패하면 그 섹션만 건너뛴다.
            try:
                paragraphs = list(section.paragraphs)
            except Exception as e:  # noqa: BLE001 — 한 섹션 실패가 전체를 막지 않게 포괄
                warnings.append(
                    f"섹션 {section_index}: 문단 접근 실패 — 건너뜀 ({type(e).__name__}: {e})"
                )
                continue

            for para in paragraphs:
                made = self._paragraph_to_nodes(para, section_index, order, warnings)
                nodes.extend(made)
                order += len(made)

        return nodes

    def _paragraph_to_nodes(
        self,
        para,
        section_index: int,
        base_order: int,
        warnings: list[str],
    ) -> list[DocumentNode]:
        """
        문단 1개를 노드 목록으로 변환한다.

        한 문단은 (a) 표를 0개 이상 품을 수 있고, (b) 텍스트를 가질 수 있다.
        읽기 순서상 표를 먼저, 그 다음 문단 텍스트를 둔다(표가 문단 흐름의
        앞쪽에 박히는 HWPX 의 일반적 배치를 반영). 빈 텍스트/빈 표는 생략한다.
        """
        nodes: list[DocumentNode] = []
        order = base_order

        # (a) 문단 내 표 — 행/열 보존 TABLE 노드.
        try:
            tables = list(para.tables)
        except Exception as e:  # noqa: BLE001 — 표 접근 실패는 경고 후 건너뜀
            warnings.append(
                f"섹션 {section_index}: 표 접근 실패 — 건너뜀 ({type(e).__name__}: {e})"
            )
            tables = []

        for table in tables:
            table_node = self._table_node(table, section_index, order, warnings)
            if table_node is not None:
                nodes.append(table_node)
                order += 1

        # (b) 문단 텍스트 — PARAGRAPH 또는 SUBHEADING.
        try:
            text = (para.text or "").strip()
        except Exception as e:  # noqa: BLE001 — 텍스트 접근 실패는 경고 후 건너뜀
            warnings.append(
                f"섹션 {section_index}: 문단 텍스트 접근 실패 — 건너뜀 ({type(e).__name__}: {e})"
            )
            text = ""

        if text:
            element_type = self._classify_paragraph(text)
            nodes.append(
                DocumentNode(
                    element_type=element_type,
                    text=text,
                    heading_path=(),
                    page=section_index,
                    order=order,
                )
            )
            order += 1

        return nodes

    @staticmethod
    def _table_node(
        table,
        section_index: int,
        order: int,
        warnings: list[str],
    ) -> DocumentNode | None:
        """
        HWPX 표 1개를 TABLE 노드로 변환한다(행/열 구조 보존).

        견고한 순회: iter_grid() 는 일부 표에서 예외를 던질 수 있어, row_count/
        column_count + cell(r, c).text 로 직접 행/열을 훑는다. 직렬화 형식은
        PPTX/PDF 파서와 동일하게 맞춘다(각 행을 " | " 로, 행은 줄바꿈으로).

        표 접근 실패는 fail-soft: 경고를 남기고 None 반환(노드 생략).
        """
        try:
            row_count = int(table.row_count)
            col_count = int(table.column_count)
        except Exception as e:  # noqa: BLE001 — 표 크기 접근 실패는 노드 생략 신호
            warnings.append(
                f"섹션 {section_index}: 표 크기 접근 실패 — 건너뜀 ({type(e).__name__}: {e})"
            )
            return None

        rows_text: list[str] = []
        for r in range(row_count):
            cells: list[str] = []
            for c in range(col_count):
                # 셀 1개 접근 실패는 그 셀만 빈칸 처리(표 전체를 버리지 않음).
                try:
                    cells.append((table.cell(r, c).text or "").strip())
                except Exception:  # noqa: BLE001 — 셀 단위 실패는 빈칸으로 흡수
                    cells.append("")
            rows_text.append(" | ".join(cells))

        content = "\n".join(rt for rt in rows_text if rt.strip(" |"))
        if not content.strip():
            return None

        return DocumentNode(
            element_type=ElementType.TABLE,
            text=content,
            heading_path=(),
            page=section_index,
            order=order,
        )

    @staticmethod
    def _classify_paragraph(text: str) -> ElementType:
        """
        문단 텍스트를 SUBHEADING 또는 PARAGRAPH 로 분류한다(보수적 휴리스틱).

        python-hwpx 가 문단 스타일 "이름"을 신뢰성 있게 노출하지 않으므로,
        구조를 단정하지 않고 길이만으로 보수적으로 판정한다:
          - 짧고(_HEADING_MAX_CHARS 이하) 문장 종결부호(. 。 ! ?)로 끝나지 않으면
            제목일 가능성 → SUBHEADING.
          - 그 외(길거나 문장으로 끝남)는 PARAGRAPH.
        애매하면 PARAGRAPH(과탐 방지) — 이번 범위의 핵심은 "본문/표 보존"이다.
        """
        if len(text) > _HEADING_MAX_CHARS:
            return ElementType.PARAGRAPH
        # 마지막 글자가 문장 종결부호이면 본문 문장으로 본다.
        if text[-1] in ".。!?":
            return ElementType.PARAGRAPH
        return ElementType.SUBHEADING

    # ─────────────────────────────────────────
    # 내부: zipfile 폴백 경로
    # ─────────────────────────────────────────

    def _fallback_parse(self, path: Path, warnings: list[str]) -> DocumentTree:
        """
        python-hwpx 가 실패했을 때의 폴백 — zipfile + xml.etree 로 본문만 긁는다.

        HWPX 내부 구조: ZIP 안에 Contents/section0.xml, section1.xml ... 형태로
        구역별 OWPML 이 들어있다. 여기서 모든 텍스트 노드(<...>...</...> 의 text)
        를 긁어 문단처럼 PARAGRAPH 노드로 만든다. 구조(표/제목 구분)는 잃지만
        본문 텍스트는 살려 검색 가능하게 한다(부분 성공 > 전체 실패).

        fail-soft: ZIP/XML 파싱 실패는 경고로 남기고 빈 트리(또는 부분 트리) 반환.
        """
        nodes: list[DocumentNode] = []
        order = 0

        try:
            with zipfile.ZipFile(str(path)) as zf:
                # Contents/section*.xml 를 번호 순으로 정렬해 읽기 순서를 맞춘다.
                section_names = sorted(
                    name
                    for name in zf.namelist()
                    if name.lower().startswith("contents/section") and name.lower().endswith(".xml")
                )
                if not section_names:
                    warnings.append("폴백: Contents/section*.xml 을 찾지 못함")

                for section_index, name in enumerate(section_names, start=1):
                    try:
                        raw = zf.read(name)
                        texts = self._extract_xml_texts(raw)
                    except (
                        zipfile.BadZipFile,
                        XmlParseError,
                        DefusedXmlException,
                        OSError,
                        ValueError,
                    ) as e:
                        warnings.append(
                            f"폴백: {name} 파싱 실패 — 건너뜀 ({type(e).__name__}: {e})"
                        )
                        continue

                    for text in texts:
                        nodes.append(
                            DocumentNode(
                                element_type=ElementType.PARAGRAPH,
                                text=text,
                                heading_path=(),
                                page=section_index,
                                order=order,
                            )
                        )
                        order += 1
        except (zipfile.BadZipFile, OSError) as e:
            msg = f"폴백 실패(ZIP 열기): {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)

        return DocumentTree(
            title=path.stem,
            source_path=str(path),
            nodes=tuple(nodes),
            doc_format="hwpx",
            warnings=tuple(warnings),
        )

    @staticmethod
    def _extract_xml_texts(raw: bytes) -> list[str]:
        """
        OWPML XML 바이트에서 모든 요소의 텍스트를 긁어 문단 단위 문자열 목록으로
        만든다. OWPML 은 네임스페이스가 많아 태그명을 일일이 매칭하지 않고,
        모든 요소의 .text 를 순회해 비어 있지 않은 것만 모은다(단순·견고).

        HWPX 본문 텍스트는 <hp:t> 요소에 들어가나, 폴백에서는 네임스페이스
        해석 없이 전체 트리의 text 를 긁는 편이 깨지지 않는다(과설계 금지).

        보안: defusedxml 로 파싱해 XML 폭탄/외부 엔티티 공격을 차단한다.
        """
        root = xml_fromstring(raw)
        texts: list[str] = []
        for elem in root.iter():
            if elem.text:
                t = elem.text.strip()
                if t:
                    texts.append(t)
        return texts
