"""
PDF 경량 파서 — pdfplumber(MIT)로 PDF 파일을 구조 트리로 변환한다.

왜 pdfplumber(경량)인가 (v7.3 로드맵 단계 5 — PDF 경량):
  pdfplumber 는 PDF 안에 "텍스트로 들어있는" 글자와 표를 좌표와 함께 추출한다.
  레이아웃 모델(Docling)이나 OCR(스캔 이미지 인식) 없이도 디지털 PDF 의
  본문/표를 그대로 읽어올 수 있어 가장 가볍다. 스캔(이미지) PDF 의 OCR 은
  이번 범위가 아니며, 별도 고품질 파서(docling/paddleocr)의 어댑터 슬롯으로
  나중에 끼워 넣는다.

처리 개요:
  파일 → 페이지 순회 → 페이지마다 "그 페이지 안의 노드들"을 생성
    └ 표(page.extract_tables())  → TABLE 노드 (행/열 보존)
    └ 단어(page.extract_words(size 포함)) → 같은 줄(top)끼리 묶어 라인 구성
        - 평균보다 뚜렷이 큰 글자 라인 → SUBHEADING (헤딩 휴리스틱)
        - 그 외 라인                   → PARAGRAPH
  읽기 순서는 좌표(top, x0) 기준으로 정렬해 좌/우 단(段)이 뒤섞이지 않게 한다.

과설계 금지 (지시 준수):
  헤딩 추정은 "폰트 크기" 휴리스틱 한 가지만 쓴다. 폰트 정보가 없거나 애매하면
  전부 PARAGRAPH 로 둔다. 핵심 목표는 "텍스트 + 표의 보존"이지 완벽한 문서
  구조 복원이 아니다.

fail-soft (anti-pattern #8):
  페이지 1장을 읽다 실패해도 전체를 중단하지 않는다. 실패 사유를 warnings 에
  모으고 나머지 페이지를 계속 처리한다. bare except 금지 — 구체 예외만 포착.

의존성 방향 (P2): core.ingest.types / parser_base 만 의존. core/rag·model 무관.
에어갭: pdfplumber 는 로컬 파일만 읽는다(외부 네트워크 없음).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.utils.exceptions import PdfminerException

from core.ingest.parser_base import DocumentParser
from core.ingest.types import DocumentNode, DocumentTree, ElementType

logger = logging.getLogger("nexus.ingest.parsers.pdf_plumber")

# PDF 파일의 매직바이트 — 파일 선두는 항상 "%PDF" 로 시작한다(PDF 규격).
# can_parse() 에서 확장자뿐 아니라 이 시그니처도 확인해 fail-closed 한다.
_PDF_MAGIC = b"%PDF"

# 같은 "줄"로 묶을 때 허용하는 세로(top) 오차(px). 글자마다 top 이 미세하게
# 다를 수 있으므로, 이 값 이내면 같은 라인으로 본다. 과하게 키우면 윗줄/아랫줄이
# 합쳐지고, 너무 작으면 한 줄이 쪼개진다. 일반 본문 폰트 기준의 보수적 값.
_LINE_TOP_TOLERANCE = 3.0

# 헤딩(SUBHEADING) 휴리스틱 — 라인 평균 글자 크기가 페이지 본문 중앙값의
# 이 배수 이상이면 소제목으로 간주한다. 1.0 에 가까운 작은 차이는 헤딩으로
# 보지 않아(과탐 방지), 1.25(=25% 더 큼) 정도를 기준으로 둔다.
_HEADING_SIZE_RATIO = 1.25

# 헤딩으로 인정할 최대 길이(글자 수). 제목은 보통 짧다 — 길면 큰 글씨 본문일
# 가능성이 높아 PARAGRAPH 로 둔다(과탐 방지).
_HEADING_MAX_CHARS = 80


class PdfPlumberParser(DocumentParser):
    """
    PDF(.pdf) 파일을 pdfplumber 로 구조 트리로 파싱하는 경량 파서.

    텍스트(글자)와 표를 좌표 기반으로 추출하므로 GPU 가 필요 없다
    (requires_gpu=False). 스캔(이미지) PDF 의 OCR 은 이번 범위가 아니다.
    """

    # ─── 정체성/플래그 ───

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        # .pdf 만 처리.
        return (".pdf",)

    @property
    def requires_gpu(self) -> bool:
        # 디지털 텍스트/표 추출 — GPU 레이아웃/OCR 모델 불필요.
        return False

    # ─── 처리 가능 여부 (fail-closed) ───

    def can_parse(self, path: Path) -> bool:
        """
        확장자(.pdf) + 매직바이트("%PDF")로 처리 가능 여부를 판단한다.

        fail-closed: 파일을 읽다 문제가 생기거나 시그니처가 다르면 False.
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
        .pdf 를 페이지 단위 구조 트리로 변환한다.

        반환: DocumentTree(title, source_path, nodes=페이지별 노드들, doc_format="pdf",
        warnings=fail-soft 경고).

        - 최상위 노드는 "페이지 안의 내용 노드(PARAGRAPH/SUBHEADING/TABLE)"들이다.
          PPTX 의 SLIDE 처럼 페이지를 감싸는 컨테이너 노드는 두지 않고, page 필드로
          페이지 번호를 표시한다(PDF 는 페이지가 곧 물리 단위라 컨테이너가 불필요).
        - 치명적 실패(파일 손상/암호화 등)는 빈 트리 + 경고로 표현한다(파이프라인
          중단 방지).
        """
        warnings: list[str] = []
        nodes: list[DocumentNode] = []

        # 1) 파일 로드 — 손상/암호화는 부분 결과(빈 트리)+경고로 처리(fail-soft).
        #    pdfplumber.open 은 손상 PDF 에서 PDFSyntaxError, 암호 PDF 에서
        #    PDFPasswordIncorrect 를 던질 수 있고, 내부 pdfminer 오류(예: "No /Root
        #    object")는 PdfminerException 으로 감싸 올린다. 이 셋은 모두 Exception
        #    직속 계열이라 OSError/ValueError 에 속하지 않으므로 명시적으로 포착해야
        #    fail-soft(빈 트리+경고)가 보장된다.
        try:
            pdf = pdfplumber.open(str(path))
        except (PdfminerException, PDFSyntaxError, PDFPasswordIncorrect, OSError, ValueError) as e:
            msg = f"PDF 로드 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            return DocumentTree(
                title=path.stem,
                source_path=str(path),
                nodes=(),
                doc_format="pdf",
                warnings=(msg,),
            )

        # with 로 감싸 항상 닫는다(파일 핸들 누수 방지).
        try:
            with pdf:
                # order 는 문서 전체 읽기 순서다. 페이지를 넘어가도 계속 증가시켜
                # 청커가 전 페이지에 걸친 순서를 알 수 있게 한다.
                order = 0
                for page_index, page in enumerate(pdf.pages, start=1):
                    page_nodes = self._parse_page(page, page_index, order, warnings)
                    nodes.extend(page_nodes)
                    order += len(page_nodes)
        except (PdfminerException, PDFSyntaxError, OSError, ValueError) as e:
            # 페이지 목록 순회 자체가 깨지는 드문 경우 — 지금까지 모은 노드는 유지.
            msg = f"PDF 페이지 순회 중단: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)

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

    def _parse_page(
        self,
        page,
        page_index: int,
        base_order: int,
        warnings: list[str],
    ) -> list[DocumentNode]:
        """
        페이지 1장을 노드 목록으로 변환한다(표 + 텍스트 라인).

        단계:
          1. 표(extract_tables) → TABLE 노드. 좌표를 모르므로 페이지 선두에 둔다.
          2. 단어(extract_words) → 같은 top 끼리 라인으로 묶고, (top,x0) 정렬로
             읽기 순서를 복원. 라인별로 PARAGRAPH/SUBHEADING 노드 생성.

        페이지 단위 실패는 warnings 에 남기고 빈 목록 반환(다음 페이지로 진행).
        """
        nodes: list[DocumentNode] = []
        order = base_order

        # 1) 표 추출 — 표 영역의 텍스트가 본문 라인에도 다시 잡힐 수 있으나,
        #    pdfplumber 의 기본 동작상 표/단어를 완전히 분리하긴 어렵다. 과설계
        #    금지 원칙에 따라 "표는 표대로, 텍스트는 텍스트대로" 둘 다 보존한다
        #    (검색 시 중복은 허용 — 누락보다 낫다).
        try:
            tables = page.extract_tables()
        except (PDFSyntaxError, ValueError, TypeError) as e:
            warnings.append(f"페이지 {page_index}: 표 추출 실패 — 건너뜀 ({type(e).__name__}: {e})")
            tables = []

        for table in tables or []:
            table_node = self._table_node(table, page_index, order)
            if table_node is not None:
                nodes.append(table_node)
                order += 1

        # 2) 텍스트 라인 추출.
        try:
            # extra_attrs=["size"] 로 글자 크기를 함께 받아 헤딩 휴리스틱에 쓴다.
            words = page.extract_words(extra_attrs=["size"])
        except (PDFSyntaxError, ValueError, TypeError) as e:
            warnings.append(
                f"페이지 {page_index}: 텍스트 추출 실패 — 건너뜀 ({type(e).__name__}: {e})"
            )
            words = []

        lines = self._group_words_into_lines(words)
        # 본문 글자 크기의 중앙값 — 헤딩 휴리스틱의 기준선.
        median_size = self._median_line_size(lines)

        for line_text, line_size in lines:
            element_type = self._classify_line(line_text, line_size, median_size)
            nodes.append(
                DocumentNode(
                    element_type=element_type,
                    text=line_text,
                    heading_path=(),
                    page=page_index,
                    order=order,
                )
            )
            order += 1

        return nodes

    @staticmethod
    def _table_node(
        table: list[list[str | None]],
        page_index: int,
        order: int,
    ) -> DocumentNode | None:
        """
        extract_tables() 가 돌려준 표 1개(행→셀 2차원 리스트)를 TABLE 노드로
        변환한다(행/열 구조 보존).

        직렬화 형식은 PPTX 파서와 동일하게 맞춘다: 각 행을 " | " 로 구분된 셀로,
        행은 줄바꿈으로 연결한다. 빈 표(내용 없음)는 None 반환(노드 생략).
        """
        rows_text: list[str] = []
        for row in table:
            # 셀이 None 일 수 있어(병합/빈칸) 빈 문자열로 정규화한다.
            cells = [(cell or "").strip() for cell in row]
            rows_text.append(" | ".join(cells))

        content = "\n".join(rt for rt in rows_text if rt.strip(" |"))
        if not content.strip():
            return None

        return DocumentNode(
            element_type=ElementType.TABLE,
            text=content,
            heading_path=(),
            page=page_index,
            order=order,
        )

    @staticmethod
    def _group_words_into_lines(words: list[dict]) -> list[tuple[str, float]]:
        """
        단어 목록을 "같은 줄"끼리 묶어 (라인 텍스트, 평균 글자 크기) 목록으로
        만든다. 읽기 순서를 (top, x0) 기준으로 복원한다.

        왜 직접 묶는가:
          pdfplumber 의 extract_text() 는 줄바꿈만 주고 글자 크기 정보를 잃는다.
          헤딩 휴리스틱(큰 글자→소제목)을 쓰려면 라인별 글자 크기가 필요하므로
          단어(좌표+size)를 받아 직접 라인으로 묶는다.

        알고리즘:
          1. 단어를 (top, x0) 로 정렬 → 위→아래, 좌→우 읽기 순서.
          2. 직전 단어와 top 차이가 허용오차 이내면 같은 라인, 아니면 새 라인.
          3. 라인 텍스트는 단어를 공백으로 잇고, 글자 크기는 단어 size 의 평균.
        """
        if not words:
            return []

        # top → x0 순 정렬(안정 정렬). top/x0 가 없을 가능성은 낮지만 안전 접근.
        ordered = sorted(
            words,
            key=lambda w: (float(w.get("top", 0.0)), float(w.get("x0", 0.0))),
        )

        lines: list[tuple[str, float]] = []
        cur_words: list[str] = []
        cur_sizes: list[float] = []
        cur_top: float | None = None

        for w in ordered:
            text = (w.get("text") or "").strip()
            if not text:
                continue
            top = float(w.get("top", 0.0))
            size = float(w.get("size", 0.0))

            # 새 라인 판정: 첫 단어이거나 top 이 허용오차를 벗어나면 라인 마감.
            if cur_top is None or abs(top - cur_top) <= _LINE_TOP_TOLERANCE:
                cur_words.append(text)
                cur_sizes.append(size)
                # 라인의 기준 top 은 첫 단어 top 으로 고정(누적 드리프트 방지).
                if cur_top is None:
                    cur_top = top
            else:
                # 직전 라인을 확정하고 새 라인을 시작.
                lines.append(PdfPlumberParser._finish_line(cur_words, cur_sizes))
                cur_words = [text]
                cur_sizes = [size]
                cur_top = top

        # 마지막 라인 마감.
        if cur_words:
            lines.append(PdfPlumberParser._finish_line(cur_words, cur_sizes))

        # 빈 라인(공백만)은 제거.
        return [(t, s) for (t, s) in lines if t.strip()]

    @staticmethod
    def _finish_line(words: list[str], sizes: list[float]) -> tuple[str, float]:
        """라인 단어들을 공백으로 잇고, 평균 글자 크기를 계산해 (텍스트, 크기) 반환."""
        text = " ".join(words).strip()
        valid_sizes = [s for s in sizes if s > 0]
        avg_size = (sum(valid_sizes) / len(valid_sizes)) if valid_sizes else 0.0
        return text, avg_size

    @staticmethod
    def _median_line_size(lines: list[tuple[str, float]]) -> float:
        """
        라인들의 글자 크기 중앙값을 구한다(헤딩 휴리스틱의 기준선).

        평균이 아니라 중앙값을 쓰는 이유: 큰 제목 몇 줄이 평균을 끌어올려 본문이
        오히려 헤딩으로 오탐되는 것을 막기 위함이다(중앙값은 이상치에 강건).
        크기 정보가 전혀 없으면 0.0 — 이 경우 헤딩 판정은 항상 거짓이 된다.
        """
        valid = sorted(s for (_t, s) in lines if s > 0)
        if not valid:
            return 0.0
        mid = len(valid) // 2
        if len(valid) % 2 == 1:
            return valid[mid]
        return (valid[mid - 1] + valid[mid]) / 2.0

    @staticmethod
    def _classify_line(text: str, line_size: float, median_size: float) -> ElementType:
        """
        라인 1개를 SUBHEADING 또는 PARAGRAPH 로 분류한다(폰트 크기 휴리스틱).

        규칙(과탐 방지 — 애매하면 PARAGRAPH):
          - 글자 크기 정보가 없으면(median 0) 무조건 PARAGRAPH.
          - 라인 크기가 본문 중앙값의 _HEADING_SIZE_RATIO 배 이상이고,
            길이가 _HEADING_MAX_CHARS 이하이면 SUBHEADING(제목은 보통 짧다).
          - 그 외는 PARAGRAPH.
        """
        if median_size <= 0 or line_size <= 0:
            return ElementType.PARAGRAPH
        if line_size >= median_size * _HEADING_SIZE_RATIO and len(text) <= _HEADING_MAX_CHARS:
            return ElementType.SUBHEADING
        return ElementType.PARAGRAPH
