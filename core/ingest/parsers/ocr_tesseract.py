"""
OCR 파서 — Tesseract 로 "스캔(이미지) PDF"와 이미지 파일을 구조 트리로 변환한다.

왜 OCR(Tesseract)인가 (v7.3 로드맵 단계 8 — 스캔 PDF/이미지 OCR, 경량 CPU):
  pdfplumber/docling 은 PDF "안에 글자로 들어있는" 디지털 텍스트를 읽는다. 그러나
  종이를 스캔해 만든 PDF 나 사진/캡처 이미지(.png/.jpg/.tiff)는 글자가 아니라
  "그림(픽셀)"이라 그 파서들로는 텍스트가 거의 안 나온다. 이런 문서는 OCR(광학
  문자 인식)로 픽셀 속 글자를 읽어내야 한다. Tesseract 는 CPU 만으로 도는 경량
  엔진(Apache-2.0)이라 GPU 가 없는 호스트에서도 쓸 수 있어 requires_gpu=False 다.

처리 개요:
  - 이미지 파일(.png/.jpg/.jpeg/.tiff): PIL 로 열어 pytesseract.image_to_string()
    으로 통째 OCR → 텍스트 1덩이를 PARAGRAPH 노드로 만든다(page=1).
  - 스캔 PDF(.pdf): pypdfium2 로 페이지를 이미지(적정 DPI)로 렌더한 뒤, 페이지마다
    OCR → 페이지별 PARAGRAPH 노드(page=페이지번호)로 만든다. "이미지로 보고 OCR"
    하므로, 디지털 텍스트가 있는 PDF 인지 여부는 이 파서가 따지지 않는다(아래
    레지스트리 정책 주석 참조).
  - 레이아웃 고급 분류(제목/표 인식 등)는 이 파서의 범위 밖이다 — 텍스트 위주
    (과설계 금지). 더 정밀한 구조가 필요하면 docling 어댑터 슬롯을 쓴다.

tesseract 경로/언어/tessdata 위치 (anti-pattern #4 — 하드코딩 금지):
  pytesseract.pytesseract.tesseract_cmd(실행 파일 경로), lang(언어), tessdata_dir
  (언어 데이터 폴더)는 모두 OcrConfig(core/config.py)에서 읽는다. config 로드가
  안 되면 환경변수(NEXUS_TESSERACT_CMD / NEXUS_TESSDATA_DIR)로 폴백하고, 그것도
  없으면 개발 기본값(Windows 설치 경로 / LOCALAPPDATA\nexus_tessdata)을 쓴다.
  배포(에어갭) 시에는 config/환경변수로 실제 경로(리눅스 등)를 덮어쓴다.

fail-soft (anti-pattern #8):
  tesseract 미설치(TesseractNotFoundError), 언어 데이터 없음/엔진 오류
  (TesseractError), 이미지 열기/렌더 실패(OSError/PdfiumError) 등은 모두 구체
  예외로 포착해 빈 트리(또는 부분 트리) + warnings 로 표현하고 예외를 전파하지
  않는다. 페이지 1장이 깨져도 나머지 페이지는 계속 OCR 한다. bare except 금지.

의존성 방향 (P2): core.ingest.types / parser_base 만 의존. core/rag·model 무관.
에어갭: pytesseract/pypdfium2/PIL 은 로컬 파일만 다룬다(외부 네트워크 없음).
  import / 호출만 하며 런타임 pip/바이너리 설치 코드는 넣지 않는다(anti-pattern #10).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pypdfium2 as pdfium
import pytesseract
from PIL import Image, UnidentifiedImageError

# pypdfium2 의 런타임 오류(손상 PDF 등)는 PdfiumError(RuntimeError 하위)로 온다.
from pypdfium2 import PdfiumError

# tesseract 미설치/엔진 오류 — fail-soft 포착 대상(실제 확인된 예외 타입).
from pytesseract.pytesseract import TesseractError, TesseractNotFoundError

from core.ingest.parser_base import DocumentParser
from core.ingest.types import DocumentNode, DocumentTree, ElementType

logger = logging.getLogger("nexus.ingest.parsers.ocr_tesseract")

# PDF 매직바이트 — 파일 선두는 항상 "%PDF"(PDF 규격). can_parse() 에서 확인한다.
_PDF_MAGIC = b"%PDF"

# 이미지 매직바이트(시그니처) — 확장자만 믿지 않고 선두 바이트로 형식을 확인해
# fail-closed 한다. 키: 사람이 읽는 이름, 값: 파일 선두에 와야 하는 바이트열.
#   PNG : \x89PNG\r\n\x1a\n
#   JPEG: \xff\xd8\xff (JFIF/EXIF 공통 SOI 마커)
#   TIFF: "II*\0"(리틀엔디언) 또는 "MM\0*"(빅엔디언)
_IMAGE_MAGICS: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"II*\x00",  # TIFF LE
    b"MM\x00*",  # TIFF BE
)

# 이 파서가 다루는 이미지 확장자(소문자, 점 포함).
_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".tif")

# OCR 결과 텍스트로 인정할 최소 길이(공백 제외). 빈 페이지/노이즈만 나온 경우
# 노드를 만들지 않아 검색 노이즈를 줄인다(과탐 방지).
_MIN_OCR_CHARS = 1

# 렌더 안전 상한 — DPI 가 비정상적으로 크게 설정돼도 메모리 폭주를 막기 위한
# 클램프. 72dpi=scale1.0 기준이므로 600dpi 면 scale≈8.3. 이 이상은 무의미하다.
_MAX_DPI = 600
_MIN_DPI = 72


class TesseractParser(DocumentParser):
    """
    스캔(이미지) PDF 및 이미지 파일(.png/.jpg/.jpeg/.tiff)을 Tesseract OCR 로
    구조 트리로 변환하는 경량 파서.

    CPU 만으로 동작하므로 requires_gpu=False 다(v7.3 단계 8 — 경량 CPU OCR).
    """

    def __init__(self) -> None:
        """
        파서 인스턴스를 만든다.

        왜 tesseract 설정을 지연(lazy)으로 적용하는가:
          tesseract_cmd(실행 파일 경로) 같은 전역 설정은 OcrConfig 에서 읽는데,
          config 로딩이 가능한지·필요한지는 첫 parse() 시점에 판단하는 편이
          안전하다(레지스트리에 등록만 되고 한 번도 안 쓰는 호스트에서 config
          접근/경고를 피한다). 따라서 첫 parse() 호출 때 한 번만 적용한다.
        """
        # OCR 설정 캐시(첫 parse 때 채운다). (tesseract_cmd, lang, tessdata_dir, dpi)
        self._cfg: tuple[str, str, str, int] | None = None

    # ─── 정체성/플래그 ───

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """
        이미지 확장자 + .pdf 를 처리한다.

        .pdf 는 pdfplumber/docling 과 확장자를 공유한다. 레지스트리에는 이
        파서를 .pdf 에 대해 가장 낮은 priority(폴백)로 등록하거나, 파이프라인이
        "디지털 텍스트 파서 결과가 비면" 호출하는 식으로 쓴다(아래 build_app
        등록 정책 주석 참조). 여기서는 "처리 가능한 확장자"만 선언한다.
        """
        return (*_IMAGE_EXTS, ".pdf")

    @property
    def requires_gpu(self) -> bool:
        # Tesseract 는 CPU 엔진 — GPU 불필요(v7.3 단계 8 의 핵심: 경량 CPU OCR).
        return False

    # ─── 처리 가능 여부 (fail-closed) ───

    def can_parse(self, path: Path) -> bool:
        """
        확장자 + 매직바이트로 처리 가능 여부를 판단한다(fail-closed).

        - 이미지 확장자면 이미지 시그니처를, .pdf 면 "%PDF" 를 선두에서 확인한다.
        - 확장자가 맞지 않거나, 파일을 열 수 없거나, 시그니처가 다르면 False.
          (불확실하면 거부 — 함부로 OCR 하지 않는다.)
        """
        suffix = path.suffix.lower()
        if suffix not in self.supported_extensions:
            return False

        try:
            with path.open("rb") as f:
                head = f.read(8)  # 가장 긴 시그니처(PNG 8바이트)에 맞춰 충분히 읽는다.
        except OSError as e:
            # 파일을 열 수 없음 — 처리 불가로 간주(fail-closed).
            logger.debug("can_parse 파일 열기 실패: %s (%s)", path, e)
            return False

        if suffix == ".pdf":
            return head.startswith(_PDF_MAGIC)
        # 이미지 — 알려진 시그니처 중 하나로 시작해야 한다.
        return any(head.startswith(magic) for magic in _IMAGE_MAGICS)

    # ─── 핵심: parse ───

    async def parse(self, path: Path) -> DocumentTree:
        """
        이미지 또는 스캔 PDF 를 OCR 해 구조 트리로 변환한다.

        반환: DocumentTree(title, source_path, nodes=페이지/이미지별 PARAGRAPH
        노드들, doc_format="ocr_image" 또는 "ocr_pdf", warnings=fail-soft 경고).

        - 치명적 상황(설정 적용 실패, 파일 열기 실패 등)도 예외를 던지지 않고
          빈 트리 + 경고로 표현한다(파이프라인 중단 방지 — fail-soft).
        """
        warnings: list[str] = []

        # 1) tesseract 전역 설정 적용(첫 호출 때 1회). 실패해도 fail-soft.
        #    여기서 tesseract_cmd 와 TESSDATA_PREFIX(언어 데이터 폴더)를 적용하므로,
        #    이후 OCR 호출에는 lang/dpi 만 넘기면 된다.
        try:
            _cmd, lang, _tessdata_dir, dpi = self._ensure_config()
        except (OSError, ValueError, RuntimeError) as e:
            # 설정 로딩/적용 실패 — OCR 불가로 보고 빈 트리+경고.
            msg = f"OCR 설정 적용 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            return self._empty_tree(path, "ocr", (msg,))

        suffix = path.suffix.lower()

        # 2) 분기: 이미지 vs 스캔 PDF.
        if suffix == ".pdf":
            nodes, doc_format = self._parse_pdf(path, lang, dpi, warnings)
        else:
            nodes, doc_format = self._parse_image(path, lang, warnings)

        return DocumentTree(
            title=path.stem,
            source_path=str(path),
            nodes=tuple(nodes),
            doc_format=doc_format,
            warnings=tuple(warnings),
        )

    # ─────────────────────────────────────────
    # 내부: 설정
    # ─────────────────────────────────────────

    def _ensure_config(self) -> tuple[str, str, str, int]:
        """
        OCR 설정(tesseract_cmd/lang/tessdata_dir/dpi)을 한 번만 읽어 캐시하고,
        tesseract 실행 파일 경로를 pytesseract 전역에 적용한다.

        우선순위: OcrConfig(config.yaml/기본값) → 실패 시 환경변수 폴백.
          - 실행 파일: NEXUS_TESSERACT_CMD
          - 데이터 폴더: NEXUS_TESSDATA_DIR
        config 로딩이 어떤 이유로든 실패해도 OCR 자체는 환경변수/기본값으로
        시도할 수 있게 한다(부분 가용성 우선 — fail-soft 정신).
        """
        if self._cfg is not None:
            return self._cfg

        tesseract_cmd = ""
        tessdata_dir = ""
        lang = "kor+eng"
        dpi = 250

        # OcrConfig 에서 읽기 — config 시스템 문제는 환경변수 폴백으로 흡수한다.
        try:
            from core.config import load_and_validate_config

            cfg = load_and_validate_config()
            ocr = cfg.ocr
            tesseract_cmd = (ocr.tesseract_cmd or "").strip()
            tessdata_dir = (ocr.tessdata_dir or "").strip()
            lang = (ocr.lang or "kor+eng").strip() or "kor+eng"
            dpi = int(ocr.dpi or 250)
        except (OSError, ValueError, RuntimeError, ImportError) as e:
            # config 로딩 실패 — 환경변수/기본값으로 폴백(경고만 남긴다).
            logger.warning("OcrConfig 로딩 실패 — 환경변수/기본값 폴백: %s", e)

        # 환경변수 폴백(config 값이 비어 있을 때만 보강).
        if not tesseract_cmd:
            tesseract_cmd = os.environ.get("NEXUS_TESSERACT_CMD", "").strip()
        if not tessdata_dir:
            tessdata_dir = os.environ.get("NEXUS_TESSDATA_DIR", "").strip()

        # DPI 안전 클램프(비정상 설정으로 인한 메모리 폭주 방지).
        dpi = max(_MIN_DPI, min(int(dpi), _MAX_DPI))

        # tesseract 실행 파일 경로를 pytesseract 전역에 적용한다.
        # (경로가 비어 있으면 PATH 상의 tesseract 를 쓴다 — 리눅스 배포 등.)
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # 언어 데이터 폴더는 TESSDATA_PREFIX 환경변수로 지정한다.
        #
        # 왜 `--tessdata-dir` config 문자열이 아니라 환경변수인가:
        #   pytesseract 는 config 문자열을 shlex.split(config, posix=(win이 아님))
        #   으로 쪼갠다. Windows(posix=False)에서는 따옴표가 인자에 그대로 남아
        #   경로가 깨지고("...nexus_tessdata"), 따옴표를 빼면 공백 있는 경로가
        #   둘로 쪼개진다. 즉 config 문자열로는 윈도/리눅스 양쪽을 안전하게
        #   다루기 어렵다. 반면 TESSDATA_PREFIX 는 tesseract 가 그대로 읽는
        #   표준 환경변수라 공백·OS 와 무관하게 안정적이다(실측으로 확인).
        #   프로세스 단위로만 설정하므로(os.environ) 같은 프로세스의 다른 코드에만
        #   영향이 있고, 외부 시스템을 건드리지 않는다(에어갭 무관).
        if tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = tessdata_dir

        self._cfg = (tesseract_cmd, lang, tessdata_dir, dpi)
        return self._cfg

    # ─────────────────────────────────────────
    # 내부: 이미지 OCR
    # ─────────────────────────────────────────

    def _parse_image(
        self,
        path: Path,
        lang: str,
        warnings: list[str],
    ) -> tuple[list[DocumentNode], str]:
        """
        이미지 파일 1개를 OCR 해 PARAGRAPH 노드 1개(page=1)로 만든다.

        실패(이미지 열기/OCR)는 warnings 에 남기고 빈 노드 목록을 돌린다(fail-soft).
        반환: (노드 목록, doc_format="ocr_image").
        """
        nodes: list[DocumentNode] = []

        # 1) 이미지 로드 — 손상/미지원 형식은 부분 결과(빈 목록)+경고로 흡수.
        #    OCR 호출은 일부러 이 블록 밖으로 분리한다: TesseractNotFoundError 는
        #    OSError 의 하위 타입이라 같은 try 안에 두면 "이미지 열기 실패" except 에
        #    그늘져(shadow) 잘못된 경고가 남는다. copy()로 픽셀을 떠서 핸들을 닫은 뒤
        #    별도 try 에서 OCR 한다(누수 방지 + 예외 분류 정확화).
        try:
            with Image.open(path) as img:
                img.load()
                pil_image = img.copy()
        except (OSError, UnidentifiedImageError) as e:
            msg = f"이미지 열기 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)
            return nodes, "ocr_image"

        # 2) OCR — tesseract 미설치/언어 데이터 없음/엔진 오류는 fail-soft.
        try:
            text = self._ocr_image(pil_image, lang)
        except (TesseractNotFoundError, TesseractError, ValueError, OSError) as e:
            msg = f"이미지 OCR 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)
            return nodes, "ocr_image"

        text = (text or "").strip()
        if len(text) >= _MIN_OCR_CHARS:
            nodes.append(
                DocumentNode(
                    element_type=ElementType.PARAGRAPH,
                    text=text,
                    heading_path=(),
                    page=1,
                    order=0,
                )
            )
        else:
            warnings.append("이미지 OCR 결과가 비어 있음(인식된 텍스트 없음).")

        return nodes, "ocr_image"

    # ─────────────────────────────────────────
    # 내부: 스캔 PDF OCR
    # ─────────────────────────────────────────

    def _parse_pdf(
        self,
        path: Path,
        lang: str,
        dpi: int,
        warnings: list[str],
    ) -> tuple[list[DocumentNode], str]:
        """
        스캔 PDF 를 페이지별로 이미지 렌더 → OCR 해 페이지마다 PARAGRAPH 노드를
        만든다(page=페이지번호).

        - pypdfium2 로 문서를 열고, 페이지마다 render(scale=dpi/72) 로 비트맵을
          만든 뒤 to_pil() 로 PIL 이미지를 얻어 OCR 한다.
        - 페이지 1장이 깨져도(렌더/OCR 실패) 그 페이지만 건너뛰고 경고를 남긴 채
          다음 페이지를 계속 처리한다(fail-soft).
        - 문서 자체를 못 여는 치명적 상황은 빈 목록 + 경고로 흡수한다.

        반환: (노드 목록, doc_format="ocr_pdf").
        """
        nodes: list[DocumentNode] = []
        # render() 의 scale 은 72dpi 기준 배율이다(예: 250dpi → scale≈3.47).
        scale = dpi / 72.0

        # 1) 문서 열기 — 손상/암호 PDF 는 PdfiumError, 입출력 문제는 OSError.
        try:
            pdf = pdfium.PdfDocument(str(path))
        except (PdfiumError, OSError, ValueError) as e:
            msg = f"스캔 PDF 열기 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)
            return nodes, "ocr_pdf"

        try:
            order = 0
            # 페이지 수를 안전하게 구한다(len 이 깨질 가능성은 낮지만 방어적).
            try:
                page_count = len(pdf)
            except (PdfiumError, TypeError) as e:
                msg = f"스캔 PDF 페이지 수 조회 실패: {type(e).__name__}: {e}"
                logger.warning("%s (%s)", msg, path)
                warnings.append(msg)
                page_count = 0

            for page_index in range(1, page_count + 1):
                node = self._ocr_pdf_page(pdf, page_index, scale, lang, order, warnings)
                if node is not None:
                    nodes.append(node)
                    order += 1
        finally:
            # pypdfium2 문서는 명시적으로 닫아 네이티브 리소스를 해제한다.
            try:
                pdf.close()
            except (PdfiumError, OSError, AttributeError) as e:
                # 닫기 실패는 본류에 영향 없음 — 디버그 로그만 남긴다.
                logger.debug("스캔 PDF 닫기 실패(무시): %s (%s)", path, e)

        if not nodes:
            warnings.append("스캔 PDF OCR 결과가 비어 있음(인식된 텍스트 없음).")

        return nodes, "ocr_pdf"

    def _ocr_pdf_page(
        self,
        pdf: pdfium.PdfDocument,
        page_index: int,
        scale: float,
        lang: str,
        order: int,
        warnings: list[str],
    ) -> DocumentNode | None:
        """
        스캔 PDF 의 페이지 1장을 이미지로 렌더해 OCR 하고, 텍스트가 있으면
        PARAGRAPH 노드를 돌려준다(없으면 None).

        페이지 단위 실패(렌더/OCR)는 warnings 에 남기고 None 을 돌려 다음
        페이지로 넘어간다(fail-soft — 한 장이 전체를 막지 않는다).
        """
        # pypdfium2 페이지 인덱스는 0부터다(사용자 page 번호는 1부터).
        try:
            page = pdf[page_index - 1]
        except (PdfiumError, IndexError, ValueError) as e:
            warnings.append(
                f"페이지 {page_index}: 페이지 접근 실패 — 건너뜀 ({type(e).__name__}: {e})"
            )
            return None

        try:
            # 페이지를 비트맵으로 렌더 → PIL 이미지로 변환.
            bitmap = page.render(scale=scale)
            try:
                pil_image = bitmap.to_pil()
            finally:
                # 비트맵은 to_pil 후 즉시 닫아 네이티브 버퍼를 해제한다.
                try:
                    bitmap.close()
                except (PdfiumError, AttributeError):
                    pass
        except (PdfiumError, OSError, ValueError, RuntimeError) as e:
            warnings.append(
                f"페이지 {page_index}: 이미지 렌더 실패 — 건너뜀 ({type(e).__name__}: {e})"
            )
            return None
        finally:
            # 페이지도 명시적으로 닫는다(렌더 성공/실패와 무관).
            try:
                page.close()
            except (PdfiumError, AttributeError):
                pass

        # OCR — tesseract 미설치/엔진 오류는 이 페이지만 건너뛴다.
        try:
            with pil_image:
                text = self._ocr_image(pil_image, lang)
        except (TesseractNotFoundError, TesseractError, ValueError, OSError) as e:
            warnings.append(f"페이지 {page_index}: OCR 실패 — 건너뜀 ({type(e).__name__}: {e})")
            return None

        text = (text or "").strip()
        if len(text) < _MIN_OCR_CHARS:
            # 빈 페이지(인식 텍스트 없음)는 노드를 만들지 않는다(노이즈 방지).
            return None

        return DocumentNode(
            element_type=ElementType.PARAGRAPH,
            text=text,
            heading_path=(),
            page=page_index,
            order=order,
        )

    # ─────────────────────────────────────────
    # 내부: 공통 OCR 호출
    # ─────────────────────────────────────────

    @staticmethod
    def _ocr_image(image: Image.Image, lang: str) -> str:
        """
        PIL 이미지 1장을 pytesseract 로 OCR 해 텍스트를 돌려준다.

        lang(언어)만 넘기면 된다 — 언어 데이터 폴더는 _ensure_config 에서 이미
        TESSDATA_PREFIX 환경변수로 지정해 두었다(여기서 config 문자열로 경로를
        넘기지 않으므로 OS/공백 quoting 문제가 없다). 예외는 잡지 않고 그대로
        올린다 — 호출부(_parse_image/_ocr_pdf_page)가 문맥에 맞게 fail-soft
        처리(경고 누적/건너뛰기)하도록 책임을 분리한다.
        """
        return pytesseract.image_to_string(image, lang=lang)

    # ─────────────────────────────────────────
    # 내부: 공통 헬퍼
    # ─────────────────────────────────────────

    @staticmethod
    def _empty_tree(path: Path, doc_format: str, warnings: tuple[str, ...]) -> DocumentTree:
        """치명적 실패 시 돌려줄 빈 트리(파일명을 제목으로, 경고만 담는다)."""
        return DocumentTree(
            title=path.stem,
            source_path=str(path),
            nodes=(),
            doc_format=doc_format,
            warnings=warnings,
        )
