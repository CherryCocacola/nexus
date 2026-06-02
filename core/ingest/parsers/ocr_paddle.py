"""
고품질 OCR 파서 — PaddleOCR 로 "스캔(이미지) PDF"와 이미지 파일을 구조 트리로
변환한다(한국어 인식 특화, GPU 권장).

왜 PaddleOCR(고품질)인가 (v7.3 로드맵 단계 8 — 한국어 표/레이아웃 OCR, GPU):
  Tesseract(경량 CPU)도 같은 "스캔 PDF/이미지 OCR" 역할을 하지만, 한국어
  인식 정확도·기울어진 텍스트·다양한 글꼴에서 한계가 있다. PaddleOCR 의
  PP-OCRv5 한국어 인식 모델은 딥러닝 기반이라 한국어 문서에서 Tesseract 보다
  정확하다. 인식 모델을 돌려야 하므로 GPU 가 권장된다(CPU 도 동작하나 느리다).
  따라서 requires_gpu=True 로 표시하고, 티어 분기(v7.3 Part 4)에서 GPU 가 있는
  호스트에서만 Tesseract 보다 우선 등록되도록 한다(GPU 없으면 Tesseract 폴백).

처리 개요 (과설계 금지 — 텍스트 보존 우선):
  - 이미지 파일(.png/.jpg/.jpeg/.tiff): PIL 로 열어 PaddleOCR 로 OCR → 인식된
    텍스트 라인들을 1덩이로 합쳐 PARAGRAPH 노드 1개(page=1)로 만든다.
  - 스캔 PDF(.pdf): pypdfium2 로 페이지를 이미지(적정 DPI)로 렌더한 뒤, 페이지마다
    OCR → 페이지별 PARAGRAPH 노드(page=페이지번호)로 만든다.
  - 표/레이아웃 고급 분류(PP-Structure 로 표→TABLE)는 이 파서의 1차 범위 밖이다.
    PP-Structure 는 레이아웃/표/방향/언어 모델을 모두 적재해 매우 무겁고, 본 단계의
    목표는 "한국어 텍스트를 정확히 보존"하는 것이므로 PP-OCR(텍스트 라인) 경로로
    PARAGRAPH 노드부터 둔다(과설계 금지). 표 구조 인식이 필요해지면 동일 어댑터
    슬롯에서 PPStructureV3 경로를 덧붙이면 된다(향후 확장).

PaddleOCR 실제 API (런타임 확인 결과 — paddleocr 3.6.0 / paddlepaddle 3.3.1):
  from paddleocr import PaddleOCR
  ocr = PaddleOCR(
      lang="korean",                       # 한국어 인식 모델(tesseract "kor" 과 다른 코드)
      enable_mkldnn=False,                 # Windows CPU oneDNN PIR 버그 회피(아래 주석)
      use_doc_orientation_classify=False,  # 방향/왜곡 보정 모델은 끄고(가볍게)
      use_doc_unwarping=False,
      use_textline_orientation=False,
  )
  results = list(ocr.predict(input))       # input: 파일 경로 str | numpy RGB ndarray
  # results 는 dict 형 결과들의 리스트. 각 결과 r 에서:
  #   r["rec_texts"]  → 인식된 텍스트 라인 리스트(list[str])
  #   r["rec_scores"] → 라인별 신뢰도(list[float], 0~1)
  # predict() 반환은 1회용 이터레이터일 수 있어 list() 로 한 번만 소진한다.

  예외 타입: PaddleOCR 은 공개 예외 베이스를 제공하지 않는다. 모델 적재 실패·
  추론 실패는 RuntimeError(그 하위 NotImplementedError 포함)/ValueError/OSError
  계열로 표면화하는 것을 실측으로 확인했다. 따라서 fail-soft 에서는 이 구체
  타입들을 명시 포착한다(bare except 금지 — anti-pattern #8).

OpenMP/KMP 주의 (Windows 개발 환경 한정):
  Windows 에서 paddle + torch 가 같은 프로세스에 올라가면 OpenMP 런타임(libiomp)
  DLL 이 중복 적재돼 import/실행이 죽는다. 개발 환경에서는 실행 전에
  환경변수 KMP_DUPLICATE_LIB_OK=TRUE 를 설정해야 한다(이 env 로 import/OCR 성공
  확인). **단, 이 파일은 os.environ["KMP_DUPLICATE_LIB_OK"] 를 코드에서 강제로
  설정하지 않는다.** Linux 배포 환경에서는 이 충돌이 없어 불필요하고, torch 의
  OpenMP 동작을 흔들어 다른 추론에 부작용을 줄 수 있기 때문이다. 환경변수는
  실행 측(쉘/서비스 유닛)에서 개발 환경에만 주입한다.

모델 다운로드(에어갭 주의):
  PaddleOCR 은 첫 실행 시 PP-OCRv5 검출/한국어 인식 모델 가중치를 인터넷에서
  내려받는다(개발 환경은 인터넷 OK). 배포(에어갭) 시에는 모델을 오프라인 번들로
  사전 배치해야 하며(PaddleX 모델 캐시 디렉토리에 사전 배치 / 환경변수로
  소스 체크 비활성화 등), 런타임 다운로드에 의존하지 않는다. 이 파서 코드는
  import / 호출만 하며 런타임 pip/모델 설치·다운로드 코드는 넣지 않는다
  (에어갭 규칙, anti-pattern #10).

fail-soft (anti-pattern #8):
  paddle/paddleocr import 실패(미설치 호스트), 모델 로드 실패, OCR 추론 예외를
  모두 구체 예외로 포착해 빈/부분 트리 + warning 으로 표현하고 예외를 전파하지
  않는다. 페이지 1장이 깨져도 나머지 페이지는 계속 OCR 한다. paddle 미설치
  호스트에서도 graceful: PaddleOCR 을 지연 import 하므로 import 실패는 첫
  parse() 에서 경고로 흡수되고(레지스트리 등록 자체는 docingest_server 가
  import 가능 여부를 사전 점검해 결정한다 — 미설치면 미등록).

의존성 방향 (P2): core.ingest.types / parser_base / core.config 만 의존.
  core/rag·core/model 무관(역방향/순환 위험 없음).
에어갭: paddleocr/pypdfium2/PIL 은 로컬 파일만 다룬다(모델 가중치는 사전 배치 전제).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError

# pypdfium2 의 런타임 오류(손상 PDF 등)는 PdfiumError(RuntimeError 하위)로 온다.
from pypdfium2 import PdfiumError

from core.ingest.parser_base import DocumentParser
from core.ingest.types import DocumentNode, DocumentTree, ElementType

logger = logging.getLogger("nexus.ingest.parsers.ocr_paddle")

# PDF 매직바이트 — 파일 선두는 항상 "%PDF"(PDF 규격). can_parse() 에서 확인한다.
# (Tesseract 파서와 동일 규칙 — 같은 확장자를 공유하므로 검증 기준을 일치시킨다.)
_PDF_MAGIC = b"%PDF"

# 이미지 매직바이트(시그니처) — 확장자만 믿지 않고 선두 바이트로 형식을 확인해
# fail-closed 한다(Tesseract 파서와 동일). 키: 사람이 읽는 이름, 값: 선두 바이트열.
#   PNG : \x89PNG\r\n\x1a\n
#   JPEG: \xff\xd8\xff (JFIF/EXIF 공통 SOI 마커)
#   TIFF: "II*\0"(리틀엔디언) 또는 "MM\0*"(빅엔디언)
_IMAGE_MAGICS: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"II*\x00",  # TIFF LE
    b"MM\x00*",  # TIFF BE
)

# 이 파서가 다루는 이미지 확장자(소문자, 점 포함) — Tesseract 파서와 동일.
_IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".tif")

# OCR 결과 텍스트로 인정할 최소 길이(공백 제외). 빈 페이지/노이즈만 나온 경우
# 노드를 만들지 않아 검색 노이즈를 줄인다(과탐 방지) — Tesseract 파서와 동일.
_MIN_OCR_CHARS = 1

# 렌더 안전 상한 — DPI 가 비정상적으로 크게 설정돼도 메모리 폭주를 막기 위한
# 클램프(Tesseract 파서와 동일). 72dpi=scale1.0 기준.
_MAX_DPI = 600
_MIN_DPI = 72

# 인식 신뢰도 하한 — 이보다 낮은 라인은 노이즈로 보고 버린다. 0.0 이면 모두
# 채택(현재 기본). 향후 config 로 뺄 여지가 있으나 과설계 금지로 상수로 둔다.
_MIN_REC_SCORE = 0.0

# PaddleOCR/paddle 런타임 추론 예외로 관찰된 타입들(공개 예외 베이스가 없음).
# NotImplementedError(oneDNN PIR 버그 등)는 RuntimeError 하위라 RuntimeError 로
# 함께 잡힌다. KeyError 는 결과 dict 키 변동에 대한 방어.
_OCR_RUNTIME_ERRORS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    OSError,
    KeyError,
    TypeError,
)


class PaddleOcrParser(DocumentParser):
    """
    스캔(이미지) PDF 및 이미지 파일(.png/.jpg/.jpeg/.tiff)을 PaddleOCR 로
    구조 트리로 변환하는 고품질 OCR 파서(한국어 인식 특화).

    딥러닝 인식 모델 추론에 GPU 가 권장되므로 requires_gpu=True 다(v7.3 단계 8 —
    고품질 OCR). GPU 없는 호스트에서는 레지스트리 우선순위상 Tesseract 경량
    파서로 폴백된다(docingest_server 등록 정책 참조).
    """

    def __init__(self) -> None:
        """
        파서 인스턴스를 만든다.

        왜 PaddleOCR 인스턴스를 지연 생성(lazy)하는가:
          PaddleOCR() 생성 시 한국어 인식/검출 모델 적재(및 첫 실행 시 다운로드)가
          트리거된다(무겁다). 레지스트리에 등록만 되고 실제 파싱이 한 번도
          일어나지 않는 호스트에서까지 모델을 적재하면 낭비이므로, 첫 parse()
          호출 때 한 번만 만들고 이후 재사용한다(_ocr 캐시).
        """
        # PaddleOCR 인스턴스 캐시(첫 parse 때 채운다). 타입은 Any — paddleocr 가
        # 미설치 호스트에서도 이 모듈이 import 되도록 paddleocr 를 모듈 상단에서
        # import 하지 않기 때문이다(지연 import).
        self._ocr: Any | None = None
        # OCR 설정 캐시: (paddle_lang, paddle_use_gpu, paddle_enable_mkldnn, dpi)
        self._cfg: tuple[str, bool, bool, int] | None = None

    # ─── 정체성/플래그 ───

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """
        이미지 확장자 + .pdf 를 처리한다(Tesseract 파서와 동일).

        .pdf 는 pdfplumber/docling 과 확장자를 공유한다. 레지스트리에는 이미지에
        대해 Tesseract 보다 높은 priority 로, .pdf 에 대해서는 디지털 텍스트
        파서보다 낮은 폴백 우선순위로 등록한다(아래 등록 정책 참조).
        """
        return (*_IMAGE_EXTS, ".pdf")

    @property
    def requires_gpu(self) -> bool:
        # PaddleOCR 인식/검출 딥러닝 모델 추론 — GPU 권장(v7.3 단계 8 의 고품질 축).
        # 티어 분기(v7.3 Part 4)가 GPU 호스트에서만 이 파서를 우선시키는 데 쓴다.
        return True

    # ─── 처리 가능 여부 (fail-closed) ───

    def can_parse(self, path: Path) -> bool:
        """
        확장자 + 매직바이트로 처리 가능 여부를 판단한다(fail-closed).

        Tesseract 파서와 동일 규칙: 이미지 확장자면 이미지 시그니처를, .pdf 면
        "%PDF" 를 선두에서 확인한다. 확장자가 맞지 않거나, 파일을 열 수 없거나,
        시그니처가 다르면 False(불확실하면 거부 — 함부로 OCR 하지 않는다).
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
        이미지 또는 스캔 PDF 를 PaddleOCR 로 OCR 해 구조 트리로 변환한다.

        반환: DocumentTree(title, source_path, nodes=페이지/이미지별 PARAGRAPH
        노드들, doc_format="ocr_image" 또는 "ocr_pdf", warnings=fail-soft 경고).

        - 치명적 상황(설정 적용 실패, PaddleOCR import/모델 적재 실패, 파일 열기
          실패 등)도 예외를 던지지 않고 빈 트리 + 경고로 표현한다(fail-soft).
        """
        warnings: list[str] = []

        # 1) 설정 로드(첫 호출 때 1회). 실패해도 fail-soft.
        cfg = self._ensure_config()
        dpi = cfg[3]

        # 2) PaddleOCR 인스턴스 준비(지연 생성 + import). import/모델 적재 실패는
        #    fail-soft — paddle 미설치 호스트에서도 빈 트리+경고로 흡수한다.
        try:
            ocr = self._ensure_ocr(cfg)
        except ImportError as e:
            # paddleocr/paddle 미설치 — 이 호스트에서는 OCR 불가(경고만).
            msg = f"PaddleOCR import 실패(미설치 추정): {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            return self._empty_tree(path, "ocr", (msg,))
        except _OCR_RUNTIME_ERRORS as e:
            # 모델 적재/초기화 실패(에어갭 모델 미배치 등) — fail-soft.
            msg = f"PaddleOCR 초기화 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            return self._empty_tree(path, "ocr", (msg,))

        suffix = path.suffix.lower()

        # 3) 분기: 이미지 vs 스캔 PDF.
        if suffix == ".pdf":
            nodes, doc_format = self._parse_pdf(ocr, path, dpi, warnings)
        else:
            nodes, doc_format = self._parse_image(ocr, path, warnings)

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

    def _ensure_config(self) -> tuple[str, bool, bool, int]:
        """
        OcrConfig 에서 PaddleOCR 설정을 한 번만 읽어 캐시한다.

        반환: (paddle_lang, paddle_use_gpu, paddle_enable_mkldnn, dpi)
        config 로딩이 어떤 이유로든 실패하면 안전한 개발 기본값으로 폴백한다
        (부분 가용성 우선 — fail-soft 정신). DPI 는 dpi 필드(스캔 PDF 렌더 해상도)
        를 Tesseract 파서와 공유한다.
        """
        if self._cfg is not None:
            return self._cfg

        paddle_lang = "korean"
        paddle_use_gpu = True
        paddle_enable_mkldnn = False
        dpi = 250

        try:
            from core.config import load_and_validate_config

            cfg = load_and_validate_config()
            ocr = cfg.ocr
            paddle_lang = (ocr.paddle_lang or "korean").strip() or "korean"
            paddle_use_gpu = bool(ocr.paddle_use_gpu)
            paddle_enable_mkldnn = bool(ocr.paddle_enable_mkldnn)
            dpi = int(ocr.dpi or 250)
        except (OSError, ValueError, RuntimeError, ImportError, AttributeError) as e:
            # config 로딩 실패 — 개발 기본값으로 폴백(경고만 남긴다).
            logger.warning("OcrConfig 로딩 실패 — PaddleOCR 기본값 폴백: %s", e)

        # DPI 안전 클램프(비정상 설정으로 인한 메모리 폭주 방지).
        dpi = max(_MIN_DPI, min(int(dpi), _MAX_DPI))

        self._cfg = (paddle_lang, paddle_use_gpu, paddle_enable_mkldnn, dpi)
        return self._cfg

    def _ensure_ocr(self, cfg: tuple[str, bool, bool, int]) -> Any:
        """
        PaddleOCR 인스턴스를 한 번만 만들어 캐시하고 재사용한다(지연 생성 + 지연 import).

        왜 지연 import 인가:
          paddleocr 는 무겁고, paddle 미설치 호스트도 있다. 모듈 상단에서 import
          하면 그런 호스트에서는 이 모듈 자체를 못 불러온다. 첫 parse() 시점에
          import 해 ImportError 를 parse() 의 fail-soft 로 흡수한다.

        구성(실측 확인된 안정 조합):
          - lang: 한국어 인식 모델("korean").
          - enable_mkldnn=False: Windows CPU paddle 3.3.1 의 oneDNN PIR 런타임
            버그 회피(모듈 docstring 참조). GPU 경로에는 영향 없음.
          - use_doc_orientation_classify / use_doc_unwarping /
            use_textline_orientation = False: 방향/왜곡 보정 등 부가 모델을 꺼서
            적재/추론을 가볍게 한다(텍스트 보존이 목표 — 과설계 금지).

        주의(GPU/device 인자): paddleocr 3.x 의 device 지정은 버전에 따라 인자명이
          다르고(paddle install 빌드가 CPU 면 무시), 호스트의 빌드가 곧 디바이스를
          결정한다. requires_gpu 분기는 레지스트리(docingest_server)가 담당하므로
          여기서는 device 를 강제하지 않고 paddle 빌드 기본을 따른다(견고·단순).
          paddle_use_gpu 는 향후 device 인자 분기 여지를 위해 cfg 에 보존한다.
        """
        if self._ocr is not None:
            return self._ocr

        paddle_lang, _use_gpu, enable_mkldnn, _dpi = cfg

        # 지연 import — 미설치 호스트에서는 ImportError 가 parse() 로 전파돼 흡수된다.
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang=paddle_lang,
            enable_mkldnn=enable_mkldnn,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        return self._ocr

    # ─────────────────────────────────────────
    # 내부: 이미지 OCR
    # ─────────────────────────────────────────

    def _parse_image(
        self,
        ocr: Any,
        path: Path,
        warnings: list[str],
    ) -> tuple[list[DocumentNode], str]:
        """
        이미지 파일 1개를 OCR 해 PARAGRAPH 노드 1개(page=1)로 만든다.

        실패(이미지 열기/OCR)는 warnings 에 남기고 빈 노드 목록을 돌린다(fail-soft).
        반환: (노드 목록, doc_format="ocr_image").
        """
        nodes: list[DocumentNode] = []

        # 1) 이미지 로드 — 손상/미지원 형식은 부분 결과(빈 목록)+경고로 흡수.
        #    PaddleOCR 은 파일 경로 문자열을 직접 받을 수 있지만, can_parse 를 이미
        #    통과했어도 손상 파일을 PaddleOCR 내부에서 만나면 예외 분류가 모호해진다.
        #    그래서 이미지는 PIL 로 먼저 열어 RGB numpy 배열로 넘긴다(예외 분류 명확화).
        try:
            with Image.open(path) as img:
                img.load()
                rgb = img.convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            msg = f"이미지 열기 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)
            return nodes, "ocr_image"

        # 2) OCR — 모델 추론 실패는 fail-soft.
        try:
            text = self._ocr_pil(ocr, rgb)
        except _OCR_RUNTIME_ERRORS as e:
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
        ocr: Any,
        path: Path,
        dpi: int,
        warnings: list[str],
    ) -> tuple[list[DocumentNode], str]:
        """
        스캔 PDF 를 페이지별로 이미지 렌더 → OCR 해 페이지마다 PARAGRAPH 노드를
        만든다(page=페이지번호). 구조·예외 처리는 Tesseract 파서와 동일한 패턴이다.

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
                node = self._ocr_pdf_page(ocr, pdf, page_index, scale, order, warnings)
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
        ocr: Any,
        pdf: pdfium.PdfDocument,
        page_index: int,
        scale: float,
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

        # OCR — 모델 추론 실패는 이 페이지만 건너뛴다.
        try:
            with pil_image as img:
                rgb = img.convert("RGB")
                text = self._ocr_pil(ocr, rgb)
        except _OCR_RUNTIME_ERRORS as e:
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
    def _ocr_pil(ocr: Any, image: Image.Image) -> str:
        """
        PIL(RGB) 이미지 1장을 PaddleOCR 로 OCR 해 인식 텍스트를 한 덩이로 합쳐
        돌려준다.

        구현 메모(실측 확인):
          - PaddleOCR 은 numpy RGB 배열을 입력으로 받는다. PIL→numpy 로 변환해
            넘긴다(numpy 는 paddleocr 의존성으로 항상 설치돼 있어 안전하게 import).
          - predict() 반환은 dict 형 결과들의 이터러블이며, 1회용일 수 있어
            list() 로 한 번만 소진한다. 각 결과의 rec_texts(list[str]) 를
            신뢰도(rec_scores) 하한 이상만 골라 줄바꿈으로 잇는다.
          - 예외는 잡지 않고 그대로 올린다 — 호출부가 문맥에 맞게 fail-soft
            (경고 누적/페이지 건너뛰기)하도록 책임을 분리한다(Tesseract 파서와 동일 철학).
        """
        import numpy as np

        arr = np.asarray(image)  # (H, W, 3) RGB uint8

        results = list(ocr.predict(arr))

        lines: list[str] = []
        for res in results:
            # 결과는 dict 처럼 키 접근이 가능하다(rec_texts / rec_scores).
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            for idx, line in enumerate(texts):
                # 신뢰도 하한 미만 라인은 노이즈로 보고 버린다(점수 없으면 채택).
                score = scores[idx] if idx < len(scores) else 1.0
                if score is None or float(score) >= _MIN_REC_SCORE:
                    text = (line or "").strip()
                    if text:
                        lines.append(text)

        return "\n".join(lines)

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
