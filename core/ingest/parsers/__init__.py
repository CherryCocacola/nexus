"""문서 인제스트(ingest) 파이프라인의 "포맷별 파서 구현체"를 한데 모으는 패키지.

이 파일(__init__.py)의 역할은 딱 하나다. 하위 모듈에 흩어져 있는 각 파서
클래스를 이 패키지의 최상위 이름공간으로 끌어올려(re-export), 바깥 코드가
    from core.ingest.parsers import PdfPlumberParser
처럼 개별 모듈 경로를 몰라도 간단히 가져다 쓰게 하는 "진입 창구"다. 즉 실제
파싱 로직은 여기 없고, 아래 하위 모듈들에 각각 들어 있다.

지원 포맷과 파서 (각 항목: 클래스 — 사용 라이브러리/라이선스 — 도입 단계):
  - pptx.PptxParser            : PowerPoint(.pptx). python-pptx(MIT) 사용.
  - pdf_plumber.PdfPlumberParser : PDF 경량 파서. pdfplumber(MIT).
                                   v7.3 단계 5 — 텍스트 위주 PDF를 빠르게 처리.
  - docling_layout.DoclingParser : PDF 고품질 파서. Docling 사용.
                                   v7.3 단계 6 — 레이아웃 보존, GPU 권장.
  - hwpx.HwpxParser            : 한글 신포맷(.hwpx). python-hwpx(OWPML).
                                   v7.3 단계 7.
  - ocr_tesseract.TesseractParser : 스캔 PDF/이미지 OCR. Tesseract(Apache-2.0).
                                   v7.3 단계 8 — 경량 CPU 경로.
  - ocr_paddle.PaddleOcrParser : 스캔 PDF/이미지 OCR. PaddleOCR 사용.
                                   v7.3 단계 8 — 고품질 한국어, GPU 권장.
                                   현재는 어댑터 슬롯(추후 연결 지점) 성격.
  - hwp_libreoffice.HwpViaLibreOfficeParser : 구포맷 한글(.hwp).
                                   LibreOffice로 변환 후 python-docx로 파싱.
                                   v7.3 단계 9.

공통 계약: 위 모든 구현체는 core.ingest.parser_base.DocumentParser 를 상속한다.
따라서 인제스트 상위 코드는 개별 파서의 내부 구현을 몰라도 동일한 인터페이스로
호출할 수 있다(다형성). 새 포맷 파서를 추가할 때는 (1) 하위 모듈에 구현체를
만들고 (2) 이 파일에서 import 후 __all__ 에 이름을 등록하면 된다.

작성자: 이현수 / 작성일: 2026-07-05
"""

# `from __future__ import annotations` : 타입 힌트를 문자열로 지연 평가하게 하여
# 순환 import 위험을 줄이고 아직 정의되지 않은 타입 참조도 허용한다. 반드시 파일
# 상단(모듈 docstring 바로 아래)에 위치해야 한다 — 위치를 바꾸면 안 된다.
from __future__ import annotations

# --- 각 하위 모듈에서 파서 구현체를 이 패키지 이름공간으로 끌어올린다(re-export) ---
# 아래 import 들은 실제 사용처가 이 파일 안이 아니라 "바깥에서 이 이름을 가져다
# 쓰기 위한" 재노출 목적이다. 그래서 ruff 등 린터가 "미사용 import"로 오해할 수
# 있으나, __all__ 에 등록돼 있으므로 공개 API로 취급된다.
from core.ingest.parsers.docling_layout import DoclingParser
from core.ingest.parsers.hwp_libreoffice import HwpViaLibreOfficeParser
from core.ingest.parsers.hwpx import HwpxParser
from core.ingest.parsers.ocr_paddle import PaddleOcrParser
from core.ingest.parsers.ocr_tesseract import TesseractParser
from core.ingest.parsers.pdf_plumber import PdfPlumberParser
from core.ingest.parsers.pptx import PptxParser

# __all__ : `from core.ingest.parsers import *` 시 공개할 이름 목록이자, 이 패키지의
# 공식 공개 API 명세다. 여기에 없는 이름은 내부용으로 간주된다. 새 파서를 추가하면
# 위 import 와 함께 이 목록에도 반드시 등록해야 외부에 노출된다.
__all__ = [
    "DoclingParser",
    "HwpViaLibreOfficeParser",
    "HwpxParser",
    "PaddleOcrParser",
    "PdfPlumberParser",
    "PptxParser",
    "TesseractParser",
]
