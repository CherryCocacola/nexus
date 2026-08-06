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
만들고 (2) 이 파일의 _LAZY_PARSERS 에 이름을 등록하면 된다.

■ 왜 "지연(lazy) 재노출"인가 (2026-08-06 수정)
  예전에는 이 파일이 7개 파서를 전부 즉시 import 했다. 그런데 각 파서 모듈은
  자기 라이브러리(docling / pdfplumber / pytesseract / pypdfium2 …)를 모듈
  최상위에서 불러온다. 파이썬은 하위 모듈을 import 할 때 부모 패키지의 __init__
  을 먼저 실행하므로,
      import core.ingest.parsers.pptx   # PPTX 하나만 쓰고 싶어도
  가 docling 까지 끌고 들어와 **그 라이브러리가 없는 환경에서는 전부 실패**했다.
  실제로 웹 컨테이너(docling 미설치)에서 PPTX/HWPX 파싱이 통째로 죽었다.

  그래서 모듈 수준 __getattr__(PEP 562)로 "이름을 실제로 꺼낼 때" 그 파서만
  import 한다. 결과적으로
    - from core.ingest.parsers import PptxParser  → PPTX 모듈만 로드(기존과 동일)
    - import core.ingest.parsers.pptx             → 다른 파서 라이브러리와 무관
  가 되어, 무거운/선택적 파서가 없는 배포에서도 있는 것만 골라 쓸 수 있다.

작성자: 이현수 / 작성일: 2026-07-05
"""

# `from __future__ import annotations` : 타입 힌트를 문자열로 지연 평가하게 하여
# 순환 import 위험을 줄이고 아직 정의되지 않은 타입 참조도 허용한다. 반드시 파일
# 상단(모듈 docstring 바로 아래)에 위치해야 한다 — 위치를 바꾸면 안 된다.
from __future__ import annotations

import importlib
from typing import Any

# 공개 이름 → 실제 구현이 들어 있는 하위 모듈 경로.
# 이 표가 "무엇을 재노출하는가"의 단일 진실원이다(새 파서는 여기에만 추가).
_LAZY_PARSERS: dict[str, str] = {
    "DoclingParser": "core.ingest.parsers.docling_layout",
    "HwpNativeParser": "core.ingest.parsers.hwp_native",
    "HwpViaLibreOfficeParser": "core.ingest.parsers.hwp_libreoffice",
    "HwpxParser": "core.ingest.parsers.hwpx",
    "PaddleOcrParser": "core.ingest.parsers.ocr_paddle",
    "PdfPlumberParser": "core.ingest.parsers.pdf_plumber",
    "PptxParser": "core.ingest.parsers.pptx",
    "TesseractParser": "core.ingest.parsers.ocr_tesseract",
}


def __getattr__(name: str) -> Any:
    """패키지 이름공간에 없는 이름을 요청받았을 때 그때서야 해당 파서를 import 한다.

    파이썬은 `from core.ingest.parsers import PptxParser` 처럼 모듈에 없는 속성을
    찾을 때 이 함수를 호출한다(PEP 562). 한 번 가져온 값은 globals() 에 캐시해
    두 번째 호출부터는 일반 속성 조회로 처리된다.

    등록되지 않은 이름은 AttributeError 로 돌려준다(오타를 조용히 넘기지 않는다).
    파서 라이브러리가 없어 import 자체가 실패하면 그 ImportError 는 그대로
    올라간다 — 호출부가 "이 포맷은 이 환경에서 못 쓴다"를 판단할 수 있어야 한다.
    """
    module_path = _LAZY_PARSERS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # 캐시 — 다음 접근부터는 __getattr__ 을 타지 않는다
    return value


def __dir__() -> list[str]:
    """dir() 과 자동완성이 지연 노출 이름까지 보이도록 한다."""
    return sorted([*globals().keys(), *_LAZY_PARSERS])


# __all__ : `from core.ingest.parsers import *` 시 공개할 이름 목록이자, 이 패키지의
# 공식 공개 API 명세다. 여기에 없는 이름은 내부용으로 간주된다.
__all__ = list(_LAZY_PARSERS)
