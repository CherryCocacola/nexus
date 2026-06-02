"""
포맷별 파서 구현체 패키지.

현재 구현:
  - pptx.PptxParser            (python-pptx, MIT)
  - pdf_plumber.PdfPlumberParser (pdfplumber, MIT — v7.3 단계 5: PDF 경량)
  - docling_layout.DoclingParser (Docling — v7.3 단계 6: PDF 고품질, GPU 권장)
  - hwpx.HwpxParser            (python-hwpx, OWPML — v7.3 단계 7: HWPX)
  - ocr_tesseract.TesseractParser (Tesseract, Apache-2.0 — v7.3 단계 8:
                                   스캔 PDF/이미지 OCR, 경량 CPU)
  - ocr_paddle.PaddleOcrParser (PaddleOCR — v7.3 단계 8: 스캔 PDF/이미지 OCR,
                                   고품질 한국어, GPU 권장 — 어댑터 슬롯)
  - hwp_libreoffice.HwpViaLibreOfficeParser (LibreOffice 변환 + python-docx —
                                   v7.3 단계 9: 구포맷 .hwp)

이 패키지의 각 구현체는 core.ingest.parser_base.DocumentParser 를 상속한다.
"""

from __future__ import annotations

from core.ingest.parsers.docling_layout import DoclingParser
from core.ingest.parsers.hwp_libreoffice import HwpViaLibreOfficeParser
from core.ingest.parsers.hwpx import HwpxParser
from core.ingest.parsers.ocr_paddle import PaddleOcrParser
from core.ingest.parsers.ocr_tesseract import TesseractParser
from core.ingest.parsers.pdf_plumber import PdfPlumberParser
from core.ingest.parsers.pptx import PptxParser

__all__ = [
    "DoclingParser",
    "HwpViaLibreOfficeParser",
    "HwpxParser",
    "PaddleOcrParser",
    "PdfPlumberParser",
    "PptxParser",
    "TesseractParser",
]
