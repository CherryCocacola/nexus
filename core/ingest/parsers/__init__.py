"""
포맷별 파서 구현체 패키지.

현재 구현:
  - pptx.PptxParser            (python-pptx, MIT)
  - pdf_plumber.PdfPlumberParser (pdfplumber, MIT — v7.3 단계 5: PDF 경량)
  - hwpx.HwpxParser            (python-hwpx, OWPML — v7.3 단계 7: HWPX)

향후(후속 단계 — v7.3 Part 9):
  - docling_layout.DoclingParser (Docling, 고품질 레이아웃 — 어댑터 슬롯)
  - ocr_*.* (PaddleOCR/Tesseract, 스캔 PDF OCR — 어댑터 슬롯)

이 패키지의 각 구현체는 core.ingest.parser_base.DocumentParser 를 상속한다.
"""

from __future__ import annotations

from core.ingest.parsers.hwpx import HwpxParser
from core.ingest.parsers.pdf_plumber import PdfPlumberParser
from core.ingest.parsers.pptx import PptxParser

__all__ = ["HwpxParser", "PdfPlumberParser", "PptxParser"]
