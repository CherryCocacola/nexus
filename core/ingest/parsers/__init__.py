"""
포맷별 파서 구현체 패키지.

현재 구현:
  - pptx.PptxParser (python-pptx, MIT)

향후(후속 단계 — v7.3 Part 9):
  - pdf_plumber.PdfPlumberParser (pdfplumber, MISSING)
  - docling_layout.DoclingParser (Docling, MISSING)
  - ocr_*.* (PaddleOCR/Tesseract, MISSING)
  - hwpx.HwpxParser (zipfile+xml.etree)

이 패키지의 각 구현체는 core.ingest.parser_base.DocumentParser 를 상속한다.
"""

from __future__ import annotations

from core.ingest.parsers.pptx import PptxParser

__all__ = ["PptxParser"]
