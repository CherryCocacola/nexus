"""
core.ingest — 문서 양식(레이아웃) 인식 임베딩 파이프라인 (v7.3).

문서(현재 PPTX)를 "구조를 보존한 채" 청킹하여 벡터 임베딩으로 tb_knowledge 에
적재하는 파이프라인을 제공한다. 핵심 목표는 "충돌 없는 임베딩"(v7.3 Part 1.1):
서로 다른 논리 영역(소제목/표/그림캡션)의 텍스트가 한 청크에 섞이지 않게 한다.

구성:
  types.py        — DocumentTree/DocumentNode/DocumentChunk, ElementType
  parser_base.py  — DocumentParser(ABC), ParserRegistry
  parsers/pptx.py — PptxParser (python-pptx, MIT)
  chunker.py      — StructureAwareChunker (계층형 청킹)
  pipeline.py     — DocumentIngestPipeline (parse→chunk→embed→적재)

향후(후속 단계 — v7.3 Part 9): PDF(pdfplumber/Docling)/HWPX/OCR 파서를
DocumentParser 인터페이스로 끼워 넣는다(어댑터 슬롯).

의존성 방향 (P2):
  core.ingest → core.rag(KnowledgeStore), core.model(ModelProvider). 역방향 없음.
"""

from __future__ import annotations

from core.ingest.chunker import StructureAwareChunker
from core.ingest.parser_base import DocumentParser, ParserRegistry
from core.ingest.parsers.pptx import PptxParser
from core.ingest.pipeline import INGEST_SOURCE, DocumentIngestPipeline
from core.ingest.types import (
    DocumentChunk,
    DocumentNode,
    DocumentTree,
    ElementType,
)

__all__ = [
    # types
    "ElementType",
    "DocumentNode",
    "DocumentTree",
    "DocumentChunk",
    # parser
    "DocumentParser",
    "ParserRegistry",
    "PptxParser",
    # chunker
    "StructureAwareChunker",
    # pipeline
    "DocumentIngestPipeline",
    "INGEST_SOURCE",
]
