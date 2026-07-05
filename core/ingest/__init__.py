"""
core.ingest — 문서 양식(레이아웃) 인식 임베딩 파이프라인 (v7.3).

이 파일은 core.ingest '패키지의 대문(진입점)' 역할을 하는 __init__.py 이다.
하위 모듈(types/parser_base/parsers/chunker/pipeline)에 흩어져 있는 공개 클래스를
여기서 한 번에 import 해서 다시 노출(re-export)한다. 덕분에 다른 코드는
`from core.ingest import DocumentIngestPipeline` 처럼 짧은 경로로 쓸 수 있고,
하위 파일 구조가 바뀌어도 이 파일만 고치면 되어 외부 영향이 줄어든다.

이 패키지가 하는 일:
  문서(PPTX·PDF·HWPX·HWP 등)를 "구조(레이아웃)를 보존한 채" 청킹하고,
  벡터 임베딩으로 변환해 tb_knowledge 테이블에 적재하는 파이프라인을 제공한다.
  핵심 목표는 "충돌 없는 임베딩"(v7.3 Part 1.1): 서로 다른 논리 영역
  (소제목·표·그림캡션 등)의 텍스트가 한 청크에 뒤섞이지 않도록 분리한다.
  이렇게 해야 검색(RAG) 단계에서 엉뚱한 문맥이 딸려오지 않는다.

구성(하위 모듈과 대표 심볼):
  types.py               — DocumentTree/DocumentNode/DocumentChunk, ElementType
  parser_base.py         — DocumentParser(ABC), ParserRegistry
  parsers/pptx.py        — PptxParser (python-pptx, MIT)
  parsers/pdf_plumber.py — PdfPlumberParser (pdfplumber, MIT — PDF 경량)
  parsers/docling_layout.py — DoclingParser (Docling — PDF 고품질, GPU 권장)
  parsers/hwpx.py        — HwpxParser (python-hwpx, OWPML — HWPX)
  parsers/hwp_libreoffice.py — HwpViaLibreOfficeParser (LibreOffice 변환 — 구포맷 .hwp)
  chunker.py             — StructureAwareChunker (계층형 청킹)
  pipeline.py            — DocumentIngestPipeline (parse→chunk→embed→적재)

처리 흐름 한눈에:
  parse(파서가 문서를 DocumentTree 로 해석) → chunk(구조 인식 청킹) →
  embed(임베딩 모델로 벡터화) → 적재(tb_knowledge 저장).
  파일 포맷마다 전용 파서가 있고, 모두 DocumentParser(ABC) 계약을 따르므로
  ParserRegistry 를 통해 포맷별로 교체·선택된다.

향후(후속 단계 — v7.3 Part 9): 스캔 OCR 파서를 DocumentParser 인터페이스로
끼워 넣는다(어댑터 슬롯). 즉 새 포맷 지원은 파서 한 개를 추가하고
레지스트리에 등록하는 방식으로 확장된다.

의존성 방향 (P2 규칙):
  core.ingest → core.rag(KnowledgeStore), core.model(ModelProvider). 역방향 없음.
  (즉 rag/model 이 ingest 를 import 하지 않는다 — 단방향 의존을 지킨다.)

작성자: 이현수 / 작성일: 2026-07-05
"""

# from __future__ 선언은 반드시 모듈 최상단(docstring 바로 아래)에 있어야 한다.
# 타입 힌트를 문자열처럼 지연 평가(lazy)해 순환 import·전방참조 문제를 줄여준다.
from __future__ import annotations

# --- 아래는 하위 모듈의 공개 심볼을 끌어와 패키지 레벨에서 다시 노출하는 부분 ---
# 여기서 미리 import 해두면, 외부는 하위 파일 경로를 몰라도 core.ingest 에서 바로 쓴다.
# 청킹기: 문서 트리를 논리 단위(계층)로 잘라 청크 리스트로 만든다.
from core.ingest.chunker import StructureAwareChunker

# 파서 공통 계약(ABC)과, 포맷별 파서를 등록·조회하는 레지스트리.
from core.ingest.parser_base import DocumentParser, ParserRegistry

# 포맷별 파서 구현들 — 각각 하나의 파일 형식을 DocumentTree 로 해석한다.
from core.ingest.parsers.docling_layout import DoclingParser  # PDF 고품질(Docling)
from core.ingest.parsers.hwp_libreoffice import HwpViaLibreOfficeParser  # 구포맷 .hwp
from core.ingest.parsers.hwpx import HwpxParser  # HWPX(OWPML)
from core.ingest.parsers.pdf_plumber import PdfPlumberParser  # PDF 경량
from core.ingest.parsers.pptx import PptxParser  # PPTX

# 파이프라인 본체와, 적재 출처(source) 태그 상수.
from core.ingest.pipeline import INGEST_SOURCE, DocumentIngestPipeline

# 데이터 모델(문서 트리·노드·청크와 요소 타입 열거형).
from core.ingest.types import (
    DocumentChunk,
    DocumentNode,
    DocumentTree,
    ElementType,
)

# __all__: `from core.ingest import *` 시 밖으로 내보낼 공개 이름 목록.
# 문서화·자동완성의 '공식 공개 API' 역할도 하므로 내부 전용 심볼은 넣지 않는다.
__all__ = [
    # 데이터 모델(types)
    "ElementType",
    "DocumentNode",
    "DocumentTree",
    "DocumentChunk",
    # 파서(parser) — 공통 계약·레지스트리 및 포맷별 구현
    "DocumentParser",
    "ParserRegistry",
    "PptxParser",
    "PdfPlumberParser",
    "DoclingParser",
    "HwpxParser",
    "HwpViaLibreOfficeParser",
    # 청킹기(chunker)
    "StructureAwareChunker",
    # 파이프라인(pipeline)
    "DocumentIngestPipeline",
    "INGEST_SOURCE",
]
