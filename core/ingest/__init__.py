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

■ 왜 "지연(lazy) 재노출"인가 (2026-08-06 수정)
  예전에는 이 파일이 하위 심볼을 전부 즉시 import 했다. 그러면 파이썬이 부모
  패키지 __init__ 을 먼저 실행하는 성질 때문에,
      from core.ingest.parser_base import ParserRegistry   # 계약 하나만 필요해도
  가 docling·pdfplumber·pipeline(→core.rag/core.model)까지 통째로 끌고 들어왔다.
  실제로 웹 컨테이너(docling 미설치)에서 이것 때문에 문서 파서 전체가 죽었다.
  모듈 수준 __getattr__(PEP 562)로 "이름을 실제로 꺼낼 때"만 로드하도록 바꾼다.
  외부 사용법(`from core.ingest import DocumentIngestPipeline`)은 그대로다.

의존성 방향 (P2 규칙):
  core.ingest → core.rag(KnowledgeStore), core.model(ModelProvider). 역방향 없음.
  (즉 rag/model 이 ingest 를 import 하지 않는다 — 단방향 의존을 지킨다.)

작성자: 이현수 / 작성일: 2026-07-05
"""

# from __future__ 선언은 반드시 모듈 최상단(docstring 바로 아래)에 있어야 한다.
# 타입 힌트를 문자열처럼 지연 평가(lazy)해 순환 import·전방참조 문제를 줄여준다.
from __future__ import annotations

import importlib
from typing import Any

# --- 공개 이름 → 실제 구현이 들어 있는 하위 모듈 경로 ---
# 이 표가 "core.ingest 가 무엇을 재노출하는가"의 단일 진실원이다. 즉시 import 하지
# 않고 표만 들고 있다가, 아래 __getattr__ 이 요청받은 이름 하나만 로드한다.
_LAZY_EXPORTS: dict[str, str] = {
    # 데이터 모델(types)
    "ElementType": "core.ingest.types",
    "DocumentNode": "core.ingest.types",
    "DocumentTree": "core.ingest.types",
    "DocumentChunk": "core.ingest.types",
    # 파서 공통 계약(ABC)과 포맷별 파서를 등록·조회하는 레지스트리
    "DocumentParser": "core.ingest.parser_base",
    "ParserRegistry": "core.ingest.parser_base",
    # 포맷별 파서 구현 — 각각 하나의 파일 형식을 DocumentTree 로 해석한다
    "PptxParser": "core.ingest.parsers.pptx",
    "PdfPlumberParser": "core.ingest.parsers.pdf_plumber",
    "DoclingParser": "core.ingest.parsers.docling_layout",
    "HwpxParser": "core.ingest.parsers.hwpx",
    "HwpNativeParser": "core.ingest.parsers.hwp_native",
    "HwpViaLibreOfficeParser": "core.ingest.parsers.hwp_libreoffice",
    # 청킹기 — 문서 트리를 논리 단위(계층)로 잘라 청크 리스트로 만든다
    "StructureAwareChunker": "core.ingest.chunker",
    # 파이프라인 본체와 적재 출처(source) 태그 상수
    "DocumentIngestPipeline": "core.ingest.pipeline",
    "INGEST_SOURCE": "core.ingest.pipeline",
}


def __getattr__(name: str) -> Any:
    """패키지에 없는 이름을 요청받으면 그때 해당 하위 모듈만 import 한다(PEP 562).

    한 번 가져온 값은 globals() 에 캐시하므로 두 번째부터는 일반 속성 조회다.
    등록되지 않은 이름은 AttributeError 로 돌려 오타를 조용히 넘기지 않는다.
    라이브러리 부재로 인한 ImportError 는 감추지 않고 그대로 올린다 — 호출부가
    "이 환경에서는 이 포맷을 못 쓴다"를 판단할 수 있어야 하기 때문이다.
    """
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """dir() 과 자동완성이 지연 노출 이름까지 보이도록 한다."""
    return sorted([*globals().keys(), *_LAZY_EXPORTS])


# __all__: `from core.ingest import *` 시 밖으로 내보낼 공개 이름 목록.
# 문서화·자동완성의 '공식 공개 API' 역할도 하므로 내부 전용 심볼은 넣지 않는다.
__all__ = list(_LAZY_EXPORTS)
