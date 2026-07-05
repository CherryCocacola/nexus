"""
docingest MCP 서버 — 문서 인제스트 파이프라인을 MCP 도구로 노출한다.

한눈에 보기(온보딩용):
  이 파일은 "문서 파일을 읽어 지식 베이스에 넣고, 그 안에서 검색"하는 기능을
  MCP(Model Context Protocol) 서버 형태로 감싼 것이다. Nexus 오케스트레이터는
  이 서버가 노출하는 parse/ingest/search 3개 도구를 원격 도구처럼 호출한다.
  실제 파싱·임베딩·DB 적재 로직은 이 파일이 아니라 core/ingest, core/rag,
  core/model 에 이미 구현돼 있고, 여기서는 그 부품들을 조립(build_app)해
  MCP 앱으로 내보내는 "얇은 서버 레이어" 역할만 한다.

구성 요소:
  · DocParseTool  / DocIngestTool / DocSearchTool — 3개 MCP 도구 클래스.
  · build_app()   — 파서 레지스트리·임베딩·DB·파이프라인을 조립하는 진입점.
  · _gpu_available() / _paddleocr_available() — 호스트 능력 감지 헬퍼.

도구 3개:
  parse  (read-only) — 파일을 파싱·청킹만 하고 적재하지 않는다(dry_run). 구조
                       트리 요약(제목/청크수/경고)을 반환한다.
  ingest (쓰기)      — 파일을 파싱→청킹→임베딩→tb_knowledge 적재(source="docingest").
  search (read-only) — 적재된 문서를 의미 기반 검색한다(source="docingest" 필터).

core 재사용:
  DocumentIngestPipeline(parse/chunk/embed/적재) + ParserRegistry +
  PptxParser/PdfPlumberParser/HwpxParser + LocalModelProvider(embed) +
  KnowledgeStore 를 그대로 조립한다.

PDF 파서 우선순위 (v7.3 단계 6 — 고품질=GPU, 경량=CPU 폴백):
  .pdf 에는 두 파서가 후보다 — DoclingParser(레이아웃 인식, GPU 권장)와
  PdfPlumberParser(경량, CPU). 이 서버가 도는 호스트에 GPU 가 있으면 Docling 을
  더 높은 priority 로 등록해 우선시키고, GPU 가 없으면 Docling 을 등록하지 않아
  자동으로 pdfplumber 경량 파서로 폴백되게 한다. requires_gpu=True 파서를 GPU
  없는 호스트에 굳이 얹지 않는 것이 v7.3 티어 분기 의도에 맞는 가장 단순한 정책.

OCR 등록 정책 (v7.3 단계 8 — 스캔 PDF/이미지 OCR, 경량 CPU):
  TesseractParser(requires_gpu=False)는 두 역할을 한다.
    (1) 이미지 파일(.png/.jpg/.jpeg/.tiff): 이 파서가 유일한 후보 — 그대로 1차.
    (2) 스캔 PDF(.pdf): .pdf 의 1차는 항상 디지털 텍스트 파서(docling/pdfplumber)다.
        OCR 은 "텍스트가 거의 없는 스캔 PDF"에만 필요한 경량 폴백이므로,
        TesseractParser 를 .pdf 에 대해 가장 낮은 priority(=-10)로 등록한다.
  왜 priority 폴백만으로 충분한가(과설계 금지):
    ParserRegistry.get_for_path 는 priority 높은 순으로 can_parse() 가 True 인
    첫 파서를 고른다. pdfplumber(priority 0)는 모든 %PDF 에 can_parse=True 이므로
    디지털/스캔을 가리지 않고 항상 OCR(priority -10)보다 먼저 선택된다. 즉 단일
    파일 라우팅에서는 OCR 이 .pdf 의 자동 대상이 되지 않는다. "디지털 파서 결과가
    비면 OCR 로 폴백"하는 노드-기반 분기는 파이프라인 레벨 정책으로, 본 단계의
    범위(파서 추가 + 등록)를 넘어선다 — v7.3 의도(스캔 OCR=경량 폴백)에 맞춰
    여기서는 파서를 "명시 호출용 + 이미지 1차 + .pdf 최저 폴백"으로만 둔다.

read-only 구분 (fail-closed 정신):
  parse/search 는 부작용이 없고, ingest 만 DB 에 쓴다. 도구 설명에 명시해 호출
  측(권한 파이프라인)이 구분할 수 있게 한다.

에어갭:
  임베딩/DB 접속은 LAN. 파일 경로는 서버 로컬 파일 시스템만 대상으로 한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI

# McpServerTool: 모든 MCP 도구가 상속하는 ABC(name/description/input_schema/call).
# create_mcp_app: 도구 목록을 받아 인증·라우팅이 붙은 FastAPI 앱을 만들어 주는 팩토리.
from mcp_servers.framework import McpServerTool, create_mcp_app

# 모듈 전용 로거 — 규칙에 따라 "nexus.{module}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.mcp_servers.docingest")

# docingest 가 적재하는 지식 소스 식별자 — pipeline.INGEST_SOURCE 와 동일해야 한다.
# 이 값으로 tb_knowledge 에 태깅하고, search 도 이 값으로 필터해 다른 소스와 섞이지
# 않게 한다(예: kowiki 덤프와 격리).
_INGEST_SOURCE = "docingest"

# 검색 질의 임베딩 접두사(e5-large 규약 — kowiki 서버와 동일).
# e5 계열 임베딩 모델은 "query: " / "passage: " 접두사로 질의와 문서를 구분하도록
# 학습돼 있어, 검색 질의에는 반드시 이 접두사를 붙여야 정확도가 나온다.
_QUERY_PREFIX = "query: "
_DEFAULT_TOP_K = 5  # search 에서 top_k 미지정 시 기본 반환 개수
_MAX_TOP_K = 50  # 과도한 결과 요청 방지를 위한 상한(초과 시 이 값으로 클램프)


def _gpu_available() -> bool:
    """
    이 호스트에 CUDA GPU 가 있는지 fail-soft 로 판정한다(레지스트리 구성용).

    DoclingParser(requires_gpu=True)를 우선 등록할지 결정하는 데만 쓴다. torch 가
    없거나(에어갭 일부 호스트) 감지 중 어떤 오류가 나도 "GPU 없음"으로 간주해
    경량 파서(pdfplumber)로 자연스럽게 폴백한다 — 감지 실패가 인제스트 전체를
    막아서는 안 된다. import 만 하며 설치 코드는 넣지 않는다(에어갭 규칙).
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception as e:  # noqa: BLE001 — 감지 실패는 GPU 없음으로 흡수(폴백)
        logger.debug("GPU 감지 실패 — GPU 없음으로 간주: %s", e)
        return False


def _paddleocr_available() -> bool:
    """
    이 호스트에서 PaddleOCR(고품질 OCR)을 import 할 수 있는지 fail-soft 로 판정한다.

    PaddleOcrParser(requires_gpu=True)를 레지스트리에 등록할지 결정하는 데만 쓴다.
    paddleocr/paddle 미설치 호스트(에어갭 일부 노드 등)에서는 import 가 실패하므로
    아예 등록하지 않고 Tesseract 경량 OCR 로 폴백한다 — 감지 실패가 인제스트
    전체를 막아서는 안 된다. import 가능 여부만 확인하며 인스턴스는 만들지 않는다
    (모델 적재 비용 회피 — 실제 적재는 첫 parse() 때 지연 생성된다). 설치 코드는
    넣지 않는다(에어갭 규칙, anti-pattern #10).

    주의(KMP/OpenMP): Windows 개발 환경에서 paddle + torch 가 같은 프로세스에
    올라가면 OpenMP DLL 중복 적재로 import 가 죽을 수 있다. 그 경우 실행 측에서
    KMP_DUPLICATE_LIB_OK=TRUE 환경변수를 주입해야 한다(파서/서버 코드에서 강제
    설정하지 않음 — Linux 배포 불필요 + torch 부작용 우려). env 미설정으로
    import 가 죽으면 여기서 False 로 흡수돼 Tesseract 폴백이 된다.
    """
    try:
        import importlib.util

        return importlib.util.find_spec("paddleocr") is not None
    except Exception as e:  # noqa: BLE001 — 감지 실패는 미가용으로 흡수(폴백)
        logger.debug("PaddleOCR 감지 실패 — 미가용으로 간주: %s", e)
        return False


class DocParseTool(McpServerTool):
    """
    파일을 파싱·청킹만 하고 적재하지 않는 read-only 도구(dry_run).

    왜 필요한가:
      실제로 DB 에 넣기(ingest) 전에 "이 파일이 제대로 파싱되는지, 청크가 몇 개
      나오는지, 구조 경고는 없는지"를 미리 확인하는 안전 점검용이다. 부작용이
      전혀 없어(DB 미접근) read-only 로 분류되며, 권한 파이프라인이 이를 보고
      확인 없이 통과시킬 수 있다.

    핵심 흐름:
      call() → pipeline.ingest_file(path, dry_run=True) → 요약(dict) 반환.
    """

    def __init__(self, pipeline: Any) -> None:
        # DocumentIngestPipeline 인스턴스를 주입받아 보관한다(파싱/청킹 로직은
        # 전부 파이프라인에 있고, 이 도구는 dry_run 플래그만 켜서 위임한다).
        self._pipeline = pipeline

    @property
    def name(self) -> str:
        return "parse"

    @property
    def description(self) -> str:
        return (
            "[read-only] 문서 파일을 파싱·청킹만 수행하고 적재하지 않는다(dry_run). "
            "제목/포맷/청크 수/구조 경고 등 요약을 반환해 적재 전 점검에 쓴다."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "서버 로컬 파일 시스템의 문서 경로(예: .pptx).",
                }
            },
            "required": ["path"],
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        """
        dry_run=True 로 파이프라인을 돌려 적재 없이 요약만 반환한다.

        매개변수:
          arguments["path"] — 서버 로컬 파일 시스템의 문서 경로(필수, 문자열).
        반환:
          파이프라인이 만든 요약 dict(제목/청크 수/구조 경고 등).
        예외:
          path 누락/빈 문자열이면 ValueError, 파싱 중 오류는 RuntimeError 로
          정규화해 올린다(호출 측이 메시지만 보고 처리하도록).
        """
        # 입력 방어 — MCP 인자는 외부에서 오므로 타입/공백을 직접 검증한다.
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("필수 인자 'path'(문자열)가 없습니다.")
        try:
            # dry_run=True 이면 파이프라인이 임베딩/DB 적재를 건너뛰고 요약만 만든다.
            return await self._pipeline.ingest_file(path, dry_run=True)
        except (OSError, ValueError, RuntimeError) as e:
            # 파일 없음/파싱 실패 등 구체 예외를 하나의 RuntimeError 로 감싸 올린다
            # (bare except 금지 — anti-pattern #8. 예외 종류를 메시지에 남긴다).
            raise RuntimeError(f"파싱 실패: {type(e).__name__}: {e}") from e


class DocIngestTool(McpServerTool):
    """
    파일을 파싱→청킹→임베딩→tb_knowledge 적재하는 쓰기 도구.

    왜 필요한가:
      실제로 지식 베이스에 문서를 넣는 유일한 도구다. DB 에 쓰기 때문에(부작용)
      read-only 가 아니며, 권한 파이프라인이 쓰기 도구로 취급한다.

    멱등성:
      동일 파일을 다시 적재해도 UPSERT 로 처리돼 중복 행이 쌓이지 않는다
      (같은 파일 재인제스트가 안전하다).

    핵심 흐름:
      call() → pipeline.ingest_file(path, dry_run=False) → 적재 행수/오류 목록 반환.
    """

    def __init__(self, pipeline: Any) -> None:
        # parse 도구와 동일한 파이프라인을 공유한다. 차이는 dry_run 플래그뿐이다.
        self._pipeline = pipeline

    @property
    def name(self) -> str:
        return "ingest"

    @property
    def description(self) -> str:
        return (
            "[쓰기] 문서 파일을 파싱→청킹→임베딩하여 지식 베이스(tb_knowledge, "
            "source='docingest')에 적재한다. 동일 파일 재적재는 멱등(UPSERT)이다. "
            "적재 행수와 오류 목록을 반환한다."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "적재할 문서 경로(서버 로컬 파일 시스템).",
                }
            },
            "required": ["path"],
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        """
        dry_run=False 로 실제 적재를 수행한다.

        매개변수:
          arguments["path"] — 적재할 문서 경로(필수, 문자열).
        반환:
          적재 행수와 오류 목록을 담은 파이프라인 결과 dict.
        예외:
          path 누락/빈 문자열이면 ValueError, 적재 중 오류는 RuntimeError.
        """
        # parse 와 동일한 입력 검증. path 만 필요하다.
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("필수 인자 'path'(문자열)가 없습니다.")
        try:
            # dry_run=False 이면 임베딩까지 태워 tb_knowledge 에 실제로 UPSERT 한다.
            return await self._pipeline.ingest_file(path, dry_run=False)
        except (OSError, ValueError, RuntimeError) as e:
            # 적재 중 발생한 구체 예외를 RuntimeError 로 정규화해 올린다.
            raise RuntimeError(f"적재 실패: {type(e).__name__}: {e}") from e


class DocSearchTool(McpServerTool):
    """
    적재된 docingest 문서를 벡터 검색하는 read-only 도구.

    왜 필요한가:
      ingest 로 넣어 둔 문서를 자연어 질의로 되찾기 위한 도구다. 질의를 임베딩해
      코사인 유사도가 높은 청크를 돌려준다. DB 를 읽기만 하므로 read-only.

    parse/ingest 와 달리 파이프라인을 쓰지 않고, 임베딩 프로바이더와 지식 저장소를
    직접 받아 "질의 임베딩 → 벡터 검색" 두 단계만 수행한다.

    핵심 흐름:
      call() → model.embed(["query: "+질의]) → store.search_by_vector(...) → 결과.
    """

    def __init__(self, model_provider: Any, knowledge_store: Any) -> None:
        # _model: 질의 문장을 임베딩 벡터로 바꾸는 프로바이더(LocalModelProvider).
        # _store: 벡터 유사도 검색을 담당하는 지식 저장소(KnowledgeStore).
        self._model = model_provider
        self._store = knowledge_store

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "[read-only] docingest 로 적재된 문서(source='docingest')에서 의미 기반 "
            "검색을 수행한다. 질의를 임베딩해 유사도 상위 청크를 반환한다."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 자연어 질의."},
                "top_k": {
                    "type": "integer",
                    "description": f"반환 결과 수(기본 {_DEFAULT_TOP_K}, 최대 {_MAX_TOP_K}).",
                },
            },
            "required": ["query"],
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        """
        질의 임베딩 → source='docingest' 로 필터한 벡터 검색.

        매개변수:
          arguments["query"]  — 검색할 자연어 질의(필수, 비어 있지 않은 문자열).
          arguments["top_k"]  — 반환 개수(선택, 기본 5, 최대 50).
        반환:
          {"query": 원질의, "count": 결과 수, "results": 청크 리스트} 형태 dict.
        예외:
          입력 검증 실패는 ValueError, 임베딩/검색 실패는 RuntimeError.
        """
        # 질의 검증 — 빈 문자열/비문자열이면 임베딩할 것이 없으므로 거부한다.
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("필수 인자 'query'(비어 있지 않은 문자열)가 없습니다.")

        # top_k 는 미지정 시에만 기본값 — `or` 관용구는 0 을 falsy 로 보아
        # 기본값으로 조용히 치환하므로, get(키, 기본값) 으로 명시적으로 처리한다.
        # (top_k=0/음수는 잘못된 입력이므로 기본값 대체가 아니라 거부해야 한다.)
        top_k = arguments.get("top_k", _DEFAULT_TOP_K)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k 는 1 이상의 정수여야 합니다.")
        top_k = min(top_k, _MAX_TOP_K)

        try:
            # e5 규약대로 "query: " 접두사를 붙여 임베딩한다. embed 는 리스트를
            # 받아 리스트를 돌려주므로, 한 문장이라도 [문장] 으로 감싸 호출한다.
            embeddings = await self._model.embed([f"{_QUERY_PREFIX}{query}"])
        except Exception as e:  # noqa: BLE001 — embed 예외 → RuntimeError 정규화
            raise RuntimeError(f"임베딩 실패: {type(e).__name__}: {e}") from e
        # 서버가 빈 리스트나 빈 벡터를 주면 이후 검색이 무의미하므로 조기 실패시킨다.
        if not embeddings or not embeddings[0]:
            raise RuntimeError("임베딩 서버가 빈 결과를 반환했습니다.")

        try:
            # source 인자로 docingest 청크만 필터(다른 소스 누설 방지).
            results = await self._store.search_by_vector(
                embeddings[0], top_k=top_k, source=_INGEST_SOURCE
            )
        except Exception as e:  # noqa: BLE001 — 검색 예외 → RuntimeError 정규화
            raise RuntimeError(f"벡터 검색 실패: {type(e).__name__}: {e}") from e

        # 호출 측이 원질의를 그대로 되돌려받고 결과 수도 바로 알 수 있게 함께 담아 준다.
        return {"query": query, "count": len(results), "results": results}


async def build_app(api_key: str = "local-key") -> FastAPI:
    """
    docingest MCP 서버 FastAPI 앱을 조립한다(이 파일의 진입점).

    이 함수 한 곳에서 필요한 부품을 전부 만들어 배선한 뒤 완성된 앱을 돌려준다.
    서버 기동 스크립트는 build_app() 을 await 해서 얻은 앱을 uvicorn 등으로 띄운다.

    매개변수:
      api_key — MCP 프레임워크가 요구하는 인증 키(LAN 내부용 기본값 "local-key").

    동작:
      1) core/config 로드(GPU 서버 URL, DB 접속 정보 등).
      2) ParserRegistry 에 PptxParser/PdfPlumberParser/HwpxParser/
         HwpViaLibreOfficeParser/TesseractParser 등록 + 호스트 능력에 따라
         PaddleOcrParser / DoclingParser(.pdf 고품질)를 더 높은 priority 로 등록.
      3) LocalModelProvider(embed) + KnowledgeStore(pg_pool) 구성
         (DB 연결 실패 시 인메모리로 폴백해 서버 자체는 뜨게 한다).
      4) DocumentIngestPipeline 조립 → parse/ingest/search 도구를 MCP 앱으로 등록.

    반환:
      create_mcp_app 이 만든 FastAPI 인스턴스(종료 훅으로 리소스 정리 포함).

    주의:
      무거운 core 모듈들은 서버 기동 시점에만 필요하므로 함수 안에서 지연 import
      한다(모듈 로드 비용 절감 + 순환 import 회피).
    """
    # --- 지연 import: 실제 서버를 조립할 때만 무거운 core 부품을 끌어온다 ---
    from core.config import load_and_validate_config
    from core.ingest.parser_base import ParserRegistry
    from core.ingest.parsers.docling_layout import DoclingParser
    from core.ingest.parsers.hwp_libreoffice import HwpViaLibreOfficeParser
    from core.ingest.parsers.hwpx import HwpxParser
    from core.ingest.parsers.ocr_tesseract import TesseractParser
    from core.ingest.parsers.pdf_plumber import PdfPlumberParser
    from core.ingest.parsers.pptx import PptxParser
    from core.ingest.pipeline import DocumentIngestPipeline
    from core.model.inference import LocalModelProvider
    from core.rag.knowledge_store import KnowledgeStore

    # 설정 로드 및 검증 — 이후 GPU/DB/모델 접속 정보를 여기서 꺼내 쓴다.
    config = load_and_validate_config()

    # 파서 레지스트리 — 청정(MIT/OWPML) 기본 파서들을 등록(어댑터 슬롯 구조).
    # 같은 확장자에 상용/고품질 파서를 더 높은 priority 로 끼우면 그것이 우선되고,
    # 미등록 시 자동으로 이 청정 기본 파서로 폴백된다(parser_base 우선순위 규칙).
    parser_registry = ParserRegistry()
    parser_registry.register(PptxParser(), priority=0)  # .pptx
    parser_registry.register(PdfPlumberParser(), priority=0)  # .pdf (경량, CPU)
    parser_registry.register(HwpxParser(), priority=0)  # .hwpx

    # v7.3 단계 9 — 구포맷 .hwp(LibreOffice headless 변환 경유, CPU).
    #  · .hwp 의 유일한 후보 파서다(HwpxParser 는 .hwpx 만 처리).
    #  · LibreOffice 가 .hwp → .docx 로 변환 후 python-docx 로 구조 보존 파싱.
    parser_registry.register(HwpViaLibreOfficeParser(), priority=0)  # .hwp

    # v7.3 단계 8 — Tesseract OCR(경량 CPU).
    #  · 이미지(.png/.jpg/.jpeg/.tiff): 유일한 후보이므로 1차 파서.
    #  · .pdf: 가장 낮은 priority(-10)로 등록 → 디지털 텍스트 파서
    #    (docling/pdfplumber)가 항상 우선하고, OCR 은 명시 호출/폴백용으로만 둔다.
    #    (상세 정책은 모듈 docstring "OCR 등록 정책" 참조.)
    parser_registry.register(TesseractParser(), priority=-10)

    # v7.3 단계 8 — PaddleOCR 고품질 OCR(어댑터 슬롯, requires_gpu=True).
    #  · GPU 가용 + paddleocr import 가능 호스트에서만 등록한다(둘 중 하나라도
    #    없으면 미등록 → 자동으로 Tesseract 경량 OCR 로 폴백).
    #  · priority=-5 로 Tesseract(-10)보다 높게 둔다 →
    #      - 이미지(.png/.jpg/.tiff): PaddleOCR 이 1차(고품질), Tesseract 폴백.
    #      - .pdf: 여전히 디지털 텍스트 파서(docling=10 / pdfplumber=0)가 먼저
    #        선택되고, OCR 끼리는 PaddleOCR 이 Tesseract 보다 앞선다. 즉 OCR 은
    #        .pdf 의 자동 대상이 아니라 명시 호출/폴백용으로만 남는다(Tesseract
    #        등록 정책과 동일 — 모듈 docstring "OCR 등록 정책" 참조).
    if _gpu_available() and _paddleocr_available():
        from core.ingest.parsers.ocr_paddle import PaddleOcrParser

        parser_registry.register(PaddleOcrParser(), priority=-5)  # 이미지/.pdf 고품질 OCR
        logger.info("docingest MCP 서버: GPU+paddleocr 감지 — PaddleOcrParser(고품질 OCR) 등록")
    else:
        logger.info("docingest MCP 서버: GPU/paddleocr 미가용 — TesseractParser(경량 OCR)만 사용")

    # .pdf 고품질 어댑터 슬롯(v7.3 단계 6) — GPU 가용 호스트에서만 우선 등록한다.
    # GPU 가 있으면 DoclingParser 를 priority=10 으로 등록해 pdfplumber(priority=0)
    # 보다 우선시키고, get_for_path 의 can_parse 폴백 덕분에 Docling 이 처리 못 하는
    # 파일은 자동으로 pdfplumber 로 넘어간다. GPU 가 없으면 아예 등록하지 않아
    # 경량 파서만 남긴다(requires_gpu=True 파서를 CPU 호스트에 얹지 않음).
    if _gpu_available():
        parser_registry.register(DoclingParser(), priority=10)  # .pdf (고품질, GPU)
        logger.info("docingest MCP 서버: GPU 감지 — DoclingParser(.pdf 고품질) 우선 등록")
    else:
        logger.info("docingest MCP 서버: GPU 미감지 — PdfPlumberParser(.pdf 경량)만 사용")

    # 임베딩 프로바이더(embed 전용) — 질의/문서를 벡터로 바꾸는 데만 쓴다.
    # 생성 LLM URL 과 임베딩 URL 을 따로 받는다(임베딩 모델은 별도 엔드포인트일 수
    # 있음). primary_model 은 여기서 직접 쓰이진 않지만 프로바이더 계약상 채워 준다.
    model_provider = LocalModelProvider(
        base_url=config.gpu_server.url,
        api_key=config.scout.api_key,
        model_id=config.model.primary_model,
        embedding_model_id=config.model.embedding_model,
        embedding_base_url=config.gpu_server.embedding_url,
    )

    # PostgreSQL 풀 — 실패 시 인메모리 폴백.
    # DB 가 없어도 서버는 떠야 하므로(개발/데모 환경 대비), 연결 실패를 흡수하고
    # pg_pool=None 으로 남긴다. None 이면 KnowledgeStore 가 인메모리 모드로 동작한다.
    pg_pool: Any | None = None
    try:
        import asyncpg

        pg_pool = await asyncpg.create_pool(
            host=config.postgresql.host,
            port=config.postgresql.port,
            database=config.postgresql.database,
            user=config.postgresql.user,
            password=config.postgresql.password,
            min_size=1,
            max_size=4,
            timeout=10.0,
        )
        logger.info("docingest MCP 서버: PostgreSQL 연결 성공")
    except ImportError:
        logger.warning("docingest MCP 서버: asyncpg 미설치 — 인메모리 폴백")
    except Exception as e:  # noqa: BLE001 — 연결 실패는 인메모리 폴백으로 흡수
        logger.warning("docingest MCP 서버: PostgreSQL 연결 실패(인메모리 폴백): %s", e)

    knowledge_store = KnowledgeStore(pg_pool=pg_pool)
    # pg_pool 이 있으면 스키마 멱등 생성(실 DB 에만 DDL 실행).
    # ensure_schema 는 테이블/인덱스가 없을 때만 만들므로 반복 호출해도 안전하다.
    if pg_pool is not None:
        await knowledge_store.ensure_schema()

    # 파서 레지스트리 + 임베딩 + 저장소를 하나로 묶은 인제스트 파이프라인.
    # parse/ingest 도구는 이 파이프라인 하나를 공유하며 dry_run 플래그로만 갈린다.
    pipeline = DocumentIngestPipeline(
        parser_registry=parser_registry,
        model_provider=model_provider,
        knowledge_store=knowledge_store,
    )

    async def _cleanup() -> None:
        # 서버 종료 시(shutdown 훅) 열려 있던 HTTP 클라이언트와 DB 풀을 닫아
        # 커넥션 누수를 막는다. create_mcp_app 에 넘겨 lifespan 종료 때 호출되게 한다.
        await model_provider.close()
        if pg_pool is not None:
            await pg_pool.close()
        logger.info("docingest MCP 서버: 리소스 정리 완료")

    # 3개 도구를 등록해 인증·라우팅이 붙은 MCP FastAPI 앱을 완성해 돌려준다.
    return create_mcp_app(
        tools=[
            DocParseTool(pipeline),
            DocIngestTool(pipeline),
            DocSearchTool(model_provider, knowledge_store),
        ],
        api_key=api_key,
        title="Nexus DocIngest MCP Server",
        shutdown_hooks=[_cleanup],
    )
