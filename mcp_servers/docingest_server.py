"""
docingest MCP 서버 — 문서 인제스트 파이프라인을 도구로 노출한다.

도구 3개:
  parse  (read-only) — 파일을 파싱·청킹만 하고 적재하지 않는다(dry_run). 구조
                       트리 요약(제목/청크수/경고)을 반환한다.
  ingest (쓰기)      — 파일을 파싱→청킹→임베딩→tb_knowledge 적재(source="docingest").
  search (read-only) — 적재된 문서를 의미 기반 검색한다(source="docingest" 필터).

core 재사용:
  DocumentIngestPipeline(parse/chunk/embed/적재) + ParserRegistry + PptxParser +
  LocalModelProvider(embed) + KnowledgeStore 를 그대로 조립한다.

read-only 구분 (fail-closed 정신):
  parse/search 는 부작용이 없고, ingest 만 DB 에 쓴다. 도구 설명에 명시해 호출
  측(권한 파이프라인)이 구분할 수 있게 한다.

에어갭:
  임베딩/DB 접속은 LAN. 파일 경로는 서버 로컬 파일 시스템만 대상으로 한다.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI

from mcp_servers.framework import McpServerTool, create_mcp_app

logger = logging.getLogger("nexus.mcp_servers.docingest")

# docingest 가 적재하는 지식 소스 식별자 — pipeline.INGEST_SOURCE 와 동일해야 한다.
_INGEST_SOURCE = "docingest"

# 검색 질의 임베딩 접두사(e5-large 규약 — kowiki 서버와 동일).
_QUERY_PREFIX = "query: "
_DEFAULT_TOP_K = 5
_MAX_TOP_K = 50


class DocParseTool(McpServerTool):
    """파일을 파싱·청킹만 하고 적재하지 않는 read-only 도구(dry_run)."""

    def __init__(self, pipeline: Any) -> None:
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
        """dry_run=True 로 파이프라인을 돌려 적재 없이 요약만 반환한다."""
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("필수 인자 'path'(문자열)가 없습니다.")
        try:
            return await self._pipeline.ingest_file(path, dry_run=True)
        except (OSError, ValueError, RuntimeError) as e:
            raise RuntimeError(f"파싱 실패: {type(e).__name__}: {e}") from e


class DocIngestTool(McpServerTool):
    """파일을 파싱→청킹→임베딩→tb_knowledge 적재하는 쓰기 도구."""

    def __init__(self, pipeline: Any) -> None:
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
        """dry_run=False 로 실제 적재를 수행한다."""
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("필수 인자 'path'(문자열)가 없습니다.")
        try:
            return await self._pipeline.ingest_file(path, dry_run=False)
        except (OSError, ValueError, RuntimeError) as e:
            raise RuntimeError(f"적재 실패: {type(e).__name__}: {e}") from e


class DocSearchTool(McpServerTool):
    """적재된 docingest 문서를 벡터 검색하는 read-only 도구."""

    def __init__(self, model_provider: Any, knowledge_store: Any) -> None:
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
        """질의 임베딩 → source='docingest' 로 필터한 벡터 검색."""
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("필수 인자 'query'(비어 있지 않은 문자열)가 없습니다.")

        top_k = arguments.get("top_k") or _DEFAULT_TOP_K
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k 는 1 이상의 정수여야 합니다.")
        top_k = min(top_k, _MAX_TOP_K)

        try:
            embeddings = await self._model.embed([f"{_QUERY_PREFIX}{query}"])
        except Exception as e:  # noqa: BLE001 — embed 예외 → RuntimeError 정규화
            raise RuntimeError(f"임베딩 실패: {type(e).__name__}: {e}") from e
        if not embeddings or not embeddings[0]:
            raise RuntimeError("임베딩 서버가 빈 결과를 반환했습니다.")

        try:
            # source 인자로 docingest 청크만 필터(다른 소스 누설 방지).
            results = await self._store.search_by_vector(
                embeddings[0], top_k=top_k, source=_INGEST_SOURCE
            )
        except Exception as e:  # noqa: BLE001 — 검색 예외 → RuntimeError 정규화
            raise RuntimeError(f"벡터 검색 실패: {type(e).__name__}: {e}") from e

        return {"query": query, "count": len(results), "results": results}


async def build_app(api_key: str = "local-key") -> FastAPI:
    """
    docingest MCP 서버 FastAPI 앱을 조립한다.

    동작:
      1) core/config 로드.
      2) ParserRegistry 에 PptxParser 등록.
      3) LocalModelProvider(embed) + KnowledgeStore(pg_pool) 구성.
      4) DocumentIngestPipeline 조립 → parse/ingest/search 도구 등록.
    """
    from core.config import load_and_validate_config
    from core.ingest.parser_base import ParserRegistry
    from core.ingest.parsers.pptx import PptxParser
    from core.ingest.pipeline import DocumentIngestPipeline
    from core.model.inference import LocalModelProvider
    from core.rag.knowledge_store import KnowledgeStore

    config = load_and_validate_config()

    # 파서 레지스트리 — 청정(MIT) PptxParser 등록(어댑터 슬롯 구조).
    parser_registry = ParserRegistry()
    parser_registry.register(PptxParser(), priority=0)

    # 임베딩 프로바이더(embed 전용).
    model_provider = LocalModelProvider(
        base_url=config.gpu_server.url,
        api_key=config.scout.api_key,
        model_id=config.model.primary_model,
        embedding_model_id=config.model.embedding_model,
        embedding_base_url=config.gpu_server.embedding_url,
    )

    # PostgreSQL 풀 — 실패 시 인메모리 폴백.
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
    if pg_pool is not None:
        await knowledge_store.ensure_schema()

    pipeline = DocumentIngestPipeline(
        parser_registry=parser_registry,
        model_provider=model_provider,
        knowledge_store=knowledge_store,
    )

    async def _cleanup() -> None:
        await model_provider.close()
        if pg_pool is not None:
            await pg_pool.close()
        logger.info("docingest MCP 서버: 리소스 정리 완료")

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
