"""
kowiki MCP 서버 — tb_knowledge(한국어 위키 등) 벡터 검색을 도구로 노출한다.

도구:
  search(query, top_k?) — 질의를 임베딩한 뒤 KnowledgeStore.search_by_vector 로
  코사인 유사도 검색을 수행하고 상위 결과(content/section/metadata/score)를 반환.

흐름 (core 재사용):
  ModelProvider.embed(["query: " + 질의]) → 1024차원 임베딩
    → KnowledgeStore.search_by_vector(embedding, top_k)
    → [{source, title, section, content, similarity, metadata, tags}, ...]

왜 "query:" 접두사인가:
  e5-large(multilingual) 권장 규약. 검색 질의는 "query:", 문서 청크는
  "passage:" 접두사를 붙여 임베딩 품질을 높인다(inference.py/pipeline.py 참고).

read-only:
  검색만 제공한다. 적재/수정 기능은 docingest 서버가 담당한다.

에어갭:
  임베딩 서버 URL/DB 접속 정보는 core/config 에서 LAN 주소로 로드. 외부 호출 없음.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI

from mcp_servers.framework import McpServerTool, create_mcp_app

logger = logging.getLogger("nexus.mcp_servers.kowiki")

# 검색 질의 임베딩 접두사(e5-large 규약).
_QUERY_PREFIX = "query: "

# 기본/최대 top_k — 과도한 결과로 토큰을 낭비하지 않도록 상한을 둔다.
_DEFAULT_TOP_K = 5
_MAX_TOP_K = 50


class KowikiSearchTool(McpServerTool):
    """
    tb_knowledge 벡터 검색 도구.

    임베딩 프로바이더와 KnowledgeStore 를 주입받아 검색을 수행한다. 둘 다
    core 의 구현(LocalModelProvider/KnowledgeStore)을 그대로 재사용한다.
    """

    def __init__(self, model_provider: Any, knowledge_store: Any) -> None:
        """
        Args:
            model_provider: embed(texts)->list[list[float]] 를 제공하는 프로바이더.
            knowledge_store: search_by_vector(...) 를 제공하는 KnowledgeStore.
        """
        self._model = model_provider
        self._store = knowledge_store

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "한국어 위키 등 적재된 지식 베이스(tb_knowledge)에서 의미 기반 검색을 수행한다. "
            "질의를 임베딩해 코사인 유사도가 높은 상위 청크를 반환한다(content/제목/섹션/점수)."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 자연어 질의.",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"반환할 결과 수(기본 {_DEFAULT_TOP_K}, 최대 {_MAX_TOP_K}).",
                },
            },
            "required": ["query"],
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        """
        질의를 임베딩해 벡터 검색을 수행한다.

        Returns:
            {
              "query": str,
              "count": int,
              "results": [
                {"title","section","content","source","similarity","tags","metadata"},
                ...
              ],
            }

        예외:
            query 누락 → ValueError(-32602).
            임베딩/검색 실패 → RuntimeError(-32603).
        """
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

        # 1) 질의 임베딩 — "query:" 접두사로 e5 규약을 따른다.
        try:
            embeddings = await self._model.embed([f"{_QUERY_PREFIX}{query}"])
        except Exception as e:  # noqa: BLE001 — embed 는 httpx 등 다양한 예외 → RuntimeError 로 정규화
            raise RuntimeError(f"임베딩 실패: {type(e).__name__}: {e}") from e

        if not embeddings or not embeddings[0]:
            raise RuntimeError("임베딩 서버가 빈 결과를 반환했습니다.")

        # 2) 벡터 검색 — KnowledgeStore.search_by_vector 재사용.
        try:
            results = await self._store.search_by_vector(embeddings[0], top_k=top_k)
        except Exception as e:  # noqa: BLE001 — asyncpg 등 예외 → RuntimeError 로 정규화
            raise RuntimeError(f"벡터 검색 실패: {type(e).__name__}: {e}") from e

        return {"query": query, "count": len(results), "results": results}


async def build_app(api_key: str = "local-key") -> FastAPI:
    """
    kowiki MCP 서버 FastAPI 앱을 조립한다.

    동작:
      1) core/config 로드 → 임베딩 서버 URL/PG 접속 정보 확보.
      2) LocalModelProvider(임베딩 전용) + KnowledgeStore(pg_pool) 구성.
         pg_pool 연결 실패 시 KnowledgeStore 인메모리 폴백으로 동작(fail-soft).
      3) KowikiSearchTool 등록한 앱 반환.
    """
    from core.config import load_and_validate_config
    from core.model.inference import LocalModelProvider
    from core.rag.knowledge_store import KnowledgeStore

    config = load_and_validate_config()

    # 임베딩 전용 프로바이더 — stream() 은 쓰지 않고 embed() 만 사용한다.
    model_provider = LocalModelProvider(
        base_url=config.gpu_server.url,
        api_key=config.scout.api_key,
        model_id=config.model.primary_model,
        embedding_model_id=config.model.embedding_model,
        embedding_base_url=config.gpu_server.embedding_url,
    )

    # PostgreSQL 풀 — 실패하면 None(KnowledgeStore 가 인메모리 폴백).
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
        logger.info("kowiki MCP 서버: PostgreSQL 연결 성공")
    except ImportError:
        logger.warning("kowiki MCP 서버: asyncpg 미설치 — 인메모리 폴백")
    except Exception as e:  # noqa: BLE001 — 연결 실패는 인메모리 폴백으로 흡수
        logger.warning("kowiki MCP 서버: PostgreSQL 연결 실패(인메모리 폴백): %s", e)

    knowledge_store = KnowledgeStore(pg_pool=pg_pool)

    async def _cleanup() -> None:
        # 임베딩 프로바이더의 httpx 클라이언트와 PG 풀을 정리한다.
        await model_provider.close()
        if pg_pool is not None:
            await pg_pool.close()
        logger.info("kowiki MCP 서버: 리소스 정리 완료")

    return create_mcp_app(
        tools=[KowikiSearchTool(model_provider, knowledge_store)],
        api_key=api_key,
        title="Nexus Kowiki RAG MCP Server",
        shutdown_hooks=[_cleanup],
    )
