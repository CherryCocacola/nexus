"""
RAG 검색기 — 쿼리와 관련된 문서 청크를 검색하여 컨텍스트로 반환한다.

RAG 파이프라인의 두 번째 단계: 사용자 쿼리를 임베딩하고 →
인덱싱된 청크에서 유사한 것을 찾아 → 토큰 예산 내에서 컨텍스트 문자열로 반환한다.

QueryEngine에서 자동으로 호출되어
시스템 프롬프트에 "--- Relevant files ---" 섹션을 추가한다.
이로써 모델이 전체 파일을 읽지 않고도 관련 내용을 참조할 수 있다.
"""

from __future__ import annotations

import logging
from typing import Any

from core.memory.types import MemoryEntry
from core.model.inference import ModelProvider

logger = logging.getLogger("nexus.rag.retriever")


class RAGRetriever:
    """
    인덱싱된 청크에서 쿼리와 관련된 것을 검색한다.

    검색 흐름:
      1. 쿼리를 e5-large로 임베딩
      2. LongTermMemory에서 벡터 유사도 검색
      3. 토큰 예산 내에서 상위 결과를 컨텍스트 문자열로 조합
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        memory_store: Any,  # LongTermMemory
    ) -> None:
        """
        Args:
            model_provider: 쿼리 임베딩 생성용
            memory_store: 인덱싱된 청크가 저장된 LongTermMemory
        """
        self._model_provider = model_provider
        self._memory = memory_store
        self._search_count: int = 0
        self._total_results: int = 0

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """
        쿼리와 관련된 청크를 검색한다.

        Args:
            query: 사용자 질문 또는 검색어
            top_k: 반환할 최대 결과 수

        Returns:
            유사도 높은 순서로 정렬된 MemoryEntry 리스트
        """
        self._search_count += 1

        try:
            # 쿼리를 임베딩으로 변환
            embeddings = await self._model_provider.embed([query])
            if not embeddings or not embeddings[0]:
                logger.warning("쿼리 임베딩 생성 실패")
                return []

            query_embedding = embeddings[0]

            # 벡터 유사도 검색 — RAG 태그가 있는 항목만 검색
            from core.memory.types import MemoryType

            results = await self._memory.search_by_vector(
                embedding=query_embedding,
                memory_type=MemoryType.SEMANTIC,
                top_k=top_k,
            )

            # RAG 인덱서가 생성한 항목만 필터링
            rag_results = [
                r for r in results
                if r.metadata.get("source") == "rag_indexer" or "rag" in r.tags
            ]

            self._total_results += len(rag_results)

            logger.debug(
                "RAG 검색: query='%s', 결과=%d개 (전체 %d개 중)",
                query[:50],
                len(rag_results),
                len(results),
            )

            return rag_results

        except Exception as e:
            logger.warning("RAG 검색 실패: %s", e)
            return []

    async def get_context(
        self,
        query: str,
        max_tokens: int = 1500,
        top_k: int = 5,
    ) -> str:
        """
        쿼리 관련 청크를 토큰 예산 내에서 컨텍스트 문자열로 반환한다.

        이 문자열이 시스템 프롬프트에 주입된다.

        Args:
            query: 사용자 질문
            max_tokens: 최대 토큰 수 (컨텍스트 내 RAG 영역)
            top_k: 검색할 최대 청크 수

        Returns:
            포맷된 컨텍스트 문자열. 결과 없으면 빈 문자열.
        """
        results = await self.search(query, top_k=top_k)
        if not results:
            return ""

        # 토큰 예산 내에서 청크를 조합한다
        parts: list[str] = []
        used_tokens = 0

        for entry in results:
            # 파일 경로와 청크 인덱스를 헤더로 추가
            file_path = entry.metadata.get("file_path", "unknown")
            chunk_idx = entry.metadata.get("chunk_index", 0)
            total_chunks = entry.metadata.get("total_chunks", 1)

            header = f"[{file_path} (chunk {chunk_idx + 1}/{total_chunks})]"
            chunk_text = f"{header}\n{entry.content}"

            # 토큰 추정 (문자수 / 3, 보수적)
            estimated = len(chunk_text) // 3
            if used_tokens + estimated > max_tokens:
                break

            parts.append(chunk_text)
            used_tokens += estimated

        if not parts:
            return ""

        context = "\n\n---\n\n".join(parts)
        logger.debug(
            "RAG 컨텍스트 생성: %d청크, ~%d토큰",
            len(parts),
            used_tokens,
        )
        return context

    @property
    def stats(self) -> dict[str, Any]:
        """검색 통계를 반환한다."""
        return {
            "search_count": self._search_count,
            "total_results": self._total_results,
        }
