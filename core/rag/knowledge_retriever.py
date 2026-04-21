"""
지식 검색 → 프롬프트 주입 — KNOWLEDGE_MODE 전용 (Part 2.5.8, 2026-04-21).

사용자의 KNOWLEDGE 질의가 들어오면:
  1. 임베딩 서버로 질의를 벡터로 변환 (e5-large, 1024차원)
  2. tb_knowledge에서 코사인 유사도로 상위 K개 청크 검색
  3. 시스템 프롬프트에 "--- Knowledge base ---" 블록으로 주입
  4. Worker가 검색 결과를 근거로 답변 생성

설계 결정:
  - TOOL 모드에는 주입하지 않는다 — 도구 호출 흐름을 방해
  - 검색 실패/임베딩 오류 시 조용히 빈 문자열 반환 (본류 응답 무영향)
  - 기존 RAGRetriever(프로젝트 코드 인덱스)와 병행 사용 가능
  - 주입 예산은 호출자가 max_tokens 인자로 제어
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.model.inference import ModelProvider
    from core.rag.knowledge_store import KnowledgeStore

logger = logging.getLogger("nexus.rag.knowledge_retriever")


class KnowledgeRetriever:
    """tb_knowledge 검색 결과를 시스템 프롬프트용 텍스트로 정리한다."""

    def __init__(
        self,
        store: KnowledgeStore,
        embedding_provider: ModelProvider | None,
        top_k: int = 5,
        min_similarity: float = 0.3,
        chars_per_token: int = 3,
    ) -> None:
        self._store = store
        self._embedding = embedding_provider
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._chars_per_token = chars_per_token

    async def get_context(
        self,
        query: str,
        max_tokens: int = 1500,
        allowed_sources: list[str] | None = None,
    ) -> str:
        """
        질의에 관련된 지식 청크를 검색해 시스템 프롬프트용 문자열로 반환한다.

        allowed_sources가 주어지면 tb_knowledge.source 값을 그 목록으로 제한한다
        (멀티테넌시, Part 5 Ch 15). 빈 리스트 = 공통 적재만 허용해 폴백 차단이
        필요할 때 명시적으로 넘긴다. None = 필터 없음(전체).

        실패하거나 결과가 없으면 빈 문자열을 반환한다 (호출자가 무영향 분기).
        """
        if not query:
            return ""

        # 1) 벡터 검색 — 임베딩 서버가 살아있을 때만
        # allowed_sources는 DB-level 필터로 넘겨 cross-tenant 누설을 구조적으로 막는다
        results: list[dict] = []
        if self._embedding is not None:
            try:
                vecs = await self._embedding.embed([query])
                if vecs and vecs[0]:
                    results = await self._store.search_by_vector(
                        embedding=vecs[0],
                        top_k=self._top_k,
                        min_similarity=self._min_similarity,
                        allowed_sources=allowed_sources,
                    )
            except Exception as e:
                logger.warning("KnowledgeRetriever 벡터 검색 실패: %s", e)

        # 2) 텍스트 검색 폴백 (DB search_by_text는 단일 source만 받으므로 클라이언트
        #    측 필터로 보정 — 텍스트 검색은 주 경로 아님)
        if not results:
            try:
                results = await self._store.search_by_text(
                    query=query, top_k=self._top_k,
                )
                if allowed_sources is not None:
                    allowed_set = set(allowed_sources)
                    results = [r for r in results if r.get("source") in allowed_set]
            except Exception as e:
                logger.debug("KnowledgeRetriever 텍스트 검색 실패: %s", e)
                return ""

        if not results:
            return ""

        # 3) 토큰 예산 내에서 블록을 조립
        budget_chars = max_tokens * self._chars_per_token
        lines: list[str] = []
        used = 0
        for r in results:
            title = r.get("title", "(untitled)")
            section = r.get("section") or ""
            source = r.get("source", "")
            sim = r.get("similarity", 0)
            content = (r.get("content") or "").strip()
            if not content:
                continue
            section_part = f" · {section}" if section else ""
            header = f"[{source} · {title}{section_part} · sim={sim:.2f}]"
            block = f"{header}\n{content}"
            if used + len(block) > budget_chars:
                # 잘라서라도 하나 더 넣을지 — 여유 있으면 자르고 중단
                remaining = budget_chars - used
                if remaining > 200:
                    lines.append(block[:remaining] + " …")
                break
            lines.append(block)
            used += len(block) + 2  # 구분자 여유

        if not lines:
            return ""

        logger.debug(
            "KnowledgeRetriever: '%s...' → %d개 청크 주입 (~%d자)",
            query[:30], len(lines), used,
        )
        return "\n\n".join(lines)
