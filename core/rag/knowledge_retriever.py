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
import re
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
        # 0.3→0.5 상향(★2 게이팅): kowiki 100만 청크에서는 어떤 질의든 약한
        # 유사도로 무언가가 잡힌다. 임계가 낮으면 무관한 청크(노이즈)가 주입돼
        # 오히려 "그럴듯한 오답"의 재료가 된다. 임계를 올려 의미적으로 관련된
        # 청크만 통과시키고, KB에 정답이 없는 질의는 0건 → "관련 자료 없음"으로
        # 흘러가 모델이 추측 대신 "모른다"고 답하도록 유도한다(할루시네이션 방지).
        min_similarity: float = 0.5,
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
        # 벡터 검색을 "실제로 끝까지 수행했는지" 추적하는 플래그.
        # 왜 필요한가:
        #   - 아래 폴백(search_by_text)은 content/title을 ILIKE로 1M행 전체
        #     풀스캔(Parallel Seq Scan)하므로 콜드 시 100초 이상 걸리는 시한폭탄이다.
        #   - 폴백의 본래 의도는 "임베딩 서버가 불능이라 벡터 검색을 못 했을 때"의
        #     최후 수단이다(아래 search_by_text 주석 참조).
        #   - 그런데 단순히 `if not results`로 분기하면, 임베딩이 멀쩡하고 벡터
        #     검색도 정상 수행됐지만 "관련 지식이 없어서 0건"인 경우까지 폴백에
        #     빠진다. 이 경우는 지식이 없는 것이므로 폴백할 이유가 없다.
        #   - 따라서 벡터 검색을 정상 수행했다면(True) 결과가 0건이어도 폴백을
        #     건너뛰고 빈 문자열을 반환한다. 임베딩 서버가 실제로 불능일 때만
        #     (False) ILIKE 폴백을 최후 수단으로 허용한다.
        vector_search_done = False
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
                    # 임베딩 생성 + 벡터 검색을 끝까지 마쳤다 → 0건이어도 폴백 불필요
                    vector_search_done = True
            except Exception as e:
                logger.warning("KnowledgeRetriever 벡터 검색 실패: %s", e)

        # 2) 텍스트 검색 폴백 — 벡터 검색을 아예 수행하지 못했을 때만 진입한다.
        #    (임베딩 서버 다운 / 빈 임베딩 / 임베딩 예외 등)
        #    벡터 검색이 정상 수행됐는데 0건이면 "관련 지식 없음"이므로 여기로 오지
        #    않고 아래에서 빈 문자열을 반환한다 — ILIKE 풀스캔 병목을 회피한다.
        #    (DB search_by_text는 단일 source만 받으므로 클라이언트 측 필터로 보정)
        if not results and not vector_search_done:
            try:
                results = await self._store.search_by_text(
                    query=query,
                    top_k=self._top_k,
                )
                if allowed_sources is not None:
                    allowed_set = set(allowed_sources)
                    results = [r for r in results if r.get("source") in allowed_set]
            except Exception as e:
                logger.debug("KnowledgeRetriever 텍스트 검색 실패: %s", e)
                return ""

        # ★ 엔티티(식별자) 매칭 게이팅 (옵션 A) — 부분 관련 함정 차단.
        #   왜 필요한가:
        #     "BWV 543"처럼 KB에 없는 특정 대상을 물으면, "바흐 일반" 청크가
        #     표면 유사도(min_similarity)만으로 잡혀 주입되고, 모델이 그 무관한
        #     청크를 근거로 그럴듯한 오답을 지어낸다(주제는 맞지만 그 작품은 아님).
        #   처리:
        #     질의에 구체 식별자(2~5자리 숫자: 카탈로그 번호·모델명·연도 등)가
        #     있으면, 그 숫자를 실제로 포함한 청크만 남긴다. 식별자가 어느 청크에도
        #     없으면 "특정 대상을 다루는 자료가 없다"는 뜻이므로 전부 드롭한다.
        #     → 0건이 되면 아래에서 빈 문자열을 반환하고, 호출자(prompt_assembler)가
        #       "관련 자료 없음"을 명시 주입해 모델이 추측 대신 "모른다"고 답하게 된다.
        #   순수 개념 질의(예: "광합성 원리")는 다자리 숫자가 없어 게이팅이
        #   적용되지 않으므로 정상 동작에 영향이 없다.
        identifiers = re.findall(r"\d{2,5}", query)
        if identifiers and results:
            results = [
                r
                for r in results
                if any(idn in (r.get("content") or "") for idn in identifiers)
            ]

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
            query[:30],
            len(lines),
            used,
        )
        return "\n\n".join(lines)
