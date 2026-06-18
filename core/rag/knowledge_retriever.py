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
        # ── 유사도 게이팅 파라미터 (2026-06-18) ──────────────────────────
        # 왜 기본값이 "게이팅 무효"인가 (하위 호환):
        #   abs_threshold=0.0  → 어떤 유사도든 0.0 이상이라 절대 임계가 작동하지
        #                        않는다(= 전부 통과).
        #   relevance_margin=1.0 → top_sim에서 1.0(코사인 유사도 최대폭) 이내면
        #                          전부 남으므로 상대 마진도 작동하지 않는다.
        #   따라서 인자를 주지 않고 생성하면 종전과 100% 동일하게 동작한다.
        #   실제 게이팅 값(0.84/0.03)은 bootstrap이 config에서 주입한다.
        abs_threshold: float = 0.0,
        relevance_margin: float = 1.0,
        chars_per_token: int = 3,
    ) -> None:
        """게이팅 임계와 검색 파라미터를 보관한다.

        bootstrap이 config.knowledge_rag(yaml 단일 소스)에서 실제 값을 주입한다.
        인자를 생략하면 게이팅이 무효(전부 통과)라 종전 동작과 100% 동일하다 —
        위 abs_threshold/relevance_margin 주석의 "하위 호환" 설명 참조.
        chars_per_token: 토큰 예산을 글자수로 환산할 때 쓰는 1토큰≈3자 근사값.
        """
        self._store = store
        self._embedding = embedding_provider
        self._top_k = top_k
        self._min_similarity = min_similarity
        # 절대 임계: 최상위 결과조차 이 값 미만이면 "관련 자료 없음"으로 전체 드롭.
        self._abs_threshold = abs_threshold
        # 상대 마진: top_sim에서 이 폭 이내 결과만 유지(노이즈 청크 절단).
        self._relevance_margin = relevance_margin
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

        # ★ 유사도 2단 게이팅 (2026-06-18) — 무관 청크 주입 차단.
        #   왜 절대+상대 2단인가 (e5-large 분포 특성):
        #     실측상 e5-large 코사인 유사도는 "무관한 문서끼리"도 0.78~0.83에
        #     몰린다. 즉 절대 유사도 하나만으로는 관련/무관을 깔끔히 가를 수 없다.
        #       - 메타질문 "rag에 이런 정보가 있었어?" → top1=0.824 (무관)
        #       - "요한 제바스티안 바흐"                → top1=0.857 (관련)
        #     그래서 두 단계로 거른다:
        #       1) 절대 임계(abs_threshold): 최상위 결과조차 이 값보다 낮으면
        #          KB에 관련 자료가 없다는 뜻이므로 전부 드롭한다. 무관 분포
        #          상한(~0.83) 바로 위(0.84)에 두어 무관 질의는 통과하지 못한다.
        #       2) 상대 마진(relevance_margin): 최상위 유사도(top_sim)에서 이 폭
        #          이내로 떨어진 결과만 남긴다. top과 동떨어진 "끼어든 노이즈
        #          청크"를 잘라, 진짜 관련 높은 소수 청크만 모델에 보여준다.
        #   벡터 검색 결과는 distance ASC 정렬이라 results[0]이 최고 유사도다.
        #   유사도 값은 각 결과 dict의 "similarity" 키에 들어 있다.
        #   (기본값 abs_threshold=0.0 / relevance_margin=1.0이면 이 블록은
        #    아무것도 거르지 않으므로 하위 호환이 보장된다.)
        if results:
            top_sim = results[0].get("similarity", 0.0)
            if top_sim < self._abs_threshold:
                # 최상위조차 무관 → 전체 드롭 (아래에서 빈 문자열 반환)
                results = []
            else:
                results = [
                    r
                    for r in results
                    if r.get("similarity", 0.0) >= top_sim - self._relevance_margin
                ]

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

        # 3) 토큰 예산 내에서 주입 블록을 조립한다.
        #    게이팅을 통과한 청크를 유사도 높은 순(results는 distance ASC)으로
        #    하나씩 쌓되, max_tokens를 글자수로 환산한 예산(budget_chars)을 넘으면
        #    멈춘다. 왜 토큰이 아니라 글자수로 자르나: 여기서 토크나이저를 돌리면
        #    비용·지연이 커서, 1토큰≈3자 근사로 가볍게 상한만 건다(시스템 프롬프트
        #    팽창 방지). 각 블록은 출처 헤더 한 줄 + 본문으로 구성한다.
        budget_chars = max_tokens * self._chars_per_token
        lines: list[str] = []
        used = 0  # 지금까지 누적한 글자수 (구분자 여유 포함)
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
