"""
지식 검색 → 프롬프트 주입 — KNOWLEDGE_MODE 전용 (Part 2.5.8, 2026-04-21).

이 모듈이 하는 일 (한눈에 보기):
  사용자가 "지식 기반(KB)에 근거해 답해줘" 유형의 질의를 던지면, 그 질의와
  의미적으로 관련된 문서 청크를 tb_knowledge에서 찾아 시스템 프롬프트에 끼워
  넣어준다. 그러면 Worker(LLM)가 자기 기억이 아니라 "실제로 주입된 근거"를
  바탕으로 답을 쓰게 되어, 없는 사실을 지어내는 할루시네이션이 줄어든다.

동작 순서 (KNOWLEDGE 질의가 들어왔을 때):
  1. 임베딩 서버로 질의를 벡터로 변환 (e5-large, 1024차원)
  2. tb_knowledge에서 코사인 유사도로 상위 K개 청크 검색
  3. (옵션) 크로스인코더 리랭킹 / 유사도 2단 게이팅 / 엔티티 게이팅 / MMR로
     무관하거나 중복된 청크를 걸러내 "진짜 관련된 소수 청크"만 남긴다
  4. 토큰 예산 안에서 "[출처 · 제목 · 점수] 본문" 블록으로 조립해 반환
  5. 호출자(prompt_assembler)가 그 블록을 시스템 프롬프트에 주입
  6. Worker가 주입된 근거를 바탕으로 답변 생성

주요 구성:
  - KnowledgeRetriever : 이 파일의 유일한 공개 클래스. 검색·필터·조립을 담당.
      · get_context()  : 외부에 노출되는 진입점. 질의 → 주입용 문자열.
      · _mmr_select()  : 내부 헬퍼. MMR로 관련도+다양성 상위 top_k 선별.

주요 의존 (호출 대상):
  - KnowledgeStore   : tb_knowledge에 대한 벡터/텍스트 검색 (DB 접근 계층)
  - ModelProvider    : 임베딩 서버 embed() / 리랭커 rerank() 호출
  - pgvector_base.cosine_similarity : MMR에서 청크끼리의 코사인 유사도 계산

설계 결정 (왜 이렇게 만들었나):
  - TOOL 모드에는 주입하지 않는다 — 도구 호출 흐름을 방해
  - 검색 실패/임베딩 오류 시 조용히 빈 문자열 반환 (본류 응답 무영향)
  - 기존 RAGRetriever(프로젝트 코드 인덱스)와 병행 사용 가능
  - 주입 예산은 호출자가 max_tokens 인자로 제어
  - 리랭커·MMR·게이팅은 기본 비활성(하위 호환)이고, 실제 값은 bootstrap이
    config.knowledge_rag(yaml 단일 소스)에서 주입한다

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

# 출처 인용 데이터 모델(KnowledgeCitation)은 core/message.py에 둔다 — StreamEvent가
# 이를 필드로 참조하기 때문이다. core/rag → core/message는 순방향 import라 의존성
# 방향 규칙(P2)을 지킨다(역방향이면 순환 import 위험).
from core.message import KnowledgeCitation
from core.rag.pgvector_base import cosine_similarity as _cosine

if TYPE_CHECKING:
    from core.model.inference import ModelProvider
    from core.rag.knowledge_store import KnowledgeStore

logger = logging.getLogger("nexus.rag.knowledge_retriever")


class KnowledgeContext(BaseModel):
    """get_context_with_citations()의 반환 — 주입 텍스트 + 출처 목록.

    출처 인용(Point 4-2)을 위해 도입한 반환 타입이다. 기존 get_context()는
    조립된 문자열 하나만 돌려줘서 상위 계층이 "어떤 청크가 주입됐는지"를 알 수
    없었다. 이 모델은 그 두 정보를 함께 담아 웹 응답의 sources 필드로 노출할 수
    있게 한다. frozen 불변(도메인 규칙 P5).

    필드:
      text      : 프롬프트 주입용 조립 문자열(기존 get_context 반환과 동일 형식).
      citations : 실제로 주입된 청크의 출처 목록. 순서 = index 순서.
                  citation 비활성이거나 주입 청크가 없으면 빈 튜플.
    """

    model_config = {"frozen": True}

    text: str = ""
    citations: tuple[KnowledgeCitation, ...] = ()


class KnowledgeRetriever:
    """tb_knowledge 검색 결과를 시스템 프롬프트용 텍스트로 정리한다.

    이 클래스가 필요한 이유:
      벡터 검색만으로는 "주제는 비슷하지만 실제로는 무관한 청크"가 함께 딸려와
      모델이 그럴듯한 오답을 짓는 재료가 된다. 그래서 검색(recall) 뒤에 여러
      단계의 필터(정밀도, precision)를 겹쳐 잡음을 걷어내고, 최종적으로 토큰
      예산 안에 들어가는 근거 텍스트만 조립해 돌려주는 역할을 한다.

    필터 파이프라인 (get_context에서 이 순서로 적용):
      1. (옵션) 크로스인코더 리랭킹 — (질의, 청크) 관련도를 직접 0~1로 점수화
      2. 유사도 2단 게이팅       — e5 코사인 분포용 절대+상대 임계 (리랭커 시 대체)
      3. 엔티티(식별자) 게이팅    — 질의 속 숫자 식별자를 포함한 청크만 유지
      4. (옵션) MMR 다양성 선별   — 관련도 높으면서 서로 다른 청크로 top_k 구성

    상태(생성자에서 주입받아 보관):
      검색·게이팅·리랭킹·MMR 파라미터를 __init__에서 받아 self._* 로 들고 있다가
      get_context 호출 시마다 재사용한다. 기본값은 모두 "무효(하위 호환)"이며
      실제 운영 값은 bootstrap이 config.knowledge_rag에서 주입한다.
    """

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
        # ── MMR(Maximal Marginal Relevance) 리랭킹 파라미터 (2026-07-05) ────
        # 왜 기본값이 "MMR 비활성"인가 (하위 호환):
        #   순수 벡터 유사도 top_k는 서로 비슷비슷한(중복) 청크를 함께 뽑아
        #   컨텍스트를 낭비하고 커버리지가 좁다. MMR은 게이팅을 통과한 후보
        #   중에서 "관련도는 높으면서 서로 다른(다양한)" 청크를 골라 이 문제를
        #   완화한다. 다만 검증 전 무회귀가 최우선이므로 기본은 꺼둔다.
        #   mmr_enabled=False면 검색을 종전대로 top_k/임베딩미포함으로 수행하고
        #   MMR 단계를 통째로 건너뛰어 동작이 현재와 100% 동일하다.
        #   (게이팅 파라미터가 abs=0.0/margin=1.0 기본으로 무효화되는 것과 동일
        #    패턴 — 실제 값은 bootstrap이 config.knowledge_rag.mmr에서 주입.)
        mmr_enabled: bool = False,
        mmr_fetch_k: int = 20,
        mmr_lambda: float = 0.7,
        # ── 크로스인코더 리랭커 파라미터 (2026-07-05) ─────────────────────────
        # 왜 기본값이 "리랭커 비활성"인가 (하위 호환):
        #   rerank_enabled=False면 벡터검색 인자·게이팅·조립이 종전과 100% 동일하고
        #   리랭커를 호출하지 않는다 → 동작이 현재와 완전히 같다(회귀 0).
        #   B200 임베딩 서버 /v1/rerank로 (질의, 각 청크) 관련도를 직접 점수화(0~1)해
        #   재정렬한다. 켜지면 e5 코사인 분포용 2단 유사도 게이팅
        #   (abs_threshold/relevance_margin)은 rerank 점수 게이팅(min_score)이
        #   대체하고, 엔티티(식별자) 게이팅은 그대로 유지된다.
        #   (게이팅/MMR 기본값이 무효화되는 것과 동일 패턴 — 실제 값은 bootstrap이
        #    config.knowledge_rag.rerank에서 주입.)
        rerank_enabled: bool = False,
        rerank_fetch_k: int = 20,
        rerank_top_k: int = 5,
        rerank_min_score: float = 0.3,
        rerank_min_similarity: float = 0.6,
        # ivfflat 재현율 튜닝 — search_by_vector에 probes로 전달.
        ivfflat_probes: int = 40,
        # ── 출처 인용(Point 4-2, 2026-07-08) ───────────────────────────────
        # 왜 기본값이 "인용 비활성"인가 (하위 호환):
        #   citation_enabled=False면 청크 헤더·조립·반환이 종전과 100% 동일하고
        #   citations도 비어(()) 있어 상위(assembler/web)가 아무 것도 노출하지 않는다
        #   → 동작이 현재와 완전히 같다(회귀 0). 실제 값은 bootstrap이 config.
        #   knowledge_rag.citation에서 주입한다(게이팅/MMR/리랭커와 동일 패턴).
        #   활성 시 청크 헤더에 `[{label}{N} | ...]` 번호를 '실제 주입된 청크에만'
        #   부여하고, 그 청크들의 출처 메타를 KnowledgeContext.citations로 돌려준다.
        citation_enabled: bool = False,
        citation_label: str = "출처",
    ) -> None:
        """게이팅 임계와 검색 파라미터를 보관한다.

        bootstrap이 config.knowledge_rag(yaml 단일 소스)에서 실제 값을 주입한다.
        인자를 생략하면 게이팅이 무효(전부 통과)이고 MMR도 비활성이라 종전 동작과
        100% 동일하다 — 위 abs_threshold/relevance_margin/mmr_enabled 주석의
        "하위 호환" 설명 참조.
        chars_per_token: 토큰 예산을 글자수로 환산할 때 쓰는 1토큰≈3자 근사값.
        """
        self._store = store
        self._embedding = embedding_provider
        self._top_k = top_k
        self._min_similarity = min_similarity
        # ivfflat.probes — 벡터검색 재현율(search_by_vector에 전달). lists=1000 인덱스에서
        # 기본 probes=1은 정답 청크를 후보에서 놓치므로 config에서 상향(기본 40).
        self._ivfflat_probes = ivfflat_probes
        # 절대 임계: 최상위 결과조차 이 값 미만이면 "관련 자료 없음"으로 전체 드롭.
        self._abs_threshold = abs_threshold
        # 상대 마진: top_sim에서 이 폭 이내 결과만 유지(노이즈 청크 절단).
        self._relevance_margin = relevance_margin
        self._chars_per_token = chars_per_token
        # MMR 리랭킹 설정 (기본 비활성 — 위 주석의 하위 호환 근거 참조).
        self._mmr_enabled = mmr_enabled
        # MMR 후보 풀 크기: 이 개수만큼 넉넉히 가져와(top_k보다 크게) 그 안에서
        # 다양성 선별로 최종 top_k를 고른다. 후보가 많을수록 다양성 여지가 커진다.
        self._mmr_fetch_k = mmr_fetch_k
        # 관련도(λ) vs 다양성(1-λ) 균형. 0.7=관련도 우선(약간의 다양성 가미).
        self._mmr_lambda = mmr_lambda
        # 크로스인코더 리랭커 설정 (기본 비활성 — 위 주석의 하위 호환 근거 참조).
        self._rerank_enabled = rerank_enabled
        # 리랭킹 후보 풀: 벡터검색을 이 개수만큼 넉넉히(리콜 그물 완화) 가져와
        # 크로스인코더로 재정렬한다. top_k보다 크게 둘수록 재현율이 오른다.
        self._rerank_fetch_k = rerank_fetch_k
        # 재정렬·게이팅 후 최종적으로 프롬프트에 넣을 청크 개수.
        self._rerank_top_k = rerank_top_k
        # rerank 점수 절대 게이팅: 최고 점수가 이 값 미만이면 전체 드롭(관련 자료 없음).
        self._rerank_min_score = rerank_min_score
        # 리랭킹용 벡터검색 1차 컷오프(리콜 그물을 넓히려 기존 min_similarity보다 완화).
        self._rerank_min_similarity = rerank_min_similarity
        # 출처 인용 설정 (기본 비활성 — 위 주석의 하위 호환 근거 참조).
        self._citation_enabled = citation_enabled
        # 라벨 접두어("출처1", "Source1" 등). 하드코딩 금지 규칙(#4)에 따라 config화.
        self._citation_label = citation_label

    @property
    def citation_enabled(self) -> bool:
        """출처 인용 활성 여부(읽기 전용) — 상위(prompt_assembler)가 trailer 판단에 쓴다."""
        return self._citation_enabled

    @property
    def citation_label(self) -> str:
        """출처 라벨 접두어(읽기 전용) — 인용 trailer 문구를 헤더와 같은 라벨로 맞춘다."""
        return self._citation_label

    async def get_context(
        self,
        query: str,
        max_tokens: int = 1500,
        allowed_sources: list[str] | None = None,
    ) -> str:
        """질의에 관련된 지식 청크를 시스템 프롬프트용 '문자열'로 반환한다.

        하위 호환 얇은 래퍼 — 실제 로직은 get_context_with_citations()에 있고,
        여기서는 그 결과의 .text만 돌려준다. 시그니처·반환 타입(str)이 종전과
        동일해 기존 호출부·테스트가 그대로 동작한다(무회귀). 출처 목록(citations)
        까지 필요한 상위 계층은 get_context_with_citations()를 직접 호출한다.
        """
        result = await self.get_context_with_citations(
            query, max_tokens, allowed_sources
        )
        return result.text

    async def get_context_with_citations(
        self,
        query: str,
        max_tokens: int = 1500,
        allowed_sources: list[str] | None = None,
    ) -> KnowledgeContext:
        """
        질의에 관련된 지식 청크를 검색해 (주입 텍스트 + 출처 목록)으로 반환한다.

        이 메서드가 클래스의 실질 진입점이다. 내부 흐름은 크게 5단계다:
          (1) 벡터 검색 → (2) 텍스트 폴백(임베딩 불능일 때만) →
          (3) 리랭킹/게이팅/엔티티/MMR 필터 → (4) top_k 컷 →
          (5) 토큰 예산 내 블록 조립(+ 인용 활성 시 [출처N] 번호 부착).

        매개변수:
          query          : 사용자 질의 원문. 빈 문자열이면 즉시 빈 컨텍스트 반환.
          max_tokens     : 주입 예산(토큰). chars_per_token으로 글자수로 환산해
                           그 상한까지만 청크를 쌓는다(기본 1500).
          allowed_sources: tb_knowledge.source 화이트리스트(멀티테넌시, Part 5
                           Ch 15). None=필터 없음(전체). 빈 리스트=공통 적재만
                           허용(폴백 차단이 필요할 때 명시적으로 넘긴다). DB-level
                           필터로 넘겨 cross-tenant 누설을 구조적으로 막는다.

        반환:
          KnowledgeContext(text, citations).
          - text: "[출처 · 제목 · 점수]\n본문" 블록들을 빈 줄로 이어 붙인 문자열
            (citation 활성 시 헤더에 "[출처N | " 번호 접두 추가).
          - citations: 실제로 주입된 청크의 출처 목록(citation 활성 시에만 채움).
          검색 실패·임베딩 오류·관련 청크 0건 등 넣을 근거가 없으면 text=""·
          citations=()인 빈 컨텍스트를 반환한다. 호출자는 빈 text를 "관련 자료
          없음"으로 처리해 모델이 추측 대신 "모른다"고 답하도록 유도한다.
        """
        if not query:
            return KnowledgeContext()

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
                    # 검색 파라미터 선택 — 우선순위: 리랭커 > MMR > 종전(하위 호환).
                    #   - 리랭커 활성: 리콜 그물을 넓게(rerank_fetch_k) + 완화된 1차
                    #     컷오프(rerank_min_similarity)로 가져온다. 크로스인코더가 그
                    #     위에서 정밀 재정렬하므로 1차는 느슨해도 된다. 임베딩은 서버가
                    #     직접 점수화하므로 되받을 필요 없음(with_embedding=False).
                    #   - MMR 활성(리랭커 비활성): 넓은 후보 풀(mmr_fetch_k)을 임베딩과
                    #     함께 가져와 다양성 선별한다.
                    #   - 둘 다 비활성: 종전대로 top_k만·임베딩 미포함(전송/파싱 비용 무증가).
                    if self._rerank_enabled:
                        fetch_k = self._rerank_fetch_k
                        min_sim = self._rerank_min_similarity
                        with_emb = False
                    elif self._mmr_enabled:
                        fetch_k = self._mmr_fetch_k
                        min_sim = self._min_similarity
                        with_emb = True
                    else:
                        fetch_k = self._top_k
                        min_sim = self._min_similarity
                        with_emb = False
                    results = await self._store.search_by_vector(
                        embedding=vecs[0],
                        top_k=fetch_k,
                        min_similarity=min_sim,
                        allowed_sources=allowed_sources,
                        with_embedding=with_emb,
                        probes=self._ivfflat_probes,
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
                return KnowledgeContext()

        # ★ 크로스인코더 리랭킹 (2026-07-05) — 임베딩 성공 경로에서만.
        #   왜 여기(게이팅 앞)인가:
        #     리랭커는 (질의, 각 청크)의 관련도를 크로스인코더로 직접 점수화(0~1)한다.
        #     e5 코사인 유사도보다 관련/무관을 훨씬 선명하게 가른다(실측: 관련 0.9~1.0,
        #     무관 0.0, 경계 ~0.28). 그래서 리랭커가 켜지면 아래 e5 2단 유사도 게이팅
        #     (abs_threshold/relevance_margin — e5 분포 전용 임계)은 '건너뛰고',
        #     rerank 점수 게이팅(min_score)이 그 역할을 대체한다. 단, 엔티티(식별자)
        #     게이팅은 rerank와 무관한 별개 안전장치이므로 아래에서 그대로 유지한다.
        #   적용 조건(하위 호환·fail-safe):
        #     - rerank_enabled일 때만. 비활성이면 이 블록 전체를 건너뛰어 종전과 동일.
        #     - vector_search_done(임베딩 성공 경로)일 때만. 임베딩 실패→ILIKE 폴백
        #       경로에서는 리랭킹을 스킵하고 기존 게이팅으로 처리한다.
        #     - rerank 호출이 예외를 던지면(서버 미로드/네트워크) 리랭킹을 통째로
        #       건너뛰고(rerank_applied=False) 기존 벡터순+기존 게이팅으로 폴백한다
        #       (fail-safe, warning 로깅).
        rerank_applied = False
        if self._rerank_enabled and vector_search_done and results:
            try:
                contents = [(r.get("content") or "") for r in results]
                scores = await self._embedding.rerank(query, contents)
                # 서버 계약: documents와 동일 길이·순서. 어긋나면 정렬이 오정렬되므로
                # 예외로 처리해 아래 except의 fail-safe 폴백으로 넘긴다.
                if len(scores) != len(results):
                    raise ValueError(
                        f"rerank 점수 개수 불일치: {len(scores)} != {len(results)}"
                    )
                # 각 후보에 rerank 점수 부착 후 내림차순 재정렬.
                for r, s in zip(results, scores, strict=True):
                    r["rerank_score"] = float(s)
                results.sort(
                    key=lambda r: r.get("rerank_score", 0.0), reverse=True
                )
                # rerank 절대 게이팅: 최고 점수조차 min_score 미만이면 전체 드롭
                # (→ "" 반환 → "관련 자료 없음"). 그 외 min_score 이상만 유지한다.
                if results[0].get("rerank_score", 0.0) < self._rerank_min_score:
                    results = []
                else:
                    results = [
                        r
                        for r in results
                        if r.get("rerank_score", 0.0) >= self._rerank_min_score
                    ]
                rerank_applied = True
            except Exception as e:
                # fail-safe: 리랭킹 실패 시 기존 벡터순+기존 게이팅으로 폴백한다.
                logger.warning(
                    "KnowledgeRetriever 리랭킹 실패 — 벡터순+기존 게이팅 폴백: %s", e
                )

        # ★ 유사도 2단 게이팅 (2026-06-18) — 무관 청크 주입 차단.
        #   (리랭커 적용 시 rerank_applied=True → 이 블록을 건너뛴다. rerank 점수
        #    게이팅이 e5 분포 전용인 abs_threshold/relevance_margin을 대체하기 때문.)
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
        if results and not rerank_applied:
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
            return KnowledgeContext()

        # ★ 리랭커 적용 시 최종 top_k 컷 — 재정렬·게이팅·엔티티게이팅을 통과한
        #   상위 rerank_top_k개만 남겨 토큰예산 조립부로 넘긴다(조립 로직은 재사용).
        if rerank_applied:
            results = results[: self._rerank_top_k]

        # ★ MMR(Maximal Marginal Relevance) 다양성 선별 (2026-07-05)
        #   위치(중요): 모든 게이팅(2단 유사도 + 엔티티) '이후'에 놓인다. 즉
        #   MMR은 게이팅을 통과한 survivors 중에서만 최종 top_k를 고르는 단계로,
        #   할루시네이션 게이팅을 훼손하거나 우회하지 않는다.
        #   왜 필요한가: 순수 벡터 top_k는 서로 비슷한(중복) 청크를 함께 뽑아
        #   컨텍스트를 낭비하고 커버리지가 좁다. MMR로 "관련도 높으면서 서로 다른"
        #   청크를 골라 컨텍스트를 절약하고 커버리지를 넓힌다.
        #   비활성(mmr_enabled=False)이면 이 블록을 통째로 건너뛰어 종전과 동일.
        #   (임베딩이 없는 폴백 경로 search_by_text에는 embedding 키가 없어
        #    _mmr_select가 다양성 항을 0으로 처리 → 사실상 관련도 순 유지로 안전.)
        #   ── 리랭커와 동시 활성이면 리랭커 우선(MMR 스킵): 리랭커가 관련도 자체를
        #      크로스인코더로 직접 다루므로, 그 위에 MMR을 겹쳐 재정렬하지 않는다.
        if self._mmr_enabled and not rerank_applied:
            results = self._mmr_select(results)
            # 임베딩 키는 주입 텍스트 조립 전에 제거한다 — 프롬프트 오염 방지 및
            # 불필요한 메모리 점유 해소(다운스트림 헤더/예산 로직은 이 키를 안 씀).
            for r in results:
                r.pop("embedding", None)

        # 3) 토큰 예산 내에서 주입 블록을 조립한다.
        #    게이팅을 통과한 청크를 유사도 높은 순(results는 distance ASC)으로
        #    하나씩 쌓되, max_tokens를 글자수로 환산한 예산(budget_chars)을 넘으면
        #    멈춘다. 왜 토큰이 아니라 글자수로 자르나: 여기서 토크나이저를 돌리면
        #    비용·지연이 커서, 1토큰≈3자 근사로 가볍게 상한만 건다(시스템 프롬프트
        #    팽창 방지). 각 블록은 출처 헤더 한 줄 + 본문으로 구성한다.
        budget_chars = max_tokens * self._chars_per_token
        lines: list[str] = []
        used = 0  # 지금까지 누적한 글자수 (구분자 여유 포함)
        # 출처 인용(Point 4-2) — 실제로 주입된 청크에만 번호를 부여하기 위해 이
        # 조립 루프 안에서 citations를 만든다. citation_idx는 '주입 확정된' 청크
        # 개수(=다음에 붙일 [출처N]의 N-1). 비활성이면 citations는 끝까지 빈 채다.
        citations: list[KnowledgeCitation] = []
        citation_idx = 0
        for r in results:
            title = r.get("title", "(untitled)")
            section = r.get("section") or ""
            source = r.get("source", "")
            sim = r.get("similarity", 0)
            content = (r.get("content") or "").strip()
            if not content:
                continue
            section_part = f" · {section}" if section else ""
            # 리랭킹된 청크는 헤더에 rerank 점수(rr=)를 표기한다. 그 외에는 종전대로
            # e5 코사인 유사도(sim=)를 쓴다(리랭킹 안 된 경로는 rerank_score 키 없음).
            rr = r.get("rerank_score")
            score_part = f"rr={rr:.2f}" if rr is not None else f"sim={sim:.2f}"
            # 이번 청크가 받게 될 출처 번호(주입이 확정될 때만 실제로 커밋한다).
            next_index = citation_idx + 1
            if self._citation_enabled:
                # 인용 활성: 기존 헤더를 대체하지 않고 "[출처N | " 번호만 앞에 덧붙여
                # 기존 e2e(광합성·바흐)와의 시각적 차이를 최소화한다.
                header = (
                    f"[{self._citation_label}{next_index} | "
                    f"{source} · {title}{section_part} · {score_part}]"
                )
            else:
                header = f"[{source} · {title}{section_part} · {score_part}]"
            block = f"{header}\n{content}"
            # 이번 청크가 실제로 주입되면(전체/부분) 그 출처 메타를 만들어 둔다
            # (citation 비활성이면 None → 아래에서 커밋하지 않는다).
            citation = (
                KnowledgeCitation(
                    index=next_index,
                    source=source,
                    title=title,
                    # 섹션은 빈 문자열이면 None으로 정규화(모델 필드 계약).
                    section=section or None,
                    # 점수는 rerank 우선, 없으면 similarity(헤더 표기와 동일 기준).
                    score=float(rr) if rr is not None else float(sim),
                    # tb_knowledge.id(감사·추적용). SELECT에 없으면 None.
                    chunk_id=r.get("id"),
                )
                if self._citation_enabled
                else None
            )

            if used + len(block) > budget_chars:
                # 잘라서라도 하나 더 넣을지 — 여유 있으면 자르고 중단
                remaining = budget_chars - used
                if remaining > 200:
                    # 잘린 청크도 헤더(번호 포함)와 함께 실제로 주입되므로 인용을 커밋한다.
                    lines.append(block[:remaining] + " …")
                    citation_idx = next_index
                    if citation is not None:
                        citations.append(citation)
                break
            lines.append(block)
            citation_idx = next_index
            if citation is not None:
                citations.append(citation)
            used += len(block) + 2  # 구분자 여유

        if not lines:
            return KnowledgeContext()

        logger.debug(
            "KnowledgeRetriever: '%s...' → %d개 청크 주입 (~%d자, citations=%d)",
            query[:30],
            len(lines),
            used,
            len(citations),
        )
        return KnowledgeContext(text="\n\n".join(lines), citations=tuple(citations))

    # ─────────────────────────────────────────────
    # MMR(Maximal Marginal Relevance) 선별 — 순수 파이썬 구현
    # ─────────────────────────────────────────────
    def _mmr_select(self, candidates: list[dict]) -> list[dict]:
        """게이팅을 통과한 후보 중 MMR로 '관련도+다양성' 상위 top_k를 고른다.

        MMR 점수:
            score(c) = λ·rel(c) − (1−λ)·max( cos(c, s) for s in selected )
          - rel(c)   : 질의-후보 유사도 = c["similarity"] (검색 단계에서 이미 계산)
          - cos(c,s) : 후보끼리의 코사인 유사도(임베딩 정규화 내적, _cosine 사용)
          - λ(mmr_lambda): 관련도 vs 다양성 균형. 1.0이면 순수 관련도(=종전),
                           0.0이면 순수 다양성. 0.7=관련도 우선.

        동작:
          - 후보 수가 top_k 이하이면 선별할 것이 없어 그대로 반환한다(스킵).
          - 첫 선택은 관련도 최대. candidates는 유사도 내림차순이라 [0]이 최대다
            (게이팅에서 results[0]=최고 유사도 기준을 그대로 승계).
          - 이후 매 단계, 남은 후보 중 위 score가 최대인 것을 하나씩 뽑아
            top_k개까지 채운다.
          - 임베딩이 없는 후보(폴백 경로 등)는 다양성 항(cos)을 0으로 처리해
            순수 관련도만으로 다룬다(안전한 열화 — MMR이 오작동하지 않음).

        반환:
          다운스트림(헤더/토큰예산 조립)이 기대하는 '유사도 내림차순'으로
          재정렬해 돌려준다. 선택 자체는 다양성 기준이지만, 표시 순서는 관련도순.
        """
        k = self._top_k
        # 후보가 뽑을 개수 이하이면 다양성 선별의 의미가 없다 → 그대로 통과.
        if len(candidates) <= k:
            return candidates

        lam = self._mmr_lambda
        remaining = list(candidates)
        # 첫 선택: 관련도 최대(= 유사도 내림차순 정렬의 선두).
        selected: list[dict] = [remaining.pop(0)]

        while remaining and len(selected) < k:
            best_idx = 0
            best_score: float | None = None
            for i, cand in enumerate(remaining):
                rel = float(cand.get("similarity", 0.0))
                # 이미 뽑힌 것들과의 최대 유사도(=중복도). 임베딩 없으면 0으로 둔다.
                max_sim = 0.0
                emb_c = cand.get("embedding")
                if emb_c:
                    for sel in selected:
                        emb_s = sel.get("embedding")
                        if emb_s:
                            s = _cosine(emb_c, emb_s)
                            if s > max_sim:
                                max_sim = s
                # 관련도는 높이고(+λ·rel) 중복도는 낮추도록(−(1−λ)·max_sim) 점수화.
                score = lam * rel - (1.0 - lam) * max_sim
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        # 표시 순서는 유사도 내림차순으로 통일(헤더 sim 표기·예산 로직 일관성).
        selected.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)
        return selected
