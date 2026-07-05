"""
RAG 검색기(Retriever) — 사용자 쿼리와 관련된 문서 청크를 찾아 컨텍스트 문자열로 만든다.

[이 파일이 하는 일 — 한눈에 보기]
RAG(Retrieval-Augmented Generation)는 "먼저 관련 자료를 찾아서(Retrieval)
그 자료를 근거로 답을 생성(Generation)한다"는 방식이다. 이 파일은 그중
'찾는(검색)' 역할을 담당한다. 즉 RAG 파이프라인의 두 번째 단계다.

전체 흐름(3단계):
  1. 사용자 쿼리를 e5-large 임베딩 모델로 벡터(숫자 배열)로 바꾼다.
  2. 장기 메모리(LongTermMemory, PostgreSQL + pgvector)에서 그 벡터와
     가장 비슷한 청크들을 유사도 검색으로 찾는다.
  3. 찾은 청크들을 토큰 예산 안에서 하나의 컨텍스트 문자열로 이어 붙인다.

[언제/누가 호출하나]
QueryEngine이 매 요청마다 자동으로 이 검색기를 호출한다. 그 결과 문자열은
시스템 프롬프트에 "--- Relevant files ---" 형태의 섹션으로 주입된다.
덕분에 모델은 파일 전체를 직접 읽지 않고도 관련 내용을 참고해 답할 수 있다.
(파일을 통째로 읽으면 토큰 낭비가 크므로, 필요한 조각만 골라 넣는 것이 핵심.)

[주요 구성 요소]
  - RAGRetriever: 검색기 본체. search()로 청크를 찾고 get_context()로 문자열화.

[의존 모듈]
  - core.memory.types.MemoryEntry / MemoryType: 저장된 청크의 데이터 모델·분류.
  - core.model.inference.ModelProvider: 쿼리 임베딩 생성(embed) 담당.
  - 실제 벡터 검색은 memory_store(LongTermMemory)의 search_by_vector()에 위임.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from typing import Any

from core.memory.types import MemoryEntry
from core.model.inference import ModelProvider

logger = logging.getLogger("nexus.rag.retriever")


class RAGRetriever:
    """
    인덱싱된 청크(문서 조각)에서 쿼리와 관련된 것을 골라내는 검색기 클래스.

    별도의 인덱서(RAG indexer)가 미리 문서를 잘게 쪼개(청크) 임베딩과 함께
    장기 메모리에 저장해 둔다. 이 클래스는 그렇게 '적재된 것'을 '읽어서 찾는'
    쪽만 담당한다(적재/색인은 이 파일의 책임이 아니다).

    검색 흐름(내부 메서드 기준):
      1. search()      — 쿼리를 e5-large로 임베딩 → 벡터 유사도 검색 → 필터링.
      2. get_context() — search() 결과를 토큰 예산 안에서 하나의 문자열로 조합.
      3. stats         — 지금까지 몇 번 검색했고 몇 건을 찾았는지 통계 조회.

    상태 필드(_search_count, _total_results)는 운영 중 모니터링/디버깅용
    누적 카운터이며, 검색 로직 자체에는 영향을 주지 않는다.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        memory_store: Any,  # LongTermMemory
    ) -> None:
        """
        검색기를 초기화한다. 외부에서 의존성(모델 제공자·메모리 저장소)을
        주입받아 보관만 한다 — 여기서 네트워크 호출 등 무거운 작업은 하지 않는다.

        Args:
            model_provider: 쿼리 문자열을 벡터로 바꾸는 임베딩 제공자.
                (search()에서 embed()를 호출할 때 사용)
            memory_store: 인덱싱된 청크가 저장된 LongTermMemory 인스턴스.
                실제 벡터 유사도 검색(search_by_vector)을 여기에 위임한다.
                타입 힌트가 Any인 이유는 순환 import를 피하기 위함이다.
        """
        # 주입받은 의존성을 인스턴스 필드로 보관(앞의 밑줄은 '내부용'이라는 관례).
        self._model_provider = model_provider
        self._memory = memory_store
        # 누적 통계 카운터 — 검색 호출 횟수와 지금까지 찾은 결과 총합.
        self._search_count: int = 0
        self._total_results: int = 0

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """
        쿼리와 관련된 청크를 벡터 유사도로 검색한다.

        이 메서드는 '어떤 청크가 관련 있는가'만 반환한다. 그 청크들을 실제
        프롬프트용 문자열로 다듬는 일은 get_context()가 담당한다.

        실패에 강하게(fail-safe) 설계되어 있다: 임베딩이 비었거나 도중에
        어떤 예외가 나더라도 예외를 위로 던지지 않고 빈 리스트를 반환한다.
        RAG는 '있으면 도움이 되는' 보조 기능이므로, 검색이 실패해도 전체
        대화 흐름은 멈추지 않아야 하기 때문이다.

        Args:
            query: 사용자 질문 또는 검색어(원문 문자열).
            top_k: 벡터 검색에서 가져올 최대 후보 수.

        Returns:
            유사도가 높은 순서로 정렬된 MemoryEntry 리스트.
            결과가 없거나 오류가 나면 빈 리스트([]).
        """
        # 검색 호출 횟수를 1 증가(통계용). 실패해도 '시도했다'는 사실은 센다.
        self._search_count += 1

        try:
            # 1) 쿼리 문자열을 임베딩 벡터로 변환한다.
            #    embed()는 리스트를 받아 리스트를 돌려주므로 [query]로 감싼다.
            embeddings = await self._model_provider.embed([query])
            # 결과가 비었거나 첫 벡터가 비어 있으면 검색을 진행할 수 없다.
            if not embeddings or not embeddings[0]:
                logger.warning("쿼리 임베딩 생성 실패")
                return []

            # 우리가 넣은 쿼리는 하나뿐이므로 첫 번째 벡터만 사용한다.
            query_embedding = embeddings[0]

            # 2) 벡터 유사도 검색.
            #    MemoryType.SEMANTIC(의미 기억)으로 한정해 검색 범위를 좁힌다.
            #    (지역 import: 순환 import 방지 + 필요한 시점에만 로드)
            from core.memory.types import MemoryType

            results = await self._memory.search_by_vector(
                embedding=query_embedding,
                memory_type=MemoryType.SEMANTIC,
                top_k=top_k,
            )

            # 3) SEMANTIC 안에는 RAG 외의 의미 기억도 섞여 있을 수 있다.
            #    그래서 'RAG 인덱서가 만든 청크'만 골라낸다. 판별 기준은 둘 중
            #    하나라도 만족하면 통과: metadata.source == "rag_indexer" 이거나
            #    tags 안에 "rag"가 포함된 경우.
            rag_results = [
                r for r in results
                if r.metadata.get("source") == "rag_indexer" or "rag" in r.tags
            ]

            # 필터링 후 실제로 채택된 결과 수를 통계에 누적한다.
            self._total_results += len(rag_results)

            # 디버깅용 로그: 쿼리 앞 50자, 필터 통과 수, 검색 원본 수를 남긴다.
            logger.debug(
                "RAG 검색: query='%s', 결과=%d개 (전체 %d개 중)",
                query[:50],
                len(rag_results),
                len(results),
            )

            return rag_results

        except Exception as e:
            # 어떤 예외든 삼키고 빈 리스트를 반환한다(위 docstring의 fail-safe
            # 설계 참고). 원인 파악을 위해 경고 로그는 남긴다.
            logger.warning("RAG 검색 실패: %s", e)
            return []

    async def get_context(
        self,
        query: str,
        max_tokens: int = 1500,
        top_k: int = 5,
    ) -> str:
        """
        쿼리 관련 청크를 검색하고, 토큰 예산 안에서 하나의 컨텍스트 문자열로 만든다.

        search()가 '무엇을' 찾을지 정한다면, 이 메서드는 그 결과를 '어떻게'
        프롬프트에 넣을지를 정한다. 반환된 문자열은 그대로 시스템 프롬프트에
        주입되어 모델이 참고할 근거가 된다.

        토큰 예산(max_tokens)을 두는 이유: RAG 내용이 너무 길면 정작 중요한
        대화 내용이 컨텍스트 창에서 밀려나기 때문이다. 그래서 상위 결과부터
        차곡차곡 담다가 예산을 넘기면 그 청크는 넣지 않고 멈춘다.

        Args:
            query: 사용자 질문(원문 문자열).
            max_tokens: RAG 영역에 허용되는 최대 토큰 수(대략적 상한).
            top_k: search()에 넘길 검색 후보 최대 수.

        Returns:
            포맷된 컨텍스트 문자열. 검색 결과가 없거나 예산상 하나도 담지
            못하면 빈 문자열("").
        """
        # 먼저 관련 청크를 검색한다. 결과가 없으면 조합할 것도 없으니 조기 반환.
        results = await self.search(query, top_k=top_k)
        if not results:
            return ""

        # 토큰 예산 안에서 청크를 순서대로 담을 버퍼와 누적 토큰 카운터.
        parts: list[str] = []
        used_tokens = 0

        for entry in results:
            # 각 청크 위에 '출처 헤더'를 붙인다. 모델이 어느 파일의 몇 번째
            # 조각인지 알 수 있어야 인용/판단에 도움이 되기 때문이다.
            # 메타데이터가 없을 때를 대비해 get()에 기본값을 준다.
            file_path = entry.metadata.get("file_path", "unknown")
            chunk_idx = entry.metadata.get("chunk_index", 0)
            total_chunks = entry.metadata.get("total_chunks", 1)

            # chunk_idx는 0부터 시작하므로 사람이 읽기 좋게 +1 해서 표기한다.
            header = f"[{file_path} (chunk {chunk_idx + 1}/{total_chunks})]"
            chunk_text = f"{header}\n{entry.content}"

            # 토큰 수를 문자수/3으로 보수적으로 추정한다(정확한 토크나이저를
            # 돌리지 않는 대신 넉넉히 잡아 예산 초과를 방지). 이 청크를 더하면
            # 예산을 넘기는 경우, 뒤 청크도 볼 필요 없이 즉시 중단한다.
            estimated = len(chunk_text) // 3
            if used_tokens + estimated > max_tokens:
                break

            parts.append(chunk_text)
            used_tokens += estimated

        # 예산이 너무 빡빡해 첫 청크조차 못 담았다면 빈 문자열을 반환한다.
        if not parts:
            return ""

        # 담긴 청크들을 구분선("---")으로 이어 하나의 문자열로 만든다.
        context = "\n\n---\n\n".join(parts)
        logger.debug(
            "RAG 컨텍스트 생성: %d청크, ~%d토큰",
            len(parts),
            used_tokens,
        )
        return context

    @property
    def stats(self) -> dict[str, Any]:
        """
        지금까지의 검색 통계를 딕셔너리로 반환한다(모니터링/디버깅용).

        Returns:
            search_count  — search()가 호출된 총 횟수.
            total_results — 필터링을 통과해 채택된 청크 수의 누적 합.
        """
        return {
            "search_count": self._search_count,
            "total_results": self._total_results,
        }
