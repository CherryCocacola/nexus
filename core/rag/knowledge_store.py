"""
지식 베이스 — tb_knowledge 기반 장기 교양 지식 저장소 (Part 2.5.8, 2026-04-21).

■ 이 파일이 하는 일 (한눈에)
  외부 지식(위키백과 덤프, 교재, 매뉴얼 등)을 잘게 쪼갠 "청크"로 만들고,
  각 청크의 임베딩 벡터를 PostgreSQL + pgvector 테이블(tb_knowledge)에 저장한다.
  사용자가 질문하면 질문 임베딩과 가장 가까운 청크를 코사인 유사도로 찾아
  모델 프롬프트에 주입한다(= 검색 증강 생성, RAG). 이렇게 하면 모델이 학습
  때 못 본 지식도 근거를 갖고 답할 수 있어 할루시네이션이 줄어든다.

■ 주요 구성 요소
  - KnowledgeEntry : 청크 1건(제목/본문/임베딩 등)을 담는 불변 데이터클래스.
  - KnowledgeStore : tb_knowledge에 add(쓰기)/search(검색)/list(요약)하는 래퍼.
                     DB가 없으면 인메모리 dict로 폴백해 테스트에서도 동작한다.
  - split_into_chunks : 긴 원문을 임베딩하기 좋은 크기로 쪼개는 순수 함수.
  - _inmemory_search  : DB 없이 파이썬으로 코사인 유사도를 직접 계산하는 폴백.

배경:
  Phase 3 LoRA가 도구 호출을 강화한 대가로 일반 지식 표현이 좁아졌고,
  Part 2.5 라우팅으로 KNOWLEDGE_MODE(베이스 Qwen)에서는 한결 정확해졌다.
  그러나 베이스 Qwen도 한국어 인문학 깊이는 한정적이므로, 외부 지식
  베이스(위키백과 등)를 pgvector로 인덱싱하여 검색-주입하는 RAG 파이프라인을
  추가한다.

설계 결정:
  - 기존 tb_memories와 별도 테이블 — EPISODIC(대화) 데이터와 혼재하지 않음
  - 단방향 의존성: rag → memory/types 없음, pgvector만 공유
  - 에어갭 준수: 외부 네트워크 호출 없음. 덤프 다운로드는 scripts/에 격리
  - 차원 1024 (e5-large multilingual)

작성자: 이현수 / 작성일: 2026-07-05

스키마:
  CREATE TABLE tb_knowledge (
    id            text PRIMARY KEY,
    source        text NOT NULL,        -- 'kowiki', 'textbook', 'manual' 등
    title         text NOT NULL,        -- 문서 제목
    section       text,                 -- 섹션 이름 (선택)
    content       text NOT NULL,        -- 청크 본문
    chunk_index   int  NOT NULL DEFAULT 0,
    total_chunks  int  NOT NULL DEFAULT 1,
    tags          text[] DEFAULT '{}',
    embedding     vector(1024),
    created_at    timestamptz NOT NULL DEFAULT now(),
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb
  );
  CREATE INDEX idx_knowledge_source ON tb_knowledge(source);
  CREATE INDEX idx_knowledge_title  ON tb_knowledge(title);
  CREATE INDEX idx_knowledge_tags   ON tb_knowledge USING GIN(tags);
  CREATE INDEX idx_knowledge_embed  ON tb_knowledge
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

# 벡터 저장 공통 기능은 pgvector_base에 모아두고 여기서 상속/재사용한다.
#  - PgVectorStore : DB 연결 보관, ensure_schema/build_vector_index/count 등 공통 로직
#  - cosine_similarity : 두 벡터의 코사인 유사도(인메모리 폴백 검색에서 사용)
#  - format_vector : float 리스트 -> pgvector가 이해하는 "[0.1,0.2,...]" 문자열
#  - parse_vector  : 위 문자열 -> float 리스트 (검색 결과 임베딩 되돌릴 때)
from core.rag.pgvector_base import PgVectorStore
from core.rag.pgvector_base import cosine_similarity as _cosine
from core.rag.pgvector_base import format_vector as _format_vector
from core.rag.pgvector_base import parse_vector as _parse_vector

logger = logging.getLogger("nexus.rag.knowledge_store")


# tb_knowledge 테이블/인덱스 생성 SQL.
# ensure_schema()가 "IF NOT EXISTS" 덕분에 몇 번을 실행해도 안전(멱등적)하게 돌린다.
# CREATE EXTENSION vector 는 pgvector 확장을 켜서 vector 타입을 쓸 수 있게 한다.
_DDL_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tb_knowledge (
    id            text PRIMARY KEY,
    source        text NOT NULL,
    title         text NOT NULL,
    section       text,
    content       text NOT NULL,
    chunk_index   int  NOT NULL DEFAULT 0,
    total_chunks  int  NOT NULL DEFAULT 1,
    tags          text[] DEFAULT '{}',
    embedding     vector(1024),
    created_at    timestamptz NOT NULL DEFAULT now(),
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_knowledge_source ON tb_knowledge(source);
CREATE INDEX IF NOT EXISTS idx_knowledge_title  ON tb_knowledge(title);
CREATE INDEX IF NOT EXISTS idx_knowledge_tags   ON tb_knowledge USING GIN(tags);
"""

# ivfflat 은 근사 최근접 이웃(ANN) 벡터 인덱스다. 이 인덱스는 데이터 분포를 보고
# 클러스터(lists)를 나누기 때문에, 행이 어느 정도 쌓인 뒤에 만들어야 품질이 좋다.
# 그래서 스키마 DDL과 분리해 두고, 적재가 끝난 다음 build_vector_index()로 실행한다.
_DDL_IVFFLAT = """
CREATE INDEX IF NOT EXISTS idx_knowledge_embed ON tb_knowledge
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


@dataclass(frozen=True)
class KnowledgeEntry:
    """단일 지식 청크 — tb_knowledge 테이블의 한 행(row)과 1:1로 대응된다.

    frozen=True 이므로 한 번 만들면 값을 바꿀 수 없다(불변). 이렇게 하면
    적재 파이프라인 여러 곳에서 같은 객체를 공유해도 실수로 수정될 위험이 없다.

    필드 설명:
      source       : 출처 식별자. 'kowiki', 'textbook', 'manual' 등.
                     검색 시 테넌트/도메인 필터로도 쓰인다.
      title        : 원문 문서 제목.
      content      : 이 청크의 실제 본문 텍스트(모델에 주입되는 근거).
      section      : 문서 내 섹션 이름(선택). 없으면 None.
      chunk_index  : 한 문서를 여러 청크로 쪼갰을 때 이 청크의 순번(0부터).
      total_chunks : 그 문서가 총 몇 조각으로 나뉘었는지.
      tags         : 분류용 태그 튜플(예: 카테고리).
      embedding    : 본문을 임베딩한 1024차원 벡터. 아직 임베딩 전이면 None.
      metadata     : 자유 형식 부가 정보(원문 URL 등)를 담는 dict.
    """

    source: str
    title: str
    content: str
    section: str | None = None
    chunk_index: int = 0
    total_chunks: int = 1
    tags: tuple[str, ...] = ()
    embedding: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """
        결정론적(deterministic) 기본키 ID를 만들어 반환한다.

        source/title/section/chunk_index 4개를 이어붙인 문자열의 SHA-256
        해시 앞 32자를 ID로 쓴다. 같은 청크를 몇 번 다시 적재해도 입력이
        같으면 결과 ID도 같으므로, add()의 UPSERT(ON CONFLICT)와 맞물려
        중복 행이 생기지 않고 덮어쓰기 된다(= 멱등성 보장).
        """
        # 4개 필드를 파이프(|)로 이어붙여 유일 키를 만든다. section이 None이면 빈 문자열.
        key = f"{self.source}|{self.title}|{self.section or ''}|{self.chunk_index}"
        # SHA-256 해시(64자 hex) 중 앞 32자만 사용 — 충돌 확률이 극히 낮아 충분하다.
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class KnowledgeStore(PgVectorStore):
    """tb_knowledge 기반 지식 베이스 — add(쓰기)/search(검색)/list(요약) 래퍼.

    이 클래스가 RAG의 "검색기(retriever)" 뒤편 저장소 역할을 한다.
    상위(예: KnowledgeRetriever)가 이 클래스의 search_by_vector 등을 호출한다.

    DB 연결(self._pg)이 있으면 실제 PostgreSQL에 질의하고, None이면 인메모리
    dict(self._store)로 폴백한다. 덕분에 DB 없는 단위 테스트/개발 환경에서도
    같은 API로 동작한다. (self._pg / self._store 는 부모 PgVectorStore가 준비.)

    장기 메모리(tb_memories, 대화 기록)와는 물리적으로 다른 테이블을 쓴다.
    대용량 지식 데이터가 대화 히스토리 검색에 섞여 품질을 떨어뜨리지 않게 하려는 것.

    스키마 생성/벡터 인덱스 빌드/행 수 세기(ensure_schema, build_vector_index,
    count) 같은 공통 로직은 부모 PgVectorStore에서 그대로 상속받는다.
    """

    # PgVectorStore 계약(약속) — 부모가 이 세 값을 읽어 DDL 실행과 COUNT 쿼리에 쓴다.
    # 자식 클래스는 "어느 테이블에 어떤 DDL을 쓸지"만 채워주면 된다.
    TABLE_NAME = "tb_knowledge"
    DDL_SCHEMA = _DDL_SCHEMA
    DDL_IVFFLAT = _DDL_IVFFLAT

    # ─── 쓰기 ───────────────────────────────────────
    async def add(self, entry: KnowledgeEntry) -> str:
        """청크 1건을 저장(UPSERT)하고 그 id를 반환한다.

        UPSERT = INSERT 시도하되, 같은 id가 이미 있으면 UPDATE로 덮어쓴다.
        entry.id가 결정론적이므로 같은 청크 재적재 시 중복 없이 최신화된다.
        DB가 없으면 인메모리 dict에 그대로 넣는다.
        """
        memory_id = entry.id
        if self._pg is None:
            # 폴백 경로 — DB 대신 메모리 dict에 저장하고 끝낸다.
            self._store[memory_id] = entry
            return memory_id

        # 임베딩 튜플을 pgvector가 받는 "[..]" 문자열로 변환($9::vector 로 캐스팅).
        embedding_str = _format_vector(entry.embedding)
        async with self._pg.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tb_knowledge (
                    id, source, title, section, content,
                    chunk_index, total_chunks, tags, embedding, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    section = EXCLUDED.section,
                    chunk_index = EXCLUDED.chunk_index,
                    total_chunks = EXCLUDED.total_chunks,
                    tags = EXCLUDED.tags,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                memory_id,
                entry.source,
                entry.title,
                entry.section,
                entry.content,
                entry.chunk_index,
                entry.total_chunks,
                list(entry.tags),
                embedding_str,
                json.dumps(entry.metadata, ensure_ascii=False),
            )
        return memory_id

    async def add_many(self, entries: list[KnowledgeEntry]) -> int:
        """여러 청크를 한 번에 적재하고, 적재한 건수를 반환한다.

        내부적으로는 단건 add()를 순서대로 반복할 뿐이다(특별한 배치 SQL 아님).
        대량 적재 파이프라인(prepare_kowiki.py 등)이 편하게 호출하는 헬퍼.
        """
        count = 0
        for e in entries:
            await self.add(e)
            count += 1
        return count

    # ─── 검색 ───────────────────────────────────────
    async def search_by_vector(
        self,
        embedding: list[float],
        top_k: int = 5,
        source: str | None = None,
        allowed_sources: list[str] | None = None,
        min_similarity: float = 0.2,
        with_embedding: bool = False,
        probes: int | None = None,
    ) -> list[dict[str, Any]]:
        """코사인 유사도 기반 벡터 검색.

        Args:
            embedding: 질의 임베딩 벡터(1024차원)
            top_k: 반환 건수
            source: 특정 소스로 제한 (예: "kowiki") — 단일값 편의 인자
            allowed_sources: 멀티테넌시 — 허용된 source 목록(`source IN (...)`).
                DB-level 필터로 cross-tenant 누설을 구조적으로 차단한다.
                source와 동시 지정 시 allowed_sources가 우선.
            min_similarity: 최소 유사도 (1 - distance). 낮은 품질은 자동 제거.
            with_embedding: True면 각 결과 dict에 파싱된 임베딩("embedding":
                list[float], 1024차원)을 함께 담는다. MMR 리랭킹처럼 후보간
                유사도를 계산해야 할 때만 켠다.
                기본 False면 SELECT에 embedding을 포함하지 않아 전송/파싱
                오버헤드가 0이고 반환 dict도 종전과 100% 동일하다(하위 호환).

        Returns:
            [{"title", "section", "content", "source", "similarity", "tags"}, ...]
            유사도 내림차순. with_embedding=True면 각 dict에 "embedding" 키 추가.
        """
        # 인자가 두 갈래(source 단일 / allowed_sources 목록)라 아래 로직이 복잡해지지
        # 않도록, 단일 source를 "원소 1개짜리 목록"으로 바꿔 한 갈래로 정규화한다.
        if allowed_sources is None and source is not None:
            allowed_sources = [source]

        if self._pg is None:
            # DB가 없으면 파이썬으로 코사인 유사도를 직접 계산하는 폴백 경로로 위임.
            return _inmemory_search(
                self._store, embedding, top_k, allowed_sources, min_similarity,
                with_embedding=with_embedding,
            )

        vec_str = _format_vector(embedding)
        # pgvector의 <=> 연산자는 "코사인 거리"(0=완전동일, 2=정반대)를 준다.
        # 거리 = 1 - 유사도 이므로, 유사도 하한(min_similarity)을 거리 상한으로 변환.
        max_distance = 1.0 - min_similarity

        # with_embedding일 때만 SELECT에 embedding 컬럼을 추가한다. 기본(False)이면
        # 종전 컬럼 집합 그대로라 전송량·파싱 비용이 늘지 않는다(하위 호환).
        embed_col = ", embedding" if with_embedding else ""

        # 쿼리 파라미터를 $1,$2,... 순서대로 리스트에 쌓는다(SQL 인젝션 방지 바인딩).
        # $1=질의벡터, $2=거리상한, $3=top_k. source 필터가 있으면 $4를 추가한다.
        params: list[Any] = [vec_str, max_distance, top_k]
        where = "WHERE embedding <=> $1::vector <= $2"
        if allowed_sources:
            # 허용 소스로 DB 단에서 걸러 cross-tenant 누설을 구조적으로 차단.
            where += " AND source = ANY($4::text[])"
            params.append(list(allowed_sources))

        query = f"""
            SELECT id, source, title, section, content, tags, metadata{embed_col},
                   (embedding <=> $1::vector) AS distance
            FROM tb_knowledge
            {where}
            ORDER BY distance ASC
            LIMIT $3
        """  # noqa: S608 — where 절/컬럼은 whitelist 기반(embed_col은 내부 bool 분기)

        # 커넥션 풀에서 연결 하나를 빌려(acquire) 쿼리를 실행하고 자동 반납한다.
        async with self._pg.acquire() as conn:
            if probes is not None:
                # ivfflat 근사검색 재현율 — SET LOCAL로 이 트랜잭션에만 probes를 적용한다
                # (풀의 다른 쿼리에 영향 없음). probes가 낮으면(기본 1) 스캔 리스트가
                # 적어 정답 청크를 후보에서 놓친다. int 캐스팅으로 SQL 인젝션 차단.
                async with conn.transaction():
                    await conn.execute(f"SET LOCAL ivfflat.probes = {int(probes)}")
                    rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch(query, *params)

        # DB 행(Record)들을 호출자가 다루기 쉬운 평범한 dict 리스트로 변환한다.
        out: list[dict[str, Any]] = []
        for r in rows:
            item: dict[str, Any] = {
                # id — tb_knowledge 기본키(출처 인용 chunk_id·감사/추적용). 반환 dict에
                # 키 하나가 늘 뿐이라 기존 소비자(키 존재를 가정 안 함)에 하위 호환.
                "id": r["id"],
                "source": r["source"],
                "title": r["title"],
                "section": r["section"],
                "content": r["content"],
                "tags": list(r["tags"] or []),
                # 거리를 다시 유사도(1 - 거리)로 되돌려 사람이 읽기 쉽게 소수 4자리 반올림.
                "similarity": round(1.0 - float(r["distance"]), 4),
                "metadata": r["metadata"] or {},
            }
            if with_embedding:
                # pgvector는 문자열("[...]")로 오므로 float 리스트로 파싱한다.
                # 파싱 실패 시 None → 다운스트림(MMR)이 임베딩 없는 후보로 스킵.
                item["embedding"] = _parse_vector(r["embedding"])
            out.append(item)
        return out

    async def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """단순 ILIKE(대소문자 무시 부분일치) 기반 텍스트 검색 — 벡터 검색의 폴백.

        의미 기반 벡터 검색이 훨씬 정확하지만, 임베딩 서버가 다운됐거나 질의가
        너무 짧아 임베딩이 무의미할 때 "그래도 뭔가는 찾아주는" 최후 수단이다.
        의미가 아니라 글자 그대로 포함 여부만 보므로 similarity는 0.5 상수로 채운다.

        Args:
            query: 검색어(문자열). content 또는 title에 이 문자열이 들어가면 매칭.
            top_k: 최대 반환 건수.
            source: 특정 소스로 한정할 때 지정(선택).

        Returns:
            search_by_vector와 같은 형태의 dict 리스트(단, similarity=0.5 고정).
        """
        if self._pg is None:
            # DB 폴백 — 메모리 dict를 순회하며 소문자 부분일치로 직접 필터링.
            needle = query.lower()
            results: list[dict[str, Any]] = []
            for e in self._store.values():
                if source and e.source != source:
                    continue
                if needle in e.content.lower() or needle in e.title.lower():
                    results.append({
                        "source": e.source,
                        "title": e.title,
                        "section": e.section,
                        "content": e.content,
                        "tags": list(e.tags),
                        "similarity": 0.5,  # 휴리스틱 상수
                        "metadata": e.metadata,
                    })
                    if len(results) >= top_k:
                        break
            return results

        # ILIKE 패턴은 앞뒤에 %를 붙여 "어디든 포함"을 의미하게 만든다.
        # $1=검색패턴, $2=top_k, (필요 시) $3=source.
        params: list[Any] = [f"%{query}%", top_k]
        where = "WHERE (content ILIKE $1 OR title ILIKE $1)"
        if source is not None:
            where += " AND source = $3"
            params.append(source)

        q = f"""
            SELECT source, title, section, content, tags, metadata
            FROM tb_knowledge
            {where}
            ORDER BY created_at DESC
            LIMIT $2
        """  # noqa: S608

        async with self._pg.acquire() as conn:
            rows = await conn.fetch(q, *params)
        return [
            {
                "source": r["source"], "title": r["title"], "section": r["section"],
                "content": r["content"], "tags": list(r["tags"] or []),
                "similarity": 0.5, "metadata": r["metadata"] or {},
            }
            for r in rows
        ]

    # ─── 운영 유틸 ──────────────────────────────────
    # count()는 PgVectorStore에서 상속한다.

    async def list_sources(self) -> list[dict[str, Any]]:
        """적재된 소스별 문서(청크) 수를 집계해 요약 리스트로 반환한다.

        운영/디버깅용 — "지금 어떤 지식이 얼마나 들어와 있나"를 한눈에 본다.
        반환 예: [{"source": "kowiki", "count": 1050000}, ...] (source 오름차순).
        """
        if self._pg is None:
            # 폴백 — 메모리 dict를 돌며 source별로 개수를 센다.
            out: dict[str, int] = {}
            for e in self._store.values():
                out[e.source] = out.get(e.source, 0) + 1
            return [{"source": k, "count": v} for k, v in sorted(out.items())]
        async with self._pg.acquire() as conn:
            rows = await conn.fetch(
                "SELECT source, COUNT(*) AS n FROM tb_knowledge GROUP BY source ORDER BY source"
            )
        return [{"source": r["source"], "count": int(r["n"])} for r in rows]


# ─────────────────────────────────────────────
# 헬퍼 — _format_vector / _cosine은 core.rag.pgvector_base에서 import (위 상단)
# ─────────────────────────────────────────────
def _inmemory_search(
    store: dict[str, KnowledgeEntry],
    embedding: list[float],
    top_k: int,
    allowed_sources: list[str] | None,
    min_similarity: float,
    with_embedding: bool = False,
) -> list[dict[str, Any]]:
    """인메모리 폴백 벡터 검색 — DB 없는 작은 데이터셋/테스트 전용.

    DB의 ivfflat 근사 검색과 달리, 저장된 모든 청크를 한 번씩 훑으며 코사인
    유사도를 전부 계산하는 완전 탐색(brute force)이다. 데이터가 크면 느리므로
    어디까지나 테스트/개발용 폴백이다.

    with_embedding=True면 DB 경로와 대칭으로 결과 dict에 "embedding"(list[float])을
    동봉한다(MMR 후보 유사도 계산용). 기본 False면 종전과 동일.
    """
    # 허용 소스를 set으로 바꿔 두면 아래 반복문에서 포함 검사(in)가 빠르다.
    allowed = set(allowed_sources) if allowed_sources else None
    # (유사도, 청크) 쌍을 모아두는 리스트 — 나중에 유사도 기준 정렬한다.
    scored: list[tuple[float, KnowledgeEntry]] = []
    for e in store.values():
        # 허용 소스 필터에 걸리지 않는 청크는 건너뛴다.
        if allowed is not None and e.source not in allowed:
            continue
        # 임베딩이 아직 없는 청크는 유사도 계산 자체가 불가하므로 제외.
        if not e.embedding:
            continue
        sim = _cosine(embedding, list(e.embedding))
        # 유사도 하한 미만은 품질이 낮다고 보고 버린다.
        if sim < min_similarity:
            continue
        scored.append((sim, e))
    # 유사도 높은 순(내림차순)으로 정렬 후 상위 top_k만 취한다.
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for sim, e in scored[:top_k]:
        item: dict[str, Any] = {
            # id — DB 경로와 대칭으로 동봉(출처 인용 chunk_id·추적용). KnowledgeEntry.id는
            # source/title/section/chunk_index 해시라 폴백 경로에서도 결정론적이다.
            "id": e.id,
            "source": e.source, "title": e.title, "section": e.section,
            "content": e.content, "tags": list(e.tags),
            "similarity": round(sim, 4), "metadata": e.metadata,
        }
        if with_embedding:
            item["embedding"] = list(e.embedding)
        out.append(item)
    return out


# ─────────────────────────────────────────────
# 청크 분할 유틸 (prepare_kowiki.py 등에서 사용)
# ─────────────────────────────────────────────
def split_into_chunks(
    text: str,
    max_chars: int = 1200,
    overlap: int = 100,
) -> list[str]:
    """
    긴 원문 텍스트를 임베딩하기 좋은 크기의 청크 리스트로 쪼갠다(순수 함수).

    왜 쪼개나: 임베딩 모델은 입력 길이에 한계가 있고, 너무 긴 덩어리를 한
    벡터로 뭉치면 검색 정확도가 떨어진다. 그래서 의미가 이어지는 선에서
    적당한 크기로 나눈다.

    규칙:
      - 기본 경계: 빈 줄(\\n\\n)로 구분된 "문단" 단위로 먼저 나눈다.
      - 문단 하나가 max_chars를 넘으면 문장 단위(마침표/물음표 등 뒤)로 재분할.
      - 이웃한 청크가 overlap 글자만큼 겹치게 만들어 경계에서 문맥이 끊기는 걸 완화.

    Args:
        text: 원문 전체 문자열.
        max_chars: 한 청크의 목표 최대 글자 수.
        overlap: 이웃 청크끼리 겹칠 글자 수(0이면 겹침 없음).

    Returns:
        청크 문자열들의 리스트. 입력이 비면 빈 리스트.
    """
    if not text:
        return []

    # 1차 분할: 빈 줄 기준으로 문단을 나누고, 앞뒤 공백 제거 + 빈 문단은 버린다.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # chunks: 완성된 청크들. buf: 현재 채우는 중인 청크 버퍼(누적 문자열).
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        # 문단 하나가 max_chars보다 크면 그대로 담을 수 없어 문장 단위로 다시 쪼갠다.
        if len(para) > max_chars:
            # 마침표/느낌표/물음표/。 뒤의 공백을 경계로 문장을 나눈다.
            sentences = re.split(r"(?<=[.!?。])\s+", para)
            for s in sentences:
                # 문장을 더해도 한도 안이면 버퍼에 이어붙인다(+1은 사이 공백 몫).
                if len(buf) + len(s) + 1 <= max_chars:
                    buf = f"{buf} {s}".strip() if buf else s
                else:
                    # 한도를 넘으면 지금까지의 버퍼를 청크로 확정하고 버퍼를 새로 시작.
                    if buf:
                        chunks.append(buf)
                    # 문장 자체가 한도보다 길면 뒤쪽 max_chars만 남겨 폭주를 막는다.
                    buf = s[-max_chars:] if len(s) > max_chars else s
        else:
            # 문단이 작으면: 버퍼에 더 담을 수 있으면 빈 줄로 이어붙이고,
            if len(buf) + len(para) + 2 <= max_chars:
                buf = f"{buf}\n\n{para}" if buf else para
            else:
                # 넘치면 현재 버퍼를 청크로 확정하고 이 문단으로 새 버퍼를 시작한다.
                chunks.append(buf)
                buf = para

    # 반복이 끝난 뒤 버퍼에 남은 내용이 있으면 마지막 청크로 추가.
    if buf:
        chunks.append(buf)

    # overlap 적용: 각 청크 앞에 "직전 청크의 꼬리(tail)"를 붙여 문맥을 이어준다.
    if overlap > 0 and len(chunks) >= 2:
        with_overlap: list[str] = [chunks[0]]  # 첫 청크는 앞에 붙일 게 없다.
        for i in range(1, len(chunks)):
            # 직전 청크 끝에서 overlap 글자를 떼어 온다(그보다 짧으면 통째로).
            tail = chunks[i - 1][-overlap:] if overlap < len(chunks[i - 1]) else chunks[i - 1]
            with_overlap.append(tail + "\n" + chunks[i])
        return with_overlap

    return chunks
