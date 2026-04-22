"""
지식 베이스 — tb_knowledge 기반 장기 교양 지식 저장소 (Part 2.5.8, 2026-04-21).

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

from core.rag.pgvector_base import PgVectorStore
from core.rag.pgvector_base import cosine_similarity as _cosine
from core.rag.pgvector_base import format_vector as _format_vector

logger = logging.getLogger("nexus.rag.knowledge_store")


# 스키마 생성 SQL — ensure_schema()가 멱등적으로 실행
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

# ivfflat 인덱스는 데이터가 어느 정도 쌓인 뒤에 만드는 것이 효율적이므로 별도 함수
_DDL_IVFFLAT = """
CREATE INDEX IF NOT EXISTS idx_knowledge_embed ON tb_knowledge
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


@dataclass(frozen=True)
class KnowledgeEntry:
    """단일 지식 청크 — tb_knowledge 1행과 대응된다."""

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
        결정론적 ID — source/title/section/chunk_index 조합의 SHA-256.
        동일 청크를 재적재해도 같은 ID가 나와 UPSERT로 멱등성이 보장된다.
        """
        key = f"{self.source}|{self.title}|{self.section or ''}|{self.chunk_index}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class KnowledgeStore(PgVectorStore):
    """tb_knowledge 기반 지식 베이스 — add/search/list 래퍼.

    DB 연결이 없으면 인메모리 폴백으로 동작하여 테스트/개발 환경에서도 쓰인다.
    장기 메모리(tb_memories)와 별도로 운영하여 대용량 지식 데이터가 대화
    히스토리 검색과 섞이지 않게 한다.

    공통 로직(ensure_schema/build_vector_index/count)은 PgVectorStore에서 상속.
    """

    # PgVectorStore 계약 — 베이스가 DDL 실행과 COUNT 쿼리에 사용한다.
    TABLE_NAME = "tb_knowledge"
    DDL_SCHEMA = _DDL_SCHEMA
    DDL_IVFFLAT = _DDL_IVFFLAT

    # ─── 쓰기 ───────────────────────────────────────
    async def add(self, entry: KnowledgeEntry) -> str:
        """청크 1건을 UPSERT한다 (같은 id는 덮어쓰기)."""
        memory_id = entry.id
        if self._pg is None:
            self._store[memory_id] = entry
            return memory_id

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
        """여러 청크를 배치로 적재한다 (단건 add 반복)."""
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

        Returns:
            [{"title", "section", "content", "source", "similarity", "tags"}, ...]
            유사도 내림차순.
        """
        # 단일 source는 allowed_sources의 특수 케이스로 정규화
        if allowed_sources is None and source is not None:
            allowed_sources = [source]

        if self._pg is None:
            # 인메모리 폴백 — 코사인 유사도 직접 계산
            return _inmemory_search(
                self._store, embedding, top_k, allowed_sources, min_similarity,
            )

        vec_str = _format_vector(embedding)
        # distance < 1 - min_similarity (cosine distance = 1 - cosine similarity)
        max_distance = 1.0 - min_similarity

        params: list[Any] = [vec_str, max_distance, top_k]
        where = "WHERE embedding <=> $1::vector <= $2"
        if allowed_sources:
            where += " AND source = ANY($4::text[])"
            params.append(list(allowed_sources))

        query = f"""
            SELECT source, title, section, content, tags, metadata,
                   (embedding <=> $1::vector) AS distance
            FROM tb_knowledge
            {where}
            ORDER BY distance ASC
            LIMIT $3
        """  # noqa: S608 — where 절은 whitelist 기반

        async with self._pg.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            {
                "source": r["source"],
                "title": r["title"],
                "section": r["section"],
                "content": r["content"],
                "tags": list(r["tags"] or []),
                "similarity": round(1.0 - float(r["distance"]), 4),
                "metadata": r["metadata"] or {},
            }
            for r in rows
        ]

    async def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """단순 ILIKE 기반 텍스트 검색 (임베딩 서버 불능 시 폴백).

        벡터 검색이 더 정확하지만, 임베딩 서버가 다운됐거나 질의가 너무 짧을 때
        최후의 수단으로 사용한다.
        """
        if self._pg is None:
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
        """적재된 소스별 문서 수 요약."""
        if self._pg is None:
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
) -> list[dict[str, Any]]:
    """인메모리 폴백 벡터 검색 — 작은 데이터셋/테스트 전용."""
    allowed = set(allowed_sources) if allowed_sources else None
    scored: list[tuple[float, KnowledgeEntry]] = []
    for e in store.values():
        if allowed is not None and e.source not in allowed:
            continue
        if not e.embedding:
            continue
        sim = _cosine(embedding, list(e.embedding))
        if sim < min_similarity:
            continue
        scored.append((sim, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "source": e.source, "title": e.title, "section": e.section,
            "content": e.content, "tags": list(e.tags),
            "similarity": round(sim, 4), "metadata": e.metadata,
        }
        for sim, e in scored[:top_k]
    ]


# ─────────────────────────────────────────────
# 청크 분할 유틸 (prepare_kowiki.py 등에서 사용)
# ─────────────────────────────────────────────
def split_into_chunks(
    text: str,
    max_chars: int = 1200,
    overlap: int = 100,
) -> list[str]:
    """
    문단 단위로 청크를 쪼갠다.

    규칙:
      - 기본 경계: 빈 줄로 구분된 문단
      - 한 문단이 max_chars를 넘으면 문장 단위로 재분할
      - 연속 청크가 overlap만큼 겹치도록 함 (문맥 단절 완화)
    """
    if not text:
        return []

    # 문단 기반 1차 분할
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        # 문단 하나가 너무 크면 문장 단위로 재분할
        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?。])\s+", para)
            for s in sentences:
                if len(buf) + len(s) + 1 <= max_chars:
                    buf = f"{buf} {s}".strip() if buf else s
                else:
                    if buf:
                        chunks.append(buf)
                    buf = s[-max_chars:] if len(s) > max_chars else s
        else:
            if len(buf) + len(para) + 2 <= max_chars:
                buf = f"{buf}\n\n{para}" if buf else para
            else:
                chunks.append(buf)
                buf = para

    if buf:
        chunks.append(buf)

    # overlap 적용: 각 청크 앞에 이전 청크의 tail을 붙인다
    if overlap > 0 and len(chunks) >= 2:
        with_overlap: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:] if overlap < len(chunks[i - 1]) else chunks[i - 1]
            with_overlap.append(tail + "\n" + chunks[i])
        return with_overlap

    return chunks
