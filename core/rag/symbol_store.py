"""
심볼 인덱스 저장소 — tb_symbols 기반 함수/클래스/메서드 검색 (Phase 10.0, 2026-04-21).

배경:
  Part 2.5.8에서 tb_knowledge(위키 등)로 일반 지식을 RAG로 주입할 수 있게 됐고,
  ProjectIndexer는 파일 청크 단위로 tb_memories에 저장 중이다. 그러나 "query_loop
  함수 어디 있어?" 같은 **심볼(함수/클래스/메서드) 단위 질의**는 청크가 경계를
  무시하고 쪼개기 때문에 정확도가 떨어진다.

  Phase 10.0은 Python AST로 심볼을 정확히 추출하여 별도 테이블에 저장하고,
  이름 매칭(pg_trgm) + 벡터 검색(pgvector)을 혼합한 전용 검색을 제공한다.

스키마 (tb_symbols):
  id              text PK   — SHA-256(path|kind|qualified_name|line_start)
  source          text      — 'nexus' 같은 프로젝트 식별자
  path            text      — 파일 상대 경로
  module          text      — Python 모듈 경로 (dotted)
  kind            text      — 'function' | 'class' | 'method' | 'async_function' | 'async_method'
  name            text      — 단일 이름 ('query_loop')
  qualified_name  text      — 전체 경로 (dotted, 예: 'core.rag.KnowledgeStore.add')
  signature       text      — 함수 시그니처 (args + return)
  docstring       text      — 원본 docstring (없으면 빈 문자열)
  summary         text      — 임베딩 입력용 요약 (kind + name + sig + docstring + 소스 발췌)
  line_start      int
  line_end        int
  embedding       vector(1024)
  tags            text[]
  created_at      timestamptz
  metadata        jsonb

인덱스:
  idx_symbols_source (btree)
  idx_symbols_path (btree)
  idx_symbols_kind (btree)
  idx_symbols_name (btree)
  idx_symbols_name_trgm (GIN, pg_trgm) — 부분매칭/오타 허용
  idx_symbols_embed (ivfflat cosine) — 대량 적재 후 별도 빌드

검색 전략:
  1) 이름 정확 일치 (가장 관련성 높음) — btree
  2) 이름 부분매칭 (query가 짧을 때) — pg_trgm similarity
  3) 벡터 검색 (의미 기반, 이름 모를 때) — 벡터 인덱스
  → SymbolRetriever가 위 세 결과를 순서대로 합치고 중복 제거한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from core.rag.pgvector_base import PgVectorStore
from core.rag.pgvector_base import cosine_similarity as _cosine
from core.rag.pgvector_base import format_vector as _format_vector

logger = logging.getLogger("nexus.rag.symbol_store")


_DDL_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS tb_symbols (
    id             text PRIMARY KEY,
    source         text NOT NULL,
    path           text NOT NULL,
    module         text NOT NULL,
    kind           text NOT NULL,
    name           text NOT NULL,
    qualified_name text NOT NULL,
    signature      text NOT NULL DEFAULT '',
    docstring      text NOT NULL DEFAULT '',
    summary        text NOT NULL,
    line_start     int NOT NULL DEFAULT 0,
    line_end       int NOT NULL DEFAULT 0,
    tags           text[] DEFAULT '{}',
    embedding      vector(1024),
    created_at     timestamptz NOT NULL DEFAULT now(),
    metadata       jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_symbols_source ON tb_symbols(source);
CREATE INDEX IF NOT EXISTS idx_symbols_path   ON tb_symbols(path);
CREATE INDEX IF NOT EXISTS idx_symbols_kind   ON tb_symbols(kind);
CREATE INDEX IF NOT EXISTS idx_symbols_name   ON tb_symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_name_trgm
    ON tb_symbols USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_symbols_qname_trgm
    ON tb_symbols USING GIN (qualified_name gin_trgm_ops);
"""

_DDL_IVFFLAT = """
CREATE INDEX IF NOT EXISTS idx_symbols_embed
    ON tb_symbols USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


@dataclass(frozen=True)
class SymbolEntry:
    """단일 심볼 — tb_symbols 1행에 대응된다."""

    source: str
    path: str
    module: str
    kind: str                  # function / class / method / async_function / async_method
    name: str
    qualified_name: str
    signature: str = ""
    docstring: str = ""
    summary: str = ""
    line_start: int = 0
    line_end: int = 0
    tags: tuple[str, ...] = ()
    embedding: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """
        결정론적 ID — path + kind + qualified_name + line_start의 SHA-256.
        같은 파일을 재인덱싱해도 같은 ID → UPSERT 멱등.
        """
        key = f"{self.path}|{self.kind}|{self.qualified_name}|{self.line_start}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class SymbolStore(PgVectorStore):
    """tb_symbols 기반 심볼 저장소.

    pg_pool이 None이면 인메모리 폴백 — 테스트/경량 개발 환경에서 쓰인다.
    공통 로직(ensure_schema/build_vector_index/count)은 PgVectorStore에서 상속.
    """

    # PgVectorStore 계약
    TABLE_NAME = "tb_symbols"
    DDL_SCHEMA = _DDL_SCHEMA
    DDL_IVFFLAT = _DDL_IVFFLAT

    # ─── 적재 ────────────────────────────────────────────
    async def add(self, entry: SymbolEntry) -> str:
        """심볼 1개를 UPSERT한다."""
        memory_id = entry.id
        if self._pg is None:
            self._store[memory_id] = entry
            return memory_id

        embedding_str = _format_vector(entry.embedding)
        async with self._pg.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tb_symbols (
                    id, source, path, module, kind, name, qualified_name,
                    signature, docstring, summary, line_start, line_end,
                    tags, embedding, metadata
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14::vector,$15::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    signature = EXCLUDED.signature,
                    docstring = EXCLUDED.docstring,
                    summary = EXCLUDED.summary,
                    line_end = EXCLUDED.line_end,
                    tags = EXCLUDED.tags,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                memory_id,
                entry.source,
                entry.path,
                entry.module,
                entry.kind,
                entry.name,
                entry.qualified_name,
                entry.signature,
                entry.docstring,
                entry.summary,
                entry.line_start,
                entry.line_end,
                list(entry.tags),
                embedding_str,
                json.dumps(entry.metadata, ensure_ascii=False),
            )
        return memory_id

    async def add_many(self, entries: list[SymbolEntry]) -> int:
        count = 0
        for e in entries:
            await self.add(e)
            count += 1
        return count

    async def delete_by_path(self, source: str, path: str) -> int:
        """파일 경로 단위 삭제 — 재인덱싱 전 기존 심볼을 비운다."""
        if self._pg is None:
            ids_to_del = [
                k for k, v in self._store.items()
                if v.source == source and v.path == path
            ]
            for k in ids_to_del:
                del self._store[k]
            return len(ids_to_del)
        async with self._pg.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM tb_symbols WHERE source = $1 AND path = $2",
                source, path,
            )
            # asyncpg의 execute는 'DELETE n' 문자열을 반환
            try:
                return int(res.split()[-1])
            except Exception:
                return 0

    # ─── 검색 ────────────────────────────────────────────
    async def search_by_name(
        self,
        query: str,
        top_k: int = 10,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """이름 기반 검색 — 정확 일치 + 부분매칭(trigram similarity).

        정확 일치가 우선(similarity=1.0으로 반환)하고, 부분매칭은 pg_trgm
        similarity 점수로 정렬한다.
        """
        if not query:
            return []
        if self._pg is None:
            q = query.lower()
            matches: list[tuple[float, SymbolEntry]] = []
            for e in self._store.values():
                if source and e.source != source:
                    continue
                ln = e.name.lower()
                lq = e.qualified_name.lower()
                if ln == q:
                    matches.append((1.0, e))
                elif q in ln or q in lq:
                    matches.append((0.6, e))
            matches.sort(key=lambda t: t[0], reverse=True)
            return [_entry_to_row(e, sim) for sim, e in matches[:top_k]]

        params: list[Any] = [query, top_k]
        where_extra = ""
        if source is not None:
            where_extra = " AND source = $3"
            params.append(source)

        q = f"""
            SELECT source, path, module, kind, name, qualified_name,
                   signature, docstring, summary, line_start, line_end,
                   tags, metadata,
                   CASE
                     WHEN name = $1 THEN 1.0
                     WHEN qualified_name = $1 THEN 0.98
                     ELSE GREATEST(similarity(name, $1), similarity(qualified_name, $1))
                   END AS sim
            FROM tb_symbols
            WHERE (name % $1 OR qualified_name % $1 OR name = $1 OR qualified_name = $1)
                  {where_extra}
            ORDER BY sim DESC
            LIMIT $2
        """  # noqa: S608 — where_extra는 whitelist(상수만)
        async with self._pg.acquire() as conn:
            rows = await conn.fetch(q, *params)
        return [_row_to_dict(r) for r in rows]

    async def search_by_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        source: str | None = None,
        min_similarity: float = 0.25,
    ) -> list[dict[str, Any]]:
        """벡터 검색 — 이름 모를 때의 의미 기반 폴백."""
        if self._pg is None:
            scored: list[tuple[float, SymbolEntry]] = []
            for e in self._store.values():
                if source and e.source != source:
                    continue
                if not e.embedding:
                    continue
                sim = _cosine(embedding, list(e.embedding))
                if sim < min_similarity:
                    continue
                scored.append((sim, e))
            scored.sort(key=lambda t: t[0], reverse=True)
            return [_entry_to_row(e, sim) for sim, e in scored[:top_k]]

        vec_str = _format_vector(embedding)
        max_distance = 1.0 - min_similarity
        params: list[Any] = [vec_str, max_distance, top_k]
        where_extra = ""
        if source is not None:
            where_extra = " AND source = $4"
            params.append(source)

        q = f"""
            SELECT source, path, module, kind, name, qualified_name,
                   signature, docstring, summary, line_start, line_end,
                   tags, metadata,
                   (1.0 - (embedding <=> $1::vector)) AS sim
            FROM tb_symbols
            WHERE embedding <=> $1::vector <= $2 {where_extra}
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """  # noqa: S608
        async with self._pg.acquire() as conn:
            rows = await conn.fetch(q, *params)
        return [_row_to_dict(r) for r in rows]

    # ─── 운영 유틸 ───────────────────────────────────────
    # count()는 PgVectorStore에서 상속한다.


# ─────────────────────────────────────────────
# 헬퍼 — _format_vector / _cosine은 core.rag.pgvector_base에서 import (위 상단)
# ─────────────────────────────────────────────
def _entry_to_row(e: SymbolEntry, sim: float) -> dict[str, Any]:
    return {
        "source": e.source, "path": e.path, "module": e.module, "kind": e.kind,
        "name": e.name, "qualified_name": e.qualified_name,
        "signature": e.signature, "docstring": e.docstring, "summary": e.summary,
        "line_start": e.line_start, "line_end": e.line_end,
        "tags": list(e.tags), "metadata": e.metadata,
        "similarity": round(float(sim), 4),
    }


def _row_to_dict(r: Any) -> dict[str, Any]:
    return {
        "source": r["source"], "path": r["path"], "module": r["module"],
        "kind": r["kind"], "name": r["name"],
        "qualified_name": r["qualified_name"],
        "signature": r["signature"], "docstring": r["docstring"],
        "summary": r["summary"],
        "line_start": int(r["line_start"]), "line_end": int(r["line_end"]),
        "tags": list(r["tags"] or []), "metadata": r["metadata"] or {},
        "similarity": round(float(r["sim"]), 4),
    }
