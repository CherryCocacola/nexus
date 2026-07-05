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

이 파일의 구성 요소:
  - SymbolEntry(dataclass): tb_symbols 1행에 대응하는 불변 데이터. id는
    path/kind/qualified_name/line_start를 해싱한 결정론적 값이라 재인덱싱해도
    같은 심볼이면 동일 ID가 나와 UPSERT가 멱등하게 동작한다.
  - SymbolStore(PgVectorStore): 실제 저장/검색을 담당. PostgreSQL 풀(self._pg)이
    있으면 DB로, 없으면 인메모리 dict(self._store) 폴백으로 동작한다.
  - _entry_to_row / _row_to_dict: 인메모리 SymbolEntry와 DB Record를 각각
    동일한 dict 형태로 변환하는 헬퍼(호출 측이 결과 출처를 신경 쓰지 않게 통일).

의존:
  - core.rag.pgvector_base.PgVectorStore — ensure_schema/build_vector_index/
    count 등 공통 로직 제공(상속으로 재사용).
  - format_vector: 파이썬 리스트를 pgvector 리터럴 문자열로 직렬화.
  - cosine_similarity: 인메모리 폴백에서 코사인 유사도 계산.

노출 API:
  - add / add_many / delete_by_path: 적재·삭제 (ProjectIndexer가 호출).
  - search_by_name / search_by_vector: 검색 (SymbolRetriever가 호출).

작성자: 이현수 / 작성일: 2026-07-05
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


# tb_symbols 테이블과 인덱스를 만드는 DDL(초기 스키마).
# PgVectorStore.ensure_schema()가 이 문자열을 실행한다.
# - vector/pg_trgm 확장을 먼저 보장(이미 있으면 무시).
# - name/qualified_name에 GIN(pg_trgm) 인덱스를 걸어 부분매칭·오타 허용 검색을 빠르게.
#   (벡터 ivfflat 인덱스는 대량 적재 후 별도로 빌드하므로 여기 포함하지 않는다.)
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

# 벡터 검색용 ivfflat 인덱스 DDL.
# 대량 적재가 끝난 뒤 PgVectorStore.build_vector_index()가 별도로 실행한다.
# 빈 테이블에 미리 만들면 클러스터링 품질이 나빠지므로 스키마 생성과 분리했다.
_DDL_IVFFLAT = """
CREATE INDEX IF NOT EXISTS idx_symbols_embed
    ON tb_symbols USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


@dataclass(frozen=True)
class SymbolEntry:
    """단일 심볼(함수/클래스/메서드) — tb_symbols 1행에 대응하는 불변 데이터.

    AST에서 추출한 심볼 하나를 담는 값 객체다. frozen=True라 생성 후 수정
    불가(불변) — 인덱싱 파이프라인에서 만들어 그대로 저장에 넘긴다.

    필드 개요:
      source        프로젝트 식별자('nexus' 등).
      path          파일 상대 경로.
      module        Python 모듈 경로(dotted).
      kind          심볼 종류(function/class/method/async_function/async_method).
      name          단일 이름('query_loop').
      qualified_name 전체 경로(dotted, 예: 'core.rag.KnowledgeStore.add').
      signature     함수 시그니처(인자 + 반환 타입).
      docstring     원본 docstring(없으면 빈 문자열).
      summary       임베딩 입력용 요약 텍스트.
      line_start/line_end 소스 파일에서의 시작/끝 줄 번호.
      tags          분류용 태그.
      embedding     summary를 임베딩한 벡터(없을 수 있음).
      metadata      부가 정보(jsonb로 저장).
    """

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
        """이 심볼의 기본키(PK)를 계산하는 결정론적 ID.

        path + kind + qualified_name + line_start를 이어 붙여 SHA-256으로
        해싱한 뒤 앞 32자를 쓴다. 같은 심볼은 항상 같은 ID가 나오므로,
        같은 파일을 다시 인덱싱해도 새 행이 아니라 기존 행이 갱신(UPSERT)된다
        → 중복 없이 멱등하게 재적재할 수 있다.
        """
        # 네 값을 '|'로 구분해 이어 붙여 해시 입력 키를 만든다.
        key = f"{self.path}|{self.kind}|{self.qualified_name}|{self.line_start}"
        # UTF-8로 인코딩해 해싱하고, 16진수 다이제스트의 앞 32자만 ID로 사용.
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class SymbolStore(PgVectorStore):
    """tb_symbols 기반 심볼 저장소 — 적재와 검색을 함께 담당한다.

    PgVectorStore를 상속해 두 가지 모드로 동작한다:
      - DB 모드: self._pg(asyncpg 풀)가 있으면 PostgreSQL의 tb_symbols에 저장/조회.
      - 인메모리 폴백: self._pg가 None이면 self._store(dict)에 저장 — 테스트나
        경량 개발 환경에서 DB 없이도 같은 API로 동작하게 한다.

    ensure_schema/build_vector_index/count 등 공통 로직은 부모 PgVectorStore에서
    상속받고, 이 클래스는 심볼 특화 적재(add)와 검색(search_by_*)만 구현한다.
    """

    # ── PgVectorStore가 요구하는 계약(클래스 속성) ──
    # 부모 공통 로직이 이 값들을 참조해 DDL 실행/카운트 등을 수행한다.
    TABLE_NAME = "tb_symbols"
    DDL_SCHEMA = _DDL_SCHEMA
    DDL_IVFFLAT = _DDL_IVFFLAT

    # ─── 적재 ────────────────────────────────────────────
    async def add(self, entry: SymbolEntry) -> str:
        """심볼 1개를 저장(UPSERT)하고 그 ID를 반환한다.

        entry.id가 결정론적이므로 같은 심볼을 다시 넣으면 기존 행이 갱신된다.
        DB가 없으면 인메모리 dict에 그대로 넣고 끝낸다.
        """
        # 결정론적 PK(재적재 시 UPSERT 키로도 쓰인다).
        memory_id = entry.id
        # 인메모리 폴백: DB 풀이 없으면 dict에 저장하고 바로 반환.
        if self._pg is None:
            self._store[memory_id] = entry
            return memory_id

        # 임베딩(파이썬 튜플/리스트)을 pgvector가 받는 리터럴 문자열로 변환.
        embedding_str = _format_vector(entry.embedding)
        # 풀에서 커넥션을 하나 빌려 INSERT ... ON CONFLICT(UPSERT) 실행.
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
        """여러 심볼을 순차로 add()해서 저장한다. 저장한 개수를 반환.

        내부적으로 하나씩 add()를 호출하므로 각 항목이 개별 UPSERT된다
        (부분 실패 시 이미 저장된 것은 남는다).
        """
        count = 0
        # 리스트를 순회하며 하나씩 저장하고 성공 개수를 센다.
        for e in entries:
            await self.add(e)
            count += 1
        return count

    async def delete_by_path(self, source: str, path: str) -> int:
        """특정 파일(source+path)에 속한 심볼을 모두 삭제하고 삭제 개수를 반환.

        파일을 다시 인덱싱하기 전에 호출해 그 파일의 옛 심볼을 비운다. 이렇게
        하면 함수가 삭제되거나 이동해도 유령(stale) 심볼이 남지 않는다.
        """
        # 인메모리 폴백: dict에서 source/path가 일치하는 키를 모아 삭제.
        if self._pg is None:
            ids_to_del = [
                k for k, v in self._store.items()
                if v.source == source and v.path == path
            ]
            # dict를 순회하는 도중 수정하지 않도록 키 목록을 먼저 모은 뒤 삭제.
            for k in ids_to_del:
                del self._store[k]
            return len(ids_to_del)
        # DB 모드: 한 번의 DELETE로 해당 파일의 심볼을 모두 지운다.
        async with self._pg.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM tb_symbols WHERE source = $1 AND path = $2",
                source, path,
            )
            # asyncpg의 execute는 'DELETE n' 문자열을 반환 → 마지막 토큰이 삭제 행수.
            try:
                # 'DELETE 3' → 공백으로 쪼갠 마지막 조각 '3'을 정수로 변환.
                return int(res.split()[-1])
            except Exception:
                # 예상치 못한 반환 형식이면 안전하게 0으로 처리(예외 전파 방지).
                return 0

    # ─── 검색 ────────────────────────────────────────────
    async def search_by_name(
        self,
        query: str,
        top_k: int = 10,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """이름 기반 검색 — 정확 일치 + 부분매칭(trigram similarity).

        "query_loop 함수 어디 있어?"처럼 심볼 이름을 알 때 쓰는 1차 검색이다.
        정확 일치가 우선(similarity=1.0으로 반환)하고, 부분매칭은 pg_trgm
        similarity 점수로 정렬한다.

        매개변수:
          query    찾을 이름(또는 그 일부).
          top_k    최대 결과 개수.
          source   특정 프로젝트로 한정하고 싶을 때 지정(None이면 전체).
        반환: 각 심볼을 dict로 표현한 리스트(similarity 점수 포함).
        """
        # 빈 질의는 검색할 것이 없으므로 즉시 빈 결과.
        if not query:
            return []
        # ── 인메모리 폴백: DB 없이 dict를 직접 훑어 이름을 비교 ──
        if self._pg is None:
            q = query.lower()  # 대소문자 무시 비교를 위해 소문자화.
            matches: list[tuple[float, SymbolEntry]] = []
            for e in self._store.values():
                # source가 지정됐고 일치하지 않으면 건너뛴다.
                if source and e.source != source:
                    continue
                ln = e.name.lower()
                lq = e.qualified_name.lower()
                if ln == q:
                    # 이름 완전 일치 → 최고 점수 1.0.
                    matches.append((1.0, e))
                elif q in ln or q in lq:
                    # 이름/전체경로에 부분 포함 → 낮은 고정 점수 0.6.
                    matches.append((0.6, e))
            # 점수 내림차순 정렬 후 상위 top_k만 dict로 변환해 반환.
            matches.sort(key=lambda t: t[0], reverse=True)
            return [_entry_to_row(e, sim) for sim, e in matches[:top_k]]

        # ── DB 모드: SQL로 정확일치 + trigram 부분매칭을 한 번에 처리 ──
        # 위치 파라미터: $1=query, $2=top_k, (선택) $3=source.
        params: list[Any] = [query, top_k]
        where_extra = ""
        # source 필터가 있으면 WHERE 절과 파라미터를 안전하게 덧붙인다.
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
        # 위 CASE 식: name 완전일치=1.0, qualified_name 완전일치=0.98,
        # 그 외에는 두 컬럼의 trigram similarity 중 큰 값을 점수로 쓴다.
        # WHERE의 '%'는 pg_trgm의 유사도 연산자(임계값 이상 유사하면 매칭).
        # (noqa: S608은 f-string SQL 경고 억제 — where_extra는 상수만 허용해 안전.)
        async with self._pg.acquire() as conn:
            rows = await conn.fetch(q, *params)
        # DB Record를 공통 dict 형태로 변환해 반환.
        return [_row_to_dict(r) for r in rows]

    async def search_by_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        source: str | None = None,
        min_similarity: float = 0.25,
    ) -> list[dict[str, Any]]:
        """벡터 검색 — 이름을 모를 때의 의미 기반 폴백.

        질의 임베딩과 각 심볼 임베딩의 코사인 유사도로 가장 가까운 심볼을 찾는다.
        이름 검색이 실패했거나 "세션을 압축하는 코드" 같은 서술형 질의에 유용하다.

        매개변수:
          embedding      질의 텍스트를 임베딩한 벡터.
          top_k          최대 결과 개수.
          source         특정 프로젝트로 한정(None이면 전체).
          min_similarity 이 값 미만의 유사도는 버림(노이즈 컷).
        반환: 심볼 dict 리스트(similarity 내림차순).
        """
        # ── 인메모리 폴백: 각 심볼과 코사인 유사도를 직접 계산 ──
        if self._pg is None:
            scored: list[tuple[float, SymbolEntry]] = []
            for e in self._store.values():
                # source 필터에 안 맞으면 제외.
                if source and e.source != source:
                    continue
                # 임베딩이 없는 심볼은 벡터 비교 불가 → 건너뛴다.
                if not e.embedding:
                    continue
                sim = _cosine(embedding, list(e.embedding))
                # 최소 유사도 미만이면 관련 없다고 보고 버린다.
                if sim < min_similarity:
                    continue
                scored.append((sim, e))
            # 유사도 내림차순으로 정렬 후 상위 top_k만 반환.
            scored.sort(key=lambda t: t[0], reverse=True)
            return [_entry_to_row(e, sim) for sim, e in scored[:top_k]]

        # ── DB 모드: pgvector의 <=> (코사인 거리) 연산으로 정렬 ──
        vec_str = _format_vector(embedding)  # 벡터를 pgvector 리터럴로 직렬화.
        # 유사도 임계값을 '거리' 임계값으로 환산(거리 = 1 - 유사도).
        max_distance = 1.0 - min_similarity
        # 위치 파라미터: $1=벡터, $2=최대거리, $3=top_k, (선택) $4=source.
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
        # SELECT의 (1.0 - (embedding <=> $1)) = 코사인 유사도로 환산해 반환.
        # WHERE는 거리 임계값 이하만 통과시키고, ORDER BY 거리로 가까운 순 정렬.
        async with self._pg.acquire() as conn:
            rows = await conn.fetch(q, *params)
        return [_row_to_dict(r) for r in rows]

    # ─── 운영 유틸 ───────────────────────────────────────
    # count()는 PgVectorStore에서 상속한다(테이블 행 수 조회).


# ─────────────────────────────────────────────
# 헬퍼 — _format_vector / _cosine은 core.rag.pgvector_base에서 import (위 상단)
# ─────────────────────────────────────────────
def _entry_to_row(e: SymbolEntry, sim: float) -> dict[str, Any]:
    """인메모리 SymbolEntry를 검색 결과 dict로 변환한다.

    DB 경로의 _row_to_dict()와 완전히 같은 키 집합을 만들어, 호출 측이
    결과가 DB에서 왔는지 인메모리에서 왔는지 신경 쓰지 않게 통일한다.
    sim은 이 심볼의 유사도 점수(소수 4자리로 반올림).
    """
    return {
        "source": e.source, "path": e.path, "module": e.module, "kind": e.kind,
        "name": e.name, "qualified_name": e.qualified_name,
        "signature": e.signature, "docstring": e.docstring, "summary": e.summary,
        "line_start": e.line_start, "line_end": e.line_end,
        "tags": list(e.tags), "metadata": e.metadata,
        "similarity": round(float(sim), 4),
    }


def _row_to_dict(r: Any) -> dict[str, Any]:
    """DB에서 가져온 Record(r)를 검색 결과 dict로 변환한다.

    _entry_to_row()와 동일한 키 형태를 보장한다. line_start/line_end는 int로,
    tags/metadata는 None일 때 빈 값으로 정규화하고, sim은 4자리로 반올림한다.
    """
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
