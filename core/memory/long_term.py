"""
장기 메모리 — PostgreSQL + pgvector 기반 영구 저장소.

중요도가 높은 메모리를 영구 저장하고, 벡터 유사도 검색을 지원한다:
  - 텍스트 검색: ILIKE 기반 키워드 매칭
  - 벡터 검색: pgvector의 cosine distance (<=>)로 의미 유사도 검색
  - 타입 필터: MemoryType별 조회

PostgreSQL이 없으면 인메모리 딕셔너리로 폴백한다.
에어갭 환경에서 PostgreSQL은 LAN 내 서버(192.168.x.x)에 위치한다.

설계 결정:
  - pgvector 확장: 벡터 컬럼에 cosine distance 인덱스 생성
  - 인메모리 폴백: DB 없이도 개발/테스트 가능
  - access_count 자동 증가: get() 호출 시 접근 횟수와 last_accessed 갱신
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

from core.memory.types import MemoryEntry, MemoryType

logger = logging.getLogger("nexus.memory.long_term")


class LongTermMemory:
    """
    PostgreSQL + pgvector 기반 장기 메모리.

    중요도가 높은 메모리를 영구 저장하고,
    텍스트/벡터 유사도 검색으로 관련 기억을 찾는다.
    """

    def __init__(self, pg_pool: Any | None = None):
        """
        장기 메모리를 초기화한다.

        Args:
            pg_pool: asyncpg.Pool 인스턴스 (None이면 인메모리 폴백)
        """
        # 인메모리 폴백 저장소: {memory_id: MemoryEntry}
        self._store: dict[str, MemoryEntry] = {}
        self._pg = pg_pool

        if self._pg is None:
            logger.info("PostgreSQL 풀 없음 — 인메모리 폴백 모드로 동작")

    # ─── CRUD 기본 연산 ───

    async def add(self, entry: MemoryEntry) -> str:
        """
        새 메모리를 저장한다.

        Args:
            entry: 저장할 MemoryEntry

        Returns:
            저장된 메모리의 ID
        """
        if self._pg is not None:
            try:
                return await self._add_pg(entry)
            except Exception as e:
                logger.warning("PostgreSQL add 실패: %s — 폴백 사용", e)

        # 인메모리 폴백
        self._store[entry.id] = entry
        logger.debug("장기 메모리 추가 (폴백): id=%s, type=%s", entry.id, entry.memory_type)
        return entry.id

    async def get(self, memory_id: str) -> MemoryEntry | None:
        """
        ID로 메모리를 조회한다.
        조회 시 access_count와 last_accessed를 자동 갱신한다.

        Args:
            memory_id: 조회할 메모리 ID

        Returns:
            MemoryEntry 또는 None
        """
        if self._pg is not None:
            try:
                return await self._get_pg(memory_id)
            except Exception as e:
                logger.warning("PostgreSQL get 실패 (id=%s): %s — 폴백 사용", memory_id, e)

        # 인메모리 폴백
        entry = self._store.get(memory_id)
        if entry is not None:
            # 접근 횟수 및 시각 갱신 (새 객체 생성 — Pydantic 모델)
            updated = entry.model_copy(
                update={
                    "access_count": entry.access_count + 1,
                    "last_accessed": datetime.now(UTC),
                }
            )
            self._store[memory_id] = updated
            return updated
        return None

    async def search_by_text(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        top_k: int = 10,
    ) -> list[MemoryEntry]:
        """
        텍스트 키워드로 메모리를 검색한다.

        content 필드에서 대소문자 무시 매칭을 수행한다.
        PostgreSQL에서는 ILIKE, 폴백에서는 파이썬 문자열 매칭을 사용한다.

        Args:
            query: 검색 쿼리 문자열
            memory_type: 특정 타입으로 필터링 (None이면 전체)
            top_k: 최대 반환 건수

        Returns:
            관련도 순 MemoryEntry 목록
        """
        if self._pg is not None:
            try:
                return await self._search_by_text_pg(query, memory_type, top_k)
            except Exception as e:
                logger.warning("PostgreSQL 텍스트 검색 실패: %s — 폴백 사용", e)

        # 인메모리 폴백: 단순 키워드 매칭
        query_lower = query.lower()
        results: list[MemoryEntry] = []

        for entry in self._store.values():
            # 타입 필터
            if memory_type is not None and entry.memory_type != memory_type:
                continue

            # content, key, tags에서 키워드 매칭
            searchable = f"{entry.content} {entry.key} {' '.join(entry.tags)}".lower()
            if query_lower in searchable:
                results.append(entry)

        # 중요도 × 접근 횟수로 정렬 (높을수록 우선)
        results.sort(
            key=lambda e: e.importance * (1 + e.access_count),
            reverse=True,
        )
        return results[:top_k]

    async def search_by_vector(
        self,
        embedding: list[float],
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        """
        벡터 유사도로 메모리를 검색한다.

        pgvector의 cosine distance (<=>)를 사용하여
        의미적으로 유사한 메모리를 찾는다.

        Args:
            embedding: 쿼리 벡터 (e5-large 등으로 생성)
            memory_type: 특정 타입으로 필터링 (None이면 전체)
            top_k: 최대 반환 건수

        Returns:
            유사도 순 MemoryEntry 목록
        """
        if self._pg is not None:
            try:
                return await self._search_by_vector_pg(embedding, memory_type, top_k)
            except Exception as e:
                logger.warning("PostgreSQL 벡터 검색 실패: %s — 폴백 사용", e)

        # 인메모리 폴백: 코사인 유사도 직접 계산
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._store.values():
            # 타입 필터
            if memory_type is not None and entry.memory_type != memory_type:
                continue

            # 임베딩이 없는 항목은 건너뛴다
            if entry.embedding is None:
                continue

            # 코사인 유사도 계산
            similarity = self._cosine_similarity(embedding, entry.embedding)
            scored.append((similarity, entry))

        # 유사도 내림차순 정렬
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def update(self, memory_id: str, **kwargs: Any) -> bool:
        """
        메모리 필드를 업데이트한다.

        Pydantic model_copy(update=...)를 사용하여 불변성을 유지한다.

        Args:
            memory_id: 업데이트할 메모리 ID
            **kwargs: 변경할 필드 (예: importance=0.8, tags=["new_tag"])

        Returns:
            업데이트 성공 여부
        """
        if self._pg is not None:
            try:
                return await self._update_pg(memory_id, **kwargs)
            except Exception as e:
                logger.warning("PostgreSQL update 실패 (id=%s): %s — 폴백 사용", memory_id, e)

        # 인메모리 폴백
        entry = self._store.get(memory_id)
        if entry is None:
            return False

        # 허용된 필드만 업데이트
        allowed_fields = {
            "content",
            "key",
            "tags",
            "importance",
            "access_count",
            "last_accessed",
            "embedding",
            "metadata",
        }
        update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not update_data:
            logger.warning("업데이트할 유효한 필드 없음 (id=%s)", memory_id)
            return False

        updated = entry.model_copy(update=update_data)
        self._store[memory_id] = updated
        logger.debug(
            "장기 메모리 업데이트 (폴백): id=%s, fields=%s",
            memory_id,
            list(update_data.keys()),
        )
        return True

    async def delete(self, memory_id: str) -> bool:
        """
        메모리를 삭제한다.

        Args:
            memory_id: 삭제할 메모리 ID

        Returns:
            삭제 성공 여부
        """
        if self._pg is not None:
            try:
                return await self._delete_pg(memory_id)
            except Exception as e:
                logger.warning("PostgreSQL delete 실패 (id=%s): %s — 폴백 사용", memory_id, e)

        # 인메모리 폴백
        if memory_id in self._store:
            del self._store[memory_id]
            logger.debug("장기 메모리 삭제 (폴백): id=%s", memory_id)
            return True
        return False

    async def get_by_type(self, memory_type: MemoryType, limit: int = 50) -> list[MemoryEntry]:
        """
        특정 타입의 메모리를 모두 조회한다.

        Args:
            memory_type: 조회할 메모리 타입
            limit: 최대 반환 건수

        Returns:
            해당 타입의 MemoryEntry 목록 (최신순)
        """
        if self._pg is not None:
            try:
                return await self._get_by_type_pg(memory_type, limit)
            except Exception as e:
                logger.warning("PostgreSQL get_by_type 실패: %s — 폴백 사용", e)

        # 인메모리 폴백
        entries = [e for e in self._store.values() if e.memory_type == memory_type]
        # 최신순 정렬
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[:limit]

    async def get_all(self, limit: int = 100) -> list[MemoryEntry]:
        """
        전체 메모리를 조회한다.
        감쇠 사이클(decay cycle)이나 통합(consolidation)에서 사용한다.

        Args:
            limit: 최대 반환 건수

        Returns:
            MemoryEntry 목록 (최신순)
        """
        if self._pg is not None:
            try:
                return await self._get_all_pg(limit)
            except Exception as e:
                logger.warning("PostgreSQL get_all 실패: %s — 폴백 사용", e)

        # 인메모리 폴백
        entries = list(self._store.values())
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[:limit]

    # ─── PostgreSQL 구현 (내부) ───

    async def _add_pg(self, entry: MemoryEntry) -> str:
        """PostgreSQL에 메모리를 INSERT한다."""
        embedding_str = str(entry.embedding) if entry.embedding else None
        import json

        await self._pg.execute(
            """
            INSERT INTO tb_memories (id, memory_type, content, key, tags, importance,
                                  access_count, created_at, last_accessed, embedding, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            entry.id,
            entry.memory_type,
            entry.content,
            entry.key,
            entry.tags,
            entry.importance,
            entry.access_count,
            entry.created_at,
            entry.last_accessed,
            embedding_str,
            json.dumps(entry.metadata, ensure_ascii=False, default=str),
        )
        logger.debug("장기 메모리 추가 (PG): id=%s, type=%s", entry.id, entry.memory_type)
        return entry.id

    async def _get_pg(self, memory_id: str) -> MemoryEntry | None:
        """PostgreSQL에서 메모리를 조회하고 access_count를 증가시킨다."""
        row = await self._pg.fetchrow("SELECT * FROM tb_memories WHERE id = $1", memory_id)
        if row is None:
            return None

        # 접근 횟수 갱신
        await self._pg.execute(
            "UPDATE tb_memories SET access_count = access_count + 1, last_accessed = $1 WHERE id = $2",
            datetime.now(UTC),
            memory_id,
        )
        return self._row_to_entry(row)

    async def _search_by_text_pg(
        self, query: str, memory_type: MemoryType | None, top_k: int
    ) -> list[MemoryEntry]:
        """PostgreSQL ILIKE 기반 텍스트 검색."""
        if memory_type is not None:
            rows = await self._pg.fetch(
                """
                SELECT * FROM tb_memories
                WHERE content ILIKE $1 AND memory_type = $2
                ORDER BY importance * (1 + access_count) DESC
                LIMIT $3
                """,
                f"%{query}%",
                memory_type,
                top_k,
            )
        else:
            rows = await self._pg.fetch(
                """
                SELECT * FROM tb_memories
                WHERE content ILIKE $1
                ORDER BY importance * (1 + access_count) DESC
                LIMIT $2
                """,
                f"%{query}%",
                top_k,
            )
        return [self._row_to_entry(row) for row in rows]

    async def _search_by_vector_pg(
        self, embedding: list[float], memory_type: MemoryType | None, top_k: int
    ) -> list[MemoryEntry]:
        """pgvector cosine distance 기반 벡터 검색."""
        embedding_str = str(embedding)
        if memory_type is not None:
            rows = await self._pg.fetch(
                """
                SELECT *, embedding <=> $1::vector AS distance
                FROM tb_memories
                WHERE embedding IS NOT NULL AND memory_type = $2
                ORDER BY distance ASC
                LIMIT $3
                """,
                embedding_str,
                memory_type,
                top_k,
            )
        else:
            rows = await self._pg.fetch(
                """
                SELECT *, embedding <=> $1::vector AS distance
                FROM tb_memories
                WHERE embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT $2
                """,
                embedding_str,
                top_k,
            )
        return [self._row_to_entry(row) for row in rows]

    async def _update_pg(self, memory_id: str, **kwargs: Any) -> bool:
        """PostgreSQL에서 메모리를 업데이트한다."""
        import json

        allowed_fields = {
            "content",
            "key",
            "tags",
            "importance",
            "access_count",
            "last_accessed",
            "embedding",
            "metadata",
        }
        update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not update_data:
            return False

        # SET 절 동적 생성
        set_clauses = []
        params: list[Any] = []
        for i, (field, value) in enumerate(update_data.items(), start=1):
            if field == "metadata":
                value = json.dumps(value, ensure_ascii=False, default=str)
            elif field == "embedding" and value is not None:
                value = str(value)
            set_clauses.append(f"{field} = ${i}")
            params.append(value)

        params.append(memory_id)
        # 안전: set_clauses는 allowed_fields 화이트리스트에서만 생성되므로 SQL injection 위험 없음
        query = f"UPDATE tb_memories SET {', '.join(set_clauses)} WHERE id = ${len(params)}"  # noqa: S608

        result = await self._pg.execute(query, *params)
        # asyncpg는 "UPDATE N" 형식의 문자열을 반환
        return result is not None and "UPDATE 0" not in str(result)

    async def _delete_pg(self, memory_id: str) -> bool:
        """PostgreSQL에서 메모리를 삭제한다."""
        result = await self._pg.execute("DELETE FROM tb_memories WHERE id = $1", memory_id)
        return result is not None and "DELETE 0" not in str(result)

    async def _get_by_type_pg(self, memory_type: MemoryType, limit: int) -> list[MemoryEntry]:
        """PostgreSQL에서 특정 타입의 메모리를 조회한다."""
        rows = await self._pg.fetch(
            """
            SELECT * FROM tb_memories
            WHERE memory_type = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            memory_type,
            limit,
        )
        return [self._row_to_entry(row) for row in rows]

    async def _get_all_pg(self, limit: int) -> list[MemoryEntry]:
        """PostgreSQL에서 전체 메모리를 조회한다."""
        rows = await self._pg.fetch(
            "SELECT * FROM tb_memories ORDER BY created_at DESC LIMIT $1",
            limit,
        )
        return [self._row_to_entry(row) for row in rows]

    # ─── 유틸리티 (내부) ───

    @staticmethod
    def _row_to_entry(row: Any) -> MemoryEntry:
        """asyncpg Record를 MemoryEntry로 변환한다."""
        import json

        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        # embedding은 pgvector가 문자열로 반환할 수 있음
        embedding = row.get("embedding")
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except (json.JSONDecodeError, TypeError):
                embedding = None

        return MemoryEntry(
            id=row["id"],
            memory_type=row["memory_type"],
            content=row["content"],
            key=row.get("key", ""),
            tags=row.get("tags", []),
            importance=float(row.get("importance", 0.5)),
            access_count=int(row.get("access_count", 0)),
            created_at=row.get("created_at", datetime.now(UTC)),
            last_accessed=row.get("last_accessed", datetime.now(UTC)),
            embedding=embedding,
            metadata=metadata,
        )

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        두 벡터의 코사인 유사도를 계산한다.

        왜 직접 계산하는가: numpy 의존 없이 인메모리 폴백에서 사용하기 위함.
        실제 운영에서는 pgvector의 <=> 연산자가 이 역할을 한다.

        Args:
            vec_a: 벡터 A
            vec_b: 벡터 B

        Returns:
            코사인 유사도 (-1.0 ~ 1.0)
        """
        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)
