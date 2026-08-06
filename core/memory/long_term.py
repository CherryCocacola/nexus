"""
장기 메모리(Long-Term Memory) — PostgreSQL + pgvector 기반 영구 저장소.

이 파일은 Nexus의 "오래 기억해야 하는" 정보를 디스크(PostgreSQL)에 영구
보관하고 다시 꺼내오는 계층이다. 단기 메모리(Redis)가 세션이 끝나면 사라지는
반면, 여기 저장된 메모리는 재부팅/재배포 후에도 남는다. 중요도가 높다고
판정된 기억만 이 계층으로 "승격(promotion)"되어 넘어온다.

세 가지 검색 방식을 제공한다:
  - 텍스트 검색: content 컬럼을 ILIKE로 키워드 부분 일치 검색
  - 벡터 검색: pgvector의 코사인 거리 연산자(<=>)로 의미 유사도 검색
  - 타입 필터: MemoryType(에피소드/사실/절차 등)별로 좁혀서 조회

핵심 클래스:
  - LongTermMemory: 위 검색 3종 + CRUD(add/get/update/delete)를 제공하는 유일한
    공개 클래스. 상위 계층인 core/memory/manager.py가 이 클래스를 사용한다.

폴백 설계(중요):
  - 생성자에 pg_pool이 주어지지 않으면(None) PostgreSQL 없이 파이썬 딕셔너리
    (_store)만으로 동작한다. DB를 띄우지 않고도 개발/단위 테스트가 가능하다.
  - 또한 PG 연산 중 예외가 나면 각 메서드가 자동으로 인메모리 폴백으로 넘어가
    서비스가 죽지 않도록 방어한다(fail-open 성격의 가용성 우선 처리).

에어갭(폐쇄망) 환경에서 PostgreSQL은 외부가 아닌 LAN 내부 서버
(예: 192.168.x.x)에 위치하며, 외부 네트워크 호출은 일절 하지 않는다.

기타 설계 결정:
  - pgvector 확장을 켜고 embedding 컬럼에 hnsw 인덱스를 만들어 벡터 검색을 가속.
  - get() 호출 시 access_count(접근 횟수)와 last_accessed(마지막 접근 시각)를
    자동 갱신 — 나중에 감쇠(decay)/통합(consolidation) 로직이 "얼마나 자주
    쓰이는 기억인가"를 판단하는 근거로 삼는다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

from core.memory.types import MemoryEntry, MemoryType

logger = logging.getLogger("nexus.memory.long_term")


# ─────────────────────────────────────────────
# tb_memories DDL — 운영 정의(2026-06-01 추출) 1:1
# ─────────────────────────────────────────────
# 아래 문자열은 실제 운영 DB에서 그대로 뽑아온 테이블 생성문이다. ensure_schema()
# 가 이 DDL을 멱등(IF NOT EXISTS)으로 실행해, 새 환경에서도 운영과 완전히 동일한
# 스키마를 자동으로 만들어 준다. 컬럼 하나하나가 운영과 1:1로 맞아야 하므로
# 임의로 타입/제약을 바꾸지 말 것.
#   - id varchar(12): importance.ImportanceAssessor가 생성하는 sha 해시 앞 12자리와
#     길이가 맞도록 정한 값. PK로 사용한다.
#   - importance CHECK(0.0~1.0): 중요도 점수가 정규화 범위를 벗어나 저장되는 것을
#     DB 레벨에서 원천 차단 — 기존 운영 제약을 그대로 유지한다.
#   - embedding vector(1024): e5-large 임베딩 모델의 출력 차원(1024)과 일치.
#     사양서 Chapter 4.5와 동일하다. 차원이 어긋나면 벡터 검색이 실패한다.
_DDL_TB_MEMORIES = """
CREATE TABLE IF NOT EXISTS tb_memories (
    id            varchar(12) PRIMARY KEY,
    memory_type   varchar(50) NOT NULL,
    content       text NOT NULL,
    key           varchar(255),
    tags          text[],
    importance    double precision,
    access_count  integer DEFAULT 0,
    created_at    timestamptz DEFAULT now(),
    last_accessed timestamptz DEFAULT now(),
    embedding     vector(1024),
    metadata      jsonb DEFAULT '{}'::jsonb,
    CONSTRAINT tb_memories_importance_check
        CHECK (importance >= 0.0 AND importance <= 1.0)
);
"""

# 보조 인덱스 목록 — 운영 DB와 동일. PK(id) 인덱스는 테이블 정의에서 자동 생성되므로
# 여기엔 포함하지 않는다. 각 CREATE INDEX는 IF NOT EXISTS로 감싸 여러 번 실행해도
# 안전(멱등)하다. 인덱스별 목적:
#   - idx_memories_type: memory_type 필터 조회 가속(get_by_type 등)
#   - idx_memories_tags(GIN): 태그 배열 검색 가속
#   - idx_memories_created_at(DESC): 최신순 정렬 조회 가속(get_all 등)
#   - idx_memories_importance(DESC): 중요도순 정렬 가속
#   - idx_memories_embedding(hnsw): 벡터 유사도 검색 가속
# hnsw 선택 근거: 200K+ 행 규모에서 ivfflat 대비 검색 정확도·재현율이 모두 우수.
# 참고: 인덱스 빌드는 행이 적은 빈 테이블에선 즉시 끝나지만, 데이터가 많으면
# 시간이 걸린다(초기 부트스트랩 시 한 번만 발생).
_DDL_TB_MEMORIES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memories_type ON tb_memories (memory_type)",
    "CREATE INDEX IF NOT EXISTS idx_memories_tags ON tb_memories USING gin (tags)",
    "CREATE INDEX IF NOT EXISTS idx_memories_created_at "
    "ON tb_memories (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_memories_importance "
    "ON tb_memories (importance DESC)",
    "CREATE INDEX IF NOT EXISTS idx_memories_embedding "
    "ON tb_memories USING hnsw (embedding vector_cosine_ops)",
    # 소유자(테넌트) 조회 가속 — 회상은 항상 metadata->>'owner' 로 좁힌 뒤 거리 계산을
    # 한다(위 _search_by_vector_pg 주석 참고). 소유자 없는 행은 인덱스에서 제외해
    # 크기를 줄인다(코드 RAG 113만 건이 여기 해당).
    "CREATE INDEX IF NOT EXISTS idx_memories_owner "
    "ON tb_memories ((metadata->>'owner')) WHERE metadata ? 'owner'",
]


class LongTermMemory:
    """
    PostgreSQL + pgvector 기반 장기 메모리 저장소.

    이 클래스 하나가 장기 메모리의 모든 입출력 창구다. 상위 계층
    (core/memory/manager.py)은 여기의 공개 메서드만 호출하고, 실제로 DB를
    쓰는지 인메모리 딕셔너리를 쓰는지는 신경 쓸 필요가 없다.

    공개 메서드 그룹:
      - CRUD: add / get / update / delete
      - 검색: search_by_text(키워드) / search_by_vector(의미 유사도)
      - 일괄 조회: get_by_type / get_all
      - 스키마: ensure_schema(부트스트랩 시 테이블·인덱스 보장)

    동작 원리: 각 공개 메서드는 pg_pool이 있으면 대응하는 _xxx_pg() 내부
    메서드로 실제 SQL을 실행하고, pg_pool이 없거나 SQL이 실패하면 파이썬
    딕셔너리(_store) 기반 폴백 로직으로 같은 기능을 흉내 낸다.
    """

    def __init__(self, pg_pool: Any | None = None):
        """
        장기 메모리를 초기화한다.

        pg_pool을 주면 실제 PostgreSQL을 사용하고, 주지 않으면(None) DB 없이
        메모리 딕셔너리만으로 동작하는 폴백 모드로 시작한다. 테스트나 로컬
        개발에서는 보통 None으로 만든다.

        Args:
            pg_pool: asyncpg.Pool 인스턴스 (None이면 인메모리 폴백 모드)
        """
        # 인메모리 폴백 저장소: {memory_id: MemoryEntry} 형태의 딕셔너리.
        # DB가 없거나 PG 연산이 실패했을 때 이 딕셔너리가 임시 저장소 역할을 한다.
        self._store: dict[str, MemoryEntry] = {}
        # asyncpg 연결 풀. None이면 폴백 모드임을 뜻하는 플래그처럼도 쓰인다.
        self._pg = pg_pool

        # 풀이 없으면 폴백 모드로 동작함을 로그로 명확히 남겨 둔다(운영 오해 방지).
        if self._pg is None:
            logger.info("PostgreSQL 풀 없음 — 인메모리 폴백 모드로 동작")

    # ─── 스키마 관리 ───
    # 운영 환경(2026-06-01 기준)에서 추출한 tb_memories 정의를 멱등 생성한다.
    # 이전엔 DDL이 nexus 코드에 없어서 새 환경 셋업 시 수동 마이그레이션이
    # 필요했다(예: 컨테이너 재빌드 후 새 PG 인스턴스). 이 메서드를 부트스트랩
    # Phase 2에서 호출하면 다음을 자동으로 보장한다:
    #   - tb_memories 테이블 (id varchar(12) PK, importance 0~1 CHECK 등)
    #   - 보조 인덱스 5종 (created_at·importance DESC, tags GIN, type, hnsw)
    # hnsw(embedding vector_cosine_ops)를 선택한 이유: 운영 DB가 이미 hnsw로
    # 운영 중이고 200K+ 행 규모에서 ivfflat보다 검색 정확도/지연 모두 우수.
    # pgvector >= 0.5.0 필요 — 운영은 0.8.0(pgvector/pgvector:pg17).
    async def ensure_schema(self) -> None:
        """
        tb_memories 스키마와 인덱스를 멱등적으로 보장한다.

        pg_pool이 None이면(인메모리 폴백) 아무 작업도 하지 않는다.
        """
        if self._pg is None:
            logger.debug("ensure_schema 건너뜀 (인메모리 폴백)")
            return

        # 풀에서 커넥션 하나를 빌려 순서대로 실행한다: 확장 → 테이블 → 인덱스.
        async with self._pg.acquire() as conn:
            # (1) pgvector 확장 먼저 켠다. 이게 없으면 vector 타입 컬럼과
            #     embedding 인덱스를 만들 수 없어 뒤 단계가 전부 실패한다.
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # (2) 테이블 생성 — 위에서 정의한 운영 정의(_DDL_TB_MEMORIES)와 1:1.
            await conn.execute(_DDL_TB_MEMORIES)
            # (3) 보조 인덱스들을 하나씩 생성. 각 DDL이 IF NOT EXISTS라 재실행 안전.
            for ddl in _DDL_TB_MEMORIES_INDEXES:
                await conn.execute(ddl)
        logger.info("tb_memories 스키마 확인/생성 완료")

    # ─── CRUD 기본 연산 ───

    async def add(self, entry: MemoryEntry) -> str:
        """
        새 메모리를 저장한다.

        이 메서드부터 delete까지 CRUD 메서드는 모두 같은 패턴을 따른다:
        "pg_pool이 있으면 _xxx_pg()로 실제 SQL 실행을 시도하고, 예외가 나면
        경고 로그를 남긴 뒤 그대로 아래 인메모리 폴백으로 흘러간다." 이렇게 하면
        DB가 잠깐 불안정해도 서비스가 죽지 않는다(가용성 우선).

        Args:
            entry: 저장할 MemoryEntry

        Returns:
            저장된 메모리의 ID(성공/폴백 모두 entry.id를 그대로 돌려준다)
        """
        # DB 경로: 풀이 있으면 먼저 PG INSERT를 시도한다. 성공하면 즉시 반환.
        if self._pg is not None:
            try:
                return await self._add_pg(entry)
            except Exception as e:
                # 실패해도 예외를 삼키고 폴백으로 진행 — 저장 자체는 이어간다.
                logger.warning("PostgreSQL add 실패: %s — 폴백 사용", e)

        # 인메모리 폴백: 딕셔너리에 그대로 넣는다.
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
            # MemoryEntry는 Pydantic 모델이라 필드를 직접 수정하지 않고,
            # model_copy(update=...)로 값만 바꾼 "새 객체"를 만든다(불변성 유지).
            # 접근할 때마다 access_count를 1 늘리고 last_accessed를 현재로 갱신 —
            # 나중에 자주/최근에 쓰인 기억을 우대하는 근거 데이터가 된다.
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
        owner: str | None = None,
    ) -> list[MemoryEntry]:
        """
        텍스트 키워드로 메모리를 검색한다.

        content 필드에서 대소문자 무시 매칭을 수행한다.
        PostgreSQL에서는 ILIKE, 폴백에서는 파이썬 문자열 매칭을 사용한다.

        Args:
            query: 검색 쿼리 문자열
            memory_type: 특정 타입으로 필터링 (None이면 전체)
            top_k: 최대 반환 건수
            owner: 소유자(테넌트) 필터. 주면 그 소유자의 기억만 본다.
                None이면 필터 없음(소유 개념이 없는 호출부의 기존 동작 유지).

        Returns:
            관련도 순 MemoryEntry 목록
        """
        if self._pg is not None:
            try:
                return await self._search_by_text_pg(query, memory_type, top_k, owner)
            except Exception as e:
                logger.warning("PostgreSQL 텍스트 검색 실패: %s — 폴백 사용", e)

        # 인메모리 폴백: 인덱스가 없으니 전체를 훑으며 파이썬 문자열 매칭을 한다.
        query_lower = query.lower()  # 대소문자 무시를 위해 소문자로 통일
        results: list[MemoryEntry] = []

        for entry in self._store.values():
            # 타입 필터: 특정 타입만 원하면 나머지는 건너뛴다.
            if memory_type is not None and entry.memory_type != memory_type:
                continue

            # 소유자 필터 — DB 경로와 같은 규칙(소유자 없는 행은 제외).
            if owner is not None and entry.metadata.get("owner") != owner:
                continue

            # content뿐 아니라 key와 tags까지 한 문자열로 합쳐 폭넓게 매칭한다.
            searchable = f"{entry.content} {entry.key} {' '.join(entry.tags)}".lower()
            if query_lower in searchable:
                results.append(entry)

        # 정렬 기준 = 중요도 × (1 + 접근횟수). 중요하고 자주 쓰인 기억일수록 위로.
        # (1 + access_count)로 감싸는 이유: access_count가 0이어도 importance가
        # 0으로 죽지 않도록 하기 위함. 이 점수 공식은 PG의 ORDER BY와 동일하다.
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
        owner: str | None = None,
    ) -> list[MemoryEntry]:
        """
        벡터 유사도로 메모리를 검색한다.

        pgvector의 cosine distance (<=>)를 사용하여
        의미적으로 유사한 메모리를 찾는다.

        Args:
            embedding: 쿼리 벡터 (e5-large 등으로 생성)
            memory_type: 특정 타입으로 필터링 (None이면 전체)
            top_k: 최대 반환 건수
            owner: 소유자(테넌트) 필터. 주면 그 소유자의 기억만 본다.
                None이면 필터 없음(코드 RAG 등 소유 개념이 없는 호출부).

        Returns:
            유사도 순 MemoryEntry 목록
        """
        if self._pg is not None:
            try:
                return await self._search_by_vector_pg(embedding, memory_type, top_k, owner)
            except Exception as e:
                logger.warning("PostgreSQL 벡터 검색 실패: %s — 폴백 사용", e)

        # 인메모리 폴백: pgvector가 없으니 코사인 유사도를 파이썬으로 직접 계산한다.
        # (유사도, 항목) 튜플을 모아 두었다가 마지막에 정렬한다.
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._store.values():
            # 타입 필터: 특정 타입만 원하면 나머지는 건너뛴다.
            if memory_type is not None and entry.memory_type != memory_type:
                continue

            # 소유자 필터 — DB 경로와 같은 규칙(소유자 없는 행은 제외).
            if owner is not None and entry.metadata.get("owner") != owner:
                continue

            # 임베딩이 없는 항목은 유사도 비교 자체가 불가능하므로 제외한다.
            if entry.embedding is None:
                continue

            # 질의 벡터와 저장된 벡터의 코사인 유사도(-1~1)를 구한다.
            similarity = self._cosine_similarity(embedding, entry.embedding)
            scored.append((similarity, entry))

        # 유사도가 높은 순으로 정렬한 뒤 상위 top_k개만 반환한다.
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

        # 아무 필드나 덮어쓰지 못하도록 화이트리스트를 둔다. id/created_at 같은
        # 불변 필드는 여기 없어서 kwargs로 넘어와도 무시된다(안전장치).
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
        # kwargs 중 화이트리스트에 든 필드만 골라낸다.
        update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}

        # 유효한 변경 항목이 하나도 없으면 조기 반환(잘못된 호출로 간주).
        if not update_data:
            logger.warning("업데이트할 유효한 필드 없음 (id=%s)", memory_id)
            return False

        # 불변 모델이므로 model_copy로 값만 바꾼 새 객체를 만들어 교체한다.
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
        """PostgreSQL에 메모리 한 건을 INSERT한다(공개 add의 DB 경로)."""
        # 벡터 컬럼은 asyncpg에 문자열 형태("[0.1, 0.2, ...]")로 넘겨야 pgvector가
        # 파싱한다. 임베딩이 없으면 NULL로 저장하기 위해 None을 그대로 둔다.
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
            # metadata(jsonb 컬럼)는 dict를 JSON 문자열로 직렬화해 넣는다.
            # ensure_ascii=False로 한글이 깨지지 않게, default=str로 datetime 등
            # 직렬화 불가 타입도 문자열로 안전하게 변환한다.
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
        self,
        query: str,
        memory_type: MemoryType | None,
        top_k: int,
        owner: str | None = None,
    ) -> list[MemoryEntry]:
        """PostgreSQL ILIKE 기반 텍스트 검색.

        owner 를 주면 metadata->>'owner' 가 일치하는 행만 본다(소유자 없는 레거시 행은
        제외 = fail-closed). 필터를 SQL 에서 거는 이유는 벡터 검색과 같다 — 파이썬에서
        걸면 LIMIT 이 남의 기억까지 세어 내 기억이 밀려난다.
        """
        if owner is not None:
            conditions = ["content ILIKE $1", "metadata->>'owner' = $2"]
            params: list[Any] = [f"%{query}%", owner]
            if memory_type is not None:
                params.append(memory_type)
                conditions.append(f"memory_type = ${len(params)}")
            params.append(top_k)
            # 안전: 위 벡터 검색과 동일 — 조건 문자열은 고정, 값은 전부 파라미터 바인딩.
            query = f"""
                SELECT * FROM tb_memories
                WHERE {" AND ".join(conditions)}
                ORDER BY importance * (1 + access_count) DESC
                LIMIT ${len(params)}
            """  # noqa: S608
            rows = await self._pg.fetch(query, *params)
            return [self._row_to_entry(row) for row in rows]

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
        self,
        embedding: list[float],
        memory_type: MemoryType | None,
        top_k: int,
        owner: str | None = None,
    ) -> list[MemoryEntry]:
        """
        pgvector 코사인 거리(<=>) 기반 벡터 검색(공개 search_by_vector의 DB 경로).

        <=>는 "거리"라서 값이 작을수록 유사하다. 그래서 distance를 ASC(오름차순)로
        정렬해 가장 가까운(=가장 비슷한) 것부터 top_k개를 가져온다.

        owner 를 주면 metadata->>'owner' 가 정확히 일치하는 행만 본다. 소유자 없는
        레거시 행은 이때 제외된다(fail-closed) — 누구 것인지 모르는 기억을 남의
        대화에 주입하지 않기 위해서다. owner=None 이면 필터를 걸지 않는다(코드 RAG
        리트리버처럼 소유 개념이 없는 호출부의 기존 동작 유지).

        ★필터를 파이썬이 아니라 SQL 에서 거는 이유: 파이썬에서 걸면 LIMIT 이 남의
        기억까지 세어 버려, 정작 내 기억은 top_k 안에 못 들어오는 일이 생긴다.
        """
        # 질의 벡터도 pgvector가 이해하도록 문자열로 바꾼 뒤 ::vector로 캐스팅한다.
        embedding_str = str(embedding)
        # WHERE 절을 조건에 맞춰 조립한다. 값은 전부 $N 파라미터 바인딩이라 인젝션 여지가 없다.
        conditions = ["embedding IS NOT NULL"]
        params: list[Any] = [embedding_str]
        if memory_type is not None:
            params.append(memory_type)
            conditions.append(f"memory_type = ${len(params)}")
        if owner is not None:
            params.append(owner)
            conditions.append(f"metadata->>'owner' = ${len(params)}")
        params.append(top_k)

        # ★소유자 필터가 걸리면 '필터 먼저, 거리계산 나중' 순서를 강제한다.
        #   왜: embedding 컬럼에는 HNSW 인덱스가 있는데, HNSW 는 후보 몇십 건을 먼저
        #   훑은 뒤 필터를 적용한다(post-filter). 전체 113만 건 중 내 소유가 몇 건뿐이면
        #   후보 안에 한 건도 안 들어와 **결과가 0건**이 된다(실측: 텍스트 검색은 1건을
        #   찾는데 벡터 검색만 0건이었다).
        #   서브쿼리 안의 OFFSET 0 은 PostgreSQL 의 최적화 펜스다 — 이게 있으면 플래너가
        #   조건을 바깥 정렬로 끌어올리지 못해, 좁혀진 집합만 정확히 거리 계산한다.
        #   소유자로 좁힌 집합은 작으므로 정확 스캔 비용이 문제되지 않는다.
        if owner is not None:
            query = f"""
                SELECT *, embedding <=> $1::vector AS distance
                FROM (
                    SELECT * FROM tb_memories
                    WHERE {" AND ".join(conditions)}
                    OFFSET 0
                ) AS scoped
                ORDER BY distance ASC
                LIMIT ${len(params)}
            """  # noqa: S608
        else:
            # 소유자 필터가 없으면 기존대로 HNSW 인덱스를 그대로 활용한다(코드 RAG 경로).
            query = f"""
                SELECT *, embedding <=> $1::vector AS distance
                FROM tb_memories
                WHERE {" AND ".join(conditions)}
                ORDER BY distance ASC
                LIMIT ${len(params)}
            """  # noqa: S608
        rows = await self._pg.fetch(query, *params)
        return [self._row_to_entry(row) for row in rows]

    async def _update_pg(self, memory_id: str, **kwargs: Any) -> bool:
        """
        PostgreSQL에서 메모리를 업데이트한다(공개 update의 DB 경로).

        어떤 필드가 넘어올지 미리 알 수 없으므로, 넘어온 필드에 맞춰 SET 절을
        동적으로 조립한다. 값 자리는 반드시 $1, $2... 파라미터 바인딩으로 처리해
        SQL 인젝션을 원천 차단한다.
        """
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
        # 화이트리스트에 든 필드만 추려낸다(폴백 update와 동일한 안전장치).
        update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not update_data:
            return False

        # SET 절을 동적으로 만든다. 필드명은 화이트리스트에서만 오고, 값은
        # $1..$N 파라미터로 분리해 바인딩한다.
        set_clauses = []
        params: list[Any] = []
        for i, (field, value) in enumerate(update_data.items(), start=1):
            # jsonb/vector 컬럼은 저장 전 문자열 형태로 변환이 필요하다.
            if field == "metadata":
                value = json.dumps(value, ensure_ascii=False, default=str)
            elif field == "embedding" and value is not None:
                value = str(value)
            # "field = $i" 조각과 그에 대응하는 값을 같은 순서로 쌓는다.
            set_clauses.append(f"{field} = ${i}")
            params.append(value)

        # 마지막 파라미터는 WHERE id = $N에 쓸 memory_id. 그래서 번호는 len(params).
        params.append(memory_id)
        # 안전: set_clauses의 필드명은 allowed_fields 화이트리스트에서만 나오고 값은
        # 전부 파라미터 바인딩이라 SQL 인젝션 위험이 없다(그래서 S608 억제).
        query = f"UPDATE tb_memories SET {', '.join(set_clauses)} WHERE id = ${len(params)}"  # noqa: S608

        result = await self._pg.execute(query, *params)
        # asyncpg의 execute는 "UPDATE N"(N=영향받은 행 수) 문자열을 돌려준다.
        # "UPDATE 0"이면 해당 id가 없어 아무것도 안 바뀐 것 → 실패(False)로 본다.
        return result is not None and "UPDATE 0" not in str(result)

    async def _delete_pg(self, memory_id: str) -> bool:
        """PostgreSQL에서 메모리를 삭제한다(공개 delete의 DB 경로)."""
        result = await self._pg.execute("DELETE FROM tb_memories WHERE id = $1", memory_id)
        # _update_pg와 같은 방식: "DELETE 0"이면 지운 게 없으니 실패로 판단한다.
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
        """
        asyncpg Record(DB 한 행)를 MemoryEntry Pydantic 모델로 변환한다.

        모든 _xxx_pg 조회 메서드가 마지막에 이 함수를 거쳐 결과를 도메인 모델로
        통일한다. DB가 문자열로 돌려주는 jsonb/vector를 파이썬 dict/list로 되돌리는
        역직렬화가 핵심이다.
        """
        import json

        # metadata(jsonb): asyncpg 설정에 따라 문자열로 올 수 있어 dict로 되돌린다.
        # 파싱이 깨지면 빈 dict로 폴백해 예외로 전체 조회가 죽지 않게 한다.
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        # embedding(vector): pgvector가 "[...]" 문자열로 반환하는 경우가 있어
        # list[float]로 되돌린다. 파싱 실패 시 None 처리(임베딩 없음으로 간주).
        embedding = row.get("embedding")
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except (json.JSONDecodeError, TypeError):
                embedding = None

        # 각 컬럼을 MemoryEntry 필드로 옮긴다. NULL 가능성이 있는 값은 .get(기본값)
        # 으로 방어하고, importance/access_count는 명시적으로 형 변환해 둔다.
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
        # 차원이 다르면 비교 자체가 무의미하므로 유사도 0으로 처리한다.
        if len(vec_a) != len(vec_b):
            return 0.0

        # 코사인 유사도 = (내적) / (각 벡터 크기의 곱).
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        # 영벡터는 크기가 0이라 0으로 나누는 것을 막기 위해 0.0을 돌려준다.
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)
