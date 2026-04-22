"""
pgvector 기반 저장소 공통 베이스 (2026-04-22 리팩토링 5).

`KnowledgeStore`와 `SymbolStore`가 똑같이 반복하던 다음 로직을 한 곳으로 묶는다:

  - `_format_vector()` / `_cosine()` — 벡터 직렬화·내적 헬퍼
  - `ensure_schema()` — DDL 멱등 실행
  - `build_vector_index()` — ivfflat 인덱스 생성
  - `count()` — 전체/소스별 레코드 수
  - `pg_pool=None`일 때 인메모리 폴백

서브클래스는 다음을 클래스 변수로 채워 넣기만 하면 된다:

  TABLE_NAME   : "tb_knowledge" / "tb_symbols" 등
  DDL_SCHEMA   : CREATE TABLE + 보조 인덱스 DDL
  DDL_IVFFLAT  : ivfflat 벡터 인덱스 DDL

이 베이스는 `async def search_*()` 같은 검색 로직은 일부러 다루지 않는다 — 검색
시그니처는 각 저장소마다 다르고(KnowledgeStore는 `allowed_sources`, SymbolStore는
이름 매칭 등), 추상화하면 오히려 콜러 쪽이 복잡해진다 (리팩토링의 원칙: 필요한
만큼만 공유).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("nexus.rag.pgvector_base")


# ─────────────────────────────────────────────
# 모듈 공용 헬퍼 — 두 스토어(KnowledgeStore/SymbolStore)가 공유한다.
# ─────────────────────────────────────────────
def format_vector(vec: list[float] | tuple[float, ...] | None) -> str | None:
    """pgvector VECTOR 타입에 넘길 수 있는 '[v1,v2,...]' 문자열을 만든다.

    None이 들어오면 None을 그대로 반환 — asyncpg가 NULL로 바인딩한다.
    """
    if vec is None:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def cosine_similarity(
    a: list[float] | tuple[float, ...],
    b: list[float] | tuple[float, ...],
) -> float:
    """인메모리 폴백용 코사인 유사도 — numpy 의존 없는 순수 파이썬 구현.

    차원이 다르거나 영벡터면 0.0을 반환해 divide-by-zero를 방어한다.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = (sum(x * x for x in a)) ** 0.5
    nb = (sum(y * y for y in b)) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ─────────────────────────────────────────────
# PgVectorStore — 공통 스키마·카운트 로직의 베이스
# ─────────────────────────────────────────────
class PgVectorStore:
    """pgvector 기반 저장소의 공통 부모.

    서브클래스 규약:
      - 클래스 변수 `TABLE_NAME`, `DDL_SCHEMA`, `DDL_IVFFLAT`을 반드시 정의
      - 인메모리 폴백 시 `self._store`는 `dict[str, <Entry>]` 형태로 유지하며,
        `<Entry>`는 최소한 `.source` 속성을 가져야 한다 (`count(source=...)`용)

    서브클래스는 자체 `add/search_*/delete_*` 메서드를 직접 구현한다.
    """

    TABLE_NAME: str = ""       # 반드시 오버라이드
    DDL_SCHEMA: str = ""       # 반드시 오버라이드
    DDL_IVFFLAT: str = ""      # 반드시 오버라이드

    def __init__(self, pg_pool: Any | None = None) -> None:
        # pg_pool이 None이면 인메모리 폴백 — 테스트/경량 실행 환경에서 쓰인다.
        self._pg = pg_pool
        # 서브클래스마다 Entry 타입이 다르므로 Any로 둔다.
        self._store: dict[str, Any] = {}

    # ─── 스키마 관리 ─────────────────────────────────
    async def ensure_schema(self) -> None:
        """DDL_SCHEMA를 멱등 실행해 테이블·기본 인덱스를 만든다."""
        if self._pg is None:
            logger.info(
                "%s: pg_pool 없음 — 스키마 건너뜀 (인메모리 폴백)",
                self.__class__.__name__,
            )
            return
        if not self.DDL_SCHEMA:
            raise NotImplementedError(
                f"{self.__class__.__name__}.DDL_SCHEMA가 비어 있음"
            )
        async with self._pg.acquire() as conn:
            await conn.execute(self.DDL_SCHEMA)
        logger.info("%s 스키마 확인/생성 완료", self.TABLE_NAME)

    async def build_vector_index(self) -> None:
        """대량 적재 후 ivfflat 벡터 인덱스를 만든다 (분포 확보 후 권장)."""
        if self._pg is None:
            return
        if not self.DDL_IVFFLAT:
            raise NotImplementedError(
                f"{self.__class__.__name__}.DDL_IVFFLAT가 비어 있음"
            )
        async with self._pg.acquire() as conn:
            await conn.execute(self.DDL_IVFFLAT)
        logger.info("%s 벡터 인덱스 생성 완료 (ivfflat cosine)", self.TABLE_NAME)

    # ─── 운영 유틸 ──────────────────────────────────
    async def count(self, source: str | None = None) -> int:
        """전체 또는 특정 소스의 레코드 수를 돌려준다."""
        if self._pg is None:
            if source is None:
                return len(self._store)
            return sum(
                1 for e in self._store.values() if getattr(e, "source", None) == source
            )
        if not self.TABLE_NAME:
            raise NotImplementedError(
                f"{self.__class__.__name__}.TABLE_NAME이 비어 있음"
            )
        async with self._pg.acquire() as conn:
            if source is None:
                return await conn.fetchval(
                    f"SELECT COUNT(*) FROM {self.TABLE_NAME}"  # noqa: S608 — TABLE_NAME은 클래스 상수
                )
            return await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.TABLE_NAME} WHERE source = $1",  # noqa: S608
                source,
            )
