"""
pgvector 기반 저장소 공통 베이스 (2026-04-22 리팩토링 5).

[이 파일이 하는 일 — 한눈에]
Nexus의 장기 기억/지식 검색은 PostgreSQL + pgvector 확장 위에 올라간다.
그런데 지식 저장소(`KnowledgeStore`, tb_knowledge)와 심볼 저장소
(`SymbolStore`, tb_symbols)가 거의 똑같은 "밑작업 코드"를 각자 중복해서
갖고 있었다. 이 파일은 그 중복을 한 곳으로 모은 공통 토대다.
즉, 두 저장소가 상속해서 재사용하는 부모 클래스와, 벡터를 다루는
모듈 공용 함수 몇 개를 제공한다.

[모듈 공용 헬퍼 함수 (클래스 밖, 어디서든 import 가능)]
  - format_vector()     — float 리스트 → pgvector가 이해하는 '[..]' 문자열
  - parse_vector()      — pgvector가 돌려준 문자열 → float 리스트 (역변환)
  - cosine_similarity() — numpy 없이 순수 파이썬으로 코사인 유사도 계산

[베이스 클래스 PgVectorStore가 대신 처리해 주는 공통 로직]
  - ensure_schema()      — 테이블/기본 인덱스 DDL을 멱등(idempotent) 실행
  - build_vector_index() — 대량 적재 후 ivfflat 벡터 인덱스 생성
  - count()              — 전체 또는 소스별 레코드 수 집계
  - pg_pool=None일 때의 인메모리 폴백 (DB 없이 테스트/경량 실행)

[서브클래스가 채워야 하는 계약]
아래 세 개의 클래스 변수만 정의하면 위 공통 로직을 그대로 물려받는다:

  TABLE_NAME   : "tb_knowledge" / "tb_symbols" 등 대상 테이블명
  DDL_SCHEMA   : CREATE TABLE + 보조 인덱스 DDL 문자열
  DDL_IVFFLAT  : ivfflat 벡터 인덱스 생성 DDL 문자열

[일부러 공유하지 않는 것 — 설계 의도]
`async def search_*()` 같은 실제 검색 로직은 여기서 다루지 않는다. 검색
시그니처가 저장소마다 크게 다르기 때문이다(KnowledgeStore는 테넌트별
`allowed_sources` 필터, SymbolStore는 심볼 이름 매칭 등). 억지로 하나로
추상화하면 오히려 호출부가 더 복잡해진다. 그래서 "필요한 만큼만 공유"
한다는 리팩토링 원칙에 따라, add/search/delete는 각 서브클래스가 직접
구현한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("nexus.rag.pgvector_base")


# ─────────────────────────────────────────────
# 모듈 공용 헬퍼 — 두 스토어(KnowledgeStore/SymbolStore)가 공유한다.
# 클래스에 묶지 않고 모듈 레벨 함수로 둔 이유: DB 유무와 무관하게,
# 벡터를 다루는 어느 코드에서나 가볍게 import해서 쓰기 위함이다.
# ─────────────────────────────────────────────
def format_vector(vec: list[float] | tuple[float, ...] | None) -> str | None:
    """임베딩 벡터(float 시퀀스)를 pgvector가 이해하는 문자열로 직렬화한다.

    pgvector의 VECTOR 컬럼에 값을 바인딩하려면 '[v1,v2,...]' 형태의
    문자열이 필요하다. 이 함수는 각 원소를 소수점 6자리로 포맷해
    콤마로 이어 붙여 그 문자열을 만든다.

    매개변수:
      vec — 임베딩 값 리스트/튜플. 임베딩이 없으면 None.

    반환:
      '[0.123456,...]' 형태의 문자열. 단, vec이 None이면 None을
      그대로 돌려줘서 asyncpg가 컬럼에 SQL NULL로 바인딩하게 한다.
    """
    if vec is None:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def parse_vector(
    raw: str | list[float] | tuple[float, ...] | None,
) -> list[float] | None:
    """pgvector VECTOR 컬럼 값을 다시 float 리스트로 되돌린다.

    format_vector()의 정확한 역방향 함수다.

    왜 필요한가:
      asyncpg는 vector 타입 전용 코덱을 따로 등록하지 않으면 pgvector
      값을 "[0.1,0.2,...]" 형태의 '문자열' 그대로 돌려준다. 그런데
      MMR(Maximal Marginal Relevance) 재랭킹은 후보들의 임베딩끼리
      코사인 유사도를 직접 계산해야 하므로, 이 문자열을 숫자 리스트로
      복원해 둘 필요가 있다.

    입력을 관대하게 받는다(들어오는 형태가 상황마다 다르기 때문):
      - str        : "[v1,v2,...]" (대괄호가 있든 없든 모두 허용) → 파싱
      - list/tuple : 이미 숫자 시퀀스면 float 리스트로만 변환
      - None       : 임베딩이 없다는 뜻이므로 None 반환

    반환:
      float 리스트. 단, 값이 비었거나 파싱에 실패하면 None을 반환해
      호출자(MMR 등)가 해당 후보를 안전하게 건너뛸 수 있게 한다.
    """
    if raw is None:
        return None
    # 이미 리스트/튜플이면 파싱할 것 없이 float로만 정규화한다.
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    text = raw.strip()
    if not text:
        return None
    # "[...]" 로 감싸여 있으면 앞뒤 대괄호를 벗겨 알맹이만 남긴다.
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return None
    try:
        # 콤마로 쪼갠 뒤 공백만 있는 조각은 버리고 나머지를 float로 변환.
        return [float(part) for part in text.split(",") if part.strip() != ""]
    except ValueError:
        # 예상 못 한 포맷이면 예외를 터뜨리지 않고 조용히 None을 준다.
        # MMR은 임베딩이 없는 후보를 자연스럽게 스킵하므로 이게 안전하다.
        return None


def cosine_similarity(
    a: list[float] | tuple[float, ...],
    b: list[float] | tuple[float, ...],
) -> float:
    """두 벡터의 코사인 유사도를 계산한다 (numpy 없는 순수 파이썬 구현).

    코사인 유사도 = (a·b) / (|a| * |b|). 값이 1에 가까울수록 두 벡터의
    방향이 비슷하다(= 의미가 유사하다)는 뜻이다.

    이 함수는 주로 DB가 없는 인메모리 폴백 경로에서 쓴다. 실제 DB가
    있으면 pgvector가 SQL 안에서 유사도를 직접 계산하므로, 그때는
    이 파이썬 구현이 필요 없다. numpy에 의존하지 않게 만든 이유는
    에어갭 경량 실행/테스트 환경에서도 추가 패키지 없이 돌리기 위함.

    매개변수:
      a, b — 같은 차원의 임베딩 벡터.

    반환:
      -1.0 ~ 1.0 범위의 코사인 유사도. 단, 두 벡터의 차원이 다르거나
      한쪽이 비었거나 영(0)벡터면 0으로 나누는 것을 막기 위해 0.0을
      반환한다(= "유사도를 판단할 수 없음"을 안전하게 표현).
    """
    # 빈 벡터거나 길이가 다르면 계산 자체가 무의미 → 0.0.
    if not a or not b or len(a) != len(b):
        return 0.0
    # 분자: 두 벡터의 내적(대응 원소 곱의 합).
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    # 분모: 각 벡터의 크기(유클리드 노름) = 제곱합의 제곱근.
    na = (sum(x * x for x in a)) ** 0.5
    nb = (sum(y * y for y in b)) ** 0.5
    # 어느 한쪽이 영벡터면 나눗셈이 불가능하므로 0.0으로 방어한다.
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ─────────────────────────────────────────────
# PgVectorStore — 공통 스키마·카운트 로직의 베이스
# ─────────────────────────────────────────────
class PgVectorStore:
    """pgvector 기반 저장소들이 공통으로 물려받는 부모 클래스.

    KnowledgeStore / SymbolStore 등이 이 클래스를 상속해서, 스키마
    생성·인덱스 생성·레코드 카운트·인메모리 폴백 같은 반복 로직을
    직접 짜지 않고 그대로 재사용한다.

    서브클래스가 지켜야 할 규약:
      - 클래스 변수 `TABLE_NAME`, `DDL_SCHEMA`, `DDL_IVFFLAT`을 반드시
        오버라이드해서 값을 채운다. (안 채우면 관련 메서드가
        NotImplementedError를 던진다.)
      - 인메모리 폴백일 때 `self._store`는 `dict[str, <Entry>]` 형태로
        유지하며, `<Entry>`는 최소한 `.source` 속성을 가져야 한다.
        (count(source=...)가 그 속성으로 소스별 개수를 세기 때문)

    add / search_* / delete_* 같은 실제 데이터 조작 메서드는 저장소마다
    시그니처가 다르므로, 이 부모가 아니라 각 서브클래스가 직접 구현한다.
    """

    # 아래 세 변수는 반드시 서브클래스에서 실제 값으로 오버라이드해야 한다.
    TABLE_NAME: str = ""       # 대상 테이블명 (예: "tb_knowledge")
    DDL_SCHEMA: str = ""       # CREATE TABLE + 보조 인덱스 DDL
    DDL_IVFFLAT: str = ""      # ivfflat 벡터 인덱스 생성 DDL

    def __init__(self, pg_pool: Any | None = None) -> None:
        # pg_pool을 주면 실제 PostgreSQL 커넥션 풀을 사용한다.
        # None이면 DB 없이 도는 인메모리 폴백 모드 — 주로 테스트나
        # 경량 실행 환경에서 쓰인다.
        self._pg = pg_pool
        # 인메모리 폴백에서 레코드를 담아둘 딕셔너리.
        # Entry 타입이 서브클래스마다 달라서 값 타입은 Any로 둔다.
        self._store: dict[str, Any] = {}

    # ─── 스키마 관리 ─────────────────────────────────
    async def ensure_schema(self) -> None:
        """테이블과 기본 인덱스를 만든다 (DDL_SCHEMA를 멱등 실행).

        멱등(idempotent)이란 여러 번 호출해도 결과가 같다는 뜻이다.
        DDL_SCHEMA는 보통 "CREATE TABLE IF NOT EXISTS ..." 형태라,
        앱을 켤 때마다 안심하고 불러도 테이블이 중복 생성되지 않는다.

        DB가 없으면(pg_pool=None, 인메모리 폴백) 만들 테이블도 없으므로
        로그만 남기고 조용히 넘어간다.
        """
        if self._pg is None:
            logger.info(
                "%s: pg_pool 없음 — 스키마 건너뜀 (인메모리 폴백)",
                self.__class__.__name__,
            )
            return
        # 서브클래스가 DDL_SCHEMA를 채우지 않았으면 명확히 실패시킨다.
        if not self.DDL_SCHEMA:
            raise NotImplementedError(
                f"{self.__class__.__name__}.DDL_SCHEMA가 비어 있음"
            )
        # 풀에서 커넥션을 하나 빌려 DDL을 실행하고, 블록을 벗어나면 반납.
        async with self._pg.acquire() as conn:
            await conn.execute(self.DDL_SCHEMA)
        logger.info("%s 스키마 확인/생성 완료", self.TABLE_NAME)

    async def build_vector_index(self) -> None:
        """ivfflat 벡터 인덱스를 만든다 (대량 적재를 끝낸 뒤 호출 권장).

        ivfflat 인덱스는 데이터를 여러 군집으로 나눠 근사 최근접 탐색을
        빠르게 해 준다. 그런데 데이터가 거의 없는 상태에서 만들면 군집이
        데이터 분포를 제대로 반영하지 못한다. 그래서 대량 적재로 벡터
        분포가 충분히 쌓인 뒤에 이 메서드를 호출하는 것이 좋다.

        인메모리 폴백이면 인덱스가 필요 없으므로 아무 것도 하지 않는다.
        """
        if self._pg is None:
            return
        # 서브클래스가 DDL_IVFFLAT을 채우지 않았으면 명확히 실패시킨다.
        if not self.DDL_IVFFLAT:
            raise NotImplementedError(
                f"{self.__class__.__name__}.DDL_IVFFLAT가 비어 있음"
            )
        async with self._pg.acquire() as conn:
            await conn.execute(self.DDL_IVFFLAT)
        logger.info("%s 벡터 인덱스 생성 완료 (ivfflat cosine)", self.TABLE_NAME)

    # ─── 운영 유틸 ──────────────────────────────────
    async def count(self, source: str | None = None) -> int:
        """저장된 레코드 수를 센다 — 전체, 또는 특정 소스만.

        매개변수:
          source — None이면 전체 레코드 수, 값이 주어지면 그 소스에
                   속한 레코드 수만 센다(예: 특정 문서 출처/테넌트).

        반환:
          정수 개수. DB가 있으면 SQL COUNT(*)로, 인메모리 폴백이면
          _store 딕셔너리를 직접 세어서 같은 의미의 값을 돌려준다.
        """
        # ── 인메모리 폴백 경로 ──
        if self._pg is None:
            if source is None:
                return len(self._store)
            # 각 Entry의 .source 속성을 보고 요청한 소스만 카운트한다.
            return sum(
                1 for e in self._store.values() if getattr(e, "source", None) == source
            )
        # ── 실제 DB 경로 ──
        # 서브클래스가 TABLE_NAME을 채우지 않았으면 명확히 실패시킨다.
        if not self.TABLE_NAME:
            raise NotImplementedError(
                f"{self.__class__.__name__}.TABLE_NAME이 비어 있음"
            )
        async with self._pg.acquire() as conn:
            if source is None:
                # noqa: S608 — TABLE_NAME은 외부 입력이 아니라 클래스 상수라
                # SQL 인젝션 위험이 없다(그래서 f-string 삽입을 허용).
                return await conn.fetchval(
                    f"SELECT COUNT(*) FROM {self.TABLE_NAME}"  # noqa: S608 — TABLE_NAME은 클래스 상수
                )
            # source 값은 반드시 $1 파라미터로 바인딩해 인젝션을 막는다.
            # (테이블명만 상수 삽입, 사용자 값은 절대 문자열 조립 안 함)
            return await conn.fetchval(
                f"SELECT COUNT(*) FROM {self.TABLE_NAME} WHERE source = $1",  # noqa: S608
                source,
            )
