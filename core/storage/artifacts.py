# 생성물(이미지·문서) 메타데이터를 PostgreSQL tb_artifacts에 기록/조회하는 fail-soft 헬퍼.
"""
tb_artifacts 저장소 헬퍼 — 생성물의 "메타데이터·소유자"만 PostgreSQL에 남긴다.

[이 파일이 하는 일 — 한눈에]
  DocumentExport / ImageGenerate 같은 생성 도구가 만든 파일의 "바이트"는
  파일시스템(exports_dir)에 두고, 여기서는 그 파일의 메타데이터(파일명·테넌트·
  세션·MIME·크기·해시)만 tb_artifacts 테이블에 기록한다.

[왜 바이너리를 DB에 넣지 않는가]
  이 프로젝트의 PostgreSQL은 지식 RAG(tb_knowledge/tb_symbols/tb_memories)용으로
  튜닝된 "공유" 컨테이너다. 여기에 문서/이미지 blob을 넣으면 버퍼 캐시가 오염되고
  백업이 비대해져 RAG 검색 성능이 나빠진다. 그래서 blob은 절대 넣지 않고
  메타데이터만 남긴다(파일은 파일시스템이 진실의 원천).

[fail-soft 원칙 — 요청/도구를 절대 깨뜨리지 않는다]
  모든 공개 함수는 pg_pool이 None(로컬/테스트/DB 미가동)이거나 SQL이 실패해도
  예외를 밖으로 던지지 않는다. 로깅만 하고 조용히 넘어간다. 생성물 메타 기록은
  "있으면 좋은" 부가 기능이지, 다운로드/생성 자체를 막을 이유가 아니기 때문이다
  (가용성 우선).

[의존성 방향]
  core/storage/ 는 core 하위의 순수 저장 계층이다. web/ 이 이 모듈을 import하며
  (web → core 단방향), 이 모듈은 web/이나 core/tools/를 import하지 않는다.

작성자: 이현수 / 작성일: 2026-07-09
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger("nexus.storage.artifacts")


# ─────────────────────────────────────────────
# "행 없음" 센티넬 — get_artifact_owner 반환값 구분용
# ─────────────────────────────────────────────
# get_artifact_owner는 세 가지 결과를 구분해서 돌려줘야 한다:
#   (1) 아예 행이 없음(레거시/기록 전 파일)        → ARTIFACT_NOT_FOUND
#   (2) 행은 있으나 tenant_id 컬럼이 NULL(소유자 미상) → None
#   (3) 행이 있고 tenant_id가 특정 테넌트           → 그 문자열
# (1)과 (2)를 모두 None으로 표현하면 다운로드 라우트가 "레거시 통과"와
# "소유자 미상"을 구분할 수 없다. 그래서 (1)에는 전용 센티넬 객체를 쓴다.
class _ArtifactNotFound:
    """tb_artifacts에 해당 파일명 행이 존재하지 않음을 나타내는 센티넬 타입."""

    __slots__ = ()

    def __repr__(self) -> str:  # 로그/디버깅 가독성용
        return "ARTIFACT_NOT_FOUND"


# 모듈 전역 단일 센티넬 인스턴스(비교는 `is`로 한다).
ARTIFACT_NOT_FOUND: Final = _ArtifactNotFound()


# ─────────────────────────────────────────────
# 스키마 DDL — 기존 tb_* 헬퍼(long_term/pgvector_base)와 동일한 멱등 패턴
# ─────────────────────────────────────────────
# filename에 UNIQUE 제약을 걸어 "한 파일명 = 한 행"을 보장한다(UNIQUE 제약은
# 조회용 인덱스를 자동 생성하므로 별도 인덱스는 만들지 않는다). ON CONFLICT
# (filename) DO NOTHING 멱등 기록의 근거가 되는 제약이다.
# 바이너리 컬럼은 의도적으로 두지 않는다(바이트는 파일시스템에만 존재).
_DDL_TB_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS tb_artifacts (
    id          BIGSERIAL PRIMARY KEY,
    filename    TEXT UNIQUE NOT NULL,
    tenant_id   TEXT,
    user_id     TEXT,
    session_id  TEXT,
    mime        TEXT,
    size_bytes  BIGINT,
    sha256      TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
"""

# 보조 인덱스 — created_at은 retention 정리(cleanup_expired_artifacts)의
# 만료 스캔을 가속한다. IF NOT EXISTS라 여러 번 실행해도 안전(멱등).
_DDL_TB_ARTIFACTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_artifacts_created_at "
    "ON tb_artifacts (created_at DESC)",
]


# ─────────────────────────────────────────────
# 스키마 보장
# ─────────────────────────────────────────────
async def ensure_artifacts_schema(pool: Any | None) -> None:
    """tb_artifacts 테이블과 인덱스를 멱등적으로 보장한다(fail-soft).

    pool이 None(DB 미가동/테스트)이면 아무 것도 하지 않는다. DDL 실행 중
    오류가 나도 예외를 밖으로 던지지 않고 경고만 남긴다 — 스키마 준비 실패가
    웹 서버 기동 자체를 막지 않게 한다.

    매개변수:
      pool — asyncpg.Pool 인스턴스(None이면 no-op).
    """
    if pool is None:
        logger.debug("tb_artifacts ensure_schema 건너뜀 (pg_pool 없음)")
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(_DDL_TB_ARTIFACTS)
            for ddl in _DDL_TB_ARTIFACTS_INDEXES:
                await conn.execute(ddl)
        logger.info("tb_artifacts 스키마 확인/생성 완료")
    except Exception as e:  # noqa: BLE001 — fail-soft: 기동을 막지 않는다
        logger.warning("tb_artifacts ensure_schema 실패(무시): %s", e)


# ─────────────────────────────────────────────
# 메타데이터 기록 (멱등)
# ─────────────────────────────────────────────
async def record_artifact(
    pool: Any | None,
    filename: str,
    tenant_id: str | None,
    session_id: str | None,
    mime: str | None,
    size_bytes: int | None,
    sha256: str | None = None,
    user_id: str | None = None,
) -> None:
    """생성물 1건의 메타데이터를 tb_artifacts에 기록한다(멱등·fail-soft).

    ON CONFLICT (filename) DO NOTHING을 쓰므로 같은 파일명으로 재호출해도
    중복 행이 생기지 않는다(재시도·중복 응답 경로에서 안전). pool이 없거나
    INSERT가 실패해도 예외를 던지지 않는다 — 기록 실패가 생성/다운로드를
    깨뜨리면 안 되기 때문이다(가용성 우선).

    매개변수:
      pool       — asyncpg.Pool(None이면 조용히 스킵).
      filename   — 저장된 파일명(exports_dir 기준 basename, UNIQUE 키).
      tenant_id  — 소유 테넌트 식별자(없으면 None → 소유자 미상).
      session_id — 생성이 일어난 세션 ID(추적용).
      mime       — MIME 타입(확장자 매핑 결과).
      size_bytes — 파일 크기(바이트). stat 실패 시 None.
      sha256     — 파일 내용 해시(선택 — 큰 파일은 성능상 생략 가능).
      user_id    — 사용자 식별자(로그인 도입 전이라 보통 None).
    """
    if pool is None:
        logger.debug("record_artifact 스킵 (pg_pool 없음): %s", filename)
        return
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tb_artifacts
                    (filename, tenant_id, user_id, session_id, mime, size_bytes, sha256)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (filename) DO NOTHING
                """,
                filename,
                tenant_id,
                user_id,
                session_id,
                mime,
                size_bytes,
                sha256,
            )
    except Exception as e:  # noqa: BLE001 — fail-soft: 요청/도구를 깨지 않는다
        logger.warning("record_artifact 실패(무시): %s — %s", filename, e)


# ─────────────────────────────────────────────
# 소유자 조회 (다운로드 IDOR 판정용)
# ─────────────────────────────────────────────
async def get_artifact_owner(pool: Any | None, filename: str) -> Any:
    """파일명으로 소유 테넌트를 조회한다. "행 없음"과 "tenant None"을 구분한다.

    반환값 세 가지(다운로드 라우트가 통과/차단을 판정하는 근거):
      - ARTIFACT_NOT_FOUND : 행이 없음(레거시/기록 전 파일). 라우트는 '통과'.
      - None               : 행은 있으나 소유자 미상(tenant_id NULL). 라우트는 '통과'.
      - "<tenant_id>"      : 특정 테넌트 소유. 요청 테넌트와 불일치 시 라우트는 '404'.

    fail-soft: pool이 없거나 조회가 실패하면 ARTIFACT_NOT_FOUND를 돌려준다
    (= 판정 불가 → 가용성 우선으로 통과시키게 한다).

    매개변수:
      pool     — asyncpg.Pool(None이면 ARTIFACT_NOT_FOUND).
      filename — 조회할 파일명(basename).
    """
    if pool is None:
        return ARTIFACT_NOT_FOUND
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT tenant_id FROM tb_artifacts WHERE filename = $1",
                filename,
            )
    except Exception as e:  # noqa: BLE001 — fail-soft: 조회 실패는 '판정 불가'로 취급
        logger.warning("get_artifact_owner 실패(무시): %s — %s", filename, e)
        return ARTIFACT_NOT_FOUND
    if row is None:
        return ARTIFACT_NOT_FOUND
    # 행이 있으면 tenant_id 컬럼 값을 그대로 돌려준다(NULL이면 None).
    return row["tenant_id"]


# ─────────────────────────────────────────────
# 보존기간 정리 (증식 방지) — 함수만 제공, 스케줄 배선은 하지 않는다
# ─────────────────────────────────────────────
async def cleanup_expired_artifacts(
    pool: Any | None,
    exports_dir: str | Path,
    retention_days: int,
) -> int:
    """보존기간이 지난 생성물의 파일과 tb_artifacts 행을 함께 삭제한다(fail-soft).

    만료 기준: created_at < now() - retention_days. 각 행에 대해 파일을 먼저
    지우고(있으면) 그 다음 행을 삭제한다. 파일 하나가 안 지워져도(권한/부재)
    전체를 중단하지 않고 다음으로 넘어간다(가용성 우선).

    TODO(nexus): 후속 — 스케줄러/기동 시 이 함수를 주기 호출하도록 배선한다.
      이번 작업에서는 함수와 테스트만 제공하고, 호출/크론 배선은 하지 않는다.

    매개변수:
      pool           — asyncpg.Pool(None이면 0 반환).
      exports_dir    — 생성물 파일이 저장된 디렉토리(파일 삭제 기준 경로).
      retention_days — 보존 일수. 0 이하이면 정리하지 않는다(안전장치).

    반환:
      삭제한 행 수(파일 삭제 실패와 무관하게 DB 행 삭제 기준). 오류 시 0.
    """
    if pool is None:
        return 0
    # 0 이하 보존기간은 "모두 삭제"로 오작동할 수 있어 안전하게 no-op 처리.
    if retention_days <= 0:
        logger.debug("cleanup 스킵 — retention_days=%s (0 이하)", retention_days)
        return 0

    base = Path(exports_dir)
    deleted = 0
    try:
        async with pool.acquire() as conn:
            # make_interval(days => $1)로 정수 일수를 안전하게 interval로 바꾼다
            # (문자열 조립 없이 파라미터 바인딩 → 인젝션 여지 없음).
            rows = await conn.fetch(
                "SELECT filename FROM tb_artifacts "
                "WHERE created_at < now() - make_interval(days => $1)",
                retention_days,
            )
            for row in rows:
                fname = row["filename"]
                # 파일 삭제 — basename만 취해 exports_dir 밖으로 나가지 못하게 한다.
                try:
                    fpath = base / Path(fname).name
                    if fpath.is_file():
                        fpath.unlink()
                except OSError as fe:
                    logger.warning("만료 파일 삭제 실패(무시): %s — %s", fname, fe)
                # 행 삭제(파일이 없어도 메타는 정리한다).
                await conn.execute(
                    "DELETE FROM tb_artifacts WHERE filename = $1",
                    fname,
                )
                deleted += 1
        if deleted:
            logger.info("만료 생성물 정리 완료: %d건 삭제", deleted)
    except Exception as e:  # noqa: BLE001 — fail-soft: 정리 실패가 서비스를 막지 않는다
        logger.warning("cleanup_expired_artifacts 실패(무시): %s", e)
    return deleted
