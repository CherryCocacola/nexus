# core/storage/artifacts.py 단위 테스트 — tb_artifacts 기록/조회/정리 fail-soft 검증.
"""
core/storage/artifacts.py 단위 테스트.

검증 대상:
  - ensure_artifacts_schema / record_artifact / get_artifact_owner /
    cleanup_expired_artifacts 의 정상·경계 동작.

핵심 시나리오(작업 지시 검증 항목):
  (a) pool=None일 때 record/get_owner/ensure/cleanup가 예외 없이 조용히 동작(fail-soft).
  (b) get_artifact_owner의 "행 없음(ARTIFACT_NOT_FOUND) vs tenant None(None) vs
      특정 테넌트(문자열)" 3분기 구분.
  (c) record_artifact의 ON CONFLICT (filename) DO NOTHING 멱등성.
  (d) DB 오류 시 fail-soft(예외 삼킴 + 안전한 기본값 반환).
  (e) cleanup_expired_artifacts가 만료 행과 파일을 함께 삭제.

테스트 격리:
  실제 PostgreSQL 없이 asyncpg 풀/커넥션을 흉내 내는 FakePool로 검증한다.
  FakePool은 tb_artifacts를 파일명 키 딕셔너리로 모사해 UNIQUE/ON CONFLICT
  멱등 의미를 재현한다. asyncio_mode="auto"(pyproject)라 async def test_* 는
  데코레이터 없이 동작한다.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from core.storage.artifacts import (
    ARTIFACT_NOT_FOUND,
    cleanup_expired_artifacts,
    ensure_artifacts_schema,
    get_artifact_owner,
    list_session_artifacts,
    record_artifact,
)


# ─────────────────────────────────────────────
# 가짜 asyncpg 커넥션/풀 — tb_artifacts를 인메모리 dict로 모사
# ─────────────────────────────────────────────
class _FakeConn:
    """asyncpg 커넥션의 execute/fetchrow/fetch만 흉내 낸다.

    tb_artifacts를 {filename: row_dict} 딕셔너리로 모사해 UNIQUE(filename)와
    ON CONFLICT DO NOTHING의 멱등 의미를 재현한다. cleanup의 만료 SELECT는
    단순화를 위해 '모든 행이 만료됐다'고 간주해 전부 돌려준다(삭제 로직 검증 목적).
    """

    def __init__(self, store: dict[str, dict[str, Any]]) -> None:
        self._store = store
        self.executed: list[str] = []  # 실행된 SQL 앞부분 기록(검증용)

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append(sql.strip())
        s = sql.strip().upper()
        if s.startswith("INSERT INTO TB_ARTIFACTS"):
            # args: (filename, tenant_id, user_id, session_id, turn, mime, size, sha256)
            filename = args[0]
            # ON CONFLICT (filename) DO NOTHING — 이미 있으면 무시(멱등).
            if filename not in self._store:
                # created_at은 DB 기본값이라 INSERT 인자에 없다. 여기서는 기록 순서대로
                # 1초씩 증가시켜 흉내 낸다 — ORDER BY created_at 과 되붙이기 시각 매칭이
                # 의미를 가지려면 이 값이 단조 증가해야 한다(2026-08-08).
                self._store[filename] = {
                    "created_at": datetime(2026, 1, 1, tzinfo=UTC)
                    + timedelta(seconds=len(self._store)),
                    "filename": filename,
                    "tenant_id": args[1],
                    "user_id": args[2],
                    "session_id": args[3],
                    "turn": args[4],
                    "mime": args[5],
                    "size_bytes": args[6],
                    "sha256": args[7],
                }
            return "INSERT 0 1"
        if s.startswith("DELETE FROM TB_ARTIFACTS WHERE FILENAME"):
            self._store.pop(args[0], None)
            return "DELETE 1"
        # CREATE TABLE/INDEX/EXTENSION 등은 no-op으로 성공 처리.
        return "OK"

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        # SELECT tenant_id FROM tb_artifacts WHERE filename = $1
        return self._store.get(args[0])

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        s = sql.strip().upper()
        if "WHERE SESSION_ID" in s:
            # list_session_artifacts — filename/mime/turn + created_at 을 반환한다.
            # created_at 은 되붙이기가 "어느 답변에 붙일지" 고르는 키다(turn 은 항상 1로
            # 리셋돼 못 쓴다 — 2026-08-08). 여기서 빠뜨리면 실제 함수가 KeyError 로
            # fail-soft 되어 **빈 목록**이 되고, 그러면 생성물이 통째로 사라진다.
            sid = args[0]
            rows = [r for r in self._store.values() if r.get("session_id") == sid]
            rows.sort(key=lambda r: r.get("created_at") or datetime(2026, 1, 1, tzinfo=UTC))
            return [
                {
                    "filename": r["filename"],
                    "mime": r.get("mime"),
                    "turn": r.get("turn"),
                    "created_at": r.get("created_at"),
                }
                for r in rows
            ]
        # cleanup 만료 SELECT — 단순화를 위해 전체를 '만료'로 간주해 돌려준다.
        return [{"filename": fn} for fn in list(self._store.keys())]


class _AcquireCtx:
    """async with pool.acquire() as conn 을 지원하는 컨텍스트 매니저."""

    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakePool:
    """asyncpg.Pool.acquire()만 흉내 내는 가짜 풀."""

    def __init__(self, store: dict[str, dict[str, Any]] | None = None) -> None:
        self.store: dict[str, dict[str, Any]] = store if store is not None else {}
        self.conn = _FakeConn(self.store)

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self.conn)


class _ErrorPool:
    """acquire()가 예외를 던지는 풀 — fail-soft 경로 검증용."""

    def acquire(self) -> Any:
        raise RuntimeError("DB 연결 끊김(모의)")


# ─────────────────────────────────────────────
# (a) pool=None — 모든 함수가 예외 없이 조용히 동작(fail-soft)
# ─────────────────────────────────────────────
async def test_record_artifact_pool_none_noop():
    """pool=None이면 record_artifact가 예외 없이 조용히 스킵한다."""
    # 예외가 나면 테스트 실패 — 반환값은 None.
    result = await record_artifact(
        None, "a.docx", "t1", "s1", "application/docx", 100
    )
    assert result is None


async def test_get_owner_pool_none_returns_sentinel():
    """pool=None이면 get_artifact_owner가 ARTIFACT_NOT_FOUND(판정 불가)를 준다."""
    owner = await get_artifact_owner(None, "a.docx")
    assert owner is ARTIFACT_NOT_FOUND


async def test_ensure_schema_pool_none_noop():
    """pool=None이면 ensure_artifacts_schema가 예외 없이 no-op."""
    await ensure_artifacts_schema(None)  # 예외 없이 통과하면 성공


async def test_cleanup_pool_none_returns_zero():
    """pool=None이면 cleanup이 0을 돌려준다(정리 대상 없음)."""
    deleted = await cleanup_expired_artifacts(None, "/tmp/x", 90)
    assert deleted == 0


# ─────────────────────────────────────────────
# (b) get_artifact_owner 3분기 구분
# ─────────────────────────────────────────────
async def test_get_owner_no_row_is_sentinel():
    """행이 없으면 ARTIFACT_NOT_FOUND(레거시 통과 판정용)."""
    pool = _FakePool()
    owner = await get_artifact_owner(pool, "missing.docx")
    assert owner is ARTIFACT_NOT_FOUND


async def test_get_owner_tenant_none_is_none():
    """행은 있으나 tenant_id가 NULL이면 None(소유자 미상)."""
    pool = _FakePool({"x.docx": {"filename": "x.docx", "tenant_id": None}})
    owner = await get_artifact_owner(pool, "x.docx")
    assert owner is None


async def test_get_owner_specific_tenant_returns_id():
    """행이 있고 tenant_id가 특정 테넌트면 그 문자열을 돌려준다."""
    pool = _FakePool({"y.docx": {"filename": "y.docx", "tenant_id": "tenant-A"}})
    owner = await get_artifact_owner(pool, "y.docx")
    assert owner == "tenant-A"


# ─────────────────────────────────────────────
# (c) record_artifact 멱등성 (ON CONFLICT DO NOTHING)
# ─────────────────────────────────────────────
async def test_record_artifact_idempotent():
    """같은 파일명으로 두 번 기록해도 행이 하나만 유지된다(멱등)."""
    pool = _FakePool()
    await record_artifact(pool, "dup.docx", "t1", "s1", "application/docx", 10)
    # 두 번째 호출은 tenant를 바꿔도 ON CONFLICT DO NOTHING으로 무시돼야 한다.
    await record_artifact(pool, "dup.docx", "t2", "s2", "application/docx", 20)
    assert len(pool.store) == 1
    # 첫 기록이 유지된다(덮어쓰지 않음).
    assert pool.store["dup.docx"]["tenant_id"] == "t1"
    # 발행된 INSERT SQL에 ON CONFLICT 절이 포함되는지 확인.
    inserts = [s for s in pool.conn.executed if s.upper().startswith("INSERT")]
    assert inserts and "ON CONFLICT (FILENAME) DO NOTHING" in inserts[0].upper()


async def test_record_artifact_stores_metadata():
    """record_artifact가 넘긴 메타데이터를 그대로 저장한다(바이트 없이 메타만)."""
    pool = _FakePool()
    await record_artifact(
        pool,
        "meta.png",
        tenant_id="tenant-B",
        session_id="sess-9",
        mime="image/png",
        size_bytes=2048,
        sha256="deadbeef",
        user_id="u1",
    )
    row = pool.store["meta.png"]
    assert row["tenant_id"] == "tenant-B"
    assert row["session_id"] == "sess-9"
    assert row["mime"] == "image/png"
    assert row["size_bytes"] == 2048
    assert row["sha256"] == "deadbeef"
    assert row["user_id"] == "u1"
    # 바이너리 컬럼은 존재하지 않는다(바이트는 파일시스템 전용).
    assert "data" not in row and "bytes" not in row


# ─────────────────────────────────────────────
# (d) DB 오류 시 fail-soft
# ─────────────────────────────────────────────
async def test_record_artifact_db_error_swallowed():
    """acquire가 예외를 던져도 record_artifact는 예외를 밖으로 던지지 않는다."""
    pool = _ErrorPool()
    # 예외가 전파되면 테스트 실패.
    result = await record_artifact(pool, "e.docx", "t1", "s1", "x", 1)
    assert result is None


async def test_get_owner_db_error_returns_sentinel():
    """조회가 실패하면 ARTIFACT_NOT_FOUND(판정 불가 → 통과)로 안전 처리."""
    pool = _ErrorPool()
    owner = await get_artifact_owner(pool, "e.docx")
    assert owner is ARTIFACT_NOT_FOUND


async def test_ensure_schema_db_error_swallowed():
    """DDL 실행 실패도 fail-soft로 삼켜 기동을 막지 않는다."""
    pool = _ErrorPool()
    await ensure_artifacts_schema(pool)  # 예외 없이 통과하면 성공


# ─────────────────────────────────────────────
# (e) cleanup_expired_artifacts — 만료 행 + 파일 삭제
# ─────────────────────────────────────────────
async def test_cleanup_deletes_rows_and_files(tmp_path: Path):
    """cleanup이 만료 행을 지우고 대응 파일도 삭제한다."""
    # 실제 파일 2개를 만들어 둔다.
    f1 = tmp_path / "old1.docx"
    f2 = tmp_path / "old2.png"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("y", encoding="utf-8")

    pool = _FakePool(
        {
            "old1.docx": {"filename": "old1.docx", "tenant_id": "t1"},
            "old2.png": {"filename": "old2.png", "tenant_id": "t2"},
        }
    )
    deleted = await cleanup_expired_artifacts(pool, tmp_path, retention_days=90)

    assert deleted == 2
    assert not f1.exists()  # 파일 삭제됨
    assert not f2.exists()
    assert len(pool.store) == 0  # 메타 행도 삭제됨


async def test_cleanup_retention_zero_is_noop(tmp_path: Path):
    """retention_days<=0이면 안전장치로 아무 것도 지우지 않는다."""
    f1 = tmp_path / "keep.docx"
    f1.write_text("x", encoding="utf-8")
    pool = _FakePool({"keep.docx": {"filename": "keep.docx", "tenant_id": "t1"}})

    deleted = await cleanup_expired_artifacts(pool, tmp_path, retention_days=0)

    assert deleted == 0
    assert f1.exists()
    assert len(pool.store) == 1


async def test_cleanup_missing_file_still_deletes_row(tmp_path: Path):
    """파일이 이미 없어도(부재) 메타 행은 정리한다."""
    # 파일을 만들지 않는다 — 행만 존재.
    pool = _FakePool({"ghost.docx": {"filename": "ghost.docx", "tenant_id": "t1"}})
    deleted = await cleanup_expired_artifacts(pool, tmp_path, retention_days=90)
    assert deleted == 1
    assert len(pool.store) == 0


# ─────────────────────────────────────────────
# (f) list_session_artifacts — 히스토리 복원용 세션별 생성물 조회
# ─────────────────────────────────────────────
async def test_list_session_artifacts_filters_by_session_with_turn():
    """세션 ID로 생성물을 걸러 filename/mime/turn을 돌려주고, 다른 세션은 제외한다."""
    pool = _FakePool()
    # 같은 세션(sess-A)에 이미지·문서 각 1건, 다른 세션(sess-B) 1건을 기록.
    await record_artifact(
        pool, "a.png", tenant_id="t1", session_id="sess-A",
        mime="image/png", size_bytes=10, turn=3,
    )
    await record_artifact(
        pool, "b.docx", tenant_id="t1", session_id="sess-A",
        mime="application/vnd.openxmlformats", size_bytes=20, turn=5,
    )
    await record_artifact(
        pool, "c.png", tenant_id="t1", session_id="sess-B",
        mime="image/png", size_bytes=30, turn=1,
    )

    arts = await list_session_artifacts(pool, "sess-A")
    names = {a["filename"] for a in arts}
    assert names == {"a.png", "b.docx"}  # 다른 세션(c.png)은 제외
    by_name = {a["filename"]: a for a in arts}
    assert by_name["a.png"]["turn"] == 3
    assert by_name["b.docx"]["turn"] == 5
    assert by_name["a.png"]["mime"] == "image/png"
    # created_at 은 되붙이기의 매칭 키다 — 빠지면 생성물이 엉뚱한 답변에 몰린다(C2).
    assert by_name["a.png"]["created_at"] is not None
    assert by_name["a.png"]["created_at"] < by_name["b.docx"]["created_at"]
    assert [a["filename"] for a in arts] == ["a.png", "b.docx"]  # 생성순 유지


async def test_list_session_artifacts_pool_none_returns_empty():
    """pool=None이면 빈 리스트(fail-soft — 복원 실패가 세션 열람을 막지 않음)."""
    assert await list_session_artifacts(None, "sess-X") == []


async def test_list_session_artifacts_db_error_returns_empty():
    """조회가 실패해도 예외를 던지지 않고 빈 리스트로 안전 처리한다."""
    assert await list_session_artifacts(_ErrorPool(), "sess-X") == []
