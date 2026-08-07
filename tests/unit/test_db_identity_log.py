# 부트스트랩이 "붙은 DB가 무엇인지"를 로그로 드러내는지 검증하는 단위 테스트.
"""
`_log_db_identity()` 의 계약을 고정한다 (2026-08-07).

[왜 이 테스트가 있나 — 실제 사고]
    재부팅 때 호스트에 설치돼 있던 다른 PostgreSQL이 같은 포트를 4초 먼저 선점해,
    NOVA가 34GB 운영 DB가 아니라 46MB짜리 엉뚱한 DB에 붙었다. 로그에는
    "PostgreSQL 연결 성공"이 찍히고 /health도 200이라, 지식 RAG도 장기기억도 없는
    상태로 48분을 정상처럼 돌았다.

    그래서 이 로그는 **주소가 아니라 내용물**을 남겨야 한다. 아래 테스트들이
    지키는 것은 그 성질이다.
      - 정상 DB   : 크기와 행수가 로그에 나온다(사람이 46MB/34GB를 구분할 수 있어야 한다)
      - 빈 DB     : WARNING 으로 올라온다 (조용히 지나가면 사고가 반복된다)
      - 테이블 없음: WARNING (신규 DB일 수도, 잘못 붙은 것일 수도 — 둘 다 알려야 한다)
      - 조회 실패 : 아무 것도 깨지 않는다 (진단 로그가 부트스트랩을 막으면 안 된다)
"""

from __future__ import annotations

import logging

import pytest

from core.bootstrap import _log_db_identity


class _FakeConn:
    """asyncpg 연결 흉내 — fetchrow/fetch 두 가지만 돌려준다."""

    def __init__(self, db_row, table_rows, raise_on: str | None = None) -> None:
        self._db_row = db_row
        self._table_rows = table_rows
        self._raise_on = raise_on

    async def fetchrow(self, *_a, **_k):
        if self._raise_on == "fetchrow":
            raise RuntimeError("permission denied")
        return self._db_row

    async def fetch(self, *_a, **_k):
        if self._raise_on == "fetch":
            raise RuntimeError("permission denied")
        return self._table_rows


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *_a):
                return False

        return _Ctx()


def _table(name: str, est_rows: int, byte_size: int, schema: str = "nexus") -> dict:
    return {"schema": schema, "name": name, "est_rows": est_rows, "bytes": byte_size}


GB = 1024**3


@pytest.mark.asyncio
async def test_healthy_db_logs_size_and_rows(caplog):
    """★정상 DB — 크기와 행수가 로그에 드러난다."""
    pool = _FakePool(
        _FakeConn(
            ("idino_ai", "34 GB"),
            [
                _table("tb_knowledge", 1_603_563, 12 * GB),
                _table("tb_memories", 1_249_825, 19 * GB),
                _table("tb_symbols", 5_024, 40 * 1024 * 1024),
            ],
        )
    )

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _log_db_identity(pool)

    text = caplog.text
    assert "idino_ai" in text and "34 GB" in text
    assert "1,603,563행" in text  # 천단위 구분 — 사람이 자릿수를 읽을 수 있어야 한다
    assert "nexus.tb_knowledge" in text  # 스키마까지 남긴다(잘못된 스키마도 드러나게)
    assert "WARNING" not in caplog.text.upper().split("DB 신원")[0]


@pytest.mark.asyncio
async def test_empty_db_raises_warning(caplog):
    """★빈 DB — 조용히 넘어가지 않고 WARNING 으로 올린다(사고 재발 방지의 핵심)."""
    pool = _FakePool(
        _FakeConn(
            ("idino_ai", "46 MB"),
            [
                _table("tb_knowledge", 0, 8192),
                _table("tb_memories", 0, 8192),
                _table("tb_symbols", 0, 8192),
            ],
        )
    )

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _log_db_identity(pool)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "빈 DB인데 경고가 없다"
    assert "비어 있습니다" in warnings[0].getMessage()
    assert "46 MB" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_partially_filled_db_is_not_warned(caplog):
    """하나라도 데이터가 있으면 경고하지 않는다 — 과잉 경고 방지."""
    pool = _FakePool(
        _FakeConn(
            ("idino_ai", "12 GB"),
            [
                _table("tb_knowledge", 1_603_563, 12 * GB),
                _table("tb_memories", 0, 8192),  # 신규 배포라 기억은 아직 없음
                _table("tb_symbols", 0, 8192),
            ],
        )
    )

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _log_db_identity(pool)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_no_signature_tables_warns(caplog):
    """테이블이 아예 없으면 알린다 — 신규 DB일 수도, 잘못 붙은 것일 수도 있다."""
    pool = _FakePool(_FakeConn(("postgres", "8 MB"), []))

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _log_db_identity(pool)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings and "테이블이 하나도 없습니다" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_unanalyzed_table_shows_unknown_not_fake_number(caplog):
    """ANALYZE 전 reltuples 는 -1 이다 — 모르는 값을 숫자인 척 적지 않는다."""
    pool = _FakePool(
        _FakeConn(
            ("idino_ai", "34 GB"),
            [_table("tb_knowledge", -1, 12 * GB)],
        )
    )

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _log_db_identity(pool)

    assert "행수미상" in caplog.text
    assert "-1" not in caplog.text


@pytest.mark.asyncio
async def test_query_failure_is_silent(caplog):
    """조회가 실패해도 예외를 올리지 않는다 — 진단 로그가 부트스트랩을 막으면 안 된다."""
    for stage in ("fetchrow", "fetch"):
        pool = _FakePool(_FakeConn(("x", "1 MB"), [], raise_on=stage))
        with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
            await _log_db_identity(pool)  # 예외가 나가면 이 줄에서 실패한다
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
