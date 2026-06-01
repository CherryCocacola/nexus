"""
mcp_servers/db_server.py 의 DbQueryTool read-only 강제 + 실행 경로 단위 테스트.

검증 대상 (fail-closed 핵심):
  1. SELECT / WITH ... SELECT 허용 → mock pool 에서 행을 받아 정형 결과 반환.
  2. INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE → ValueError(거부).
  3. 세미콜론 다중문 → ValueError. 끝 세미콜론 1개는 허용.
  4. SQL 주석으로 금지 키워드를 숨기는 우회 → 주석 제거 후 거부.
  5. 행 상한(_MAX_ROWS) 초과 시 truncated=True + 잘림 안내.
  6. sql 누락/타입 오류 → ValueError. DB 실행 예외 → RuntimeError.

운영 DB 오염 방지: asyncpg.Pool 은 전부 AsyncMock 으로 대체한다 — 실 DB 미접속,
실 tb_knowledge(105만행) 절대 미접근.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_servers.db_server import _MAX_ROWS, DbQueryTool, _validate_read_only


# ─────────────────────────────────────────────
# mock asyncpg pool — READ ONLY 트랜잭션 컨텍스트까지 모사
# ─────────────────────────────────────────────
class _FakeRecord(dict):
    """asyncpg.Record 흉내 — dict(r)/r.keys() 가 동작하면 충분하다."""


def _make_pool(fetch_return: Any) -> AsyncMock:
    """
    DbQueryTool 이 기대하는 풀 형태를 모사한다:
      async with pool.acquire() as conn:
          async with conn.transaction(readonly=True):
              records = await conn.fetch(sql, *params)

    Args:
        fetch_return: conn.fetch 가 돌려줄 값(레코드 리스트 또는 side_effect 예외).
    """
    conn = AsyncMock()
    if isinstance(fetch_return, Exception):
        conn.fetch = AsyncMock(side_effect=fetch_return)
    else:
        conn.fetch = AsyncMock(return_value=fetch_return)

    # conn.transaction(readonly=True) 는 async 컨텍스트 매니저여야 한다.
    # (await 대상이 아니라 동기 호출이 async-ctx 를 반환 → MagicMock 으로 구성)
    txn = AsyncMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn)

    pool = AsyncMock()
    acquire_ctx = AsyncMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=acquire_ctx)
    # 테스트가 conn.fetch 호출 인자를 검증할 수 있도록 노출.
    pool._conn = conn
    return pool


# ─────────────────────────────────────────────
# _validate_read_only — 순수 함수(거부 규칙)
# ─────────────────────────────────────────────
class TestValidateReadOnly:
    """SQL read-only 검증 규칙을 mock 없이 직접 검사한다."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "select id from tb_knowledge limit 5",
            "WITH x AS (SELECT 1) SELECT * FROM x",
            "  SELECT 1  ",  # 앞뒤 공백
            "SELECT 1;",  # 끝 세미콜론 1개는 허용
            "(SELECT 1)",  # 괄호로 감싼 SELECT
        ],
    )
    def test_allows_select_and_with(self, sql: str):
        """SELECT/WITH 로 시작하는 단일 조회문은 통과해야 한다."""
        # 예외가 발생하지 않고 정규화된 문자열을 반환해야 한다.
        assert _validate_read_only(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET a=1",
            "DELETE FROM t",
            "DROP TABLE t",
            "ALTER TABLE t ADD COLUMN c int",
            "CREATE TABLE t (id int)",
            "TRUNCATE t",
            "GRANT ALL ON t TO u",
            "COPY t FROM '/etc/passwd'",
        ],
    )
    def test_rejects_write_and_ddl(self, sql: str):
        """쓰기/DDL 문은 ValueError 로 거부되어야 한다."""
        with pytest.raises(ValueError):
            _validate_read_only(sql)

    def test_rejects_multi_statement_semicolon(self):
        """중간 세미콜론(다중문)은 거부되어야 한다."""
        with pytest.raises(ValueError):
            _validate_read_only("SELECT 1; DROP TABLE t")

    def test_rejects_select_with_embedded_write_keyword(self):
        """SELECT 로 시작해도 본문에 쓰기 키워드가 있으면 거부(CTE 내부 DELETE 등)."""
        with pytest.raises(ValueError):
            _validate_read_only("WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d")

    def test_rejects_line_comment_smuggled_keyword(self):
        """라인 주석으로 금지 키워드를 숨겨도, 주석 제거 후 거부되어야 한다."""
        # 주석을 제거하면 'SELECT 1 DROP TABLE t' 가 되어 DROP 이 드러난다.
        with pytest.raises(ValueError):
            _validate_read_only("SELECT 1 --\nDROP TABLE t")

    def test_rejects_block_comment_smuggled_keyword(self):
        """블록 주석으로 숨긴 키워드도 제거 후 거부되어야 한다."""
        with pytest.raises(ValueError):
            _validate_read_only("SELECT 1 /* hi */ DELETE FROM t")

    def test_rejects_empty_sql(self):
        """빈 SQL 은 거부되어야 한다."""
        with pytest.raises(ValueError):
            _validate_read_only("")

    def test_rejects_comment_only_sql(self):
        """주석만 있는 SQL 은 거부되어야 한다(제거 후 실행할 내용 없음)."""
        with pytest.raises(ValueError):
            _validate_read_only("-- just a comment")

    def test_rejects_non_select_leading_keyword(self):
        """SELECT/WITH 가 아닌 다른 키워드로 시작하면 거부."""
        with pytest.raises(ValueError):
            _validate_read_only("EXPLAIN SELECT 1")


# ─────────────────────────────────────────────
# DbQueryTool.call — 실행 경로(mock pool)
# ─────────────────────────────────────────────
class TestDbQueryToolCall:
    """DbQueryTool.call 의 결과 정형/거부/예외 정규화를 검증한다."""

    async def test_call_select_returns_structured_rows(self):
        """SELECT 호출 → columns/rows/row_count/truncated 정형 결과를 반환해야 한다."""
        records = [_FakeRecord(id=1, title="가"), _FakeRecord(id=2, title="나")]
        pool = _make_pool(records)
        tool = DbQueryTool(pool)

        result = await tool.call({"sql": "SELECT id, title FROM tb_knowledge LIMIT 2"})

        assert result["columns"] == ["id", "title"]
        assert result["rows"] == [{"id": 1, "title": "가"}, {"id": 2, "title": "나"}]
        assert result["row_count"] == 2
        assert result["truncated"] is False
        assert result["note"] is None

    async def test_call_wraps_query_in_limit_guard(self):
        """검증된 SELECT 를 LIMIT(_MAX_ROWS+1) 서브쿼리로 감싸 실행해야 한다(상한 보호)."""
        pool = _make_pool([])
        tool = DbQueryTool(pool)
        await tool.call({"sql": "SELECT 1"})

        # conn.fetch 의 첫 인자(SQL)에 LIMIT _MAX_ROWS+1 가드가 포함돼야 한다.
        executed_sql = pool._conn.fetch.await_args.args[0]
        assert f"LIMIT {_MAX_ROWS + 1}" in executed_sql
        assert "SELECT 1" in executed_sql

    async def test_call_forwards_params_to_fetch(self):
        """params 배열이 conn.fetch 에 위치 인자로 전달되어야 한다($1..$N 바인딩)."""
        pool = _make_pool([])
        tool = DbQueryTool(pool)
        await tool.call({"sql": "SELECT * FROM t WHERE id=$1", "params": [42]})

        # fetch(sql, *params) → 두 번째 인자가 42.
        assert pool._conn.fetch.await_args.args[1] == 42

    async def test_call_truncates_over_max_rows(self):
        """상한 초과(_MAX_ROWS+1 행)면 truncated=True + 안내, 행은 상한까지만."""
        # 상한+1 개를 돌려주도록 모사.
        records = [_FakeRecord(id=i) for i in range(_MAX_ROWS + 1)]
        pool = _make_pool(records)
        tool = DbQueryTool(pool)

        result = await tool.call({"sql": "SELECT id FROM big"})

        assert result["truncated"] is True
        assert result["row_count"] == _MAX_ROWS
        assert len(result["rows"]) == _MAX_ROWS
        assert result["note"]  # 잘림 안내 문구 존재

    async def test_call_rejects_write_sql_with_value_error(self):
        """쓰기 SQL 은 fetch 까지 가기 전에 ValueError 로 거부되어야 한다."""
        pool = _make_pool([])
        tool = DbQueryTool(pool)
        with pytest.raises(ValueError):
            await tool.call({"sql": "DELETE FROM tb_knowledge"})
        # DB 에 절대 도달하지 않아야 한다(운영 오염 방지).
        pool._conn.fetch.assert_not_called()

    async def test_call_missing_sql_raises_value_error(self):
        """sql 인자 누락 → ValueError."""
        pool = _make_pool([])
        tool = DbQueryTool(pool)
        with pytest.raises(ValueError):
            await tool.call({})

    async def test_call_non_string_sql_raises_value_error(self):
        """sql 이 문자열이 아니면 ValueError."""
        pool = _make_pool([])
        tool = DbQueryTool(pool)
        with pytest.raises(ValueError):
            await tool.call({"sql": 123})

    async def test_call_non_list_params_raises_value_error(self):
        """params 가 배열이 아니면 ValueError."""
        pool = _make_pool([])
        tool = DbQueryTool(pool)
        with pytest.raises(ValueError):
            await tool.call({"sql": "SELECT 1", "params": "oops"})

    async def test_call_db_error_normalized_to_runtime_error(self):
        """DB 실행 중 예외는 RuntimeError 로 정규화되어야 한다(프레임워크 -32603)."""
        pool = _make_pool(OSError("connection reset"))
        tool = DbQueryTool(pool)
        with pytest.raises(RuntimeError):
            await tool.call({"sql": "SELECT 1"})


# ─────────────────────────────────────────────
# 도구 메타 — schema/name/description 정합
# ─────────────────────────────────────────────
class TestDbQueryToolMetadata:
    """도구 식별/스키마 노출을 검증한다."""

    def test_name_and_required_schema(self):
        """name='query', input_schema.required=['sql'] 여야 한다."""
        tool = DbQueryTool(_make_pool([]))
        assert tool.name == "query"
        assert tool.input_schema["required"] == ["sql"]
        assert "read-only" in tool.description or "SELECT" in tool.description
