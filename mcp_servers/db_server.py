"""
db MCP 서버 — 사내 PostgreSQL 을 read-only 로 노출한다.

도구:
  query(sql, params?) — SELECT(또는 WITH ... SELECT) 조회만 허용. 결과를 행
  리스트(컬럼명→값 dict)로 반환한다.

왜 read-only 강제인가 (fail-closed):
  MCP 로 노출되는 DB 는 모델/에이전트가 직접 호출하므로, 쓰기/DDL 을 허용하면
  데이터 손상 위험이 크다. 따라서 SQL 진입부에서 SELECT/WITH 만 통과시키고,
  세미콜론 다중문·DDL/DML 키워드를 거부한다. 그 위에 asyncpg 트랜잭션을
  READ ONLY 로 한 번 더 잠가 이중 방어한다.

에어갭:
  접속 정보는 core/config(PostgreSQLConfig)에서 가져온다. 외부 네트워크 호출
  없음. asyncpg 미설치 시 명확한 오류를 반환(런타임 install 코드 없음).

의존성 방향 (P2):
  mcp_servers → core/config (재사용). core 는 이 모듈을 import 하지 않는다.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import FastAPI

from mcp_servers.framework import McpServerTool, create_mcp_app

logger = logging.getLogger("nexus.mcp_servers.db")

# 결과 행수 상한 — 거대 결과셋이 메모리/네트워크를 압박하지 않도록 잘라낸다.
_MAX_ROWS = 1000

# read-only 위반으로 간주하는 키워드(단어 경계로 매칭). SELECT 만 허용하되,
# 본문 어딘가에 쓰기/DDL 키워드가 섞여 있으면(예: CTE 내부 INSERT) 거부한다.
_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "grant",
    "revoke",
    "merge",
    "call",
    "copy",
    "vacuum",
    "analyze",
    "reindex",
    "comment",
    "set",
    "do",
)
_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _strip_sql_comments(sql: str) -> str:
    """
    SQL 주석을 제거한다(키워드 검사 회피 방지).

    -- 라인 주석과 /* ... */ 블록 주석을 모두 지운다. 주석 안에 'delete' 를
    숨겨 검사를 우회하려는 시도를 차단하기 위함이다.
    """
    no_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_line = re.sub(r"--[^\n]*", " ", no_block)
    return no_line


def _validate_read_only(sql: str) -> str:
    """
    SQL 이 read-only 단일 SELECT 인지 검증한다.

    규칙(fail-closed):
      1) 비어 있으면 거부.
      2) 주석 제거 후, SELECT 또는 WITH 로 시작해야 한다.
      3) 세미콜론으로 두 개 이상의 문장을 합칠 수 없다(다중문 금지).
      4) 본문에 쓰기/DDL 키워드가 단어 경계로 나타나면 거부.

    Returns:
        정규화된(앞뒤 공백 제거) SQL. 위반 시 ValueError.
    """
    if not sql or not sql.strip():
        raise ValueError("sql 이 비어 있습니다.")

    cleaned = _strip_sql_comments(sql).strip()
    if not cleaned:
        raise ValueError("주석을 제거하면 실행할 SQL 이 없습니다.")

    # 3) 다중문 금지 — 끝의 세미콜론 1개는 허용하되, 중간 세미콜론은 거부.
    trimmed = cleaned.rstrip(";").strip()
    if ";" in trimmed:
        raise ValueError("여러 문장(세미콜론 구분)은 허용되지 않습니다. 단일 SELECT 만 가능합니다.")

    # 2) SELECT/WITH 로 시작하는지 — 대소문자 무시.
    head = trimmed.lstrip("(").lstrip()  # 괄호로 감싼 (SELECT ...) 도 허용
    if not re.match(r"(?is)^\s*(select|with)\b", head):
        raise ValueError("read-only 위반: SELECT 또는 WITH ... SELECT 만 허용됩니다.")

    # 4) 쓰기/DDL 키워드 포함 여부.
    match = _FORBIDDEN_RE.search(trimmed)
    if match:
        raise ValueError(
            f"read-only 위반: 금지된 키워드 '{match.group(1).upper()}' 가 포함되어 있습니다."
        )

    return trimmed


class DbQueryTool(McpServerTool):
    """
    read-only SELECT 조회 도구.

    asyncpg 풀에서 커넥션을 얻어 READ ONLY 트랜잭션 안에서 실행한다. 이렇게
    하면 검증을 우회한 어떤 쓰기 시도도 DB 레벨에서 차단된다(이중 방어).
    """

    def __init__(self, pg_pool: Any) -> None:
        """
        Args:
            pg_pool: asyncpg.Pool 인스턴스. 반드시 유효해야 한다(없으면 build_app
                단계에서 오류 — db 서버는 인메모리 폴백을 제공하지 않는다).
        """
        self._pg = pg_pool

    @property
    def name(self) -> str:
        return "query"

    @property
    def description(self) -> str:
        return (
            "사내 PostgreSQL 에 read-only SELECT 조회를 실행한다. "
            "SELECT 또는 WITH ... SELECT 만 허용되며, 쓰기/DDL 은 거부된다. "
            "결과는 최대 1000행까지 컬럼명→값 dict 의 리스트로 반환된다."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "실행할 SELECT 쿼리. $1, $2 형태의 매개변수 사용 가능.",
                },
                "params": {
                    "type": "array",
                    "description": "쿼리 매개변수($1..$N)에 바인딩할 값 목록(선택).",
                    "items": {},
                },
            },
            "required": ["sql"],
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        """
        SELECT 를 검증 후 READ ONLY 트랜잭션에서 실행한다.

        Returns:
            {
              "columns": [컬럼명, ...],
              "rows": [{컬럼명: 값, ...}, ...],
              "row_count": int,        # 반환된 행수(상한 적용 후)
              "truncated": bool,       # 상한(_MAX_ROWS) 초과로 잘렸는지
              "note": str | None,      # 잘림 안내(선택)
            }

        예외:
            sql 누락/검증 실패 → ValueError (프레임워크가 -32602 로 변환).
            DB 실행 오류        → RuntimeError (프레임워크가 -32603 으로 변환).
        """
        sql = arguments.get("sql")
        if not isinstance(sql, str):
            raise ValueError("필수 인자 'sql'(문자열)이 없습니다.")

        params = arguments.get("params") or []
        if not isinstance(params, list):
            raise ValueError("'params' 는 배열이어야 합니다.")

        # read-only 검증 — 위반 시 ValueError.
        safe_sql = _validate_read_only(sql)

        # READ ONLY 트랜잭션에서 실행(이중 방어).
        try:
            async with self._pg.acquire() as conn:
                async with conn.transaction(readonly=True):
                    # 상한+1 만큼 가져와 잘림 여부를 정확히 판정한다.
                    records = await conn.fetch(
                        f"SELECT * FROM ({safe_sql}) AS _mcp_q LIMIT {_MAX_ROWS + 1}",  # noqa: S608 — safe_sql 은 SELECT 만 통과한 검증된 쿼리
                        *params,
                    )
        except Exception as e:  # noqa: BLE001 — asyncpg 예외 계층이 넓어 한 번에 포착 후 RuntimeError 로 정규화
            # asyncpg.PostgresError 등 다양한 예외를 표준 RuntimeError 로 변환해
            # 프레임워크가 -32603 으로 매핑하게 한다(서버는 죽지 않는다).
            raise RuntimeError(f"쿼리 실행 실패: {type(e).__name__}: {e}") from e

        truncated = len(records) > _MAX_ROWS
        used = records[:_MAX_ROWS]

        # 컬럼 순서를 첫 행 기준으로 안정적으로 추출한다.
        columns = list(used[0].keys()) if used else []
        rows = [dict(r) for r in used]

        result: dict[str, Any] = {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "note": (
                f"결과가 {_MAX_ROWS}행으로 잘렸습니다. LIMIT/WHERE 로 범위를 좁히세요."
                if truncated
                else None
            ),
        }
        return result


async def build_app(api_key: str = "local-key") -> FastAPI:
    """
    db MCP 서버 FastAPI 앱을 조립한다.

    동작:
      1) core/config 에서 PostgreSQL 접속 정보를 읽어 asyncpg 풀을 만든다.
      2) DbQueryTool 을 등록한 FastAPI 앱을 반환한다.
      3) lifespan 종료 시 풀을 닫는다.

    Args:
        api_key: Bearer 인증 키(run.py 에서 전달).

    Returns:
        FastAPI 앱. asyncpg 미설치/연결 실패 시 RuntimeError.
    """
    # core 재사용 — 설정 로딩(P2: mcp_servers → core 허용).
    from core.config import load_and_validate_config

    config = load_and_validate_config()

    try:
        import asyncpg
    except ImportError as e:
        # db 서버는 인메모리 폴백이 의미 없으므로(실 DB 노출이 목적) 명확히 실패.
        raise RuntimeError("asyncpg 패키지가 필요합니다(db MCP 서버).") from e

    try:
        pg_pool = await asyncpg.create_pool(
            host=config.postgresql.host,
            port=config.postgresql.port,
            database=config.postgresql.database,
            user=config.postgresql.user,
            password=config.postgresql.password,
            min_size=config.postgresql.min_connections,
            max_size=config.postgresql.max_connections,
            timeout=10.0,
        )
    except Exception as e:  # noqa: BLE001 — 연결 실패를 RuntimeError 로 정규화
        raise RuntimeError(
            f"PostgreSQL 연결 실패: {config.postgresql.host}:{config.postgresql.port} "
            f"({type(e).__name__}: {e})"
        ) from e

    logger.info(
        "db MCP 서버: PostgreSQL 연결 성공 %s:%d/%s",
        config.postgresql.host,
        config.postgresql.port,
        config.postgresql.database,
    )

    # 풀 정리 — 앱 종료 시 커넥션 반환(lifespan shutdown 훅).
    async def _close_pool() -> None:
        await pg_pool.close()
        logger.info("db MCP 서버: PostgreSQL 풀 종료")

    return create_mcp_app(
        tools=[DbQueryTool(pg_pool)],
        api_key=api_key,
        title="Nexus DB MCP Server",
        shutdown_hooks=[_close_pool],
    )
