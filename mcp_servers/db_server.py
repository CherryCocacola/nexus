"""
db MCP 서버 — 사내 PostgreSQL 을 read-only 로 노출한다.

■ 이 파일이 하는 일 (한 문단 요약)
  Nexus 의 모델/에이전트가 MCP 프로토콜을 통해 사내 DB 를 "읽기 전용"으로
  조회할 수 있게 해 주는 작은 FastAPI 서버다. 오직 SELECT 조회만 허용하며,
  쓰기(INSERT/UPDATE/DELETE)나 DDL(CREATE/DROP/ALTER 등)은 두 겹의 방어로
  철저히 막는다. 조회 결과는 최대 1000행까지 dict 리스트 형태로 돌려준다.

■ 공개(노출) 도구
  query(sql, params?) — SELECT(또는 WITH ... SELECT) 조회만 허용. 결과를 행
  리스트(컬럼명→값 dict)로 반환한다.

■ 주요 구성 요소 (읽는 순서 추천)
  - _strip_sql_comments()  : SQL 주석 제거(검사 우회 방지)
  - _validate_read_only()  : SELECT 여부·다중문·금지 키워드 검증(1차 방어)
  - DbQueryTool            : 실제 조회를 수행하는 MCP 도구 클래스
  - build_app()            : asyncpg 풀 생성 + FastAPI 앱 조립(진입점)

■ 왜 read-only 를 강제하는가 (fail-closed 설계)
  MCP 로 노출되는 DB 는 모델/에이전트가 직접 호출하므로, 쓰기/DDL 을 허용하면
  데이터 손상 위험이 크다. 따라서 두 겹으로 막는다.
    1차 방어(_validate_read_only): SQL 진입부에서 SELECT/WITH 만 통과시키고,
           세미콜론 다중문·DDL/DML 키워드를 거부한다.
    2차 방어(DB 트랜잭션): asyncpg 트랜잭션을 READ ONLY 로 한 번 더 잠가,
           1차를 뚫더라도 DB 엔진 레벨에서 쓰기가 물리적으로 실패하게 한다.
  이렇게 '가장 안전한 쪽이 기본값'인 fail-closed 원칙을 지킨다.

■ 에어갭(폐쇄망) 준수
  접속 정보는 core/config(PostgreSQLConfig)에서 가져온다. 외부 네트워크 호출
  없음. asyncpg 미설치 시 명확한 오류를 반환(런타임 install 코드 없음).

■ 의존성 방향 (아키텍처 규칙 P2)
  mcp_servers → core/config (재사용). core 는 이 모듈을 import 하지 않는다.
  즉 이 파일은 core 를 가져다 쓰기만 하고, core 가 이 파일에 의존하지 않는다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import FastAPI

from mcp_servers.framework import McpServerTool, create_mcp_app

# 이 모듈 전용 로거. 이름을 계층적으로 지정해 두면 나중에 로그를 이 서버만
# 따로 켜고 끌 수 있다("nexus.mcp_servers.db" 만 DEBUG 로 올리는 식).
logger = logging.getLogger("nexus.mcp_servers.db")

# 결과 행수 상한 — 거대 결과셋이 메모리/네트워크를 압박하지 않도록 잘라낸다.
# 실제 조회 시에는 이 값 +1 만큼 가져와서 "1000행을 넘겼는지"를 정확히 판단한다.
_MAX_ROWS = 1000

# read-only 위반으로 간주하는 키워드(단어 경계로 매칭). SELECT 만 허용하되,
# 본문 어딘가에 쓰기/DDL 키워드가 섞여 있으면(예: CTE 내부 INSERT) 거부한다.
# 아래 튜플에 새 위험 키워드를 추가하면 자동으로 정규식(_FORBIDDEN_RE)에 반영된다.
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
# 위 금지 키워드들을 하나의 정규식으로 미리 컴파일해 둔다.
#   - r"\b(...)\b" : 단어 경계(\b)로 감싸 'select' 안의 'set' 같은 부분 일치를 막는다.
#   - "|".join(...) : 키워드들을 OR 로 이어 붙여 하나라도 걸리면 매치되게 한다.
#   - re.IGNORECASE : 대소문자를 구분하지 않는다(Insert, INSERT, insert 모두 탐지).
# 모듈 로드 시 한 번만 컴파일해 두면 매 요청마다 재컴파일하는 비용을 아낄 수 있다.
_FORBIDDEN_RE = re.compile(
    r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def _strip_sql_comments(sql: str) -> str:
    """
    SQL 주석을 제거한다(키워드 검사 회피 방지).

    왜 필요한가:
      공격자가 주석 안에 위험 키워드를 숨겨 검사를 우회하려 할 수 있다.
      예) "SELECT 1 /* delete */" 또는 "SELECT 1 -- delete" 처럼 주석 속에
      'delete' 를 넣어도, 여기서 먼저 주석을 지운 뒤 검사하므로 소용없다.

    처리 순서:
      1) /* ... */ 블록 주석 제거 — DOTALL 로 줄바꿈을 넘는 여러 줄 주석도 잡는다.
      2) -- 로 시작하는 라인 주석 제거 — 줄 끝(\n)까지를 공백으로 치환한다.
      주석을 빈 문자열이 아니라 공백(" ")으로 바꾸는 이유는, 주석 앞뒤 토큰이
      의도치 않게 붙어버리는 것을 막기 위함이다.

    Args:
        sql: 원본 SQL 문자열.

    Returns:
        주석이 모두 공백으로 치환된 SQL 문자열(길이/구조는 대략 유지).
    """
    no_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    no_line = re.sub(r"--[^\n]*", " ", no_block)
    return no_line


def _validate_read_only(sql: str) -> str:
    """
    SQL 이 read-only 단일 SELECT 인지 검증한다(이 서버의 1차 방어선).

    이 함수는 사용자가 보낸 SQL 을 실제로 실행하기 "전에" 통과시켜야 하는
    문지기다. 아래 네 가지 규칙을 모두 만족해야만 통과되며, 하나라도
    어기면 즉시 ValueError 를 던져 실행을 막는다(fail-closed).

    규칙(fail-closed):
      1) 비어 있으면 거부.
      2) 주석 제거 후, SELECT 또는 WITH 로 시작해야 한다.
      3) 세미콜론으로 두 개 이상의 문장을 합칠 수 없다(다중문 금지).
         → "SELECT 1; DROP TABLE t" 같은 SQL 인젝션 형태를 원천 차단.
      4) 본문에 쓰기/DDL 키워드가 단어 경계로 나타나면 거부.

    Args:
        sql: 검증할 원본 SQL 문자열.

    Returns:
        정규화된(앞뒤 공백·끝 세미콜론 제거) SQL 문자열. 이 반환값이 실제
        실행에 쓰이는 "안전하다고 판정된" SQL 이다.

    Raises:
        ValueError: 위 네 규칙 중 하나라도 위반한 경우. 호출부(도구)에서
            그대로 위로 전파되어, MCP 프레임워크가 -32602 오류로 변환한다.
    """
    # 1) 빈 입력 방어 — None 이거나 공백만 있으면 실행할 게 없으므로 거부.
    if not sql or not sql.strip():
        raise ValueError("sql 이 비어 있습니다.")

    # 주석을 먼저 걷어낸다. 주석 안에 숨긴 키워드로 아래 검사를 우회하지 못하게 함.
    cleaned = _strip_sql_comments(sql).strip()
    if not cleaned:
        raise ValueError("주석을 제거하면 실행할 SQL 이 없습니다.")

    # 3) 다중문 금지 — 끝의 세미콜론 1개는 허용하되, 중간 세미콜론은 거부.
    #    rstrip(";") 로 맨 끝 세미콜론을 떼어낸 뒤에도 세미콜론이 남아 있다면,
    #    그것은 문장이 둘 이상 이어졌다는 뜻이므로 차단한다.
    trimmed = cleaned.rstrip(";").strip()
    if ";" in trimmed:
        raise ValueError("여러 문장(세미콜론 구분)은 허용되지 않습니다. 단일 SELECT 만 가능합니다.")

    # 2) SELECT/WITH 로 시작하는지 — 대소문자 무시.
    #    맨 앞 여는 괄호를 벗겨(lstrip) "(SELECT ...)" 처럼 괄호로 감싼 형태도 허용한다.
    head = trimmed.lstrip("(").lstrip()  # 괄호로 감싼 (SELECT ...) 도 허용
    #    (?is): i=대소문자 무시, s=DOTALL. 문두가 select 또는 with 여야만 통과.
    if not re.match(r"(?is)^\s*(select|with)\b", head):
        raise ValueError("read-only 위반: SELECT 또는 WITH ... SELECT 만 허용됩니다.")

    # 4) 쓰기/DDL 키워드 포함 여부 — 앞서 컴파일한 정규식으로 본문 전체를 훑는다.
    #    WITH 절(CTE) 내부에 INSERT 를 숨기는 식의 우회도 여기서 함께 걸린다.
    match = _FORBIDDEN_RE.search(trimmed)
    if match:
        raise ValueError(
            f"read-only 위반: 금지된 키워드 '{match.group(1).upper()}' 가 포함되어 있습니다."
        )

    return trimmed


class DbQueryTool(McpServerTool):
    """
    read-only SELECT 조회 도구 (MCP 로 노출되는 'query' 도구의 실체).

    McpServerTool 을 상속하며, MCP 프레임워크는 이 클래스의 name/description/
    input_schema 로 도구를 광고하고, 모델이 호출하면 call() 을 실행한다.

    핵심 동작:
      asyncpg 풀에서 커넥션을 하나 빌려, READ ONLY 트랜잭션 안에서 조회한다.
      이렇게 하면 1차 검증(_validate_read_only)을 어쩌다 우회한 쓰기 시도라도
      DB 엔진이 트랜잭션 레벨에서 거부하므로, 두 겹의 방어가 완성된다.
    """

    def __init__(self, pg_pool: Any) -> None:
        """
        도구를 생성하면서 사용할 커넥션 풀을 주입받는다.

        풀을 도구가 직접 만들지 않고 밖에서 받는 이유는, 여러 요청이 커넥션을
        공유·재사용하도록 build_app() 에서 한 번만 풀을 만들어 넘기기 위함이다.

        Args:
            pg_pool: asyncpg.Pool 인스턴스. 반드시 유효해야 한다(없으면 build_app
                단계에서 오류 — db 서버는 인메모리 폴백을 제공하지 않는다).
        """
        self._pg = pg_pool

    @property
    def name(self) -> str:
        # MCP 클라이언트(모델)가 이 도구를 호출할 때 쓰는 이름. 서버 내에서 유일해야 함.
        return "query"

    @property
    def description(self) -> str:
        # 모델에게 보여줄 도구 설명. 무엇을 할 수 있고 무엇이 금지인지 명확히 적어,
        # 모델이 애초에 쓰기 쿼리를 시도하지 않도록 유도한다(프롬프트 차원의 안내).
        return (
            "사내 PostgreSQL 에 read-only SELECT 조회를 실행한다. "
            "SELECT 또는 WITH ... SELECT 만 허용되며, 쓰기/DDL 은 거부된다. "
            "결과는 최대 1000행까지 컬럼명→값 dict 의 리스트로 반환된다."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        # 도구 입력 규격(JSON Schema). MCP 클라이언트는 이 스키마를 보고 어떤
        # 인자를 넘길지 결정한다. sql 은 필수, params 는 선택이다.
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
        SELECT 를 검증 후 READ ONLY 트랜잭션에서 실행한다(이 도구의 심장부).

        전체 흐름(위에서 아래로):
          1) 인자에서 sql/params 를 꺼내 타입을 확인한다.
          2) _validate_read_only 로 1차 검증(SELECT 여부·키워드 등)을 통과시킨다.
          3) 커넥션을 빌려 READ ONLY 트랜잭션 안에서 조회한다(2차 방어).
          4) 상한(_MAX_ROWS)에 맞춰 행을 잘라내고, 잘렸는지 여부까지 담아 돌려준다.

        Args:
            arguments: MCP 클라이언트가 넘긴 인자 dict. "sql"(필수, 문자열),
                "params"(선택, 리스트)를 담는다.

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
        # 1) sql 인자 검증 — 문자열이 아니면(누락 포함) 진행할 수 없다.
        sql = arguments.get("sql")
        if not isinstance(sql, str):
            raise ValueError("필수 인자 'sql'(문자열)이 없습니다.")

        # params 는 선택 인자. 없으면 빈 리스트로 두고, 있으면 리스트인지 확인한다.
        # ($1, $2 자리에 바인딩될 값들이며, SQL 인젝션을 막는 안전한 바인딩 방식이다.)
        params = arguments.get("params") or []
        if not isinstance(params, list):
            raise ValueError("'params' 는 배열이어야 합니다.")

        # 2) read-only 검증 — 위반 시 ValueError. 통과한 safe_sql 만 실행에 쓴다.
        safe_sql = _validate_read_only(sql)

        # 3) READ ONLY 트랜잭션에서 실행(이중 방어).
        #    acquire() 로 풀에서 커넥션을 잠깐 빌리고, async with 블록을 벗어나면
        #    자동으로 풀에 반납된다. transaction(readonly=True) 안에서는 DB 엔진이
        #    쓰기를 아예 거부하므로, 1차 검증을 뚫었더라도 여기서 최종 차단된다.
        try:
            async with self._pg.acquire() as conn:
                async with conn.transaction(readonly=True):
                    # 상한+1 만큼 가져와 잘림 여부를 정확히 판정한다.
                    #   사용자 SQL 을 서브쿼리로 감싸고 LIMIT 을 바깥에서 씌워,
                    #   원본에 LIMIT 이 없어도 과도한 행이 넘어오지 않게 한다.
                    #   1001행이 오면 "1000행을 넘겼다"는 걸 알 수 있어 truncated 판정이 정확하다.
                    records = await conn.fetch(
                        f"SELECT * FROM ({safe_sql}) AS _mcp_q LIMIT {_MAX_ROWS + 1}",  # noqa: S608 — safe_sql 은 SELECT 만 통과한 검증된 쿼리
                        *params,
                    )
        except Exception as e:  # noqa: BLE001 — asyncpg 예외 계층이 넓어 한 번에 포착 후 RuntimeError 로 정규화
            # asyncpg.PostgresError 등 다양한 예외를 표준 RuntimeError 로 변환해
            # 프레임워크가 -32603 으로 매핑하게 한다(서버는 죽지 않는다).
            raise RuntimeError(f"쿼리 실행 실패: {type(e).__name__}: {e}") from e

        # 4) 상한 초과 판정 및 잘라내기.
        #    상한+1 을 요청했으므로, 실제로 그만큼 왔다면 원본은 더 많았다는 뜻이다.
        truncated = len(records) > _MAX_ROWS
        used = records[:_MAX_ROWS]  # 클라이언트에 돌려줄 실제 행은 상한까지만.

        # 컬럼 순서를 첫 행 기준으로 안정적으로 추출한다.
        #    (행이 하나도 없으면 컬럼 목록도 알 수 없으므로 빈 리스트를 준다.)
        columns = list(used[0].keys()) if used else []
        # asyncpg 의 Record 객체를 일반 dict 로 변환해 JSON 직렬화가 가능하게 한다.
        rows = [dict(r) for r in used]

        # 응답 조립 — 컬럼/행/행수/잘림여부/안내문을 하나의 dict 로 묶어 반환한다.
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
    db MCP 서버 FastAPI 앱을 조립한다(이 모듈의 진입점/팩토리 함수).

    run.py 같은 실행 스크립트가 이 함수를 호출해 서버 앱을 얻는다. 여기서
    DB 연결 준비까지 끝내므로, 앱이 반환되는 시점엔 이미 조회 가능한 상태다.

    동작:
      1) core/config 에서 PostgreSQL 접속 정보를 읽어 asyncpg 풀을 만든다.
      2) DbQueryTool 을 등록한 FastAPI 앱을 반환한다.
      3) lifespan 종료 시 풀을 닫는다(_close_pool 훅).

    Args:
        api_key: Bearer 인증 키(run.py 에서 전달). MCP 요청 인증에 쓰인다.

    Returns:
        조립이 끝난 FastAPI 앱.

    Raises:
        RuntimeError: asyncpg 미설치, 또는 PostgreSQL 연결 실패 시. 어느 쪽이든
            원인을 담은 명확한 메시지로 정규화해 던진다.
    """
    # core 재사용 — 설정 로딩(P2: mcp_servers → core 허용).
    # 함수 안에서 늦게(lazy) import 하는 이유는, 모듈 로드 시점의 불필요한
    # 의존성/순환 import 를 피하고 실제 앱을 만들 때만 core 설정을 끌어오기 위함이다.
    from core.config import load_and_validate_config

    config = load_and_validate_config()

    # asyncpg 는 PostgreSQL 비동기 드라이버. 에어갭 규칙상 런타임 설치는 금지이므로,
    # 없으면 설치를 시도하지 않고 그대로 명확한 오류로 실패시킨다.
    try:
        import asyncpg
    except ImportError as e:
        # db 서버는 인메모리 폴백이 의미 없으므로(실 DB 노출이 목적) 명확히 실패.
        raise RuntimeError("asyncpg 패키지가 필요합니다(db MCP 서버).") from e

    # 커넥션 풀 생성 — 요청마다 새 연결을 여는 대신, 미리 만들어 둔 연결들을
    # 돌려쓴다. min/max 로 유지할 연결 수 범위를 정하고, timeout 으로 연결
    # 대기 한도를 둔다. 모든 접속 정보는 config(YAML)에서 오며 하드코딩하지 않는다.
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
    # 서버가 내려갈 때 이 함수가 호출되어, 열려 있던 모든 커넥션을 깔끔히 닫는다.
    # 정리하지 않으면 DB 쪽에 유휴 커넥션이 남아 자원을 낭비할 수 있다.
    async def _close_pool() -> None:
        await pg_pool.close()
        logger.info("db MCP 서버: PostgreSQL 풀 종료")

    # 준비된 도구·인증키·종료훅을 프레임워크에 넘겨 최종 FastAPI 앱을 만든다.
    # create_mcp_app 이 MCP 프로토콜 엔드포인트(도구 목록/호출)와 인증을 붙여 준다.
    return create_mcp_app(
        tools=[DbQueryTool(pg_pool)],
        api_key=api_key,
        title="Nexus DB MCP Server",
        shutdown_hooks=[_close_pool],
    )
