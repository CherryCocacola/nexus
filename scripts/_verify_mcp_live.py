"""MCP 서버 ↔ 클라이언트 실 백엔드 e2e 검증 스크립트 (개발/운영자 진단용).

[이 파일이 하는 일]
Nexus 의 MCP(Model Context Protocol) 서버 두 개(db / kowiki)를 이 프로세스
안에서 임의 포트의 uvicorn 서버로 직접 띄운 뒤, 우리가 만든 McpClient 로
`list_tools`(도구 목록 조회) → `call_tool`(도구 실제 호출) 순서로 불러본다.
그렇게 해서 다음의 "전 구간(end-to-end)" 연결이 살아 있는지를 한 번에 확인한다.

  McpClient → (HTTP) → MCP 서버 → 실 PostgreSQL(.39) / 실 임베딩 서버(.28:8002)

핵심은 mock 이 아니라 **실제 백엔드**에 붙는다는 점이다. 그래서 서버·DB·임베딩
어느 한 곳이라도 죽어 있으면 여기서 바로 드러난다. 단, 던지는 요청은 전부
read-only(SELECT / search)라 운영 데이터(tb_knowledge)를 절대 변경하지 않는다.

[주요 함수]
  - _free_port() : 충돌 없는 빈 로컬 포트 확보
  - _serve()     : FastAPI 앱을 uvicorn 백그라운드 태스크로 기동 + 준비 대기
  - _stop()      : uvicorn 서버 정상 종료
  - check_db()   : db MCP 서버 → 실 PG 로 SELECT 검증
  - check_kowiki(): kowiki MCP 서버 → 실 임베딩 + pgvector 검색 검증
  - main()       : 위 두 검증을 순서대로 실행

[의존 모듈]
  core.tools.mcp.client.McpClient, mcp_servers.db_server, mcp_servers.kowiki_server

[실행 방법]
  python -m scripts._verify_mcp_live   (또는 python scripts/_verify_mcp_live.py)

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import socket
import sys

import uvicorn

from core.tools.mcp.client import McpClient
from mcp_servers.db_server import build_app as build_db_app
from mcp_servers.kowiki_server import build_app as build_kowiki_app

# 윈도우 콘솔 등에서 한글/이모지 출력이 깨지지 않도록 표준출력을 UTF-8 로 강제.
sys.stdout.reconfigure(encoding="utf-8")

# MCP 서버와 클라이언트가 공유하는 인증 키. 로컬 검증용 고정값이며,
# 서버 기동 시(build_app)와 클라이언트 생성 시 동일하게 넣어 인증을 통과시킨다.
API_KEY = "local-key"


def _free_port() -> int:
    """지금 당장 비어 있는 로컬 포트 번호 1개를 OS 로부터 받아 반환한다.

    포트 0 에 bind 하면 OS 가 사용 중이지 않은 포트를 알아서 골라준다. 그
    번호를 읽어 소켓을 닫고 돌려주므로, 여러 번 실행하거나 다른 서버와
    포트가 겹쳐 충돌하는 일을 막을 수 있다.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _serve(app, port: int):
    """FastAPI(app)를 지정 포트의 uvicorn 서버로 띄우고 준비될 때까지 기다린다.

    서버를 백그라운드 asyncio 태스크로 돌리기 때문에, 이 함수를 호출한 쪽은
    같은 프로세스 안에서 곧바로 클라이언트로 접속해 검증을 이어갈 수 있다.
    반환값은 (server, task) 튜플이며, 뒤에서 _stop() 에 그대로 넘겨 종료한다.

    매개변수:
        app  : uvicorn 이 서빙할 ASGI/FastAPI 앱 객체.
        port : 서버가 listen 할 포트 번호(보통 _free_port() 로 받은 값).
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    # serve() 는 무한 루프라 await 하지 않고 태스크로 분리해 뒤에서 계속 돌린다.
    task = asyncio.create_task(server.serve())
    # 서버가 실제로 소켓을 열기 전에 접속하면 실패하므로, started 플래그가
    # True 가 될 때까지 0.05초 간격으로 최대 10초(200회) 폴링해 레이스를 막는다.
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    return server, task


async def _stop(server, task) -> None:
    """_serve() 로 띄운 uvicorn 서버를 깨끗하게(정상) 종료한다.

    should_exit 플래그를 세우면 serve() 루프가 스스로 빠져나오며, 그 태스크가
    끝날 때까지 await 로 기다려 자원 누수 없이 마무리한다.
    """
    server.should_exit = True
    await task


async def check_db() -> None:
    """db MCP 서버를 실제로 띄우고, read-only SELECT 를 실 PostgreSQL 에 던져 검증한다.

    확인하는 흐름:
      1) build_db_app() 으로 db MCP 서버 앱을 만들고 빈 포트에 기동.
      2) McpClient 로 접속해 list_tools() 로 노출 도구 목록을 받아 출력.
      3) query 도구를 호출해 tb_knowledge 의 행수를 세는 SELECT 를 실 PG(.39)에 실행.

    조회만 하므로 운영 데이터는 절대 바뀌지 않는다. 끝나면 클라이언트와
    서버를 모두 정리한다.
    """
    print("\n=== db MCP 서버 (실 PostgreSQL read-only) ===")
    app = await build_db_app(api_key=API_KEY)
    port = _free_port()
    server, task = await _serve(app, port)
    try:
        # 방금 띄운 서버 주소로 MCP 클라이언트를 붙인다(동일 API_KEY 로 인증).
        client = McpClient(f"http://127.0.0.1:{port}", api_key=API_KEY)
        # 서버가 어떤 도구를 노출하는지 이름만 뽑아 확인.
        tools = await client.list_tools()
        print("  tools/list   :", [t["name"] for t in tools])
        # 실 PG 에 SELECT — tb_knowledge 행수를 세어 연결과 조회가 되는지 본다
        # (count 만 세므로 운영 데이터 변경 없음).
        result = await client.call_tool(
            "query", {"sql": "SELECT count(*) AS n FROM tb_knowledge"}
        )
        print("  query result :", result)
        # 주의: 쓰기 차단(DELETE 거부 등) 검증은 실 DB 에 던지지 않고 단위 테스트
        # (test_mcp_db_server.py, 32건)에서 mock 으로만 확인한다. 운영 DB 보호 목적.
        await client.aclose()
    finally:
        # 예외가 나더라도 서버는 반드시 종료(포트/자원 누수 방지).
        await _stop(server, task)


async def check_kowiki() -> None:
    """kowiki MCP 서버를 띄우고, 실 임베딩 서버 + pgvector 로 의미 검색을 검증한다.

    확인하는 흐름:
      1) build_kowiki_app() 으로 kowiki MCP 서버 앱을 만들고 빈 포트에 기동.
      2) McpClient 로 접속해 list_tools() 로 노출 도구 목록 출력.
      3) search 도구에 한글 질의("니체")를 던져, 질의문을 임베딩(.28:8002)한 뒤
         pgvector 유사도 검색으로 상위 3개 청크를 받아오는 전 구간을 확인.

    임베딩 서버가 쉬고 있었다면 첫 호출에서 모델 로딩(cold start)에 시간이
    걸릴 수 있어, 이 호출만 timeout 을 넉넉히 준다. 결과는 요약만 출력한다.
    """
    print("\n=== kowiki MCP 서버 (실 임베딩 + pgvector search) ===")
    app = await build_kowiki_app(api_key=API_KEY)
    port = _free_port()
    server, task = await _serve(app, port)
    try:
        # 방금 띄운 kowiki 서버로 클라이언트를 붙인다.
        client = McpClient(f"http://127.0.0.1:{port}", api_key=API_KEY)
        # 노출 도구 목록 확인(search 등이 보이는지).
        tools = await client.list_tools()
        print("  tools/list   :", [t["name"] for t in tools])
        # 임베딩 서버가 idle 이면 첫 호출에 cold start(~60초)가 걸린다(v7.1 Part 5).
        # 기본 5초로는 부족하므로 이 호출만 timeout 을 넉넉히(120초) 준다.
        print("  (임베딩 cold start 가능 — 최대 120초 대기)")
        # 실제 의미 검색 호출: "니체" 를 임베딩해 상위 3개 청크를 유사도 검색.
        result = await client.call_tool(
            "search", {"query": "니체", "top_k": 3}, timeout=120
        )
        # 결과는 보통 청크(리스트) — 정확한 형식은 서버 구현에 따르므로,
        # 여기서는 총 개수와 첫 항목만 짧게 요약해 콘솔이 넘치지 않게 출력한다.
        if isinstance(result, list):
            print(f"  search hits  : {len(result)}개")
            if result:
                # 첫 항목만 미리보기. dict 면 앞 4개 키만 값 60자로 잘라 보여주고,
                # 임베딩 벡터(embedding)는 길고 무의미하니 <vec> 로 대체한다.
                first = result[0]
                preview = str(first)[:120] if not isinstance(first, dict) else {
                    k: (str(v)[:60] if k != "embedding" else "<vec>")
                    for k, v in list(first.items())[:4]
                }
                print("  first hit    :", preview)
        else:
            # 리스트가 아닌 형태(에러 문자열 등)면 앞부분만 잘라 출력.
            print("  search result:", str(result)[:160])
        await client.aclose()
    finally:
        # 예외 여부와 무관하게 서버는 반드시 종료.
        await _stop(server, task)


async def main() -> None:
    """두 검증(check_db → check_kowiki)을 순서대로 실행하는 진입점.

    앞의 것이 실패(예외)하면 뒤 검증은 실행되지 않으므로, 문제 지점을
    위에서부터 차례로 좁혀갈 수 있다.
    """
    await check_db()
    await check_kowiki()
    print("\n=== MCP 실 백엔드 e2e 검증 종료 ===")


if __name__ == "__main__":
    # 스크립트로 직접 실행할 때만 asyncio 이벤트 루프를 열어 main() 을 돌린다.
    asyncio.run(main())
