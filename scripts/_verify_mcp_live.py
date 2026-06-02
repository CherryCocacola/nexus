"""MCP 서버 ↔ 클라이언트 실 백엔드 e2e 검증 (개발용).

실제 uvicorn 서버(db / kowiki)를 임의 포트에 띄우고, 우리 McpClient 로
list_tools → call_tool 을 호출해 **실 PostgreSQL(.39) / 임베딩 서버(.28:8002)**
와의 전 구간 연결을 확인한다. 전부 read-only 라 운영 데이터(tb_knowledge)를
변경하지 않는다.

실행: python -m scripts._verify_mcp_live   (또는 python scripts/_verify_mcp_live.py)
"""

from __future__ import annotations

import asyncio
import socket
import sys

import uvicorn

from core.tools.mcp.client import McpClient
from mcp_servers.db_server import build_app as build_db_app
from mcp_servers.kowiki_server import build_app as build_kowiki_app

sys.stdout.reconfigure(encoding="utf-8")

API_KEY = "local-key"


def _free_port() -> int:
    """사용 가능한 로컬 포트 1개를 OS에서 할당받는다(테스트 충돌 방지)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _serve(app, port: int):
    """FastAPI 앱을 임의 포트의 uvicorn 서버로 띄우고, 준비될 때까지 대기한다."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # server.started 가 True 가 될 때까지 짧게 폴링(레이스 방지)
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    return server, task


async def _stop(server, task) -> None:
    """uvicorn 서버를 깨끗하게 종료한다."""
    server.should_exit = True
    await task


async def check_db() -> None:
    """db MCP 서버를 띄우고 read-only SELECT 를 실 PG 에 던져본다."""
    print("\n=== db MCP 서버 (실 PostgreSQL read-only) ===")
    app = await build_db_app(api_key=API_KEY)
    port = _free_port()
    server, task = await _serve(app, port)
    try:
        client = McpClient(f"http://127.0.0.1:{port}", api_key=API_KEY)
        tools = await client.list_tools()
        print("  tools/list   :", [t["name"] for t in tools])
        # 실 PG 에 SELECT — tb_knowledge 행수(운영 데이터 변경 없음)
        result = await client.call_tool(
            "query", {"sql": "SELECT count(*) AS n FROM tb_knowledge"}
        )
        print("  query result :", result)
        # 주의: 쓰기 차단(DELETE 거부)은 실 DB 에 던지지 않고 단위 테스트
        # (test_mcp_db_server.py, 32건)에서 mock 으로 검증한다. 운영 DB 보호.
        await client.aclose()
    finally:
        await _stop(server, task)


async def check_kowiki() -> None:
    """kowiki MCP 서버를 띄우고 실 임베딩+pgvector 로 검색해본다."""
    print("\n=== kowiki MCP 서버 (실 임베딩 + pgvector search) ===")
    app = await build_kowiki_app(api_key=API_KEY)
    port = _free_port()
    server, task = await _serve(app, port)
    try:
        client = McpClient(f"http://127.0.0.1:{port}", api_key=API_KEY)
        tools = await client.list_tools()
        print("  tools/list   :", [t["name"] for t in tools])
        # 임베딩 서버가 idle 이면 첫 호출에 cold start(~60초)가 걸린다(v7.1 Part 5).
        # 기본 5초로는 부족하므로 이 호출만 timeout 을 넉넉히(120초) 준다.
        print("  (임베딩 cold start 가능 — 최대 120초 대기)")
        result = await client.call_tool(
            "search", {"query": "니체", "top_k": 3}, timeout=120
        )
        # 결과는 청크 리스트(형식은 서버 구현에 따름) — 개수/첫 항목 요약만 출력
        if isinstance(result, list):
            print(f"  search hits  : {len(result)}개")
            if result:
                first = result[0]
                preview = str(first)[:120] if not isinstance(first, dict) else {
                    k: (str(v)[:60] if k != "embedding" else "<vec>")
                    for k, v in list(first.items())[:4]
                }
                print("  first hit    :", preview)
        else:
            print("  search result:", str(result)[:160])
        await client.aclose()
    finally:
        await _stop(server, task)


async def main() -> None:
    await check_db()
    await check_kowiki()
    print("\n=== MCP 실 백엔드 e2e 검증 종료 ===")


if __name__ == "__main__":
    asyncio.run(main())
