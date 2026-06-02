"""
mcp_servers 실행 entrypoint — LAN MCP 서버 1개를 uvicorn 으로 기동한다.

사용법:
  python -m mcp_servers.run <name> [--host 0.0.0.0] [--port <p>] [--api-key ...]

  <name> 은 다음 중 하나:
    db        — PostgreSQL read-only 조회 서버  (기본 포트 8810)
    diag      — 인프라 진단 서버                (기본 포트 8811)
    kowiki    — tb_knowledge RAG 검색 서버      (기본 포트 8813)
    docingest — 문서 인제스트 서버              (기본 포트 8814)

설계:
  각 서버 모듈은 async def build_app(api_key) -> FastAPI 를 노출한다. 이 entrypoint
  는 선택된 서버의 build_app() 을 호출(asyncio.run)해 FastAPI 인스턴스를 얻은 뒤
  uvicorn 으로 띄운다. build_app 은 DB 풀 등 await 가 필요한 자원을 만들기 때문에
  동기 팩토리가 아니라 코루틴이다.

에어갭:
  기본 host 는 0.0.0.0(LAN 바인드). 외부로의 발신 호출은 각 서버 도구가 LAN
  주소만 사용하므로 구조적으로 차단된다. 기본 api_key 는 placeholder 이며 운영
  시 --api-key 또는 환경변수로 주입한다.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger("nexus.mcp_servers.run")

# 서버 이름 → (모듈 build_app 경로, 기본 포트) 매핑.
# 값에 코루틴 팩토리 자체가 아니라 "지연 import 하는 람다" 를 두어, 선택된 서버의
# 의존성만 import 되게 한다(예: db 만 띄울 때 ingest 파서까지 import 하지 않음).
_SERVERS: dict[str, int] = {
    "db": 8810,
    "diag": 8811,
    "kowiki": 8813,
    "docingest": 8814,
}


async def _build(name: str, api_key: str) -> FastAPI:
    """선택된 서버의 build_app(api_key) 을 호출해 FastAPI 앱을 만든다."""
    if name == "db":
        from mcp_servers.db_server import build_app
    elif name == "diag":
        from mcp_servers.diag_server import build_app
    elif name == "kowiki":
        from mcp_servers.kowiki_server import build_app
    elif name == "docingest":
        from mcp_servers.docingest_server import build_app
    else:
        # argparse choices 로 이미 걸러지지만 방어적으로 한 번 더.
        raise ValueError(f"알 수 없는 서버 이름: {name!r}")
    return await build_app(api_key=api_key)


def main() -> None:
    """CLI 인자를 파싱하고 uvicorn 으로 서버를 기동한다."""
    parser = argparse.ArgumentParser(
        prog="mcp_servers.run",
        description="LAN MCP 서버 기동(db/diag/kowiki/docingest)",
    )
    parser.add_argument(
        "name",
        choices=sorted(_SERVERS.keys()),
        help="기동할 MCP 서버 이름",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104 — LAN 바인드(에어갭 내부망). 외부 노출은 방화벽으로 통제.
        help="바인드 호스트(기본 0.0.0.0 — LAN)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="바인드 포트(기본: 서버별 기본 포트)",
    )
    parser.add_argument(
        "--api-key",
        default="local-key",
        help="Bearer 인증 키(기본 placeholder, 운영 시 명시 주입)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    port = args.port if args.port is not None else _SERVERS[args.name]

    # build_app 은 코루틴 — DB 풀 등 비동기 자원을 만들기 위해 먼저 실행한다.
    app = asyncio.run(_build(args.name, args.api_key))

    logger.info(
        "MCP 서버 '%s' 기동: http://%s:%d (POST '/' JSON-RPC, GET '/health')",
        args.name,
        args.host,
        port,
    )
    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
