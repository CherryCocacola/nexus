"""
mcp_servers — 사내 시스템을 LAN MCP 서버로 노출하는 독립 서비스 패키지.

이 패키지는 Nexus 오케스트레이터가 아니라 "사내 시스템을 감싸는 서버 측" 이며,
이미 구현된 클라이언트(core/tools/mcp/client.py)와 JSON-RPC 2.0 으로 통신한다.

의존성 방향 (P2):
  mcp_servers/ → core/ (코어 재사용)만 허용. core/ 는 mcp_servers/ 를 절대 import
  하지 않는다.

구성:
  framework.py        — McpServerTool(ABC) + create_mcp_app(JSON-RPC 디스패치/인증)
  run.py              — `python -m mcp_servers.run <name>` uvicorn entrypoint
  db_server.py        — PostgreSQL read-only 조회(query)
  diag_server.py      — 인프라 진단(reachability, rag_latency)
  kowiki_server.py    — tb_knowledge RAG 검색(search)
  docingest_server.py — 문서 인제스트(parse/ingest/search)

각 서버 모듈은 `async def build_app(api_key) -> FastAPI` 를 노출한다.
"""

from __future__ import annotations

from mcp_servers.framework import McpServerTool, create_mcp_app

__all__ = ["McpServerTool", "create_mcp_app"]
