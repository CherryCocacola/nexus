"""
core.tools.mcp — v7.2 MCP(Model Context Protocol) 클라이언트 어댑터 패키지.

LAN 내부 MCP 서버의 도구를 Nexus 도구 풀에 흡수한다.
MCP JSON-RPC는 이 패키지 내부(McpClient/McpToolAdapter)에만 캡슐화되며,
query_loop·executor는 일반 BaseTool로만 본다(P3 계약 유지).

공개 심볼:
  - McpClient            : JSON-RPC over LAN HTTP/SSE 저수준 클라이언트
  - McpToolAdapter       : 원격 도구 1개 → BaseTool 래퍼
  - McpConnectionManager : 발견·등록·정리 관리자
"""

from __future__ import annotations

from core.tools.mcp.adapter import McpToolAdapter
from core.tools.mcp.client import McpClient
from core.tools.mcp.connection_manager import McpConnectionManager

__all__ = [
    "McpClient",
    "McpToolAdapter",
    "McpConnectionManager",
]
