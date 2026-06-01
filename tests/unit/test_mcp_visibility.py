"""
MCP 가시성(GlobalState.mcp_servers / mcp_connected) — 부트스트랩 채움 로직과
/metrics 노출을 검증한다.

대상 (v7.2/v7.3 MCP 통합):
  - core/bootstrap.py init_phase2 의 MCP 등록 결과 → GlobalState 반영 로직.
  - web/app.py /metrics 의 result["mcp"] 구조(connected_count/connected/tool_counts).

외부 서비스 미접근:
  - 실 MCP 서버/임베딩/PG/GPU 에 접근하지 않는다.
  - connect_and_register 는 AsyncMock 으로 가짜 등록 결과를 주입한다.
  - /metrics 는 ASGITransport(인프로세스)로 호출하고, _app_state["state"] 에
    가짜 GlobalState 를 주입해 mcp 섹션만 검증한다.

왜 init_phase2 전체를 돌리지 않는가:
  init_phase2 는 ModelProvider/24개 도구/Redis/PG/QueryEngine 까지 조립하는
  무거운 진입점이라, 본 단위 테스트는 "MCP 등록 결과 → GlobalState 반영" 이라는
  실제 변경 지점만 격리해 검증한다(McpConnectionManager 를 mock 으로 대체).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from core.state import GlobalState


# ─────────────────────────────────────────────
# 부트스트랩 MCP 반영 로직
# ─────────────────────────────────────────────
def _apply_mcp_visibility(state: GlobalState, registered: dict[str, list[str]]) -> None:
    """
    bootstrap.init_phase2 의 MCP 가시성 반영 블록과 동일한 변환을 수행한다.

    bootstrap.py(:391~395)와 1:1 대응:
      state.mcp_servers = {name: {"tools": [...], "tool_count": len(...)} ...}
      state.mcp_connected = {도구 1개 이상 등록된 서버명}
    이 헬퍼는 그 로직을 그대로 옮겨 단위 검증한다(무거운 init_phase2 우회).
    """
    state.mcp_servers = {
        name: {"tools": list(tools), "tool_count": len(tools)} for name, tools in registered.items()
    }
    state.mcp_connected = {name for name, tools in registered.items() if len(tools) >= 1}


class TestBootstrapMcpVisibility:
    """connect_and_register 결과가 GlobalState 에 올바르게 반영되는지 검증한다."""

    async def test_connect_and_register_populates_state(self):
        """{'kowiki': ['search'], 'failed': []} → tool_count=1, mcp_connected={'kowiki'}.

        빈 도구 리스트(failed)는 '연결 실패'로 보아 mcp_connected 에서 제외된다
        (도구가 1개 이상이어야 살아 있는 서버로 간주 — fail-closed 요약)."""
        # McpConnectionManager.connect_and_register 를 AsyncMock 으로 대체해
        # 실 MCP 서버 접속 없이 가짜 등록 결과를 만든다.
        fake_manager = AsyncMock()
        fake_manager.connect_and_register.return_value = {
            "kowiki": ["search"],
            "failed": [],  # 빈 리스트 — 연결 실패 서버
        }

        registered = await fake_manager.connect_and_register(object())
        state = GlobalState()
        _apply_mcp_visibility(state, registered)

        # kowiki 는 도구 1개 → 상세에 tool_count=1.
        assert state.mcp_servers["kowiki"]["tool_count"] == 1
        assert state.mcp_servers["kowiki"]["tools"] == ["search"]
        # failed 는 상세에는 남지만 tool_count=0.
        assert state.mcp_servers["failed"]["tool_count"] == 0
        # mcp_connected 는 도구가 있는 서버만(빈 리스트 서버 제외).
        assert state.mcp_connected == {"kowiki"}

    async def test_multiple_servers_connected_set(self):
        """여러 서버 중 도구가 있는 서버만 mcp_connected 에 포함된다."""
        fake_manager = AsyncMock()
        fake_manager.connect_and_register.return_value = {
            "kowiki": ["search"],
            "docingest": ["parse", "ingest", "search"],
            "diag": [],  # 연결 실패
        }
        registered = await fake_manager.connect_and_register(object())
        state = GlobalState()
        _apply_mcp_visibility(state, registered)

        assert state.mcp_connected == {"kowiki", "docingest"}
        assert state.mcp_servers["docingest"]["tool_count"] == 3
        assert state.mcp_servers["diag"]["tool_count"] == 0

    async def test_empty_registration_leaves_state_empty(self):
        """등록 결과가 비면 mcp_servers=={} 이고 mcp_connected==set() 이다."""
        state = GlobalState()
        _apply_mcp_visibility(state, {})
        assert state.mcp_servers == {}
        assert state.mcp_connected == set()


# ─────────────────────────────────────────────
# /metrics MCP 섹션
# ─────────────────────────────────────────────
@pytest.mark.asyncio
class TestMetricsMcpSection:
    """web/app.py /metrics 의 result['mcp'] 구조를 검증한다."""

    async def _get_metrics(self):
        """ASGITransport 로 /metrics 를 인프로세스 호출해 JSON 을 반환한다."""
        from httpx import ASGITransport, AsyncClient

        from web.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/metrics")
        assert resp.status_code == 200
        return resp.json()

    async def test_metrics_mcp_section_from_state(self):
        """state 에 MCP 가시성이 있으면 /metrics 에 connected_count/connected/tool_counts 노출.

        connected 는 정렬되어야 하고, tool_counts 는 서버별 tool_count 를 담아야 한다."""
        from web.app import _app_state

        # 가짜 state 를 주입한다(실 부트스트랩 미실행).
        fake_state = GlobalState()
        fake_state.mcp_servers = {
            "kowiki": {"tools": ["search"], "tool_count": 1},
            "docingest": {"tools": ["parse", "ingest", "search"], "tool_count": 3},
            "diag": {"tools": [], "tool_count": 0},
        }
        # diag 는 도구 0개라 connected 에서 제외(부트스트랩 요약 규칙과 동일).
        fake_state.mcp_connected = {"kowiki", "docingest"}

        saved = _app_state.get("state")
        _app_state["state"] = fake_state
        try:
            data = await self._get_metrics()
        finally:
            _app_state["state"] = saved

        assert "mcp" in data
        mcp = data["mcp"]
        assert mcp["connected_count"] == 2
        # connected 는 정렬된 리스트여야 한다.
        assert mcp["connected"] == ["docingest", "kowiki"]
        # tool_counts 는 서버별 개수(0 포함).
        assert mcp["tool_counts"] == {"kowiki": 1, "docingest": 3, "diag": 0}

    async def test_metrics_no_mcp_key_when_state_absent(self):
        """_app_state['state'] 가 없으면 result 에 mcp 키가 포함되지 않아야 한다.

        /metrics 는 state 가 있을 때만 session/mcp 섹션을 채운다(코드: if state:)."""
        from web.app import _app_state

        saved = _app_state.get("state")
        _app_state["state"] = None
        try:
            data = await self._get_metrics()
        finally:
            _app_state["state"] = saved

        # state 가 없으면 session 도 mcp 도 없어야 한다.
        assert "mcp" not in data
        assert "session" not in data
