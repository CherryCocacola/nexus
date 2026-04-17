"""
Phase 8.0 통합 테스트 — Web API 엔드포인트.

FastAPI 앱의 모든 엔드포인트가 올바르게 응답하는지 검증한다.
httpx.AsyncClient + ASGITransport로 실제 HTTP 요청 없이 테스트한다.

사양서 Ch.22.6 Test 7에 해당한다.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from web.app import app


# ─────────────────────────────────────────────
# Test 7: Web API 통합 테스트
# ─────────────────────────────────────────────
@pytest.mark.asyncio
class TestWebAPIIntegration:
    """FastAPI 앱의 엔드포인트를 통합 테스트한다."""

    @pytest_asyncio.fixture
    async def client(self):
        """
        httpx AsyncClient를 생성한다.
        ASGITransport로 FastAPI 앱에 직접 요청한다 (실제 HTTP 서버 불필요).
        """
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    async def test_health_endpoint_returns_ok(self, client: AsyncClient) -> None:
        """GET /health가 200 OK + status="ok"를 반환한다."""
        resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    async def test_chat_endpoint_returns_session_id(self, client: AsyncClient) -> None:
        """POST /v1/chat가 session_id를 포함한 응답을 반환한다."""
        resp = await client.post("/v1/chat", json={"message": "안녕하세요"})

        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0
        assert "response" in data

    async def test_chat_endpoint_with_custom_session_id(self, client: AsyncClient) -> None:
        """POST /v1/chat에 session_id를 지정하면 그대로 반환한다."""
        custom_id = "my-test-session-123"
        resp = await client.post(
            "/v1/chat",
            json={"message": "Hello", "session_id": custom_id},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == custom_id

    async def test_chat_stream_endpoint_returns_sse(self, client: AsyncClient) -> None:
        """POST /v1/chat/stream이 SSE 형식으로 응답한다."""
        resp = await client.post("/v1/chat/stream", json={"message": "안녕"})

        assert resp.status_code == 200
        # Content-Type이 SSE 형식이어야 한다
        content_type = resp.headers.get("content-type", "")
        assert "text/event-stream" in content_type
        # 응답 본문에 SSE "data:" 접두사가 포함되어야 한다
        body = resp.text
        assert "data:" in body

    async def test_sessions_endpoint(self, client: AsyncClient) -> None:
        """GET /v1/sessions가 세션 목록을 반환한다."""
        resp = await client.get("/v1/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    async def test_tools_endpoint(self, client: AsyncClient) -> None:
        """GET /v1/tools가 도구 목록을 반환한다."""
        resp = await client.get("/v1/tools")

        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert "total" in data

    async def test_models_endpoint(self, client: AsyncClient) -> None:
        """GET /v1/models가 모델 목록을 반환한다."""
        resp = await client.get("/v1/models")

        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "total" in data
        # 최소 2개 모델 (primary + auxiliary)
        assert data["total"] >= 2

    async def test_metrics_endpoint(self, client: AsyncClient) -> None:
        """GET /metrics가 메트릭스 딕셔너리를 반환한다."""
        resp = await client.get("/metrics")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    async def test_metrics_exposes_agents_section(self, client: AsyncClient) -> None:
        """
        v7.0 Phase 9 이후: /metrics에 "agents" 섹션이 존재해야 한다.

        AgentTool.get_stats()가 subagent_type별 통계를 집계하며, 호출이 없으면
        빈 dict를 반환한다 (키 자체는 항상 존재).
        """
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert isinstance(data["agents"], dict)
        # 하위 호환 scout 섹션도 유지된다
        assert "scout" in data
        # Scout 호출이 아직 없으면 0이어야 한다
        assert data["scout"]["scout_calls"] >= 0
