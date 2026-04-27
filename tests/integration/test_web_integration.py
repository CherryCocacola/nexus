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

    async def test_metrics_exposes_agent_cache_section(
        self, client: AsyncClient
    ) -> None:
        """v0.14.2: /metrics에 "agent_cache" 섹션이 있어야 한다.

        Scout 결과 캐시의 히트/미스/크기를 관측할 수 있다.
        """
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_cache" in data
        cache = data["agent_cache"]
        for key in ("hits", "misses", "stored", "evicted", "size"):
            assert key in cache, f"'{key}' 누락"
            assert isinstance(cache[key], int)
            assert cache[key] >= 0

    # ─── Part 5 Ch 15 — 테넌트 목록 엔드포인트 ───
    async def test_tenants_endpoint_returns_list(self, client: AsyncClient) -> None:
        """GET /v1/tenants가 200 OK + 기본 구조를 반환한다.

        부트스트랩이 tenant_registry를 초기화하지 못한 환경에서도 빈 목록으로
        graceful하게 응답해야 하므로 'tenants', 'default_tenant', 'total' 키가
        반드시 존재한다.
        """
        resp = await client.get("/v1/tenants")

        assert resp.status_code == 200
        data = resp.json()
        assert "tenants" in data
        assert "default_tenant" in data
        assert "total" in data
        assert isinstance(data["tenants"], list)
        assert data["total"] == len(data["tenants"])

    async def test_tenants_endpoint_default_tenant_present(
        self, client: AsyncClient
    ) -> None:
        """레지스트리가 초기화된 경우 default 테넌트는 반드시 목록에 존재한다.

        bootstrap이 tenant_registry를 주입하면 default_tenant(보통 'default')가
        tenants 목록 안에도 있어야 정합성이 맞다.
        """
        resp = await client.get("/v1/tenants")
        data = resp.json()

        if data["total"] == 0:
            pytest.skip("tenant_registry가 초기화되지 않은 테스트 환경")

        ids = {t["id"] for t in data["tenants"]}
        assert data["default_tenant"] in ids

    async def test_tenants_endpoint_hides_raw_api_keys(
        self, client: AsyncClient
    ) -> None:
        """api_keys 원본은 응답에 절대 포함되지 않는다 (보안).

        대신 api_key_count(정수)만 노출한다.
        """
        resp = await client.get("/v1/tenants")
        data = resp.json()

        for tenant in data["tenants"]:
            assert "api_keys" not in tenant, (
                f"api_keys 원본이 응답에 노출됨: tenant={tenant['id']}"
            )
            assert "api_key_count" in tenant
            assert isinstance(tenant["api_key_count"], int)
            assert tenant["api_key_count"] >= 0

    async def test_tenants_endpoint_full_response_with_injected_registry(
        self, client: AsyncClient
    ) -> None:
        """tenant_registry를 직접 주입하여 완전한 응답 구조를 검증한다.

        보안 관점에서 가장 중요한 테스트 — 실제 api_keys가 있는 테넌트에 대해
        원본 값이 응답 본문 어디에도 새어나가지 않는지 plaintext grep으로 확인.
        """
        from core.config import TenantConfig, TenantRegistry
        from web.app import _app_state

        saved = _app_state.get("tenant_registry")
        try:
            _app_state["tenant_registry"] = TenantRegistry(
                default_tenant="default",
                tenants=[
                    TenantConfig(
                        id="default",
                        name="Default",
                        description="공통 테넌트",
                        allowed_knowledge_sources=["kowiki", "sample"],
                    ),
                    TenantConfig(
                        id="school-a",
                        name="A학교",
                        description="학교 A 전용 LoRA",
                        model_override="nexus-school-a",
                        allowed_knowledge_sources=["kowiki", "school-a-textbook"],
                        api_keys=["sk-secret-alpha", "sk-secret-beta"],
                        metadata={"dept": "engineering"},
                    ),
                ],
            )

            resp = await client.get("/v1/tenants")
            assert resp.status_code == 200
            data = resp.json()

            assert data["total"] == 2
            assert data["default_tenant"] == "default"

            by_id = {t["id"]: t for t in data["tenants"]}
            school = by_id["school-a"]
            assert school["name"] == "A학교"
            assert school["model_override"] == "nexus-school-a"
            assert school["allowed_knowledge_sources"] == [
                "kowiki",
                "school-a-textbook",
            ]
            assert school["api_key_count"] == 2
            assert school["metadata"] == {"dept": "engineering"}
            assert "api_keys" not in school

            # 응답 본문 어디에도 API 키 원본이 새어나가지 않아야 한다.
            body_text = resp.text
            assert "sk-secret-alpha" not in body_text
            assert "sk-secret-beta" not in body_text
        finally:
            _app_state["tenant_registry"] = saved

    # ─── Ch 16 — 세션 메시지 복원 엔드포인트 ───
    async def test_session_messages_404_for_unknown_session(
        self, client: AsyncClient
    ) -> None:
        """존재하지 않는 세션은 404를 반환해야 프론트가 '세션 없음'으로 분기 가능."""
        resp = await client.get("/v1/sessions/definitely-not-exist-xyz-999/messages")
        assert resp.status_code == 404

    @pytest.mark.parametrize("bad_id", ["foo\\bar", "foo..bar", "\x00"])
    async def test_session_messages_400_for_invalid_chars(
        self, client: AsyncClient, bad_id: str
    ) -> None:
        """핸들러까지 도달하는 의심 문자(백슬래시/연속 점/NUL)는 400으로 차단."""
        from urllib.parse import quote

        resp = await client.get(f"/v1/sessions/{quote(bad_id, safe='')}/messages")
        assert resp.status_code == 400, (
            f"bad_id={bad_id!r} 가 400이 아님: {resp.status_code}"
        )

    @pytest.mark.parametrize("bad_id", ["foo/bar", "a/../../etc/passwd", ".."])
    async def test_session_messages_router_blocks_slashes(
        self, client: AsyncClient, bad_id: str
    ) -> None:
        """슬래시/단독 ..는 FastAPI 라우터가 path 매칭에 실패시켜 404로 차단.

        핸들러 도달 전에 막히는 1차 방어선이 정상 동작함을 확인한다 (200이
        아니기만 하면 트랜스크립트 디렉토리 탈출은 발생할 수 없다).
        """
        from urllib.parse import quote

        resp = await client.get(f"/v1/sessions/{quote(bad_id, safe='')}/messages")
        # 200이 아니면 안전 — 200이면 디렉토리 탈출 가능성을 의미하므로 즉시 실패
        assert resp.status_code != 200, (
            f"bad_id={bad_id!r} 가 200을 반환 — 경로 탈출 가능성!"
        )
        assert resp.status_code in (400, 404), (
            f"bad_id={bad_id!r} 가 예상 외 코드: {resp.status_code}"
        )

    async def test_sessions_includes_title_hint(
        self, client: AsyncClient, tmp_path
    ) -> None:
        """GET /v1/sessions 응답에 title_hint 필드가 포함된다 (v0.14.5).

        첫 user 메시지의 앞부분이 요약되어 사이드바 제목으로 쓰이도록 백엔드가
        미리 계산해 돌려주는지 검증.
        """
        import json as _json
        from unittest.mock import MagicMock

        from web.app import _app_state

        sid = "title-hint-test-session"
        sdir = tmp_path / sid
        sdir.mkdir(parents=True)
        with (sdir / "transcript.jsonl").open("w", encoding="utf-8") as f:
            f.write(_json.dumps({
                "ts": "2026-04-23T00:00:00+00:00",
                "session_id": sid, "turn": 1,
                "role": "user", "content": "니체 철학 핵심 3가지",
            }, ensure_ascii=False) + "\n")

        saved = _app_state.get("config")
        try:
            mock_cfg = MagicMock()
            mock_cfg.session.sessions_dir = str(tmp_path)
            _app_state["config"] = mock_cfg

            resp = await client.get("/v1/sessions")
            assert resp.status_code == 200
            data = resp.json()
            # 새 디렉토리가 목록에 있고 title_hint가 있어야 한다
            by_id = {s["session_id"]: s for s in data["sessions"]}
            assert sid in by_id
            assert by_id[sid]["title_hint"] == "니체 철학 핵심 3가지"
        finally:
            _app_state["config"] = saved

    # ─── Ch 16 — DELETE /v1/sessions/{id} (v0.14.5) ───
    async def test_delete_session_removes_transcript(
        self, client: AsyncClient, tmp_path
    ) -> None:
        """DELETE는 트랜스크립트 디렉토리를 통째로 제거하고 결과를 알린다."""
        import json as _json
        from unittest.mock import MagicMock

        from web.app import _app_state

        sid = "delete-test-session"
        sdir = tmp_path / sid
        sdir.mkdir(parents=True)
        with (sdir / "transcript.jsonl").open("w", encoding="utf-8") as f:
            f.write(_json.dumps({
                "ts": "2026-04-23T00:00:00+00:00",
                "session_id": sid, "turn": 1,
                "role": "user", "content": "삭제될 대화",
            }, ensure_ascii=False) + "\n")

        saved = _app_state.get("config")
        saved_mm = _app_state.get("memory_manager")
        try:
            mock_cfg = MagicMock()
            mock_cfg.session.sessions_dir = str(tmp_path)
            _app_state["config"] = mock_cfg
            # Redis 없이 디스크 삭제만 검증
            _app_state["memory_manager"] = None

            resp = await client.delete(f"/v1/sessions/{sid}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == sid
            assert data["deleted_disk"] is True
            assert data["deleted_redis"] is False  # Redis 미주입
            assert not sdir.exists()
        finally:
            _app_state["config"] = saved
            _app_state["memory_manager"] = saved_mm

    async def test_delete_session_idempotent_when_missing(
        self, client: AsyncClient
    ) -> None:
        """없는 세션 삭제는 200 + 둘 다 False — 재시도 안전."""
        resp = await client.delete("/v1/sessions/never-existed-xyz-999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted_disk"] is False
        assert data["deleted_redis"] is False

    @pytest.mark.parametrize("bad_id", ["foo\\bar", "foo..bar", "\x00"])
    async def test_delete_session_400_for_invalid_chars(
        self, client: AsyncClient, bad_id: str
    ) -> None:
        """의심 문자가 포함된 session_id는 400으로 차단."""
        from urllib.parse import quote

        resp = await client.delete(f"/v1/sessions/{quote(bad_id, safe='')}")
        assert resp.status_code == 400

    async def test_session_messages_returns_transcript_content(
        self, client: AsyncClient, tmp_path
    ) -> None:
        """
        트랜스크립트 파일이 있으면 그 내용을 반환한다 (Redis 폴백 경로).

        Redis는 테스트 환경에서 실제 연결이 없으므로 자연스럽게 실패하고,
        파일 폴백으로 떨어진다. config.session.sessions_dir를 tmp로 주입하여
        파일 내용이 그대로 응답에 실리는지 검증.
        """
        import json
        from unittest.mock import MagicMock

        from web.app import _app_state

        # 트랜스크립트 파일을 수동으로 생성
        sid = "integration-test-transcript"
        sdir = tmp_path / sid
        sdir.mkdir(parents=True)
        tfile = sdir / "transcript.jsonl"
        with tfile.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": "2026-04-23T00:00:00+00:00",
                "session_id": sid, "turn": 1,
                "role": "user", "content": "안녕",
            }, ensure_ascii=False) + "\n")
            f.write(json.dumps({
                "ts": "2026-04-23T00:00:01+00:00",
                "session_id": sid, "turn": 1,
                "role": "assistant", "content": "안녕하세요!",
                "usage": {"input_tokens": 100, "output_tokens": 5},
            }, ensure_ascii=False) + "\n")

        # config 모킹 — session.sessions_dir만 tmp로 지정
        saved = _app_state.get("config")
        try:
            mock_cfg = MagicMock()
            mock_cfg.session.sessions_dir = str(tmp_path)
            _app_state["config"] = mock_cfg

            # memory_manager가 있으면 Redis 쪽이 먼저 시도되므로 None으로 덮어
            # 파일 폴백을 강제한다
            saved_mm = _app_state.get("memory_manager")
            _app_state["memory_manager"] = None

            resp = await client.get(f"/v1/sessions/{sid}/messages")
            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == sid
            assert data["source"] == "transcript"
            assert data["total"] == 2
            roles = [m["role"] for m in data["messages"]]
            contents = [m["content"] for m in data["messages"]]
            assert roles == ["user", "assistant"]
            assert contents == ["안녕", "안녕하세요!"]
        finally:
            _app_state["config"] = saved
            _app_state["memory_manager"] = saved_mm
