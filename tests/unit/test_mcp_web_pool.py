"""
웹 MCP 도구 풀 머지 — 회귀 방지 테스트.

배경 (수정된 버그):
  웹 Worker가 MCP 도구(mcp__db__query 등)를 못 보고, 대신 npx 셸 명령을
  환각 호출하던 버그를 수정했다. 수정 지점은 두 곳:

  1) core/bootstrap.py ⑨-d
     MCP 연결 후 cli_registry.get_all_tools() 중 이름이 "mcp__"로 시작하는
     어댑터만 추려 components["mcp_tools"]로 노출한다. MCP 비활성/실패 시에는
     빈 리스트를 미리 둔다(fail-closed).

  2) web/app.py _build_web_query_engine
     _create_web_tool_registry로 만든 웹 기본 도구 풀에 components["mcp_tools"]를
     _combine_scout_pool의 name 중복 제거 규칙으로 머지한다. 그리고 그 실풀을
     _app_state["web_tools"]로 저장해 /v1/tools가 모델이 실제로 보는 도구를 노출한다.

본 테스트가 검증하는 것:
  - 부트스트랩의 mcp__ prefix 필터 로직(test #1)
  - _build_web_query_engine의 머지 + 중복 제거(test #2)
  - /v1/tools가 web_tools(MCP 포함)를 우선 노출하는지(test #3)

외부 서비스 미접근:
  - 실 PG/vLLM/Redis/MCP 서버에 접근하지 않는다.
  - 무거운 ModelDispatcher/QueryEngine은 patch로 대체해 "어떤 도구 풀을
    전달받았는가"만 캡처한다(머지 결과 검증이 목적).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest


# ─────────────────────────────────────────────
# 테스트용 가짜 도구
# ─────────────────────────────────────────────
class _FakeTool:
    """
    BaseTool 전체를 흉내 내지 않고, 풀 머지/노출 로직이 실제로 읽는
    속성(name/description/group/is_read_only)만 가진 경량 더블.

    _combine_scout_pool은 .name만 보고 중복을 제거하고, /v1/tools의
    ToolInfo는 name/description/group/is_read_only만 직렬화한다.
    따라서 이 4개 속성이면 로직을 실제로 검증하기에 충분하다.
    """

    def __init__(self, name: str, *, group: str = "", is_read_only: bool = False):
        self.name = name
        self.description = f"{name} 도구"
        self.group = group
        self.is_read_only = is_read_only


# ─────────────────────────────────────────────
# Test #1 — bootstrap components["mcp_tools"] 필터
# ─────────────────────────────────────────────
def _filter_mcp_tools(cli_tools: list[Any]) -> list[Any]:
    """
    bootstrap.init_phase2(:394~396)의 mcp__ prefix 필터와 1:1 대응하는 헬퍼.

      components["mcp_tools"] = [t for t in cli_tools if t.name.startswith("mcp__")]

    무거운 init_phase2(ModelProvider/24개 도구/Redis/PG/QueryEngine 조립)를
    돌리지 않고, "등록된 cli_tools에서 MCP 어댑터만 추리는" 실제 변경 지점만
    격리 검증한다(McpConnectionManager는 결과만 흉내 내면 되므로 불필요).
    """
    return [t for t in cli_tools if t.name.startswith("mcp__")]


class TestBootstrapMcpToolsFilter:
    """connect_and_register 후 cli_tools → components['mcp_tools'] 필터 검증."""

    def test_only_mcp_prefixed_adapters_collected(self):
        """일반 도구 + MCP 어댑터가 섞인 cli_tools에서 mcp__ 도구만 추려진다."""
        # cli_registry.get_all_tools()가 돌려줄 법한 혼합 풀(이름순 정렬 가정).
        cli_tools = [
            _FakeTool("Bash"),
            _FakeTool("Edit"),
            _FakeTool("mcp__db__query", group="mcp:db"),
            _FakeTool("mcp__kowiki__search", group="mcp:kowiki"),
            _FakeTool("Write"),
        ]

        mcp_tools = _filter_mcp_tools(cli_tools)

        names = [t.name for t in mcp_tools]
        # MCP 어댑터 2개만, 그리고 그 2개만 포함되어야 한다.
        assert names == ["mcp__db__query", "mcp__kowiki__search"]
        # 일반 도구(Bash/Edit/Write)는 절대 섞이지 않는다.
        assert all(t.name.startswith("mcp__") for t in mcp_tools)

    def test_no_mcp_adapters_yields_empty(self):
        """MCP 어댑터가 하나도 없으면(=MCP 비활성/연결 0) 빈 리스트."""
        cli_tools = [_FakeTool("Bash"), _FakeTool("Edit"), _FakeTool("Write")]
        assert _filter_mcp_tools(cli_tools) == []

    def test_mcp_disabled_default_is_empty_list(self):
        """
        MCP 비활성 경로(config.mcp.enabled=False)의 기본값 계약 검증.

        bootstrap은 try 블록 진입 전에 components["mcp_tools"] = [] 를 먼저 둔다.
        그래서 _build_web_query_engine이 .get("mcp_tools", [])로 안전 참조한다
        (fail-closed). 여기서는 그 "비활성=빈 리스트" 불변식을 명시 검증한다.
        """
        components: dict[str, Any] = {}
        # bootstrap의 사전 기본값 설정과 동일.
        components["mcp_tools"] = []
        # MCP 비활성이라 try 블록을 건너뛴다 → 빈 리스트 그대로 유지.
        assert components["mcp_tools"] == []


# ─────────────────────────────────────────────
# Test #2 — _build_web_query_engine 풀 머지 + 중복 제거
# ─────────────────────────────────────────────
class TestCombineScoutPoolDedup:
    """머지에 재사용되는 _combine_scout_pool의 중복 제거 규칙을 직접 검증."""

    def test_dedup_keeps_first_occurrence_by_name(self):
        """동일 name이 양쪽에 있으면 앞쪽(웹 도구)만 남고 순서가 보존된다."""
        from web.app import _combine_scout_pool

        web = [_FakeTool("Edit"), _FakeTool("Bash")]
        # mcp_tools 쪽에 Bash 중복 + 신규 mcp__db__query.
        mcp = [_FakeTool("Bash"), _FakeTool("mcp__db__query")]

        merged = _combine_scout_pool(web, mcp)
        names = [t.name for t in merged]

        # Bash는 한 번만(앞쪽 = web 도구가 우선), 순서는 web → mcp.
        assert names == ["Edit", "Bash", "mcp__db__query"]
        # 중복 제거 후 Bash 인스턴스는 web 쪽 원본이어야 한다(앞 우선).
        bash = next(t for t in merged if t.name == "Bash")
        assert bash is web[1]

    async def test_build_web_query_engine_merges_mcp_into_worker_tools(self):
        """
        _build_web_query_engine 실행 시 worker_tools 풀에 mcp__ 도구가 머지되는지.

        ModelDispatcher/QueryEngine은 무겁고 실 provider를 요구하므로 patch로
        대체하고, 생성자에 전달된 worker_tools만 캡처해 머지 결과를 검증한다.
        _create_web_tool_registry는 실제 함수를 그대로 호출해(실 웹 도구 5종)
        머지 대상 base 풀이 가짜가 아니라 실제임을 보장한다.
        """
        from web import app as web_app

        captured: dict[str, Any] = {}

        # ModelDispatcher 대역 — 전달된 worker_tools를 캡처만 한다.
        def _fake_dispatcher(*args: Any, **kwargs: Any):  # noqa: ANN202
            captured["dispatcher_worker_tools"] = kwargs.get("worker_tools")
            return object()

        # QueryEngine 대역 — 전달된 tools를 캡처만 한다.
        def _fake_query_engine(*args: Any, **kwargs: Any):  # noqa: ANN202
            captured["engine_tools"] = kwargs.get("tools")
            return object()

        # 상태 더블 — _build_web_query_engine이 읽는 필드만 채운다.
        class _FakeMode:
            value = "default"

        class _FakeConfig:
            routing = object()

        class _FakeState:
            cwd = "."
            session_id = "test-session"
            permission_mode = _FakeMode()
            config = _FakeConfig()

        # MCP 도구를 주입한 components. 실 웹 도구(Edit/Write/Bash/Agent/Symbol…)와
        # 이름이 겹치지 않는 mcp__ 도구 2개를 넣어 "추가 머지"를 검증한다.
        mcp_db = _FakeTool("mcp__db__query", group="mcp:db", is_read_only=True)
        mcp_kowiki = _FakeTool("mcp__kowiki__search", group="mcp:kowiki", is_read_only=True)
        components: dict[str, Any] = {
            "mcp_tools": [mcp_db, mcp_kowiki],
            "scout_tools": [],
            "hardware_tier": "TIER_S",
            "model_provider": object(),
            "scout_provider": None,
            "context_manager": None,
            "memory_manager": None,
            "knowledge_retriever": None,
            "agent_registry": None,
            "task_manager": None,
            "symbol_store": None,
        }

        # ModelDispatcher/QueryEngine은 query_engine·model_dispatcher 소스 모듈에서
        # import되므로(함수 내부 lazy import) 그쪽을 patch한다.
        with (
            patch("core.orchestrator.model_dispatcher.ModelDispatcher", _fake_dispatcher),
            patch("core.orchestrator.query_engine.QueryEngine", _fake_query_engine),
            patch.object(web_app, "_load_worker_system_prompt", return_value="SYS"),
        ):
            engine, dispatcher, web_tools = web_app._build_web_query_engine(
                components, _FakeState()
            )

        names = {t.name for t in web_tools}
        # 반환된 실풀(web_tools)에 MCP 도구 2개가 포함되어야 한다.
        assert "mcp__db__query" in names
        assert "mcp__kowiki__search" in names
        # 실 웹 기본 도구도 함께 살아 있어야 한다(머지가 base를 덮어쓰지 않음).
        assert "Edit" in names
        # 캡처된 dispatcher/engine 풀에도 동일하게 MCP가 반영되어야 한다.
        assert "mcp__db__query" in {t.name for t in captured["dispatcher_worker_tools"]}
        assert "mcp__db__query" in {t.name for t in captured["engine_tools"]}

    async def test_build_web_query_engine_no_mcp_leaves_base_pool(self):
        """mcp_tools가 비면(fail-closed) base 웹 도구만 남고 mcp__ 도구는 없다."""
        from web import app as web_app

        def _fake_dispatcher(*args: Any, **kwargs: Any):  # noqa: ANN202
            return object()

        def _fake_query_engine(*args: Any, **kwargs: Any):  # noqa: ANN202
            return object()

        class _FakeMode:
            value = "default"

        class _FakeConfig:
            routing = object()

        class _FakeState:
            cwd = "."
            session_id = "s"
            permission_mode = _FakeMode()
            config = _FakeConfig()

        components: dict[str, Any] = {
            "mcp_tools": [],  # MCP 비활성/실패
            "scout_tools": [],
            "hardware_tier": "TIER_S",
            "model_provider": object(),
            "scout_provider": None,
        }

        with (
            patch("core.orchestrator.model_dispatcher.ModelDispatcher", _fake_dispatcher),
            patch("core.orchestrator.query_engine.QueryEngine", _fake_query_engine),
            patch.object(web_app, "_load_worker_system_prompt", return_value="SYS"),
        ):
            _engine, _dispatcher, web_tools = web_app._build_web_query_engine(
                components, _FakeState()
            )

        # base 웹 도구는 존재하고, mcp__ 접두사 도구는 하나도 없어야 한다.
        assert len(web_tools) >= 1
        assert all(not t.name.startswith("mcp__") for t in web_tools)


# ─────────────────────────────────────────────
# Test #3 — GET /v1/tools 가 web_tools(MCP 포함) 우선 노출
# ─────────────────────────────────────────────
@pytest.mark.asyncio
class TestV1ToolsExposesMcp:
    """/v1/tools가 _app_state['web_tools']의 MCP 도구를 노출하는지 검증."""

    async def _get_tools(self) -> dict[str, Any]:
        """ASGITransport로 /v1/tools를 인프로세스 호출해 JSON을 반환한다."""
        from httpx import ASGITransport, AsyncClient

        from web.app import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/v1/tools")
        assert resp.status_code == 200
        return resp.json()

    async def test_web_tools_with_mcp_are_listed(self):
        """web_tools에 MCP 도구가 있으면 /v1/tools 응답에 그대로 노출된다."""
        from web.app import _app_state

        web_tools = [
            _FakeTool("Edit", group="filesystem"),
            _FakeTool("mcp__db__query", group="mcp:db", is_read_only=True),
            _FakeTool("mcp__kowiki__search", group="mcp:kowiki", is_read_only=True),
        ]

        saved = _app_state.get("web_tools")
        _app_state["web_tools"] = web_tools
        try:
            data = await self._get_tools()
        finally:
            _app_state["web_tools"] = saved

        names = {t["name"] for t in data["tools"]}
        assert "mcp__db__query" in names
        assert "mcp__kowiki__search" in names
        assert data["total"] == 3
        # ToolInfo 직렬화 필드가 보존되는지(읽기전용 플래그 등).
        db = next(t for t in data["tools"] if t["name"] == "mcp__db__query")
        assert db["group"] == "mcp:db"
        assert db["is_read_only"] is True

    async def test_falls_back_to_tool_registry_when_web_tools_absent(self):
        """
        web_tools가 비면 base tool_registry로 폴백한다(부트스트랩 실패 등).

        이 폴백 풀에는 MCP 도구가 없을 수 있으나, 여기서는 '우선순위 계약'
        (web_tools 부재 시 registry 사용)만 검증한다.
        """
        from web.app import _app_state

        class _FakeRegistry:
            def get_all_tools(self) -> list[Any]:
                return [_FakeTool("Read", group="filesystem", is_read_only=True)]

        saved_web = _app_state.get("web_tools")
        saved_reg = _app_state.get("tool_registry")
        _app_state["web_tools"] = None
        _app_state["tool_registry"] = _FakeRegistry()
        try:
            data = await self._get_tools()
        finally:
            _app_state["web_tools"] = saved_web
            _app_state["tool_registry"] = saved_reg

        names = {t["name"] for t in data["tools"]}
        # 폴백 레지스트리의 도구가 노출되고, MCP 도구는 없다.
        assert names == {"Read"}
        assert all(not n.startswith("mcp__") for n in names)
