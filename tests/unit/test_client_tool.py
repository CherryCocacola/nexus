# 클라이언트 실행 도구 — 서버가 절대 실행하지 않는다는 성질을 고정한다.
"""
`ClientTool` 과 query_loop 의 실행 차단 배선을 검증한다 (2026-08-08).

[왜 생겼나]
    VSCode 플러그인이 "CLI 와 같은 코딩 능력"을 가지려면 모델이 파일을 읽고 고치는
    판단을 해야 한다. 그런데 그 실행을 **서버가 하면 안 된다**.

      ① 보안 — 실측으로 테넌트 키 하나에 /app/config/tenants.yaml(전 테넌트 API 키)이
         읽혔다. 그래서 웹에서 Bash 를 제거했는데, 코딩용이라고 서버 도구를 되돌려
         주면 같은 구멍이 다시 열린다.
      ② 쓸모 — 개발자의 코드는 개발자 PC 에 있다. 서버의 Read 는 /app 을 읽는다.

    그래서 실행은 클라이언트가 하고 서버는 tool_calls 만 돌려준다.
    아래 테스트는 그 경계가 다시 무너지지 않게 고정한다.
"""

from __future__ import annotations

import pytest

from core.tools.base import PermissionBehavior, ToolUseContext
from core.tools.implementations.client_tool import (
    MAX_CLIENT_TOOLS,
    ClientTool,
    build_client_tools,
)

CTX = ToolUseContext(cwd=".")

OPENAI_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the user's workspace",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


# ─────────────────────────────────────────────
# 스키마 변환
# ─────────────────────────────────────────────


def test_builds_from_openai_tool_spec():
    tools = build_client_tools(OPENAI_SPEC)

    assert len(tools) == 1
    assert tools[0].name == "read_file"
    assert tools[0].input_schema["properties"]["path"]["type"] == "string"


def test_malformed_entries_are_skipped_not_fatal():
    """외부 클라이언트가 보내는 값이라, 하나 이상하다고 요청 전체를 죽이지 않는다."""
    tools = build_client_tools(
        [*OPENAI_SPEC, {"function": {"name": ""}}, "문자열", None, {"function": {}}]
    )

    assert [t.name for t in tools] == ["read_file"]


def test_duplicate_names_are_deduped():
    tools = build_client_tools(OPENAI_SPEC + OPENAI_SPEC)

    assert len(tools) == 1


def test_tool_count_is_capped():
    """무제한이면 프롬프트가 도구 목록으로 채워져 본래 작업이 밀린다."""
    many = [
        {"function": {"name": f"t{i}", "parameters": {}}}
        for i in range(MAX_CLIENT_TOOLS + 20)
    ]

    assert len(build_client_tools(many)) == MAX_CLIENT_TOOLS


# ─────────────────────────────────────────────
# ★서버는 실행하지 않는다
# ─────────────────────────────────────────────


def test_marked_as_client_executed():
    """query_loop 이 이 플래그를 보고 실행 대기열에서 제외한다."""
    assert ClientTool("x", "", {}).is_client_executed is True


@pytest.mark.asyncio
async def test_call_fails_loudly_if_ever_reached():
    """★불렸다는 것은 차단 배선이 깨졌다는 뜻이다 — 조용히 넘어가면 안 된다.

    빈 결과를 돌려주면 "서버가 실행한 척"이 되어 문제가 숨는다.
    """
    result = await ClientTool("read_file", "", {}).call({"path": "/etc/passwd"}, CTX)

    assert result.is_error
    assert "클라이언트가 실행" in result.error_message


@pytest.mark.asyncio
async def test_permission_is_not_a_server_decision():
    """서버 권한 판정 대상이 아니다 — 실행 주체가 클라이언트이기 때문이다."""
    result = await ClientTool("read_file", "", {}).check_permissions({}, CTX)

    assert result.behavior == PermissionBehavior.ALLOW


def test_query_loop_collects_client_tool_names():
    """query_loop 이 도구 목록에서 클라이언트 실행 대상을 골라내는지 확인한다.

    실제 루프를 돌리지 않고, 이름 수집에 쓰는 판정(`is_client_executed`)이
    서버 도구와 클라이언트 도구를 정확히 가르는지만 본다.
    """
    from core.tools.implementations.write_tool import WriteTool

    tools = [WriteTool(), *build_client_tools(OPENAI_SPEC)]
    client_names = {t.name for t in tools if getattr(t, "is_client_executed", False)}

    assert client_names == {"read_file"}
    assert "Write" not in client_names  # 서버 도구는 종전대로 서버가 실행한다


def test_server_tools_are_not_client_executed():
    """기존 도구에 이 플래그가 새어 들어가면 서버 실행이 통째로 멈춘다."""
    from core.bootstrap import _create_tool_registry

    for tool in _create_tool_registry().get_all_tools():
        assert getattr(tool, "is_client_executed", False) is False, tool.name
