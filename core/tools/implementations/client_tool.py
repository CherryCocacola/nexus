# 클라이언트가 대신 실행하는 도구 — 서버는 호출 판단만 하고 실행하지 않는다.
"""
ClientTool — 도구 실행을 호출자(클라이언트)에게 넘기기 위한 껍데기.

[왜 필요한가 — 2026-08-08]
  VSCode 플러그인이 "CLI 와 같은 코딩 능력"을 갖게 하려면, 모델이 파일을 읽고
  고치고 테스트를 돌리는 **판단**을 해야 한다. 그런데 그 실행을 서버가 하면 두 가지가
  깨진다.

    ① 보안 — 서버 도구는 서버 파일시스템을 만진다. 실측으로 테넌트 키 하나에
       `/app/config/tenants.yaml`(전 테넌트 API 키)이 읽혔다(그래서 웹에서 Bash 를
       제거했다). 코딩용이라고 되돌려 주면 같은 구멍이 다시 열린다.
    ② 쓸모 — 개발자의 코드는 **개발자 PC** 에 있다. 서버의 Read 는 `/app` 을 읽지
       VSCode 에서 열어 놓은 프로젝트를 읽지 않는다.

  그래서 실행은 클라이언트가 한다. 서버는 "이 도구를 이 인자로 부르라"까지만 정하고
  그대로 돌려준다. 이것이 OpenAI `tools` / `tool_calls` 규약의 본래 동작이기도 하다 —
  루프의 주인은 클라이언트다.

[이 클래스가 하는 일]
  클라이언트가 보낸 JSON 스키마(OpenAI function 형식)를 BaseTool 로 감싼다.
  모델에게는 여느 도구와 똑같이 보이지만, 서버는 이것을 **절대 실행하지 않는다**
  (query_loop 이 `is_client_executed` 를 보고 실행 대기열에 넣지 않는다).

  call() 이 불리면 그것은 배선이 잘못됐다는 뜻이므로 조용히 넘어가지 않고 오류를 낸다.

작성자: 이현수 / 작성일: 2026-08-08
"""

from __future__ import annotations

from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 클라이언트가 보낸 스키마에서 받아들일 최대 도구 개수·이름 길이.
#   무제한이면 프롬프트가 통째로 도구 목록으로 채워져 본래 작업이 밀린다.
MAX_CLIENT_TOOLS = 64
MAX_TOOL_NAME_CHARS = 64


class ClientTool(BaseTool):
    """클라이언트가 실행할 도구의 스키마만 담는 껍데기(서버는 실행하지 않는다)."""

    # query_loop 이 이 플래그를 보고 실행 대기열에서 제외한다.
    is_client_executed = True

    def __init__(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        self._name = name
        self._description = description
        self._input_schema = input_schema or {"type": "object", "properties": {}}

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def group(self) -> str:
        return "client"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return self._input_schema

    # ═══ 3. Behavior Flags ═══
    # 서버는 아무 것도 하지 않으므로 부작용이 없다. 다만 "안전하다"는 뜻이 아니라
    # "서버가 실행하지 않는다"는 뜻이다 — 실제 위험은 클라이언트 쪽 정책이 진다.

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """서버 권한 판정 대상이 아니다 — 실행 주체가 클라이언트이기 때문이다."""
        return PermissionResult(
            behavior=PermissionBehavior.ALLOW,
            message=f"client-executed: {self._name}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """호출되면 안 된다 — 불렸다는 것은 실행 차단 배선이 깨졌다는 뜻이다.

        조용히 빈 결과를 돌려주면 "서버가 실행한 척" 이 되어 문제가 숨는다.
        그래서 명시적으로 실패시킨다.
        """
        return ToolResult.error(
            f"'{self._name}' 은 클라이언트가 실행하는 도구입니다. "
            "서버는 이 도구를 실행하지 않습니다(배선 오류)."
        )


def build_client_tools(specs: list[dict[str, Any]]) -> list[ClientTool]:
    """OpenAI `tools` 배열을 ClientTool 목록으로 바꾼다.

    형식이 어긋난 항목은 조용히 건너뛴다 — 외부 클라이언트가 보내는 값이라
    하나가 이상하다고 요청 전체를 실패시키면 연동이 취약해진다.

    Args:
        specs: `[{"type": "function", "function": {"name", "description", "parameters"}}]`

    Returns:
        ClientTool 목록(중복 이름 제거, 최대 MAX_CLIENT_TOOLS 개).
    """
    tools: list[ClientTool] = []
    seen: set[str] = set()
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        fn = spec.get("function") if isinstance(spec.get("function"), dict) else spec
        name = str(fn.get("name") or "").strip()
        if not name or len(name) > MAX_TOOL_NAME_CHARS or name in seen:
            continue
        seen.add(name)
        tools.append(
            ClientTool(
                name=name,
                description=str(fn.get("description") or ""),
                input_schema=fn.get("parameters") or fn.get("input_schema") or {},
            )
        )
        if len(tools) >= MAX_CLIENT_TOOLS:
            break
    return tools
