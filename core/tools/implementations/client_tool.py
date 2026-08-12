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

# ─────────────────────────────────────────────
# ★서버 실행으로 남겨 두는 도구 — 이 목록을 늘리는 것은 보안 결정이다
# ─────────────────────────────────────────────
# 원칙은 "클라이언트 도구가 오면 서버 도구 풀을 통째로 교체" 다(위 ① 참조).
# 딱 하나 예외를 둔다 — **비전 분석**.
#
#   왜 예외가 필요한가(2026-08-12):
#     A.X-4.0 은 텍스트 전용이라 이미지를 볼 수 없고, 비전은 별도 서버(Gemma3-27B)에
#     있다. 그 서버를 부르는 것은 `AnalyzeImage` 뿐인데, 도구를 전부 교체해 버리면
#     플러그인 요청에는 이 도구가 **존재하지 않는다**. 클라이언트가 대신 실행할 수도
#     없다 — 개발자 PC 에는 비전 서버가 없다.
#
#   왜 안전한가:
#     이 도구는 `is_read_only=True` 이고, `_resolve_uploaded_image()` 가 업로드
#     디렉토리 **밖의 경로를 DENY** 한다(경로 순회 차단). Bash/Read 처럼 임의 파일에
#     닿지 않으므로, 테넌트 키로 tenants.yaml 이 읽혔던 그 부류의 위험이 아니다.
#
#   왜 목록을 고정하는가:
#     여기에 도구를 하나 더 넣는 순간 "서버 도구를 다시 여는" 결정이 된다. 조용히
#     늘어나지 않도록 테스트로 목록 자체를 고정한다.
SERVER_TOOLS_KEPT_WITH_CLIENT_TOOLS: tuple[str, ...] = ("AnalyzeImage",)


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


def build_client_tools(
    specs: list[dict[str, Any]],
) -> tuple[list[ClientTool], list[str]]:
    """OpenAI `tools` 배열을 ClientTool 목록으로 바꾸고, **버린 것을 함께 돌려준다**.

    [왜 경고를 함께 돌려주나 — 2026-08-08]
      예전에는 형식이 어긋나거나 상한을 넘은 항목을 **조용히 건너뛰었다.** 요청 전체를
      실패시키지 않으려는 의도였는데, 그 대가가 컸다. 빠진 도구는 모델이 **보지도 부르지도
      못하고**, 잘렸다는 표식을 남길 자리도 없다. 도구 결과가 잘릴 때는 본문에
      `…[중략]…` 을 넣어 모델이 "여기 더 있었다"를 알 수 있지만, 도구 목록에는 그런
      자리가 없다 — 그냥 존재하지 않는 것이 된다.

      그래서 플러그인 개발자는 "왜 내 도구를 안 쓰지?" 만 보고 원인을 찾을 방법이 없었다.
      이 리포가 반복해서 문제 삼아 온 조용한 실패다(구조화 출력이 꺼져 있을 때 조용히
      무시하지 않고 400 으로 거부하는 것과 같은 이유).

      **자르는 것 자체는 유지한다** — 상한은 폭주 방지 백스톱으로 필요하다. 다만 소리를
      낸다. 판단은 호출부가 한다(경고만 실을지, 유효 0건이면 거부할지).

    Args:
        specs: `[{"type": "function", "function": {"name", "description", "parameters"}}]`

    Returns:
        (ClientTool 목록, 경고 문구 목록). 버린 것이 없으면 경고는 빈 목록.
        목록은 중복 이름 제거 + 최대 MAX_CLIENT_TOOLS 개.
    """
    tools: list[ClientTool] = []
    seen: set[str] = set()
    warnings: list[str] = []

    # 버린 것을 사유별로 모은다 — 개발자가 무엇을 고쳐야 하는지 바로 알 수 있게.
    malformed = 0  # dict 가 아니거나 이름이 없는 항목
    too_long: list[str] = []  # 이름이 상한을 넘은 항목
    duplicated: list[str] = []  # 같은 이름이 두 번 온 항목
    over_limit: list[str] = []  # 개수 상한을 넘어 못 들어간 항목

    for spec in specs or []:
        if not isinstance(spec, dict):
            malformed += 1
            continue
        fn = spec.get("function") if isinstance(spec.get("function"), dict) else spec
        name = str(fn.get("name") or "").strip()
        if not name:
            malformed += 1
            continue
        if len(name) > MAX_TOOL_NAME_CHARS:
            too_long.append(name[:MAX_TOOL_NAME_CHARS])
            continue
        if name in seen:
            duplicated.append(name)
            continue
        # 상한에 도달했으면 **중단하지 않고** 남은 이름을 모은다. 예전에는 break 로
        # 빠져나가 "몇 개가 더 있었는지"조차 알 수 없었다.
        if len(tools) >= MAX_CLIENT_TOOLS:
            over_limit.append(name)
            continue
        seen.add(name)
        tools.append(
            ClientTool(
                name=name,
                description=str(fn.get("description") or ""),
                input_schema=fn.get("parameters") or fn.get("input_schema") or {},
            )
        )

    if malformed:
        warnings.append(
            f"형식이 올바르지 않아 제외된 도구 {malformed}건 "
            "(객체가 아니거나 function.name 이 없음)."
        )
    if too_long:
        warnings.append(
            f"이름이 {MAX_TOOL_NAME_CHARS}자를 넘어 제외된 도구 "
            f"{len(too_long)}건: {', '.join(too_long[:5])}" + (" 외" if len(too_long) > 5 else "")
        )
    if duplicated:
        warnings.append(
            f"이름이 중복되어 제외된 도구 {len(duplicated)}건: "
            f"{', '.join(duplicated[:5])}" + (" 외" if len(duplicated) > 5 else "")
        )
    if over_limit:
        warnings.append(
            f"도구 개수 상한({MAX_CLIENT_TOOLS})을 넘어 제외된 도구 "
            f"{len(over_limit)}건: {', '.join(over_limit[:5])}"
            + (" 외" if len(over_limit) > 5 else "")
            + ". 이번 작업에 필요한 도구만 선언하면 프롬프트 여유도 늘어납니다."
        )

    return tools, warnings
