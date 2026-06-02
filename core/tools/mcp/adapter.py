"""
McpToolAdapter — 원격 MCP 도구 1개를 Nexus BaseTool로 래핑한다.

핵심 아이디어:
  원격 MCP 서버가 보고한 도구 1개를 BaseTool 서브클래스 1개로 감싸면,
  registry·executor·query_loop은 이 객체를 다른 24개 도구와 완전히 동일하게
  취급한다. MCP라는 사실을 전혀 모른다(P3 계약 — OpenAI tool_calls만 노출).

경계선:
  MCP JSON-RPC(tools/call)는 McpClient 내부에만 존재한다. 이 어댑터의 call()은
  client.call_tool()을 한 번 부를 뿐이며, 결과를 ToolResult로 정규화한다.

의존성 방향(P2): core/tools/mcp/ → core/tools/base.py (단방향, 역방향 금지).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 타입 체크 전용 import — 런타임 순환/불필요 의존을 피한다
if TYPE_CHECKING:
    from core.tools.mcp.client import McpClient

logger = logging.getLogger("nexus.tools.mcp.adapter")


class McpToolAdapter(BaseTool):
    """
    원격 MCP 서버의 tool 1개를 Nexus BaseTool로 래핑하는 어댑터.

    도구 이름 규칙: mcp__{server_name}__{remote_tool_name}
      → 이 접두사로 PermissionPipeline이 자동으로 ToolCategory.MCP로 분류한다
        (core/permission/pipeline.py — 이미 존재. 어댑터는 이름만 맞추면 정합).
    """

    def __init__(
        self,
        server_name: str,
        remote_tool_name: str,
        remote_schema: dict[str, Any],
        client: McpClient,
        description: str,
        is_read_only: bool = False,
    ):
        """
        Args:
            server_name: MCP 서버 이름 (예: "db", "diag", "docutil", "kowiki").
            remote_tool_name: 원격 서버가 보고한 원본 도구 이름.
            remote_schema: tools/list가 준 inputSchema(변환 없이 그대로 노출).
            client: 이 서버와 통신하는 McpClient.
            description: 모델에 보여줄 도구 설명.
            is_read_only: fail-closed 기본 False. 조회 전용 도구만 명시적으로
                          True로 완화한다(server.trust.read_only 기반).
        """
        self._server = server_name
        self._remote = remote_tool_name
        self._schema = remote_schema
        self._client = client
        self._description = description
        self._read_only = is_read_only

    # ═══ Identity ═══

    @property
    def name(self) -> str:
        """권한 식별 규칙과 정합: mcp__{server}__{tool}."""
        return f"mcp__{self._server}__{self._remote}"

    @property
    def description(self) -> str:
        return self._description

    @property
    def group(self) -> str:
        """UI 카테고리 — MCP 도구는 서버별로 묶어 보여준다."""
        return f"mcp:{self._server}"

    # ═══ Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        """MCP tools/list의 inputSchema를 그대로 노출(변환 없음)."""
        return self._schema

    # ═══ Behavior Flags ═══

    @property
    def is_read_only(self) -> bool:
        """fail-closed: 기본 False. 조회 전용 서버면 생성 시 True로 완화됨."""
        return self._read_only

    # ═══ Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        원격 서버가 준 input_schema로 입력을 기본 검증한다(도메인 검증 단계).

        목적(제품 견고성): 잘못된 입력이 LAN을 건너 원격 MCP 서버까지
        나가기 전에 어댑터 단에서 차단한다. 원격 호출 1회를 아끼고,
        에러 메시지를 모델에 더 빨리 돌려줘 자가 교정을 돕는다.

        검증 범위(과설계 금지 — 최소 견고성):
          1. required 필드 존재 검사 — schema.required의 키가 입력에 모두 있는지
          2. 최상위 type 검사 — schema.type이 "object"면 입력이 dict인지

        반환값 계약(BaseTool.validate_input): None=유효, str=에러 메시지(무효).

        주의: JSON Schema 전체(중첩 type, format, enum 등)를 구현하지 않는다.
        깊은 검증은 원격 서버의 책임이며, 여기서는 명백한 오류만 빠르게 거른다.
        """
        schema = self._schema or {}

        # 1) 최상위 type 검사 — object로 선언됐는데 dict가 아니면 무효
        #    (input_data는 항상 dict로 들어오지만, 방어적으로 명시 확인)
        schema_type = schema.get("type")
        if schema_type == "object" and not isinstance(input_data, dict):
            return f"MCP 도구 '{self.name}' 입력은 객체(object)여야 합니다."

        # 2) required 필드 존재 검사 — 누락 필드가 있으면 그 목록을 에러로
        required = schema.get("required")
        if isinstance(required, list):
            missing = [
                key for key in required if isinstance(key, str) and key not in (input_data or {})
            ]
            if missing:
                return f"MCP 도구 '{self.name}' 필수 입력 누락: {', '.join(missing)}"

        # 통과 — 유효
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        MCP 도구의 Layer 2(도구 고유 검사)는 항상 ALLOW를 반환하고,
        최종 권한 판단을 전적으로 5계층 파이프라인에 위임한다.

        제품 보안 정책(신뢰 경계는 "등록 단계"에서 통제):
          신뢰 통제는 McpConnectionManager.connect_and_register에서 수행한다.
          read-only로 신뢰된 서버(또는 allow_write=True로 명시 허용된 쓰기 서버)의
          도구만 애초에 등록되므로, 신뢰되지 않은 쓰기 도구는 도구 풀에 들어오지
          못한다(모델이 호출 자체 불가). 따라서 Layer 2에서 다시 신뢰를 따질
          필요가 없다.

        왜 ASK 조기 단락을 제거했는가(설계 결정):
          직전 강화는 쓰기 도구(is_read_only=False)를 여기서 ASK로 반환했는데,
          그 ASK가 권한 파이프라인을 조기에 단락시켜 PLAN 모드의 최종 보정
          (ASK→DENY)을 무력화하는 문제가 있었다. Layer 2가 read-only/쓰기에
          관계없이 ALLOW로 통과시키면, MODE_BEHAVIOR_MAP과 Layer 5 보정이
          정상 적용된다:
            - DEFAULT 모드  → ASK
            - PLAN 모드     → DENY (쓰기 차단 보정 정상 동작)
            - BYPASS 모드   → ALLOW
          즉 allow_write로 의도적으로 켠 쓰기 도구도 표준 권한 정책을 받는다.

        anti-pattern #11(권한 레이어 건너뛰기) 위반 아님:
          파이프라인을 우회/단축하지 않는다. Layer 2가 ALLOW를 내더라도
          Layer 3~5가 그대로 이어져 모드별 최종 결정을 내린다. 신뢰 통제는
          등록 단계로, 권한 판단은 표준 5계층으로 관심사를 분리한 것이다.

        details에 서버명·원격도구·신뢰상태를 실어 감사 추적(PermissionAuditEntry)을 돕는다.
        """
        # 감사 추적용 메타 — 어떤 서버의 어떤 신뢰 상태에서 내린 결정인지 기록
        audit_details = {
            "mcp_server": self._server,
            "remote_tool": self._remote,
            "is_read_only": self._read_only,
        }

        # Layer 2는 read-only든 (명시 허용된) 쓰기든 항상 ALLOW로 통과시킨다.
        # 최종 판단(모드별 ALLOW/ASK/DENY)은 전적으로 5계층 파이프라인에 위임한다.
        return PermissionResult(
            behavior=PermissionBehavior.ALLOW,
            message=(
                f"MCP 도구 (서버 '{self._server}', read_only={self._read_only}) — "
                f"최종 판단은 5계층 파이프라인에 위임"
            ),
            details=audit_details,
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        MCP tools/call을 1회 호출하고 결과를 ToolResult로 정규화한다.

        에러 처리(anti-pattern #8 — bare except 금지):
          연결 실패/타임아웃/LAN 검증 실패만 구체적으로 포착해
          tool_use_error로 래핑한다. 그 외 예상치 못한 예외는 상위 executor의
          13단계 파이프라인이 일괄 래핑하도록 의도적으로 전파한다.
        """
        try:
            result = await self._client.call_tool(
                self._remote,
                input_data,
                timeout=self.timeout_seconds,
            )
            return ToolResult.success(result)
        except (TimeoutError, ConnectionError, ValueError) as e:
            # 원격 MCP 서버 장애는 본류(다른 도구·턴)에 영향 주지 않도록 격리
            logger.warning("MCP 도구 '%s' 호출 실패: %s", self.name, e)
            return ToolResult.error(f"MCP 서버 '{self._server}' 호출 실패: {e}")
