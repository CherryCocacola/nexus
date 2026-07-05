"""
McpToolAdapter — 원격 MCP 도구 1개를 Nexus BaseTool로 래핑하는 어댑터 모듈.

이 파일이 하는 일 (한눈에):
  MCP(Model Context Protocol) 서버는 자기가 제공하는 "원격 도구" 목록을
  tools/list로 알려준다. 이 모듈은 그 원격 도구 하나하나를 Nexus 내부의
  표준 도구 인터페이스(BaseTool)로 옷 입혀(adapter 패턴) 주는 역할을 한다.

왜 필요한가 (핵심 아이디어):
  원격 MCP 도구 1개를 BaseTool 서브클래스 1개로 감싸 두면,
  registry(도구 등록)·executor(실행)·query_loop(에이전트 루프)은 이 객체를
  나머지 로컬 24개 도구와 "완전히 똑같이" 다룰 수 있다. 즉 상위 계층은
  이게 원격 MCP 도구라는 사실을 전혀 몰라도 된다.
  (P3 표준 계약 — 내부에는 OpenAI tool_calls 형식만 노출한다.)

경계선 (책임 분리):
  MCP의 실제 통신 프로토콜인 JSON-RPC(tools/call)는 McpClient 내부에만
  숨어 있다. 이 어댑터의 call()이 하는 일은 client.call_tool()을 딱 한 번
  호출하고, 그 응답을 Nexus 표준 결과 타입인 ToolResult로 정규화하는 것뿐이다.
  (통신은 McpClient가, 표준화·권한·검증 껍데기는 이 어댑터가 담당.)

주요 구성:
  - McpToolAdapter: 원격 도구 1개 = 이 클래스 인스턴스 1개. BaseTool 상속.

의존성 방향 (P2 규칙): core/tools/mcp/ → core/tools/base.py 로만 흐른다.
  (단방향만 허용 — base.py가 이 모듈을 거꾸로 import 하면 안 된다.)

작성자: 이현수 / 작성일: 2026-07-05
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
    원격 MCP 서버의 도구(tool) 1개를 Nexus BaseTool로 래핑하는 어댑터 클래스.

    한 서버가 N개의 원격 도구를 보고하면, 이 클래스 인스턴스도 N개 만들어진다
    (도구 1개 ↔ 어댑터 1개). 인스턴스는 자기가 감쌀 서버 이름·원격 도구 이름·
    원격 스키마·통신용 McpClient를 생성 시점에 주입받아 보관한다.

    BaseTool의 생애주기 규약(아래 순서로 상위 executor가 호출):
      validate_input() → check_permissions() → call() → (결과 매핑)
    이 클래스는 그중 세 메서드를 오버라이드해 MCP에 맞게 구현한다.

    도구 이름 규칙: mcp__{server_name}__{remote_tool_name}
      이 "mcp__" 접두사만 맞춰 두면 PermissionPipeline이 이름만 보고 자동으로
      ToolCategory.MCP로 분류한다(core/permission/pipeline.py에 이미 구현됨).
      즉 어댑터는 별도 등록 로직 없이 name 규칙만 지키면 권한 분류에 정합된다.
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
        어댑터 생성자 — 원격 도구 1개를 감싸는 데 필요한 정보를 모두 주입받아
        내부 필드에 보관한다(이후 프로퍼티/메서드가 이 값들을 참조).

        보통 McpConnectionManager가 서버에 연결한 뒤 tools/list 응답을 돌면서
        도구마다 이 생성자를 호출해 어댑터를 하나씩 만든다.

        Args:
            server_name: MCP 서버 이름 (예: "db", "diag", "docutil", "kowiki").
                         name 프로퍼티와 group 분류의 기준이 된다.
            remote_tool_name: 원격 서버가 보고한 원본 도구 이름. call() 시
                              이 이름 그대로 client.call_tool()에 넘긴다.
            remote_schema: tools/list가 준 inputSchema. 변환 없이 그대로 노출하며
                           validate_input()의 검증 기준으로도 쓰인다.
            client: 이 서버와 실제로 통신하는 McpClient(JSON-RPC 담당).
            description: 모델(LLM)에 보여줄 도구 설명 문자열.
            is_read_only: fail-closed 원칙상 기본 False(=쓰기 도구로 간주).
                          조회 전용 도구만 server.trust.read_only 값을 근거로
                          명시적으로 True로 완화한다.
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
        """
        도구 고유 이름. 권한 파이프라인의 MCP 식별 규칙과 반드시 정합해야 한다:
        mcp__{server}__{tool}. 이 이름은 registry 등록 키이자 모델이 호출할 때
        쓰는 이름이며, "mcp__" 접두사가 곧 ToolCategory.MCP 분류의 트리거다.
        """
        return f"mcp__{self._server}__{self._remote}"

    @property
    def description(self) -> str:
        """모델에게 보여줄 도구 설명. 생성 시 받은 값을 그대로 돌려준다."""
        return self._description

    @property
    def group(self) -> str:
        """
        UI/도구 목록에서의 묶음(카테고리) 키. MCP 도구는 서버별로 묶어 보여주기
        위해 "mcp:{서버명}" 형태를 쓴다(예: 같은 db 서버 도구들은 한 그룹).
        """
        return f"mcp:{self._server}"

    # ═══ Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        """
        도구 입력 JSON Schema. 원격 MCP 서버가 tools/list로 준 inputSchema를
        가공 없이 그대로 노출한다. 이 스키마가 모델에게 전달되어 어떤 인자를
        채워야 하는지 알려주고, validate_input()의 검증 기준으로도 재사용된다.
        """
        return self._schema

    # ═══ Behavior Flags ═══

    @property
    def is_read_only(self) -> bool:
        """
        조회 전용 여부. fail-closed 원칙상 기본은 False(쓰기 가능으로 간주)이며,
        신뢰된 조회 전용 서버의 도구만 생성 시 True로 완화된다. 이 값은 권한
        판단(모드별 ALLOW/ASK/DENY)과 동시 실행 안전성 판정 등에 영향을 준다.
        """
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
        실제 도구 실행 담당. 원격 MCP 서버의 tools/call을 1회 호출하고,
        그 응답을 Nexus 표준 결과 타입 ToolResult로 정규화해 돌려준다.
        (권한·검증을 모두 통과한 뒤 상위 executor가 이 메서드를 호출한다.)

        흐름:
          1. McpClient.call_tool()에 원격 도구 이름·입력·타임아웃을 넘겨 호출.
          2. 성공하면 ToolResult.success로 감싸 반환.
          3. 실패하면 아래 except에서 ToolResult.error로 변환.

        에러 처리(anti-pattern #8 — bare except 금지):
          연결 실패/타임아웃/LAN 검증 실패처럼 "예상 가능한" 예외만 구체 타입으로
          포착해 tool_use_error로 래핑한다. 그 밖의 예상치 못한 예외는 삼키지 않고
          상위 executor의 13단계 파이프라인이 일괄 래핑하도록 일부러 전파시킨다.
        """
        try:
            # 원격 서버에 실제 실행 요청. 원본 도구 이름과 입력을 그대로 전달하고,
            # BaseTool이 제공하는 타임아웃(초)을 함께 넘겨 무한 대기를 방지한다.
            result = await self._client.call_tool(
                self._remote,
                input_data,
                timeout=self.timeout_seconds,
            )
            # 정상 응답 — 표준 성공 결과로 감싼다.
            return ToolResult.success(result)
        except (TimeoutError, ConnectionError, ValueError) as e:
            # 원격 MCP 서버 장애(타임아웃/연결 끊김/LAN 검증 실패 등)는
            # 본류(다른 도구·다음 턴)에 영향을 주지 않도록 여기서 격리한다.
            # 경고 로그만 남기고, 모델이 이해할 에러 결과로 변환해 반환한다.
            logger.warning("MCP 도구 '%s' 호출 실패: %s", self.name, e)
            return ToolResult.error(f"MCP 서버 '{self._server}' 호출 실패: {e}")
