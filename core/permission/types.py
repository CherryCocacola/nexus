"""
권한 시스템 타입 정의 — Permission System 전체에서 사용하는 타입.

Ch.8.2 사양서 기반으로 권한 관련 모든 타입을 한 파일에 정의한다.
  - PermissionMode: 7가지 권한 모드
  - PermissionBehavior: 3가지 권한 결정 결과 (base.py에서 재export)
  - PermissionRuleSource: 규칙의 출처 (CLI, 설정, 세션 등)
  - PermissionDecisionReason: 결정 이유 (감사 로그용)
  - ToolCategory: 도구 분류 (7가지)
  - MODE_BEHAVIOR_MAP: 모드×카테고리 → 동작 매핑 (7×7)
  - PermissionContext: 권한 판단에 필요한 불변 컨텍스트
  - PermissionDecision: AllowDecision | DenyDecision | AskDecision (Union)
  - PermissionAuditEntry: 감사 로그 엔트리

핵심 원칙 — fail-closed:
  모든 기본값은 가장 제한적인 설정(DENY 또는 ASK)이다.
  명시적으로 허용하지 않으면 차단된다.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# base.py에 이미 정의된 PermissionBehavior를 재export한다
from core.tools.base import PermissionBehavior

logger = logging.getLogger("nexus.permission")


# ─────────────────────────────────────────────
# PermissionMode — 7가지 권한 모드
# ─────────────────────────────────────────────
class PermissionMode(str, Enum):
    """
    권한 모드. CLI 플래그 또는 세션 설정으로 결정된다.
    각 모드는 도구 카테고리별로 다른 동작을 지정한다.
    """

    # 기본 모드: 읽기만 허용, 나머지는 사용자에게 물어본다
    DEFAULT = "DEFAULT"

    # 편집 허용: 파일 쓰기는 자동 허용, 위험 작업은 여전히 물어본다
    ACCEPT_EDITS = "ACCEPT_EDITS"

    # 권한 우회: 모든 도구를 허용한다 (개발/테스트용)
    BYPASS_PERMISSIONS = "BYPASS_PERMISSIONS"

    # 질문 금지: 물어보는 대신 거부한다 (CI/CD 파이프라인용)
    DONT_ASK = "DONT_ASK"

    # 계획 모드: 읽기와 계획만 허용, 쓰기는 모두 거부
    PLAN = "PLAN"

    # 자동 모드: 대부분 허용하되 위험한 것만 물어본다
    AUTO = "AUTO"

    # 버블 모드: 결정을 상위 에이전트로 전달한다 (서브 에이전트용)
    BUBBLE = "BUBBLE"


# ─────────────────────────────────────────────
# PermissionRuleSource — 규칙의 출처
# ─────────────────────────────────────────────
class PermissionRuleSource(str, Enum):
    """규칙이 어디서 왔는지 추적한다. 감사 로그와 디버깅에 사용."""

    CLI_FLAG = "cli_flag"  # --bypass-permissions 같은 CLI 플래그
    PROJECT_CONFIG = "project_config"  # .nexus/permission_rules.yaml
    USER_CONFIG = "user_config"  # ~/.nexus/config.yaml
    SESSION_GRANT = "session_grant"  # 세션 중 사용자가 허용한 것
    HOOK = "hook"  # Hook 시스템에서 결정한 것
    MODE_DEFAULT = "mode_default"  # MODE_BEHAVIOR_MAP 기본값
    SYSTEM = "system"  # 시스템 하드코딩 규칙


# ─────────────────────────────────────────────
# PermissionDecisionReason — 결정 이유
# ─────────────────────────────────────────────
class PermissionDecisionReason(str, Enum):
    """왜 이 결정이 내려졌는지 기록한다. 감사 로그에 포함."""

    # 허용 이유
    READ_ONLY_TOOL = "read_only_tool"  # 읽기 전용 도구는 항상 허용
    MODE_ALLOWS = "mode_allows"  # 현재 모드가 이 카테고리를 허용
    SESSION_GRANT = "session_grant"  # 세션 중 사용자가 허용
    HOOK_APPROVED = "hook_approved"  # Hook이 승인
    BYPASS_MODE = "bypass_mode"  # BYPASS 모드 (Layer 5)

    # 거부 이유
    DENY_RULE = "deny_rule"  # Layer 1 deny rule
    TOOL_CHECK_DENIED = "tool_check_denied"  # Layer 2 도구 자체 검사
    MODE_DENIES = "mode_denies"  # 현재 모드가 이 카테고리를 거부
    HOOK_BLOCKED = "hook_blocked"  # Hook이 차단
    PLAN_MODE_WRITE = "plan_mode_write"  # PLAN 모드에서 쓰기 시도 (Layer 5)


# ─────────────────────────────────────────────
# ToolCategory — 도구 분류 (7가지)
# ─────────────────────────────────────────────
class ToolCategory(str, Enum):
    """
    도구를 보안 카테고리로 분류한다.
    MODE_BEHAVIOR_MAP의 열(column)로 사용된다.
    """

    READONLY = "readonly"  # Read, Glob, Grep 등 읽기 전용
    FILE_WRITE = "file_write"  # Write, Edit 등 파일 변경
    BASH = "bash"  # Bash 명령어 실행
    DANGEROUS = "dangerous"  # 시스템에 큰 영향 (rm -rf 등)
    NETWORK = "network"  # 네트워크 접근 (에어갭에서 차단)
    AGENT = "agent"  # 서브 에이전트 생성
    MCP = "mcp"  # MCP 프로토콜 도구


# ─────────────────────────────────────────────
# MODE_BEHAVIOR_MAP — 7×7 매핑 테이블
# ─────────────────────────────────────────────
# 행(row) = PermissionMode, 열(column) = ToolCategory
# 값 = PermissionBehavior (ALLOW / DENY / ASK)
#
# 이 테이블이 Layer 3 (canUseTool)의 핵심이다.
# 모드와 카테고리의 조합으로 기본 동작을 결정한다.
MODE_BEHAVIOR_MAP: dict[PermissionMode, dict[ToolCategory, PermissionBehavior]] = {
    # DEFAULT: 읽기만 자동 허용, 나머지는 물어본다
    PermissionMode.DEFAULT: {
        ToolCategory.READONLY: PermissionBehavior.ALLOW,
        ToolCategory.FILE_WRITE: PermissionBehavior.ASK,
        ToolCategory.BASH: PermissionBehavior.ASK,
        ToolCategory.DANGEROUS: PermissionBehavior.ASK,
        ToolCategory.NETWORK: PermissionBehavior.DENY,
        ToolCategory.AGENT: PermissionBehavior.ASK,
        ToolCategory.MCP: PermissionBehavior.ASK,
    },
    # ACCEPT_EDITS: 파일 쓰기도 허용, 위험 작업은 물어본다
    PermissionMode.ACCEPT_EDITS: {
        ToolCategory.READONLY: PermissionBehavior.ALLOW,
        ToolCategory.FILE_WRITE: PermissionBehavior.ALLOW,
        ToolCategory.BASH: PermissionBehavior.ASK,
        ToolCategory.DANGEROUS: PermissionBehavior.ASK,
        ToolCategory.NETWORK: PermissionBehavior.DENY,
        ToolCategory.AGENT: PermissionBehavior.ASK,
        ToolCategory.MCP: PermissionBehavior.ASK,
    },
    # BYPASS_PERMISSIONS: 모든 것을 허용 (Layer 5에서 ASK→ALLOW 변환)
    PermissionMode.BYPASS_PERMISSIONS: {
        ToolCategory.READONLY: PermissionBehavior.ALLOW,
        ToolCategory.FILE_WRITE: PermissionBehavior.ALLOW,
        ToolCategory.BASH: PermissionBehavior.ALLOW,
        ToolCategory.DANGEROUS: PermissionBehavior.ALLOW,
        ToolCategory.NETWORK: PermissionBehavior.ALLOW,
        ToolCategory.AGENT: PermissionBehavior.ALLOW,
        ToolCategory.MCP: PermissionBehavior.ALLOW,
    },
    # DONT_ASK: 물어볼 것을 거부로 변환 (CI/CD용)
    PermissionMode.DONT_ASK: {
        ToolCategory.READONLY: PermissionBehavior.ALLOW,
        ToolCategory.FILE_WRITE: PermissionBehavior.DENY,
        ToolCategory.BASH: PermissionBehavior.DENY,
        ToolCategory.DANGEROUS: PermissionBehavior.DENY,
        ToolCategory.NETWORK: PermissionBehavior.DENY,
        ToolCategory.AGENT: PermissionBehavior.DENY,
        ToolCategory.MCP: PermissionBehavior.DENY,
    },
    # PLAN: 읽기만 허용, 모든 쓰기 거부
    PermissionMode.PLAN: {
        ToolCategory.READONLY: PermissionBehavior.ALLOW,
        ToolCategory.FILE_WRITE: PermissionBehavior.DENY,
        ToolCategory.BASH: PermissionBehavior.DENY,
        ToolCategory.DANGEROUS: PermissionBehavior.DENY,
        ToolCategory.NETWORK: PermissionBehavior.DENY,
        ToolCategory.AGENT: PermissionBehavior.DENY,
        ToolCategory.MCP: PermissionBehavior.DENY,
    },
    # AUTO: 대부분 허용, 위험한 것만 물어본다
    PermissionMode.AUTO: {
        ToolCategory.READONLY: PermissionBehavior.ALLOW,
        ToolCategory.FILE_WRITE: PermissionBehavior.ALLOW,
        ToolCategory.BASH: PermissionBehavior.ALLOW,
        ToolCategory.DANGEROUS: PermissionBehavior.ASK,
        ToolCategory.NETWORK: PermissionBehavior.DENY,
        ToolCategory.AGENT: PermissionBehavior.ALLOW,
        ToolCategory.MCP: PermissionBehavior.ALLOW,
    },
    # BUBBLE: 결정을 상위로 전달 (서브 에이전트용 — 모두 ASK로 설정)
    PermissionMode.BUBBLE: {
        ToolCategory.READONLY: PermissionBehavior.ALLOW,
        ToolCategory.FILE_WRITE: PermissionBehavior.ASK,
        ToolCategory.BASH: PermissionBehavior.ASK,
        ToolCategory.DANGEROUS: PermissionBehavior.ASK,
        ToolCategory.NETWORK: PermissionBehavior.DENY,
        ToolCategory.AGENT: PermissionBehavior.ASK,
        ToolCategory.MCP: PermissionBehavior.ASK,
    },
}


# ─────────────────────────────────────────────
# PermissionRule — 권한 규칙 모델
# ─────────────────────────────────────────────
class PermissionRule(BaseModel):
    """
    하나의 권한 규칙.
    YAML 설정이나 CLI 플래그에서 로드되어 파이프라인에 전달된다.
    """

    source: PermissionRuleSource  # 이 규칙이 어디서 왔는지
    behavior: PermissionBehavior  # ALLOW / DENY / ASK
    tool_name: str | None = None  # 특정 도구에만 적용 (None이면 전체)
    tool_category: ToolCategory | None = None  # 특정 카테고리에만 적용
    rule_content: str = ""  # 규칙 설명 (감사 로그용)
    path_pattern: str | None = None  # 파일 경로 패턴 (glob)
    command_pattern: str | None = None  # 명령어 패턴 (regex)

    def matches_tool(self, tool_name: str) -> bool:
        """이 규칙이 주어진 도구에 적용되는지 확인한다."""
        # tool_name이 None이거나 "*"이면 모든 도구에 적용
        if self.tool_name is None or self.tool_name == "*":
            return True
        return self.tool_name.lower() == tool_name.lower()

    def matches_input(self, tool_input: dict[str, Any]) -> bool:
        """이 규칙이 주어진 입력에 적용되는지 확인한다."""
        import fnmatch
        import re

        # 경로 패턴 검사 (파일 경로가 있는 경우)
        if self.path_pattern is not None:
            file_path = tool_input.get("file_path") or tool_input.get("path", "")
            if file_path and not fnmatch.fnmatch(file_path, self.path_pattern):
                return False

        # 명령어 패턴 검사 (bash 명령어가 있는 경우)
        if self.command_pattern is not None:
            command = tool_input.get("command", "")
            if command and not re.search(self.command_pattern, command):
                return False

        return True


# ─────────────────────────────────────────────
# PermissionDecision — 권한 결정 (3종류 Union)
# ─────────────────────────────────────────────
class AllowDecision(BaseModel):
    """허용 결정. 도구 실행을 진행한다."""

    type: Literal["allow"] = "allow"
    behavior: PermissionBehavior = PermissionBehavior.ALLOW
    reason: PermissionDecisionReason
    source: PermissionRuleSource = PermissionRuleSource.MODE_DEFAULT
    message: str = ""


class DenyDecision(BaseModel):
    """거부 결정. 도구 실행을 차단하고 에러를 반환한다."""

    type: Literal["deny"] = "deny"
    behavior: PermissionBehavior = PermissionBehavior.DENY
    reason: PermissionDecisionReason
    source: PermissionRuleSource = PermissionRuleSource.MODE_DEFAULT
    message: str = ""


class AskDecision(BaseModel):
    """사용자 확인 결정. CLI에서 사용자에게 물어본다."""

    type: Literal["ask"] = "ask"
    behavior: PermissionBehavior = PermissionBehavior.ASK
    reason: PermissionDecisionReason
    source: PermissionRuleSource = PermissionRuleSource.MODE_DEFAULT
    message: str = ""
    tool_name: str = ""  # 어떤 도구에 대한 질문인지
    tool_input_summary: str = ""  # 입력 요약 (사용자 표시용)


# 3가지 결정의 Union 타입
PermissionDecision = AllowDecision | DenyDecision | AskDecision


# ─────────────────────────────────────────────
# PermissionContext — 불변 권한 컨텍스트
# ─────────────────────────────────────────────
class PermissionContext(BaseModel):
    """
    권한 판단에 필요한 컨텍스트 (frozen).
    파이프라인 실행 중 변경되지 않는다.
    새 값이 필요하면 with_*() 메서드로 새 인스턴스를 생성한다.
    """

    model_config = {"frozen": True}

    mode: PermissionMode = PermissionMode.DEFAULT
    working_directory: str = "."
    rules: tuple[PermissionRule, ...] = ()  # frozen이라 tuple 사용
    session_grants: tuple[str, ...] = ()  # 세션 중 허용된 도구 이름 목록
    session_id: str = ""
    agent_id: str | None = None

    def with_session_grant(self, tool_name: str) -> PermissionContext:
        """새 세션 허용을 추가한 새 컨텍스트를 반환한다."""
        if tool_name in self.session_grants:
            return self
        return self.model_copy(update={"session_grants": (*self.session_grants, tool_name)})

    def with_mode(self, mode: PermissionMode) -> PermissionContext:
        """모드를 변경한 새 컨텍스트를 반환한다."""
        return self.model_copy(update={"mode": mode})

    def has_session_grant(self, tool_name: str) -> bool:
        """이 도구가 세션 중 허용되었는지 확인한다."""
        return tool_name in self.session_grants


# ─────────────────────────────────────────────
# PermissionAuditEntry — 감사 로그 엔트리
# ─────────────────────────────────────────────
class PermissionAuditEntry(BaseModel):
    """
    하나의 권한 결정을 기록하는 감사 로그 엔트리.
    JSONL 형식으로 직렬화되어 로그 파일에 기록된다.
    """

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str = ""
    agent_id: str | None = None
    tool_name: str = ""
    tool_category: str = ""  # ToolCategory 값
    tool_input_summary: str = ""  # 입력 요약 (민감 정보 제거)
    decision: str = ""  # "allow" / "deny" / "ask"
    reason: str = ""  # PermissionDecisionReason 값
    source: str = ""  # PermissionRuleSource 값
    mode: str = ""  # PermissionMode 값
    message: str = ""  # 추가 설명
    layers_checked: list[str] = Field(default_factory=list)  # 어떤 레이어를 거쳤는지
