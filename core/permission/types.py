"""
권한 시스템 타입 정의 — Permission System 전체에서 사용하는 공용 타입 모음.

[이 파일이 하는 일]
Nexus의 5계층 권한 파이프라인(Layer 1~5)이 서로 주고받는 데이터 구조를
한 곳에 모아 정의한다. 즉 "권한 판단에 쓰이는 단어장"에 해당하는 파일이다.
파이프라인 로직 자체(pipeline.py, layer2_* 등)는 여기 정의된 타입을 import 해
사용하므로, 이 파일은 권한 모듈의 최하위 기반(의존성 뿌리)에 위치한다.

[사양서 근거]
사양서 Ch.8.2를 기반으로 한다. 7가지 권한 모드, 7가지 도구 카테고리,
그리고 그 둘을 교차한 7×7 동작 매핑 테이블(MODE_BEHAVIOR_MAP)이 핵심이다.

[정의하는 주요 타입]
  - PermissionMode         : 7가지 권한 모드 (DEFAULT, PLAN, BYPASS 등)
  - PermissionBehavior     : 3가지 결정 결과 (ALLOW/DENY/ASK). base.py에서 재export
  - PermissionRuleSource   : 규칙의 출처 (CLI 플래그, 설정, 세션 등)
  - PermissionDecisionReason : 결정 이유 코드 (감사 로그·디버깅용)
  - ToolCategory           : 도구 보안 분류 (7가지)
  - MODE_BEHAVIOR_MAP      : 모드 × 카테고리 → 기본 동작 매핑 (7×7 테이블)
  - PermissionRule         : YAML/CLI에서 로드되는 개별 권한 규칙 모델
  - PermissionDecision     : AllowDecision | DenyDecision | AskDecision (Union)
  - PermissionContext      : 권한 판단에 필요한 불변(frozen) 컨텍스트
  - PermissionAuditEntry   : 감사 로그 한 줄(엔트리) 모델

[의존 관계]
  core.tools.base 의 PermissionBehavior 를 재export 한다(단일 출처 유지).
  이 파일은 다른 권한 모듈을 import 하지 않는다 — 순환 의존을 피하기 위함.

[핵심 원칙 — fail-closed(안전측 차단)]
  모든 기본값은 가장 제한적인 설정(DENY 또는 ASK)이다.
  즉 "명시적으로 허용하지 않으면 무조건 막힌다"가 이 시스템의 대원칙이다.
  새 모드나 카테고리를 추가할 때도 이 원칙을 깨지 않도록 주의한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# PermissionBehavior(ALLOW/DENY/ASK)는 도구 계층(base.py)에서 이미 정의돼 있다.
# 권한 코드가 이 파일만 import 하면 되도록 여기서 그대로 재export 한다.
# (같은 개념을 두 곳에서 중복 정의하면 값이 어긋날 수 있어 단일 출처를 유지)
from core.tools.base import PermissionBehavior

# 권한 관련 로그는 모두 이 로거로 남긴다. 규칙에 따라 이름은 "nexus.{module}" 형식.
logger = logging.getLogger("nexus.permission")


# ─────────────────────────────────────────────
# PermissionMode — 7가지 권한 모드
# ─────────────────────────────────────────────
class PermissionMode(str, Enum):
    """
    권한 모드 — "지금 세션이 얼마나 관대한가"를 나타내는 큰 스위치.

    CLI 플래그(예: --bypass-permissions)나 세션 설정으로 하나가 선택된다.
    선택된 모드는 아래 MODE_BEHAVIOR_MAP에서 도구 카테고리별 기본 동작
    (ALLOW/DENY/ASK)을 찾는 "행(row) 키"로 쓰인다.

    str을 함께 상속하므로 이 enum 멤버는 문자열처럼도 다룰 수 있다
    (예: JSON 직렬화 시 "DEFAULT" 문자열로 저장됨).
    """

    # 기본 모드: 읽기만 자동 허용하고, 쓰기·실행 등은 사용자에게 물어본다.
    # 대화형 CLI에서 가장 흔히 쓰이는 안전한 기본값.
    DEFAULT = "DEFAULT"

    # 편집 허용 모드: 파일 쓰기(Write/Edit)는 자동 허용, 위험 작업은 여전히 물어본다.
    # 코드를 연속 수정하는 작업에서 매번 확인을 안 받으려 할 때 사용.
    ACCEPT_EDITS = "ACCEPT_EDITS"

    # 권한 우회 모드: 모든 도구를 무조건 허용한다. (개발/테스트 전용 — 위험)
    # 실제로는 Layer 5에서 ASK 결정을 ALLOW로 뒤집는 방식으로 동작한다.
    BYPASS_PERMISSIONS = "BYPASS_PERMISSIONS"

    # 질문 금지 모드: 사용자에게 물어볼 수 없는 환경(CI/CD 등)에서 쓴다.
    # "물어봐야 하는(ASK)" 상황을 전부 거부(DENY)로 바꿔 무인 실행을 안전하게 만든다.
    DONT_ASK = "DONT_ASK"

    # 계획 모드: 읽기만 허용하고 모든 쓰기·실행은 거부한다.
    # 코드를 건드리지 않고 조사·계획만 세우게 할 때 사용.
    PLAN = "PLAN"

    # 자동 모드: 대부분 자동 허용하되, 진짜 위험한(DANGEROUS) 것만 물어본다.
    # 신뢰할 수 있는 반자율 실행에 적합.
    AUTO = "AUTO"

    # 버블 모드: 스스로 결정하지 않고 판단을 상위 에이전트로 "떠올려(bubble up)" 보낸다.
    # 서브 에이전트가 자체 권한을 갖지 않도록 대부분 ASK로 설정된다.
    BUBBLE = "BUBBLE"


# ─────────────────────────────────────────────
# PermissionRuleSource — 규칙의 출처
# ─────────────────────────────────────────────
class PermissionRuleSource(str, Enum):
    """
    규칙의 출처 — "이 결정을 만든 규칙이 어디서 왔는가"를 추적한다.

    권한 판단 결과를 감사 로그에 남길 때 함께 기록하여, 나중에
    "왜 이게 허용/거부됐지?"를 역추적(디버깅)할 수 있게 한다.
    우선순위가 높은 순서대로 대략: CLI 플래그 > 설정 파일 > 세션 > 기본값.
    """

    CLI_FLAG = "cli_flag"  # 실행 시 넘긴 CLI 플래그 (예: --bypass-permissions)
    PROJECT_CONFIG = "project_config"  # 프로젝트 설정 파일 .nexus/permission_rules.yaml
    USER_CONFIG = "user_config"  # 사용자 홈의 전역 설정 ~/.nexus/config.yaml
    SESSION_GRANT = "session_grant"  # 이번 세션 중 사용자가 직접 허용해 준 것
    HOOK = "hook"  # Hook 시스템(Layer 4)이 내린 결정
    MODE_DEFAULT = "mode_default"  # 아무 규칙도 안 맞아 MODE_BEHAVIOR_MAP 기본값 적용
    SYSTEM = "system"  # 코드에 하드코딩된 시스템 규칙(최후의 안전장치)


# ─────────────────────────────────────────────
# PermissionDecisionReason — 결정 이유
# ─────────────────────────────────────────────
class PermissionDecisionReason(str, Enum):
    """
    결정 이유 코드 — "왜 허용/거부/질문으로 정해졌는가"를 한 단어로 표현한다.

    출처(PermissionRuleSource)가 "어디서"라면, 이 값은 "왜"에 해당한다.
    감사 로그와 사용자 안내 메시지에 포함되어 결정 근거를 명확히 한다.
    아래는 이유를 허용/거부 두 묶음으로 나눠 정리한 것이다.
    """

    # ── 허용(ALLOW) 계열 이유 ──
    READ_ONLY_TOOL = "read_only_tool"  # 읽기 전용 도구라 어느 모드에서든 항상 허용
    MODE_ALLOWS = "mode_allows"  # 현재 모드가 이 카테고리를 기본 허용
    SESSION_GRANT = "session_grant"  # 이번 세션에서 사용자가 미리 허용해 둠
    HOOK_APPROVED = "hook_approved"  # Hook(Layer 4)이 명시적으로 승인
    BYPASS_MODE = "bypass_mode"  # BYPASS 모드가 ASK를 ALLOW로 승격 (Layer 5)

    # ── 거부(DENY) 계열 이유 ──
    DENY_RULE = "deny_rule"  # Layer 1 차단 규칙에 걸림 (도구 자체가 제거됨)
    TOOL_CHECK_DENIED = "tool_check_denied"  # Layer 2 경로/명령어 검사에서 거부
    MODE_DENIES = "mode_denies"  # 현재 모드가 이 카테고리를 기본 거부
    HOOK_BLOCKED = "hook_blocked"  # Hook(Layer 4)이 명시적으로 차단
    PLAN_MODE_WRITE = "plan_mode_write"  # PLAN 모드에서 쓰기 시도라 거부 (Layer 5)


# ─────────────────────────────────────────────
# ToolCategory — 도구 분류 (7가지)
# ─────────────────────────────────────────────
class ToolCategory(str, Enum):
    """
    도구 보안 카테고리 — 각 도구가 "얼마나 위험한 부류인가"를 나타낸다.

    모든 도구는 실행 전에 이 7가지 중 하나로 분류되며, 그 카테고리가
    MODE_BEHAVIOR_MAP에서 동작을 찾는 "열(column) 키"로 쓰인다.
    (행=모드, 열=카테고리 → 교차점에서 ALLOW/DENY/ASK 결정)
    """

    READONLY = "readonly"  # 읽기 전용: Read, Glob, Grep 등 (상태를 안 바꿈)
    FILE_WRITE = "file_write"  # 파일 변경: Write, Edit 등
    BASH = "bash"  # 셸 명령 실행: 일반 Bash 명령어
    DANGEROUS = "dangerous"  # 파괴적 작업: rm -rf 등 시스템에 큰 영향
    NETWORK = "network"  # 네트워크 접근: 에어갭 원칙상 기본 차단
    AGENT = "agent"  # 서브 에이전트 생성/실행
    MCP = "mcp"  # MCP 프로토콜을 통해 노출된 외부 도구


# ─────────────────────────────────────────────
# MODE_BEHAVIOR_MAP — 7×7 매핑 테이블
# ─────────────────────────────────────────────
# [읽는 법] 2차원 표(dict of dict)다.
#   행(row)    = PermissionMode  (현재 권한 모드)
#   열(column) = ToolCategory    (도구의 보안 분류)
#   값(value)  = PermissionBehavior (ALLOW / DENY / ASK)
# 즉 MODE_BEHAVIOR_MAP[mode][category] 로 한 번에 기본 동작을 얻는다.
#
# [역할] 이 테이블이 Layer 3(canUseTool 단계)의 핵심 근거다.
#   앞선 Layer 1(차단 규칙)·Layer 2(경로/명령 검사)를 통과한 뒤,
#   "이 모드에서 이 부류의 도구는 기본적으로 어떻게 다룰까?"를 여기서 정한다.
#
# [불변식] NETWORK 열은 에어갭 원칙상 사실상 항상 DENY이며(BYPASS만 예외),
#   READONLY 열은 항상 ALLOW다. 새 모드를 추가할 때도 이 성질을 유지한다.
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
    하나의 권한 규칙 — "어떤 도구/입력에 어떤 동작을 적용할지" 한 줄.

    YAML 설정 파일이나 CLI 플래그에서 여러 개가 로드되어 파이프라인에 전달되고,
    파이프라인은 각 규칙에 대해 matches_tool()·matches_input()으로
    "이 호출에 이 규칙이 해당되는가"를 판정한다.
    규칙은 도구 이름, 카테고리, 경로 패턴, 명령어 패턴으로 좁힐 수 있으며
    지정하지 않은(None) 조건은 "제한 없음(모두 매칭)"으로 취급된다.
    """

    source: PermissionRuleSource  # 이 규칙의 출처 (감사·우선순위 판단용)
    behavior: PermissionBehavior  # 매칭 시 적용할 동작 — ALLOW / DENY / ASK
    tool_name: str | None = None  # 특정 도구에만 적용 (None 또는 "*"이면 전체)
    tool_category: ToolCategory | None = None  # 특정 카테고리에만 적용
    rule_content: str = ""  # 사람이 읽을 규칙 설명 (감사 로그에 남김)
    path_pattern: str | None = None  # 파일 경로 매칭 패턴 (glob 문법, fnmatch)
    command_pattern: str | None = None  # 명령어 매칭 패턴 (정규식, re.search)

    def matches_tool(self, tool_name: str) -> bool:
        """
        이 규칙이 주어진 도구 이름에 적용되는지 확인한다.

        와일드카드("*")나 미지정(None)이면 모든 도구에 해당된다.
        그 외에는 대소문자를 무시하고 이름이 정확히 같은지 비교한다.
        """
        # tool_name이 None이거나 "*"이면 도구를 가리지 않고 모두 적용된다.
        if self.tool_name is None or self.tool_name == "*":
            return True
        # 대소문자 차이로 규칙이 빗나가지 않도록 양쪽을 소문자로 맞춰 비교한다.
        return self.tool_name.lower() == tool_name.lower()

    def matches_input(self, tool_input: dict[str, Any]) -> bool:
        """
        이 규칙이 주어진 도구 입력(인자)에 적용되는지 확인한다.

        경로 패턴과 명령어 패턴 두 조건을 검사한다. 지정된 패턴이 있는데
        실제 입력이 그 패턴에 맞지 않으면 False(=이 규칙은 해당 없음)를 반환한다.
        패턴이 지정되지 않았거나 입력에 해당 값이 없으면 그 조건은 건너뛴다.
        결과적으로 "지정한 모든 패턴을 통과할 때만" True가 된다.
        """
        # 표준 라이브러리라 상단 import 없이 필요할 때 지역 import 한다(순환/비용 최소화).
        import fnmatch
        import re

        # (1) 경로 패턴 검사 — 파일을 다루는 도구일 때만 의미가 있다.
        if self.path_pattern is not None:
            # file_path 키를 우선 보고, 없으면 path 키를 본다(도구마다 키 이름이 다름).
            file_path = tool_input.get("file_path") or tool_input.get("path", "")
            # 경로가 있는데 glob 패턴과 안 맞으면 이 규칙은 해당되지 않는다.
            if file_path and not fnmatch.fnmatch(file_path, self.path_pattern):
                return False

        # (2) 명령어 패턴 검사 — bash 같은 명령 실행 도구일 때만 의미가 있다.
        if self.command_pattern is not None:
            command = tool_input.get("command", "")
            # 명령어가 있는데 정규식에 걸리지 않으면 이 규칙은 해당되지 않는다.
            if command and not re.search(self.command_pattern, command):
                return False

        # 위 두 관문을 모두 통과 → 이 규칙이 이 입력에 적용된다.
        return True


# ─────────────────────────────────────────────
# PermissionDecision — 권한 결정 (3종류 Union)
# ─────────────────────────────────────────────
class AllowDecision(BaseModel):
    """
    허용 결정 — 파이프라인이 "이 도구를 실행해도 된다"고 판단한 결과.

    type 필드가 "allow" 리터럴로 고정돼 있어, 세 결정 타입을 하나로 묶은
    PermissionDecision Union에서 이 값으로 종류를 구분(discriminated union)한다.
    reason/source는 왜·어디서 이 허용이 나왔는지를 감사 로그에 남기기 위한 값.
    """

    type: Literal["allow"] = "allow"  # Union 판별용 태그 (항상 "allow")
    behavior: PermissionBehavior = PermissionBehavior.ALLOW  # 결과 동작 = 허용
    reason: PermissionDecisionReason  # 허용된 근거 이유 (필수)
    source: PermissionRuleSource = PermissionRuleSource.MODE_DEFAULT  # 결정의 출처
    message: str = ""  # 사람이 읽을 부가 설명(선택)


class DenyDecision(BaseModel):
    """
    거부 결정 — "이 도구는 실행할 수 없다"고 판단한 결과.

    이 결정을 받으면 도구는 실행되지 않고, 대신 에러(tool_use_error)로 처리된다.
    message에 사용자/모델에게 보여줄 거부 사유를 담는 것이 좋다.
    """

    type: Literal["deny"] = "deny"  # Union 판별용 태그 (항상 "deny")
    behavior: PermissionBehavior = PermissionBehavior.DENY  # 결과 동작 = 거부
    reason: PermissionDecisionReason  # 거부된 근거 이유 (필수)
    source: PermissionRuleSource = PermissionRuleSource.MODE_DEFAULT  # 결정의 출처
    message: str = ""  # 거부 사유 설명(선택)


class AskDecision(BaseModel):
    """
    질문 결정 — "실행 전에 사용자에게 물어봐야 한다"는 보류 상태.

    CLI가 이 결정을 받으면 사용자에게 확인을 요청하고, 응답에 따라
    최종적으로 허용/거부로 이어진다. 사용자에게 무엇을 물어볼지 보여주기 위해
    tool_name과 입력 요약을 함께 담는다.
    """

    type: Literal["ask"] = "ask"  # Union 판별용 태그 (항상 "ask")
    behavior: PermissionBehavior = PermissionBehavior.ASK  # 결과 동작 = 질문
    reason: PermissionDecisionReason  # 질문이 필요한 근거 이유 (필수)
    source: PermissionRuleSource = PermissionRuleSource.MODE_DEFAULT  # 결정의 출처
    message: str = ""  # 확인 프롬프트에 덧붙일 설명(선택)
    tool_name: str = ""  # 어떤 도구에 대한 질문인지 (사용자 표시용)
    tool_input_summary: str = ""  # 입력 인자 요약 (사용자 표시용, 민감정보 제외)


# 위 세 결정을 하나로 묶은 Union 타입. type 필드("allow"/"deny"/"ask")로
# 어느 종류인지 구분한다. 파이프라인 함수들의 반환 타입으로 사용된다.
PermissionDecision = AllowDecision | DenyDecision | AskDecision


# ─────────────────────────────────────────────
# PermissionContext — 불변 권한 컨텍스트
# ─────────────────────────────────────────────
class PermissionContext(BaseModel):
    """
    권한 판단에 필요한 불변(frozen) 컨텍스트 — "지금 어떤 상황인지"의 스냅샷.

    현재 모드, 작업 디렉터리, 적용할 규칙 목록, 세션 중 허용 내역 등
    권한 결정에 필요한 배경 정보를 한 덩어리로 담아 파이프라인에 넘긴다.
    frozen=True라 한 번 만들면 값을 바꿀 수 없다(변경 실수를 방지).
    값을 바꾸려면 아래 with_*() 메서드로 "일부만 바뀐 새 인스턴스"를 만든다.
    """

    model_config = {"frozen": True}  # 불변 객체로 고정 — 필드 재할당 불가

    mode: PermissionMode = PermissionMode.DEFAULT  # 현재 권한 모드
    working_directory: str = "."  # 기준 작업 디렉터리(상대경로 해석 기준)
    rules: tuple[PermissionRule, ...] = ()  # 적용 규칙 목록 (frozen이라 list 대신 tuple)
    session_grants: tuple[str, ...] = ()  # 이번 세션에서 사용자가 허용한 도구 이름들
    session_id: str = ""  # 세션 식별자 (감사 로그 연결용)
    agent_id: str | None = None  # 서브 에이전트 식별자 (없으면 최상위 세션)

    def with_session_grant(self, tool_name: str) -> PermissionContext:
        """
        주어진 도구를 세션 허용 목록에 추가한 새 컨텍스트를 반환한다.

        frozen 객체라 직접 수정하지 못하므로 model_copy로 복제본을 만든다.
        이미 허용돼 있으면 불필요한 복제를 피하려고 self를 그대로 돌려준다.
        """
        # 이미 허용 목록에 있으면 새 객체를 만들 필요가 없다.
        if tool_name in self.session_grants:
            return self
        # 기존 튜플 뒤에 새 도구를 붙인 튜플로 교체한 복제본을 만든다.
        return self.model_copy(update={"session_grants": (*self.session_grants, tool_name)})

    def with_mode(self, mode: PermissionMode) -> PermissionContext:
        """모드만 바꾼 새 컨텍스트를 반환한다(나머지 필드는 그대로 복제)."""
        return self.model_copy(update={"mode": mode})

    def has_session_grant(self, tool_name: str) -> bool:
        """이 도구가 이번 세션에서 이미 허용되었는지 여부를 알려준다."""
        return tool_name in self.session_grants


# ─────────────────────────────────────────────
# PermissionAuditEntry — 감사 로그 엔트리
# ─────────────────────────────────────────────
class PermissionAuditEntry(BaseModel):
    """
    감사 로그 한 줄 — 권한 결정 한 건의 전체 이력을 남기는 기록 모델.

    권한 판단이 끝날 때마다 하나씩 만들어져 JSONL(한 줄에 JSON 하나) 형식으로
    로그 파일에 append 된다. 나중에 "누가·언제·어떤 도구를·왜 허용/거부했는지"를
    추적·감사할 수 있게 하는 것이 목적이다. enum 값들은 문자열로 평탄화해 저장한다.
    """

    # 기록 시각. 기본값으로 지금(UTC) 시각을 ISO 8601 문자열로 자동 채운다.
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str = ""  # 어느 세션에서 일어난 결정인지
    agent_id: str | None = None  # 서브 에이전트가 유발했다면 그 식별자
    tool_name: str = ""  # 대상 도구 이름
    tool_category: str = ""  # 도구 카테고리 (ToolCategory 값을 문자열로)
    tool_input_summary: str = ""  # 입력 요약 — 민감 정보는 제거하고 기록
    decision: str = ""  # 최종 결정: "allow" / "deny" / "ask"
    reason: str = ""  # 결정 이유 (PermissionDecisionReason 값)
    source: str = ""  # 결정 출처 (PermissionRuleSource 값)
    mode: str = ""  # 당시 권한 모드 (PermissionMode 값)
    message: str = ""  # 사람이 읽을 추가 설명
    layers_checked: list[str] = Field(default_factory=list)  # 통과한 권한 레이어 목록
