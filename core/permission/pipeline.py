"""
5계층 권한 파이프라인 — Permission Pipeline (Nexus 보안의 관문).

이 파일은 Nexus에서 "도구를 실행해도 되는가?"를 판정하는 유일한 관문이다.
모델(LLM)이 어떤 도구를 호출하든, 실제 실행 직전에 반드시 이 파이프라인을
통과해야 한다. 5개 레이어를 위에서 아래로 순서대로 검사하고, 그 결과를
하나의 PermissionDecision(ALLOW / DENY / ASK) 객체로 뭉쳐서 돌려준다.

왜 5계층으로 나눴나?
  한 곳에서 모든 조건을 if-else로 판정하면 규칙이 늘어날수록 읽기 어렵고
  실수가 생긴다. 그래서 "관심사"별로 레이어를 분리했다. 각 레이어는 자기
  책임만 검사하고, DENY(거부)가 나오면 즉시 멈춘다(뒤 레이어는 보지 않는다).
  이렇게 하면 "왜 막혔는가"를 레이어 이름만 봐도 바로 알 수 있다.

5계층 구조 (위에서 아래 순서로 검사):
  Layer 1: DenyRuleFilter — YAML/세션에서 온 deny 규칙에 걸리면 즉시 거부
  Layer 2: 보안 사전검사 + Tool.check_permissions() — 경로/명령어 위험 차단
  Layer 3: canUseTool — 현재 모드 + 도구 카테고리로 기본 동작 결정
  Layer 4: Hook — 외부 Hook 스크립트가 추가로 승인/차단 (주입 시에만)
  Layer 5: 최종 모드 보정 — BYPASS→ALLOW, PLAN→쓰기DENY, DONT_ASK→DENY 등

핵심 원칙 (반드시 기억):
  - 5개 레이어를 모두 통과해야만 최종 ALLOW가 된다.
  - 어느 레이어든 DENY가 나오면 그 자리에서 즉시 중단하고 DENY를 반환한다.
  - ASK(사용자 확인 필요)는 여기서 결정만 하고, 실제 사용자에게 묻는 것은
    상위 CLI/웹 레이어의 몫이다(이 파일은 UI를 직접 다루지 않는다).
  - Fail-closed: 판단이 애매하면 항상 더 안전한 쪽(ASK/DENY)으로 기운다.

주요 클래스:
  - PermissionPipeline: check()가 진입점. 나머지 _check_* 메서드는 각 레이어.

의존 모듈:
  - core.permission.types — 결정/규칙/카테고리 등 모든 데이터 타입
  - core.tools.base — BaseTool, ToolUseContext (검사 대상 도구와 실행 문맥)
  - core.security.path_guard / command_filter — Layer 2 사전검사기(선택 주입)
  - core.hooks.hook_manager — Layer 4 Hook 실행기(선택 주입)

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from core.permission.types import (
    MODE_BEHAVIOR_MAP,
    AllowDecision,
    AskDecision,
    DenyDecision,
    PermissionAuditEntry,
    PermissionBehavior,
    PermissionContext,
    PermissionDecision,
    PermissionDecisionReason,
    PermissionMode,
    PermissionRule,
    PermissionRuleSource,
    ToolCategory,
)
from core.tools.base import BaseTool, ToolUseContext

if TYPE_CHECKING:
    from core.hooks.hook_manager import HookManager
    from core.security.command_filter import CommandFilter
    from core.security.path_guard import PathGuard

logger = logging.getLogger("nexus.permission")


# ─────────────────────────────────────────────
# 이름 기반 도구 분류 테이블 (공개)
# ─────────────────────────────────────────────
# 왜 모듈 레벨 공개 함수로 뺐나 (CLI Stage 1 A1):
#   CLI의 accept-edits 모드는 "이 도구가 파일 수정(FILE_WRITE) 부류인가"를 알아야
#   자동 승인 여부를 정할 수 있다. 분류표를 CLI에 복붙하면 여기와 어긋나는
#   드리프트가 생기므로, 파이프라인 내부에서만 쓰던 이름 매칭 로직을 공개 함수
#   categorize_tool_name()으로 추출해 CLI가 재사용한다.
#   (behavior flag 기반 2단계·FILE_WRITE 폴백 3단계는 tool 객체가 필요하므로
#    종전대로 _categorize_tool 안에 남긴다 — 판정 결과는 기존과 100% 동일.)
_READONLY_TOOL_NAMES = {
    "read",
    "glob",
    "grep",
    "ls",
    "cat",
    "head",
    "tail",
    "taskget",
    "tasklist",
    # TodoRead/TodoWrite: 세션 내부의 계획 메타데이터만 갱신하며
    # 파일·프로세스·네트워크 등 외부 부작용이 전혀 없다.
    # PLAN 모드에서도 계획 수립 자체는 허용되어야 하므로 READONLY로 취급한다
    # (READONLY 열은 MODE_BEHAVIOR_MAP에서 항상 ALLOW). 도구 자체의
    # is_read_only 플래그는 정직하게 유지된다(TodoWrite=False) — executor의
    # 동시성 파티셔닝은 플래그를, 권한 카테고리는 이 집합을 본다.
    # Layer 1(deny rule)·Layer 4(hook)는 그대로 적용되므로 운영자가
    # permission_rules.yaml에서 여전히 차단할 수 있다(anti-pattern #11 준수).
    "todoread",
    "todowrite",
}
# MultiEdit 포함 — 종전 표에 빠져 있었으나 flag 폴백(3단계)으로 어차피
# FILE_WRITE 판정이었다. 이름 표에 명시해 CLI(accept-edits 자동 승인)에서도
# 같은 분류를 받도록 한다(판정 변화 없음, 도달 경로만 1단계로 앞당김).
# ScaffoldWeb(2026-08-05): 검증 템플릿을 대상 폴더로 복사하는 파일 쓰기 도구 —
# Write/Edit과 같은 정책(accept_edits 자동 승인 대상)으로 분류한다.
_FILE_WRITE_TOOL_NAMES = {"write", "edit", "multiedit", "notebookedit", "scaffoldweb"}
_BASH_TOOL_NAMES = {"bash"}
_NETWORK_TOOL_NAMES = {"webfetch", "websearch"}
_AGENT_TOOL_NAMES = {"agent", "taskcreate", "taskstop"}


def categorize_tool_name(tool_name: str) -> ToolCategory | None:
    """
    도구 "이름"만으로 보안 카테고리를 분류한다 (분류표의 단일 공식 출처).

    파이프라인 _categorize_tool()의 1단계(이름 매칭)와 CLI의 accept-edits
    자동 승인 판정이 모두 이 함수를 쓴다 — 분류 규칙이 한 군데에만 있어야
    두 사용처가 어긋나지 않는다(드리프트 방지).

    Args:
        tool_name: 도구 이름(대소문자 무관 — 내부에서 소문자로 통일).

    Returns:
        이름으로 확정 가능한 ToolCategory. 표에 없는 이름이면 None.
        None의 의미는 호출처마다 다르다:
          - 파이프라인: behavior flag → FILE_WRITE 폴백으로 이어감(fail-closed).
          - CLI 자동 승인: "모르는 도구는 자동 승인하지 않음"(fail-closed —
            여기서 FILE_WRITE로 근사하면 미지의 도구가 무확인 통과하므로 금지).
    """
    name = tool_name.lower()
    if name in _READONLY_TOOL_NAMES:
        return ToolCategory.READONLY
    if name in _FILE_WRITE_TOOL_NAMES:
        return ToolCategory.FILE_WRITE
    if name in _BASH_TOOL_NAMES:
        return ToolCategory.BASH
    if name in _NETWORK_TOOL_NAMES:
        return ToolCategory.NETWORK
    if name in _AGENT_TOOL_NAMES:
        return ToolCategory.AGENT
    # MCP 외부 도구는 이름이 "mcp__"로 시작하도록 규약돼 있어 접두사로 판별한다.
    if name.startswith("mcp__"):
        return ToolCategory.MCP
    return None


class PermissionPipeline:
    """
    5계층 권한 파이프라인의 본체 클래스.

    사용법(호출 측 입장):
      1) 세션 시작 시 PermissionContext(현재 모드/작업 디렉토리/규칙 등)와
         선택적 사전검사기/Hook 매니저를 넣어 인스턴스를 하나 만든다.
      2) 도구 실행 직전마다 await pipeline.check(tool, tool_input, ctx)를 호출한다.
      3) 반환된 PermissionDecision을 보고:
         - AllowDecision → 도구를 실제로 실행한다.
         - DenyDecision  → 실행하지 않고 거부 메시지를 모델/사용자에게 돌려준다.
         - AskDecision   → 사용자에게 확인을 받은 뒤 실행 여부를 정한다.

    상태(state)로 무엇을 들고 있나?
      - _context: 현재 권한 컨텍스트(모드가 바뀌면 update_context로 교체).
      - _rules: 컨텍스트 규칙 + 추가 규칙을 합친 최종 규칙 목록(Layer 1에서 사용).
      - _hook_manager / _path_guard / _command_filter: 선택 주입 협력자.
      - _audit_log: 최근 판정 이력(디버깅/감사용, 메모리 상한 있음).
    """

    def __init__(
        self,
        context: PermissionContext,
        rules: list[PermissionRule] | None = None,
        hook_manager: HookManager | None = None,
        path_guard: PathGuard | None = None,
        command_filter: CommandFilter | None = None,
    ) -> None:
        """
        파이프라인을 초기화한다.

        협력자(hook_manager/path_guard/command_filter)는 모두 선택 주입이다.
        주입하지 않으면(None) 해당 검사 단계를 통째로 건너뛰므로, 예전부터
        이 인자들을 넘기지 않던 호출부와 단위 테스트의 판정 결과가 1비트도
        바뀌지 않는다. 이것이 "무회귀(no-regression)" 설계의 핵심이다.

        Args:
            context: 불변 권한 컨텍스트 (현재 모드, 작업 디렉토리, 세션 규칙 등).
            rules: 추가 권한 규칙 목록 (주로 YAML에서 로드된 deny/allow 규칙).
            hook_manager: Hook 매니저 (Layer 4용). None이면 Layer 4를 건너뛴다.
            path_guard: 경로 보호기(Layer 2 사전검사용). None이면 경로 사전검사를
                건너뛴다 — 이 인자를 주지 않는 기존 호출부/단위 테스트의 동작을
                그대로 유지하기 위한 무회귀 안전장치다(부트스트랩만 주입).
            command_filter: 명령어 필터(Layer 2 사전검사용). None이면 명령어
                사전검사를 건너뛴다(위와 동일한 무회귀 안전장치).
        """
        self._context = context
        # context에 이미 들어 있는 규칙과, 생성자로 별도 전달된 규칙을 합쳐
        # 하나의 평평한 리스트로 만든다. Layer 1(deny 검사)이 이 목록을 순회한다.
        # rules가 None이면 (rules or []) 로 빈 리스트를 써서 TypeError를 막는다.
        self._rules = list(context.rules) + (rules or [])
        self._hook_manager = hook_manager
        # Layer 2 사전검사기(선택 주입). 주입이 없으면(None) 사전검사를 통째로
        # 건너뛰므로, PathGuard/CommandFilter를 주입하지 않는 기존 파이프라인
        # 단위 테스트의 판정 경로가 1비트도 바뀌지 않는다(무회귀 핵심).
        self._path_guard = path_guard
        self._command_filter = command_filter
        # 감사 로그 엔트리 목록. 매 판정마다 한 줄씩 append 되며, 나중에
        # get_audit_log()/get_recent_audit()로 조회한다. 무한히 쌓이면 메모리를
        # 먹으므로 _record_audit에서 상한(1000개)을 두고 오래된 것을 잘라낸다.
        self._audit_log: list[PermissionAuditEntry] = []

    @property
    def context(self) -> PermissionContext:
        """현재 권한 컨텍스트를 반환한다(외부에서 모드/디렉토리 등을 읽을 때 사용)."""
        return self._context

    def update_context(self, context: PermissionContext) -> None:
        """
        권한 컨텍스트를 통째로 교체한다.

        예를 들어 사용자가 세션 도중 권한 모드를 DEFAULT → ACCEPT_EDITS로
        바꾸면, 상위 레이어가 새 컨텍스트를 만들어 이 메서드로 갈아끼운다.
        PermissionContext는 불변(frozen)이라 필드 수정 대신 '교체'로 반영한다.
        """
        self._context = context

    async def check(
        self,
        tool: BaseTool,
        tool_input: dict,
        tool_use_context: ToolUseContext,
    ) -> PermissionDecision:
        """
        5계층 파이프라인을 실행하여 최종 권한 결정을 반환한다(이 클래스의 진입점).

        전체 흐름을 한눈에:
          Layer1(deny) → Layer2(보안검사) → Layer3(모드동작)
          → Layer4(Hook) → Layer5(모드보정) → 감사기록 → 반환

        각 레이어의 규칙:
          - DENY가 나오면 즉시 반환한다(뒤 레이어는 아예 실행하지 않는다).
          - Layer 3의 결과(ALLOW/ASK/DENY)는 곧바로 반환하지 않고, Layer 5의
            모드 보정을 거친 뒤 최종 확정된다(예: BYPASS면 ASK가 ALLOW로 바뀜).
          - 어떤 경로로 끝나든 마지막에 _record_audit로 판정 이력을 남긴다.

        layers_checked 리스트는 "이번 판정에서 어느 레이어까지 검사했는지"를
        문자열로 쌓아, 감사 로그에 함께 저장한다(사후 디버깅에 유용).

        Args:
            tool: 실행하려는 도구(BaseTool 구현체).
            tool_input: 모델이 채워 넣은 도구 입력 인자(dict).
            tool_use_context: 실행 문맥(작업 디렉토리 cwd 등)을 담은 객체.

        Returns:
            PermissionDecision: AllowDecision, DenyDecision, 또는 AskDecision 중 하나.
        """
        # 감사 로그용 추적 정보 — 통과한 레이어 이름을 순서대로 쌓는다.
        layers_checked: list[str] = []
        # 도구를 7개 보안 카테고리 중 하나로 분류한다. Layer 3/5의 판정 기준이 된다.
        category = self._categorize_tool(tool)

        # ── Layer 1: Deny Rules ──
        # 명시적 deny 규칙(YAML/세션)에 걸리는 도구는 다른 검사 없이 즉시 거부한다.
        # 가장 강력하고 우선순위가 높은 차단이라 맨 앞에 둔다.
        layers_checked.append("layer1_deny_rules")
        layer1_result = self._check_deny_rules(tool, tool_input)
        if layer1_result is not None:
            self._record_audit(tool, category, tool_input, layer1_result, layers_checked)
            return layer1_result

        # ── Layer 2: 보안 사전검사 + Tool.check_permissions() ──
        # 경로 순회/보호경로/위험 명령어 등 "도구 고유의 보안 위험"을 검사한다.
        # 사전검사기 주입 여부와 무관하게 도구 자체의 check_permissions도 함께 돈다.
        layers_checked.append("layer2_tool_check")
        layer2_result = await self._check_tool_permissions(tool, tool_input, tool_use_context)
        if layer2_result is not None:
            self._record_audit(tool, category, tool_input, layer2_result, layers_checked)
            return layer2_result

        # ── Layer 3: canUseTool — MODE_BEHAVIOR_MAP 기반 ──
        # 여기서는 즉시 반환하지 않는다. 현재 모드+카테고리의 "기본 동작"만 뽑아
        # layer3_result에 담아 두고, Layer 5에서 모드별 보정을 거쳐 확정한다.
        layers_checked.append("layer3_mode_behavior")
        layer3_result = self._check_mode_behavior(tool, category)

        # ── Layer 4: Hook 기반 승인/차단 ──
        # Hook 매니저가 주입된 경우에만 실행한다. 외부 Hook이 BLOCK/APPROVE를
        # 내면 그 결정으로 즉시 끝나고, CONTINUE(None)면 Layer 5로 넘어간다.
        if self._hook_manager is not None:
            layers_checked.append("layer4_hooks")
            layer4_result = await self._check_hooks(tool, tool_input)
            if layer4_result is not None:
                self._record_audit(tool, category, tool_input, layer4_result, layers_checked)
                return layer4_result

        # ── Layer 5: 최종 모드 보정 ──
        # Layer 3의 잠정 결정을 현재 모드에 맞게 마지막으로 다듬는다.
        #   BYPASS 모드: ASK → ALLOW (확인 요청을 자동 통과)
        #   PLAN 모드: 쓰기성 ASK → DENY (계획 중엔 부작용 금지)
        layers_checked.append("layer5_context_resolution")
        final_result = self._apply_context_resolution(layer3_result, category)

        # 확정된 최종 결정을 감사 로그에 남기고 반환한다.
        self._record_audit(tool, category, tool_input, final_result, layers_checked)
        return final_result

    def _categorize_tool(self, tool: BaseTool) -> ToolCategory:
        """
        도구를 7개 보안 카테고리(ToolCategory) 중 하나로 분류한다.

        이 카테고리는 Layer 3(모드별 동작)과 Layer 5(모드 보정)의 판정 기준이므로
        분류가 곧 보안 강도를 결정한다. 그래서 애매하면 무조건 더 위험한 쪽으로
        분류하는 fail-closed 원칙을 따른다.

        분류 순서(위에서 먼저 매칭되는 것을 채택):
        1. 도구 이름으로 매칭 — 우리가 아는 표준 도구는 이름만 보고 확정한다.
           (분류표는 모듈 레벨 공개 함수 categorize_tool_name()에 있다 — CLI와 공유.)
        2. behavior flag로 매칭 — 이름을 모르는 도구는 읽기전용/파괴적 플래그로 본다.
        3. 그래도 못 정하면 FILE_WRITE로 간주 — "읽기 전용이 아니면 쓰기" (fail-closed).
        """
        # 1단계: 이름 매칭 — 공개 분류 함수에 위임(단일 출처, CLI와 드리프트 방지).
        by_name = categorize_tool_name(tool.name)
        if by_name is not None:
            return by_name

        # 2단계: behavior flag 기반 분류 — 이름을 모르는 도구(플러그인 등)에 대비.
        # 읽기 전용이면 READONLY, 파괴적이면 가장 위험한 DANGEROUS로 본다.
        if tool.is_read_only:
            return ToolCategory.READONLY
        if tool.is_destructive:
            return ToolCategory.DANGEROUS

        # 3단계: 기본값 FILE_WRITE — fail-closed. "읽기 전용이라 증명되지 않았으면
        # 쓰기로 취급"해서, 알 수 없는 도구가 무방비로 통과하는 일을 막는다.
        return ToolCategory.FILE_WRITE

    def _resolve_by_mode(self, category: ToolCategory) -> PermissionBehavior:
        """
        MODE_BEHAVIOR_MAP에서 (현재 모드, 카테고리)의 기본 동작을 조회하는 헬퍼.

        MODE_BEHAVIOR_MAP은 "모드마다 각 카테고리를 ALLOW/ASK/DENY 중 무엇으로
        다룰지"를 미리 정의한 2차원 표다. 여기서는 그 표를 읽기만 한다.
        표에 없는 조합이면 가장 안전한 ASK로 떨어뜨린다(fail-closed).

        Returns:
            PermissionBehavior: ALLOW / ASK / DENY 중 하나.
        """
        mode = self._context.mode
        # 현재 모드에 해당하는 하위 매핑(카테고리→동작)을 꺼낸다. 없으면 빈 dict.
        mode_map = MODE_BEHAVIOR_MAP.get(mode, {})
        # 그 안에서 카테고리를 조회. 없으면 ASK를 기본값으로 반환(가장 안전).
        return mode_map.get(category, PermissionBehavior.ASK)

    def _check_deny_rules(self, tool: BaseTool, tool_input: dict) -> PermissionDecision | None:
        """
        Layer 1: Deny Rule 검사.

        _rules 목록을 훑어, 이 (도구, 입력)에 들어맞는 DENY 규칙이 하나라도
        있으면 곧바로 DenyDecision을 만들어 반환한다. 매칭되는 규칙이 없으면
        None을 반환해 "이 레이어는 통과"임을 알린다(호출 측이 다음 레이어로 감).

        반환 타입이 PermissionDecision | None인 이유: None은 "판단 보류(통과)",
        객체는 "여기서 확정(거부)"이라는 두 신호를 구분하기 위함이다.
        """
        for rule in self._rules:
            # 관심사는 오직 DENY 규칙뿐. allow/ask 규칙은 이 레이어에서 무시한다.
            if rule.behavior != PermissionBehavior.DENY:
                continue
            # 도구 이름과 입력이 모두 규칙 조건에 맞아야 실제로 적용되는 규칙이다.
            if rule.matches_tool(tool.name) and rule.matches_input(tool_input):
                return DenyDecision(
                    reason=PermissionDecisionReason.DENY_RULE,
                    source=rule.source,
                    message=rule.rule_content or f"Deny rule: {tool.name}",
                )
        # 어떤 DENY 규칙에도 걸리지 않음 → 통과(다음 레이어에서 계속 검사).
        return None

    @staticmethod
    def _normalize_path(path: str, cwd: str) -> str:
        """
        파일 경로를 cwd 기준 절대경로로 정규화한다(보호경로 매칭 정확도 향상용).

        왜 필요한가? 상대경로 상태로는 PathGuard의 글롭 패턴(예: `**/.ssh/*`)에
        부모 경로 정보가 없어 매칭이 새는 경우가 있다. cwd와 결합한 절대경로로
        펴 놓으면 보호경로/순회 검사가 훨씬 정확해진다.

        동작:
        - 이미 절대경로면 그대로 둔다(불필요한 변형 방지).
        - 상대경로면 cwd와 이어 붙인 뒤 os.path.abspath로 접는다(`..`/`.` 정리).
        정규화 도중 예외(비정상 문자 등)가 나면 원본 경로를 그대로 돌려준다.
        여기서 예외로 죽지 않는 것이 목적이며, 실제 차단 판단은 뒤 검사에 맡긴다.

        참고: @staticmethod 인 이유는 인스턴스 상태(self)를 전혀 쓰지 않는
        순수 경로 계산이기 때문이다(테스트/재사용이 쉬워진다).
        """
        try:
            if os.path.isabs(path):
                return path
            return os.path.abspath(os.path.join(cwd, path))
        except (OSError, ValueError):
            return path

    def _precheck_security_guards(
        self,
        tool: BaseTool,
        tool_input: dict,
        tool_use_context: ToolUseContext,
    ) -> PermissionDecision | None:
        """
        Layer 2 공통 사전검사 — PathGuard(경로) + CommandFilter(명령어).

        도구가 스스로 하는 check_permissions()와는 별개로, 파이프라인 차원에서
        "위험한 경로 접근"과 "위험한 셸 명령"을 한 번 더 걸러 주는 안전망이다.
        도구 구현이 실수로 검사를 빠뜨려도 여기서 잡을 수 있다(방어적 이중화).

        무회귀 원칙: path_guard/command_filter가 주입되지 않았으면(둘 다 None)
        아무 것도 하지 않고 None을 반환한다. 즉 사전검사기를 넘기지 않는 기존
        파이프라인 사용처의 판정은 전혀 바뀌지 않는다.

        검사 대상:
          1) 파일계열 도구 — tool_input에 file_path/path 키가 있거나 카테고리가
             FILE_WRITE면 PathGuard로 경로를 검사한다. 쓰기 도구면 is_path_writable
             (읽기전용 경로까지 차단), 그 외에는 is_path_safe(순회/보호경로/UNC).
          2) Bash 도구 — tool_input에 command 키가 있으면 CommandFilter로 검사한다.
             critical/high/medium 위험 severity면 DENY. "unknown"(안전 목록에 없음)은
             여기서 차단하지 않는다 — 그건 확인(ASK) 영역이라 Layer 3에 맡긴다.

        Returns:
            DenyDecision(차단) 또는 None(사전검사 통과/미적용).
        """
        # 작업 디렉토리 — 상대경로 해석과 순회 검사의 기준점.
        cwd = tool_use_context.cwd or "."

        # ── ① 경로 사전검사(PathGuard) ──
        # path_guard가 주입된 경우에만 수행한다(미주입이면 이 블록 자체를 건너뜀).
        if self._path_guard is not None:
            # 파일 경로 입력을 표준 키에서 추출한다(도구마다 file_path 또는 path 사용).
            # 앞의 키가 없거나 빈 값이면 or 로 다음 키를 시도한다.
            path = tool_input.get("file_path") or tool_input.get("path")
            category = self._categorize_tool(tool)
            # 경로 키가 있거나 카테고리가 FILE_WRITE면 "파일을 다루는 도구"로 본다.
            is_file_tool = path is not None or category == ToolCategory.FILE_WRITE
            # 실제 검사할 경로 문자열이 존재할 때만 검사에 들어간다.
            if is_file_tool and path:
                # cwd 기준 절대경로로 정규화해 보호경로 매칭 정확도를 높인다.
                # 왜: 상대경로 ".ssh/id_rsa"는 PathGuard의 `**/.ssh/*` 글롭에 부모
                # 세그먼트가 없어 걸리지 않는다. cwd와 결합한 절대경로
                # "{cwd}/.ssh/id_rsa"는 정확히 매칭돼 보호된다. 순회(../..)는
                # 절대경로로 바꿔도 is_path_safe의 cwd-scope 검사가 그대로 잡는다
                # (abspath가 `..`를 접어 cwd 밖으로 나가면 relative_to 실패 → 차단).
                norm_path = self._normalize_path(str(path), cwd)
                # 쓰기 여부 판정: 쓰기/위험 카테고리이거나 읽기전용이 아닌 도구는
                # 쓰기로 간주해 읽기전용 경로까지 차단한다(fail-closed).
                is_write = category in (
                    ToolCategory.FILE_WRITE,
                    ToolCategory.DANGEROUS,
                ) or not tool.is_read_only
                # 쓰기 도구는 더 엄격한 is_path_writable(읽기전용 경로까지 차단)로,
                # 읽기 도구는 is_path_safe(순회/보호경로/UNC만 차단)로 검사한다.
                if is_write:
                    ok, reason = self._path_guard.is_path_writable(norm_path, cwd)
                else:
                    ok, reason = self._path_guard.is_path_safe(norm_path, cwd)
                # ok가 False면 경로 정책 위반 → 즉시 거부(뒤 명령어 검사도 안 함).
                if not ok:
                    return DenyDecision(
                        reason=PermissionDecisionReason.TOOL_CHECK_DENIED,
                        source=PermissionRuleSource.SYSTEM,
                        message=f"경로 정책 위반: {reason} (path={path})",
                    )

        # ── ② 명령어 사전검사(CommandFilter) ──
        # command_filter가 주입된 경우에만 수행한다(주로 Bash 도구 대상).
        if self._command_filter is not None:
            command = tool_input.get("command")
            # command 키가 문자열이고 공백만 있는 게 아닐 때만 검사한다.
            if isinstance(command, str) and command.strip():
                safe, severity, reason = self._command_filter.check_command(command)
                # 실제 "위험" 판정(critical/high/medium)만 차단한다. unknown은 통과시켜
                # Layer 3의 ASK 흐름이 처리하도록 남긴다(과잉 차단 방지).
                if not safe and severity in ("critical", "high", "medium"):
                    return DenyDecision(
                        reason=PermissionDecisionReason.TOOL_CHECK_DENIED,
                        source=PermissionRuleSource.SYSTEM,
                        message=f"명령어 정책 위반[{severity}]: {reason}",
                    )

        # 사전검사 통과(또는 미적용) — 다음 검사(도구 자체 check_permissions)로.
        return None

    async def _check_tool_permissions(
        self,
        tool: BaseTool,
        tool_input: dict,
        tool_use_context: ToolUseContext,
    ) -> PermissionDecision | None:
        """
        Layer 2: PathGuard/CommandFilter 사전검사 + 도구 자체의 check_permissions().

        이 레이어는 두 단계로 이뤄진다:
          (1) 파이프라인 사전검사 — 주입된 path_guard/command_filter로 경로·명령을
              먼저 훑는다. 여기서 DENY가 나오면 도구 자체 검사까지 갈 것도 없이
              즉시 거부한다(fail-closed). 미주입(None)이면 이 단계는 건너뛰어
              기존 동작을 그대로 유지한다.
          (2) 도구 자체 검사 — tool.check_permissions()를 호출해 도구가 자기만의
              규칙으로 판단하게 한다. 이 도구 검사는 사전검사기 주입 여부와 무관하게
              항상 실행된다.

        도구 검사 결과의 변환:
          DENY  → DenyDecision (거부 확정)
          ASK   → AskDecision  (사용자 확인 필요)
          ALLOW → None         (이 레이어 통과 — 다음 레이어에서 계속 판단)
        """
        # ── Layer 2 사전검사: PathGuard / CommandFilter ──
        # 사전검사에서 거부가 나오면 도구 자체 검사는 생략하고 바로 반환한다.
        guard_deny = self._precheck_security_guards(tool, tool_input, tool_use_context)
        if guard_deny is not None:
            return guard_deny

        # 도구 자체의 권한 검사(각 BaseTool 구현이 자기 규칙으로 판단).
        result = await tool.check_permissions(tool_input, tool_use_context)

        if result.behavior == PermissionBehavior.DENY:
            return DenyDecision(
                reason=PermissionDecisionReason.TOOL_CHECK_DENIED,
                source=PermissionRuleSource.SYSTEM,
                message=result.message or f"Tool {tool.name} denied",
            )
        if result.behavior == PermissionBehavior.ASK:
            return AskDecision(
                reason=PermissionDecisionReason.TOOL_CHECK_DENIED,
                source=PermissionRuleSource.SYSTEM,
                message=result.message or f"Tool {tool.name} requires confirmation",
                tool_name=tool.name,
            )
        # ALLOW — 이 레이어는 통과. None을 돌려 다음 레이어(Layer 3)로 넘긴다.
        return None

    def _check_mode_behavior(self, tool: BaseTool, category: ToolCategory) -> PermissionDecision:
        """
        Layer 3: MODE_BEHAVIOR_MAP 기반 동작 결정(잠정 결정 생성).

        판정 우선순위:
          1) 세션 허용(session grant): 사용자가 이번 세션에서 "이 도구는 계속
             허용"이라고 이미 승인한 경우. 모드 표보다 우선해 바로 ALLOW.
          2) 위가 없으면 MODE_BEHAVIOR_MAP에서 (모드, 카테고리) 기본 동작을 조회해
             ALLOW/DENY/ASK 결정을 만든다.

        주의: 여기 결정은 '잠정'이다. ASK로 나와도 Layer 5(모드 보정)에서
        ALLOW나 DENY로 바뀔 수 있다. 그래서 check()는 이 값을 즉시 반환하지 않고
        Layer 5를 거친다.

        Returns:
            PermissionDecision: Allow/Deny/Ask 중 하나(항상 결정을 반환한다).
        """
        # 세션 중 이미 사용자 승인을 받아 둔 도구라면 더 볼 것 없이 허용한다.
        if self._context.has_session_grant(tool.name):
            return AllowDecision(
                reason=PermissionDecisionReason.SESSION_GRANT,
                source=PermissionRuleSource.SESSION_GRANT,
                message=f"Session grant: {tool.name}",
            )

        # 세션 허용이 없으면 모드+카테고리 기본 동작을 표에서 조회한다.
        behavior = self._resolve_by_mode(category)

        # 조회된 동작(ALLOW/DENY/ASK)을 그에 맞는 Decision 객체로 감싸 반환한다.
        if behavior == PermissionBehavior.ALLOW:
            return AllowDecision(
                reason=PermissionDecisionReason.MODE_ALLOWS,
                source=PermissionRuleSource.MODE_DEFAULT,
                message=f"Mode {self._context.mode.value} allows {category.value}",
            )
        if behavior == PermissionBehavior.DENY:
            return DenyDecision(
                reason=PermissionDecisionReason.MODE_DENIES,
                source=PermissionRuleSource.MODE_DEFAULT,
                message=f"Mode {self._context.mode.value} denies {category.value}",
            )
        # ASK
        return AskDecision(
            reason=PermissionDecisionReason.MODE_DENIES,
            source=PermissionRuleSource.MODE_DEFAULT,
            message=f"Mode {self._context.mode.value} asks for {category.value}",
            tool_name=tool.name,
        )

    async def _check_hooks(self, tool: BaseTool, tool_input: dict) -> PermissionDecision | None:
        """
        Layer 4: Hook 기반 승인/차단.

        운영자가 정의한 외부 Hook(스크립트/콜백)에게 "이 도구 실행 직전에 개입할
        기회"를 주는 확장 지점이다. 예: 특정 파일 수정을 감사 시스템에 알리거나,
        회사 정책상 금지된 명령을 추가로 차단하는 용도.

        동작:
          - Hook 매니저가 없으면(None) 아무 것도 하지 않고 None(통과).
          - PRE_TOOL_USE 이벤트를 실행하고 그 결정을 변환한다:
              BLOCK    → DenyDecision (차단)
              APPROVE  → AllowDecision (승인 — 이후 레이어를 건너뛰고 실행 확정)
              CONTINUE → None (개입 없음, 다음 레이어로)

        참고: HookDecision 등은 순환 import를 피하려고 함수 안에서 지연 import 한다
        (모듈 최상단에서 import하면 core.hooks ↔ core.permission가 서로를 물 수 있음).
        """
        if self._hook_manager is None:
            return None

        # 지연 import — 순환 의존을 끊기 위해 함수 실행 시점에만 불러온다.
        from core.hooks.hook_manager import HookDecision, HookEvent, HookInput

        hook_input = HookInput(
            event=HookEvent.PRE_TOOL_USE,
            tool_name=tool.name,
            tool_input=tool_input,
        )
        # 등록된 PRE_TOOL_USE Hook들을 실행하고 종합 결정을 받는다.
        hook_result = await self._hook_manager.run(HookEvent.PRE_TOOL_USE, hook_input)

        if hook_result.decision == HookDecision.BLOCK:
            return DenyDecision(
                reason=PermissionDecisionReason.HOOK_BLOCKED,
                source=PermissionRuleSource.HOOK,
                message=hook_result.block_reason or "Hook blocked",
            )
        if hook_result.decision == HookDecision.APPROVE:
            return AllowDecision(
                reason=PermissionDecisionReason.HOOK_APPROVED,
                source=PermissionRuleSource.HOOK,
                message=hook_result.message or "Hook approved",
            )
        # CONTINUE — Hook이 개입하지 않음. None을 돌려 Layer 5로 진행한다.
        return None

    def _apply_context_resolution(
        self,
        layer3_decision: PermissionDecision,
        category: ToolCategory,
    ) -> PermissionDecision:
        """
        Layer 5: 최종 모드 보정 — 잠정 결정에 현재 모드의 정책을 마지막으로 입힌다.

        Layer 3가 만든 잠정 결정(layer3_decision)을 특정 모드에서 다시 손본다.
        핵심은 "ASK를 어느 쪽으로 확정할지"가 모드마다 다르다는 점이다:
        - BYPASS_PERMISSIONS: ASK → ALLOW (모든 확인 요청을 자동 허용. 신뢰 환경용)
        - PLAN: 쓰기성 ASK → DENY (계획만 세우는 모드라 부작용 있는 쓰기는 전부 거부)
        - DONT_ASK: ASK → DENY (사용자에게 묻지 않기로 했으니 확인 대신 거부)

        위 조건에 해당하지 않으면 Layer 3 결정을 손대지 않고 그대로 반환한다.
        (ALLOW/DENY로 이미 확정된 결정은 여기서 뒤집지 않는다.)

        Returns:
            PermissionDecision: 보정이 반영된 최종 결정.
        """
        mode = self._context.mode

        # BYPASS 모드: 확인 요청(ASK)을 자동 허용(ALLOW)으로 승격한다.
        if mode == PermissionMode.BYPASS_PERMISSIONS:
            if isinstance(layer3_decision, AskDecision):
                return AllowDecision(
                    reason=PermissionDecisionReason.BYPASS_MODE,
                    source=PermissionRuleSource.MODE_DEFAULT,
                    message="BYPASS mode: ASK → ALLOW",
                )

        # PLAN 모드: 부작용을 내는 "쓰기성" 카테고리의 ASK만 DENY로 낮춘다.
        # (읽기 전용 ASK는 계획 단계에서도 위험하지 않으므로 여기선 건드리지 않음.)
        if mode == PermissionMode.PLAN:
            # 시스템에 변화를 줄 수 있는 카테고리 = "쓰기성"으로 간주하는 집합.
            write_categories = {
                ToolCategory.FILE_WRITE,
                ToolCategory.BASH,
                ToolCategory.DANGEROUS,
                ToolCategory.NETWORK,
                ToolCategory.AGENT,
                ToolCategory.MCP,
            }
            if isinstance(layer3_decision, AskDecision) and category in write_categories:
                return DenyDecision(
                    reason=PermissionDecisionReason.PLAN_MODE_WRITE,
                    source=PermissionRuleSource.MODE_DEFAULT,
                    message="PLAN mode: write ASK → DENY",
                )

        # DONT_ASK 모드: 물어보지 않기로 한 모드이므로 ASK를 DENY로 확정한다.
        if mode == PermissionMode.DONT_ASK:
            if isinstance(layer3_decision, AskDecision):
                return DenyDecision(
                    reason=PermissionDecisionReason.MODE_DENIES,
                    source=PermissionRuleSource.MODE_DEFAULT,
                    message="DONT_ASK mode: ASK → DENY",
                )

        # 위 어떤 모드 조건에도 해당하지 않으면 Layer 3 결정을 그대로 최종 확정.
        return layer3_decision

    def _record_audit(
        self,
        tool: BaseTool,
        category: ToolCategory,
        tool_input: dict,
        decision: PermissionDecision,
        layers_checked: list[str],
    ) -> None:
        """
        한 번의 권한 판정 결과를 감사 로그(_audit_log)에 한 줄로 남긴다.

        왜 필요한가? "이 도구가 왜 허용/거부됐는지"를 나중에 추적하려면 결정 근거와
        통과 레이어를 기록해 둬야 한다. 보안 사고 조사·디버깅의 핵심 자료가 된다.

        민감 정보 주의: 도구 입력에는 비밀번호/토큰 같은 값이 있을 수 있으므로,
        원본 대신 tool.backfill_observable_input()로 관찰 가능한(마스킹된) 형태로
        바꾼 뒤 200자로 잘라서 저장한다.
        """
        # 입력 요약 — 도구의 backfill로 민감 정보를 걸러낸 관찰용 입력을 얻는다.
        safe_input = tool.backfill_observable_input(tool_input)
        input_summary = str(safe_input)[:200]  # 200자로 제한(로그 비대화 방지)

        # 판정 이력 한 건을 구조화된 엔트리로 만든다. reason/source는 Enum일 수도
        # 문자열일 수도 있어, .value 속성이 있으면 그 값을 쓰고 없으면 str()로 감싼다.
        entry = PermissionAuditEntry(
            session_id=self._context.session_id,
            agent_id=self._context.agent_id,
            tool_name=tool.name,
            tool_category=category.value,
            tool_input_summary=input_summary,
            decision=decision.type,
            reason=(
                decision.reason.value if hasattr(decision.reason, "value") else str(decision.reason)
            ),
            source=(
                decision.source.value if hasattr(decision.source, "value") else str(decision.source)
            ),
            mode=self._context.mode.value,
            message=decision.message,
            layers_checked=layers_checked,
        )
        self._audit_log.append(entry)

        # 무한정 쌓이면 메모리를 잠식하므로 상한을 둔다. 1000개를 넘어서면
        # 최근 500개만 남기고 앞부분을 버린다(가장 오래된 이력부터 폐기).
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]

    def get_audit_log(self) -> list[PermissionAuditEntry]:
        """
        감사 로그 전체를 복사본으로 반환한다.

        내부 리스트를 그대로 넘기지 않고 list(...)로 얕은 복사를 만들어,
        호출 측이 실수로 원본을 수정하는 것을 막는다(방어적 복사).
        """
        return list(self._audit_log)

    def get_recent_audit(self, n: int = 50) -> list[PermissionAuditEntry]:
        """
        가장 최근 n개의 감사 로그만 잘라서 반환한다(기본 50개).

        전체가 아니라 최근 이력만 빠르게 확인하고 싶을 때 사용한다.
        여기서도 슬라이스로 새 리스트를 만들어 원본을 보호한다.
        """
        return list(self._audit_log[-n:])
