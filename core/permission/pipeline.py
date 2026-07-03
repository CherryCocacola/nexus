"""
5계층 권한 파이프라인 — Permission Pipeline.

Ch.8.8 사양서 기반. 모든 도구 실행 전에 이 파이프라인을 통과해야 한다.
5개 레이어를 순서대로 검사하여 최종 PermissionDecision을 반환한다.

5계층 구조:
  Layer 1: DenyRuleFilter — deny rule에 해당하면 즉시 거부
  Layer 2: Tool.check_permissions() — 도구 자체의 경로/명령어 검증
  Layer 3: canUseTool — MODE_BEHAVIOR_MAP 기반 모드별 동작 결정
  Layer 4: Hook — Hook 시스템에서 추가 승인/차단 (추후 연동)
  Layer 5: 최종 모드 보정 — BYPASS→ALLOW, PLAN→쓰기DENY 등

핵심 원칙:
  - 5개 레이어 모두 통과해야만 ALLOW
  - 하나라도 DENY면 즉시 중단
  - ASK는 사용자에게 확인 요청 (CLI 레이어에서 처리)
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


class PermissionPipeline:
    """
    5계층 권한 파이프라인.

    도구 실행 전에 check()를 호출하여 PermissionDecision을 받는다.
    ALLOW일 때만 도구를 실행하고, DENY/ASK는 적절히 처리한다.
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

        Args:
            context: 불변 권한 컨텍스트 (모드, 작업 디렉토리 등)
            rules: 추가 권한 규칙 목록 (YAML에서 로드된 것)
            hook_manager: Hook 매니저 (Layer 4용, None이면 건너뜀)
            path_guard: 경로 보호기(Layer 2 사전검사용). None이면 경로 사전검사를
                건너뛴다 — 이 인자를 주지 않는 기존 호출부/단위 테스트의 동작을
                그대로 유지하기 위한 무회귀 안전장치다(부트스트랩만 주입).
            command_filter: 명령어 필터(Layer 2 사전검사용). None이면 명령어
                사전검사를 건너뛴다(위와 동일한 무회귀 안전장치).
        """
        self._context = context
        # context에 이미 포함된 규칙 + 추가 규칙을 합친다
        self._rules = list(context.rules) + (rules or [])
        self._hook_manager = hook_manager
        # Layer 2 사전검사기(선택 주입). 주입이 없으면(None) 사전검사를 통째로
        # 건너뛰므로, PathGuard/CommandFilter를 주입하지 않는 기존 파이프라인
        # 단위 테스트의 판정 경로가 1비트도 바뀌지 않는다(무회귀 핵심).
        self._path_guard = path_guard
        self._command_filter = command_filter
        # 감사 로그 엔트리 목록 (최근 결정 추적)
        self._audit_log: list[PermissionAuditEntry] = []

    @property
    def context(self) -> PermissionContext:
        """현재 권한 컨텍스트를 반환한다."""
        return self._context

    def update_context(self, context: PermissionContext) -> None:
        """권한 컨텍스트를 업데이트한다 (모드 변경 등)."""
        self._context = context

    async def check(
        self,
        tool: BaseTool,
        tool_input: dict,
        tool_use_context: ToolUseContext,
    ) -> PermissionDecision:
        """
        5계층 파이프라인을 실행하여 권한 결정을 반환한다.

        각 레이어를 순서대로 실행하고:
        - DENY가 나오면 즉시 반환 (후속 레이어 건너뜀)
        - ASK가 나오면 일단 기록하고 계속 진행
        - 모든 레이어 통과 후 최종 결정 반환

        Args:
            tool: 실행하려는 도구
            tool_input: 도구 입력 데이터
            tool_use_context: 도구 실행 컨텍스트

        Returns:
            PermissionDecision: AllowDecision, DenyDecision, 또는 AskDecision
        """
        # 감사 로그용 추적 정보
        layers_checked: list[str] = []
        category = self._categorize_tool(tool)

        # ── Layer 1: Deny Rules ──
        # deny rule에 해당하는 도구는 즉시 거부한다
        layers_checked.append("layer1_deny_rules")
        layer1_result = self._check_deny_rules(tool, tool_input)
        if layer1_result is not None:
            self._record_audit(tool, category, tool_input, layer1_result, layers_checked)
            return layer1_result

        # ── Layer 2: Tool.check_permissions() ──
        # 도구 자체의 보안 검사 (경로 검증, 명령어 필터 등)
        layers_checked.append("layer2_tool_check")
        layer2_result = await self._check_tool_permissions(tool, tool_input, tool_use_context)
        if layer2_result is not None:
            self._record_audit(tool, category, tool_input, layer2_result, layers_checked)
            return layer2_result

        # ── Layer 3: canUseTool — MODE_BEHAVIOR_MAP 기반 ──
        # 현재 모드와 도구 카테고리로 기본 동작을 결정한다
        layers_checked.append("layer3_mode_behavior")
        layer3_result = self._check_mode_behavior(tool, category)

        # ── Layer 4: Hook 기반 승인/차단 ──
        # Hook 매니저가 있으면 추가 검사를 실행한다
        if self._hook_manager is not None:
            layers_checked.append("layer4_hooks")
            layer4_result = await self._check_hooks(tool, tool_input)
            if layer4_result is not None:
                self._record_audit(tool, category, tool_input, layer4_result, layers_checked)
                return layer4_result

        # ── Layer 5: 최종 모드 보정 ──
        # BYPASS 모드: ASK → ALLOW
        # PLAN 모드: 쓰기 ASK → DENY
        layers_checked.append("layer5_context_resolution")
        final_result = self._apply_context_resolution(layer3_result, category)

        # 감사 로그 기록
        self._record_audit(tool, category, tool_input, final_result, layers_checked)
        return final_result

    def _categorize_tool(self, tool: BaseTool) -> ToolCategory:
        """
        도구를 보안 카테고리로 분류한다.

        분류 기준:
        1. 도구 이름으로 먼저 분류 (알려진 도구)
        2. behavior flag로 분류 (is_read_only, is_destructive 등)
        3. 기본값은 가장 제한적인 카테고리 (fail-closed)
        """
        name = tool.name.lower()

        # 이름 기반 분류 — 알려진 도구를 먼저 매칭
        readonly_tools = {
            "read",
            "glob",
            "grep",
            "ls",
            "cat",
            "head",
            "tail",
            "taskget",
            "tasklist",
        }
        file_write_tools = {"write", "edit", "notebookedit"}
        bash_tools = {"bash"}
        network_tools = {"webfetch", "websearch"}
        agent_tools = {"agent", "taskcreate", "taskstop"}
        # MCP 도구는 이름에 "mcp__" 접두사가 붙는다 (아래에서 startswith로 검사)

        if name in readonly_tools:
            return ToolCategory.READONLY
        if name in file_write_tools:
            return ToolCategory.FILE_WRITE
        if name in bash_tools:
            return ToolCategory.BASH
        if name in network_tools:
            return ToolCategory.NETWORK
        if name in agent_tools:
            return ToolCategory.AGENT
        if name.startswith("mcp__"):
            return ToolCategory.MCP

        # behavior flag 기반 분류
        if tool.is_read_only:
            return ToolCategory.READONLY
        if tool.is_destructive:
            return ToolCategory.DANGEROUS

        # 기본값: FILE_WRITE (fail-closed — 읽기 전용이 아니면 쓰기로 간주)
        return ToolCategory.FILE_WRITE

    def _resolve_by_mode(self, category: ToolCategory) -> PermissionBehavior:
        """
        MODE_BEHAVIOR_MAP에서 현재 모드와 카테고리의 동작을 조회한다.

        매핑 테이블에 없는 조합이면 ASK를 반환한다 (fail-closed).
        """
        mode = self._context.mode
        mode_map = MODE_BEHAVIOR_MAP.get(mode, {})
        # 매핑에 없으면 ASK (가장 안전한 기본값)
        return mode_map.get(category, PermissionBehavior.ASK)

    def _check_deny_rules(self, tool: BaseTool, tool_input: dict) -> PermissionDecision | None:
        """
        Layer 1: Deny Rule 검사.

        deny rule에 매칭되면 DenyDecision을 반환한다.
        매칭되는 rule이 없으면 None을 반환하고 다음 레이어로 넘어간다.
        """
        for rule in self._rules:
            # DENY 규칙만 검사한다
            if rule.behavior != PermissionBehavior.DENY:
                continue
            # 이 도구에 적용되는 규칙인지 확인
            if rule.matches_tool(tool.name) and rule.matches_input(tool_input):
                return DenyDecision(
                    reason=PermissionDecisionReason.DENY_RULE,
                    source=rule.source,
                    message=rule.rule_content or f"Deny rule: {tool.name}",
                )
        return None

    @staticmethod
    def _normalize_path(path: str, cwd: str) -> str:
        """
        파일 경로를 cwd 기준 절대경로로 정규화한다(보호경로 매칭 정확도 향상용).

        - 이미 절대경로면 그대로 둔다.
        - 상대경로면 cwd와 결합해 os.path.abspath로 접는다(`..`/`.` 정리).
        정규화 실패(비정상 문자 등) 시 원본을 그대로 반환한다(fail-safe).
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
        if self._path_guard is not None:
            # 파일 경로 입력을 표준 키에서 추출한다(도구마다 file_path 또는 path 사용).
            path = tool_input.get("file_path") or tool_input.get("path")
            category = self._categorize_tool(tool)
            is_file_tool = path is not None or category == ToolCategory.FILE_WRITE
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
                if is_write:
                    ok, reason = self._path_guard.is_path_writable(norm_path, cwd)
                else:
                    ok, reason = self._path_guard.is_path_safe(norm_path, cwd)
                if not ok:
                    return DenyDecision(
                        reason=PermissionDecisionReason.TOOL_CHECK_DENIED,
                        source=PermissionRuleSource.SYSTEM,
                        message=f"경로 정책 위반: {reason} (path={path})",
                    )

        # ── ② 명령어 사전검사(CommandFilter) ──
        if self._command_filter is not None:
            command = tool_input.get("command")
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

        먼저 주입된 보안 사전검사기(path_guard/command_filter)로 파일 경로·Bash
        명령어를 검사한다. 여기서 DENY가 나오면 도구 자체 검사까지 갈 것도 없이
        즉시 거부한다(fail-closed). 사전검사기가 주입되지 않았으면(None) 이 단계를
        건너뛰어 기존 동작을 그대로 유지한다.

        그 다음 도구 자체 검사를 수행한다:
        DENY면 DenyDecision, ALLOW면 None (통과), ASK면 AskDecision.
        """
        # ── Layer 2 사전검사: PathGuard / CommandFilter ──
        guard_deny = self._precheck_security_guards(tool, tool_input, tool_use_context)
        if guard_deny is not None:
            return guard_deny

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
        # ALLOW — 통과, 다음 레이어로
        return None

    def _check_mode_behavior(self, tool: BaseTool, category: ToolCategory) -> PermissionDecision:
        """
        Layer 3: MODE_BEHAVIOR_MAP 기반 동작 결정.

        세션 허용(session grant)이 있으면 그것을 우선한다.
        없으면 MODE_BEHAVIOR_MAP에서 조회한다.
        """
        # 세션 중 이미 허용된 도구인지 확인
        if self._context.has_session_grant(tool.name):
            return AllowDecision(
                reason=PermissionDecisionReason.SESSION_GRANT,
                source=PermissionRuleSource.SESSION_GRANT,
                message=f"Session grant: {tool.name}",
            )

        # MODE_BEHAVIOR_MAP에서 동작 조회
        behavior = self._resolve_by_mode(category)

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

        Hook 매니저가 있으면 pre_tool_use 이벤트를 실행한다.
        BLOCK이면 DenyDecision, APPROVE면 AllowDecision, CONTINUE면 None.
        """
        if self._hook_manager is None:
            return None

        from core.hooks.hook_manager import HookDecision, HookEvent, HookInput

        hook_input = HookInput(
            event=HookEvent.PRE_TOOL_USE,
            tool_name=tool.name,
            tool_input=tool_input,
        )
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
        # CONTINUE — 다음 레이어로
        return None

    def _apply_context_resolution(
        self,
        layer3_decision: PermissionDecision,
        category: ToolCategory,
    ) -> PermissionDecision:
        """
        Layer 5: 최종 모드 보정.

        특정 모드에서 Layer 3 결정을 수정한다:
        - BYPASS_PERMISSIONS: ASK → ALLOW (모든 확인 요청을 자동 허용)
        - PLAN: 쓰기 ASK → DENY (계획 모드에서 쓰기는 모두 거부)
        - DONT_ASK: ASK → DENY (사용자에게 묻지 않고 거부)
        """
        mode = self._context.mode

        # BYPASS 모드: ASK를 ALLOW로 변환
        if mode == PermissionMode.BYPASS_PERMISSIONS:
            if isinstance(layer3_decision, AskDecision):
                return AllowDecision(
                    reason=PermissionDecisionReason.BYPASS_MODE,
                    source=PermissionRuleSource.MODE_DEFAULT,
                    message="BYPASS mode: ASK → ALLOW",
                )

        # PLAN 모드: 쓰기 카테고리의 ASK를 DENY로 변환
        if mode == PermissionMode.PLAN:
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

        # DONT_ASK 모드: ASK를 DENY로 변환
        if mode == PermissionMode.DONT_ASK:
            if isinstance(layer3_decision, AskDecision):
                return DenyDecision(
                    reason=PermissionDecisionReason.MODE_DENIES,
                    source=PermissionRuleSource.MODE_DEFAULT,
                    message="DONT_ASK mode: ASK → DENY",
                )

        # 보정 없이 그대로 반환
        return layer3_decision

    def _record_audit(
        self,
        tool: BaseTool,
        category: ToolCategory,
        tool_input: dict,
        decision: PermissionDecision,
        layers_checked: list[str],
    ) -> None:
        """감사 로그 엔트리를 기록한다."""
        # 입력 요약 (민감 정보 제거를 위해 도구의 backfill 사용)
        safe_input = tool.backfill_observable_input(tool_input)
        input_summary = str(safe_input)[:200]  # 200자로 제한

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

        # 최근 1000개만 유지 (메모리 제한)
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]

    def get_audit_log(self) -> list[PermissionAuditEntry]:
        """감사 로그 전체를 반환한다."""
        return list(self._audit_log)

    def get_recent_audit(self, n: int = 50) -> list[PermissionAuditEntry]:
        """최근 n개 감사 로그를 반환한다."""
        return list(self._audit_log[-n:])
