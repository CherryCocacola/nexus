"""
core/permission/ + core/security/ 단위 테스트.

5계층 권한, 경로 보호, 명령어 필터를 검증한다.
"""

from __future__ import annotations

from core.permission.types import (
    MODE_BEHAVIOR_MAP,
    PermissionBehavior,
    PermissionContext,
    PermissionMode,
    PermissionRule,
    PermissionRuleSource,
    ToolCategory,
)
from core.security.command_filter import CommandFilter
from core.security.path_guard import PathGuard


class TestPermissionMode:
    """PermissionMode 테스트."""

    def test_seven_modes(self):
        """7가지 권한 모드가 정의되어 있는지 확인한다."""
        assert len(PermissionMode) == 7

    def test_plan_mode_denies_writes(self):
        """PLAN 모드에서 파일 쓰기가 DENY인지 확인한다."""
        behavior = MODE_BEHAVIOR_MAP[PermissionMode.PLAN][ToolCategory.FILE_WRITE]
        assert behavior == PermissionBehavior.DENY

    def test_bypass_allows_everything(self):
        """BYPASS 모드에서 모든 카테고리가 ALLOW인지 확인한다."""
        for category in ToolCategory:
            if category in MODE_BEHAVIOR_MAP[PermissionMode.BYPASS_PERMISSIONS]:
                behavior = MODE_BEHAVIOR_MAP[PermissionMode.BYPASS_PERMISSIONS][category]
                assert behavior == PermissionBehavior.ALLOW

    def test_default_allows_readonly(self):
        """DEFAULT 모드에서 읽기 전용이 ALLOW인지 확인한다."""
        behavior = MODE_BEHAVIOR_MAP[PermissionMode.DEFAULT][ToolCategory.READONLY]
        assert behavior == PermissionBehavior.ALLOW

    def test_default_asks_for_bash(self):
        """DEFAULT 모드에서 Bash가 ASK인지 확인한다."""
        behavior = MODE_BEHAVIOR_MAP[PermissionMode.DEFAULT][ToolCategory.BASH]
        assert behavior == PermissionBehavior.ASK


class TestPermissionContext:
    """PermissionContext 테스트."""

    def test_frozen(self):
        """PermissionContext가 불변인지 확인한다."""
        ctx = PermissionContext(
            mode=PermissionMode.DEFAULT,
            working_directory="/workspace",
        )
        # frozen이므로 직접 수정 불가 — ValidationError가 발생해야 정상
        import pydantic
        import pytest
        with pytest.raises(pydantic.ValidationError):
            ctx.mode = PermissionMode.PLAN

    def test_with_mode_creates_new(self):
        """with_mode()가 새 객체를 반환하는지 확인한다."""
        ctx = PermissionContext(
            mode=PermissionMode.DEFAULT,
            working_directory="/workspace",
        )
        new_ctx = ctx.with_mode(PermissionMode.PLAN)
        assert new_ctx.mode == PermissionMode.PLAN
        assert ctx.mode == PermissionMode.DEFAULT  # 원본 불변

    def test_with_session_grant(self):
        """with_session_grant()가 규칙을 추가하는지 확인한다."""
        ctx = PermissionContext(
            mode=PermissionMode.DEFAULT,
            working_directory="/workspace",
        )
        rule = PermissionRule(
            source=PermissionRuleSource.SESSION_GRANT,
            behavior=PermissionBehavior.ALLOW,
            tool_name="Bash",
        )
        new_ctx = ctx.with_session_grant(rule)
        assert len(new_ctx.session_grants) == 1


class TestPermissionRule:
    """PermissionRule 테스트."""

    def test_matches_tool_exact(self):
        """정확한 도구 이름 매칭을 확인한다."""
        rule = PermissionRule(
            source=PermissionRuleSource.USER_CONFIG,
            behavior=PermissionBehavior.ALLOW,
            tool_name="Read",
        )
        assert rule.matches_tool("Read") is True
        assert rule.matches_tool("Write") is False

    def test_wildcard_matches_all(self):
        """'*' 패턴이 모든 도구에 매칭되는지 확인한다."""
        rule = PermissionRule(
            source=PermissionRuleSource.SYSTEM,
            behavior=PermissionBehavior.DENY,
            tool_name="*",
        )
        assert rule.matches_tool("Read") is True
        assert rule.matches_tool("Bash") is True


class TestPathGuard:
    """PathGuard 테스트."""

    def test_etc_passwd_blocked(self):
        """보호 경로 /etc/passwd가 차단되는지 확인한다."""
        pg = PathGuard()
        safe, _ = pg.is_path_safe("/etc/passwd", "/workspace")
        assert safe is False

    def test_normal_path_allowed(self, tmp_path):
        """일반 경로가 허용되는지 확인한다."""
        # resolve된 경로로 테스트 (Windows 심볼릭 링크 대응)
        test_file = tmp_path / "test.py"
        test_file.write_text("hello")
        resolved_tmp = str(tmp_path.resolve())
        resolved_file = str(test_file.resolve())
        pg = PathGuard()
        safe, reason = pg.is_path_safe(resolved_file, resolved_tmp)
        assert safe is True, f"safe={safe}, reason={reason}"

    def test_traversal_blocked(self, tmp_path):
        """경로 순회 공격이 차단되는지 확인한다."""
        pg = PathGuard()
        safe, _ = pg.is_path_safe(
            str(tmp_path / ".." / ".." / "etc" / "shadow"),
            str(tmp_path),
        )
        assert safe is False


class TestCommandFilter:
    """CommandFilter 테스트."""

    def test_rm_rf_root_blocked(self):
        """rm -rf /가 차단되는지 확인한다."""
        cf = CommandFilter()
        safe, severity, _ = cf.check_command("rm -rf /")
        assert safe is False
        assert severity == "critical"

    def test_curl_blocked(self):
        """curl이 차단되는지 확인한다 (에어갭 위반)."""
        cf = CommandFilter()
        safe, _, _ = cf.check_command("curl http://example.com")
        assert safe is False

    def test_safe_command_allowed(self):
        """안전한 명령어가 허용되는지 확인한다."""
        cf = CommandFilter()
        safe, _, _ = cf.check_command("ls -la")
        assert safe is True

    def test_git_allowed(self):
        """git 명령어가 허용되는지 확인한다."""
        cf = CommandFilter()
        safe, _, _ = cf.check_command("git status")
        assert safe is True

    def test_python_allowed(self):
        """python 명령어가 허용되는지 확인한다."""
        cf = CommandFilter()
        safe, _, _ = cf.check_command("python -m pytest")
        assert safe is True
