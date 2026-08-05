# Bash 세션 allow-list(A4) 검증 — 프리픽스 등록 조건과 자동 승인 안전성.
"""
NexusREPL의 Bash "이 세션 항상 허용" 계약을 고정한다 (2026-08-05).

[왜 도구명이 아니라 명령 프리픽스인가]
  Bash를 도구명 단위로 풀면 이후 **어떤 명령이든** 확인 없이 실행된다. 그래서
  첫 토큰(`git`, `ls` …) 단위로만 등록하고, 등록·사용 양쪽 시점에 CommandFilter
  검사를 다시 통과해야 한다.

[반드시 지켜야 하는 안전 계약]
  - 셸 메타문자(; | & ` $( > < 개행)가 있으면 등록도 자동 승인도 하지 않는다.
    "git status && rm -rf /" 처럼 첫 토큰이 안전해도 뒤가 위험할 수 있기 때문.
  - 위험 명령(rm 등)은 CommandFilter가 걸러 등록되지 않는다.
  - 경로가 섞인 실행(./x, /bin/x)은 이름만으로 동일성을 보장할 수 없어 제외.
  - 등록된 프리픽스라도 이번 명령이 위험하면 자동 승인하지 않는다.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from cli.repl import NexusREPL


def _make_repl():
    """allow-list 판정에 필요한 필드만 심은 REPL을 만든다."""
    repl = NexusREPL.__new__(NexusREPL)
    repl._bash_allow_prefixes = set()
    repl._session_allow = set()
    repl._tool_ctx = None  # 파이프라인 없음 → CommandFilter를 자체 생성하는 경로
    repl.console = Console(file=io.StringIO(), force_terminal=False, width=200)
    return repl


class TestPrefixRegistration:
    @pytest.mark.parametrize("command", ["git status", "ls -la", "cat README.md"])
    def test_safe_simple_commands_registrable(self, command: str) -> None:
        """안전 목록에 있는 단순 명령은 첫 토큰으로 등록 가능하다."""
        repl = _make_repl()
        assert repl._bash_prefix_if_registrable(command) == command.split()[0]

    @pytest.mark.parametrize(
        "command",
        [
            "git status && rm -rf /",
            "ls; rm -rf /tmp",
            "cat f | sh",
            "echo `whoami`",
            "echo $(rm -rf /)",
            "ls > /etc/passwd",
            "ls & sleep 1",
            "git status\nrm -rf /",
        ],
    )
    def test_shell_metachars_blocked(self, command: str) -> None:
        """셸 메타문자가 있으면 첫 토큰이 안전해도 등록 불가(fail-closed)."""
        repl = _make_repl()
        assert repl._bash_prefix_if_registrable(command) is None

    @pytest.mark.parametrize(
        "command", ["rm -rf /", "sudo rm -rf /var", "dd if=/dev/zero of=/dev/sda"]
    )
    def test_dangerous_commands_not_registrable(self, command: str) -> None:
        """CommandFilter가 위험하다고 판정한 명령은 등록되지 않는다."""
        repl = _make_repl()
        assert repl._bash_prefix_if_registrable(command) is None

    @pytest.mark.parametrize("command", ["./deploy.sh", "/usr/bin/env python", "..\\x.bat"])
    def test_path_qualified_commands_excluded(self, command: str) -> None:
        """경로가 섞인 실행은 이름만으로 동일성을 보장할 수 없어 제외한다."""
        repl = _make_repl()
        assert repl._bash_prefix_if_registrable(command) is None

    @pytest.mark.parametrize("command", ["", "   ", None])
    def test_empty_or_invalid_returns_none(self, command) -> None:
        """빈 명령·비문자열은 안전하게 None을 돌려준다."""
        repl = _make_repl()
        assert repl._bash_prefix_if_registrable(command) is None


class TestAutoAllow:
    def test_registered_prefix_auto_allows_same_command_family(self) -> None:
        """등록된 프리픽스의 안전한 명령은 자동 승인된다."""
        repl = _make_repl()
        repl._bash_allow_prefixes.add("git")

        assert repl._is_bash_autoallowed("git status") is True
        assert repl._is_bash_autoallowed("git diff HEAD") is True

    def test_unregistered_prefix_not_allowed(self) -> None:
        """등록되지 않은 프리픽스는 자동 승인되지 않는다."""
        repl = _make_repl()
        repl._bash_allow_prefixes.add("git")

        assert repl._is_bash_autoallowed("ls -la") is False

    def test_registered_prefix_with_metachars_not_allowed(self) -> None:
        """등록된 프리픽스라도 뒤에 위험한 명령이 붙으면 자동 승인하지 않는다.

        이 테스트가 A4의 핵심 안전 계약이다 — 등록만 믿고 통과시키면
        `git status && rm -rf /` 가 확인 없이 실행된다.
        """
        repl = _make_repl()
        repl._bash_allow_prefixes.add("git")

        assert repl._is_bash_autoallowed("git status && rm -rf /") is False
        assert repl._is_bash_autoallowed("git log | sh") is False

    def test_empty_allowlist_never_auto_allows(self) -> None:
        """등록된 것이 없으면 어떤 명령도 자동 승인되지 않는다."""
        repl = _make_repl()
        assert repl._is_bash_autoallowed("git status") is False
