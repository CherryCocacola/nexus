# `!` bash 패스스루(D1)·상태줄(C2) 검증 — 보안 게이트가 핵심.
"""
`!<명령>` 패스스루의 안전 계약을 고정한다 (2026-08-05).

[왜 게이트가 여러 겹인가]
  이 기능은 모델을 거치지 않고 사용자가 직접 셸을 실행하는 통로다. 권한
  파이프라인을 우회하므로, 도구 실행과 같은 수준의 방어가 필요하다.
    ① plan/deny_all 모드에서는 아예 실행하지 않는다(그 모드의 의미가 부작용 금지)
    ② 권한 파이프라인이 쓰는 CommandFilter로 위험 명령을 차단한다
    ③ 허용/차단 무관하게 감사 로그를 남긴다
    ④ v1은 표시 전용 — 결과가 대화 맥락에 들어가지 않는다
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

from rich.console import Console

from cli.repl import NexusREPL


class _FakeAudit:
    """감사 기록을 메모리에 모으는 대역."""

    def __init__(self) -> None:
        self.entries: list = []

    def log_decision(self, entry) -> None:
        self.entries.append(entry)


def _make_repl(mode: str = "default", *, tmp_path=None):
    repl = NexusREPL.__new__(NexusREPL)
    repl._permission_mode = mode
    buf = io.StringIO()
    repl.console = Console(file=buf, force_terminal=False, width=200)
    audit = _FakeAudit()
    repl._tool_ctx = SimpleNamespace(options={"audit_logger": audit})
    repl._state = SimpleNamespace(
        session_id="s-1",
        cwd=str(tmp_path) if tmp_path else ".",
        get_session_summary=lambda: {
            "session_id": "s-1",
            "total_input_tokens": 10,
            "total_output_tokens": 20,
        },
    )
    return repl, buf, audit


class TestModeGate:
    def test_plan_mode_blocks_execution(self) -> None:
        """plan 모드에서는 실행하지 않는다(부작용 금지 모드의 우회 통로 차단)."""
        repl, buf, audit = _make_repl("plan")

        asyncio.run(repl._run_bash_passthrough("ls"))

        assert "plan 모드에서는" in buf.getvalue()
        assert audit.entries and audit.entries[0].decision == "deny"

    def test_deny_all_mode_blocks_execution(self) -> None:
        """deny_all 모드도 동일하게 차단한다."""
        repl, buf, audit = _make_repl("deny_all")
        asyncio.run(repl._run_bash_passthrough("ls"))
        assert "deny_all 모드에서는" in buf.getvalue()
        assert audit.entries[0].decision == "deny"


class TestCommandFilterGate:
    def test_dangerous_command_blocked(self) -> None:
        """위험 명령은 CommandFilter가 걸러 실행되지 않는다."""
        repl, buf, audit = _make_repl("default")

        asyncio.run(repl._run_bash_passthrough("rm -rf /"))

        out = buf.getvalue()
        assert "차단됨" in out
        assert audit.entries and audit.entries[0].decision == "deny"

    def test_empty_command_reported(self) -> None:
        """빈 명령은 사용법을 안내하고 아무것도 실행하지 않는다."""
        repl, buf, audit = _make_repl("default")
        asyncio.run(repl._run_bash_passthrough("   "))
        assert "실행할 명령이 없습니다" in buf.getvalue()
        assert audit.entries == []


class TestExecution:
    def test_safe_command_runs_and_is_audited(self, tmp_path) -> None:
        """안전한 명령은 실행되고, 허용 기록이 감사 로그에 남는다."""
        repl, buf, audit = _make_repl("default", tmp_path=tmp_path)

        asyncio.run(repl._run_bash_passthrough("echo nexus-passthrough-ok"))

        out = buf.getvalue()
        assert "nexus-passthrough-ok" in out
        assert "exit=0" in out
        assert audit.entries and audit.entries[0].decision == "allow"
        assert audit.entries[0].tool_name == "!bash"

    def test_display_only_notice_shown(self, tmp_path) -> None:
        """결과가 대화 맥락에 포함되지 않는다는 점을 매번 알린다(v1 한계 명시)."""
        repl, buf, _ = _make_repl("default", tmp_path=tmp_path)
        asyncio.run(repl._run_bash_passthrough("echo hi"))
        assert "대화 맥락에 포함되지 않습니다" in buf.getvalue()

    def test_audit_failure_does_not_break_execution(self, tmp_path) -> None:
        """감사 로거가 없어도(또는 실패해도) 명령 실행은 정상 진행된다."""
        repl, buf, _ = _make_repl("default", tmp_path=tmp_path)
        repl._tool_ctx = None  # 감사 로거 없음

        asyncio.run(repl._run_bash_passthrough("echo still-works"))

        assert "still-works" in buf.getvalue()


class TestBottomToolbar:
    def test_shows_mode_and_tokens(self) -> None:
        """상태줄에 권한 모드·토큰·세션이 나온다."""
        repl, _, _ = _make_repl("accept_edits")
        bar = repl._bottom_toolbar()
        assert "accept_edits" in bar and "10" in bar and "20" in bar

    def test_safe_before_bootstrap(self) -> None:
        """부트스트랩 전에도 예외 없이 문자열을 돌려준다(프롬프트가 떠야 하므로)."""
        repl, _, _ = _make_repl("default")
        repl._state = None
        assert "초기화 중" in repl._bottom_toolbar()

    def test_never_raises_on_broken_state(self) -> None:
        """상태 객체가 깨져 있어도 예외를 내지 않는다(입력 불가 방지)."""
        repl, _, _ = _make_repl("default")
        repl._state = SimpleNamespace(get_session_summary=lambda: 1 / 0)
        assert isinstance(repl._bottom_toolbar(), str)
