# cli/repl.py prompt_permission_v2 검증 — 모드 인지형 numbered 권한 프롬프트(A1/A3).
"""
NexusREPL.prompt_permission_v2 의 4가지 핵심 동작을 검증한다.

1) accept_edits 모드: FILE_WRITE 부류(Write/Edit/MultiEdit/NotebookEdit)는
   프롬프트 없이 즉시 승인, Bash·미지의 도구는 여전히 프롬프트를 띄운다(fail-closed).
2) numbered 선택: 1=예 / 2=예(세션 항상 허용) / 3=아니오+피드백.
3) 세션 allow 목록: 2번 선택 후 같은 도구는 재프롬프트 없이 자동 승인.
   단 Bash·미지의 도구는 목록에 등록되지 않는다(이번 1회만 허용).
4) 취소(Ctrl+C/EOF): 안전한 쪽(거부)으로 떨어진다.

REPL 생성은 test_cli_repl.py 패턴을 따른다 — __init__ 이 PromptSession을 만들며
실콘솔을 요구하므로 __new__ 로 우회하고 필요한 필드만 직접 심는다.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from rich.console import Console

from cli.repl import NexusREPL


class _FakePromptSession:
    """미리 정해둔 응답을 순서대로 돌려주는 가짜 prompt-toolkit 세션."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.prompts_shown: list[str] = []  # 어떤 프롬프트가 표시됐는지 기록

    def prompt(self, text: str = "") -> str:
        self.prompts_shown.append(text)
        if not self._answers:
            raise AssertionError("예상보다 많은 프롬프트 호출")
        return self._answers.pop(0)


def _make_repl(mode: str, answers: list[str] | None = None):
    """권한 모드와 프롬프트 응답 시나리오를 심은 REPL 껍데기를 만든다."""
    repl = NexusREPL.__new__(NexusREPL)
    repl._permission_mode = mode
    repl._session_allow = set()
    # A4(2026-08-05): Bash는 도구명이 아니라 명령 프리픽스로 등록된다.
    repl._bash_allow_prefixes = set()
    repl._tool_ctx = None  # CommandFilter를 자체 생성하는 경로를 태운다
    buf = io.StringIO()
    repl.console = Console(file=buf, force_terminal=False, width=200)
    fake_session = _FakePromptSession(answers or [])
    repl._prompt_session = fake_session
    return repl, buf, fake_session


# ─── ① accept_edits 모드 인지 ───


@pytest.mark.parametrize("tool_name", ["Write", "Edit", "MultiEdit", "NotebookEdit"])
def test_accept_edits_file_write_auto_approved(tool_name: str) -> None:
    """accept_edits 모드에서 파일 수정 부류는 프롬프트 없이 즉시 승인된다."""
    repl, buf, session = _make_repl("accept_edits")

    result = asyncio.run(repl.prompt_permission_v2(tool_name, "파일 수정", {}))

    assert result["approved"] is True
    assert session.prompts_shown == []  # 프롬프트를 띄우지 않았다
    assert "자동 승인" in buf.getvalue()  # 1줄 안내는 남긴다


def test_accept_edits_bash_still_prompts() -> None:
    """accept_edits 모드에서도 Bash는 자동 승인되지 않고 프롬프트를 띄운다."""
    repl, _, session = _make_repl("accept_edits", answers=["1"])

    result = asyncio.run(repl.prompt_permission_v2("Bash", "명령 실행", {}))

    assert result["approved"] is True  # 사용자가 1(예)을 골랐으므로 허용
    assert len(session.prompts_shown) == 1  # 프롬프트가 실제로 떴다


def test_accept_edits_unknown_tool_still_prompts() -> None:
    """accept_edits 모드에서 분류표에 없는 미지의 도구는 자동 승인하지 않는다(fail-closed)."""
    repl, _, session = _make_repl("accept_edits", answers=["1"])

    result = asyncio.run(repl.prompt_permission_v2("UnknownTool", "뭔가 실행", {}))

    assert result["approved"] is True
    assert len(session.prompts_shown) == 1  # 자동 승인 아님 — 반드시 물어봄


def test_default_mode_file_write_prompts() -> None:
    """default 모드에서는 파일 수정 부류도 종전대로 프롬프트를 띄운다."""
    repl, _, session = _make_repl("default", answers=["1"])

    result = asyncio.run(repl.prompt_permission_v2("Write", "파일 쓰기", {}))

    assert result["approved"] is True
    assert len(session.prompts_shown) == 1


# ─── ② numbered 선택 ───


def test_choice_3_denies_with_feedback() -> None:
    """3(아니오) 선택 시 피드백을 입력받아 거부 응답에 담는다."""
    repl, _, session = _make_repl("default", answers=["3", "그 파일 말고 config만 고쳐"])

    result = asyncio.run(repl.prompt_permission_v2("Write", "파일 쓰기", {}))

    assert result["approved"] is False
    assert result["feedback"] == "그 파일 말고 config만 고쳐"
    assert len(session.prompts_shown) == 2  # 선택 1회 + 피드백 1회


def test_unrecognized_input_treated_as_deny() -> None:
    """1/2/3 이외의 입력은 거부로 처리된다(fail-closed) — 피드백 프롬프트로 이어진다."""
    repl, _, _ = _make_repl("default", answers=["뭐라고?", ""])

    result = asyncio.run(repl.prompt_permission_v2("Write", "파일 쓰기", {}))

    assert result["approved"] is False
    assert result["feedback"] == ""


# ─── ③ 세션 항상 허용 ───


def test_choice_2_registers_session_allow_and_skips_next_prompt() -> None:
    """2번 선택으로 등록된 도구는 다음 요청부터 프롬프트 없이 자동 승인된다."""
    repl, _, session = _make_repl("default", answers=["2"])

    first = asyncio.run(repl.prompt_permission_v2("Write", "파일 쓰기", {}))
    second = asyncio.run(repl.prompt_permission_v2("Write", "파일 또 쓰기", {}))

    assert first["approved"] is True
    assert first["always_allow"] is True
    assert "Write" in repl._session_allow
    assert second["approved"] is True
    assert len(session.prompts_shown) == 1  # 두 번째는 프롬프트 없이 통과


def test_choice_2_bash_registers_command_prefix() -> None:
    """Bash는 2번 선택 시 도구명이 아니라 **명령 프리픽스**로 등록된다(A4).

    등록 후 같은 계열의 안전한 명령은 프롬프트 없이 통과하고, 도구명(Bash)은
    allow 목록에 들어가지 않는다 — 도구 단위로 풀면 모든 명령이 통과하기 때문.
    """
    repl, _, session = _make_repl("default", answers=["2"])

    first = asyncio.run(
        repl.prompt_permission_v2("Bash", "명령 실행", {"command": "git status"})
    )
    second = asyncio.run(
        repl.prompt_permission_v2("Bash", "명령 실행", {"command": "git diff HEAD"})
    )

    assert first["approved"] is True
    assert first["always_allow"] is True
    assert "git" in repl._bash_allow_prefixes
    assert "Bash" not in repl._session_allow  # 도구 단위 등록은 여전히 금지
    assert second["approved"] is True
    assert len(session.prompts_shown) == 1  # 두 번째는 자동 승인


def test_choice_2_bash_unsafe_command_not_registered() -> None:
    """복합·위험 명령은 2번을 선택해도 등록되지 않고 1회성 허용에 그친다."""
    repl, _, session = _make_repl("default", answers=["2", "1"])

    first = asyncio.run(
        repl.prompt_permission_v2("Bash", "명령 실행", {"command": "git status && rm -rf /"})
    )
    # 등록되지 않았으므로 두 번째도 다시 물어본다.
    asyncio.run(
        repl.prompt_permission_v2("Bash", "명령 실행", {"command": "git status && rm -rf /"})
    )

    assert first["approved"] is True
    assert first["always_allow"] is False
    assert repl._bash_allow_prefixes == set()
    assert len(session.prompts_shown) == 2


# ─── ④ 취소 안전 처리 ───


def test_cancel_during_prompt_denies() -> None:
    """프롬프트 도중 Ctrl+C(KeyboardInterrupt)는 안전하게 거부로 처리된다."""
    repl, _, _ = _make_repl("default")

    class _InterruptSession:
        def prompt(self, text: str = "") -> str:
            raise KeyboardInterrupt

    repl._prompt_session = _InterruptSession()

    result = asyncio.run(repl.prompt_permission_v2("Write", "파일 쓰기", {}))

    assert result["approved"] is False
    assert result["feedback"] == ""
