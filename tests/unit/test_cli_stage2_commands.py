# Stage 2 슬래시 명령 검증 — /cost·/save·/diff·/copy.
"""
CLI Stage 2 배치1 명령들의 계약을 고정한다 (2026-08-05).

  - /cost : 세션 누적 토큰을 표로. 과금이 아니라 토큰임을 명시한다.
  - /save : 트랜스크립트를 사람이 읽는 Markdown으로 내보낸다.
  - /diff : 로컬 git 변경사항(에어갭 안전). 리포가 아니면 안내만.
  - /copy : 마지막 응답을 복사(pyperclip → OSC52 폴백).
모든 명령은 실패해도 세션을 끊지 않는다.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

from rich.console import Console

from cli.repl import NexusREPL


def _make_repl(tmp_path=None, *, with_state: bool = True):
    repl = NexusREPL.__new__(NexusREPL)
    buf = io.StringIO()
    repl.console = Console(file=buf, force_terminal=False, width=200)
    repl._last_response = ""
    if with_state:
        repl._state = SimpleNamespace(
            session_id="abcdef12-0000-0000-0000-000000000000",
            cwd=str(tmp_path) if tmp_path else ".",
            config=SimpleNamespace(sessions_dir=str(tmp_path) if tmp_path else "."),
            get_session_summary=lambda: {
                "turns": 3,
                "total_input_tokens": 1200,
                "total_output_tokens": 340,
                "total_tool_calls": 2,
                "total_duration_seconds": 12.5,
            },
        )
    else:
        repl._state = None
    return repl, buf


class TestCost:
    def test_shows_token_totals(self) -> None:
        """누적 토큰과 합계를 보여주고, 과금이 없음을 알린다."""
        repl, buf = _make_repl()
        asyncio.run(repl._cmd_cost([]))
        out = buf.getvalue()
        assert "1,200" in out and "340" in out
        assert "1,540" in out  # 합계
        assert "과금은 없습니다" in out

    def test_without_state_is_safe(self) -> None:
        """부트스트랩 전이면 안내만 하고 예외를 내지 않는다."""
        repl, buf = _make_repl(with_state=False)
        asyncio.run(repl._cmd_cost([]))
        assert "초기화되지 않았습니다" in buf.getvalue()


class TestSave:
    def test_writes_markdown(self, tmp_path) -> None:
        """트랜스크립트를 Markdown으로 저장한다(역할 라벨 포함)."""
        sid = "abcdef12-0000-0000-0000-000000000000"
        d = tmp_path / "cli" / sid
        d.mkdir(parents=True)
        (d / "transcript.jsonl").write_text(
            '{"role":"user","content":"안녕"}\n{"role":"assistant","content":"반가워요"}\n',
            encoding="utf-8",
        )
        repl, buf = _make_repl(tmp_path)
        target = tmp_path / "out.md"

        asyncio.run(repl._cmd_save([str(target)]))

        text = target.read_text(encoding="utf-8")
        assert "# Nexus 대화 기록" in text
        assert "## 사용자" in text and "안녕" in text
        assert "## NOVA" in text and "반가워요" in text
        assert "저장했습니다" in buf.getvalue()

    def test_no_transcript_is_reported(self, tmp_path) -> None:
        """저장할 대화가 없으면 그 사실만 알린다(파일을 만들지 않는다)."""
        repl, buf = _make_repl(tmp_path)
        asyncio.run(repl._cmd_save([str(tmp_path / "none.md")]))
        assert "저장할 대화가 없습니다" in buf.getvalue()
        assert not (tmp_path / "none.md").exists()


class TestDiff:
    def test_non_repo_reports_gracefully(self, tmp_path) -> None:
        """git 저장소가 아니면 오류를 던지지 않고 안내만 한다."""
        repl, buf = _make_repl(tmp_path)
        asyncio.run(repl._cmd_diff([]))
        out = buf.getvalue()
        # git 미설치·비저장소 어느 쪽이든 사용자에게 문장으로 알린다.
        assert out.strip() != ""


class TestCopy:
    def test_empty_response_reported(self) -> None:
        """복사할 응답이 없으면 알린다."""
        repl, buf = _make_repl()
        asyncio.run(repl._cmd_copy([]))
        assert "복사할 응답이 없습니다" in buf.getvalue()

    def test_copies_last_response(self, monkeypatch) -> None:
        """마지막 응답을 클립보드로 보낸다(pyperclip 경로)."""
        repl, buf = _make_repl()
        repl._last_response = "복사 대상 텍스트"
        copied: list[str] = []

        import sys
        import types

        fake = types.ModuleType("pyperclip")
        fake.copy = lambda t: copied.append(t)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pyperclip", fake)

        asyncio.run(repl._cmd_copy([]))

        assert copied == ["복사 대상 텍스트"]
        assert "복사했습니다" in buf.getvalue()

    def test_falls_back_to_osc52(self, monkeypatch) -> None:
        """pyperclip이 없으면 OSC52 이스케이프로 폴백한다(원격 세션 대응)."""
        repl, buf = _make_repl()
        repl._last_response = "원격 복사"

        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "pyperclip":
                raise ImportError("no pyperclip")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        written: list[str] = []
        monkeypatch.setattr("sys.stdout.write", lambda t: written.append(t))
        monkeypatch.setattr("sys.stdout.flush", lambda: None)

        asyncio.run(repl._cmd_copy([]))

        assert any("\033]52;c;" in w for w in written)
        assert "OSC52" in buf.getvalue()
