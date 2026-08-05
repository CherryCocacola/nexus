# /compact(D2)·/resume(D3) 검증 — 강제 압축과 세션 재바인드.
"""
CLI Stage 2 배치3 계약을 고정한다 (2026-08-05).

  - /compact: ContextManager의 강제 압축(force=True)을 태우고 결과를 엔진
    히스토리에 **in-place로** 반영한다(다른 곳이 든 참조가 어긋나지 않도록).
  - /resume: 이전 CLI 세션을 골라 메시지를 복원하되, **session_id까지 재바인드**
    한다. 그러지 않으면 이어서 한 대화가 새 세션에 쌓여 원본과 갈라진다.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

from rich.console import Console

from cli.repl import NexusREPL
from core.message import Message


class _FakeCM:
    """강제 압축만 흉내내는 ContextManager 대역."""

    def __init__(self) -> None:
        self.forced: list[bool] = []

    def _estimate_tokens(self, messages) -> int:
        return sum(len(str(getattr(m, "content", ""))) for m in messages)

    async def auto_compact_if_needed(self, messages, force: bool = False):
        self.forced.append(force)
        return [Message.user("요약본")]


class _FakeEngine:
    def __init__(self, messages=None, cm=None) -> None:
        self._messages = messages if messages is not None else []
        self._context_manager = cm
        self.bound: list[dict] = []

    def bind_request(self, **kwargs) -> None:
        self.bound.append(kwargs)


def _make_repl(engine=None, tmp_path=None):
    repl = NexusREPL.__new__(NexusREPL)
    buf = io.StringIO()
    repl.console = Console(file=buf, force_terminal=False, width=200)
    repl._query_engine = engine
    repl._state = SimpleNamespace(
        session_id="current-session",
        config=SimpleNamespace(sessions_dir=str(tmp_path) if tmp_path else "."),
    )
    return repl, buf


class TestCompact:
    def test_forces_compaction_and_replaces_in_place(self) -> None:
        """강제 압축을 호출하고 결과를 같은 리스트 객체에 반영한다."""
        cm = _FakeCM()
        msgs = [Message.user("a" * 50), Message.assistant(text="b" * 50)]
        engine = _FakeEngine(msgs, cm)
        repl, buf = _make_repl(engine)

        asyncio.run(repl._cmd_compact([]))

        assert cm.forced == [True]  # force=True로 호출됐다
        assert msgs is engine._messages  # 같은 객체를 유지(in-place)
        assert len(msgs) == 1 and "요약본" in str(msgs[0].content)
        out = buf.getvalue()
        assert "컨텍스트 압축" in out and "절약" in out

    def test_no_engine_is_safe(self) -> None:
        """부트스트랩 전이면 안내만 한다."""
        repl, buf = _make_repl(None)
        asyncio.run(repl._cmd_compact([]))
        assert "초기화되지 않았습니다" in buf.getvalue()

    def test_empty_history_reported(self) -> None:
        """대화가 없으면 압축하지 않는다."""
        engine = _FakeEngine([], _FakeCM())
        repl, buf = _make_repl(engine)
        asyncio.run(repl._cmd_compact([]))
        assert "압축할 대화가 없습니다" in buf.getvalue()

    def test_compaction_failure_is_reported(self) -> None:
        """압축이 실패해도 예외를 밖으로 내지 않는다."""

        class _Broken(_FakeCM):
            async def auto_compact_if_needed(self, messages, force: bool = False):
                raise RuntimeError("모델 호출 실패")

        engine = _FakeEngine([Message.user("x")], _Broken())
        repl, buf = _make_repl(engine)
        asyncio.run(repl._cmd_compact([]))
        assert "압축 실패" in buf.getvalue()


class TestResume:
    def _seed(self, tmp_path, sid: str, text: str = "이전 대화") -> None:
        d = tmp_path / "cli" / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "transcript.jsonl").write_text(
            f'{{"role":"user","content":"{text}"}}\n'
            f'{{"role":"assistant","content":"답변"}}\n',
            encoding="utf-8",
        )

    def test_lists_previous_sessions(self, tmp_path) -> None:
        """인자 없이 부르면 이전 세션 목록을 보여 준다."""
        self._seed(tmp_path, "old-session-1")
        engine = _FakeEngine()
        repl, buf = _make_repl(engine, tmp_path)

        asyncio.run(repl._cmd_resume([]))

        out = buf.getvalue()
        assert "이전 CLI 세션" in out and "old-sess" in out
        assert engine.bound == []  # 목록만 — 복원하지 않는다

    def test_current_session_excluded(self, tmp_path) -> None:
        """지금 쓰고 있는 세션은 목록에서 제외한다."""
        self._seed(tmp_path, "current-session")
        engine = _FakeEngine()
        repl, buf = _make_repl(engine, tmp_path)

        asyncio.run(repl._cmd_resume([]))

        assert "이어받을 이전 세션이 없습니다" in buf.getvalue()

    def test_resume_rebinds_session_id(self, tmp_path) -> None:
        """번호를 주면 메시지를 복원하고 session_id까지 재바인드한다."""
        self._seed(tmp_path, "old-session-1")
        engine = _FakeEngine()
        repl, buf = _make_repl(engine, tmp_path)

        asyncio.run(repl._cmd_resume(["1"]))

        assert engine.bound, "bind_request가 호출되어야 한다"
        call = engine.bound[0]
        assert call["session_id"] == "old-session-1"
        assert call["channel"] == "cli"
        assert len(call["restore_messages"]) == 2
        # GlobalState의 세션 ID도 갈아끼워야 같은 트랜스크립트에 이어 쓴다.
        assert repl._state.session_id == "old-session-1"
        assert "이어받았습니다" in buf.getvalue()

    def test_invalid_index_reported(self, tmp_path) -> None:
        """범위를 벗어난 번호는 안내만 하고 아무것도 바꾸지 않는다."""
        self._seed(tmp_path, "old-session-1")
        engine = _FakeEngine()
        repl, buf = _make_repl(engine, tmp_path)

        asyncio.run(repl._cmd_resume(["99"]))

        assert "번호를 지정하세요" in buf.getvalue()
        assert engine.bound == []
        assert repl._state.session_id == "current-session"
