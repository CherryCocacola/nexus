"""
세션 영속화 단위 테스트 — Ch 16 (2026-04-21).

검증 대상:
  1. SessionTranscript
     - 경로 규약: {sessions_dir}/{session_id}/transcript.jsonl
     - append_entry: JSON Lines 형식, append-only
     - enabled=False: I/O 없음
     - list_transcript_sessions: 최근 수정 시각 기준 정렬
  2. ShortTermMemory.list_sessions (인메모리 폴백 경로)
  3. QueryEngine._finalize_turn
     - MemoryManager.on_turn_end 호출
     - Transcript.append_entry user/assistant 쌍 기록
     - 예외 발생 시 swallow (본류 응답 영향 없음)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.memory.short_term import ShortTermMemory
from core.memory.transcript import SessionTranscript, list_transcript_sessions
from core.message import Message


# ─────────────────────────────────────────────
# SessionTranscript
# ─────────────────────────────────────────────
def test_transcript_creates_dir_and_appends(tmp_path: Path) -> None:
    """append_entry가 JSONL 라인을 파일에 추가한다."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="s1", enabled=True)
    t.append_entry(role="user", content="hello", turn=1)
    t.append_entry(
        role="assistant", content="hi", turn=1,
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    path = tmp_path / "s1" / "transcript.jsonl"
    assert path.exists(), "transcript 파일이 생성되어야 한다"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["role"] == "user"
    assert first["content"] == "hello"
    assert first["turn"] == 1
    assert first["session_id"] == "s1"

    second = json.loads(lines[1])
    assert second["role"] == "assistant"
    assert second["usage"]["input_tokens"] == 10


def test_transcript_disabled_writes_nothing(tmp_path: Path) -> None:
    """enabled=False면 파일을 만들지 않고 append도 하지 않는다."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="s2", enabled=False)
    t.append_entry(role="user", content="hello", turn=1)
    assert not (tmp_path / "s2").exists()


def test_transcript_empty_content_skipped(tmp_path: Path) -> None:
    """빈 content는 기록하지 않는다 (공백 턴 오염 방지)."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="s3", enabled=True)
    t.append_entry(role="user", content="", turn=1)
    path = tmp_path / "s3" / "transcript.jsonl"
    if path.exists():
        assert path.read_text(encoding="utf-8") == ""


def test_list_transcript_sessions_sorted_by_mtime(tmp_path: Path) -> None:
    """list_transcript_sessions는 최근 수정 순으로 세션을 반환한다."""
    # 세션 3개 생성 — 각기 다른 시점에 마지막 기록
    for sid in ("a", "b", "c"):
        t = SessionTranscript(sessions_dir=tmp_path, session_id=sid, enabled=True)
        t.append_entry(role="user", content=f"q-{sid}", turn=1)

    # 세션 b만 최신 내용을 덮어써서 최종 수정 시각 갱신
    import time as _t
    _t.sleep(0.02)
    t_b = SessionTranscript(sessions_dir=tmp_path, session_id="b", enabled=True)
    t_b.append_entry(role="assistant", content="updated", turn=2)

    out = list_transcript_sessions(tmp_path, limit=10)
    ids = [s["session_id"] for s in out]
    assert "b" in ids and "a" in ids and "c" in ids
    # 가장 최근 수정된 b가 목록 앞쪽
    assert ids[0] == "b"
    # entries 카운트가 파일 라인 수와 일치
    b_entry = next(s for s in out if s["session_id"] == "b")
    assert b_entry["entries"] == 2


def test_list_transcript_sessions_empty_dir(tmp_path: Path) -> None:
    """디렉토리 자체가 비어 있으면 빈 리스트."""
    assert list_transcript_sessions(tmp_path / "nonexistent") == []
    assert list_transcript_sessions(tmp_path) == []


# ─────────────────────────────────────────────
# ShortTermMemory.list_sessions (인메모리 폴백)
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_short_term_list_sessions_inmemory() -> None:
    """Redis 없이 save_conversation_context → list_sessions가 session_id를 반환한다."""
    stm = ShortTermMemory(redis_client=None)
    await stm.save_conversation_context("sess-a", [{"role": "user", "content": "hi"}])
    await stm.save_conversation_context("sess-b", [{"role": "user", "content": "hi"}])
    # 무관한 키도 넣어서 접두 필터가 동작하는지 확인
    await stm.set("tool_cache:Read:xyz", "cached")

    sessions = await stm.list_sessions(limit=10)
    assert set(sessions) == {"sess-a", "sess-b"}


@pytest.mark.asyncio
async def test_short_term_list_sessions_respects_limit() -> None:
    """limit 이상은 반환하지 않는다."""
    stm = ShortTermMemory(redis_client=None)
    for i in range(5):
        await stm.save_conversation_context(f"s-{i}", [])
    out = await stm.list_sessions(limit=3)
    assert len(out) <= 3


# ─────────────────────────────────────────────
# QueryEngine._finalize_turn — MemoryManager + Transcript 호출
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_finalize_turn_invokes_memory_and_transcript(tmp_path: Path) -> None:
    """턴 종료 시 memory_manager.on_turn_end와 transcript.append_entry가 호출된다."""
    from core.orchestrator.query_engine import QueryEngine
    from core.tools.base import ToolUseContext

    fake_mm = MagicMock()
    fake_mm.on_turn_end = AsyncMock()
    fake_transcript = MagicMock()

    engine = QueryEngine(
        model_provider=MagicMock(),
        tools=[],
        context=ToolUseContext(cwd=".", session_id="sess-test"),
        system_prompt="",
        memory_manager=fake_mm,
        transcript=fake_transcript,
    )
    # messages에 user/assistant 쌍 주입
    engine._messages = [
        Message.user("질문"),
        Message.assistant("답변"),
    ]
    engine._total_turns = 1

    await engine._finalize_turn("질문")

    fake_mm.on_turn_end.assert_awaited_once()
    kwargs = fake_mm.on_turn_end.await_args.kwargs
    assert kwargs["session_id"] == "sess-test"
    assert kwargs["messages"] == engine._messages

    # user + assistant 각 1회씩 append_entry 호출
    assert fake_transcript.append_entry.call_count == 2
    calls = [c.kwargs for c in fake_transcript.append_entry.call_args_list]
    assert calls[0]["role"] == "user" and calls[0]["content"] == "질문"
    assert calls[1]["role"] == "assistant" and calls[1]["content"] == "답변"


@pytest.mark.asyncio
async def test_finalize_turn_swallows_memory_errors(tmp_path: Path) -> None:
    """memory_manager 호출이 실패해도 예외가 올라오지 않는다."""
    from core.orchestrator.query_engine import QueryEngine
    from core.tools.base import ToolUseContext

    fake_mm = MagicMock()
    fake_mm.on_turn_end = AsyncMock(side_effect=RuntimeError("redis down"))

    engine = QueryEngine(
        model_provider=MagicMock(),
        tools=[],
        context=ToolUseContext(cwd=".", session_id="sess-err"),
        system_prompt="",
        memory_manager=fake_mm,
        transcript=None,
    )
    engine._messages = [Message.user("q"), Message.assistant("a")]

    # 예외가 propagate되지 않아야 한다
    await engine._finalize_turn("q")
    fake_mm.on_turn_end.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_turn_no_memory_manager_is_noop() -> None:
    """memory_manager/transcript가 둘 다 None이면 조용히 아무 것도 하지 않는다."""
    from core.orchestrator.query_engine import QueryEngine
    from core.tools.base import ToolUseContext

    engine = QueryEngine(
        model_provider=MagicMock(),
        tools=[],
        context=ToolUseContext(cwd=".", session_id="sess-none"),
        system_prompt="",
        memory_manager=None,
        transcript=None,
    )
    engine._messages = [Message.user("q")]
    # 예외 없어야 함
    await engine._finalize_turn("q")


# ─────────────────────────────────────────────
# SessionTranscript + QueryEngine 통합 — 실제 파일 기록
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_finalize_turn_writes_real_transcript_file(tmp_path: Path) -> None:
    """QueryEngine → 실제 SessionTranscript 인스턴스 → 디스크 파일로 기록이 남는다."""
    from core.orchestrator.query_engine import QueryEngine
    from core.tools.base import ToolUseContext

    trans = SessionTranscript(sessions_dir=tmp_path, session_id="sess-disk", enabled=True)
    engine = QueryEngine(
        model_provider=MagicMock(),
        tools=[],
        context=ToolUseContext(cwd=".", session_id="sess-disk"),
        system_prompt="",
        memory_manager=None,
        transcript=trans,
    )
    engine._messages = [Message.user("hello"), Message.assistant("world")]
    engine._total_turns = 1
    await engine._finalize_turn("hello")

    path = tmp_path / "sess-disk" / "transcript.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["content"] == "hello"
    assert json.loads(lines[1])["content"] == "world"
