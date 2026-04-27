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
from core.memory.transcript import (
    SessionTranscript,
    delete_transcript_session,
    list_transcript_sessions,
    read_transcript_messages,
)
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
# read_transcript_messages (Ch 16 프론트 복원용)
# ─────────────────────────────────────────────
def test_read_transcript_messages_filters_by_role(tmp_path: Path) -> None:
    """기본 호출은 user/assistant만 반환하고 system/에러 엔트리는 제외한다."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="sx", enabled=True)
    t.append_entry(role="user", content="hi", turn=1)
    t.append_entry(role="assistant", content="hello", turn=1)
    t.append_entry(
        role="system", content="[stream aborted]", turn=2,
        extra={"error": "TimeoutError"},
    )
    t.append_entry(role="user", content="again", turn=3)

    msgs = read_transcript_messages(tmp_path, "sx")
    roles = [m["role"] for m in msgs]
    contents = [m["content"] for m in msgs]
    assert roles == ["user", "assistant", "user"]
    assert contents == ["hi", "hello", "again"]
    # 필드 보존
    assert msgs[0]["turn"] == 1
    assert msgs[0]["ts"] is not None


def test_read_transcript_messages_missing_file(tmp_path: Path) -> None:
    """파일이 없으면 빈 리스트 — 404 분기와 별개로 예외 없이 처리된다."""
    assert read_transcript_messages(tmp_path, "unknown-session") == []


def test_read_transcript_messages_limit_recent(tmp_path: Path) -> None:
    """limit 지정 시 가장 최근 N개만 반환 (오래된 것부터 순서 유지)."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="sy", enabled=True)
    for i in range(5):
        t.append_entry(role="user", content=f"q{i}", turn=i)
    msgs = read_transcript_messages(tmp_path, "sy", limit=2)
    assert [m["content"] for m in msgs] == ["q3", "q4"]


def test_read_transcript_messages_skips_corrupt_lines(tmp_path: Path) -> None:
    """손상된 JSON 라인은 조용히 스킵되고 정상 라인은 그대로 반환된다."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="sz", enabled=True)
    t.append_entry(role="user", content="ok1", turn=1)

    # 정상 라인 사이에 손상된 라인을 끼워 넣는다
    path = tmp_path / "sz" / "transcript.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write("THIS IS NOT JSON\n")

    t.append_entry(role="assistant", content="ok2", turn=1)

    msgs = read_transcript_messages(tmp_path, "sz")
    assert [m["content"] for m in msgs] == ["ok1", "ok2"]


def test_read_transcript_messages_custom_roles(tmp_path: Path) -> None:
    """roles 인자를 명시하면 system 엔트리도 포함된다."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="sw", enabled=True)
    t.append_entry(role="user", content="hi", turn=1)
    t.append_entry(role="system", content="[info]", turn=1)

    msgs = read_transcript_messages(
        tmp_path, "sw", roles=("user", "assistant", "system")
    )
    assert [m["role"] for m in msgs] == ["user", "system"]


# ─────────────────────────────────────────────
# list_transcript_sessions — title_hint (v0.14.5)
# ─────────────────────────────────────────────
def test_list_transcript_sessions_includes_title_hint(tmp_path: Path) -> None:
    """첫 user 메시지를 title_hint로 포함하여 사이드바에서 대화 식별이 가능하다."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="s1", enabled=True)
    t.append_entry(role="user", content="니체 철학에 대해 설명해줘", turn=1)
    t.append_entry(role="assistant", content="위버멘쉬...", turn=1)

    out = list_transcript_sessions(tmp_path, limit=10)
    assert len(out) == 1
    assert out[0]["title_hint"] == "니체 철학에 대해 설명해줘"


def test_list_transcript_sessions_title_hint_truncated(tmp_path: Path) -> None:
    """title_hint는 60자를 넘으면 말줄임표가 붙는다."""
    long_msg = "가" * 80
    t = SessionTranscript(sessions_dir=tmp_path, session_id="s2", enabled=True)
    t.append_entry(role="user", content=long_msg, turn=1)

    out = list_transcript_sessions(tmp_path, limit=10)
    hint = out[0]["title_hint"]
    assert hint is not None
    assert hint.endswith("…")
    # 60자 + 말줄임표 1자
    assert len(hint) == 61


def test_list_transcript_sessions_title_hint_skips_newlines(tmp_path: Path) -> None:
    """멀티라인 첫 메시지도 공백 하나로 정규화된 title_hint로 나온다."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="s3", enabled=True)
    t.append_entry(role="user", content="첫줄\n둘째줄", turn=1)

    out = list_transcript_sessions(tmp_path, limit=10)
    assert out[0]["title_hint"] == "첫줄 둘째줄"


def test_list_transcript_sessions_title_hint_none_if_no_user(tmp_path: Path) -> None:
    """assistant만 있고 user 메시지가 없으면 title_hint는 None."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="s4", enabled=True)
    t.append_entry(role="assistant", content="먼저 말을 걸었음", turn=1)

    out = list_transcript_sessions(tmp_path, limit=10)
    assert out[0]["title_hint"] is None


# ─────────────────────────────────────────────
# delete_transcript_session (v0.14.5)
# ─────────────────────────────────────────────
def test_delete_transcript_session_removes_dir(tmp_path: Path) -> None:
    """세션 디렉토리가 통째로 지워진다."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="del1", enabled=True)
    t.append_entry(role="user", content="bye", turn=1)
    sdir = tmp_path / "del1"
    assert sdir.exists()

    deleted = delete_transcript_session(tmp_path, "del1")
    assert deleted is True
    assert not sdir.exists()


def test_delete_transcript_session_returns_false_if_missing(tmp_path: Path) -> None:
    """없는 세션 삭제는 False 반환 (에러 아님)."""
    assert delete_transcript_session(tmp_path, "never-existed") is False


@pytest.mark.parametrize(
    "bad_id", ["../etc", "a/b", "a\\b", "..", "\x00evil", "sess/\x00"],
)
def test_delete_transcript_session_rejects_path_traversal(
    tmp_path: Path, bad_id: str
) -> None:
    """경로 분리자/'..'/'NUL'이 들어온 session_id는 ValueError로 거부."""
    with pytest.raises(ValueError):
        delete_transcript_session(tmp_path, bad_id)


def test_delete_transcript_session_does_not_touch_sibling(tmp_path: Path) -> None:
    """한 세션 삭제가 다른 세션 파일에 영향을 주지 않는다 (디렉토리 고립)."""
    t_a = SessionTranscript(sessions_dir=tmp_path, session_id="a", enabled=True)
    t_a.append_entry(role="user", content="hi-a", turn=1)
    t_b = SessionTranscript(sessions_dir=tmp_path, session_id="b", enabled=True)
    t_b.append_entry(role="user", content="hi-b", turn=1)

    assert delete_transcript_session(tmp_path, "a") is True
    assert not (tmp_path / "a").exists()
    # b는 그대로
    assert (tmp_path / "b" / "transcript.jsonl").exists()
    msgs = read_transcript_messages(tmp_path, "b")
    assert msgs[0]["content"] == "hi-b"


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
