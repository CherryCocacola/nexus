# 진입점 채널(web/cli/api)별 히스토리 격리 검증 — 한 채널이 다른 채널 세션을 못 본다.
"""
web/cli/api 히스토리 격리의 회귀 방지 테스트.

배경: web 사이드바에는 web에서 진행한 세션만 보여야 하고, cli/api 기록은
web에서 조회되면 안 된다. 저장 계층 두 곳(Redis 단기 short_term, JSONL
transcript)에 channel 네임스페이스를 도입해 격리한다. channel=None(레거시 flat)은
기존 동작을 유지하며 채널 세션과 상호 비침범이어야 한다.

이 테스트는 저장 계층의 격리 계약만 순수하게 검증한다(웹/CLI 진입점 배선은
각 표면의 통합 테스트가 담당). short_term은 Redis 없이 인메모리 폴백으로 돈다.
"""

from __future__ import annotations

from core.memory.short_term import ShortTermMemory
from core.memory.transcript import (
    SessionTranscript,
    delete_transcript_session,
    list_transcript_sessions,
    read_transcript_messages,
)

# ─────────────────────────────────────────────
# ShortTermMemory — Redis 키 네임스페이스 격리
# ─────────────────────────────────────────────


async def test_short_term_same_session_id_isolated_across_channels():
    """같은 session_id라도 채널이 다르면 컨텍스트가 섞이지 않는다."""
    mem = ShortTermMemory(redis_client=None)
    await mem.save_conversation_context(
        "sid", [{"role": "user", "content": "web쪽"}], channel="web"
    )
    await mem.save_conversation_context(
        "sid", [{"role": "user", "content": "cli쪽"}], channel="cli"
    )

    assert await mem.get_conversation_context("sid", channel="web") == [
        {"role": "user", "content": "web쪽"}
    ]
    assert await mem.get_conversation_context("sid", channel="cli") == [
        {"role": "user", "content": "cli쪽"}
    ]


async def test_short_term_get_wrong_channel_returns_empty():
    """web에 저장한 세션을 다른 채널로 조회하면 안 보인다."""
    mem = ShortTermMemory(redis_client=None)
    await mem.save_conversation_context(
        "s1", [{"role": "user", "content": "x"}], channel="web"
    )
    assert await mem.get_conversation_context("s1", channel="cli") == []
    assert await mem.get_conversation_context("s1", channel="api") == []


async def test_short_term_list_sessions_filters_by_channel():
    """list_sessions는 지정 채널의 세션만 돌려준다(다른 채널·flat 미포함)."""
    mem = ShortTermMemory(redis_client=None)
    await mem.save_conversation_context("w1", [{"role": "user", "content": "a"}], channel="web")
    await mem.save_conversation_context("w2", [{"role": "user", "content": "b"}], channel="web")
    await mem.save_conversation_context("c1", [{"role": "user", "content": "c"}], channel="cli")

    assert set(await mem.list_sessions(channel="web")) == {"w1", "w2"}
    assert set(await mem.list_sessions(channel="cli")) == {"c1"}
    assert await mem.list_sessions(channel="api") == []


async def test_short_term_legacy_flat_isolated_from_channels():
    """channel=None(레거시 flat)은 채널 세션을 집지 않고 그 역도 성립(상호 비침범)."""
    mem = ShortTermMemory(redis_client=None)
    await mem.save_conversation_context("flat1", [{"role": "user", "content": "old"}])  # flat
    await mem.save_conversation_context("web1", [{"role": "user", "content": "new"}], channel="web")

    # flat 목록은 flat 세션만, web 목록은 web 세션만 — 서로 새지 않는다.
    assert set(await mem.list_sessions()) == {"flat1"}
    assert set(await mem.list_sessions(channel="web")) == {"web1"}
    # flat 저장은 flat 조회로 그대로 복원된다(하위호환).
    assert await mem.get_conversation_context("flat1") == [{"role": "user", "content": "old"}]


async def test_short_term_clear_session_channel_scoped():
    """clear_session은 지정 채널의 세션만 지운다(다른 채널은 보존)."""
    mem = ShortTermMemory(redis_client=None)
    await mem.save_conversation_context("sid", [{"role": "user", "content": "web"}], channel="web")
    await mem.save_conversation_context("sid", [{"role": "user", "content": "cli"}], channel="cli")

    await mem.clear_session("sid", channel="web")
    assert await mem.get_conversation_context("sid", channel="web") == []
    assert await mem.get_conversation_context("sid", channel="cli") == [
        {"role": "user", "content": "cli"}
    ]


# ─────────────────────────────────────────────
# SessionTranscript / 조회 함수 — 디스크 경로 격리
# ─────────────────────────────────────────────


def test_transcript_writes_under_channel_folder(tmp_path):
    """channel을 주면 {sessions_dir}/{channel}/{id}/ 아래에 기록된다(flat 경로엔 안 생김)."""
    t = SessionTranscript(sessions_dir=tmp_path, session_id="s1", channel="web")
    t.append_entry(role="user", content="안녕", turn=0)
    assert (tmp_path / "web" / "s1" / "transcript.jsonl").exists()
    assert not (tmp_path / "s1" / "transcript.jsonl").exists()


def test_transcript_list_filters_by_channel(tmp_path):
    """list_transcript_sessions는 지정 채널 세션만 반환한다."""
    SessionTranscript(tmp_path, "w1", channel="web").append_entry(role="user", content="w", turn=0)
    SessionTranscript(tmp_path, "c1", channel="cli").append_entry(role="user", content="c", turn=0)

    web_ids = {s["session_id"] for s in list_transcript_sessions(tmp_path, channel="web")}
    cli_ids = {s["session_id"] for s in list_transcript_sessions(tmp_path, channel="cli")}
    assert web_ids == {"w1"}
    assert cli_ids == {"c1"}


def test_transcript_legacy_flat_excluded_from_channel_lists(tmp_path):
    """레거시 flat 세션은 채널 목록에 안 나오고, 채널 세션은 flat 목록에 안 나온다."""
    SessionTranscript(tmp_path, "flat1").append_entry(role="user", content="old", turn=0)  # flat
    SessionTranscript(tmp_path, "web1", channel="web").append_entry(
        role="user", content="new", turn=0
    )

    # channel=None(flat) 목록은 top-level만 훑고, 채널 폴더(web)는 직속 파일이 없어 제외.
    flat_ids = {s["session_id"] for s in list_transcript_sessions(tmp_path)}
    web_ids = {s["session_id"] for s in list_transcript_sessions(tmp_path, channel="web")}
    assert flat_ids == {"flat1"}
    assert web_ids == {"web1"}


def test_transcript_read_and_delete_channel_scoped(tmp_path):
    """읽기·삭제 모두 채널 스코프 — 다른 채널로는 안 읽히고 안 지워진다."""
    SessionTranscript(tmp_path, "s1", channel="web").append_entry(role="user", content="hi", turn=0)

    assert len(read_transcript_messages(tmp_path, "s1", channel="web")) == 1
    assert read_transcript_messages(tmp_path, "s1", channel="cli") == []

    # 다른 채널 삭제는 대상 없음 → False, 맞는 채널 삭제는 True.
    assert delete_transcript_session(tmp_path, "s1", channel="cli") is False
    assert delete_transcript_session(tmp_path, "s1", channel="web") is True
    assert read_transcript_messages(tmp_path, "s1", channel="web") == []
