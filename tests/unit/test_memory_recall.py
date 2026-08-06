# 장기 기억 회상 배선과 코드 RAG 분리 동작을 검증하는 단위 테스트.
"""
회상(recall) 배선 + 코드 RAG 청크 분리 검증.

배경 (2026-08-06 실측):
    MemoryManager.on_turn_start(회상 훅)은 구현돼 있었지만 프로덕션 어디서도 호출되지
    않아 장기 기억이 **쓰기 전용**이었다. 새 세션에서 이전 대화를 전혀 회상하지 못했다.
    동시에 tb_memories 에는 코드 RAG 청크가 113만 건(전체의 99.98%) 쌓여 있고 이들의
    importance 가 0.8 로 높아, 필터 없이 회상을 켜면 소스코드가 컨텍스트를 덮는다.

무엇을 지키나:
    - is_rag_chunk 가 RAG 청크(태그/메타데이터)만 정확히 골라낸다.
    - 회상 검색 결과에서 RAG 청크가 제외된다(on_turn_start / search_relevant 양쪽).
    - 회상 주입은 **기본 비활성**이다 — 설정을 켜지 않으면 아무 것도 주입되지 않는다.
    - 켜면 주입 블록이 만들어지고, 건수·길이 상한이 지켜진다.
    - 회상 실패는 대화를 막지 않는다(fail-soft).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from core.memory.types import MemoryEntry, MemoryType, is_rag_chunk
from core.orchestrator.query_engine import QueryEngine


def _entry(content: str, *, rag: bool = False, importance: float = 0.5) -> MemoryEntry:
    """대화 기억 또는 코드 RAG 청크 한 건을 만든다."""
    if rag:
        return MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content=content,
            key="rag:core/bootstrap.py:chunk_3",
            tags=["rag", "py", "bootstrap.py"],
            importance=0.8,
            metadata={"source": "rag_indexer"},
        )
    return MemoryEntry(
        memory_type=MemoryType.EPISODIC,
        content=content,
        key="turn:s1:abc",
        tags=["conversation", "s1"],
        importance=importance,
        metadata={"session_id": "s1"},
    )


# ─────────────────────────────────────────────
# RAG 청크 판별
# ─────────────────────────────────────────────


def test_is_rag_chunk_detects_indexer_metadata():
    """metadata.source 가 rag_indexer 면 RAG 청크다."""
    assert is_rag_chunk(_entry("def foo(): ...", rag=True)) is True


def test_is_rag_chunk_detects_tag_only():
    """metadata 가 비어도 tags 에 'rag' 가 있으면 RAG 청크다(리트리버와 동일 기준)."""
    entry = MemoryEntry(
        memory_type=MemoryType.SEMANTIC, content="x", key="k", tags=["rag"], metadata={}
    )
    assert is_rag_chunk(entry) is True


def test_is_rag_chunk_keeps_conversation_memory():
    """대화 기억은 RAG 청크가 아니다."""
    assert is_rag_chunk(_entry("사용자 사번은 NX-8842")) is False


# ─────────────────────────────────────────────
# 회상 주입 (QueryEngine._recall_memories)
# ─────────────────────────────────────────────


class _StubManager:
    """on_turn_start 만 흉내 내는 MemoryManager 스텁."""

    def __init__(self, entries: list[MemoryEntry] | None = None, fail: bool = False) -> None:
        self._entries = entries or []
        self._fail = fail
        self.called = False

    async def on_turn_start(self, session_id: str, user_message: str) -> list[MemoryEntry]:
        self.called = True
        if self._fail:
            raise RuntimeError("검색 실패")
        return self._entries


def _engine(manager: Any, options: dict | None = None) -> QueryEngine:
    """_recall_memories 만 호출할 수 있는 최소 엔진을 만든다.

    QueryEngine 전체를 세우면 모델·도구까지 필요하므로, 실제로 읽는 속성만 갖춘
    가벼운 객체에 메서드를 빌려 붙인다(이 테스트의 관심사는 회상 로직 하나다).
    """
    engine = SimpleNamespace(
        _memory_manager=manager,
        _session_id="s1",
        _context=SimpleNamespace(options=options or {}),
    )
    engine._recall_memories = QueryEngine._recall_memories.__get__(engine, QueryEngine)
    return engine  # type: ignore[return-value]


async def test_recall_disabled_by_default():
    """설정을 켜지 않으면 회상 조회 자체를 하지 않는다(무회귀 보장)."""
    manager = _StubManager([_entry("과거 기억")])

    block = await _engine(manager, options={})._recall_memories("질문")

    assert block == ""
    assert manager.called is False, "꺼져 있는데 검색을 수행했다"


async def test_recall_enabled_builds_block():
    """켜면 과거 기억이 주입 블록으로 만들어진다."""
    manager = _StubManager([_entry("사용자는 정보전산원 소속이다")])
    engine = _engine(manager, {"memory_recall": {"enabled": True}})

    block = await engine._recall_memories("제 소속이 어디죠?")

    assert "사용자는 정보전산원 소속이다" in block
    # 모델이 이걸 사용자 발화로 오해하지 않도록 성격을 명시해야 한다.
    assert "참고용" in block
    assert manager.called is True


async def test_recall_respects_item_limit():
    """주입 건수 상한을 지킨다(컨텍스트 보호)."""
    entries = [_entry(f"기억{i}") for i in range(10)]
    engine = _engine(_StubManager(entries), {"memory_recall": {"enabled": True, "max_items": 3}})

    block = await engine._recall_memories("질문")

    assert block.count("- 기억") == 3


async def test_recall_truncates_long_entries():
    """기억 1건이 길어도 지정 길이로 잘라 넣는다."""
    engine = _engine(
        _StubManager([_entry("가" * 900)]),
        {"memory_recall": {"enabled": True, "max_chars": 50}},
    )

    block = await engine._recall_memories("질문")

    assert "가" * 50 in block
    assert "가" * 51 not in block


async def test_recall_skips_empty_entries():
    """내용이 빈 기억만 있으면 블록을 만들지 않는다."""
    engine = _engine(_StubManager([_entry("   ")]), {"memory_recall": {"enabled": True}})

    assert await engine._recall_memories("질문") == ""


async def test_recall_failure_is_fail_soft():
    """검색이 실패해도 예외를 올리지 않고 빈 블록으로 넘어간다."""
    engine = _engine(_StubManager(fail=True), {"memory_recall": {"enabled": True}})

    assert await engine._recall_memories("질문") == ""


async def test_recall_without_manager_is_noop():
    """MemoryManager 가 없으면(테스트/경량 경로) 아무 것도 하지 않는다."""
    engine = _engine(None, {"memory_recall": {"enabled": True}})

    assert await engine._recall_memories("질문") == ""
