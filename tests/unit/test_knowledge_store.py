"""
KnowledgeStore/KnowledgeRetriever 단위 테스트 — Part 2.5.8 (2026-04-21).

검증:
  1. split_into_chunks 경계 동작 (빈 문자열, 짧은 텍스트, overlap)
  2. KnowledgeEntry.id 결정론성 (source/title/section/chunk_index 기준)
  3. KnowledgeStore 인메모리 폴백 — add/add_many/count/list_sources
  4. KnowledgeStore.search_by_vector 인메모리 폴백 — 코사인 정렬
  5. KnowledgeStore.search_by_text 인메모리 폴백 — ILIKE
  6. KnowledgeRetriever.get_context — 빈 결과/정상 포맷/예산 제한
  7. KnowledgeRetriever는 임베딩 실패 시 텍스트 검색으로 폴백
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.rag.knowledge_retriever import KnowledgeRetriever
from core.rag.knowledge_store import (
    KnowledgeEntry,
    KnowledgeStore,
    _format_vector,
    split_into_chunks,
)


# ─────────────────────────────────────────────
# split_into_chunks
# ─────────────────────────────────────────────
def test_split_into_chunks_empty() -> None:
    assert split_into_chunks("") == []


def test_split_into_chunks_short_single_chunk() -> None:
    out = split_into_chunks("짧은 문단입니다.", max_chars=1000)
    assert len(out) == 1
    assert out[0] == "짧은 문단입니다."


def test_split_into_chunks_preserves_paragraph_boundaries() -> None:
    text = "문단 A.\n\n문단 B.\n\n문단 C."
    out = split_into_chunks(text, max_chars=1000)
    assert len(out) == 1
    assert "문단 A" in out[0] and "문단 B" in out[0] and "문단 C" in out[0]


def test_split_into_chunks_applies_overlap() -> None:
    """여러 청크로 쪼개지면 2번째부터는 앞 청크의 tail을 이어 붙인다."""
    # max_chars를 작게 설정해 강제로 2개 이상 청크가 나오도록
    text = ("A" * 200) + "\n\n" + ("B" * 200) + "\n\n" + ("C" * 200)
    chunks = split_into_chunks(text, max_chars=250, overlap=30)
    assert len(chunks) >= 2
    # 두 번째 청크 앞부분에 첫 청크의 tail이 포함되어야 한다
    assert chunks[1].startswith(chunks[0][-30:])


# ─────────────────────────────────────────────
# KnowledgeEntry.id 결정론성
# ─────────────────────────────────────────────
def test_entry_id_is_deterministic() -> None:
    a = KnowledgeEntry(source="s", title="T", content="x", chunk_index=0)
    b = KnowledgeEntry(source="s", title="T", content="y", chunk_index=0)
    # content가 달라도 (source, title, section, chunk_index)가 같으면 같은 id
    assert a.id == b.id


def test_entry_id_differs_by_chunk_index() -> None:
    a = KnowledgeEntry(source="s", title="T", content="x", chunk_index=0)
    b = KnowledgeEntry(source="s", title="T", content="x", chunk_index=1)
    assert a.id != b.id


# ─────────────────────────────────────────────
# KnowledgeStore 인메모리 폴백
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_store_inmemory_add_and_count() -> None:
    store = KnowledgeStore(pg_pool=None)
    await store.add(
        KnowledgeEntry(source="sample", title="A", content="foo", chunk_index=0)
    )
    await store.add(
        KnowledgeEntry(source="sample", title="B", content="bar", chunk_index=0)
    )
    await store.add(
        KnowledgeEntry(source="other", title="C", content="baz", chunk_index=0)
    )
    assert await store.count() == 3
    assert await store.count("sample") == 2


@pytest.mark.asyncio
async def test_store_inmemory_list_sources_sorted() -> None:
    store = KnowledgeStore(pg_pool=None)
    await store.add(KnowledgeEntry(source="z", title="A", content="x"))
    await store.add(KnowledgeEntry(source="a", title="B", content="x"))
    await store.add(KnowledgeEntry(source="a", title="C", content="x"))
    sources = await store.list_sources()
    # 이름순 정렬
    assert [s["source"] for s in sources] == ["a", "z"]
    assert sources[0]["count"] == 2


@pytest.mark.asyncio
async def test_store_inmemory_search_by_vector_sorts_by_cosine() -> None:
    """임베딩이 다른 3개 청크 중 가장 가까운 것이 먼저."""
    store = KnowledgeStore(pg_pool=None)
    # 질의 벡터와 각기 다른 유사도를 갖는 엔트리 구성
    query_vec = [1.0, 0.0, 0.0]
    await store.add(KnowledgeEntry(
        source="s", title="same", content="close", chunk_index=0,
        embedding=(1.0, 0.0, 0.0),  # cosine=1.0
    ))
    await store.add(KnowledgeEntry(
        source="s", title="rotated", content="medium", chunk_index=0,
        embedding=(0.5, 0.5, 0.0),  # cosine ~ 0.707
    ))
    await store.add(KnowledgeEntry(
        source="s", title="orth", content="far", chunk_index=0,
        embedding=(0.0, 1.0, 0.0),  # cosine=0
    ))
    results = await store.search_by_vector(
        embedding=query_vec, top_k=3, min_similarity=0.1,
    )
    # 최소 유사도 0.1 → orth는 제외됨
    assert len(results) == 2
    assert results[0]["title"] == "same"
    assert results[1]["title"] == "rotated"
    assert results[0]["similarity"] > results[1]["similarity"]


@pytest.mark.asyncio
async def test_store_inmemory_search_by_text() -> None:
    store = KnowledgeStore(pg_pool=None)
    await store.add(KnowledgeEntry(
        source="s", title="니체와 초인", content="차라투스트라는 초인을 말한다.",
    ))
    await store.add(KnowledgeEntry(
        source="s", title="카프카 변신", content="그레고르 잠자는 벌레가 된다.",
    ))
    out = await store.search_by_text("초인", top_k=5)
    assert len(out) == 1
    assert out[0]["title"] == "니체와 초인"


# ─────────────────────────────────────────────
# _format_vector
# ─────────────────────────────────────────────
def test_format_vector_returns_pgvector_literal() -> None:
    s = _format_vector([1.0, -0.5, 0.25])
    assert s == "[1.000000,-0.500000,0.250000]"


def test_format_vector_none_returns_none() -> None:
    assert _format_vector(None) is None


# ─────────────────────────────────────────────
# KnowledgeRetriever
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_retriever_empty_query_returns_empty() -> None:
    store = KnowledgeStore(pg_pool=None)
    retr = KnowledgeRetriever(store=store, embedding_provider=None)
    assert await retr.get_context("") == ""


@pytest.mark.asyncio
async def test_retriever_formats_header_with_title_source_similarity() -> None:
    """정상 경로 — 헤더에 title/source/sim이 포함되고 content가 이어진다."""
    store = KnowledgeStore(pg_pool=None)
    await store.add(KnowledgeEntry(
        source="kowiki", title="니체", content="독일 철학자, 1844~1900",
        embedding=(1.0, 0.0, 0.0),
    ))
    emb = MagicMock()
    emb.embed = AsyncMock(return_value=[[1.0, 0.0, 0.0]])
    retr = KnowledgeRetriever(store=store, embedding_provider=emb, top_k=3, min_similarity=0.1)
    ctx = await retr.get_context("니체 누구야?", max_tokens=500)
    assert "kowiki" in ctx
    assert "니체" in ctx
    assert "독일 철학자" in ctx
    assert "sim=" in ctx


@pytest.mark.asyncio
async def test_retriever_falls_back_to_text_when_embedding_fails() -> None:
    """임베딩 서버가 예외를 내도 텍스트 검색으로 답을 돌려줘야 한다."""
    store = KnowledgeStore(pg_pool=None)
    await store.add(KnowledgeEntry(
        source="kowiki", title="카프카", content="변신의 저자",
    ))
    broken = MagicMock()
    broken.embed = AsyncMock(side_effect=RuntimeError("embedding server down"))
    retr = KnowledgeRetriever(store=store, embedding_provider=broken, top_k=3)
    ctx = await retr.get_context("카프카", max_tokens=500)
    assert "카프카" in ctx
    assert "변신" in ctx


@pytest.mark.asyncio
async def test_retriever_respects_budget() -> None:
    """max_tokens 예산 내에서만 청크를 연결한다."""
    store = KnowledgeStore(pg_pool=None)
    # 큰 청크를 여러 개 넣어서 예산 초과 유도
    for i in range(3):
        await store.add(KnowledgeEntry(
            source="s", title=f"Doc-{i}", content="X" * 2000, chunk_index=0,
            embedding=(1.0, 0.0, 0.0),
        ))
    emb = MagicMock()
    emb.embed = AsyncMock(return_value=[[1.0, 0.0, 0.0]])
    retr = KnowledgeRetriever(store=store, embedding_provider=emb, top_k=5)
    # max_tokens=300 → chars_per_token=3 → 예산 900자
    ctx = await retr.get_context("anything", max_tokens=300)
    assert 0 < len(ctx) <= 900 + 200  # 마지막 청크는 잘려도 200자 내


@pytest.mark.asyncio
async def test_retriever_no_results_returns_empty_string() -> None:
    """빈 스토어에는 빈 문자열을 돌려준다."""
    store = KnowledgeStore(pg_pool=None)
    emb = MagicMock()
    emb.embed = AsyncMock(return_value=[[0.1] * 3])
    retr = KnowledgeRetriever(store=store, embedding_provider=emb)
    assert await retr.get_context("anything") == ""
