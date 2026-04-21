"""
심볼 인덱서/스토어/도구 단위 테스트 — Phase 10.0 (2026-04-21).

검증:
  1. SymbolEntry.id 결정론성 (path/kind/qualified_name/line_start)
  2. extract_symbols_from_source — 함수/클래스/메서드 정확 추출, 시그니처 포함
  3. SymbolStore 인메모리 — add/count/delete_by_path/search_by_name
  4. SymbolStore.search_by_name — 정확 일치 우선, 부분매칭 후순위
  5. SymbolStore.search_by_vector — 코사인 정렬
  6. SymbolProjectIndexer — 임시 파일 트리에서 자동 추출
  7. SymbolSearchTool — context.options["symbol_store"] 조회
  8. _looks_like_identifier 분기
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.rag.symbol_indexer import (
    SymbolProjectIndexer,
    extract_symbols_from_source,
)
from core.rag.symbol_store import SymbolEntry, SymbolStore
from core.tools.base import ToolUseContext
from core.tools.implementations.symbol_search_tool import (
    SymbolSearchTool,
    _looks_like_identifier,
)

# ─────────────────────────────────────────────
# extract_symbols_from_source
# ─────────────────────────────────────────────
SAMPLE_PY = '''\
"""Sample module."""

def top_level(x: int, y: int = 0) -> int:
    """Add two ints."""
    return x + y


async def fetch(url: str) -> str:
    return url


class Greeter:
    """A greeter class."""

    def hello(self, name: str) -> str:
        """Return greeting."""
        return f"hi {name}"

    async def async_bye(self) -> None:
        pass
'''


def test_extract_counts_and_kinds() -> None:
    entries = extract_symbols_from_source(
        SAMPLE_PY, path="sample.py", module="sample"
    )
    # top_level(function) + fetch(async_function) + Greeter(class) +
    # Greeter.hello(method) + Greeter.async_bye(async_method) = 5
    kinds = sorted(e.kind for e in entries)
    assert kinds == ["async_function", "async_method", "class", "function", "method"]
    names = [e.qualified_name for e in entries]
    assert "top_level" in names
    assert "Greeter.hello" in names
    assert "Greeter.async_bye" in names


def test_extract_signature_and_docstring() -> None:
    entries = extract_symbols_from_source(
        SAMPLE_PY, path="sample.py", module="sample"
    )
    top = next(e for e in entries if e.name == "top_level")
    assert "x: int" in top.signature
    assert "-> int" in top.signature
    assert top.docstring.strip().startswith("Add two ints")
    greeter = next(e for e in entries if e.kind == "class" and e.name == "Greeter")
    assert greeter.docstring.strip().startswith("A greeter class")


def test_extract_source_excerpt_in_summary() -> None:
    entries = extract_symbols_from_source(
        SAMPLE_PY, path="sample.py", module="sample"
    )
    top = next(e for e in entries if e.name == "top_level")
    assert "def top_level" in top.summary
    # docstring + source 발췌 포함
    assert "Add two ints" in top.summary


def test_extract_handles_syntax_error_gracefully() -> None:
    entries = extract_symbols_from_source(
        "def bad(:::", path="bad.py", module="bad"
    )
    assert entries == []


# ─────────────────────────────────────────────
# SymbolEntry.id 결정론성
# ─────────────────────────────────────────────
def test_entry_id_is_deterministic() -> None:
    a = SymbolEntry(source="s", path="a.py", module="a", kind="function",
                    name="f", qualified_name="f", line_start=10)
    b = SymbolEntry(source="s", path="a.py", module="a", kind="function",
                    name="f", qualified_name="f", line_start=10,
                    summary="different summary")
    assert a.id == b.id


def test_entry_id_differs_by_line() -> None:
    a = SymbolEntry(source="s", path="a.py", module="a", kind="function",
                    name="f", qualified_name="f", line_start=10)
    b = SymbolEntry(source="s", path="a.py", module="a", kind="function",
                    name="f", qualified_name="f", line_start=20)
    assert a.id != b.id


# ─────────────────────────────────────────────
# SymbolStore 인메모리 폴백
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_store_inmemory_add_count_delete() -> None:
    store = SymbolStore(pg_pool=None)
    await store.add(SymbolEntry(
        source="nexus", path="a.py", module="a", kind="function",
        name="foo", qualified_name="foo", line_start=1,
    ))
    await store.add(SymbolEntry(
        source="nexus", path="b.py", module="b", kind="class",
        name="Bar", qualified_name="Bar", line_start=1,
    ))
    assert await store.count() == 2
    n = await store.delete_by_path("nexus", "a.py")
    assert n == 1
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_store_search_by_name_exact_then_partial() -> None:
    store = SymbolStore(pg_pool=None)
    await store.add(SymbolEntry(source="s", path="x.py", module="x",
                                kind="function", name="query_loop",
                                qualified_name="x.query_loop", line_start=1))
    await store.add(SymbolEntry(source="s", path="y.py", module="y",
                                kind="function", name="query_loop_helper",
                                qualified_name="y.query_loop_helper", line_start=1))
    results = await store.search_by_name("query_loop", top_k=5)
    assert results[0]["name"] == "query_loop"
    assert results[0]["similarity"] == 1.0
    assert results[1]["name"] == "query_loop_helper"


@pytest.mark.asyncio
async def test_store_search_by_vector_cosine() -> None:
    store = SymbolStore(pg_pool=None)
    await store.add(SymbolEntry(
        source="s", path="x.py", module="x", kind="function",
        name="same", qualified_name="same", line_start=1,
        embedding=(1.0, 0.0, 0.0),
    ))
    await store.add(SymbolEntry(
        source="s", path="y.py", module="y", kind="function",
        name="near", qualified_name="near", line_start=1,
        embedding=(0.8, 0.6, 0.0),
    ))
    res = await store.search_by_vector(
        embedding=[1.0, 0.0, 0.0], top_k=5, min_similarity=0.1,
    )
    assert res[0]["name"] == "same"
    assert res[1]["name"] == "near"
    assert res[0]["similarity"] > res[1]["similarity"]


# ─────────────────────────────────────────────
# SymbolProjectIndexer (임시 파일 트리)
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_project_indexer_walks_and_ingests(tmp_path: Path) -> None:
    """실제 파일 트리에서 심볼 추출 + 스토어 적재."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text(SAMPLE_PY, encoding="utf-8")

    store = SymbolStore(pg_pool=None)
    indexer = SymbolProjectIndexer(store=store, embedder=None, project_source="test")
    stats = await indexer.index_project(tmp_path)
    assert stats["files"] == 1
    assert stats["symbols"] == 5
    assert await store.count("test") == 5


@pytest.mark.asyncio
async def test_project_indexer_reembeds_with_embedder(tmp_path: Path) -> None:
    """embedder 주입 시 모든 엔트리에 임베딩이 채워진다."""
    (tmp_path / "m.py").write_text(SAMPLE_PY, encoding="utf-8")
    store = SymbolStore(pg_pool=None)

    async def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    indexer = SymbolProjectIndexer(store=store, embedder=fake_embed, project_source="t")
    await indexer.index_project(tmp_path)
    # 내부 저장소 확인
    embedded = [e for e in store._store.values() if e.embedding is not None]
    assert len(embedded) == 5


# ─────────────────────────────────────────────
# SymbolSearchTool
# ─────────────────────────────────────────────
def test_looks_like_identifier() -> None:
    assert _looks_like_identifier("query_loop")
    assert _looks_like_identifier("Config.load")
    assert _looks_like_identifier("a.b.c.d")
    assert not _looks_like_identifier("what is query loop?")
    assert not _looks_like_identifier("")
    assert not _looks_like_identifier("foo bar")


@pytest.mark.asyncio
async def test_tool_returns_error_when_store_missing() -> None:
    tool = SymbolSearchTool()
    ctx = ToolUseContext(cwd=".", session_id="t", options={})
    res = await tool.call({"query": "foo"}, ctx)
    assert res.is_error
    assert "symbol" in res.error_message.lower() or "bootstrap" in res.error_message.lower()


@pytest.mark.asyncio
async def test_tool_returns_found_result(tmp_path: Path) -> None:
    store = SymbolStore(pg_pool=None)
    await store.add(SymbolEntry(
        source="nexus", path="core/orchestrator/query_loop.py",
        module="core.orchestrator.query_loop",
        kind="function", name="query_loop", qualified_name="query_loop",
        signature="(messages, ...) -> AsyncGenerator[StreamEvent, None]",
        docstring="Main agent turn loop.", line_start=130, line_end=600,
    ))
    tool = SymbolSearchTool()
    ctx = ToolUseContext(cwd=".", session_id="t",
                         options={"symbol_store": store})
    res = await tool.call({"query": "query_loop"}, ctx)
    assert not res.is_error
    assert res.metadata["count"] == 1
    assert "query_loop" in res.data
    assert "query_loop.py" in res.data


@pytest.mark.asyncio
async def test_tool_falls_back_to_vector_for_natural_language() -> None:
    """query가 자연어면 벡터 검색을 우선 시도한다."""
    store = SymbolStore(pg_pool=None)
    await store.add(SymbolEntry(
        source="nexus", path="core/x.py", module="core.x",
        kind="function", name="parse_tool_use",
        qualified_name="parse_tool_use", line_start=1,
        summary="parse tool use",
        embedding=(0.9, 0.1, 0.0),
    ))
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[[1.0, 0.0, 0.0]])

    tool = SymbolSearchTool()
    ctx = ToolUseContext(
        cwd=".", session_id="t",
        options={"symbol_store": store, "model_provider": provider},
    )
    res = await tool.call(
        {"query": "function that parses tool calls from model output"}, ctx,
    )
    assert not res.is_error
    assert "parse_tool_use" in res.data
    provider.embed.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_validates_empty_query() -> None:
    tool = SymbolSearchTool()
    msg = tool.validate_input({"query": "   "})
    assert msg is not None
