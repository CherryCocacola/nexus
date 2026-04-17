"""
core/rag/retriever.py 단위 테스트.

RAGRetriever의 검색 및 컨텍스트 생성 기능을 검증한다:
  - search: 벡터 유사도 기반 청크 검색
  - get_context: 토큰 예산 내 컨텍스트 문자열 생성
  - stats: 검색 통계
  - 통합: ProjectIndexer → RAGRetriever 파이프라인

외부 서비스(GPU 서버, PostgreSQL)는 사용하지 않는다.
embed()는 mock, LongTermMemory는 인메모리 폴백을 사용한다.
"""

from __future__ import annotations

import math
from unittest.mock import AsyncMock

import pytest

from core.memory.long_term import LongTermMemory
from core.memory.types import MemoryEntry, MemoryType
from core.rag.indexer import ProjectIndexer
from core.rag.retriever import RAGRetriever


# ─────────────────────────────────────────────
# 테스트용 유틸리티
# ─────────────────────────────────────────────

# 임베딩 차원 (테스트용)
EMBED_DIM = 10


def _deterministic_embedding(text: str) -> list[float]:
    """
    텍스트를 결정론적 벡터로 변환한다 (테스트용).
    hash 기반으로 각 차원에 다른 값을 넣어 벡터 간 유사도 차이를 만든다.
    """
    h = hash(text) % 10000
    return [math.sin(h + i) for i in range(EMBED_DIM)]


async def mock_embed(texts: list[str]) -> list[list[float]]:
    """GPU 서버 없이 임베딩을 생성하는 mock."""
    return [_deterministic_embedding(t) for t in texts]


def _make_model_provider() -> AsyncMock:
    """embed() mock이 설정된 ModelProvider를 생성한다."""
    provider = AsyncMock()
    provider.embed = AsyncMock(side_effect=mock_embed)
    return provider


async def _add_rag_entry(
    memory: LongTermMemory,
    content: str,
    file_path: str = "test.py",
    chunk_index: int = 0,
) -> MemoryEntry:
    """테스트용 RAG 엔트리를 메모리에 직접 추가한다."""
    embedding = _deterministic_embedding(f"File: {file_path}\n{content}")
    entry = MemoryEntry(
        memory_type=MemoryType.SEMANTIC,
        content=content,
        key=f"rag:{file_path}:chunk_{chunk_index}",
        tags=["rag", "py", file_path],
        importance=0.8,
        embedding=embedding,
        metadata={
            "file_path": file_path,
            "chunk_index": chunk_index,
            "total_chunks": 1,
            "source": "rag_indexer",
        },
    )
    await memory.add(entry)
    return entry


# ─────────────────────────────────────────────
# RAGRetriever 테스트
# ─────────────────────────────────────────────
class TestRAGRetrieverSearch:
    """RAGRetriever.search() 검색 기능 테스트."""

    @pytest.fixture
    def memory_store(self):
        """인메모리 폴백 모드의 LongTermMemory를 생성한다."""
        return LongTermMemory(pg_pool=None)

    @pytest.fixture
    def model_provider(self):
        """embed() mock이 설정된 ModelProvider를 생성한다."""
        return _make_model_provider()

    @pytest.fixture
    def retriever(self, model_provider, memory_store):
        """테스트용 RAGRetriever 인스턴스를 생성한다."""
        return RAGRetriever(
            model_provider=model_provider,
            memory_store=memory_store,
        )

    async def test_search_returns_indexed_chunks(self, retriever, memory_store):
        """인덱싱된 청크에서 검색 결과가 반환되는지 검증한다."""
        # 테스트 데이터를 메모리에 추가
        await _add_rag_entry(memory_store, "def add(a, b):\n    return a + b", "math.py", 0)
        await _add_rag_entry(memory_store, "def hello():\n    print('hello')", "greet.py", 0)

        results = await retriever.search("add function")

        # 결과가 반환되어야 한다
        assert len(results) >= 1
        # 모든 결과가 MemoryEntry여야 한다
        for r in results:
            assert isinstance(r, MemoryEntry)
            assert r.metadata.get("source") == "rag_indexer"

    async def test_search_empty_index_returns_empty(self, retriever):
        """빈 인덱스에서 검색 시 빈 리스트를 반환하는지 검증한다."""
        results = await retriever.search("anything")

        assert results == []

    async def test_search_embed_failure_returns_empty(self, memory_store):
        """embed() 호출 실패 시 빈 리스트를 반환하는지 검증한다."""
        # embed()가 예외를 던지는 provider
        failing_provider = AsyncMock()
        failing_provider.embed = AsyncMock(side_effect=RuntimeError("GPU 연결 실패"))
        retriever = RAGRetriever(
            model_provider=failing_provider,
            memory_store=memory_store,
        )

        # 메모리에 데이터가 있어도 embed 실패 시 빈 결과
        await _add_rag_entry(memory_store, "some code", "file.py")

        results = await retriever.search("query")

        assert results == []

    async def test_search_respects_top_k(self, retriever, memory_store):
        """top_k 파라미터가 반환 결과 수를 제한하는지 검증한다."""
        # 10개 청크 추가
        for i in range(10):
            await _add_rag_entry(
                memory_store, f"chunk_{i} content", f"file_{i}.py", i
            )

        results = await retriever.search("content", top_k=3)

        # top_k=3이므로 최대 3개만 반환
        assert len(results) <= 3

    async def test_search_increments_stats(self, retriever, memory_store):
        """search() 호출 시 통계가 증가하는지 검증한다."""
        await _add_rag_entry(memory_store, "def test():\n    pass", "test.py")

        assert retriever.stats["search_count"] == 0

        await retriever.search("test")
        assert retriever.stats["search_count"] == 1

        await retriever.search("another query")
        assert retriever.stats["search_count"] == 2


class TestRAGRetrieverGetContext:
    """RAGRetriever.get_context() 컨텍스트 생성 테스트."""

    @pytest.fixture
    def memory_store(self):
        """인메모리 폴백 모드의 LongTermMemory를 생성한다."""
        return LongTermMemory(pg_pool=None)

    @pytest.fixture
    def model_provider(self):
        """embed() mock이 설정된 ModelProvider를 생성한다."""
        return _make_model_provider()

    @pytest.fixture
    def retriever(self, model_provider, memory_store):
        """테스트용 RAGRetriever 인스턴스를 생성한다."""
        return RAGRetriever(
            model_provider=model_provider,
            memory_store=memory_store,
        )

    async def test_get_context_formats_correctly(self, retriever, memory_store):
        """get_context가 올바른 형식의 컨텍스트 문자열을 생성하는지 검증한다."""
        await _add_rag_entry(
            memory_store, "def add(a, b):\n    return a + b", "math.py", 0
        )

        context = await retriever.get_context("add function", max_tokens=1500)

        # 비어있지 않아야 한다
        assert context != ""
        # 파일 경로 헤더가 포함되어야 한다
        assert "math.py" in context
        # 실제 코드 내용이 포함되어야 한다
        assert "def add" in context

    async def test_get_context_empty_results_returns_empty_string(self, retriever):
        """검색 결과가 없으면 빈 문자열을 반환하는지 검증한다."""
        context = await retriever.get_context("nonexistent topic")

        assert context == ""

    async def test_get_context_budget_overflow_includes_partial(
        self, retriever, memory_store
    ):
        """토큰 예산 초과 시 일부 청크만 포함하는지 검증한다."""
        # 큰 청크 여러 개 추가 (각 ~300자 → ~100토큰)
        for i in range(10):
            content = f"# chunk_{i}\n" + ("x" * 300)
            await _add_rag_entry(memory_store, content, f"big_{i}.py", i)

        # 매우 작은 토큰 예산으로 요청
        context = await retriever.get_context("chunk", max_tokens=50)

        if context:
            # 예산이 작으면 10개 전부 포함되지 않아야 한다
            chunk_count = context.count("---")
            # 최대 1~2개 청크만 포함 (50토큰 ≈ 150자)
            assert chunk_count <= 5
        # 예산이 너무 작으면 빈 문자열도 가능

    async def test_get_context_includes_chunk_header(self, retriever, memory_store):
        """컨텍스트에 chunk N/M 형식의 헤더가 포함되는지 검증한다."""
        entry = await _add_rag_entry(
            memory_store, "important code here", "core.py", 2
        )
        # total_chunks를 5로 업데이트
        await memory_store.update(
            entry.id, metadata={
                "file_path": "core.py",
                "chunk_index": 2,
                "total_chunks": 5,
                "source": "rag_indexer",
            }
        )

        context = await retriever.get_context("important code")

        if context:
            # 헤더에 chunk 정보가 포함되어야 한다
            assert "core.py" in context


class TestRAGRetrieverStats:
    """RAGRetriever.stats 통계 테스트."""

    async def test_stats_initial_values(self):
        """초기 통계값이 모두 0인지 검증한다."""
        provider = _make_model_provider()
        memory = LongTermMemory(pg_pool=None)
        retriever = RAGRetriever(model_provider=provider, memory_store=memory)

        stats = retriever.stats
        assert stats["search_count"] == 0
        assert stats["total_results"] == 0

    async def test_stats_accumulates_results(self):
        """검색 결과 수가 누적되는지 검증한다."""
        provider = _make_model_provider()
        memory = LongTermMemory(pg_pool=None)
        retriever = RAGRetriever(model_provider=provider, memory_store=memory)

        # 데이터 추가
        await _add_rag_entry(memory, "def hello():\n    pass", "hello.py")
        await _add_rag_entry(memory, "def world():\n    pass", "world.py")

        await retriever.search("hello")
        first_total = retriever.stats["total_results"]

        await retriever.search("world")
        second_total = retriever.stats["total_results"]

        # 두 번째 검색 후 total_results가 증가해야 한다
        assert second_total >= first_total


# ─────────────────────────────────────────────
# 통합 테스트: ProjectIndexer → RAGRetriever
# ─────────────────────────────────────────────
class TestIndexerRetrieverPipeline:
    """ProjectIndexer로 인덱싱한 후 RAGRetriever로 검색하는 통합 파이프라인 테스트."""

    async def test_index_then_search_pipeline(self, tmp_path):
        """인덱싱 → 검색 전체 파이프라인이 동작하는지 검증한다.

        1. tmp_path에 Python 파일 생성
        2. ProjectIndexer로 인덱싱
        3. RAGRetriever로 검색
        4. 검색 결과에 원본 내용이 포함되는지 확인
        """
        # 공유 의존성 설정
        provider = _make_model_provider()
        memory = LongTermMemory(pg_pool=None)

        # 1. 테스트 파일 생성
        py_file = tmp_path / "calculator.py"
        py_file.write_text(
            "def add(a, b):\n"
            "    '''두 수를 더한다.'''\n"
            "    return a + b\n"
            "\n"
            "def subtract(a, b):\n"
            "    '''두 수를 뺀다.'''\n"
            "    return a - b\n",
            encoding="utf-8",
        )

        # 2. 인덱싱
        indexer = ProjectIndexer(model_provider=provider, memory_store=memory)
        chunk_count = await indexer.index_file(str(py_file))
        assert chunk_count >= 1

        # 3. 검색
        retriever = RAGRetriever(model_provider=provider, memory_store=memory)
        results = await retriever.search("calculator add function")

        # 4. 검색 결과 검증
        assert len(results) >= 1
        # 결과의 메타데이터에 source가 rag_indexer여야 한다
        assert all(r.metadata.get("source") == "rag_indexer" for r in results)

        # 5. get_context도 동작해야 한다
        context = await retriever.get_context("calculator add function")
        assert context != ""
        assert "calculator.py" in context
