"""
core/rag/indexer.py 단위 테스트.

ProjectIndexer와 청크 분할 함수를 검증한다:
  - _split_code_chunks: 코드 파일의 함수/클래스 경계 분할
  - _split_text_chunks: 문서 파일의 단락 경계 분할
  - ProjectIndexer.index_file: 단일 파일 인덱싱
  - ProjectIndexer.index_directory: 디렉토리 재귀 인덱싱

외부 서비스(GPU 서버)는 mock으로 처리한다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.memory.long_term import LongTermMemory
from core.rag.indexer import (
    CODE_CHUNK_SIZE,
    DOC_CHUNK_SIZE,
    EXCLUDED_DIRS,
    ProjectIndexer,
    _split_code_chunks,
    _split_text_chunks,
)


# ─────────────────────────────────────────────
# 테스트용 mock embed 함수
# ─────────────────────────────────────────────
async def mock_embed(texts: list[str]) -> list[list[float]]:
    """
    GPU 서버 없이 임베딩을 생성하는 테스트용 mock.
    각 텍스트의 hash를 기반으로 10차원 벡터를 반환한다.
    """
    return [[float(hash(t) % 100) / 100.0] * 10 for t in texts]


def _make_model_provider() -> AsyncMock:
    """embed() 메서드가 mock_embed를 반환하는 ModelProvider mock을 생성한다."""
    provider = AsyncMock()
    provider.embed = AsyncMock(side_effect=mock_embed)
    return provider


# ─────────────────────────────────────────────
# _split_code_chunks 테스트
# ─────────────────────────────────────────────
class TestSplitCodeChunks:
    """코드 파일 청크 분할 함수 테스트."""

    def test_split_code_chunks_python_function_boundary(self):
        """Python 함수 정의 경계에서 청크가 분할되는지 검증한다."""
        # 두 개의 함수가 포함된 코드
        code = (
            "# 모듈 설명\n"
            "import os\n"
            "\n"
            "def hello():\n"
            "    print('hello')\n"
            "    return True\n"
            "\n"
            "def world():\n"
            "    print('world')\n"
            "    return False\n"
        )
        chunks = _split_code_chunks(code, max_size=500)

        # 최소 2개 청크로 분할되어야 한다 (함수 경계)
        assert len(chunks) >= 2
        # 첫 번째 청크에 import와 hello 함수가 포함
        assert "import os" in chunks[0]
        assert "hello" in chunks[0]
        # 두 번째 청크에 world 함수가 포함
        assert "world" in chunks[1]

    def test_split_code_chunks_empty_text(self):
        """빈 텍스트 입력 시 빈 리스트를 반환하는지 검증한다."""
        chunks = _split_code_chunks("", max_size=500)
        assert chunks == []

    def test_split_code_chunks_whitespace_only(self):
        """공백만 있는 텍스트 입력 시 빈 리스트를 반환하는지 검증한다."""
        chunks = _split_code_chunks("   \n\n  \n", max_size=500)
        assert chunks == []

    def test_split_code_chunks_max_size_overflow(self):
        """max_size를 초과하는 긴 함수는 강제 분할되는지 검증한다."""
        # max_size보다 긴 단일 함수 생성
        long_body = "\n".join([f"    line_{i} = {i}" for i in range(100)])
        code = f"def very_long_function():\n{long_body}\n"

        # 작은 max_size로 강제 분할 유도
        chunks = _split_code_chunks(code, max_size=200)

        # 강제 분할로 여러 청크가 생성되어야 한다
        assert len(chunks) >= 2
        # 모든 원본 라인이 청크에 포함되어야 한다
        all_text = "\n".join(chunks)
        assert "very_long_function" in all_text
        assert "line_99" in all_text

    def test_split_code_chunks_async_def_boundary(self):
        """async def 키워드도 경계로 인식되는지 검증한다."""
        # current_size > 50 조건을 충족하기 위해 충분히 긴 앞부분이 필요하다
        code = (
            "# header comment that is long enough\n"
            "x = 'some_long_variable_value_here'\n"
            "y = 'another_long_variable_value'\n"
            "\n"
            "async def fetch_data():\n"
            "    return await get()\n"
        )
        chunks = _split_code_chunks(code, max_size=500)

        # async def가 새 청크를 시작해야 한다
        assert len(chunks) >= 2
        assert "async def fetch_data" in chunks[-1]

    def test_split_code_chunks_class_boundary(self):
        """class 키워드도 경계로 인식되는지 검증한다."""
        code = (
            "# 앞부분 코드 padding\n"
            "import sys\n"
            "import os\n"
            "CONSTANT = 42\n"
            "\n"
            "class MyClass:\n"
            "    def method(self):\n"
            "        pass\n"
        )
        chunks = _split_code_chunks(code, max_size=500)

        assert len(chunks) >= 2
        assert "class MyClass" in chunks[-1]


# ─────────────────────────────────────────────
# _split_text_chunks 테스트
# ─────────────────────────────────────────────
class TestSplitTextChunks:
    """문서 파일 청크 분할 함수 테스트."""

    def test_split_text_chunks_paragraph_boundary(self):
        """단락 경계(빈 줄)에서 분할되는지 검증한다."""
        text = (
            "첫 번째 단락입니다.\n"
            "이어지는 문장입니다.\n"
            "\n"
            "두 번째 단락입니다.\n"
            "계속되는 내용입니다.\n"
            "\n"
            "세 번째 단락입니다.\n"
        )
        # 작은 max_size로 분할 유도
        chunks = _split_text_chunks(text, max_size=50)

        assert len(chunks) >= 2
        # 각 청크는 비어있지 않아야 한다
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_split_text_chunks_empty_text(self):
        """빈 텍스트 입력 시 빈 리스트를 반환하는지 검증한다."""
        chunks = _split_text_chunks("", max_size=1000)
        assert chunks == []

    def test_split_text_chunks_whitespace_only(self):
        """공백/개행만 있는 텍스트는 빈 리스트를 반환하는지 검증한다."""
        chunks = _split_text_chunks("  \n\n  \n\n  ", max_size=1000)
        assert chunks == []

    def test_split_text_chunks_single_long_paragraph(self):
        """단일 긴 단락은 하나의 청크로 반환되는지 검증한다.

        _split_text_chunks는 단락 경계에서만 분할하므로,
        단락 내부의 긴 텍스트는 하나의 청크에 그대로 포함된다.
        """
        long_text = "이것은 매우 긴 단락입니다. " * 200
        chunks = _split_text_chunks(long_text, max_size=100)

        # 단락 구분이 없으므로 하나의 청크에 전체가 포함된다
        assert len(chunks) >= 1
        # 원본 내용이 보존되어야 한다
        combined = "\n\n".join(chunks)
        assert "매우 긴 단락" in combined

    def test_split_text_chunks_within_max_size(self):
        """max_size 이하의 짧은 텍스트는 하나의 청크로 반환되는지 검증한다."""
        text = "짧은 문서입니다."
        chunks = _split_text_chunks(text, max_size=1000)

        assert len(chunks) == 1
        assert chunks[0] == "짧은 문서입니다."

    def test_split_text_chunks_preserves_content(self):
        """분할 후에도 원본 내용이 누락 없이 보존되는지 검증한다."""
        paragraphs = [f"단락 {i}번 내용입니다." for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = _split_text_chunks(text, max_size=50)

        # 모든 단락이 어떤 청크에든 포함되어야 한다
        combined = " ".join(chunks)
        for p in paragraphs:
            assert p in combined


# ─────────────────────────────────────────────
# ProjectIndexer 테스트
# ─────────────────────────────────────────────
class TestProjectIndexer:
    """ProjectIndexer 인덱싱 기능 테스트."""

    @pytest.fixture
    def memory_store(self):
        """인메모리 폴백 모드의 LongTermMemory를 생성한다."""
        return LongTermMemory(pg_pool=None)

    @pytest.fixture
    def model_provider(self):
        """embed() mock이 설정된 ModelProvider를 생성한다."""
        return _make_model_provider()

    @pytest.fixture
    def indexer(self, model_provider, memory_store):
        """테스트용 ProjectIndexer 인스턴스를 생성한다."""
        return ProjectIndexer(
            model_provider=model_provider,
            memory_store=memory_store,
        )

    async def test_index_file_python_file(self, indexer, model_provider, tmp_path):
        """Python 파일을 인덱싱하면 청크가 생성되고 메모리에 저장되는지 검증한다."""
        # 테스트용 Python 파일 생성
        py_file = tmp_path / "example.py"
        py_file.write_text(
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "def multiply(a, b):\n"
            "    return a * b\n",
            encoding="utf-8",
        )

        chunk_count = await indexer.index_file(str(py_file))

        # 최소 1개 이상의 청크가 생성되어야 한다
        assert chunk_count >= 1
        # embed()가 호출되어야 한다
        model_provider.embed.assert_called()
        # 통계가 갱신되어야 한다
        assert indexer.stats["indexed_files"] == 1
        assert indexer.stats["indexed_chunks"] == chunk_count

    async def test_index_file_empty_file_returns_zero(self, indexer, tmp_path):
        """빈 파일은 0개 청크를 반환하는지 검증한다."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("", encoding="utf-8")

        chunk_count = await indexer.index_file(str(empty_file))

        assert chunk_count == 0
        assert indexer.stats["indexed_files"] == 0

    async def test_index_file_nonexistent_returns_zero(self, indexer, tmp_path):
        """존재하지 않는 파일은 0개 청크를 반환하는지 검증한다."""
        fake_path = str(tmp_path / "nonexistent.py")

        chunk_count = await indexer.index_file(fake_path)

        assert chunk_count == 0
        assert indexer.stats["indexed_files"] == 0

    async def test_index_file_whitespace_only_returns_zero(self, indexer, tmp_path):
        """공백만 있는 파일은 0개 청크를 반환하는지 검증한다."""
        ws_file = tmp_path / "whitespace.py"
        ws_file.write_text("   \n\n  \n", encoding="utf-8")

        chunk_count = await indexer.index_file(str(ws_file))

        assert chunk_count == 0

    async def test_index_directory_indexes_all_files(self, indexer, model_provider, tmp_path):
        """디렉토리 내 모든 대상 파일이 인덱싱되는지 검증한다."""
        # 테스트 파일 구조 생성
        (tmp_path / "main.py").write_text(
            "def main():\n    print('hello')\n",
            encoding="utf-8",
        )
        (tmp_path / "utils.py").write_text(
            "def helper():\n    return 42\n",
            encoding="utf-8",
        )
        (tmp_path / "readme.md").write_text(
            "# 프로젝트\n\n설명입니다.\n",
            encoding="utf-8",
        )
        # 인덱싱 대상이 아닌 파일 (확장자 필터링)
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")

        stats = await indexer.index_directory(str(tmp_path))

        # .py 2개 + .md 1개 = 3개 파일 인덱싱
        assert stats["indexed_files"] == 3
        assert stats["indexed_chunks"] >= 3
        assert stats["failed_files"] == 0

    async def test_index_directory_excludes_dirs(self, indexer, tmp_path):
        """EXCLUDED_DIRS에 포함된 디렉토리는 인덱싱에서 제외되는지 검증한다."""
        # 일반 파일
        (tmp_path / "app.py").write_text(
            "def app():\n    pass\n",
            encoding="utf-8",
        )
        # EXCLUDED_DIRS 중 하나(__pycache__) 안에 파일 생성
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text(
            "def cached():\n    pass\n",
            encoding="utf-8",
        )
        # .git 안에 파일 생성
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config.py").write_text(
            "GIT_CONFIG = True\n",
            encoding="utf-8",
        )

        stats = await indexer.index_directory(str(tmp_path))

        # __pycache__와 .git 안의 파일은 제외되어야 한다
        assert stats["indexed_files"] == 1

    async def test_index_directory_nonexistent_returns_empty_stats(self, indexer):
        """존재하지 않는 디렉토리는 빈 통계를 반환하는지 검증한다."""
        stats = await indexer.index_directory("/nonexistent/path")

        assert stats["indexed_files"] == 0
        assert stats["indexed_chunks"] == 0
        assert stats["failed_files"] == 0

    async def test_stats_initial_values(self, indexer):
        """초기 통계값이 모두 0인지 검증한다."""
        stats = indexer.stats

        assert stats["indexed_files"] == 0
        assert stats["indexed_chunks"] == 0
        assert stats["failed_files"] == 0

    async def test_index_file_stores_correct_metadata(
        self, indexer, memory_store, tmp_path
    ):
        """인덱싱된 청크의 메타데이터가 올바르게 저장되는지 검증한다."""
        py_file = tmp_path / "meta_test.py"
        py_file.write_text(
            "def test_func():\n    return 1\n",
            encoding="utf-8",
        )

        await indexer.index_file(str(py_file))

        # 인메모리 저장소에서 직접 확인
        from core.memory.types import MemoryType

        entries = await memory_store.get_by_type(MemoryType.SEMANTIC)
        assert len(entries) >= 1

        # 메타데이터 검증
        entry = entries[0]
        assert entry.metadata.get("source") == "rag_indexer"
        assert "file_path" in entry.metadata
        assert "chunk_index" in entry.metadata
        assert "total_chunks" in entry.metadata
        # 태그에 "rag"가 포함되어야 한다
        assert "rag" in entry.tags

    async def test_index_file_embed_failure_continues(
        self, memory_store, tmp_path
    ):
        """embed() 실패 시 에러 없이 0개 청크를 반환하는지 검증한다."""
        # embed()가 예외를 던지는 provider
        provider = AsyncMock()
        provider.embed = AsyncMock(side_effect=RuntimeError("GPU 오류"))
        indexer = ProjectIndexer(
            model_provider=provider,
            memory_store=memory_store,
        )

        py_file = tmp_path / "fail_test.py"
        py_file.write_text(
            "def broken():\n    pass\n",
            encoding="utf-8",
        )

        # 예외가 전파되지 않고 0을 반환해야 한다
        chunk_count = await indexer.index_file(str(py_file))

        # embed 실패 시 청크가 저장되지 않는다 (파일 자체는 indexed_files에 카운트됨)
        # index_file은 embed 실패 배치를 건너뛰고 계속 진행하므로
        # chunk_count는 0이 된다
        assert chunk_count == 0
