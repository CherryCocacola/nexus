"""
프로젝트 파일 인덱서 — 파일을 청크로 분할하고 임베딩으로 인덱싱한다.

RAG 파이프라인의 첫 단계: 프로젝트 파일을 읽고 → 청크로 분할하고 →
e5-large 임베딩을 생성하여 → LongTermMemory에 저장한다.

왜 인덱싱하는가:
  - 8K 컨텍스트에서 파일 전체를 읽으면 2~3개가 한계
  - 인덱싱하면 질문과 관련된 청크만 검색하여 컨텍스트에 주입 가능
  - 프로젝트 전체 분석이 실시간 대화에서 가능해진다

인덱싱 시점:
  - 부트스트랩 시 백그라운드 (asyncio.create_task)
  - 인덱싱 완료 전에도 채팅 가능 (검색 결과가 없을 뿐)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from core.memory.types import MemoryEntry, MemoryType
from core.model.inference import ModelProvider

logger = logging.getLogger("nexus.rag.indexer")

# 인덱싱 대상 파일 확장자
INDEXABLE_EXTENSIONS = {
    # 소스 코드
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    # 설정/데이터
    ".yaml", ".yml", ".json", ".toml",
    # 문서
    ".md", ".txt", ".rst",
    # 웹
    ".html", ".css",
    # SQL
    ".sql",
    # 셸
    ".sh", ".bash",
}

# 인덱싱 제외 디렉토리
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "egg-info",
    "models", "checkpoints", "data", "logs",
}

# 코드 파일 청크 크기 (문자)
CODE_CHUNK_SIZE = 500
# 문서 파일 청크 크기 (문자)
DOC_CHUNK_SIZE = 1000
# 한 번에 임베딩할 배치 크기
EMBED_BATCH_SIZE = 10
# 파일 크기 제한 (100KB 초과 파일은 건너뜀)
MAX_FILE_SIZE = 100 * 1024


class ProjectIndexer:
    """
    프로젝트 파일을 인덱싱하여 벡터 검색이 가능하게 한다.

    흐름: 파일 탐색 → 텍스트 읽기 → 청크 분할 → 임베딩 → 메모리 저장
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        memory_store: Any,  # LongTermMemory
    ) -> None:
        """
        Args:
            model_provider: 임베딩 생성용 (embed() 메서드)
            memory_store: 인덱스 저장소 (LongTermMemory)
        """
        self._model_provider = model_provider
        self._memory = memory_store
        self._indexed_files: int = 0
        self._indexed_chunks: int = 0
        self._failed_files: int = 0

    async def index_directory(
        self,
        path: str,
        extensions: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        디렉토리를 재귀 탐색하여 모든 대상 파일을 인덱싱한다.

        Args:
            path: 인덱싱할 루트 디렉토리
            extensions: 대상 확장자 (None이면 기본값 사용)

        Returns:
            인덱싱 통계: indexed_files, indexed_chunks, failed_files
        """
        target_exts = extensions or INDEXABLE_EXTENSIONS
        self._indexed_files = 0
        self._indexed_chunks = 0
        self._failed_files = 0

        root = Path(path)
        if not root.is_dir():
            logger.warning("인덱싱 대상이 디렉토리가 아닙니다: %s", path)
            return self.stats

        # 대상 파일 수집
        files_to_index: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # 제외 디렉토리 필터링 (os.walk의 dirnames를 수정하면 하위 탐색 제외)
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() in target_exts and fpath.stat().st_size <= MAX_FILE_SIZE:
                    files_to_index.append(fpath)

        logger.info("인덱싱 대상: %d개 파일 (%s)", len(files_to_index), path)

        # 파일별 인덱싱
        for fpath in files_to_index:
            try:
                await self.index_file(str(fpath))
            except Exception as e:
                self._failed_files += 1
                logger.debug("파일 인덱싱 실패: %s — %s", fpath, e)

        logger.info(
            "인덱싱 완료: %d파일, %d청크, %d실패",
            self._indexed_files,
            self._indexed_chunks,
            self._failed_files,
        )
        return self.stats

    async def index_file(self, file_path: str) -> int:
        """
        단일 파일을 인덱싱한다.

        파일 읽기 → 청크 분할 → 임베딩 → 메모리 저장

        Args:
            file_path: 파일 절대 경로

        Returns:
            인덱싱된 청크 수
        """
        fpath = Path(file_path)
        if not fpath.is_file():
            return 0

        # 파일 읽기
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("파일 읽기 실패: %s — %s", file_path, e)
            return 0

        if not text.strip():
            return 0

        # 청크 분할 (코드 vs 문서)
        is_code = fpath.suffix.lower() in {".py", ".js", ".ts", ".go", ".rs", ".java"}
        chunks = _split_into_chunks(text, is_code=is_code)

        if not chunks:
            return 0

        # 배치 임베딩 생성
        # 각 청크에 파일 경로를 접두어로 붙여서 검색 정확도를 높인다
        relative_path = str(fpath)
        texts_to_embed = [
            f"File: {relative_path}\n{chunk}" for chunk in chunks
        ]

        chunk_count = 0
        for i in range(0, len(texts_to_embed), EMBED_BATCH_SIZE):
            batch = texts_to_embed[i : i + EMBED_BATCH_SIZE]
            batch_chunks = chunks[i : i + EMBED_BATCH_SIZE]

            try:
                embeddings = await self._model_provider.embed(batch)
            except Exception as e:
                logger.debug("임베딩 생성 실패 (배치 %d): %s", i, e)
                continue

            # 메모리에 저장
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                chunk_idx = i + j
                entry = MemoryEntry(
                    memory_type=MemoryType.SEMANTIC,
                    content=chunk,
                    key=f"rag:{relative_path}:chunk_{chunk_idx}",
                    tags=["rag", fpath.suffix.lstrip("."), fpath.name],
                    importance=0.8,  # RAG 청크는 장기 보존
                    embedding=embedding,
                    metadata={
                        "file_path": relative_path,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "source": "rag_indexer",
                    },
                )
                await self._memory.add(entry)
                chunk_count += 1

        self._indexed_files += 1
        self._indexed_chunks += chunk_count
        return chunk_count

    @property
    def stats(self) -> dict[str, Any]:
        """인덱싱 통계를 반환한다."""
        return {
            "indexed_files": self._indexed_files,
            "indexed_chunks": self._indexed_chunks,
            "failed_files": self._failed_files,
        }


# ─────────────────────────────────────────────
# 청크 분할 함수
# ─────────────────────────────────────────────
def _split_into_chunks(text: str, is_code: bool = False) -> list[str]:
    """
    텍스트를 청크로 분할한다.

    코드 파일: 함수/클래스 경계를 인식하여 분할 (500자)
    문서 파일: 단락 경계에서 분할 (1000자)

    왜 크기가 다른가:
      - 코드는 함수 단위가 의미 있는 검색 단위
      - 문서는 단락 단위가 자연스러운 의미 단위
      - 8K 컨텍스트에서 top_k=3 × 500자 ≈ 500토큰으로 여유
    """
    if is_code:
        return _split_code_chunks(text, max_size=CODE_CHUNK_SIZE)
    return _split_text_chunks(text, max_size=DOC_CHUNK_SIZE)


def _split_code_chunks(text: str, max_size: int = CODE_CHUNK_SIZE) -> list[str]:
    """
    코드를 함수/클래스 경계에서 분할한다.

    Python 함수/클래스 정의 패턴을 인식하여
    의미 있는 단위로 분할한다.
    """
    # 함수/클래스 정의 시작 패턴 (Python, JS, Go 등)
    boundary_pattern = re.compile(
        r"^(?:def |async def |class |function |const |export |func )",
        re.MULTILINE,
    )

    chunks: list[str] = []
    lines = text.split("\n")
    current_chunk: list[str] = []
    current_size = 0

    for line in lines:
        # 함수/클래스 경계를 만나면 현재 청크를 저장하고 새 청크 시작
        if boundary_pattern.match(line.strip()) and current_size > 50:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_chunk = []
            current_size = 0

        current_chunk.append(line)
        current_size += len(line) + 1

        # 최대 크기 초과 시 강제 분할
        if current_size >= max_size:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_chunk = []
            current_size = 0

    # 마지막 청크
    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _split_text_chunks(text: str, max_size: int = DOC_CHUNK_SIZE) -> list[str]:
    """
    문서 텍스트를 단락 경계에서 분할한다.

    빈 줄(단락 구분)에서 우선 분할하고,
    단락이 max_size를 초과하면 문장 경계에서 분할한다.
    """
    # 빈 줄로 단락 분리
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_size + len(para) > max_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(para)
        current_size += len(para)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
