"""
프로젝트 파일 인덱서 — 파일을 청크로 쪼개고 임베딩으로 색인한다.

이 모듈은 RAG(검색 증강 생성) 파이프라인의 "적재(indexing)" 단계를 담당한다.
프로젝트 폴더 안의 소스/문서 파일을 훑어서 → 적당한 크기의 청크로 나누고 →
e5-large 임베딩 벡터를 만든 뒤 → 장기 메모리(LongTermMemory)에 저장한다.
이렇게 미리 색인해 두면, 나중에 사용자가 질문했을 때 질문과 의미가 가까운
청크만 골라 컨텍스트에 넣어 줄 수 있다.

왜 굳이 인덱싱을 하는가 (배경):
  - 우리 모델은 8K 컨텍스트가 한계라, 파일 원문을 통째로 넣으면 2~3개면 꽉 찬다.
  - 미리 색인해 두면 질문과 관련된 청크만 뽑아 넣을 수 있어 훨씬 효율적이다.
  - 결과적으로 프로젝트 전체를 대상으로 한 분석이 실시간 대화에서도 가능해진다.

언제 인덱싱이 도는가 (호출 시점):
  - 부트스트랩 시 백그라운드 태스크로 실행된다 (asyncio.create_task).
  - 인덱싱이 끝나기 전에도 채팅은 가능하다 — 단지 검색 결과가 아직 없을 뿐이다.

주요 구성:
  - ProjectIndexer 클래스: 디렉토리/파일 단위 인덱싱의 진입점.
  - _split_into_chunks / _split_code_chunks / _split_text_chunks: 청크 분할 헬퍼.

외부 의존:
  - ModelProvider.embed()  — 임베딩 벡터 생성 (core.model.inference)
  - LongTermMemory.add()   — 청크를 벡터와 함께 저장 (core.memory)

작성자: 이현수 / 작성일: 2026-07-05
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

# 인덱싱 대상으로 삼을 파일 확장자 집합.
# 여기에 없는 확장자(이미지·바이너리 등)는 아예 건너뛴다.
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

# 탐색 도중 통째로 건너뛸 디렉토리 이름 집합.
# 빌드 산출물·캐시·가상환경·모델 파일 등은 색인해도 의미가 없어서 제외한다.
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "egg-info",
    "models", "checkpoints", "data", "logs",
}

# 코드 파일을 나눌 때 목표로 하는 청크 크기 (문자 수 기준).
# 코드는 함수 하나가 대략 이 정도라 검색 단위로 알맞다.
CODE_CHUNK_SIZE = 500
# 문서 파일을 나눌 때 목표로 하는 청크 크기 (문자 수 기준).
# 문서는 단락이 더 길어서 코드보다 크게 잡는다.
DOC_CHUNK_SIZE = 1000
# 임베딩을 만들 때 한 번의 호출에 몰아 넣는 청크 개수(배치 크기).
# 한 청크씩 부르면 왕복이 잦아 느리므로 묶어서 처리한다.
EMBED_BATCH_SIZE = 10
# 인덱싱을 시도할 최대 파일 크기(바이트). 100KB를 넘으면 건너뛴다.
# 지나치게 큰 파일은 청크가 폭증하고 대개 자동 생성물이라 색인 가치가 낮다.
MAX_FILE_SIZE = 100 * 1024


class ProjectIndexer:
    """
    프로젝트 파일을 색인하여 벡터 기반 의미 검색이 가능하게 만드는 클래스.

    전체 흐름은 다음 5단계로 요약된다:
      파일 탐색 → 텍스트 읽기 → 청크 분할 → 임베딩 생성 → 메모리 저장

    한 번의 인덱싱 작업 동안 처리한 파일/청크/실패 개수를 인스턴스 필드로
    누적해 두었다가, stats 프로퍼티로 통계를 돌려준다.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        memory_store: Any,  # LongTermMemory
    ) -> None:
        """
        인덱서를 초기화한다.

        Args:
            model_provider: 임베딩을 만들어 주는 객체. embed() 메서드를 쓴다.
            memory_store: 청크를 저장할 장기 메모리 저장소(LongTermMemory).
                          순환 import를 피하려고 타입은 Any로 느슨하게 둔다.
        """
        # 임베딩 생성기와 저장소는 외부에서 주입받는다(의존성 주입).
        self._model_provider = model_provider
        self._memory = memory_store
        # 아래 세 개는 이번 인덱싱 작업의 진행 상황을 세는 카운터다.
        self._indexed_files: int = 0
        self._indexed_chunks: int = 0
        self._failed_files: int = 0

    async def index_directory(
        self,
        path: str,
        extensions: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        디렉토리를 재귀적으로 훑어 대상이 되는 모든 파일을 인덱싱한다.

        부트스트랩 시 백그라운드에서 호출되는 최상위 진입점이다.
        먼저 대상 파일 목록을 모두 수집한 뒤, 파일별로 index_file()을 돌린다.

        Args:
            path: 인덱싱을 시작할 루트 디렉토리 경로.
            extensions: 대상 확장자 집합. None이면 INDEXABLE_EXTENSIONS 기본값.

        Returns:
            인덱싱 통계 딕셔너리(indexed_files, indexed_chunks, failed_files).
        """
        # 호출자가 확장자를 안 주면 모듈 기본 집합을 쓴다.
        target_exts = extensions or INDEXABLE_EXTENSIONS
        # 이번 작업의 카운터를 0으로 초기화한다(이전 호출 값이 남지 않도록).
        self._indexed_files = 0
        self._indexed_chunks = 0
        self._failed_files = 0

        root = Path(path)
        # 애초에 디렉토리가 아니면 할 일이 없으니 경고 후 현재 통계만 돌려준다.
        if not root.is_dir():
            logger.warning("인덱싱 대상이 디렉토리가 아닙니다: %s", path)
            return self.stats

        # 1단계: 조건에 맞는 대상 파일을 먼저 전부 수집한다.
        files_to_index: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # os.walk가 넘겨주는 dirnames 리스트를 그 자리에서(slice 대입) 걸러내면
            # os.walk가 이후에 그 하위 디렉토리로 내려가지 않는다 → 제외 폴더 스킵.
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

            for fname in filenames:
                fpath = Path(dirpath) / fname
                # 확장자가 대상이고, 크기 제한 이하인 파일만 목록에 담는다.
                if fpath.suffix.lower() in target_exts and fpath.stat().st_size <= MAX_FILE_SIZE:
                    files_to_index.append(fpath)

        logger.info("인덱싱 대상: %d개 파일 (%s)", len(files_to_index), path)

        # 2단계: 수집한 파일을 하나씩 인덱싱한다.
        for fpath in files_to_index:
            try:
                await self.index_file(str(fpath))
            except Exception as e:
                # 파일 하나가 실패해도 전체 작업은 계속 진행한다(집계만 하고 넘어감).
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
        단일 파일 하나를 인덱싱한다.

        처리 단계: 파일 읽기 → 청크 분할 → 배치 임베딩 → 메모리에 저장.
        읽기 실패나 빈 내용 등 진행할 수 없는 경우에는 0을 돌려주고 조용히 끝낸다.

        Args:
            file_path: 인덱싱할 파일의 절대 경로(문자열).

        Returns:
            실제로 저장에 성공한 청크 개수.
        """
        fpath = Path(file_path)
        # 파일이 아니면(경로가 사라졌거나 디렉토리면) 처리하지 않는다.
        if not fpath.is_file():
            return 0

        # 파일 읽기. UTF-8로 디코드하되 깨진 바이트는 대체 문자로 바꿔(errors=replace)
        # 인코딩 오류로 통째로 실패하는 일을 막는다.
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("파일 읽기 실패: %s — %s", file_path, e)
            return 0

        # 공백만 있는 빈 파일이면 색인할 내용이 없으니 건너뛴다.
        if not text.strip():
            return 0

        # 청크 분할 방식은 파일 종류에 따라 다르다.
        # 아래 확장자들은 "코드"로 취급해 함수/클래스 경계 기준으로 나눈다.
        is_code = fpath.suffix.lower() in {".py", ".js", ".ts", ".go", ".rs", ".java"}
        chunks = _split_into_chunks(text, is_code=is_code)

        # 나눈 청크가 하나도 없으면 저장할 것도 없다.
        if not chunks:
            return 0

        # 임베딩에 넣을 텍스트를 준비한다.
        # 각 청크 앞에 "File: 경로"를 붙이는데, 이렇게 하면 임베딩에 파일 위치
        # 맥락이 섞여 들어가 검색 정확도가 올라간다(원문 청크는 따로 저장).
        relative_path = str(fpath)
        texts_to_embed = [
            f"File: {relative_path}\n{chunk}" for chunk in chunks
        ]

        chunk_count = 0
        # 청크를 EMBED_BATCH_SIZE개씩 잘라 배치 단위로 처리한다.
        for i in range(0, len(texts_to_embed), EMBED_BATCH_SIZE):
            # batch: 임베딩에 넣을(경로 접두어 포함) 텍스트 묶음.
            batch = texts_to_embed[i : i + EMBED_BATCH_SIZE]
            # batch_chunks: 실제 저장할 원문 청크 묶음(접두어 없이 순수 내용).
            batch_chunks = chunks[i : i + EMBED_BATCH_SIZE]

            try:
                # 배치를 통째로 임베딩 서버에 보내 벡터 리스트를 받는다.
                embeddings = await self._model_provider.embed(batch)
            except Exception as e:
                # 이 배치만 실패한 것이므로 다음 배치는 계속 시도한다.
                logger.debug("임베딩 생성 실패 (배치 %d): %s", i, e)
                continue

            # 배치 안의 (원문 청크, 임베딩 벡터) 쌍을 하나씩 저장한다.
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                # 전체 청크 목록에서의 실제 인덱스(배치 시작 위치 + 배치 내 위치).
                chunk_idx = i + j
                entry = MemoryEntry(
                    memory_type=MemoryType.SEMANTIC,
                    content=chunk,
                    # key는 청크를 유일하게 식별한다: rag:파일경로:chunk_번호
                    key=f"rag:{relative_path}:chunk_{chunk_idx}",
                    # 태그로 나중에 rag/확장자/파일명 기준 필터링이 가능하다.
                    tags=["rag", fpath.suffix.lstrip("."), fpath.name],
                    importance=0.8,  # RAG 청크는 중요도를 높게 줘 장기 보존시킨다
                    embedding=embedding,
                    metadata={
                        "file_path": relative_path,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "source": "rag_indexer",
                    },
                )
                # 장기 메모리에 저장(내부적으로 pgvector에 벡터가 들어간다).
                await self._memory.add(entry)
                chunk_count += 1

        # 파일 하나를 끝냈으니 전체 통계 카운터를 갱신한다.
        self._indexed_files += 1
        self._indexed_chunks += chunk_count
        return chunk_count

    @property
    def stats(self) -> dict[str, Any]:
        """이번 인덱싱 작업의 누적 통계를 딕셔너리로 반환한다."""
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
    텍스트를 검색하기 좋은 크기의 청크 리스트로 나눈다.

    파일 종류에 따라 분할 전략과 목표 크기가 다르다:
      - 코드 파일: 함수/클래스 경계를 인식해서 분할한다 (약 500자).
      - 문서 파일: 단락(빈 줄) 경계에서 분할한다 (약 1000자).

    왜 크기를 다르게 두는가:
      - 코드는 함수 하나가 의미 있는 검색 단위라 작게 잡는다.
      - 문서는 단락이 자연스러운 의미 단위라 더 크게 잡는다.
      - 8K 컨텍스트에서 top_k=3 × 500자 ≈ 500토큰 정도라 여유가 있다.

    Args:
        text: 나눌 원본 텍스트.
        is_code: 코드 파일이면 True, 문서면 False.

    Returns:
        분할된 청크 문자열 리스트.
    """
    # 코드/문서에 따라 전용 분할 함수로 위임한다.
    if is_code:
        return _split_code_chunks(text, max_size=CODE_CHUNK_SIZE)
    return _split_text_chunks(text, max_size=DOC_CHUNK_SIZE)


def _split_code_chunks(text: str, max_size: int = CODE_CHUNK_SIZE) -> list[str]:
    """
    코드를 함수/클래스 경계를 기준으로 청크로 나눈다.

    def/class 같은 정의 시작 지점을 만나면 그 앞까지를 한 청크로 끊는다.
    이렇게 하면 함수 하나가 되도록 온전히 한 청크에 담겨 검색 품질이 좋아진다.
    다만 한 함수가 너무 길면 max_size에서 강제로 끊어 청크가 과도하게
    커지는 것을 막는다.

    Args:
        text: 나눌 코드 텍스트.
        max_size: 청크 하나의 최대 문자 수(초과 시 강제 분할).

    Returns:
        분할된 코드 청크 리스트.
    """
    # 함수/클래스 정의가 시작되는 줄을 잡는 정규식 패턴.
    # Python(def/async def/class), JS(function/const/export), Go(func) 등을 커버한다.
    # re.MULTILINE으로 각 줄 맨 앞(^)에서 매칭되게 한다.
    boundary_pattern = re.compile(
        r"^(?:def |async def |class |function |const |export |func )",
        re.MULTILINE,
    )

    chunks: list[str] = []
    lines = text.split("\n")
    current_chunk: list[str] = []  # 지금 모으고 있는 청크의 줄 목록
    current_size = 0  # 지금 청크의 대략적인 문자 수

    for line in lines:
        # 함수/클래스 경계를 만났고, 이미 어느 정도(50자 초과) 모았다면
        # 지금까지 모은 것을 한 청크로 확정하고 새 청크를 시작한다.
        # 50자 조건은 def 바로 위 데코레이터/주석 몇 줄에서 성급히 끊기는 것을 막는다.
        if boundary_pattern.match(line.strip()) and current_size > 50:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_chunk = []
            current_size = 0

        # 현재 줄을 청크에 추가한다. +1은 join으로 복원될 개행 문자 몫이다.
        current_chunk.append(line)
        current_size += len(line) + 1

        # 경계를 못 만난 채 최대 크기를 넘겼으면 여기서 강제로 끊는다.
        if current_size >= max_size:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_chunk = []
            current_size = 0

    # 반복이 끝난 뒤 아직 안 비운 마지막 청크가 있으면 마저 담는다.
    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _split_text_chunks(text: str, max_size: int = DOC_CHUNK_SIZE) -> list[str]:
    """
    문서 텍스트를 단락 경계를 기준으로 청크로 나눈다.

    먼저 빈 줄(단락 구분)로 단락을 나눈 뒤, 단락들을 max_size를 넘지 않는
    선에서 하나의 청크로 모아 붙인다. 단락 사이 구분은 빈 줄 두 개("\\n\\n")로
    복원한다.

    Args:
        text: 나눌 문서 텍스트.
        max_size: 청크 하나의 최대 문자 수.

    Returns:
        분할된 문서 청크 리스트.
    """
    # 빈 줄(공백만 있는 줄 포함)을 기준으로 단락 단위로 쪼갠다.
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current_chunk: list[str] = []  # 지금 모으는 청크에 담긴 단락들
    current_size = 0  # 지금 청크의 대략적인 문자 수

    for para in paragraphs:
        para = para.strip()
        # 공백뿐인 단락은 건너뛴다.
        if not para:
            continue

        # 이 단락을 더하면 최대 크기를 넘고, 이미 모아 둔 단락이 있다면
        # 여기서 한 청크로 확정하고 새 청크를 시작한다.
        if current_size + len(para) > max_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        # 현재 단락을 청크에 추가한다.
        current_chunk.append(para)
        current_size += len(para)

    # 반복이 끝난 뒤 남은 마지막 청크를 마저 담는다.
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
