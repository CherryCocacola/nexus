"""
문서 인제스트 파이프라인 — DocumentIngestPipeline.

역할:
  "파일 1개 → 파싱 → 청킹 → 임베딩 → tb_knowledge 적재"의 오케스트레이션.
  v7.3 Part 2.1 전체 파이프라인의 코드 구현이다.

흐름:
  ingest_file(path)
    1. parser_registry 에서 확장자에 맞는 파서를 찾아 parse() → DocumentTree
    2. StructureAwareChunker.chunk() → list[DocumentChunk]
    3. model_provider.embed(["passage: ...", ...]) → 임베딩(1024차원)
    4. DocumentChunk → KnowledgeEntry 변환(metadata jsonb 에 구조 메타)
    5. KnowledgeStore.add() UPSERT 적재(멱등)

의존성 주입:
  parser_registry / model_provider / knowledge_store 를 생성자로 받는다.
  이렇게 하면 테스트에서 가짜(fake) 구현을 끼워 넣기 쉽고, 본 모듈은 구체
  구현(LocalModelProvider 등)에 직접 의존하지 않는다.

의존성 방향 (P2):
  core.ingest → core.rag(KnowledgeStore/KnowledgeEntry), core.model(ModelProvider).
  역방향 import 없음.

에어갭/fail-soft:
  임베딩/적재 실패는 구체 예외로 포착해 로깅하고, 파일 단위 본류를 중단하지
  않는다(가능한 만큼 적재). dry_run 으로 적재 없이 청킹까지만 점검 가능.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.ingest.chunker import StructureAwareChunker
from core.ingest.parser_base import ParserRegistry
from core.ingest.types import DocumentChunk, DocumentTree
from core.model.inference import ModelProvider
from core.rag.knowledge_store import KnowledgeEntry, KnowledgeStore

logger = logging.getLogger("nexus.ingest.pipeline")

# 적재 소스 식별자 — kowiki 등과 구분된다(v7.3 Part 2.5).
INGEST_SOURCE = "docingest"

# e5-large 권장 접두사(v7.1 Part 2). 문서 청크는 "passage:" 로 임베딩한다.
# 검색 질의는 "query:" 를 쓰지만, 인제스트는 적재(passage)만 담당한다.
_PASSAGE_PREFIX = "passage: "

# 임베딩 배치 크기 — 한 번의 /v1/embed 요청에 보낼 청크 수.
# 너무 크면 임베딩 서버 메모리/타임아웃, 너무 작으면 호출 오버헤드.
_EMBED_BATCH_SIZE = 32


class DocumentIngestPipeline:
    """
    문서 파일을 구조 보존 청킹해 tb_knowledge 에 적재하는 파이프라인.

    공개 메서드:
      - ingest_file(path, dry_run=False): 파일 1건 처리, 요약 dict 반환.
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        model_provider: ModelProvider,
        knowledge_store: KnowledgeStore,
        chunker: StructureAwareChunker | None = None,
    ) -> None:
        """
        Args:
            parser_registry: 확장자→파서 매핑(PPTX 파서 등록되어 있어야 함).
            model_provider: embed(texts) 를 제공하는 모델 프로바이더(LAN /v1/embed).
            knowledge_store: tb_knowledge UPSERT 적재소.
            chunker: 청커(미지정 시 기본값으로 생성).
        """
        self._registry = parser_registry
        self._model = model_provider
        self._store = knowledge_store
        self._chunker = chunker or StructureAwareChunker()

    # ─── 공개 진입점 ───

    async def ingest_file(self, path: str | Path, dry_run: bool = False) -> dict[str, Any]:
        """
        파일 1건을 인제스트한다.

        Args:
            path: 입력 파일 경로.
            dry_run: True 면 임베딩/적재를 건너뛰고 파싱·청킹까지만 수행
                (적재 부작용 없이 청크 결과를 점검).

        Returns:
            요약 dict:
              {
                "file": str, "doc_format": str, "title": str,
                "chunk_count": int, "parent_count": int, "child_count": int,
                "warning_count": int, "warnings": list[str],
                "ingested": int,            # 실제 적재 행수(dry_run 이면 0)
                "errors": list[str],        # 임베딩/적재 중 발생한 오류 메시지
                "dry_run": bool,
              }
        """
        file_path = Path(path)

        # 1) 파서 선택 — fail-closed: 지원 파서가 없으면 빈 요약 반환.
        parser = self._registry.get_for_path(file_path)
        if parser is None:
            logger.warning("지원하는 파서 없음 — 건너뜀: %s", file_path)
            return self._summary(
                file_path,
                tree=None,
                chunks=[],
                ingested=0,
                errors=[f"지원하는 파서가 없습니다: {file_path.suffix}"],
                dry_run=dry_run,
            )

        # 2) 파싱 (fail-soft — 파서가 부분 트리+warnings 로 표현).
        tree = await parser.parse(file_path)

        # 3) 청킹.
        chunks = self._chunker.chunk(tree)

        # dry_run 이면 여기서 종료(적재 없음).
        if dry_run:
            return self._summary(
                file_path, tree=tree, chunks=chunks, ingested=0, errors=[], dry_run=True
            )

        # 4) 임베딩 + 적재.
        ingested, errors = await self._embed_and_store(tree, chunks)

        return self._summary(
            file_path,
            tree=tree,
            chunks=chunks,
            ingested=ingested,
            errors=errors,
            dry_run=False,
        )

    # ─── 내부: 임베딩 + 적재 ───

    async def _embed_and_store(
        self, tree: DocumentTree, chunks: list[DocumentChunk]
    ) -> tuple[int, list[str]]:
        """
        청크들을 배치 임베딩한 뒤 KnowledgeEntry 로 변환해 UPSERT 적재한다.

        반환: (적재 성공 행수, 오류 메시지 목록).

        fail-soft: 임베딩 배치 또는 개별 적재가 실패해도 전체를 중단하지 않고
        오류를 모아 계속 진행한다(구체 예외만 포착 — anti-pattern #8).
        """
        if not chunks:
            return 0, []

        total = len(chunks)
        ingested = 0
        errors: list[str] = []

        # 청크를 임베딩 배치 단위로 나눠 처리한다.
        for start in range(0, total, _EMBED_BATCH_SIZE):
            batch = chunks[start : start + _EMBED_BATCH_SIZE]
            texts = [f"{_PASSAGE_PREFIX}{c.content}" for c in batch]

            # 임베딩 — 실패 시 이 배치는 건너뛰고 다음 배치로(fail-soft).
            try:
                embeddings = await self._model.embed(texts)
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                # ModelProvider.embed 는 httpx 오류 등을 raise 할 수 있다.
                # 구체 예외 군만 포착해 배치를 건너뛴다.
                msg = f"임베딩 실패(배치 {start}~{start + len(batch)}): {type(e).__name__}: {e}"
                logger.error(msg)
                errors.append(msg)
                continue

            # 임베딩 개수가 청크 수와 다르면 정합성 문제 — 이 배치 스킵.
            if len(embeddings) != len(batch):
                msg = (
                    f"임베딩 개수 불일치(배치 {start}): "
                    f"청크 {len(batch)} vs 임베딩 {len(embeddings)}"
                )
                logger.error(msg)
                errors.append(msg)
                continue

            # 배치 내 각 청크를 KnowledgeEntry 로 변환해 적재.
            for offset, (chunk, embedding) in enumerate(zip(batch, embeddings, strict=False)):
                chunk_index = start + offset
                entry = self._to_entry(
                    tree=tree,
                    chunk=chunk,
                    embedding=embedding,
                    chunk_index=chunk_index,
                    total_chunks=total,
                )
                try:
                    await self._store.add(entry)
                    ingested += 1
                except (OSError, ValueError, KeyError, RuntimeError) as e:
                    msg = f"적재 실패(청크 {chunk_index}): {type(e).__name__}: {e}"
                    logger.error(msg)
                    errors.append(msg)

        return ingested, errors

    # ─── 내부: DocumentChunk → KnowledgeEntry ───

    @staticmethod
    def _to_entry(
        tree: DocumentTree,
        chunk: DocumentChunk,
        embedding: list[float],
        chunk_index: int,
        total_chunks: int,
    ) -> KnowledgeEntry:
        """
        DocumentChunk 를 KnowledgeEntry 로 변환한다(v7.3 Part 2.5 매핑).

        - section: heading_path 를 " > " 로 이은 문자열(기존 section 컬럼 활용).
        - metadata jsonb: 구조 메타(element_type/heading_path/page/
          parent_chunk_id/is_parent/doc_format)를 컬럼 추가 없이 전부 담는다.
        - KnowledgeEntry.id 는 source/title/section/chunk_index 의 SHA-256 →
          재적재 멱등(UPSERT) 자동 보장.

        주의: KnowledgeEntry.embedding 은 tuple[float, ...] 타입이므로 tuple 로
        변환해 넘긴다(KnowledgeStore.add 가 _format_vector 로 직렬화).
        """
        section = " > ".join(chunk.heading_path) if chunk.heading_path else None

        return KnowledgeEntry(
            source=INGEST_SOURCE,
            title=tree.title,
            section=section,
            content=chunk.content,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            embedding=tuple(embedding),
            metadata={
                "element_type": chunk.element_type.value,
                "heading_path": list(chunk.heading_path),
                "page": chunk.page,
                "parent_chunk_id": chunk.parent_chunk_id,
                "is_parent": chunk.is_parent,
                "ingested_by": "doc_ingest_pipeline",
                "doc_format": tree.doc_format,
            },
        )

    # ─── 내부: 요약 ───

    @staticmethod
    def _summary(
        file_path: Path,
        tree: DocumentTree | None,
        chunks: list[DocumentChunk],
        ingested: int,
        errors: list[str],
        dry_run: bool,
    ) -> dict[str, Any]:
        """ingest_file 의 반환 요약 dict 를 구성한다."""
        parent_count = sum(1 for c in chunks if c.is_parent)
        child_count = len(chunks) - parent_count
        warnings = list(tree.warnings) if tree else []
        return {
            "file": str(file_path),
            "doc_format": tree.doc_format if tree else "",
            "title": tree.title if tree else file_path.stem,
            "chunk_count": len(chunks),
            "parent_count": parent_count,
            "child_count": child_count,
            "warning_count": len(warnings),
            "warnings": warnings,
            "ingested": ingested,
            "errors": errors,
            "dry_run": dry_run,
        }
