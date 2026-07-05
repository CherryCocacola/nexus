"""
문서 인제스트(수집/적재) 파이프라인 — DocumentIngestPipeline.

이 파일 한 줄 요약:
  "문서 파일 1개를 읽어 → 구조를 살려 잘게 나누고 → 각 조각을 벡터로 바꿔 →
  지식 DB(tb_knowledge)에 저장"하는 전 과정을 하나로 묶어 지휘(오케스트레이션)한다.
  RAG(검색 증강 생성)가 참고할 지식을 쌓아 두는 '입구' 역할이며, v7.3 사양서
  Part 2.1 에 정의된 전체 파이프라인을 코드로 구현한 것이다.

전체 흐름 (ingest_file 한 번 호출로 아래 5단계가 순서대로 진행):
  1. parser_registry 에서 파일 확장자에 맞는 파서를 찾아 parse() 실행
     → DocumentTree(문서를 제목/절/문단 등 트리 구조로 표현한 객체)
  2. StructureAwareChunker.chunk() 로 트리를 검색 단위 조각으로 분할
     → list[DocumentChunk] (제목 경로/페이지 같은 구조 정보를 유지한 조각들)
  3. model_provider.embed(["passage: ...", ...]) 로 각 조각을 임베딩
     → 1024차원 실수 벡터(의미가 비슷한 글은 벡터도 가깝다)
  4. DocumentChunk → KnowledgeEntry 로 변환(구조 메타는 metadata jsonb 에 담음)
  5. KnowledgeStore.add() 로 UPSERT 적재 — 같은 글을 다시 넣어도 중복이 안 생김(멱등)

주요 구성 요소:
  - 클래스 DocumentIngestPipeline: 위 5단계를 조율하는 본체.
  - 공개 메서드 ingest_file(): 유일한 진입점. 파일 1건을 처리하고 요약 dict 반환.
  - 내부 헬퍼: _embed_and_store(임베딩+적재), _to_entry(형 변환), _summary(결과 요약).

의존성 주입(DI)을 쓰는 이유:
  parser_registry / model_provider / knowledge_store 를 직접 만들지 않고 생성자로
  받는다. 이렇게 하면 (1) 테스트에서 실제 서버 대신 가짜(fake) 구현을 끼워 넣기
  쉽고, (2) 이 모듈이 특정 구체 구현(LocalModelProvider 등)에 딱 붙지 않아
  나중에 구현을 갈아 끼워도 이 파일은 손댈 필요가 없다.

의존성 방향 (아키텍처 규칙 P2 준수):
  core.ingest → core.rag(KnowledgeStore/KnowledgeEntry), core.model(ModelProvider)
  방향으로만 import 한다. 반대 방향(하위가 상위를 import)은 없어 순환 import 위험이 없다.

에어갭/fail-soft 설계 원칙:
  이 프로젝트는 폐쇄망에서 도는 만큼 외부 호출이 없고, 부분 실패에 강해야 한다.
  임베딩/적재가 일부 실패해도 파일 전체 처리를 멈추지 않고(가능한 조각만 적재),
  실패는 구체 예외로만 잡아 로깅한 뒤 계속 진행한다. 또 dry_run 옵션을 주면
  DB에 아무것도 쓰지 않고 파싱·청킹 결과만 미리 점검할 수 있다.

작성자: 이현수 / 작성일: 2026-07-05
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

# 적재 소스 식별자 — tb_knowledge.source 컬럼에 그대로 저장된다.
# 이 값으로 "문서 인제스트로 들어온 지식"을 kowiki(위키 덤프) 등 다른 출처와
# 구분할 수 있다. 나중에 특정 출처만 재적재/삭제할 때 필터 키로도 쓴다. (v7.3 Part 2.5)
INGEST_SOURCE = "docingest"

# 임베딩 모델(e5-large)이 권장하는 접두사. e5 계열은 "문서(passage)"인지
# "질문(query)"인지에 따라 앞에 붙이는 말이 달라야 성능이 잘 나온다. (v7.1 Part 2)
# 이 파이프라인은 '적재' 담당이라 문서 조각에만 "passage: " 를 붙인다.
# (검색 시점의 질의는 별도 코드에서 "query: " 를 붙여 임베딩한다.)
_PASSAGE_PREFIX = "passage: "

# 임베딩 배치 크기 — 임베딩 서버(/v1/embed)에 한 번에 몇 개 조각을 보낼지.
# 여러 조각을 한 요청에 묶어 보내면 왕복 횟수가 줄어 빠르다. 단,
# 너무 크면 서버 메모리 부담/타임아웃 위험이 커지고, 너무 작으면 호출
# 오버헤드가 늘어난다. 32는 그 절충값이다.
_EMBED_BATCH_SIZE = 32


class DocumentIngestPipeline:
    """
    문서 파일을 '구조를 보존한 채' 조각내어 tb_knowledge 에 적재하는 파이프라인.

    '구조 보존'이란 제목/절/페이지 같은 문서의 뼈대를 조각에 함께 저장한다는 뜻이다.
    덕분에 나중에 검색했을 때 "이 내용이 어느 문서의 어느 절에서 왔는지"를 알 수 있다.

    사용 예:
        pipeline = DocumentIngestPipeline(registry, model, store)
        summary = await pipeline.ingest_file("보고서.pptx")
        print(summary["ingested"])  # 실제로 몇 조각이 DB에 들어갔는지

    공개 메서드:
      - ingest_file(path, dry_run=False): 파일 1건을 처리하고 결과 요약 dict 를 반환.
        (그 외 메서드는 모두 내부용 헬퍼로, 앞에 밑줄(_)을 붙여 구분한다.)
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        model_provider: ModelProvider,
        knowledge_store: KnowledgeStore,
        chunker: StructureAwareChunker | None = None,
    ) -> None:
        """
        파이프라인이 쓸 협력 객체들을 주입받아 인스턴스 필드에 보관한다.

        생성자는 실제 일을 하지 않고 '준비'만 한다. 실제 처리는 ingest_file()에서 일어난다.

        Args:
            parser_registry: 파일 확장자를 보고 알맞은 파서를 골라 주는 레지스트리.
                (예: .pptx 를 처리하려면 PPTX 파서가 여기 등록돼 있어야 한다.)
            model_provider: embed(texts) 로 텍스트를 벡터로 바꿔 주는 모델 프로바이더.
                내부적으로 LAN 임베딩 서버(/v1/embed)를 호출한다.
            knowledge_store: tb_knowledge 에 UPSERT(있으면 갱신, 없으면 삽입)하는 적재소.
            chunker: 문서를 조각으로 나누는 청커. 넘기지 않으면 기본 구현을 새로 만든다.
                (테스트에서 다른 분할 규칙을 주입하고 싶을 때를 위해 열어 둔 매개변수.)
        """
        # 아래 4개 필드는 모두 밑줄로 시작 — "밖에서 직접 건드리지 말라"는 관례상 표시.
        self._registry = parser_registry
        self._model = model_provider
        self._store = knowledge_store
        # chunker 가 None(미지정)이면 or 뒤의 기본 청커가 대신 쓰인다.
        self._chunker = chunker or StructureAwareChunker()

    # ─── 공개 진입점 ───

    async def ingest_file(self, path: str | Path, dry_run: bool = False) -> dict[str, Any]:
        """
        파일 1건을 인제스트한다 — 이 클래스의 유일한 공개 진입점이자 전체 흐름의 지휘자.

        모듈 상단에서 설명한 5단계를 여기서 순서대로 호출한다. 각 단계는 실패에
        관대하게(fail-soft) 설계돼 있어, 문제가 생겨도 가능한 만큼 처리하고
        무엇이 잘못됐는지는 반환 dict 에 담아 호출자에게 알린다.

        Args:
            path: 입력 파일 경로. 문자열이든 Path 든 받아서 내부에서 Path 로 통일한다.
            dry_run: True 면 임베딩/적재를 건너뛰고 파싱·청킹까지만 수행한다.
                DB에 아무 부작용 없이 "이 문서가 몇 조각으로, 어떻게 나뉘는지"만
                미리 확인하고 싶을 때 쓴다.

        Returns:
            처리 결과를 담은 요약 dict. 주요 키:
              {
                "file": str, "doc_format": str, "title": str,
                "chunk_count": int, "parent_count": int, "child_count": int,
                "warning_count": int, "warnings": list[str],
                "ingested": int,            # 실제 적재된 행 수(dry_run 이면 항상 0)
                "errors": list[str],        # 임베딩/적재 도중 발생한 오류 메시지들
                "dry_run": bool,
              }
        """
        # 입력을 문자열/Path 어느 쪽으로 받든 Path 객체로 통일 — 이후 코드가 한 종류만 다룸.
        file_path = Path(path)

        # 1) 파서 선택 — 지원하는 확장자가 아니면 처리할 방법이 없다.
        #    fail-closed 원칙: 애매하면 '안전하게 아무것도 안 함'. 여기선 경고만 남기고
        #    빈 요약(오류 메시지 포함)을 돌려줘, 배치 처리 중 한 파일 때문에 멈추지 않게 한다.
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

        # 2) 파싱 — 파일을 읽어 DocumentTree(제목/절/문단 트리)로 변환.
        #    파서는 fail-soft: 깨진 부분이 있어도 예외를 던지기보다 '부분 트리 +
        #    경고(warnings) 목록'으로 표현해 반환한다. 그래서 여기선 그냥 받으면 된다.
        tree = await parser.parse(file_path)

        # 3) 청킹 — 트리를 검색에 알맞은 크기의 조각(DocumentChunk) 리스트로 분할.
        chunks = self._chunker.chunk(tree)

        # dry_run 모드면 여기서 조기 반환한다. 임베딩/적재는 건너뛰므로 DB 변경이 전혀 없다.
        if dry_run:
            return self._summary(
                file_path, tree=tree, chunks=chunks, ingested=0, errors=[], dry_run=True
            )

        # 4~5) 각 조각을 임베딩해 KnowledgeEntry 로 바꾼 뒤 DB에 적재한다.
        #    실제 적재된 행 수와 오류 목록을 돌려받는다.
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
        조각들을 배치 단위로 임베딩한 뒤 KnowledgeEntry 로 변환해 UPSERT 적재한다.

        여기가 파이프라인에서 실제로 GPU/DB 자원을 쓰는 핵심 구간이다. 성능을 위해
        조각을 낱개가 아니라 _EMBED_BATCH_SIZE(32) 묶음 단위로 임베딩한다.

        Args:
            tree: 원본 문서 트리. 제목/포맷 등 조각 공통 메타를 여기서 가져온다.
            chunks: 적재 대상 조각 리스트(청킹 결과 그대로).

        Returns:
            (적재에 성공한 행 수, 오류 메시지 목록) 튜플.

        fail-soft 설계: 임베딩 배치 하나가 실패하거나 개별 적재가 실패해도 전체를
        멈추지 않는다. 오류는 모아 두고 남은 것을 계속 처리한다. 단, 예외는
        광범위한 bare except 로 삼키지 않고 구체적인 예외 군만 포착한다
        (프로젝트 anti-pattern #8 '에러 무시 금지' 준수).
        """
        # 조각이 하나도 없으면 할 일이 없다 — 즉시 빈 결과 반환(불필요한 반복 회피).
        if not chunks:
            return 0, []

        total = len(chunks)  # 전체 조각 수 — 각 조각의 total_chunks 메타에 넣을 값.
        ingested = 0  # 지금까지 DB에 성공적으로 넣은 행 수(누적 카운터).
        errors: list[str] = []  # 발생한 오류 메시지를 모아 마지막에 함께 반환.

        # 조각 리스트를 앞에서부터 32개씩 잘라 배치 단위로 순회한다.
        # start 는 각 배치의 시작 인덱스(0, 32, 64, ...).
        for start in range(0, total, _EMBED_BATCH_SIZE):
            batch = chunks[start : start + _EMBED_BATCH_SIZE]
            # 임베딩 서버에 보낼 텍스트를 만든다. e5 규칙대로 앞에 "passage: " 를 붙인다.
            texts = [f"{_PASSAGE_PREFIX}{c.content}" for c in batch]

            # 배치 임베딩 시도 — 실패하면 이 배치만 통째로 건너뛰고 다음 배치로 넘어간다.
            try:
                embeddings = await self._model.embed(texts)
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                # embed()는 내부에서 httpx 통신 오류/응답 파싱 오류 등을 던질 수 있다.
                # 그런 '예상 가능한' 예외 군만 잡아 기록하고 넘어간다. 그 외 예상 못 한
                # 예외는 일부러 잡지 않아 상위로 전파시킨다(진짜 버그를 숨기지 않기 위해).
                msg = f"임베딩 실패(배치 {start}~{start + len(batch)}): {type(e).__name__}: {e}"
                logger.error(msg)
                errors.append(msg)
                continue

            # 방어적 검증: 보낸 조각 수와 돌아온 임베딩 수는 반드시 같아야 한다.
            # 다르면 어느 조각이 어느 벡터인지 짝짓기가 어긋나므로, 잘못 적재하느니
            # 이 배치를 통째로 버린다(데이터 오염 방지).
            if len(embeddings) != len(batch):
                msg = (
                    f"임베딩 개수 불일치(배치 {start}): "
                    f"청크 {len(batch)} vs 임베딩 {len(embeddings)}"
                )
                logger.error(msg)
                errors.append(msg)
                continue

            # 배치 내부를 돌며 (조각, 그 조각의 임베딩) 쌍을 하나씩 적재한다.
            # zip 으로 둘을 짝지어 순회하고, offset 으로 배치 내 상대 위치를 센다.
            for offset, (chunk, embedding) in enumerate(zip(batch, embeddings, strict=False)):
                # 전체 문서 기준의 절대 조각 번호 = 배치 시작 위치 + 배치 내 위치.
                chunk_index = start + offset
                # 조각 + 임베딩 + 위치 정보를 DB 저장용 KnowledgeEntry 로 변환.
                entry = self._to_entry(
                    tree=tree,
                    chunk=chunk,
                    embedding=embedding,
                    chunk_index=chunk_index,
                    total_chunks=total,
                )
                # 적재도 조각 단위로 감싼다 — 한 조각 저장이 실패해도 나머지는 계속 진행.
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
        DocumentChunk(청킹 결과) 를 KnowledgeEntry(DB 저장용 모델) 로 변환한다.
        (v7.3 Part 2.5 에 정의된 필드 매핑을 그대로 코드로 옮긴 것.)

        staticmethod 인 이유: self 상태를 전혀 안 쓰고 입력만으로 결과가 정해지는
        순수 변환이라, 인스턴스에 묶어 둘 필요가 없다.

        핵심 매핑 규칙:
        - section: 조각의 heading_path(제목 경로)를 " > " 로 이어 한 문자열로 만든다.
          예: ["1장", "개요"] → "1장 > 개요". 기존 section 컬럼을 그대로 활용한다.
        - metadata(jsonb): 구조 메타(element_type/heading_path/page/
          parent_chunk_id/is_parent/doc_format 등)를 새 컬럼을 만들지 않고 전부
          이 jsonb 한 곳에 담는다. 스키마 변경 없이 유연하게 정보를 싣는 방식.
        - id: KnowledgeEntry 가 source/title/section/chunk_index 를 SHA-256 해시해
          자동 생성한다. 같은 문서를 다시 적재하면 같은 id 가 나와 UPSERT 로 덮어써지고,
          그래서 중복 없이 멱등(idempotent)하게 재적재된다.

        주의: KnowledgeEntry.embedding 필드는 tuple[float, ...] 타입을 기대한다.
        그래서 list 로 온 embedding 을 tuple(...) 로 감싸 넘긴다.
        (실제 벡터 문자열 직렬화는 KnowledgeStore.add 안의 _format_vector 가 처리한다.)
        """
        # heading_path 가 비어 있으면(최상위 등) section 은 None 으로 둔다.
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
        """
        ingest_file 이 호출자에게 돌려줄 결과 요약 dict 를 만든다.

        성공/부분성공/실패 어느 경로에서든 같은 모양의 dict 를 만들어 반환 형태를
        일관되게 유지하려고 별도 헬퍼로 분리했다. tree 가 None 일 수 있어(파서 없음
        등) 그 경우에도 안전하게 기본값을 채운다.
        """
        # 부모 조각(상위 컨텍스트용) 수와 자식 조각 수를 나눠 센다 — 통계/디버깅용.
        parent_count = sum(1 for c in chunks if c.is_parent)
        child_count = len(chunks) - parent_count
        # tree 가 없으면(파싱 자체가 없었으면) 경고 목록도 빈 리스트로 둔다.
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
