"""
대량 문서 배치 인제스트 스크립트 (v7.3 단계 10, 2026-06-02).

역할:
  디렉토리 하나를 재귀 순회하며 지원 포맷(PPTX/PDF/HWPX/이미지/.hwp)의 문서를
  일괄로 파싱→구조보존 청킹→임베딩→tb_knowledge(source="docingest") 적재한다.
  prepare_kowiki.py 가 "위키 덤프 1개"를 적재하는 것과 짝을 이루는, "사내 문서
  디렉토리"용 배치 적재기다.

이 스크립트는 **운영 Nexus 런타임 코드가 아니라 "데이터 준비" 용도**다. 에어갭
원칙상 Nexus 런타임은 외부 네트워크를 호출하지 않지만, 최초 코퍼스 준비는 별도
단계로 허용된다(progress.md RAG 지식 베이스 계획 근거). 임베딩 서버(:8002)와
PostgreSQL(:5440)은 모두 LAN 주소이므로 에어갭을 준수한다.

설계(prepare_kowiki.py 패턴 답습):
  - argparse 로 입력 디렉토리/포맷/임베딩 URL/PG dsn/스택/한계를 받는다.
  - core 의 DocumentIngestPipeline + ParserRegistry + LocalModelProvider +
    KnowledgeStore 를 그대로 조립한다(docingest MCP 서버와 동일 조립).
  - 파일별 try/except 격리 — 한 파일이 실패해도 전체 배치를 중단하지 않는다
    (anti-pattern #8: bare except 금지, 구체 예외만 포착).
  - 진행 카운터를 주기적으로 출력해 tmux 등에서 장기 실행을 관찰할 수 있게 한다.

실행 예시:
  1. 디렉토리 적재(자동 스택 선택 — GPU 감지):
       python scripts/prepare_documents.py \\
         --input /opt/nexus-gpu/corpora/docs \\
         --embed-url http://192.168.21.112:8002 \\
         --pg "postgresql://nexus:idino%4012@192.168.10.39:5440/nexus"

  2. 스모크(첫 5개 파일, 적재 없이 파싱·청킹만):
       python scripts/prepare_documents.py --input ./docs --dry-run --limit 5

  3. 대량 적재 후 벡터 인덱스(ivfflat) 재빌드:
       python scripts/prepare_documents.py --build-index \\
         --pg "postgresql://nexus:idino%4012@192.168.10.39:5440/nexus"

안전장치:
  - `--dry-run`: 파싱/청킹까지만 수행하고 임베딩·DB 쓰기 생략.
  - `--limit N`: 상위 N개 파일만 처리(스모크 테스트).
  - 적재는 UPSERT(KnowledgeEntry.id = SHA-256)이므로 재실행해도 중복 없음.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus.scripts.prepare_documents")

# 프로젝트 루트를 sys.path에 추가 (스크립트를 독립 실행할 때 core/* import 가능하게).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# 진행 로그 출력 주기 — N개 파일마다 누적 카운터를 찍는다(장기 실행 관찰용).
_PROGRESS_EVERY = 20


# ─────────────────────────────────────────────
# GPU 감지 — 스택(light/high) 자동 선택용
# ─────────────────────────────────────────────
def _gpu_available() -> bool:
    """
    이 호스트에 CUDA GPU 가 있는지 fail-soft 로 판정한다.

    docingest MCP 서버의 동명 헬퍼와 동일한 정책:
      - GPU 가 있으면 DoclingParser(.pdf 고품질, requires_gpu=True)를 우선 등록.
      - torch 가 없거나 감지 중 오류가 나도 "GPU 없음"으로 간주해 경량 파서
        (pdfplumber)로 자연스럽게 폴백한다. 감지 실패가 배치 전체를 막지 않는다.
      - import 만 하며 설치 코드는 넣지 않는다(에어갭 규칙).
    """
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception as e:  # noqa: BLE001 — 감지 실패는 GPU 없음으로 흡수(폴백)
        logger.debug("GPU 감지 실패 — GPU 없음으로 간주: %s", e)
        return False


# ─────────────────────────────────────────────
# 파서 레지스트리 구성 — docingest MCP 서버와 동일 정책
# ─────────────────────────────────────────────
def _build_parser_registry(stack: str) -> Any:
    """
    스택(light/high)에 맞춰 ParserRegistry 를 구성한다.

    docingest_server.build_app 의 등록 정책을 그대로 따른다(어댑터 슬롯 구조):
      - PptxParser(.pptx) / PdfPlumberParser(.pdf 경량) / HwpxParser(.hwpx) /
        HwpViaLibreOfficeParser(.hwp) — 모두 청정(MIT/OWPML) CPU 파서, priority=0.
      - TesseractParser — 이미지(.png/.jpg/...)의 1차 후보이자 .pdf 최저 폴백
        (priority=-10). 디지털 텍스트 파서가 항상 .pdf 를 우선 처리한다.
      - DoclingParser(.pdf 고품질, requires_gpu=True) — stack="high" 일 때만
        priority=10 으로 등록해 pdfplumber 보다 우선시킨다. get_for_path 의
        can_parse 폴백 덕분에 Docling 이 못 다루는 파일은 자동으로 pdfplumber 로
        넘어간다. stack="light" 이면 아예 등록하지 않아 경량 파서만 남긴다.

    Args:
        stack: "light"(pdfplumber/tesseract, CPU) 또는 "high"(docling 추가, GPU).

    Returns:
        구성된 ParserRegistry.
    """
    from core.ingest.parser_base import ParserRegistry
    from core.ingest.parsers.hwp_libreoffice import HwpViaLibreOfficeParser
    from core.ingest.parsers.hwpx import HwpxParser
    from core.ingest.parsers.ocr_tesseract import TesseractParser
    from core.ingest.parsers.pdf_plumber import PdfPlumberParser
    from core.ingest.parsers.pptx import PptxParser

    registry = ParserRegistry()
    registry.register(PptxParser(), priority=0)  # .pptx
    registry.register(PdfPlumberParser(), priority=0)  # .pdf (경량, CPU)
    registry.register(HwpxParser(), priority=0)  # .hwpx
    registry.register(HwpViaLibreOfficeParser(), priority=0)  # .hwp (LibreOffice 변환)
    # 이미지 1차 + .pdf 최저 폴백 — 자세한 정책은 docingest_server 모듈 docstring 참조.
    registry.register(TesseractParser(), priority=-10)

    if stack == "high":
        # 고품질 PDF 어댑터 슬롯 — GPU 호스트에서만 의미가 있다.
        from core.ingest.parsers.docling_layout import DoclingParser

        registry.register(DoclingParser(), priority=10)  # .pdf (고품질, GPU)
        logger.info("스택=high — DoclingParser(.pdf 고품질) 우선 등록")
    else:
        logger.info("스택=light — PdfPlumberParser(.pdf 경량)만 사용")

    return registry


# ─────────────────────────────────────────────
# 입력 파일 순회 — 확장자 필터 + 재귀
# ─────────────────────────────────────────────
def iter_input_files(
    input_dir: Path,
    allowed_exts: set[str],
) -> list[Path]:
    """
    입력 디렉토리를 재귀 순회하며 허용 확장자 파일만 정렬해 반환한다.

    규칙:
      - 확장자 비교는 소문자·점 포함으로 정규화한다(대소문자 무시: .PDF == .pdf).
      - 디렉토리 트리를 정렬(sorted)해 처리 순서를 결정론적으로 만든다(재실행 시
        같은 순서 → --limit 스모크 결과 재현 가능).
      - 숨김/임시 파일(점으로 시작)은 제외한다.

    Args:
        input_dir: 순회할 루트 디렉토리.
        allowed_exts: 허용 확장자 집합(소문자·점 포함). 예: {".pdf", ".pptx"}.

    Returns:
        처리 대상 파일 경로 리스트(정렬됨).
    """
    files: list[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):  # 숨김/임시 파일 제외
            continue
        if path.suffix.lower() in allowed_exts:
            files.append(path)
    return files


# ─────────────────────────────────────────────
# 모델 프로바이더 / PG 풀 구성
# ─────────────────────────────────────────────
def _build_model_provider(config: Any, embed_url: str | None) -> Any:
    """
    임베딩 전용 LocalModelProvider 를 구성한다(docingest 서버와 동일).

    --embed-url 로 임베딩 서버를 덮어쓸 수 있고, 미지정 시 config 의
    gpu_server.embedding_url 을 쓴다(에어갭: 둘 다 LAN 주소여야 함).
    """
    from core.model.inference import LocalModelProvider

    embedding_base_url = embed_url or config.gpu_server.embedding_url
    return LocalModelProvider(
        base_url=config.gpu_server.url,
        api_key=config.scout.api_key,
        model_id=config.model.primary_model,
        embedding_model_id=config.model.embedding_model,
        embedding_base_url=embedding_base_url,
    )


async def _create_pg_pool(config: Any, dsn: str | None) -> Any | None:
    """
    PostgreSQL 커넥션 풀을 만든다 — 실패 시 None(인메모리 폴백).

    --pg dsn 이 주어지면 DSN 문자열로 연결하고(prepare_kowiki 방식), 없으면
    config 의 postgresql 섹션(host/port/...)으로 연결한다(docingest 방식).
    연결 실패는 구체 예외로 포착해 None 을 반환한다(배치를 막지 않음).
    """
    try:
        import asyncpg
    except ImportError:
        logger.warning("asyncpg 미설치 — 인메모리 폴백(실제 적재 불가)")
        return None

    try:
        if dsn:
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, timeout=10.0)
        else:
            pool = await asyncpg.create_pool(
                host=config.postgresql.host,
                port=config.postgresql.port,
                database=config.postgresql.database,
                user=config.postgresql.user,
                password=config.postgresql.password,
                min_size=1,
                max_size=4,
                timeout=10.0,
            )
        logger.info("PostgreSQL 연결 성공")
        return pool
    except Exception as e:  # noqa: BLE001 — 연결 실패는 인메모리 폴백으로 흡수
        logger.warning("PostgreSQL 연결 실패(인메모리 폴백): %s", e)
        return None


# ─────────────────────────────────────────────
# 메인 배치 적재 흐름
# ─────────────────────────────────────────────
async def run_ingest(args: argparse.Namespace) -> int:
    """디렉토리 재귀 순회 → 파일별 ingest_file → 진행 로그."""
    from core.config import load_and_validate_config
    from core.ingest.pipeline import DocumentIngestPipeline
    from core.rag.knowledge_store import KnowledgeStore

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error("입력 디렉토리가 없거나 디렉토리가 아닙니다: %s", input_dir)
        return 1

    config = load_and_validate_config()

    # 스택 결정 — 명시값 우선, "auto" 면 GPU 감지로 light/high 자동 선택.
    stack = args.stack
    if stack == "auto":
        stack = "high" if _gpu_available() else "light"
        logger.info("스택 자동 선택: %s", stack)

    # 파서 레지스트리 구성 후, 실제 등록된 확장자를 기준으로 처리 대상을 좁힌다.
    parser_registry = _build_parser_registry(stack)
    registered_exts = set(parser_registry.supported_extensions())

    # --formats 가 주어지면 그 교집합만, 없으면 등록된 전체 확장자를 처리한다.
    if args.formats:
        # "pptx,pdf" / ".pptx,.pdf" 모두 허용 — 점·소문자로 정규화.
        wanted = {
            ("." + f.strip().lstrip(".")).lower() for f in args.formats.split(",") if f.strip()
        }
        allowed_exts = wanted & registered_exts
        # 등록되지 않은 확장자를 요청하면 경고만 하고 무시한다(처리 불가).
        unsupported = wanted - registered_exts
        if unsupported:
            logger.warning("등록된 파서가 없는 확장자는 무시: %s", sorted(unsupported))
    else:
        allowed_exts = registered_exts

    if not allowed_exts:
        logger.error("처리할 확장자가 없습니다(등록 파서/--formats 확인).")
        return 1
    logger.info("처리 대상 확장자: %s", sorted(allowed_exts))

    files = iter_input_files(input_dir, allowed_exts)
    if args.limit and args.limit > 0:
        files = files[: args.limit]
    logger.info("처리 대상 파일: %d개 (input=%s)", len(files), input_dir)
    if not files:
        logger.warning("처리할 파일이 없습니다 — 종료.")
        return 0

    # 모델 프로바이더(embed) + PG 풀 구성. dry_run 이면 임베딩/적재가 없으므로
    # PG 풀을 만들지 않는다(불필요한 DB 접속 회피).
    model_provider = _build_model_provider(config, args.embed_url)
    pg_pool: Any | None = None
    if not args.dry_run:
        pg_pool = await _create_pg_pool(config, args.pg)

    knowledge_store = KnowledgeStore(pg_pool=pg_pool)
    if pg_pool is not None:
        await knowledge_store.ensure_schema()

    pipeline = DocumentIngestPipeline(
        parser_registry=parser_registry,
        model_provider=model_provider,
        knowledge_store=knowledge_store,
    )

    # 누적 카운터.
    processed = 0  # 처리 시도한 파일 수
    succeeded = 0  # 예외 없이 ingest_file 이 반환한 파일 수
    failed = 0  # 파일 단위로 예외가 난 수(격리됨)
    chunks_total = 0  # 생성된 청크 수 누적
    ingested_total = 0  # 실제 적재된 행수 누적(dry_run 이면 0)
    warnings_total = 0  # 파서 경고 누적

    try:
        for file_path in files:
            processed += 1
            # 파일별 격리 — 한 파일의 실패가 배치 전체를 중단시키지 않는다.
            # ingest_file 내부도 fail-soft(임베딩/적재 실패는 errors 로 수집)지만,
            # 파서가 던지는 치명적 예외(파일 손상 등)는 여기서 잡아 격리한다.
            try:
                summary = await pipeline.ingest_file(file_path, dry_run=args.dry_run)
            except (OSError, ValueError, RuntimeError) as e:
                failed += 1
                logger.error(
                    "파일 처리 실패(격리): %s — %s: %s",
                    file_path,
                    type(e).__name__,
                    e,
                )
                continue

            succeeded += 1
            chunks_total += summary["chunk_count"]
            ingested_total += summary["ingested"]
            warnings_total += summary["warning_count"]

            # 파서가 경고를 남겼으면(구조 일부 손실 등) 가시화한다.
            if summary["warning_count"]:
                logger.warning(
                    "경고 %d건 — %s: %s",
                    summary["warning_count"],
                    summary["title"],
                    summary["warnings"][:3],  # 앞 3건만 미리보기
                )
            # 임베딩/적재 단계 오류(ingest_file 가 errors 로 수집)도 가시화.
            if summary["errors"]:
                logger.warning(
                    "적재 오류 %d건 — %s: %s",
                    len(summary["errors"]),
                    summary["title"],
                    summary["errors"][:3],
                )

            # 주기적 진행 로그(장기 실행 관찰용).
            if processed % _PROGRESS_EVERY == 0:
                logger.info(
                    "진행: 처리=%d, 성공=%d, 실패=%d, 청크=%d, 적재=%d",
                    processed,
                    succeeded,
                    failed,
                    chunks_total,
                    ingested_total,
                )
    finally:
        # 리소스 정리 — 예외/정상 종료 모두에서 httpx 클라이언트와 PG 풀을 닫는다.
        await model_provider.close()
        if pg_pool is not None:
            await pg_pool.close()

    logger.info(
        "완료: 파일 처리=%d, 성공=%d, 실패=%d, 청크=%d, 적재=%d, 경고=%d%s",
        processed,
        succeeded,
        failed,
        chunks_total,
        ingested_total,
        warnings_total,
        " (dry-run: 적재 없음)" if args.dry_run else "",
    )
    return 0


async def run_build_index(args: argparse.Namespace) -> int:
    """대량 적재 후 벡터 검색 인덱스(ivfflat)를 만든다(prepare_kowiki 방식)."""
    from core.config import load_and_validate_config
    from core.rag.knowledge_store import KnowledgeStore

    config = load_and_validate_config()
    pool = await _create_pg_pool(config, args.pg)
    if pool is None:
        logger.error("PostgreSQL 연결 실패 — 인덱스를 빌드할 수 없습니다.")
        return 1

    store = KnowledgeStore(pg_pool=pool)
    await store.ensure_schema()
    await store.build_vector_index()
    await pool.close()
    logger.info("ivfflat 인덱스 빌드 완료")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="대량 문서 배치 인제스트기 (v7.3 단계 10) — 디렉토리를 재귀 "
        "순회해 PPTX/PDF/HWPX/이미지/.hwp 를 tb_knowledge(source='docingest')에 적재"
    )
    parser.add_argument("--input", type=str, help="문서 디렉토리(재귀 순회)")
    parser.add_argument(
        "--formats",
        type=str,
        default="",
        help="처리할 확장자 쉼표구분(예: 'pptx,pdf,hwpx,png,jpg,hwp'). "
        "미지정 시 등록된 모든 파서 확장자를 처리한다.",
    )
    parser.add_argument(
        "--embed-url",
        type=str,
        default="",
        help="임베딩 서버 base URL(미지정 시 config 의 gpu_server.embedding_url)",
    )
    parser.add_argument(
        "--pg",
        type=str,
        default="",
        help="PostgreSQL DSN(asyncpg). 미지정 시 config 의 postgresql 섹션 사용.",
    )
    parser.add_argument(
        "--stack",
        type=str,
        default="auto",
        choices=["auto", "light", "high"],
        help="파싱 스택: light(pdfplumber/tesseract, CPU) / "
        "high(docling 추가, GPU) / auto(GPU 감지로 자동 선택). 기본 auto.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="파싱/청킹까지만 수행(임베딩·DB 쓰기 생략)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="상위 N개 파일만 처리(0 = 무제한, 스모크 테스트용)",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="적재 없이 ivfflat 벡터 인덱스만 빌드",
    )
    args = parser.parse_args()

    if args.build_index:
        return asyncio.run(run_build_index(args))

    if not args.input:
        parser.error("--input 디렉토리가 필요합니다 (또는 --build-index)")
    return asyncio.run(run_ingest(args))


if __name__ == "__main__":
    sys.exit(main())
