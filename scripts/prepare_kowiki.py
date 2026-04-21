"""
위키백과 덤프를 tb_knowledge로 적재하는 준비 스크립트 (Part 2.5.8, 2026-04-21).

이 스크립트는 **운영 Nexus 코드가 아닌 "데이터 준비" 용도**다. 에어갭 원칙상
Nexus 런타임이 외부 네트워크를 호출하지 않지만, 최초 코퍼스 준비는 별도 단계로
허용된다 (progress.md 2단계 RAG 지식 베이스 계획 근거).

실행 순서:
  1. 덤프 획득 (GPU 서버에서 수동 실행 권장):
       wget -P /opt/nexus-gpu/corpora/kowiki/ \\
            https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
  2. 이 스크립트 실행:
       python scripts/prepare_kowiki.py \\
         --dump /opt/nexus-gpu/corpora/kowiki/kowiki-latest-pages-articles.xml.bz2 \\
         --categories "철학,문학,역사,인물" \\
         --limit 500 \\
         --embed-url http://192.168.22.28:8002 \\
         --pg "postgresql://nexus:idino@12@192.168.10.39:5440/nexus"

  3. 벡터 인덱스 빌드 (대량 적재 후 1회):
       python scripts/prepare_kowiki.py --build-index \\
         --pg "postgresql://nexus:idino@12@192.168.10.39:5440/nexus"

설계:
  - 본 스크립트는 **mwparserfromhell / bz2 / asyncpg / httpx만** 사용한다
  - 외부 URL은 `--dump`로 받는 "로컬 파일 경로" — 네트워크 호출 없음
  - 임베딩 서버는 LAN 내 e5-large (:8002) — 에어갭 준수
  - 카테고리 필터로 인문학만 좁혀 적재 크기를 통제한다

안전장치:
  - `--dry-run`: 파싱/청크까지만 수행하고 DB 쓰기·임베딩 호출 생략
  - `--limit N`: 상위 N개 문서만 처리 (스모크 테스트)
  - 적재는 UPSERT이므로 재실행해도 중복 없음
"""

from __future__ import annotations

import argparse
import asyncio
import bz2
import logging
import re
import sys
import xml.etree.ElementTree as ET  # noqa: S405 — kowiki 덤프는 신뢰 가능, 로컬 파일만 파싱
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus.scripts.prepare_kowiki")

# 프로젝트 루트를 sys.path에 추가 (스크립트를 독립 실행할 때)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rag.knowledge_store import (  # noqa: E402
    KnowledgeEntry,
    KnowledgeStore,
    split_into_chunks,
)

# 기본 제외 네임스페이스 (템플릿/토론/파일 등은 본문이 아님)
_SKIP_PREFIXES = (
    "위키백과:", "특수기능:", "틀:", "분류:", "파일:",
    "사용자:", "사용자토론:", "문서토론:", "토론:", "미디어위키:",
)

# 위키 마크업 정리용 정규식 (mwparserfromhell 의존 없이 경량 처리)
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")  # {{틀|인수}} 제거 (단일 레벨)
_RE_REF = re.compile(r"<ref[^>]*?>.*?</ref>|<ref[^>]*/>", re.DOTALL)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")  # [[링크|표시]] → 표시
_RE_EXTLINK = re.compile(r"\[https?://\S+\s+([^\]]+)\]")  # [http://... 제목]
_RE_EXTLINK_PLAIN = re.compile(r"\[https?://\S+\]")
_RE_BOLD_ITALIC = re.compile(r"'''''|'''|''")
_RE_HEADING = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$", re.MULTILINE)
_RE_FILE_LINK = re.compile(r"\[\[(?:파일|File|이미지|Image):[^\]]*\]\]", re.IGNORECASE)
_RE_CAT_LINK = re.compile(r"\[\[(?:분류|Category):[^\]]+\]\]", re.IGNORECASE)


def strip_wiki_markup(text: str) -> str:
    """경량 위키 마크업 제거 — mwparserfromhell 없이 필요한 만큼만 처리."""
    if not text:
        return text
    # 반복 적용 (중첩 템플릿 대응)
    prev = ""
    t = text
    for _ in range(5):
        if t == prev:
            break
        prev = t
        t = _RE_REF.sub("", t)
        t = _RE_FILE_LINK.sub("", t)
        t = _RE_CAT_LINK.sub("", t)
        t = _RE_TEMPLATE.sub("", t)
        t = _RE_HTML.sub("", t)
        t = _RE_LINK.sub(r"\1", t)
        t = _RE_EXTLINK.sub(r"\1", t)
        t = _RE_EXTLINK_PLAIN.sub("", t)
        t = _RE_BOLD_ITALIC.sub("", t)
        # 헤딩 == 제목 == → # 제목
        t = _RE_HEADING.sub(lambda m: ("\n\n" + "#" * len(m.group(1)) + " " + m.group(2) + "\n"), t)
    # 과도한 공백 정리
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def _local(elem: ET.Element) -> str:
    """`{namespace}name` 태그에서 로컬 이름만 추출."""
    return elem.tag.split("}", 1)[-1]


def _find_child(elem: ET.Element, local_name: str) -> ET.Element | None:
    """namespace-agnostic 자식 탐색. 위키 덤프가 export-0.10/0.11 등으로
    바뀌어도 항상 동작한다."""
    for child in elem:
        if _local(child) == local_name:
            return child
    return None


def iter_pages(xml_stream) -> Iterator[dict[str, Any]]:
    """위키 덤프 XML을 스트리밍 파싱하여 (title, text, categories) 튜플 방출.

    namespace-agnostic — export-0.10, 0.11, 향후 버전 모두 동작.
    """
    for _, elem in ET.iterparse(xml_stream, events=("end",)):  # noqa: S314 — 로컬 신뢰 파일
        if _local(elem) != "page":
            continue

        title_el = _find_child(elem, "title")
        ns_el = _find_child(elem, "ns")
        rev_el = _find_child(elem, "revision")
        text_el = _find_child(rev_el, "text") if rev_el is not None else None

        try:
            title = title_el.text if title_el is not None else None
            namespace = (ns_el.text or "0") if ns_el is not None else "0"
            wikitext = text_el.text if text_el is not None else None

            if title and namespace == "0" and wikitext:
                if not any(title.startswith(p) for p in _SKIP_PREFIXES):
                    # 카테고리 추출
                    cats = re.findall(r"\[\[(?:분류|Category):([^\]|]+)", wikitext, re.IGNORECASE)
                    yield {
                        "title": title,
                        "text": wikitext,
                        "categories": [c.strip() for c in cats],
                    }
        finally:
            elem.clear()


def category_matches(cats: list[str], wanted: list[str]) -> bool:
    """문서의 카테고리 리스트 중 하나라도 wanted 부분문자열을 포함하면 True."""
    if not wanted:
        return True
    for c in cats:
        for w in wanted:
            if w in c:
                return True
    return False


# ─────────────────────────────────────────────
# 임베딩 + 적재
# ─────────────────────────────────────────────
async def _embed_texts(base_url: str, texts: list[str]) -> list[list[float]]:
    """임베딩 서버(:8002)에 배치 요청.

    Nexus 임베딩 서버 스펙:
      POST /v1/embed
      body: {"texts": ["...", ...]}
      response: {"embeddings": [[1024-vec], ...], "dimension": 1024}
    """
    import httpx  # 선택적 의존성 — 스크립트에서만 사용

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            base_url.rstrip("/") + "/v1/embed",
            json={"texts": texts},
        )
        r.raise_for_status()
        data = r.json()
    return data["embeddings"]


async def run_ingest(args: argparse.Namespace) -> int:
    """덤프 파싱 → 청크 → 임베딩 → tb_knowledge 적재."""
    dump_path = Path(args.dump)
    if not dump_path.exists():
        logger.error("덤프 파일 없음: %s", dump_path)
        return 1

    wanted_cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    logger.info("카테고리 필터: %s", wanted_cats or "(전부)")

    # PG 연결
    pool = None
    if args.pg and not args.dry_run:
        import asyncpg
        pool = await asyncpg.create_pool(args.pg, min_size=1, max_size=3)
    store = KnowledgeStore(pg_pool=pool)
    if pool is not None:
        await store.ensure_schema()

    processed = 0
    adopted = 0
    chunks_total = 0

    # bz2 스트림으로 XML 파싱 → 문서 단위 방출
    opener = bz2.open if str(dump_path).endswith(".bz2") else open
    with opener(dump_path, "rb") as f:
        for page in iter_pages(f):
            processed += 1
            if not category_matches(page["categories"], wanted_cats):
                continue

            body = strip_wiki_markup(page["text"])
            if len(body) < 300:  # 너무 짧은 문서는 스킵
                continue

            chunks = split_into_chunks(body, max_chars=1200, overlap=100)
            if not chunks:
                continue

            # 임베딩 (배치)
            embeddings: list[list[float]] = []
            if not args.dry_run:
                try:
                    # 5개씩 배치
                    for i in range(0, len(chunks), 5):
                        batch = chunks[i:i + 5]
                        emb = await _embed_texts(args.embed_url, batch)
                        embeddings.extend(emb)
                except Exception as e:
                    logger.warning("임베딩 실패 (%s): %s", page["title"][:40], e)
                    continue

            # 적재
            for idx, chunk in enumerate(chunks):
                entry = KnowledgeEntry(
                    source="kowiki",
                    title=page["title"],
                    content=chunk,
                    section=None,
                    chunk_index=idx,
                    total_chunks=len(chunks),
                    tags=tuple(page["categories"][:5]),
                    embedding=tuple(embeddings[idx]) if embeddings else None,
                    metadata={"ingested_by": "prepare_kowiki.py"},
                )
                if not args.dry_run:
                    await store.add(entry)
                chunks_total += 1

            adopted += 1
            if adopted % 50 == 0:
                logger.info("진행: 처리=%d, 적재=%d, 청크=%d", processed, adopted, chunks_total)

            if args.limit and adopted >= args.limit:
                break

    logger.info(
        "완료: 전체 처리=%d, 적재 문서=%d, 청크=%d",
        processed, adopted, chunks_total,
    )

    if pool is not None:
        await pool.close()
    return 0


async def run_build_index(args: argparse.Namespace) -> int:
    """대량 적재 후 벡터 검색 인덱스(ivfflat)를 만든다."""
    import asyncpg
    pool = await asyncpg.create_pool(args.pg, min_size=1, max_size=2)
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
    parser = argparse.ArgumentParser(description="kowiki 덤프 적재기 (Part 2.5.8)")
    parser.add_argument("--dump", type=str, help="kowiki-*.xml.bz2 로컬 경로")
    parser.add_argument("--categories", type=str, default="철학,문학,역사,인물",
                        help="쉼표구분 카테고리 부분문자열 필터")
    parser.add_argument("--limit", type=int, default=0,
                        help="상위 N개 문서만 처리 (0 = 무제한)")
    parser.add_argument("--embed-url", type=str,
                        default="http://192.168.22.28:8002",
                        help="임베딩 서버 base URL")
    parser.add_argument("--pg", type=str,
                        default="postgresql://nexus:idino%40%4012@192.168.10.39:5440/nexus",
                        help="PostgreSQL 연결 문자열 (asyncpg)")
    parser.add_argument("--dry-run", action="store_true",
                        help="파싱/청크까지만 수행 (DB·임베딩 호출 생략)")
    parser.add_argument("--build-index", action="store_true",
                        help="적재 없이 ivfflat 벡터 인덱스만 빌드")
    args = parser.parse_args()

    if args.build_index:
        return asyncio.run(run_build_index(args))

    if not args.dump:
        parser.error("--dump 경로가 필요합니다 (또는 --build-index)")
    return asyncio.run(run_ingest(args))


if __name__ == "__main__":
    sys.exit(main())
