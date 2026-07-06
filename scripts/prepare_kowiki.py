"""
위키백과(kowiki) 덤프를 지식 RAG 테이블(tb_knowledge)로 적재하는 준비 스크립트.
(사양서 Part 2.5.8, 최초 작성 2026-04-21)

── 이 파일이 하는 일 (한눈에) ──────────────────────────────
한국어 위키백과 XML 덤프 파일을 읽어서, 위키 마크업을 제거해 평문으로 바꾸고,
적당한 크기의 청크(조각)로 나눈 뒤, 임베딩 서버로 벡터를 뽑아 PostgreSQL의
tb_knowledge 테이블에 저장한다. 이렇게 쌓인 지식은 나중에 Nexus가 답변할 때
"지식 검색(RAG)"으로 근거를 찾는 데 쓰인다.

── 왜 별도 스크립트인가 (에어갭 관련 주의) ────────────────
Nexus 런타임(운영 코드)은 에어갭 원칙상 외부 네트워크를 절대 호출하지 않는다.
하지만 이 파일은 운영 코드가 아니라 **최초 1회 데이터 준비(코퍼스 구축)** 용도라
별도 단계로 허용된다 (progress.md의 "2단계 RAG 지식 베이스 계획" 근거).
실제로 이 스크립트도 외부 인터넷을 직접 호출하지 않는다 — 덤프는 미리 받아둔
로컬 파일 경로(`--dump`)로 넘겨받고, 임베딩은 LAN 안의 서버(:8002)만 쓴다.

── 주요 함수 (실행 흐름 순서) ─────────────────────────────
  - strip_wiki_markup(): 위키 문법을 정규식으로 걷어내 평문으로 정리
  - iter_pages(): 덤프 XML을 스트리밍으로 훑어 문서를 하나씩 방출
  - category_matches(): 원하는 카테고리(인문학 등)인지 필터링
  - _embed_texts(): LAN 임베딩 서버(:8002)에 배치로 벡터 요청
  - run_ingest(): 위 조각들을 엮어 파싱→청크→임베딩→적재를 수행
  - run_build_index(): 대량 적재가 끝난 뒤 벡터 검색 인덱스를 생성
  - main(): CLI 인자를 파싱해 위 두 모드 중 하나를 실행

── 외부 의존 / 다른 모듈 ──────────────────────────────────
  - core.rag.knowledge_store: KnowledgeEntry(적재 단위), KnowledgeStore(저장소),
    split_into_chunks(청크 분할) — 실제 DB 스키마·UPSERT 로직은 여기에 있다.
  - 런타임 라이브러리: bz2(표준), httpx(임베딩 HTTP), asyncpg(PG 접속).

── 실행 순서 (실제 운영 절차) ─────────────────────────────
  1. 덤프 획득 (GPU 서버에서 수동 실행 권장):
       wget -P /opt/nexus-gpu/corpora/kowiki/ \\
            https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2
  2. 이 스크립트 실행 (파싱→임베딩→적재):
       python scripts/prepare_kowiki.py \\
         --dump /opt/nexus-gpu/corpora/kowiki/kowiki-latest-pages-articles.xml.bz2 \\
         --categories "철학,문학,역사,인물" \\
         --limit 500 \\
         --embed-url http://192.168.21.112:8002 \\
         --pg "postgresql://nexus:idino@12@192.168.10.39:5440/nexus"

  3. 벡터 인덱스 빌드 (대량 적재가 모두 끝난 뒤 1회만):
       python scripts/prepare_kowiki.py --build-index \\
         --pg "postgresql://nexus:idino@12@192.168.10.39:5440/nexus"
     (인덱스는 데이터가 충분히 쌓인 뒤에 만들어야 ivfflat 품질이 좋다.)

── 설계 원칙 ──────────────────────────────────────────────
  - 위키 마크업 제거는 무거운 mwparserfromhell 없이 경량 정규식만으로 처리한다.
  - 외부 URL은 `--dump`로 받는 "로컬 파일 경로"뿐 — 네트워크 호출 없음.
  - 임베딩 서버는 LAN 내 e5-large(:8002) — 에어갭 준수.
  - 카테고리 필터로 인문학 위주로 좁혀 적재 크기를 통제한다.

── 안전장치 ───────────────────────────────────────────────
  - `--dry-run`: 파싱/청크까지만 하고 DB 쓰기·임베딩 호출은 생략(리허설).
  - `--limit N`: 상위 N개 문서만 처리(스모크 테스트용).
  - 적재는 UPSERT라서 같은 문서를 다시 넣어도 중복이 생기지 않는다.

작성자: 이현수 / 작성일: 2026-07-05
"""

# from __future__: 타입 힌트를 문자열로 지연 평가한다. 최상단에 있어야 한다(기능성).
from __future__ import annotations

import argparse  # CLI 인자 파싱
import asyncio  # 비동기 실행(임베딩 HTTP, PG 접속이 비동기라 이벤트 루프가 필요)
import bz2  # kowiki 덤프는 .bz2로 압축돼 있어 스트리밍 해제하며 읽는다
import logging
import re  # 위키 마크업 제거 및 카테고리 추출 정규식
import sys

# ElementTree로 덤프 XML을 스트리밍 파싱. S405 경고는 "신뢰 못 할 XML" 주의인데,
# 여기선 우리가 미리 받아둔 로컬 kowiki 덤프만 파싱하므로 안전(그래서 noqa).
import xml.etree.ElementTree as ET  # noqa: S405 — kowiki 덤프는 신뢰 가능, 로컬 파일만 파싱
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# 모듈 전용 로거. 이름을 "nexus.scripts.*"로 맞춰 프로젝트 로깅 규칙과 일관되게 한다.
logger = logging.getLogger("nexus.scripts.prepare_kowiki")

# 이 파일을 `python scripts/prepare_kowiki.py`처럼 단독 실행하면 core 패키지를
# import할 수 없다. 그래서 프로젝트 루트(=이 파일의 부모의 부모)를 sys.path에 넣어
# core.rag.* 를 찾을 수 있게 한다.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# E402(모듈 최상단이 아닌 import) 경고를 끄는 이유: 위의 sys.path 조작이 먼저
# 실행돼야만 이 import가 성립하므로, 일부러 여기에 둔다.
from core.rag.knowledge_store import (  # noqa: E402
    KnowledgeEntry,  # DB에 넣을 지식 한 조각(청크)을 표현하는 데이터 모델
    KnowledgeStore,  # tb_knowledge에 대한 저장소(스키마 보장/추가/인덱스 빌드)
    split_into_chunks,  # 긴 본문을 겹침(overlap) 포함해 청크로 나누는 유틸
)

# 본문이 아닌 네임스페이스(템플릿/토론/파일/사용자 문서 등)는 제외 대상이다.
# 문서 제목이 이 접두어들로 시작하면 건너뛴다.
_SKIP_PREFIXES = (
    "위키백과:", "특수기능:", "틀:", "분류:", "파일:",
    "사용자:", "사용자토론:", "문서토론:", "토론:", "미디어위키:",
)

# ── 위키 마크업 정리용 정규식 (무거운 mwparserfromhell 없이 경량 처리) ──
# 각 정규식은 위키 문법의 한 종류를 잡아내 제거하거나 표시 텍스트만 남긴다.
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")  # {{틀|인수}} 제거 (한 겹씩)
# <ref>...</ref> 각주 블록과 <ref/> 자가닫힘 태그를 통째로 제거
_RE_REF = re.compile(r"<ref[^>]*?>.*?</ref>|<ref[^>]*/>", re.DOTALL)
_RE_HTML = re.compile(r"<[^>]+>")  # 남은 일반 HTML 태그 제거
_RE_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")  # [[링크|표시]] → 표시
_RE_EXTLINK = re.compile(r"\[https?://\S+\s+([^\]]+)\]")  # [http://... 제목] → 제목
_RE_EXTLINK_PLAIN = re.compile(r"\[https?://\S+\]")  # 표시 텍스트 없는 외부링크 제거
_RE_BOLD_ITALIC = re.compile(r"'''''|'''|''")  # 굵게/기울임 마크업('' ''' ''''') 제거
# == 제목 == 형태의 문단 헤딩. 그룹1=등호개수(깊이), 그룹2=제목 텍스트.
_RE_HEADING = re.compile(r"^(=+)\s*(.+?)\s*\1\s*$", re.MULTILINE)
# 파일/이미지 삽입 링크 제거 (한글·영문 접두어 모두, 대소문자 무시)
_RE_FILE_LINK = re.compile(r"\[\[(?:파일|File|이미지|Image):[^\]]*\]\]", re.IGNORECASE)
# 분류(카테고리) 링크 제거 — 카테고리 값은 iter_pages에서 따로 뽑아 쓴다
_RE_CAT_LINK = re.compile(r"\[\[(?:분류|Category):[^\]]+\]\]", re.IGNORECASE)


def strip_wiki_markup(text: str) -> str:
    """위키 마크업을 제거해 평문에 가깝게 정리한다(경량 방식).

    무거운 파서(mwparserfromhell) 없이 위에서 정의한 정규식만 순서대로 적용한다.
    핵심 흐름:
      - 각주/파일/분류/템플릿/HTML 등 "본문이 아닌 것"을 지운다.
      - 내부·외부 링크는 사람이 읽는 표시 텍스트만 남긴다.
      - == 제목 == 헤딩은 마크다운 '#' 헤딩으로 바꿔 구조를 살린다.
    같은 치환을 최대 5번 반복하는 이유: 템플릿이 중첩돼 한 번에 다 안 지워질 수
    있어서, 더 이상 바뀌지 않을 때까지(수렴) 다시 적용한다.

    매개변수:
      text: 위키 원문(wikitext). None/빈 문자열이면 그대로 반환.
    반환:
      마크업이 제거되고 공백이 정리된 평문 문자열.
    """
    if not text:
        return text
    # 이전 결과(prev)와 같아지면 = 더 지울 게 없으면 반복을 멈춘다(수렴 판정).
    prev = ""
    t = text
    for _ in range(5):  # 중첩 템플릿 대응: 최대 5번까지 반복 적용
        if t == prev:
            break
        prev = t
        t = _RE_REF.sub("", t)  # <ref> 각주 제거
        t = _RE_FILE_LINK.sub("", t)  # 파일/이미지 링크 제거
        t = _RE_CAT_LINK.sub("", t)  # 분류 링크 제거(값은 별도 추출)
        t = _RE_TEMPLATE.sub("", t)  # {{틀}} 한 겹 제거(반복으로 중첩 해소)
        t = _RE_HTML.sub("", t)  # 남은 HTML 태그 제거
        t = _RE_LINK.sub(r"\1", t)  # [[링크|표시]] → 표시 텍스트만
        t = _RE_EXTLINK.sub(r"\1", t)  # [url 제목] → 제목만
        t = _RE_EXTLINK_PLAIN.sub("", t)  # [url] (표시 없음) 제거
        t = _RE_BOLD_ITALIC.sub("", t)  # 굵게/기울임 기호 제거
        # 헤딩 "== 제목 ==" → 마크다운 "## 제목" (등호 개수만큼 '#' 깊이 유지)
        t = _RE_HEADING.sub(lambda m: ("\n\n" + "#" * len(m.group(1)) + " " + m.group(2) + "\n"), t)
    # 마지막으로 과도한 공백을 정리한다.
    t = re.sub(r"\n{3,}", "\n\n", t)  # 빈 줄 3개 이상 → 2개로 축소
    t = re.sub(r"[ \t]{2,}", " ", t)  # 연속 공백/탭 → 한 칸으로
    return t.strip()


def _local(elem: ET.Element) -> str:
    """XML 태그에서 네임스페이스를 뗀 로컬 이름만 뽑는다.

    ElementTree는 태그를 `{네임스페이스}이름` 형태로 준다. '}' 뒤쪽만 취해서
    "title", "text" 같은 순수 태그명을 얻는다.
    """
    return elem.tag.split("}", 1)[-1]


def _find_child(elem: ET.Element, local_name: str) -> ET.Element | None:
    """네임스페이스에 상관없이 이름이 일치하는 첫 자식 엘리먼트를 찾는다.

    위키 덤프 스키마가 export-0.10 → 0.11 등으로 바뀌면 네임스페이스 URL도
    달라진다. 로컬 이름만 비교하므로 어느 버전이 와도 동일하게 동작한다.
    없으면 None을 반환한다.
    """
    for child in elem:
        if _local(child) == local_name:
            return child
    return None


def iter_pages(xml_stream) -> Iterator[dict[str, Any]]:
    """위키 덤프 XML을 스트리밍 파싱해 문서를 하나씩 dict로 방출한다.

    왜 스트리밍인가: kowiki 덤프는 수 GB라 통째로 메모리에 올릴 수 없다.
    iterparse로 <page> 태그가 끝날 때마다 처리하고, 처리 후에는 elem.clear()로
    메모리를 즉시 비운다(누수 방지).

    필터링 규칙:
      - namespace == "0" (일반 문서)만 채택. 틀/토론/파일 등은 제외.
      - 제목이 _SKIP_PREFIXES로 시작하면 제외.
    방출 형태:
      {"title": 제목, "text": 위키원문, "categories": [분류명, ...]}
    네임스페이스에 무관하게 동작하므로 export-0.10/0.11/향후 버전 모두 OK.
    """
    # S314: 신뢰할 수 없는 XML 파싱 경고. 로컬 신뢰 덤프만 다루므로 무시(noqa).
    for _, elem in ET.iterparse(xml_stream, events=("end",)):  # noqa: S314 — 로컬 신뢰 파일
        if _local(elem) != "page":  # <page> 종료 이벤트만 관심 대상
            continue

        # 한 문서(page) 안에서 필요한 하위 엘리먼트를 찾는다.
        # 본문은 <page><revision><text> 구조라 revision을 먼저 찾고 그 안의 text를 본다.
        title_el = _find_child(elem, "title")
        ns_el = _find_child(elem, "ns")
        rev_el = _find_child(elem, "revision")
        text_el = _find_child(rev_el, "text") if rev_el is not None else None

        try:
            # 각 엘리먼트가 없을 수도 있으니 안전하게 기본값을 준다.
            title = title_el.text if title_el is not None else None
            namespace = (ns_el.text or "0") if ns_el is not None else "0"
            wikitext = text_el.text if text_el is not None else None

            # 제목·본문이 있고 일반 문서(ns=0)일 때만 채택 후보로 본다.
            if title and namespace == "0" and wikitext:
                if not any(title.startswith(p) for p in _SKIP_PREFIXES):
                    # 본문에서 [[분류:XXX]] 형태의 카테고리명만 뽑아둔다.
                    # (뒤의 category_matches 필터와 적재 태그에 쓰인다.)
                    cats = re.findall(r"\[\[(?:분류|Category):([^\]|]+)", wikitext, re.IGNORECASE)
                    yield {
                        "title": title,
                        "text": wikitext,
                        "categories": [c.strip() for c in cats],
                    }
        finally:
            # 처리 여부와 무관하게 항상 메모리를 비운다(대용량 덤프 필수 처리).
            elem.clear()


def category_matches(cats: list[str], wanted: list[str]) -> bool:
    """문서 카테고리가 원하는 카테고리(부분문자열) 중 하나라도 포함하면 True.

    예: wanted=["역사"] 이고 문서 카테고리에 "한국의 역사"가 있으면 매칭된다.
    wanted가 비어 있으면(필터 없음) 모든 문서를 통과시킨다.

    매개변수:
      cats: 문서에서 추출한 카테고리 이름 목록.
      wanted: 사용자가 --categories로 준 관심 카테고리(부분문자열) 목록.
    """
    if not wanted:  # 필터 미지정 → 전부 통과
        return True
    for c in cats:
        for w in wanted:
            if w in c:  # 부분문자열 포함이면 매칭 성공
                return True
    return False


# ─────────────────────────────────────────────
# 임베딩 + 적재
# ─────────────────────────────────────────────
async def _embed_texts(base_url: str, texts: list[str]) -> list[list[float]]:
    """LAN 임베딩 서버(:8002)에 텍스트 배치를 보내 벡터를 받아온다.

    Nexus 임베딩 서버 API 규격:
      POST /v1/embed
      요청 body: {"texts": ["...", ...]}
      응답 body: {"embeddings": [[1024차원 벡터], ...], "dimension": 1024}

    매개변수:
      base_url: 임베딩 서버 기본 주소(끝 슬래시는 자동 정리).
      texts: 벡터로 바꿀 문자열 목록(청크 배치).
    반환:
      입력 순서와 동일하게 정렬된 임베딩 벡터 목록.
    """
    import httpx  # 선택적 의존성 — 이 스크립트에서만 필요해 함수 안에서 import

    # timeout=60초: 배치 임베딩이 느릴 수 있어 넉넉히 준다.
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            base_url.rstrip("/") + "/v1/embed",  # 이중 슬래시 방지 위해 rstrip
            json={"texts": texts},
        )
        r.raise_for_status()  # 4xx/5xx면 예외를 던져 호출부에서 스킵 처리하게 함
        data = r.json()
    return data["embeddings"]


async def run_ingest(args: argparse.Namespace) -> int:
    """적재 파이프라인 본체: 덤프 파싱 → 마크업 제거 → 청크 → 임베딩 → DB 적재.

    전체 흐름:
      1) 덤프 파일 존재 확인, 카테고리 필터 준비.
      2) (dry-run이 아니면) PG 풀을 열고 스키마를 보장.
      3) bz2/평문 덤프를 스트리밍하며 문서를 하나씩 처리.
         - 카테고리 필터 통과 → 마크업 제거 → 너무 짧으면 스킵.
         - split_into_chunks로 청크 분할 → 배치 임베딩 → KnowledgeEntry로 적재.
      4) 처리·적재·청크 개수를 로그로 남기고 풀을 닫는다.
    반환: 종료 코드(0=성공, 1=덤프 없음).
    """
    dump_path = Path(args.dump)
    if not dump_path.exists():
        logger.error("덤프 파일 없음: %s", dump_path)
        return 1

    # "철학,문학" 같은 쉼표 문자열을 리스트로 쪼갠다(빈 항목 제거).
    wanted_cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    logger.info("카테고리 필터: %s", wanted_cats or "(전부)")

    # PG 연결: dry-run이면 DB를 아예 열지 않는다(리허설). pool이 None이면
    # KnowledgeStore는 실제 쓰기를 하지 않는다.
    pool = None
    if args.pg and not args.dry_run:
        import asyncpg  # 선택적 의존성 — 실제 적재 시에만 필요
        pool = await asyncpg.create_pool(args.pg, min_size=1, max_size=3)
    store = KnowledgeStore(pg_pool=pool)
    if pool is not None:
        await store.ensure_schema()  # tb_knowledge 테이블/컬럼이 없으면 만든다

    # 진행 상황 카운터: 훑은 문서 수 / 실제 적재한 문서 수 / 누적 청크 수
    processed = 0
    adopted = 0
    chunks_total = 0

    # 확장자가 .bz2면 bz2.open, 아니면 일반 open으로 스트림을 연다.
    opener = bz2.open if str(dump_path).endswith(".bz2") else open
    with opener(dump_path, "rb") as f:
        for page in iter_pages(f):  # 문서 단위로 하나씩 받아 처리
            processed += 1
            # 관심 카테고리가 아니면 건너뛴다.
            if not category_matches(page["categories"], wanted_cats):
                continue

            body = strip_wiki_markup(page["text"])  # 위키 문법 제거 → 평문
            if len(body) < 300:  # 300자 미만은 토막글로 보고 스킵(품질 확보)
                continue

            # 본문을 최대 1200자, 100자 겹침으로 청크 분할.
            # 겹침을 두는 이유: 청크 경계에서 문맥이 끊겨 검색 품질이 떨어지는 걸 완화.
            chunks = split_into_chunks(body, max_chars=1200, overlap=100)
            if not chunks:
                continue

            # ── 임베딩 단계 (dry-run이면 통째로 건너뜀) ──
            embeddings: list[list[float]] = []
            if not args.dry_run:
                try:
                    # 서버 부하를 고려해 한 번에 5개씩 나눠 요청한다.
                    for i in range(0, len(chunks), 5):
                        batch = chunks[i:i + 5]
                        emb = await _embed_texts(args.embed_url, batch)
                        embeddings.extend(emb)
                except Exception as e:
                    # 한 문서 임베딩이 실패해도 전체를 멈추지 않고 그 문서만 건너뛴다.
                    logger.warning("임베딩 실패 (%s): %s", page["title"][:40], e)
                    continue

            # ── 적재 단계: 청크마다 KnowledgeEntry를 만들어 저장 ──
            for idx, chunk in enumerate(chunks):
                entry = KnowledgeEntry(
                    source="kowiki",  # 출처 표시(나중에 필터/추적용)
                    title=page["title"],
                    content=chunk,
                    section=None,  # 위키는 섹션 정보를 따로 안 붙임
                    chunk_index=idx,  # 이 문서 내 청크 순번
                    total_chunks=len(chunks),  # 이 문서의 전체 청크 수
                    tags=tuple(page["categories"][:5]),  # 카테고리 상위 5개를 태그로
                    # dry-run이면 embeddings가 비어 있으므로 벡터는 None으로 둔다.
                    embedding=tuple(embeddings[idx]) if embeddings else None,
                    metadata={"ingested_by": "prepare_kowiki.py"},  # 적재 출처 기록
                )
                if not args.dry_run:
                    await store.add(entry)  # UPSERT — 재실행해도 중복 없음
                chunks_total += 1

            adopted += 1
            if adopted % 50 == 0:  # 50개마다 중간 진행 상황 로그
                logger.info("진행: 처리=%d, 적재=%d, 청크=%d", processed, adopted, chunks_total)

            # --limit로 상한을 걸었으면 그만큼 적재하고 종료(스모크 테스트).
            if args.limit and adopted >= args.limit:
                break

    logger.info(
        "완료: 전체 처리=%d, 적재 문서=%d, 청크=%d",
        processed, adopted, chunks_total,
    )

    if pool is not None:
        await pool.close()  # 열었던 커넥션 풀을 반드시 정리
    return 0


async def run_build_index(args: argparse.Namespace) -> int:
    """대량 적재가 끝난 뒤 벡터 검색용 ivfflat 인덱스를 생성한다.

    왜 적재와 분리했나: ivfflat은 데이터 분포를 보고 클러스터를 나누므로,
    데이터가 충분히 쌓인 뒤에 한 번 만들어야 검색 품질이 좋다. 그래서 적재
    루프와 별도 모드(--build-index)로 실행한다.
    반환: 종료 코드(0=성공).
    """
    import asyncpg
    pool = await asyncpg.create_pool(args.pg, min_size=1, max_size=2)
    store = KnowledgeStore(pg_pool=pool)
    await store.ensure_schema()  # 테이블이 있는지 먼저 확인/보장
    await store.build_vector_index()  # 실제 인덱스 생성은 저장소에 위임
    await pool.close()
    logger.info("ivfflat 인덱스 빌드 완료")
    return 0


def main() -> int:
    """CLI 진입점 — 인자를 파싱해 두 모드(적재 / 인덱스 빌드) 중 하나를 실행한다.

    흐름:
      - 로깅 기본 설정(INFO 레벨) 후 argparse로 옵션을 받는다.
      - --build-index가 있으면 인덱스 빌드 모드로 분기.
      - 아니면 --dump가 필수이며, 적재 파이프라인(run_ingest)을 실행한다.
    반환: 프로세스 종료 코드(그대로 sys.exit에 전달됨).
    """
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
                        default="http://192.168.21.112:8002",
                        help="임베딩 서버 base URL")
    parser.add_argument("--pg", type=str,
                        default="postgresql://nexus:idino%4012@192.168.10.39:5440/nexus",
                        help="PostgreSQL 연결 문자열 (asyncpg) — config/nexus_config.yaml의 postgresql 섹션과 동일 자격 사용")
    parser.add_argument("--dry-run", action="store_true",
                        help="파싱/청크까지만 수행 (DB·임베딩 호출 생략)")
    parser.add_argument("--build-index", action="store_true",
                        help="적재 없이 ivfflat 벡터 인덱스만 빌드")
    args = parser.parse_args()

    # 인덱스 빌드 모드: 적재 없이 벡터 인덱스만 만든다.
    if args.build_index:
        return asyncio.run(run_build_index(args))

    # 적재 모드에서는 덤프 경로가 반드시 필요하다.
    if not args.dump:
        parser.error("--dump 경로가 필요합니다 (또는 --build-index)")
    return asyncio.run(run_ingest(args))


if __name__ == "__main__":
    # 스크립트로 직접 실행될 때 main()의 반환 코드를 프로세스 종료 코드로 넘긴다.
    sys.exit(main())
