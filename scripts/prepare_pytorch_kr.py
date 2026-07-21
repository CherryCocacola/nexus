# PyTorch 한국어 자료(공식 한국어 튜토리얼 + 읽을거리&정보공유)를 코딩 지식 RAG로 적재하는 스크립트.
"""
PyTorch 한국어 자료를 tb_knowledge로 적재한다(AI/딥러닝 지식 보강).

── 이 파일이 하는 일 (한눈에) ─────────────────────────────
두 개의 서로 다른 소스를 하나의 스크립트에서 처리한다. 성격도 라이선스도 다르므로
tb_knowledge.source 를 반드시 분리한다(상용 전환 시 선택적 제거를 위해).

  1) --mode tutorials → source='pytorch_kr'
     PyTorchKorea/tutorials-kr 저장소(로컬 git clone)의 .rst / .py 를 파싱한다.
     sphinx-gallery 형식이라 .py 안에 산문(주석)과 코드가 섞여 있다.
     라이선스 BSD-3-Clause → 상용 배포 가능. PyTorch v2.8 기준이라 API가 최신이다.

  2) --mode forum → source='pytorch_forum_kr'
     discuss.pytorch.kr 의 '읽을거리&정보공유'(category id=14, slug='news') 토픽.
     Discourse 공개 JSON API를 쓰므로 HTML 파싱이 필요 없다(OKKY의 RSC 파싱과 대조).

기존 자산을 재사용한다: coding_corpus.chunk_code_aware(코드 인지 청킹),
html_to_markdown(Discourse cooked HTML 변환), KnowledgeStore(UPSERT 적재).
prepare_okky.py / prepare_stackoverflow.py 와 대칭 구조다.

── 왜 한국어 소스인가 ─────────────────────────────────────
기존 코딩 RAG(so_ko)는 영어 SO를 한국어 글로스로 우회한 것이라 교차언어 recall이
근본 한계였다. 이 두 소스는 **원본이 한국어**라 글로스·재임베딩이 아예 불필요하다.
또 so_ko의 AI 관련 비중은 pytorch 334청크(0.13%)·tensorflow 1015청크로 TF 편중이며
torch 1.x 시대 지식이라 노후했다. 이 적재는 그 갭을 정조준한다.

── 법적·스크래핑 준수 (중요) ─────────────────────────────
사용자 결정: **내부 전용** 사용. 상용 전환 시 provenance(source)로 선택 제거한다.
  - pytorch_kr      : BSD-3-Clause. 상용 배포 가능(저작자 표시 유지).
  - pytorch_forum_kr: 사용자 저작물(UGC), 재사용 라이선스 없음 → 상용 전 반드시
                      `DELETE FROM tb_knowledge WHERE source='pytorch_forum_kr'`.
robots.txt(2026-07-21 확인) 준수: /c/, /t/ 는 허용(차단은 *.rss·/search·/my·/g 등).
Crawl-delay 명시는 없으나 예의로 요청 간 1.2초 간격을 둔다(OKKY와 동일).

── 검증 범위 ──────────────────────────────────────────────
파싱·필터·엔트리 생성은 합성 입력 단위테스트로 검증한다. 임베딩·DB 쓰기(run_ingest)는
실 인프라가 필요하므로 --dry-run 으로 분리한다(prepare_okky 대칭).

작성자: Nexus 팀 / 작성일: 2026-07-21
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (coding_corpus)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 루트 (core)

from coding_corpus import chunk_code_aware, html_to_markdown  # noqa: E402

from core.rag.knowledge_store import KnowledgeEntry, KnowledgeStore  # noqa: E402

logger = logging.getLogger("nexus.scripts.prepare_pytorch_kr")

_FORUM_BASE = "https://discuss.pytorch.kr"
_TUTORIAL_BASE = "https://tutorials.pytorch.kr"
_NEWS_CATEGORY_ID = 14  # '읽을거리&정보공유' (slug='news', 2026-07-21 기준 토픽 3,120)
_NEWS_CATEGORY_SLUG = "news"

_TUTORIAL_LICENSE = "BSD-3-Clause (PyTorchKorea/tutorials-kr, 원본 pytorch/tutorials와 동일)"
_FORUM_LICENSE = "discuss.pytorch.kr UGC (재사용 라이선스 없음; 내부 전용, source로 제거 가능)"

_RATE_DELAY = 1.2  # 예의: 요청 간 최소 간격(초). robots에 Crawl-delay 없음.
_UA = "Mozilla/5.0 (compatible; NexusResearch/1.0; internal RAG corpus prep)"

# 한글 음절 영역. 번역 미완(영문 잔존) 문서를 걸러낼 때 쓴다.
_HANGUL_RE = re.compile(r"[가-힣]")
# 알파벳/한글만 세어 비율을 낸다(코드·기호는 분모에서 제외해야 코드 많은 문서가 억울하지 않다).
_LETTER_RE = re.compile(r"[가-힣A-Za-z]")

# sphinx-gallery .py 의 블록 구분선(#### 또는 ---- 이 길게 이어진 주석 줄).
_GALLERY_SEP_RE = re.compile(r"^#\s*[#\-=*]{10,}\s*$")
# reStructuredText 제목 밑줄(=== --- ~~~ 등이 3자 이상 반복).
_RST_UNDERLINE_RE = re.compile(r"^[=\-~^\"'`*+#]{3,}\s*$")


def hangul_ratio(text: str) -> float:
    """글자(한글+영문) 중 한글이 차지하는 비율을 돌려준다(0.0~1.0).

    번역 미완 튜토리얼을 걸러내기 위한 지표다. tutorials-kr는 오픈 이슈가 많아
    영문 원문이 그대로 남은 문서가 섞여 있는데, 그런 문서를 그대로 적재하면
    한국어 질의에 대해 so_ko와 같은 교차언어 문제를 다시 만든다.
    코드·기호는 분모에서 빼므로 "코드가 많은 한국어 문서"가 불리해지지 않는다.
    """
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    hangul = _HANGUL_RE.findall(text)
    return len(hangul) / len(letters)


# ── 1) 튜토리얼(tutorials-kr) 파싱 ────────────────────────────────────────


def parse_rst(text: str) -> tuple[str, str]:
    """.rst 문서에서 (제목, 본문 마크다운)을 뽑는다.

    완전한 rst 파서가 아니라 RAG 적재에 필요한 만큼만 처리한다.
      - 제목: 밑줄(===, ---)이 붙은 첫 줄
      - `.. code-block:: python` 지시자 → 마크다운 코드 펜스(```)로 변환
      - 그 밖의 `.. xxx::` 지시자 줄은 버리되 내용은 남긴다(정보 손실 최소화)
    """
    lines = text.splitlines()
    title = ""
    out: list[str] = []
    i = 0
    in_code = False       # 코드 펜스를 열어둔 상태인지
    code_indent = 0       # 코드 블록의 기준 들여쓰기(이보다 얕아지면 블록 종료)

    while i < len(lines):
        line = lines[i]

        # 코드 블록 안: 들여쓰기가 유지되는 동안 계속 코드로 취급한다.
        if in_code:
            if line.strip() == "":
                out.append("")
                i += 1
                continue
            indent = len(line) - len(line.lstrip())
            if indent >= code_indent:
                out.append(line[code_indent:])
                i += 1
                continue
            out.append("```")
            in_code = False
            # 여기서 continue 하지 않고 아래 일반 처리로 흘려보낸다.

        # 코드 블록 시작 지시자.
        m = re.match(r"^(\s*)\.\.\s+(?:code-block|sourcecode)::\s*(\w+)?\s*$", line)
        if m:
            lang = m.group(2) or ""
            out.append(f"```{lang}")
            in_code = True
            # 지시자 다음의 옵션(:linenos: 등)과 빈 줄을 건너뛰고 본문 들여쓰기를 잡는다.
            j = i + 1
            while j < len(lines) and (lines[j].strip() == "" or re.match(r"^\s+:\w+:", lines[j])):
                j += 1
            code_indent = (len(lines[j]) - len(lines[j].lstrip())) if j < len(lines) else 0
            i = j
            continue

        # 제목: 다음 줄이 밑줄이면 제목으로 인정한다.
        if (
            line.strip()
            and i + 1 < len(lines)
            and _RST_UNDERLINE_RE.match(lines[i + 1])
            and len(lines[i + 1].strip()) >= len(line.strip()) // 2
        ):
            if not title:
                title = line.strip()
            out.append(f"## {line.strip()}")
            i += 2
            continue

        # 남은 지시자/주석 줄은 버린다(내용 줄은 살아남는다).
        if re.match(r"^\s*\.\.\s", line):
            i += 1
            continue

        out.append(line)
        i += 1

    if in_code:
        out.append("```")

    return title, "\n".join(out).strip()


def parse_gallery_py(text: str) -> tuple[str, str]:
    """sphinx-gallery 형식 .py 에서 (제목, 본문 마크다운)을 뽑는다.

    구조: 맨 위 모듈 docstring(rst)이 서론이고, 그 뒤로
      ######## 구분선 + `#` 주석 블록(산문)  /  일반 파이썬 코드
    가 번갈아 나온다. 산문은 그대로, 코드는 ```python 펜스로 감싼다.
    """
    title = ""
    body: list[str] = []

    # 1) 모듈 docstring(서론) 분리.
    m = re.match(r'^\s*(?:#[^\n]*\n)*\s*(?P<q>"""|\'\'\')(?P<doc>.*?)(?P=q)', text, re.DOTALL)
    rest = text
    if m:
        title, doc_md = parse_rst(m.group("doc"))
        body.append(doc_md)
        rest = text[m.end():]

    # 2) 나머지를 산문(주석)/코드로 번갈아 모은다.
    buf_code: list[str] = []
    buf_prose: list[str] = []

    def flush_code() -> None:
        # 쌓인 코드 줄을 펜스로 감싸 본문에 넣는다(빈 줄만 있으면 버린다).
        code = "\n".join(buf_code).strip("\n")
        buf_code.clear()
        if code.strip():
            body.append(f"```python\n{code}\n```")

    def flush_prose() -> None:
        # 쌓인 주석 줄을 산문으로 넣는다(rst 문법이 섞일 수 있어 parse_rst를 태운다).
        prose = "\n".join(buf_prose).strip("\n")
        buf_prose.clear()
        if prose.strip():
            _, prose_md = parse_rst(prose)
            body.append(prose_md)

    for line in rest.splitlines():
        if _GALLERY_SEP_RE.match(line):
            # 구분선 = 산문 블록의 경계. 코드를 먼저 닫는다.
            flush_code()
            flush_prose()
            continue
        if line.startswith("#"):
            flush_code()
            buf_prose.append(re.sub(r"^#\s?", "", line))
            continue
        if buf_prose:
            flush_prose()
        buf_code.append(line)

    flush_code()
    flush_prose()

    return title, "\n\n".join(x for x in body if x.strip()).strip()


def iter_tutorial_docs(repo_root: Path) -> list[dict[str, Any]]:
    """tutorials-kr 저장소에서 튜토리얼 문서를 훑어 (경로/제목/본문) 목록을 만든다.

    *_source/ 디렉토리(beginner_source, intermediate_source, advanced_source,
    recipes_source, unstable_source 등)만 대상으로 한다. 저장소 루트의 빌드 설정·
    스크립트는 지식이 아니므로 제외한다.
    """
    docs: list[dict[str, Any]] = []
    for src_dir in sorted(repo_root.glob("*_source")):
        for path in sorted(src_dir.rglob("*")):
            # .txt 는 CMakeLists.txt 등 빌드 파일이라 지식이 아니다(제외).
            if path.suffix not in (".rst", ".py"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                logger.warning("읽기 실패 %s: %s", path, e)
                continue
            rel = path.relative_to(repo_root).as_posix()
            if path.suffix == ".py":
                title, body = parse_gallery_py(text)
            else:
                title, body = parse_rst(text)
            if not body.strip():
                continue
            docs.append({
                "path": rel,
                "title": title or path.stem.replace("_", " "),
                "body": body,
            })
    return docs


def build_tutorial_entries(
    docs: list[dict[str, Any]],
    *,
    source: str = "pytorch_kr",
    min_doc_chars: int = 200,
    min_hangul_ratio: float = 0.15,
    max_chunk_chars: int = 1500,
) -> list[KnowledgeEntry]:
    """튜토리얼 문서를 청킹해 KnowledgeEntry로 만든다.

    min_hangul_ratio: 번역 미완(영문 잔존) 문서를 제외하는 게이트. tutorials-kr는
      비정기 번역이라 영문 원문이 남은 문서가 섞여 있고, 그대로 넣으면 한국어 질의에
      대한 교차언어 문제가 재발한다. 코드 비중이 큰 문서를 고려해 임계는 낮게(0.15) 둔다.
    PK 유일성: section=저장소 상대경로(문서마다 유일).
    """
    entries: list[KnowledgeEntry] = []
    skipped_en = 0
    for d in docs:
        body = d["body"]
        if len(body) < min_doc_chars:
            continue
        if hangul_ratio(body) < min_hangul_ratio:
            skipped_en += 1
            continue
        chunks = chunk_code_aware(body, max_chars=max_chunk_chars)
        # 저장소 상대경로 → 공개 문서 URL(확장자를 .html로 교체, *_source/ 접두 제거).
        doc_url = f"{_TUTORIAL_BASE}/{re.sub(r'_source/', '/', d['path'])}"
        doc_url = re.sub(r"\.(rst|py|txt)$", ".html", doc_url).replace("//", "/").replace(":/", "://")
        meta = {
            "path": d["path"],
            "url": doc_url,
            "license": _TUTORIAL_LICENSE,
            "repo": "PyTorchKorea/tutorials-kr",
        }
        for i, ch in enumerate(chunks):
            entries.append(KnowledgeEntry(
                source=source, title=d["title"], content=ch, section=d["path"],
                chunk_index=i, total_chunks=len(chunks), tags=("pytorch", "tutorial"),
                metadata=meta,
            ))
    if skipped_en:
        logger.info("번역 미완(한글비율<%.2f) 문서 %d건 제외", min_hangul_ratio, skipped_en)
    return entries


# ── 2) 포럼(discuss.pytorch.kr '읽을거리&정보공유') 수집 ──────────────────


def _get_json(url: str) -> dict[str, Any] | None:
    """Discourse 공개 JSON을 rate-limit을 지키며 받아온다(실패는 None)."""
    import httpx

    time.sleep(_RATE_DELAY)  # 예의: 요청 간 간격
    try:
        r = httpx.get(url, headers={"User-Agent": _UA, "Accept": "application/json"},
                      timeout=20, follow_redirects=True)
    except httpx.HTTPError as e:
        logger.warning("요청 실패 %s: %s", url, e)
        return None
    if r.status_code != 200:
        logger.warning("HTTP %d %s", r.status_code, url)
        return None
    try:
        return r.json()
    except json.JSONDecodeError:
        logger.warning("JSON 파싱 실패 %s", url)
        return None


def iter_category_topics(max_pages: int, start_page: int = 0) -> list[dict[str, Any]]:
    """'읽을거리&정보공유' 카테고리의 토픽 목록을 열거한다(robots 허용 경로 /c/).

    Discourse 카테고리 JSON은 페이지당 30개 안팎을 준다. 빈 페이지가 나오면 멈춘다.
    """
    topics: list[dict[str, Any]] = []
    seen: set[int] = set()
    for page in range(start_page, start_page + max_pages):
        url = f"{_FORUM_BASE}/c/{_NEWS_CATEGORY_SLUG}/{_NEWS_CATEGORY_ID}.json?page={page}"
        data = _get_json(url)
        if not data:
            break
        page_topics = (data.get("topic_list") or {}).get("topics") or []
        new = [t for t in page_topics if t.get("id") not in seen]
        if not new:
            break
        for t in new:
            seen.add(t["id"])
        topics.extend(new)
        logger.info("열거: page %d → 누적 %d개 토픽", page, len(topics))
    return topics


def fetch_topic(topic_id: int) -> dict[str, Any] | None:
    """토픽 1건의 본문+답글을 받아온다(robots 허용 경로 /t/)."""
    return _get_json(f"{_FORUM_BASE}/t/{topic_id}.json")


def build_topic_document(topic: dict[str, Any]) -> tuple[str, list[int]]:
    """토픽 JSON에서 (결합 마크다운 문서, 사용한 post id 목록)을 만든다.

    Discourse는 렌더된 HTML을 `cooked` 필드로 준다. html_to_markdown으로 변환해
    코드 블록(```)을 보존한 채 본문을 합친다. 첫 글(원 게시물)과 답글을 이어붙인다.
    """
    posts = (topic.get("post_stream") or {}).get("posts") or []
    parts: list[str] = []
    used: list[int] = []
    for p in posts:
        cooked = p.get("cooked") or ""
        if not cooked:
            continue
        md = html_to_markdown(cooked).strip()
        if not md:
            continue
        parts.append(md)
        used.append(p.get("id"))
    return "\n\n".join(parts), used


def normalize_tags(raw: Any) -> tuple[str, ...]:
    """Discourse 태그를 문자열 튜플로 정규화한다.

    같은 'tags' 필드라도 엔드포인트에 따라 모양이 다르다.
      - 카테고리 목록(/c/...json): ["paper", "llm"]        (문자열)
      - 토픽 상세(/t/{id}.json)  : [{"id":432,"name":"paper",...}]  (dict)
    dict를 그대로 넘기면 tb_knowledge.tags(text[]) 바인딩에서 DataError가 난다.
    """
    out: list[str] = []
    for t in raw or ():
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, dict):
            name = t.get("name") or t.get("slug")
            if name:
                out.append(str(name))
    return tuple(out)


def build_forum_entries(
    topics: list[dict[str, Any]],
    *,
    source: str = "pytorch_forum_kr",
    min_doc_chars: int = 300,
    max_chunk_chars: int = 1500,
) -> list[KnowledgeEntry]:
    """토픽을 청킹해 KnowledgeEntry로 만든다.

    min_doc_chars 를 300으로 둔 이유: 이 카테고리는 토픽당 게시물이 1.21개
    (3,782/3,120, 2026-07-21 실측)로 대부분 답글 없는 단발 공유글이다. 그중
    링크만 달랑 있는 글은 RAG에 노이즈이므로 본문 길이로 걸러낸다.
    PK 유일성: section=f"t{topic_id}".
    """
    entries: list[KnowledgeEntry] = []
    for t in topics:
        tid = t.get("id")
        if tid is None:
            continue
        doc, used = build_topic_document(t)
        if not used or len(doc) < min_doc_chars:
            continue
        title = t.get("title") or (t.get("unicode_title") or "")
        slug = t.get("slug") or ""
        chunks = chunk_code_aware(doc, max_chars=max_chunk_chars)
        meta = {
            "topic_id": tid,
            "url": f"{_FORUM_BASE}/t/{slug}/{tid}" if slug else f"{_FORUM_BASE}/t/{tid}",
            "license": _FORUM_LICENSE,
            "post_ids": used,
            "category": "읽을거리&정보공유",
        }
        for i, ch in enumerate(chunks):
            entries.append(KnowledgeEntry(
                source=source, title=title, content=ch, section=f"t{tid}",
                chunk_index=i, total_chunks=len(chunks),
                tags=normalize_tags(t.get("tags")) or ("pytorch", "news"),
                metadata=meta,
            ))
    return entries


# ── 3) 적재 파이프라인 (prepare_okky 대칭) ────────────────────────────────


async def _embed_texts(base_url: str, texts: list[str]) -> list[list[float]]:
    """LAN 임베딩 서버(:8002) /v1/embed 로 벡터를 받는다(prepare_kowiki 계약)."""
    import httpx

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(base_url.rstrip("/") + "/v1/embed", json={"texts": texts})
        r.raise_for_status()
        return r.json()["embeddings"]


async def _store_one(store, entry, attempts: int = 3) -> bool:
    """엔트리 1건을 UPSERT한다. 커넥션이 끊기면 잠시 쉬고 재시도한다.

    장시간(1시간+) 적재에서는 유휴 커넥션이 서버/네트워크에 의해 끊겨
    ConnectionDoesNotExistError가 난다(실제로 948토픽 지점에서 겪음). 풀에서 새
    커넥션을 다시 받으면 대개 복구되므로 몇 번 재시도한다.
    """
    import asyncpg

    for n in range(attempts):
        try:
            await store.add(entry)
            return True
        except (asyncpg.PostgresConnectionError, ConnectionError, OSError) as e:
            if n == attempts - 1:
                logger.warning("적재 실패(재시도 소진): %s", e)
                return False
            await asyncio.sleep(2 * (n + 1))  # 2초 → 4초 백오프
    return False


async def _embed_and_store(store, embed_url: str, entries, batch_size: int) -> int:
    """엔트리 목록을 배치 임베딩해 UPSERT하고 적재 수를 돌려준다."""
    stored = 0
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        try:
            embs = await _embed_texts(embed_url, [e.content for e in batch])
        except Exception as e:
            logger.warning("임베딩 실패 배치: %s", e)
            continue
        for entry, emb in zip(batch, embs, strict=True):
            if await _store_one(store, replace(entry, embedding=tuple(emb))):
                stored += 1
    return stored


async def _existing_keys(pool, source: str, meta_key: str) -> set[str]:
    """이미 적재된 문서 키를 모은다(이어받기용 — 재스크랩·재임베딩을 피한다).

    meta_key는 파라미터 바인딩($2)으로 넘긴다. 내부 상수라도 SQL에 문자열을
    직접 끼워 넣지 않는 것이 원칙이다.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT metadata->>$2 AS k FROM tb_knowledge WHERE source=$1",
            source, meta_key,
        )
    return {r["k"] for r in rows if r["k"]}


async def run_ingest(args: argparse.Namespace) -> int:
    """모드에 따라 수집 → 결합 → 임베딩 → tb_knowledge UPSERT를 수행한다.

    --dry-run 이면 수집·결합까지만(임베딩·DB 없음). 재실행은 UPSERT라 멱등이며,
    이미 적재된 문서는 건너뛰므로 중단 후 이어받기가 가능하다.
    """
    if args.mode == "tutorials":
        repo = Path(args.repo).expanduser().resolve()
        if not repo.is_dir():
            logger.error("저장소 경로가 없습니다: %s", repo)
            return 1
        docs = iter_tutorial_docs(repo)
        logger.info("튜토리얼 문서 %d건 파싱", len(docs))
        if args.dry_run:
            entries = build_tutorial_entries(
                docs, source=args.source, min_hangul_ratio=args.min_hangul,
            )
            print(f"[dry-run] mode=tutorials docs={len(docs)} entries={len(entries)} "
                  f"source={args.source}")
            return 0
    else:
        topics_meta = iter_category_topics(args.pages, args.start_page)
        logger.info("열거 %d개 토픽(page %d~, delay %.1fs)",
                    len(topics_meta), args.start_page, _RATE_DELAY)
        if args.dry_run:
            ids = [t["id"] for t in topics_meta][: (args.limit or 10)]
            full = [t for tid in ids if (t := fetch_topic(tid))]
            entries = build_forum_entries(full, source=args.source)
            print(f"[dry-run] mode=forum topics={len(full)} entries={len(entries)} "
                  f"source={args.source}")
            return 0

    import asyncpg

    pool = await asyncpg.create_pool(args.pg, min_size=1, max_size=3)
    store = KnowledgeStore(pg_pool=pool)
    await store.ensure_schema()
    stored = 0
    try:
        if args.mode == "tutorials":
            done = await _existing_keys(pool, args.source, "path")
            todo = [d for d in docs if d["path"] not in done]
            if args.limit:
                todo = todo[: args.limit]
            logger.info("신규 문서 %d건 (기존 %d건 스킵)", len(todo), len(done))
            batch = 50  # 문서 50건마다 임베딩→저장(증분)
            for i in range(0, len(todo), batch):
                entries = build_tutorial_entries(
                    todo[i : i + batch], source=args.source, min_hangul_ratio=args.min_hangul,
                )
                stored += await _embed_and_store(store, args.embed_url, entries, args.batch_size)
                logger.info("진행: 문서 %d/%d, 적재 %d청크", min(i + batch, len(todo)),
                            len(todo), stored)
        else:
            done = await _existing_keys(pool, args.source, "topic_id")
            ids = [t["id"] for t in topics_meta if str(t["id"]) not in done]
            if args.limit:
                ids = ids[: args.limit]
            logger.info("신규 토픽 %d건 (기존 %d건 스킵)", len(ids), len(done))
            batch = 20  # 토픽 20건마다 스크랩→임베딩→저장(증분)
            for i in range(0, len(ids), batch):
                full = [t for tid in ids[i : i + batch] if (t := fetch_topic(tid))]
                entries = build_forum_entries(full, source=args.source)
                stored += await _embed_and_store(store, args.embed_url, entries, args.batch_size)
                logger.info("진행: 토픽 %d/%d, 적재 %d청크", min(i + batch, len(ids)),
                            len(ids), stored)
    finally:
        await pool.close()
    logger.info("완료: %d청크 적재(source=%s)", stored, args.source)
    print(f"stored={stored} source={args.source}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PyTorch 한국어 자료(튜토리얼/읽을거리&정보공유) → tb_knowledge",
    )
    parser.add_argument("--mode", choices=("tutorials", "forum"), required=True,
                        help="tutorials=tutorials-kr 저장소, forum=읽을거리&정보공유")
    parser.add_argument("--repo", help="[tutorials] tutorials-kr git clone 경로")
    parser.add_argument("--min-hangul", type=float, default=0.15,
                        help="[tutorials] 번역 미완 제외 임계(한글 비율)")
    parser.add_argument("--pages", type=int, default=120,
                        help="[forum] 열거할 카테고리 페이지 수(30개/페이지)")
    parser.add_argument("--start-page", type=int, default=0, help="[forum] 열거 시작 페이지")
    parser.add_argument("--limit", type=int, default=None, help="처리 상한(문서/토픽 수)")
    parser.add_argument("--source", help="tb_knowledge.source (기본: 모드별 표준값)")
    parser.add_argument("--embed-url", help="LAN 임베딩 서버(:8002)")
    parser.add_argument("--pg", help="PostgreSQL DSN")
    parser.add_argument("--batch-size", type=int, default=16, help="임베딩·적재 배치 크기")
    parser.add_argument("--dry-run", action="store_true", help="수집·결합까지만(임베딩·DB 없음)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # source 기본값은 모드별로 다르다(라이선스가 다르므로 반드시 분리 저장).
    if not args.source:
        args.source = "pytorch_kr" if args.mode == "tutorials" else "pytorch_forum_kr"
    if args.mode == "tutorials" and not args.repo:
        parser.error("--mode tutorials 에는 --repo(저장소 clone 경로)가 필요합니다")
    if not args.dry_run and (not args.pg or not args.embed_url):
        parser.error("실제 적재에는 --pg 와 --embed-url 이 필요합니다(리허설은 --dry-run)")
    return asyncio.run(run_ingest(args))


if __name__ == "__main__":
    raise SystemExit(main())
