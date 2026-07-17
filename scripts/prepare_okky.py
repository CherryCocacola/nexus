# OKKY(okky.kr) 코딩 Q&A를 코딩 지식 RAG(tb_knowledge, source='okky')로 적재하는 준비 스크립트.
"""
OKKY Q&A(/questions/{id})를 스크래핑·큐레이션해 tb_knowledge(source='okky')로 적재한다.

── 이 파일이 하는 일 (한눈에) ─────────────────────────────
OKKY는 Next.js App Router(RSC) 앱이라 질문/답변이 일반 HTML 태그가 아니라
`self.__next_f` RSC 스트리밍 페이로드(정형 JSON)에 들어 있다. 이를 파싱해 질문 +
채택답변 + 고득점답변을 하나의 문서로 결합하고, 코드 인지 청킹으로 나눠 적재한다.
prepare_stackoverflow.py와 대칭 구조(그쪽은 XML 덤프, 이쪽은 라이브 페이지 RSC).

── 법적·스크래핑 준수 (중요) ─────────────────────────────
사용자 결정: 코딩 모드는 **팀 내부 전용** 사용이며, 상용 전환 시 provenance
(source='okky')로 `DELETE WHERE source='okky'`+재인덱스로 제거한다(progress.md 근거).
OKKY 콘텐츠는 사용자 저작물(저작권 작성자 귀속·재사용 라이선스 없음)이므로 상용
배포 전 법무 검토/eBrain 허가가 게이트다. 스크래핑은 robots.txt를 준수한다:
  - 허용 경로만: /questions/{id}, /articles/{id}, /sitemap.xml
  - 금지 경로 접근 안 함: /api/ (그래서 페이지 RSC를 파싱), /users/*/questions, /auth/
  - rate-limit(기본 1.2초/요청, robots Crawl-delay 존중), 소량·야간 권장.

── 검증 범위 ──────────────────────────────────────────────
RSC 파싱·결합·엔트리 생성은 --probe(실 페이지 1건) + 합성 페이로드 단위테스트로
검증한다. 임베딩·DB 쓰기(run_ingest)는 실 인프라 필요라 가드(prepare_stackoverflow 대칭).

작성자: Nexus 팀 / 작성일: 2026-07-17
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

logger = logging.getLogger("nexus.scripts.prepare_okky")

_BASE = "https://okky.kr"
_OKKY_LICENSE = "OKKY-UGC (no reuse license; internal-only, removable by source)"
# self.__next_f.push([1,"..."]) 조각에서 JSON 문자열 리터럴을 뽑는 정규식.
_RSC_RE = re.compile(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)')
# 예의: robots Crawl-delay=1 준수 + 여유. 요청 간 최소 간격(초).
_RATE_DELAY = 1.2
_UA = "Mozilla/5.0 (compatible; NexusResearch/1.0; internal RAG corpus prep)"


def rsc_payload(html: str) -> str:
    """HTML에서 self.__next_f RSC 조각을 이어붙여 하나의 페이로드 문자열로 만든다.

    각 조각은 JSON 문자열 리터럴이라 json.loads로 이스케이프를 풀어 이어붙이면
    정형 JSON 조각들이 담긴 평문 페이로드가 된다.
    """
    parts = []
    for m in _RSC_RE.finditer(html):
        try:
            parts.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue
    return "".join(parts)


def _json_value_after(payload: str, key: str, start: int = 0) -> tuple[Any, int] | None:
    """payload에서 `"key":` 뒤의 JSON 값(객체/배열)을 균형괄호로 잘라 파싱한다.

    RSC 페이로드는 순수 JSON이 아니지만, "answers":{...} 같은 개별 값은 유효
    JSON이다. 여는 괄호({ 또는 [)부터 짝이 맞는 닫는 괄호까지 슬라이스해 json.loads.
    반환: (값, 다음 탐색 위치) 또는 None.
    """
    i = payload.find(f'"{key}":', start)
    if i < 0:
        return None
    j = i + len(key) + 3
    while j < len(payload) and payload[j] not in "{[":
        j += 1
    if j >= len(payload):
        return None
    open_ch = payload[j]
    close_ch = "}" if open_ch == "{" else "]"
    depth, k, in_str, esc = 0, j, False, False
    while k < len(payload):
        c = payload[k]
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(payload[j : k + 1]), k + 1
                    except json.JSONDecodeError:
                        return None
        k += 1
    return None


def parse_question(html: str) -> dict[str, Any] | None:
    """OKKY 질문 페이지 HTML(RSC)에서 질문/답변 데이터를 추출한다.

    반환: {id, title, text(HTML), tags[], selected_answer_id, answers:[{id,text,votes,selected}]}
    필수 필드가 없으면 None.
    """
    p = rsc_payload(html)
    if not p:
        return None

    def _str(key: str) -> str | None:
        m = re.search(r'"' + key + r'":"((?:[^"\\]|\\.)*)"', p)
        if not m:
            return None
        try:
            return json.loads('"' + m.group(1) + '"')
        except json.JSONDecodeError:
            return m.group(1)

    title = _str("title")
    if not title:
        return None
    sel = re.search(r'"selectedAnswerId":(\d+|null)', p)
    selected_id = int(sel.group(1)) if sel and sel.group(1) != "null" else None
    tags_v = _json_value_after(p, "tags")
    tags = []
    if tags_v and isinstance(tags_v[0], list):
        tags = [t.get("name") or t.get("text") for t in tags_v[0] if isinstance(t, dict)]
        tags = [t for t in tags if t]

    # 질문 본문 text: answers 앞쪽의 첫 "text"가 질문 본문(답변 text는 answers 안).
    ai = p.find('"answers":')
    q_body = ""
    if ai > 0:
        qm = re.search(r'"text":"((?:[^"\\]|\\.)*)"', p[:ai])
        if qm:
            try:
                q_body = json.loads('"' + qm.group(1) + '"')
            except json.JSONDecodeError:
                q_body = qm.group(1)

    answers: list[dict[str, Any]] = []
    av = _json_value_after(p, "answers")
    if av and isinstance(av[0], dict):
        for a in av[0].get("content", []):
            if not isinstance(a, dict):
                continue
            answers.append({
                "id": a.get("id"),
                "text": a.get("text") or "",
                "votes": a.get("voteCount", 0) + a.get("assentCount", 0),
                "selected": bool(a.get("selected")),
            })

    qid_m = re.search(r'"id":(\d+)', p)
    return {
        "id": qid_m and int(qid_m.group(1)),
        "title": title,
        "text": q_body,
        "tags": tags,
        "selected_answer_id": selected_id,
        "answers": answers,
    }


def build_combined_document(q: dict[str, Any], min_answer_votes: int) -> tuple[str, list[int]]:
    """질문 + (채택답변 + 고득점답변)을 하나의 마크다운 문서로 결합한다(SO와 동일 패턴).

    반환: (문서, 포함 답변 id 목록). 포함 답변 없으면 목록이 비어 큐레이션에서 제외.
    """
    sel = q.get("selected_answer_id")
    parts: list[str] = [f"# {q['title']}".strip()]
    qb = html_to_markdown(q.get("text") or "")
    if qb:
        parts.append(qb)
    ordered = sorted(
        q.get("answers", []),
        key=lambda a: (0 if a["id"] == sel or a["selected"] else 1, -a["votes"]),
    )
    used: list[int] = []
    for a in ordered:
        is_acc = a["id"] == sel or a["selected"]
        if not is_acc and a["votes"] < min_answer_votes:
            continue
        label = "채택된 답변" if is_acc else f"답변 (추천 {a['votes']})"
        parts.append(f"## {label}")
        body = html_to_markdown(a["text"])
        if body:
            parts.append(body)
        used.append(a["id"])
    return "\n\n".join(x for x in parts if x.strip()), used


def build_entries(
    questions: list[dict[str, Any]],
    *,
    source: str = "okky",
    min_answer_votes: int = 1,
    min_question_votes: int = 0,
    require_code: bool = False,
    min_doc_chars: int = 80,
    max_chunk_chars: int = 1500,
) -> list[KnowledgeEntry]:
    """큐레이션 통과 질문을 결합·청킹해 KnowledgeEntry로 만든다(source='okky').

    PK 유일성: section=f"q{qid}"로 질문마다 유일한 id 공간. provenance metadata에
    url·license·question_id·answer_ids 기록(상용 전환 시 소스별 분리용).
    """
    entries: list[KnowledgeEntry] = []
    for q in questions:
        qid = q.get("id")
        if qid is None:
            continue
        doc, used = build_combined_document(q, min_answer_votes)
        if not used or len(doc) < min_doc_chars:
            continue
        if require_code and "```" not in doc:
            continue
        chunks = chunk_code_aware(doc, max_chars=max_chunk_chars)
        meta = {
            "question_id": qid,
            "url": f"{_BASE}/questions/{qid}",
            "license": _OKKY_LICENSE,
            "answer_ids": used,
        }
        for i, ch in enumerate(chunks):
            entries.append(KnowledgeEntry(
                source=source, title=q["title"], content=ch, section=f"q{qid}",
                chunk_index=i, total_chunks=len(chunks), tags=tuple(q.get("tags") or []),
                metadata=meta,
            ))
    return entries


def fetch_question(qid: int) -> dict[str, Any] | None:
    """robots 허용 경로 /questions/{id}를 rate-limit 준수해 받아 파싱한다."""
    import httpx

    time.sleep(_RATE_DELAY)  # 예의: 요청 간 간격
    r = httpx.get(f"{_BASE}/questions/{qid}", headers={"User-Agent": _UA},
                  timeout=20, follow_redirects=True)
    if r.status_code != 200:
        return None
    q = parse_question(r.text)
    if q is not None:
        # PK 정확성: 질문 id는 URL의 qid가 권위값(페이로드 첫 "id"는 모듈/라우트 id일 수 있음).
        q["id"] = qid
    return q


def run_probe(args: argparse.Namespace) -> int:
    """실 페이지 1건을 받아 파싱 결과를 출력한다(검증용)."""
    q = fetch_question(args.probe)
    if not q:
        print(f"파싱 실패 qid={args.probe}")
        return 1
    doc, used = build_combined_document(q, args.min_votes)
    print(f"title: {q['title']}")
    print(f"tags: {q['tags']}  selected: {q['selected_answer_id']}  answers: {len(q['answers'])}")
    print(f"결합문서 {len(doc)}자, 포함답변 {used}, 코드블록: {'```' in doc}")
    print("--- 문서 앞부분 ---")
    print(doc[:600])
    return 0


def iter_listing_ids(board: str, max_pages: int) -> list[int]:
    """기술 Q&A 목록(/questions/{board}?page=N)에서 질문 id를 열거한다(robots 허용).

    각 페이지 RSC에서 "id":N,"title": 패턴으로 질문 항목 id를 뽑는다(20개/페이지).
    rate-limit을 준수하며 max_pages까지 수집한다.
    """
    import httpx

    ids: list[int] = []
    seen: set[int] = set()
    for page in range(1, max_pages + 1):
        time.sleep(_RATE_DELAY)
        # 1페이지는 파라미터 없는 기본 URL이 SSR로 목록을 렌더한다(?page=1은 빈 응답 가능).
        base = f"{_BASE}/questions/{board}"
        url = base if page == 1 else f"{base}?page={page}"
        r = httpx.get(url, headers={"User-Agent": _UA}, timeout=20, follow_redirects=True)
        if r.status_code != 200:
            break
        p = rsc_payload(r.text)
        page_ids = [int(m) for m in re.findall(r'"id":(\d+),"title":', p)]
        new = [i for i in page_ids if i not in seen]
        if not new:
            break
        for i in new:
            seen.add(i)
        ids.extend(new)
        logger.info("열거: page %d → 누적 %d개 질문", page, len(ids))
    return ids


async def _embed_texts(base_url: str, texts: list[str]) -> list[list[float]]:
    """LAN 임베딩 서버(:8002) /v1/embed 로 벡터를 받는다(prepare_kowiki 계약)."""
    import httpx

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(base_url.rstrip("/") + "/v1/embed", json={"texts": texts})
        r.raise_for_status()
        return r.json()["embeddings"]


async def run_ingest(args: argparse.Namespace) -> int:
    """열거 → 스크랩(rate-limit) → 결합 → 임베딩 → tb_knowledge(source) UPSERT.

    소규모 파일럿용. --dry-run이면 스크랩·결합까지만(임베딩·DB 없음). source는 파일럿
    격리를 위해 'okky-pilot' 등으로 지정 가능(상용 전환 시 source로 제거).
    """
    ids = iter_listing_ids(args.board, args.pages)
    if args.limit:
        ids = ids[: args.limit]
    logger.info("스크랩 대상 %d개 질문 (rate-limit %.1f초/건)", len(ids), _RATE_DELAY)

    questions: list[dict[str, Any]] = []
    for n, qid in enumerate(ids, 1):
        q = fetch_question(qid)
        if q:
            questions.append(q)
        if n % 20 == 0:
            logger.info("스크랩 %d/%d", n, len(ids))

    entries = build_entries(
        questions,
        source=args.source,
        min_answer_votes=args.min_votes,
        require_code=args.require_code,
    )
    logger.info("엔트리 %d개 생성(질문 %d, source=%s)", len(entries), len(questions), args.source)
    if args.dry_run:
        print(f"[dry-run] questions={len(questions)} entries={len(entries)} source={args.source}")
        return 0

    import asyncpg

    pool = await asyncpg.create_pool(args.pg, min_size=1, max_size=3)
    store = KnowledgeStore(pg_pool=pool)
    await store.ensure_schema()
    stored = 0
    try:
        for i in range(0, len(entries), args.batch_size):
            batch = entries[i : i + args.batch_size]
            try:
                embs = await _embed_texts(args.embed_url, [e.content for e in batch])
            except Exception as e:
                logger.warning("임베딩 실패 배치 %d: %s", i, e)
                continue
            for entry, emb in zip(batch, embs, strict=True):
                await store.add(replace(entry, embedding=tuple(emb)))
                stored += 1
    finally:
        await pool.close()
    logger.info("완료: %d청크 적재(source=%s)", stored, args.source)
    print(f"stored={stored} source={args.source}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OKKY Q&A → tb_knowledge(source='okky')")
    parser.add_argument("--probe", type=int, help="질문 1건 파싱 검증(qid)")
    parser.add_argument("--board", default="tech", help="Q&A 게시판(tech/career/qna-etc)")
    parser.add_argument("--pages", type=int, default=5, help="열거할 목록 페이지 수(20개/페이지)")
    parser.add_argument("--limit", type=int, default=None, help="스크랩 질문 상한")
    parser.add_argument("--min-votes", type=int, default=1, help="비채택 답변 최소 추천")
    parser.add_argument("--require-code", action="store_true", help="코드 블록 있는 질문만")
    parser.add_argument("--source", default="okky", help="tb_knowledge.source(파일럿=okky-pilot)")
    parser.add_argument("--embed-url", help="LAN 임베딩 서버(:8002)")
    parser.add_argument("--pg", help="PostgreSQL DSN")
    parser.add_argument("--batch-size", type=int, default=16, help="임베딩·적재 배치 크기")
    parser.add_argument("--dry-run", action="store_true", help="스크랩·결합까지만(임베딩·DB 없음)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.probe:
        return run_probe(args)
    if not args.dry_run and (not args.pg or not args.embed_url):
        parser.error("실제 적재에는 --pg 와 --embed-url 이 필요합니다(리허설은 --dry-run)")
    return asyncio.run(run_ingest(args))


if __name__ == "__main__":
    raise SystemExit(main())
