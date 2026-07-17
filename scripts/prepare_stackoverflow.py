# Stack Overflow Posts.xml 덤프를 코딩 지식 RAG(tb_knowledge, source='so')로 적재하는 준비 스크립트.
"""
Stack Overflow 공식 데이터 덤프(Posts.xml)를 파싱·큐레이션해 tb_knowledge로 적재한다.

── 이 파일이 하는 일 (한눈에) ─────────────────────────────
SO 덤프의 질문/답변을 짝지어 "질문 + 채택답변 + 고득점답변"을 하나의 문서로
결합하고, 코드 인지 청킹(coding_corpus.chunk_code_aware)으로 나눠 임베딩 후
tb_knowledge(source='so')에 넣는다. prepare_kowiki.py와 같은 "준비 단계" 스크립트다.

── 왜 별도 스크립트인가 (에어갭) ─────────────────────────
prepare_kowiki.py와 동일 — 운영 코드가 아닌 최초 1회 코퍼스 구축용이라 별도 단계로
허용된다. 덤프는 미리 받아둔 로컬 파일(--dump)로 받고, 임베딩은 LAN 서버(:8002)만 쓴다.

── FABLE5 검토가 요구한 핵심 설계(반드시 지킴) ───────────
  C1/N1 (PK 무손실): SO 질문 제목은 유일하지 않고 한 질문에 채택+고득점 답변이 여럿
      있을 수 있다. tb_knowledge PK는 SHA256(source|title|section|chunk_index)뿐이라,
      제목·section만으로는 서로 다른 질문/답변이 같은 id로 조용히 덮어써진다. 그래서
      **질문+선택된 답변들을 '단일 문서'로 결합**하고 **section=q{question_id}로 앵커링**해
      질문마다 유일한 id 공간을 준다(답변 다수 → 한 문서라 충돌 원천 소멸).
  Posts.xml 구조: 요소 트리가 아니라 <row Id=".." PostTypeId="1|2" Body=".." .../> 형태의
      **속성 기반 평면 XML**이다(kowiki의 <page><revision><text>와 다름). 답변은 ParentId로
      질문을 참조하므로 질문/답변을 버킷팅해 페어링한다.

── 이 파일의 검증 범위 ────────────────────────────────────
파싱·페어링·결합·큐레이션·엔트리 생성(iter_posts~build_entries)은 합성 Posts.xml
fixture로 단위테스트한다(tests/unit/test_prepare_stackoverflow.py). 임베딩 호출과
tb_knowledge 쓰기(run_ingest)는 실 덤프 + LAN 인프라가 필요하므로 가드해 두고,
Phase 0b/1에서 실서버로 검증한다(mock 금지 원칙 [[feedback_real_servers_no_mock]]).

── 실행(개요, 실 덤프 준비 후) ────────────────────────────
  1. 덤프 획득(연결된 준비 PC에서 수동):
       # archive.org의 stackoverflow.com-Posts.7z → Posts.xml 로 해제
  2. 통계 스캔(임베딩·DB 없이 규모 파악 — N 결정용):
       python scripts/prepare_stackoverflow.py --dump Posts.xml --stats --min-score 5
  3. 적재(파싱→결합→청킹→임베딩→UPSERT):
       python scripts/prepare_stackoverflow.py --dump Posts.xml \\
         --min-score 5 --embed-url http://192.168.21.112:8002 --pg "postgresql://..."
  4. 벡터 인덱스(전량 적재 후 1회 — 재인덱스 절차는 별도 문서):
       python scripts/prepare_stackoverflow.py --build-index --pg "postgresql://..."

작성자: Nexus 팀 / 작성일: 2026-07-17
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys

# 속성 기반 평면 XML 스트리밍 파싱. 로컬 신뢰 덤프만 파싱하므로 S405/S314는 noqa.
import xml.etree.ElementTree as ET  # noqa: S405 — 로컬 신뢰 SO 덤프만 파싱
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

# 이 파일 단독 실행 시 core/coding_corpus를 import할 수 있게 경로를 잡는다.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (coding_corpus)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 루트 (core)

from coding_corpus import chunk_code_aware, html_to_markdown  # noqa: E402

from core.rag.knowledge_store import KnowledgeEntry, KnowledgeStore  # noqa: E402

logger = logging.getLogger("nexus.scripts.prepare_stackoverflow")

# SO Posts.xml PostTypeId: 1=질문, 2=답변.
_POST_QUESTION = 1
_POST_ANSWER = 2

# Tags 속성 파싱: 신형 "<python><list>" 또는 구형 "|python|list|" 모두 대응.
_TAG_ANGLE_RE = re.compile(r"<([^>]+)>")

# SO 콘텐츠 라이선스(덤프 기준). provenance에 남겨 나중에 소스별 분리가 가능하게 한다.
_SO_LICENSE = "CC BY-SA 4.0"


def parse_tags(raw: str) -> list[str]:
    """Tags 속성 문자열을 태그 리스트로 변환한다(신형 <>·구형 || 모두)."""
    if not raw:
        return []
    angle = _TAG_ANGLE_RE.findall(raw)
    if angle:
        return [t.strip().lower() for t in angle if t.strip()]
    # 구형 파이프 구분 형식 폴백
    return [t.strip().lower() for t in raw.strip("|").split("|") if t.strip()]


def _int_or_none(value: str | None) -> int | None:
    """정수 속성을 안전하게 파싱한다(없거나 비정상이면 None)."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def iter_posts(xml_stream: Any) -> Iterator[dict[str, Any]]:
    """Posts.xml을 스트리밍 파싱해 게시물(row)을 하나씩 dict로 방출한다.

    왜 스트리밍인가: Posts.xml은 비압축 수십~100GB라 통째로 못 올린다. <row> 종료
    이벤트마다 처리 후 elem.clear()로 메모리를 즉시 비운다(대용량 필수).
    질문(type 1)·답변(type 2)만 방출하고 그 외(태그위키 등)는 건너뛴다.
    """
    for _, elem in ET.iterparse(xml_stream, events=("end",)):  # noqa: S314 — 로컬 신뢰 파일
        if elem.tag != "row":
            continue
        a = elem.attrib
        post_type = _int_or_none(a.get("PostTypeId"))
        try:
            if post_type not in (_POST_QUESTION, _POST_ANSWER):
                continue
            pid = _int_or_none(a.get("Id"))
            if pid is None:
                continue
            # CreationDate="2008-09-02T03:41:06.880" → 연도(정수). 날짜 필터용.
            cd = a.get("CreationDate", "") or ""
            year = int(cd[:4]) if cd[:4].isdigit() else 0
            yield {
                "id": pid,
                "type": post_type,
                "parent_id": _int_or_none(a.get("ParentId")),
                "accepted_answer_id": _int_or_none(a.get("AcceptedAnswerId")),
                "score": _int_or_none(a.get("Score")) or 0,
                "year": year,
                "title": a.get("Title", "") or "",
                "body": a.get("Body", "") or "",
                "tags": parse_tags(a.get("Tags", "")),
            }
        finally:
            # 처리 여부와 무관하게 항상 메모리 비움(대용량 덤프 필수).
            elem.clear()


def load_posts(
    xml_stream: Any,
    limit: int | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """게시물을 질문(id→질문)과 답변(부모 질문 id→답변 리스트)으로 버킷팅한다.

    반환: (questions_by_id, answers_by_parent).
    주의(규모): 이 함수는 (limit까지의) 게시물을 메모리에 올린다. 소규모(파일럿·
    fixture·limit)엔 충분하나, 전량 덤프(수천만 행)엔 SQLite 디스크 스테이징 2-pass가
    필요하다(run_ingest에서 실덤프 배선 시 도입 예정). 페어링 '로직'은 여기서 확정·검증한다.

    Args:
        xml_stream: Posts.xml 파일 객체(또는 파일류).
        limit: 처리할 최대 row 수(None=전량). 실덤프(96GB, 수천만 행)를 OOM 없이
               표본 검증하거나 파일럿 규모를 뽑을 때 쓴다.
    """
    questions: dict[int, dict[str, Any]] = {}
    answers: dict[int, list[dict[str, Any]]] = {}
    for count, post in enumerate(iter_posts(xml_stream)):
        if limit is not None and count >= limit:
            break  # 표본 상한 도달 → 조기 종료(전량 덤프 OOM 방지)
        if post["type"] == _POST_QUESTION:
            questions[post["id"]] = post
        elif post["type"] == _POST_ANSWER and post["parent_id"] is not None:
            answers.setdefault(post["parent_id"], []).append(post)
    return questions, answers


def build_combined_document(
    question: dict[str, Any],
    answers: list[dict[str, Any]],
    min_answer_score: int,
) -> tuple[str, list[int]]:
    """질문 + (채택답변 + 고득점답변)을 하나의 마크다운 문서로 결합한다(N1 해소).

    답변은 '채택된 것 먼저, 그다음 점수 내림차순'으로 배치한다. 채택답변은 점수와
    무관하게 포함하고, 나머지는 min_answer_score 이상만 포함한다. HTML Body는
    html_to_markdown으로 변환해 코드 블록을 펜스로 보존한다.

    반환: (결합 문서 문자열, 실제로 포함된 답변 id 목록). 포함 답변이 없으면
    답변 id 목록이 비어, 호출부(build_entries)가 그 질문을 큐레이션에서 제외한다.
    """
    acc_id = question.get("accepted_answer_id")
    parts: list[str] = [f"# {question['title']}".strip()]
    q_body = html_to_markdown(question["body"])
    if q_body:
        parts.append(q_body)

    # 채택(0) 우선, 그다음 점수 높은 순.
    ordered = sorted(answers, key=lambda ans: (0 if ans["id"] == acc_id else 1, -ans["score"]))
    used_ids: list[int] = []
    for ans in ordered:
        is_accepted = ans["id"] == acc_id
        if not is_accepted and ans["score"] < min_answer_score:
            continue
        label = "채택된 답변" if is_accepted else f"답변 (추천 {ans['score']})"
        parts.append(f"## {label}")
        body_md = html_to_markdown(ans["body"])
        if body_md:
            parts.append(body_md)
        used_ids.append(ans["id"])

    doc = "\n\n".join(p for p in parts if p.strip())
    return doc, used_ids


def build_entries(
    questions: dict[int, dict[str, Any]],
    answers_by_parent: dict[int, list[dict[str, Any]]],
    *,
    source: str = "so",
    min_answer_score: int = 5,
    min_question_score: int = 0,
    require_code: bool = False,
    min_doc_chars: int = 80,
    max_chunk_chars: int = 1500,
) -> list[KnowledgeEntry]:
    """큐레이션을 통과한 질문을 결합·청킹해 KnowledgeEntry 리스트로 만든다.

    큐레이션 기준:
      - 질문 점수 >= min_question_score (질문 자체의 품질 게이트, 0=무필터).
      - 채택답변 OR 점수 >= min_answer_score 답변이 하나라도 있어야 한다.
      - require_code=True면 결합 문서에 코드 블록(```)이 있어야 한다.
      - 결합 문서 길이가 min_doc_chars 이상.
    PK 유일성(C1): section=f"q{qid}" + chunk_index로 질문마다 유일한 id 공간을 준다.

    왜 min_question_score가 필요한가(실측 근거): min_answer_score만으로는 규모가
    거의 안 줄어든다 — 대부분 질문이 '채택 답변'(점수 무관 포함)으로 통과하기
    때문이다. 질문 점수 게이트가 코퍼스 규모·품질을 조이는 실질 레버다.
    """
    entries: list[KnowledgeEntry] = []
    # id 순서로 처리해 결과가 결정론적이게 한다(재현성).
    for qid in sorted(questions):
        q = questions[qid]
        # 질문 품질 게이트 — 저품질 질문을 결합·청킹 전에 싸게 걸러낸다.
        if q["score"] < min_question_score:
            continue
        doc, used_ids = build_combined_document(
            q, answers_by_parent.get(qid, []), min_answer_score
        )
        if not used_ids:  # 자격 있는 답변이 없음 → 제외
            continue
        if require_code and "```" not in doc:
            continue
        if len(doc) < min_doc_chars:
            continue

        chunks = chunk_code_aware(doc, max_chars=max_chunk_chars)
        # provenance — 소스별 분리·출처표시(CC BY-SA)를 위한 metadata.
        metadata: dict[str, Any] = {
            "question_id": qid,
            "url": f"https://stackoverflow.com/q/{qid}",
            "license": _SO_LICENSE,
            "score": q["score"],
            "answer_ids": used_ids,
        }
        for i, chunk in enumerate(chunks):
            entries.append(
                KnowledgeEntry(
                    source=source,
                    title=q["title"],
                    content=chunk,
                    section=f"q{qid}",  # ← C1 앵커링: 질문마다 유일
                    chunk_index=i,
                    total_chunks=len(chunks),
                    tags=tuple(q["tags"]),
                    metadata=metadata,
                )
            )
    return entries


def stage_to_sqlite(dump_path: Path, db_path: str, min_year: int, min_qscore: int,
                    max_rows: int | None = None) -> int:
    """Pass 1: 덤프를 스트리밍하며 (연도>=min_year, 질문점수>=min_qscore) 질문과 그
    답변만 SQLite에 스테이징한다(전량 메모리 회피 = OOM 방지, 예전·저품질 제외).

    Id 오름차순이라 답변의 부모(kept 질문)는 이미 kept 집합에 있다. 본문은 SQLite(디스크)
    에 저장하고, 메모리엔 kept 질문 id 집합만 유지한다. 반환: 스테이징한 질문 수.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.executescript(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
        "DROP TABLE IF EXISTS q; DROP TABLE IF EXISTS a;"
        "CREATE TABLE q(id INTEGER PRIMARY KEY, title TEXT, body TEXT, tags TEXT,"
        " score INT, accepted INT);"
        "CREATE TABLE a(id INTEGER, parent INTEGER, body TEXT, votes INT);"
    )
    kept: set[int] = set()
    nq = na = seen = 0
    with open(dump_path, "rb") as f:  # noqa: ASYNC230 — 배치 준비 스크립트
        cur = conn.cursor()
        for post in iter_posts(f):
            seen += 1
            if post["type"] == _POST_QUESTION:
                if post["year"] >= min_year and post["score"] >= min_qscore:
                    kept.add(post["id"])
                    cur.execute(
                        "INSERT OR IGNORE INTO q VALUES(?,?,?,?,?,?)",
                        (post["id"], post["title"], post["body"], json.dumps(post["tags"]),
                         post["score"], post["accepted_answer_id"] or 0),
                    )
                    nq += 1
            elif post["type"] == _POST_ANSWER and post["parent_id"] in kept:
                cur.execute("INSERT INTO a VALUES(?,?,?,?)",
                            (post["id"], post["parent_id"], post["body"], post["score"]))
                na += 1
            if seen % 2000000 == 0:
                conn.commit()
                logger.info("스테이징: %dM행 스캔 → 질문 %d, 답변 %d", seen // 1000000, nq, na)
            if max_rows and seen >= max_rows:
                break
    conn.execute("CREATE INDEX ia ON a(parent)")
    conn.commit()
    conn.close()
    logger.info("스테이징 완료: 질문 %d, 답변 %d (총 %d행 스캔)", nq, na, seen)
    return nq


async def run_staged_ingest(args: argparse.Namespace) -> int:
    """SQLite 스테이징 기반 대량 적재: (Pass1 스테이징) → Pass2 질문배치 build→임베딩→UPSERT.

    --skip-stage면 기존 SQLite를 재사용(Pass2만). require_code/min_answer_score는 build에서.
    """
    import sqlite3

    if not args.skip_stage:
        stage_to_sqlite(Path(args.dump), args.sqlite, args.min_year,
                        args.min_question_score, args.stage_max_rows)
    if args.dry_run:
        conn = sqlite3.connect(args.sqlite)
        nq = conn.execute("SELECT count(*) FROM q").fetchone()[0]
        na = conn.execute("SELECT count(*) FROM a").fetchone()[0]
        conn.close()
        print(f"[dry-run] staged questions={nq} answers={na}")
        return 0

    import asyncpg

    conn = sqlite3.connect(args.sqlite)
    conn.row_factory = sqlite3.Row
    qids = [r[0] for r in conn.execute("SELECT id FROM q").fetchall()]
    logger.info("스테이징 질문 %d개 → 적재 시작", len(qids))
    pool = await asyncpg.create_pool(args.pg, min_size=1, max_size=3)
    store = KnowledgeStore(pg_pool=pool)
    await store.ensure_schema()
    stored = 0
    q_batch = 500
    try:
        for i in range(0, len(qids), q_batch):
            batch_ids = qids[i : i + q_batch]
            marks = ",".join("?" * len(batch_ids))
            questions: dict[int, dict[str, Any]] = {}
            for r in conn.execute(f"SELECT * FROM q WHERE id IN ({marks})", batch_ids):  # noqa: S608 — marks=플레이스홀더, 값 바인딩
                questions[r["id"]] = {
                    "id": r["id"], "title": r["title"], "body": r["body"],
                    "tags": json.loads(r["tags"]), "score": r["score"],
                    "accepted_answer_id": r["accepted"] or None,
                }
            answers_by_parent: dict[int, list[dict[str, Any]]] = {}
            for r in conn.execute(f"SELECT * FROM a WHERE parent IN ({marks})", batch_ids):  # noqa: S608 — marks=플레이스홀더, 값 바인딩
                answers_by_parent.setdefault(r["parent"], []).append(
                    {"id": r["id"], "body": r["body"], "score": r["votes"]})
            entries = build_entries(
                questions, answers_by_parent, source=args.source,
                min_answer_score=args.min_score, min_question_score=0,
                require_code=args.require_code,
            )
            for j in range(0, len(entries), args.batch_size):
                b = entries[j : j + args.batch_size]
                try:
                    embs = await _embed_texts(args.embed_url, [e.content for e in b])
                except Exception as e:
                    logger.warning("임베딩 실패: %s", e)
                    continue
                for entry, emb in zip(b, embs, strict=True):
                    await store.add(replace(entry, embedding=tuple(emb)))
                    stored += 1
            done = min(i + q_batch, len(qids))
            logger.info("적재 진행: 질문 %d/%d, %d청크", done, len(qids), stored)
    finally:
        conn.close()
        await pool.close()
    logger.info("완료: %d청크 적재(source=%s)", stored, args.source)
    print(f"stored={stored} source={args.source}")
    return 0


def run_stats(args: argparse.Namespace) -> int:
    """임베딩·DB 없이 덤프를 스캔해 큐레이션 규모를 집계한다(N 결정용, 저비용).

    FABLE5 N5 대응 — 큐레이션 임계(min_answer_score)별로 통과 질문/청크 수를 미리
    파악해 적재 시간·디스크를 추정할 수 있게 한다.
    """
    dump = Path(args.dump)
    if not dump.exists():
        logger.error("덤프 파일 없음: %s", dump)
        return 1
    with open(dump, "rb") as f:  # noqa: ASYNC230 — 준비 스크립트(동기 배치)
        questions, answers = load_posts(f, limit=args.limit)
    entries = build_entries(
        questions,
        answers,
        min_answer_score=args.min_score,
        min_question_score=args.min_question_score,
        require_code=args.require_code,
    )
    kept_q = len({e.section for e in entries})
    limit_note = f" [표본 {args.limit} rows]" if args.limit else " [전량]"
    logger.info(
        "통계%s: 질문 %d개 / 답변 그룹 %d개 → 큐레이션 통과 질문 %d개, 청크 %d개 "
        "(min_score=%d, require_code=%s)",
        limit_note, len(questions), len(answers), kept_q, len(entries),
        args.min_score, args.require_code,
    )
    print(
        f"questions={len(questions)} answer_groups={len(answers)} "
        f"kept_questions={kept_q} chunks={len(entries)} "
        f"min_score={args.min_score} min_qscore={args.min_question_score} "
        f"require_code={args.require_code} limit={args.limit}"
    )
    return 0


async def _embed_texts(base_url: str, texts: list[str]) -> list[list[float]]:
    """LAN 임베딩 서버(:8002)에 텍스트 배치를 보내 벡터를 받는다(prepare_kowiki와 동일 계약).

    POST /v1/embed {"texts": [...]} → {"embeddings": [[..1024..], ...]}.
    """
    import httpx  # 선택적 의존성 — 적재 시에만 필요

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(base_url.rstrip("/") + "/v1/embed", json={"texts": texts})
        r.raise_for_status()
        return r.json()["embeddings"]


async def run_ingest(args: argparse.Namespace) -> int:
    """SO 덤프(표본) → 엔트리 생성 → 임베딩 → tb_knowledge(source) UPSERT.

    파일럿 규모는 --limit로 바운드해 인메모리로 처리한다(전량은 SQLite 스테이징
    2-pass가 별도 과제). --dry-run이면 엔트리 생성까지만 하고 임베딩·DB를 건드리지
    않는다(리허설, 인프라 없이 안전 검증). source는 파일럿 격리를 위해 'so-pilot'
    등으로 지정할 수 있다.
    """
    dump = Path(args.dump)
    if not dump.exists():  # noqa: ASYNC240 — 준비 스크립트(동기 파일 확인)
        logger.error("덤프 파일 없음: %s", dump)
        return 1

    with open(dump, "rb") as f:  # noqa: ASYNC230 — 준비 스크립트(동기 배치 파싱)
        questions, answers = load_posts(f, limit=args.limit)
    entries = build_entries(
        questions,
        answers,
        source=args.source,
        min_answer_score=args.min_score,
        min_question_score=args.min_question_score,
        require_code=args.require_code,
    )
    logger.info(
        "엔트리 %d개 생성(질문 %d, 답변그룹 %d, source=%s)",
        len(entries), len(questions), len(answers), args.source,
    )
    if args.dry_run:
        print(f"[dry-run] entries={len(entries)} source={args.source} (임베딩·DB 없음)")
        return 0

    import asyncpg  # 선택적 의존성 — 실제 적재 시에만 필요

    pool = await asyncpg.create_pool(args.pg, min_size=1, max_size=3)
    store = KnowledgeStore(pg_pool=pool)
    await store.ensure_schema()

    stored = 0
    batch_size = args.batch_size
    try:
        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            try:
                embs = await _embed_texts(args.embed_url, [e.content for e in batch])
            except Exception as e:
                # 한 배치 임베딩 실패는 그 배치만 건너뛰고 계속한다(전체 중단 방지).
                logger.warning("임베딩 실패 배치 %d: %s", i, e)
                continue
            for entry, emb in zip(batch, embs, strict=True):
                # KnowledgeEntry는 frozen이라 embedding을 담은 새 객체로 교체 후 UPSERT.
                await store.add(replace(entry, embedding=tuple(emb)))
                stored += 1
            if stored % 500 == 0:
                logger.info("적재 진행: %d/%d 청크", stored, len(entries))
    finally:
        await pool.close()

    logger.info("완료: %d청크 적재(source=%s)", stored, args.source)
    print(f"stored={stored} source={args.source}")
    return 0


async def run_build_index(args: argparse.Namespace) -> int:
    """벡터 인덱스 빌드(전량 적재 후). 주의: build_vector_index는 IF NOT EXISTS라
    기존 idx_knowledge_embed가 있으면 no-op이다(FABLE5 C2). 전량 적재 시 실제
    재인덱스(CREATE INDEX CONCURRENTLY + lists 재산정)는 별도 운영 절차로 수행한다.
    파일럿(so-pilot 소량)은 기존 인덱스에 흡수되므로 이 단계가 필요 없다.
    """
    import asyncpg

    pool = await asyncpg.create_pool(args.pg, min_size=1, max_size=2)
    store = KnowledgeStore(pg_pool=pool)
    await store.ensure_schema()
    await store.build_vector_index()
    await pool.close()
    logger.info("build_vector_index 호출 완료(no-op 여부는 C2 주석 참조)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점 — --stats(규모), --dump 적재(임베딩+UPSERT), --build-index."""
    parser = argparse.ArgumentParser(description="Stack Overflow 덤프 → tb_knowledge 준비")
    parser.add_argument("--dump", help="Posts.xml 로컬 경로")
    parser.add_argument("--stats", action="store_true", help="적재 없이 규모만 집계")
    parser.add_argument("--min-score", type=int, default=5, help="비채택 답변 최소 점수")
    parser.add_argument(
        "--min-question-score", type=int, default=0, help="질문 최소 점수(품질 게이트)"
    )
    parser.add_argument("--require-code", action="store_true", help="코드 블록 있는 질문만")
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 row 수(표본 검증용)")
    parser.add_argument("--source", default="so", help="tb_knowledge.source 값(파일럿=so-pilot)")
    parser.add_argument("--embed-url", help="LAN 임베딩 서버(:8002)")
    parser.add_argument("--pg", help="PostgreSQL DSN")
    parser.add_argument("--batch-size", type=int, default=16, help="임베딩·적재 배치 크기")
    parser.add_argument("--dry-run", action="store_true", help="엔트리 생성까지만(임베딩·DB 없음)")
    parser.add_argument("--build-index", action="store_true", help="벡터 인덱스 빌드(C2 주석 참조)")
    # 대량 적재(SQLite 스테이징) — 예전·저품질 제외하고 전량에서 최근·고품질만.
    parser.add_argument("--stage-ingest", action="store_true", help="SQLite 스테이징 대량 적재")
    parser.add_argument("--min-year", type=int, default=2018, help="이 연도 이상만(예전 제외)")
    parser.add_argument("--sqlite", default="so_stage.db", help="스테이징 SQLite 경로")
    parser.add_argument("--skip-stage", action="store_true", help="기존 SQLite 재사용")
    parser.add_argument("--stage-max-rows", type=int, default=None, help="스캔 행 상한(테스트)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.build_index:
        if not args.pg:
            parser.error("--build-index 에는 --pg 가 필요합니다")
        return asyncio.run(run_build_index(args))

    if args.stats:
        if not args.dump:
            parser.error("--stats 에는 --dump 가 필요합니다")
        return run_stats(args)

    if args.stage_ingest:
        if not args.skip_stage and not args.dump:
            parser.error("--stage-ingest 에는 --dump 가 필요합니다(--skip-stage면 불요)")
        if not args.dry_run and (not args.pg or not args.embed_url):
            parser.error("실제 적재에는 --pg 와 --embed-url 이 필요합니다(리허설은 --dry-run)")
        return asyncio.run(run_staged_ingest(args))

    # 적재 모드(인메모리, 소규모)
    if not args.dump:
        parser.error("적재에는 --dump 가 필요합니다(또는 --stats)")
    if not args.dry_run and (not args.pg or not args.embed_url):
        parser.error("실제 적재에는 --pg 와 --embed-url 이 필요합니다(리허설은 --dry-run)")
    return asyncio.run(run_ingest(args))


if __name__ == "__main__":
    raise SystemExit(main())
