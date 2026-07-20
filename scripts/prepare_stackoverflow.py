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
from core.rag.pgvector_base import parse_vector as _parse_vector  # noqa: E402

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


# ── 문서측 번역(B안, 2026-07-19) ────────────────────────────────────────────
# 왜(실측 근거): e5·리랭커가 한국어질의↔영어SO 문서를 매칭 못 해(리랭크 0.0~0.09)
#   빈주입된다. 각 청크에 한국어 제목 글로스를 붙이면 리랭커가 통과시키고(0.0→1.0)
#   벡터 회수도 top-40 임계 위로 올라온다(프로브 검증, progress.md 2026-07-19).
#   질의를 번역하면 한국어 코퍼스(kowiki/okky)를 파괴하므로(실측 5중 3 손실),
#   번역을 '문서측'에 두어 한국어 검색 경로를 전혀 건드리지 않는다.


def apply_gloss(content: str, ko_gloss: str) -> str:
    """청크 본문 맨 앞에 한국어 제목 글로스를 헤더로 덧붙인다.

    왜 임베딩·저장 본문 둘 다에 넣나(프로브 검증): 검색 회수(벡터)뿐 아니라 리랭커도
    이 본문을 보고 점수를 매긴다. 글로스가 리랭크 대상 텍스트에 있어야 한국어질의가
    min_score 게이트를 통과한다(본문만이면 0.0~0.08로 드롭). 한국어 사용자에겐 영어
    답변 위 한글 제목이라 노이즈가 아니라 도움이 된다.

    ko_gloss가 비었으면(번역 실패 폴백) 원문을 그대로 둔다(무손상 — 그 행은 오늘과 동일).

    NUL(0x00)은 항상 제거한다 — PostgreSQL text 컬럼은 널바이트를 거부(UTF8 invalid)하는데,
    A.X 번역 글로스나 원문에 드물게 섞여 들어와 적재가 중단될 수 있다.
    """
    content = content.replace("\x00", "")
    g = ko_gloss.replace("\x00", "").strip()
    if not g:
        return content
    return f"# {g}\n{content}"


def _extract_json_array(text: str) -> Any:
    """모델 응답에서 첫 JSON 배열을 관대하게 추출한다(코드펜스·부연이 섞여도).

    ```json ... ``` 펜스나 앞뒤 잡텍스트가 있어도 첫 '['~마지막 ']' 구간만 파싱한다.
    실패하면 None을 돌려 호출부가 원문 유지 폴백을 타게 한다.
    """
    s = text.find("[")
    e = text.rfind("]")
    if s == -1 or e == -1 or e < s:
        return None
    try:
        return json.loads(text[s : e + 1])
    except json.JSONDecodeError:
        return None


async def _translate_batch(
    client: Any, base_url: str, model: str, titles: list[str]
) -> list[str]:
    """제목 배치를 A.X /v1/chat/completions로 EN→KO 번역해 리스트로 돌려준다.

    번호 매긴 입력을 같은 개수의 JSON 문자열 배열로 받아 파싱한다. 파싱 실패·개수
    불일치·요청 오류면 원문(영어) 리스트를 그대로 돌려준다(무손상 폴백).
    """
    numbered = "\n".join(f"{j + 1}. {t}" for j, t in enumerate(titles))
    prompt = (
        "다음 영어 Stack Overflow 질문 제목들을 자연스러운 한국어 검색어로 번역하라. "
        "기술 용어(함수명·라이브러리·키워드·기호)는 통용 표기를 쓰되 의미를 살려라. "
        "설명·부연 없이 입력과 같은 개수의 JSON 문자열 배열로만 출력하라.\n\n"
        f"{numbered}\n\n"
        '출력 예: ["첫 번째 한국어 번역", "두 번째 한국어 번역"]'
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    try:
        r = await client.post(base_url.rstrip("/") + "/v1/chat/completions", json=payload)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        arr = _extract_json_array(text)
        if isinstance(arr, list) and len(arr) == len(titles):
            return [str(x).strip() for x in arr]
        logger.warning(
            "번역 배치 파싱 불가/개수 불일치(%s vs %d) → 원문 유지",
            len(arr) if isinstance(arr, list) else "None", len(titles),
        )
    except Exception as e:  # noqa: BLE001 — 번역 실패는 원문 유지로 흡수(적재 계속)
        logger.warning("번역 배치 실패 → 원문 유지: %s", e)
    return list(titles)


async def translate_titles(
    base_url: str, model: str, titles: list[str], *,
    batch_size: int = 20,
    timeout: float = 180.0,  # noqa: ASYNC109 — 오프라인 배치, httpx 클라이언트 생성용
) -> dict[str, str]:
    """영어 제목 리스트를 배치로 EN→KO 번역해 {영어제목: 한국어제목}을 돌려준다.

    오프라인 1회 배치라 생성모델(A.X)로 품질 우선 번역한다. 실패 배치는 원문을
    유지하므로 그 행은 오늘과 동일하게 동작한다(부분실패 무손상).
    """
    import httpx

    result: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i in range(0, len(titles), batch_size):
            batch = titles[i : i + batch_size]
            ko = await _translate_batch(client, base_url, model, batch)
            for en, k in zip(batch, ko, strict=True):
                result[en] = k or en
    return result


def _load_gloss_cache(path: str | None) -> dict[str, str]:
    """번역 캐시(JSON) 로드 — 재실행 시 이미 번역한 제목을 건너뛰기 위함(이어받기)."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_gloss_cache(path: str | None, cache: dict[str, str]) -> None:
    """번역 캐시를 원자적으로 저장(중단 안전 — 임시파일 쓰고 교체)."""
    if not path:
        return
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def _as_dict(value: Any) -> dict[str, Any]:
    """asyncpg의 jsonb 컬럼이 str/dict 어느 쪽으로 오든 dict로 정규화한다."""
    if isinstance(value, str):
        return json.loads(value)
    return dict(value) if value else {}


async def _embed_store(
    store: KnowledgeStore, embed_url: str, entries: list[KnowledgeEntry]
) -> int:
    """엔트리 배치를 임베딩(:8002)해 UPSERT하고 적재 건수를 돌려준다.

    임베딩 실패는 그 배치만 건너뛴다(전체 중단 방지) — run_ingest와 동일 정책.
    """
    try:
        embs = await _embed_texts(embed_url, [e.content for e in entries])
    except Exception as e:  # noqa: BLE001 — 배치 임베딩 실패는 스킵(다음 배치 계속)
        logger.warning("임베딩 실패 배치(%d개): %s", len(entries), e)
        return 0
    stored = 0
    for entry, emb in zip(entries, embs, strict=True):
        await store.add(replace(entry, embedding=tuple(emb)))
        stored += 1
    return stored


async def run_glossify(args: argparse.Namespace) -> int:
    """기존 source(예: 'so')를 한국어 글로스 부착 + 재임베딩해 target source('so_ko')로 적재.

    Posts.xml 재파싱 없이 이미 큐레이션된 tb_knowledge 행을 변환한다(저렴).
      1) source 고유 제목을 A.X로 EN→KO 번역(캐시 파일로 재실행 안전).
      2) 각 청크 content에 글로스 프리픽스 → 재임베딩(:8002) → source=target UPSERT.
      3) --resume-glossify면 target에 이미 있는 id는 건너뛴다(중단 후 이어받기).
    섀도 소스라 무중단·롤백(DROP source=target) 안전. 활성화는 tenants 스왑(별도 승인).
    keyset 페이지네이션(id > last)으로 스트리밍해 255k를 인메모리에 올리지 않는다.
    """
    import asyncpg

    src, tgt = args.glossify_source, args.target_source
    pool = await asyncpg.create_pool(args.pg, min_size=1, max_size=3)
    store = KnowledgeStore(pg_pool=pool)
    await store.ensure_schema()
    try:
        # 1) 고유 제목 수집 + 번역(캐시로 이어받기)
        title_rows = await pool.fetch(
            "SELECT DISTINCT title FROM tb_knowledge WHERE source=$1", src
        )
        titles = [r["title"] for r in title_rows]
        cache = _load_gloss_cache(args.gloss_cache)
        todo = [t for t in titles if t not in cache]
        logger.info("고유 제목 %d개(미번역 %d) — A.X EN→KO 번역", len(titles), len(todo))
        # 500개 청크마다 번역·캐시 저장(중단 시 진행분 보존)
        for i in range(0, len(todo), 500):
            chunk = todo[i : i + 500]
            new = await translate_titles(
                args.translate_url, args.translate_model, chunk,
                batch_size=args.translate_batch,
            )
            cache.update(new)
            _save_gloss_cache(args.gloss_cache, cache)
            logger.info("번역 진행: %d/%d 제목", min(i + 500, len(todo)), len(todo))
        if args.translate_only:
            print(f"translated_titles={len(cache)} (glossify 적재는 --translate-only 없이 재실행)")
            return 0

        # 2) 이어받기: target에 이미 적재된 id
        done_ids: set[str] = set()
        if args.resume_glossify:
            for r in await pool.fetch("SELECT id FROM tb_knowledge WHERE source=$1", tgt):
                done_ids.add(r["id"])
            logger.info("이어받기: target 기존 %d행 건너뜀", len(done_ids))

        # 3) keyset 페이지네이션으로 스트리밍 변환 → (재임베딩 or 임베딩 복사) → UPSERT
        #    --reuse-embedding: 기존 source 임베딩을 그대로 복사(재임베딩 0). 문서측 번역의
        #    저렴경로 — 리랭커는 글로스 본문을 보고 통과하고, 벡터 fetch는 fetch_k 상향으로
        #    보완한다(재임베딩 18h 회피). 실측 근거는 progress.md 2026-07-19 참조.
        reuse = args.reuse_embedding
        cols = "id, title, section, content, chunk_index, total_chunks, tags, metadata"
        if reuse:
            cols += ", embedding"
        total = await pool.fetchval(
            "SELECT count(*) FROM tb_knowledge WHERE source=$1", src
        )
        stored = skipped = 0
        last_id = ""
        while True:
            rows = await pool.fetch(
                f"SELECT {cols} FROM tb_knowledge WHERE source=$1 AND id > $2 "  # noqa: S608 — cols=화이트리스트 고정
                "ORDER BY id LIMIT $3",
                src, last_id, 500,
            )
            if not rows:
                break
            last_id = rows[-1]["id"]
            batch: list[KnowledgeEntry] = []
            for r in rows:
                emb = None
                if reuse and r["embedding"] is not None:
                    emb = tuple(_parse_vector(r["embedding"]))
                entry = KnowledgeEntry(
                    source=tgt,
                    title=r["title"],
                    content=apply_gloss(r["content"], cache.get(r["title"], "")),
                    section=r["section"],
                    chunk_index=r["chunk_index"],
                    total_chunks=r["total_chunks"],
                    tags=tuple(r["tags"] or ()),
                    embedding=emb,
                    metadata=_as_dict(r["metadata"]),
                )
                if entry.id in done_ids:
                    skipped += 1
                    continue
                batch.append(entry)
            if reuse:
                # 임베딩 복사 경로 — 임베딩 서버 호출 없이 바로 UPSERT.
                for entry in batch:
                    await store.add(entry)
                    stored += 1
            else:
                for j in range(0, len(batch), args.batch_size):
                    sub = batch[j : j + args.batch_size]
                    stored += await _embed_store(store, args.embed_url, sub)
            logger.info("글로시파이: 적재 %d(+skip %d) / 총 %d", stored, skipped, total)
    finally:
        await pool.close()
    logger.info("완료: %d청크 적재(source=%s, skip %d)", stored, tgt, skipped)
    print(f"stored={stored} source={tgt} skipped={skipped}")
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
    # 문서측 번역(B안) — 기존 source를 EN→KO 글로스 부착·재임베딩해 target source로 적재.
    parser.add_argument("--glossify-source", help="변환할 기존 source(예: so)")
    parser.add_argument("--target-source", default="so_ko", help="글로스 적재 target source")
    parser.add_argument("--translate-url", help="A.X 번역 서버(OpenAI 호환, 예: http://127.0.0.1:18001)")
    parser.add_argument("--translate-model", default="ax-4.0", help="번역 모델 served-name")
    parser.add_argument("--translate-batch", type=int, default=20, help="1요청당 제목 개수")
    parser.add_argument("--gloss-cache", default="gloss_cache.json", help="EN→KO 캐시(이어받기)")
    parser.add_argument("--translate-only", action="store_true", help="번역·캐시만(적재 안 함)")
    parser.add_argument("--resume-glossify", action="store_true", help="target 기존 id 건너뛰기")
    parser.add_argument("--reuse-embedding", action="store_true",
                        help="기존 source 임베딩 복사(재임베딩 0, 저렴경로). --embed-url 불요")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.build_index:
        if not args.pg:
            parser.error("--build-index 에는 --pg 가 필요합니다")
        return asyncio.run(run_build_index(args))

    if args.glossify_source:
        if not args.pg or not args.translate_url:
            parser.error("--glossify-source 에는 --pg 와 --translate-url 이 필요합니다")
        if not args.translate_only and not args.reuse_embedding and not args.embed_url:
            parser.error("글로스 적재엔 --embed-url 필요(번역만=--translate-only, 복사=--reuse)")
        return asyncio.run(run_glossify(args))

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
