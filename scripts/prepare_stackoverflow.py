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
import logging
import re
import sys

# 속성 기반 평면 XML 스트리밍 파싱. 로컬 신뢰 덤프만 파싱하므로 S405/S314는 noqa.
import xml.etree.ElementTree as ET  # noqa: S405 — 로컬 신뢰 SO 덤프만 파싱
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# 이 파일 단독 실행 시 core/coding_corpus를 import할 수 있게 경로를 잡는다.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (coding_corpus)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 루트 (core)

from coding_corpus import chunk_code_aware, html_to_markdown  # noqa: E402

from core.rag.knowledge_store import KnowledgeEntry  # noqa: E402

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
            yield {
                "id": pid,
                "type": post_type,
                "parent_id": _int_or_none(a.get("ParentId")),
                "accepted_answer_id": _int_or_none(a.get("AcceptedAnswerId")),
                "score": _int_or_none(a.get("Score")) or 0,
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


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점 — 현재는 --stats(규모 스캔)만 완전 배선. 적재/인덱스는 후속."""
    parser = argparse.ArgumentParser(description="Stack Overflow 덤프 → tb_knowledge 준비")
    parser.add_argument("--dump", help="Posts.xml 로컬 경로")
    parser.add_argument("--stats", action="store_true", help="적재 없이 규모만 집계")
    parser.add_argument("--min-score", type=int, default=5, help="비채택 답변 최소 점수")
    parser.add_argument(
        "--min-question-score", type=int, default=0, help="질문 최소 점수(품질 게이트)"
    )
    parser.add_argument("--require-code", action="store_true", help="코드 블록 있는 질문만")
    parser.add_argument("--limit", type=int, default=None, help="처리할 최대 row 수(표본 검증용)")
    parser.add_argument("--build-index", action="store_true", help="(후속) 벡터 인덱스 빌드")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.stats:
        if not args.dump:
            parser.error("--stats 에는 --dump 가 필요합니다")
        return run_stats(args)
    # 적재/인덱스 배선은 실덤프+LAN 인프라 준비 후(Phase 0b/1)에 추가한다.
    parser.error("현재 --stats 만 지원합니다(적재/인덱스는 실덤프 배선 후 활성화).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
