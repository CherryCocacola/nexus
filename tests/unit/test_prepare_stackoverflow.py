# scripts/prepare_stackoverflow.py SO 덤프 파싱·페어링·큐레이션·PK 유일성 검증.
"""
합성 Posts.xml fixture로 SO 적재 준비 파이프라인의 순수 로직을 검증한다.

특히 FABLE5 CRITICAL 수정(C1/N1)을 회귀로 못 박는다:
  - 제목이 같은 서로 다른 질문이 같은 PK로 덮어써지지 않는다.
  - 한 질문에 답변이 여럿이어도 단일 문서로 결합돼 PK 충돌이 없다.
임베딩·DB 쓰기는 실서버 대상이라 여기서 다루지 않는다.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from prepare_stackoverflow import (  # noqa: E402
    build_combined_document,
    build_entries,
    load_posts,
    parse_tags,
)

# 합성 Posts.xml — 속성 기반 평면 XML(실제 SO 덤프 구조). Body의 HTML은 XML
# 엔티티로 이스케이프돼 있고, ET가 파싱 시 <p> 등으로 복원한다.
_FIXTURE = """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" AcceptedAnswerId="3" Score="42"
       Title="How to reverse a list in Python?"
       Body="&lt;p&gt;I want to reverse a list.&lt;/p&gt;" Tags="&lt;python&gt;&lt;list&gt;" />
  <row Id="2" PostTypeId="2" ParentId="1" Score="10"
       Body="&lt;pre&gt;&lt;code&gt;y = x[::-1]&lt;/code&gt;&lt;/pre&gt;" />
  <row Id="3" PostTypeId="2" ParentId="1" Score="55"
       Body="&lt;pre&gt;&lt;code&gt;y = list(reversed(x))&lt;/code&gt;&lt;/pre&gt;" />
  <row Id="4" PostTypeId="1" Score="5"
       Title="How to reverse a list in Python?"
       Body="&lt;p&gt;Different question, same title.&lt;/p&gt;" Tags="&lt;python&gt;" />
  <row Id="5" PostTypeId="2" ParentId="4" Score="8"
       Body="&lt;p&gt;Use the reverse method.&lt;/p&gt;" />
  <row Id="6" PostTypeId="1" Score="1" Title="Weak question"
       Body="&lt;p&gt;No good answers.&lt;/p&gt;" Tags="&lt;java&gt;" />
  <row Id="7" PostTypeId="2" ParentId="6" Score="0" Body="&lt;p&gt;weak&lt;/p&gt;" />
  <row Id="99" PostTypeId="3" Body="tag wiki - 무시돼야 함" />
</posts>"""


def _load():
    return load_posts(io.BytesIO(_FIXTURE.encode("utf-8")))


# ── parse_tags ───────────────────────────────────────────
def test_parse_tags_angle_and_pipe_formats():
    assert parse_tags("<python><list>") == ["python", "list"]
    assert parse_tags("|python|list|") == ["python", "list"]
    assert parse_tags("") == []


# ── 파싱·버킷팅 ──────────────────────────────────────────
def test_load_posts_buckets_questions_and_answers():
    questions, answers = _load()
    # 질문 3개(1,4,6), PostTypeId=3(99)은 무시.
    assert set(questions) == {1, 4, 6}
    # 답변은 부모 질문별로 묶인다.
    assert {a["id"] for a in answers[1]} == {2, 3}
    assert {a["id"] for a in answers[4]} == {5}


# ── 결합 문서: 채택 우선 + 고득점 포함 ──────────────────
def test_build_combined_document_orders_accepted_first():
    questions, answers = _load()
    doc, used = build_combined_document(questions[1], answers[1], min_answer_score=5)
    # 채택답변(3)이 비채택답변(2)보다 앞에 온다.
    assert used == [3, 2]
    assert doc.index("채택된 답변") < doc.index("답변 (추천 10)")
    # 코드가 펜스로 보존된다.
    assert "y = list(reversed(x))" in doc
    assert "y = x[::-1]" in doc


# ── 큐레이션: 자격 답변 없으면 제외 ─────────────────────
def test_curation_excludes_question_without_qualifying_answer():
    questions, answers = _load()
    entries = build_entries(questions, answers, min_answer_score=5)
    kept_sections = {e.section for e in entries}
    # q6은 답변 점수 0(<5)·채택 없음 → 제외. q1·q4만 남는다.
    assert kept_sections == {"q1", "q4"}


# ── C1/N1: 제목 같은 다른 질문이 PK 충돌하지 않는다 ─────
def test_pk_unique_across_same_title_different_questions():
    questions, answers = _load()
    entries = build_entries(questions, answers, min_answer_score=5)
    # q1과 q4는 제목이 완전히 같지만 서로 다른 질문이다.
    q1_ids = {e.id for e in entries if e.section == "q1"}
    q4_ids = {e.id for e in entries if e.section == "q4"}
    assert q1_ids and q4_ids
    # 제목이 같아도 section(q{id})이 달라 PK가 겹치지 않는다(무언 덮어쓰기 방지).
    assert q1_ids.isdisjoint(q4_ids)


# ── C1/N1: 한 질문의 복수 답변이 단일 문서라 충돌 없음 ──
def test_pk_unique_within_multi_answer_question():
    questions, answers = _load()
    entries = build_entries(questions, answers, min_answer_score=5)
    q1_entries = [e for e in entries if e.section == "q1"]
    ids = [e.id for e in q1_entries]
    # 답변이 2개(2,3)여도 하나의 결합 문서 → 청크 id가 서로 유일하다.
    assert len(ids) == len(set(ids))


# ── provenance metadata ─────────────────────────────────
def test_provenance_metadata_present():
    questions, answers = _load()
    entries = build_entries(questions, answers, min_answer_score=5)
    e = next(e for e in entries if e.section == "q1")
    assert e.source == "so"
    assert e.metadata["question_id"] == 1
    assert e.metadata["url"] == "https://stackoverflow.com/q/1"
    assert e.metadata["license"] == "CC BY-SA 4.0"
    assert e.metadata["answer_ids"] == [3, 2]
    assert set(e.tags) == {"python", "list"}


# ── require_code 필터 ───────────────────────────────────
def test_require_code_filters_codeless_questions():
    questions, answers = _load()
    entries = build_entries(questions, answers, min_answer_score=5, require_code=True)
    kept = {e.section for e in entries}
    # q1만 코드 블록 보유(q4는 산문 답변뿐) → q1만 남는다.
    assert kept == {"q1"}
