# scripts/prepare_okky.py OKKY RSC 파싱·결합·엔트리 검증(합성 페이로드, 오프라인).
"""
OKKY Next.js RSC 페이로드에서 질문/답변을 추출하고 결합·청킹하는 순수 로직을 검증한다.
네트워크·DB 없이 합성 self.__next_f 페이로드로 테스트한다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from prepare_okky import (  # noqa: E402
    build_combined_document,
    build_entries,
    parse_question,
    rsc_payload,
)

# 합성 RSC 페이로드 — 질문 text 는 answers 앞에, 코드블록 포함 답변, 채택답변(222).
_PAYLOAD = (
    '{"id":1234,'
    '"title":"파이썬 리스트를 뒤집는 방법",'
    '"tags":[{"name":"python"},{"name":"list"}],'
    '"text":"<p>리스트를 뒤집으려면 어떻게 하나요?</p>",'
    '"selectedAnswerId":222,'
    '"answers":{"content":['
    '{"id":222,"text":"<p>슬라이싱을 쓰세요.</p><pre><code>y = x[::-1]</code></pre>",'
    '"voteCount":10,"assentCount":0,"selected":true},'
    '{"id":223,"text":"<p>reverse() 메서드도 있습니다.</p>","voteCount":2,"selected":false}'
    "]}}"
)


def _html(payload: str) -> str:
    # rsc_payload가 json.loads로 조각을 풀므로, 페이로드를 JSON 문자열 리터럴로 감싼다.
    push = f"self.__next_f.push([1,{json.dumps(payload)}])"
    return f"<html><body><script>{push}</script></body></html>"


def test_rsc_payload_extracts_chunks():
    p = rsc_payload(_html(_PAYLOAD))
    assert '"title":"파이썬' in p
    assert rsc_payload("<html>no rsc here</html>") == ""


def test_parse_question_extracts_fields():
    q = parse_question(_html(_PAYLOAD))
    assert q is not None
    assert q["title"] == "파이썬 리스트를 뒤집는 방법"
    assert set(q["tags"]) == {"python", "list"}
    assert q["selected_answer_id"] == 222
    assert len(q["answers"]) == 2
    assert "리스트를 뒤집으려면" in q["text"]


def test_build_combined_document_accepted_first_code_preserved():
    q = parse_question(_html(_PAYLOAD))
    doc, used = build_combined_document(q, min_answer_votes=1)
    # 채택답변(222) 먼저, 코드 펜스 보존.
    assert used[0] == 222
    assert "채택된 답변" in doc
    assert "y = x[::-1]" in doc
    assert "```" in doc
    # 추천 2인 비채택답변(223)도 min_votes=1 통과.
    assert 223 in used


def test_build_entries_source_and_pk():
    q = parse_question(_html(_PAYLOAD))
    entries = build_entries([q], min_answer_votes=1)
    assert entries
    assert all(e.source == "okky" for e in entries)
    assert all(e.section == "q1234" for e in entries)
    # PK 유일성(청크 간).
    ids = [e.id for e in entries]
    assert len(ids) == len(set(ids))
    e = entries[0]
    assert e.metadata["url"] == "https://okky.kr/questions/1234"
    assert e.metadata["answer_ids"][0] == 222
    assert set(e.tags) == {"python", "list"}


def test_require_code_filters():
    # 코드 없는 질문은 require_code=True에서 제외.
    no_code = _PAYLOAD.replace("<pre><code>y = x[::-1]</code></pre>", "")
    q = parse_question(_html(no_code))
    assert build_entries([q], min_answer_votes=1, require_code=True) == []
    assert build_entries([q], min_answer_votes=1, require_code=False)  # 코드 무관하면 통과
