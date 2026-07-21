# prepare_pytorch_kr.py 의 rst/py 파싱·포럼 결합·엔트리 생성 검증(합성 입력, 오프라인).
"""
PyTorch 한국어 자료 적재 스크립트의 순수 로직을 검증한다.

네트워크·DB 없이 합성 입력으로 테스트한다(prepare_okky 대칭). 검증 대상은
번역 미완 게이트(hangul_ratio), rst/sphinx-gallery 파싱, Discourse cooked HTML
결합, 그리고 라이선스가 다른 두 source의 분리다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from prepare_pytorch_kr import (  # noqa: E402
    _FORUM_LICENSE,
    _TUTORIAL_LICENSE,
    build_forum_entries,
    build_topic_document,
    build_tutorial_entries,
    hangul_ratio,
    normalize_tags,
    parse_gallery_py,
    parse_rst,
)

# ── 합성 입력 ─────────────────────────────────────────────────────────────

_RST_KO = '''\
텐서(Tensor) 다루기
===================

파이토치에서 텐서는 배열과 비슷한 자료구조입니다. 아래 예시를 봅시다.

.. code-block:: python

    import torch
    x = torch.rand(3, 4)
    print(x.shape)

이렇게 하면 무작위 텐서를 만들 수 있습니다.

.. note::

    이 문서는 예시입니다.
'''

_GALLERY_PY_KO = '''\
"""
신경망 학습하기
===============

이 튜토리얼에서는 간단한 신경망을 학습하는 방법을 다룹니다.
"""

import torch
import torch.nn as nn

######################################################################
# 모델 정의하기
# --------------
# 아래와 같이 모듈을 상속받아 모델을 정의합니다.

model = nn.Linear(10, 2)
'''

# 번역이 안 된 영문 원문(제외 대상).
_RST_EN = '''\
Custom C++ Operators
====================

This tutorial explains how to author a custom operator in C++ and register it
with the PyTorch dispatcher so that it can be used from Python code directly.
'''


def _topic(topic_id: int, title: str, cooked_list: list[str]) -> dict:
    """Discourse 토픽 JSON의 최소 형태를 만든다(post_stream.posts[].cooked).

    tags는 **실제 /t/{id}.json 응답과 같이 dict 목록**으로 둔다. 처음에 이걸
    문자열 목록으로 만들어 뒀다가 실적재에서 DataError(text[] 바인딩 실패)를
    놓쳤다 — 픽스처는 실제 API 모양을 따라야 한다.
    """
    return {
        "id": topic_id,
        "title": title,
        "slug": "test-topic",
        "tags": [{"id": 432, "name": "paper", "slug": "paper"}],
        "post_stream": {
            "posts": [{"id": 1000 + i, "cooked": c} for i, c in enumerate(cooked_list)]
        },
    }


# ── hangul_ratio (번역 미완 게이트) ───────────────────────────────────────


def test_hangul_ratio_korean_text_high():
    assert hangul_ratio("파이토치는 딥러닝 프레임워크입니다") > 0.9


def test_hangul_ratio_english_text_zero():
    assert hangul_ratio("This is an English tutorial about tensors.") == 0.0


def test_hangul_ratio_code_heavy_korean_still_counts():
    """코드·기호는 분모에서 빠지므로 코드가 많은 한국어 문서도 임계를 넘어야 한다."""
    text = "텐서를 만듭니다.\n\n```python\nx = torch.rand(3, 4)\nprint(x.shape)\n```"
    assert hangul_ratio(text) >= 0.15


def test_hangul_ratio_empty_returns_zero():
    assert hangul_ratio("") == 0.0
    assert hangul_ratio("```\n{}\n```") == 0.0


# ── rst 파싱 ──────────────────────────────────────────────────────────────


def test_parse_rst_extracts_title_from_underline():
    title, _ = parse_rst(_RST_KO)
    assert title == "텐서(Tensor) 다루기"


def test_parse_rst_converts_code_block_to_fence():
    _, body = parse_rst(_RST_KO)
    assert "```python" in body
    assert "import torch" in body
    # 코드 블록의 들여쓰기가 제거되어 실행 가능한 형태로 남아야 한다.
    assert "\nimport torch" in body


def test_parse_rst_drops_directive_lines_but_keeps_prose():
    _, body = parse_rst(_RST_KO)
    assert ".. note::" not in body
    assert "이 문서는 예시입니다." in body


def test_parse_rst_closes_unterminated_code_fence():
    """문서가 코드 블록으로 끝나도 펜스가 닫혀야 한다(짝이 안 맞으면 청킹이 깨진다)."""
    _, body = parse_rst("제목\n====\n\n.. code-block:: python\n\n    x = 1\n")
    assert body.count("```") % 2 == 0


# ── sphinx-gallery .py 파싱 ───────────────────────────────────────────────


def test_parse_gallery_py_extracts_title_from_docstring():
    title, _ = parse_gallery_py(_GALLERY_PY_KO)
    assert title == "신경망 학습하기"


def test_parse_gallery_py_separates_prose_and_code():
    _, body = parse_gallery_py(_GALLERY_PY_KO)
    # 주석 블록은 산문으로(# 접두 제거), 코드는 펜스로.
    assert "모듈을 상속받아 모델을 정의합니다." in body
    assert "# 아래와 같이" not in body
    assert "```python" in body
    assert "model = nn.Linear(10, 2)" in body


def test_parse_gallery_py_fences_are_balanced():
    _, body = parse_gallery_py(_GALLERY_PY_KO)
    assert body.count("```") % 2 == 0


# ── 튜토리얼 엔트리 ───────────────────────────────────────────────────────


def _tutorial_doc(path: str, text: str) -> dict:
    title, body = (parse_gallery_py(text) if path.endswith(".py") else parse_rst(text))
    return {"path": path, "title": title, "body": body}


def test_build_tutorial_entries_skips_untranslated_english():
    """영문 원문 문서는 제외되어야 한다(교차언어 문제 재발 방지)."""
    docs = [_tutorial_doc("advanced_source/cpp_custom_ops.rst", _RST_EN)]
    assert build_tutorial_entries(docs) == []


def test_build_tutorial_entries_keeps_korean_doc():
    docs = [_tutorial_doc("beginner_source/tensor.rst", _RST_KO * 3)]
    entries = build_tutorial_entries(docs)
    assert entries
    assert all(e.source == "pytorch_kr" for e in entries)


def test_build_tutorial_entries_skips_short_doc():
    docs = [{"path": "a.rst", "title": "짧은 글", "body": "너무 짧은 한국어 문서."}]
    assert build_tutorial_entries(docs) == []


def test_build_tutorial_entries_metadata_has_bsd_license_and_url():
    docs = [_tutorial_doc("beginner_source/tensor.rst", _RST_KO * 3)]
    e = build_tutorial_entries(docs)[0]
    assert e.metadata["license"] == _TUTORIAL_LICENSE
    assert "BSD-3-Clause" in e.metadata["license"]
    assert e.metadata["url"].startswith("https://tutorials.pytorch.kr/")
    assert e.metadata["url"].endswith(".html")
    assert e.metadata["path"] == "beginner_source/tensor.rst"


def test_build_tutorial_entries_section_is_unique_per_document():
    """section=상대경로 → 문서마다 유일해야 PK 충돌 없이 UPSERT된다."""
    docs = [
        _tutorial_doc("beginner_source/a.rst", _RST_KO * 3),
        _tutorial_doc("beginner_source/b.rst", _RST_KO * 3),
    ]
    entries = build_tutorial_entries(docs)
    sections = {e.section for e in entries}
    assert sections == {"beginner_source/a.rst", "beginner_source/b.rst"}


def test_build_tutorial_entries_chunk_index_is_sequential():
    docs = [_tutorial_doc("beginner_source/tensor.rst", _RST_KO * 8)]
    entries = build_tutorial_entries(docs)
    assert [e.chunk_index for e in entries] == list(range(len(entries)))
    assert all(e.total_chunks == len(entries) for e in entries)


# ── 포럼 결합·엔트리 ──────────────────────────────────────────────────────


def test_build_topic_document_joins_posts_in_order():
    t = _topic(11, "AI 소식", ["<p>첫 번째 글입니다.</p>", "<p>두 번째 답글입니다.</p>"])
    doc, used = build_topic_document(t)
    assert used == [1000, 1001]
    assert doc.index("첫 번째") < doc.index("두 번째")


def test_build_topic_document_skips_empty_cooked():
    t = _topic(12, "빈 글", ["<p>본문입니다.</p>", ""])
    doc, used = build_topic_document(t)
    assert used == [1000]
    assert "본문입니다" in doc


def test_build_forum_entries_filters_link_only_short_posts():
    """링크만 있는 단발 공유글은 RAG 노이즈이므로 본문 길이로 걸러야 한다."""
    t = _topic(13, "링크 공유", ["<p>참고하세요</p>"])
    assert build_forum_entries([t]) == []


def test_build_forum_entries_keeps_substantial_post():
    body = "<p>" + ("파이토치 모델 학습에 대한 자세한 설명입니다. " * 20) + "</p>"
    entries = build_forum_entries([_topic(14, "학습 가이드", [body])])
    assert entries
    assert all(e.source == "pytorch_forum_kr" for e in entries)
    assert entries[0].title == "학습 가이드"


def test_build_forum_entries_metadata_marks_ugc_license():
    """상용 전환 시 source로 제거해야 하므로 UGC 라이선스가 명시되어야 한다."""
    body = "<p>" + ("긴 한국어 본문입니다. " * 30) + "</p>"
    e = build_forum_entries([_topic(15, "제목", [body])])[0]
    assert e.metadata["license"] == _FORUM_LICENSE
    assert "재사용 라이선스 없음" in e.metadata["license"]
    assert e.metadata["topic_id"] == 15
    assert e.metadata["url"] == "https://discuss.pytorch.kr/t/test-topic/15"
    assert e.section == "t15"


def test_normalize_tags_converts_discourse_dicts_to_strings():
    """회귀 방지: /t/{id}.json 의 tags는 dict라 그대로 넘기면 PG text[] 바인딩이 깨진다."""
    raw = [{"id": 432, "name": "paper", "slug": "paper"}, {"id": 9, "name": "llm"}]
    assert normalize_tags(raw) == ("paper", "llm")


def test_normalize_tags_accepts_plain_strings():
    """카테고리 목록 엔드포인트는 문자열 목록을 주므로 둘 다 받아야 한다."""
    assert normalize_tags(["paper", "llm"]) == ("paper", "llm")


def test_normalize_tags_handles_empty_and_malformed():
    assert normalize_tags(None) == ()
    assert normalize_tags([]) == ()
    assert normalize_tags([{"id": 1}, 42]) == ()  # name/slug 없음·비문자열은 버린다


def test_build_forum_entries_tags_are_all_strings():
    """엔트리의 tags는 반드시 문자열 튜플이어야 한다(적재 시 DataError 방지)."""
    body = "<p>" + ("긴 한국어 본문입니다. " * 30) + "</p>"
    e = build_forum_entries([_topic(17, "제목", [body])])[0]
    assert e.tags == ("paper",)
    assert all(isinstance(x, str) for x in e.tags)


def test_forum_and_tutorial_sources_are_separated():
    """라이선스가 다르므로 두 소스는 절대 같은 source 값을 쓰면 안 된다."""
    body = "<p>" + ("긴 한국어 본문입니다. " * 30) + "</p>"
    forum = build_forum_entries([_topic(16, "포럼 글", [body])])
    tutorial = build_tutorial_entries([_tutorial_doc("beginner_source/t.rst", _RST_KO * 3)])
    assert {e.source for e in forum}.isdisjoint({e.source for e in tutorial})
