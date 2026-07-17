# scripts/coding_corpus.py 코드 인지 청커의 불변식 검증 단위테스트.
"""
코드 인지 청커(chunk_code_aware)의 불변식 (A)~(E)를 검증한다.

핵심은 기존 split_into_chunks의 M2 결함(코드 파괴·앞부분 폐기)이 재발하지
않음을 회귀 테스트로 못 박는 것이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/를 import 경로에 넣는다(prepare_kowiki.py와 동일한 단독 실행 대비 패턴).
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from coding_corpus import chunk_code_aware  # noqa: E402


def _content_lines(text: str) -> set[str]:
    """펜스(```)·빈 줄을 제외한 실제 내용 라인 집합(무손실 비교용)."""
    return {
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("```")
    }


# ── (기본) 빈 입력 ────────────────────────────────────────
def test_chunk_empty_input_returns_empty_list():
    assert chunk_code_aware("") == []
    assert chunk_code_aware("   \n  \n") == []


# ── (A) 코드 블록 원자성 — 한도 이하면 안 쪼갠다 ──────────
def test_chunk_short_code_block_kept_intact():
    text = "리스트를 뒤집는 방법입니다.\n\n```python\nx = [1, 2, 3]\nx.reverse()\n```\n\n끝."
    chunks = chunk_code_aware(text, max_chars=1500)
    joined = "\n".join(chunks)
    # 코드 블록이 통째로 한 청크 안에 온전히 들어 있어야 한다.
    assert "```python\nx = [1, 2, 3]\nx.reverse()\n```" in joined
    # 짧은 Q&A는 한 청크로 유지(불변식 D).
    assert len(chunks) == 1


# ── (B)(C) 무손실 + 앞부분 보존 — M2 버그 정면 회귀 ───────
def test_chunk_oversized_code_preserves_head_no_data_loss():
    # import·시그니처가 맨 앞에 오는 긴 코드 블록(문장부호 없음 → 기존 청커가
    # 하나의 '문장'으로 보고 뒤 max_chars만 남기며 앞을 버리던 케이스).
    body_lines = [f"    step_{i} = compute({i})" for i in range(60)]
    code = (
        "```python\nimport os\nimport sys\n\ndef pipeline():\n"
        + "\n".join(body_lines)
        + "\n    return step_0\n```"
    )
    text = "다음은 긴 파이프라인 예시입니다.\n\n" + code
    chunks = chunk_code_aware(text, max_chars=300)

    all_lines = _content_lines("\n".join(chunks))
    # 맨 앞 import/시그니처가 반드시 살아 있어야 한다(앞부분 폐기 금지).
    assert "import os" in all_lines
    assert "import sys" in all_lines
    assert "def pipeline():" in all_lines
    # 원문의 모든 내용 라인이 어느 청크엔가 빠짐없이 존재(무손실).
    assert _content_lines(code).issubset(all_lines)


# ── (C) 긴 코드는 라인 경계로 나뉘고 각 조각이 유효한 펜스 ──
def test_chunk_oversized_code_split_at_line_boundaries_and_refenced():
    code_lines = [f"const value_{i} = {i};" for i in range(40)]
    code = "```javascript\n" + "\n".join(code_lines) + "\n```"
    chunks = chunk_code_aware(code, max_chars=200)
    assert len(chunks) > 1  # 실제로 분할됐다
    for ch in chunks:
        # 각 코드 조각은 여는/닫는 펜스를 갖춘 유효 블록이어야 한다.
        assert ch.startswith("```")
        assert ch.rstrip().endswith("```")
    # 어떤 원본 라인도 글자 중간에서 잘리지 않았다(모든 라인이 온전히 존재).
    all_lines = _content_lines("\n".join(chunks))
    assert _content_lines(code).issubset(all_lines)


# ── (E) 모든 청크가 max_chars 이하 ───────────────────────
def test_chunk_all_within_max_chars():
    body_lines = [f"    line_{i} = do_something_useful({i})" for i in range(80)]
    code = "```python\n" + "\n".join(body_lines) + "\n```"
    text = "설명 " * 200 + "\n\n" + code + "\n\n마무리 설명 " * 200
    max_chars = 400
    chunks = chunk_code_aware(text, max_chars=max_chars)
    for ch in chunks:
        assert len(ch) <= max_chars, f"청크가 한도 초과: {len(ch)} > {max_chars}"


# ── (D) 설명문 + 인접 짧은 코드는 한 청크에 동거 ─────────
def test_chunk_prose_and_adjacent_code_paired():
    text = "이 함수는 합을 구합니다.\n\n```python\ndef add(a, b):\n    return a + b\n```"
    chunks = chunk_code_aware(text, max_chars=1500)
    assert len(chunks) == 1
    assert "이 함수는 합을 구합니다." in chunks[0]
    assert "def add(a, b):" in chunks[0]


# ── (B) 산문도 앞부분 보존(하드분할이 앞→뒤) ─────────────
def test_chunk_oversized_prose_preserves_head():
    # 문장부호 없이 이어지는 아주 긴 산문(하드분할 대상).
    head = "AAAA_시작표지"
    tail = "ZZZZ_끝표지"
    long_prose = head + "_" + ("가나다라마바사" * 200) + "_" + tail
    chunks = chunk_code_aware(long_prose, max_chars=300)
    joined = "".join(chunks)
    assert head in joined  # 앞부분이 남아 있어야 한다(뒤만 남기던 버그 방지)
    assert tail in joined  # 뒷부분도 남아 있어야 한다(무손실)


# ── 전체 무손실 라운드트립 ───────────────────────────────
def test_chunk_roundtrip_completeness_mixed():
    text = (
        "먼저 슬라이싱을 씁니다.\n\n"
        "```python\nx = [1,2,3]\ny = x[::-1]\n```\n\n"
        "다음으로 reversed()를 씁니다.\n\n"
        "```python\ny = list(reversed(x))\n```\n\n"
        "마지막으로 in-place reverse()입니다.\n\n"
        "```python\nx.reverse()\n```\n\n"
        "상황에 맞게 고르세요."
    )
    chunks = chunk_code_aware(text, max_chars=120)
    all_lines = _content_lines("\n".join(chunks))
    assert _content_lines(text).issubset(all_lines)
