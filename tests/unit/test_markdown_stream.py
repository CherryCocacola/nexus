# 스트리밍 마크다운 렌더러의 블록 확정 규칙을 검증한다.
"""
`cli/markdown_stream.py` 단위 테스트.

핵심 관심사는 셋이다.
  1) 빈 줄을 만나야 블록이 확정된다(마크다운 블록 경계).
  2) ``` 코드펜스 안의 빈 줄은 경계가 아니다(코드가 쪼개지면 안 된다).
  3) 스트림이 끝나면(flush) 남은 것이 반드시 나온다 — 안 그러면 마지막 문단이 사라진다.
"""

from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text

from cli.markdown_stream import MarkdownStreamRenderer


def _texts(blocks) -> list[str]:
    """확정된 블록에서 원본 마크다운 문자열만 뽑아낸다."""
    return [b.markup if isinstance(b, Markdown) else b.plain for b in blocks]


def test_feed_without_blank_line_holds_block() -> None:
    """빈 줄이 없으면 아직 확정하지 않는다 — 뒤에 더 붙을 수 있기 때문."""
    r = MarkdownStreamRenderer()
    assert r.feed("## 제목\n") == []
    assert r.pending is True


def test_blank_line_finalizes_block() -> None:
    """빈 줄이 오면 그 앞까지가 한 블록으로 확정된다."""
    r = MarkdownStreamRenderer()
    r.feed("## 제목\n본문 첫 줄\n")
    blocks = r.feed("\n다음 문단")
    assert _texts(blocks) == ["## 제목\n본문 첫 줄"]


def test_code_fence_blank_line_is_not_a_boundary() -> None:
    """코드블록 안의 빈 줄로 쪼개면 코드가 깨진다 — 닫는 펜스까지 한 블록이어야 한다."""
    r = MarkdownStreamRenderer()
    blocks = r.feed("```python\na = 1\n\nb = 2\n```\n")
    assert _texts(blocks) == ["```python\na = 1\n\nb = 2\n```"]


def test_flush_emits_remaining_tail() -> None:
    """★스트림이 빈 줄로 끝나지 않아도 마지막 문단이 반드시 출력돼야 한다."""
    r = MarkdownStreamRenderer()
    assert r.feed("마지막 문단인데 빈 줄이 없다") == []
    assert _texts(r.flush()) == ["마지막 문단인데 빈 줄이 없다"]
    assert r.pending is False


def test_long_run_without_blank_line_is_bounded() -> None:
    """빈 줄 없이 길게 이어져도 일정 줄 수에서 끊어 내보낸다(체감 지연 상한)."""
    r = MarkdownStreamRenderer()
    out: list = []
    for i in range(30):
        out += r.feed(f"- 항목 {i}\n")
    assert out, "상한에 걸렸는데도 아무것도 확정되지 않았다"


def test_disabled_returns_literal_text() -> None:
    """토글을 끄면 렌더하지 않고 원문 그대로 — 단 markup 오해석은 막아야 한다(B-4)."""
    r = MarkdownStreamRenderer(enabled=False)
    blocks = r.feed("대괄호 [INFO] 와 list[int] 가 그대로 남아야 한다")
    assert len(blocks) == 1
    assert isinstance(blocks[0], Text)
    assert "[INFO]" in blocks[0].plain


def test_bracket_content_survives_rendering() -> None:
    """마크다운 렌더 경로에서도 대괄호가 콘솔 markup 으로 먹히지 않아야 한다."""
    r = MarkdownStreamRenderer()
    r.feed("로그에 [INFO] 가 있고 타입은 list[int] 다\n")
    blocks = r.flush()
    assert "[INFO]" in _texts(blocks)[0]
    assert "list[int]" in _texts(blocks)[0]


def test_reset_discards_buffer() -> None:
    """턴이 취소되면 남은 찌꺼기가 다음 턴으로 새지 않아야 한다."""
    r = MarkdownStreamRenderer()
    r.feed("버려질 내용")
    r.reset()
    assert r.pending is False
    assert r.flush() == []
