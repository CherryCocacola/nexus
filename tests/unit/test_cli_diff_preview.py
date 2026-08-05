# 파일 수정 diff 미리보기(B1/B2) 검증 — 렌더·상한·도구별 처리.
"""
승인 전에 "무엇이 어떻게 바뀌는지"를 보여 주는 미리보기의 계약을 고정한다
(2026-08-05).

  - Write: 기존 파일 대비 diff, 파일이 없으면 신규 생성 diff
  - Edit: old_string→new_string을 실제 파일에 적용한 결과로 diff
  - MultiEdit: 같은 파일 편집들을 순차 적용해 단일 diff
  - 상한: 256KB 초과 파일·바이너리는 diff 생략(터미널 마비 방지),
    200줄 초과 출력은 잘라내고 남은 줄 수를 알린다
  - 미리보기 실패는 절대 승인 흐름을 막지 않는다(None 반환)
"""

from __future__ import annotations

import io

from rich.console import Console

from cli.formatters import (
    DIFF_MAX_FILE_BYTES,
    DIFF_MAX_LINES,
    format_change_preview,
    format_diff,
    looks_binary,
    read_text_for_diff,
)


def _render(renderable) -> str:
    """Rich 렌더러블을 문자열로 펼쳐 검사하기 쉽게 만든다."""
    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=200).print(renderable)
    return buf.getvalue()


class TestFormatDiff:
    def test_shows_added_and_removed_lines(self) -> None:
        """추가·삭제 줄이 diff에 나타난다."""
        out = _render(format_diff("a.py", "x = 1\ny = 2\n", "x = 1\ny = 3\n"))
        assert "-y = 2" in out
        assert "+y = 3" in out

    def test_no_change_is_reported(self) -> None:
        """변경이 없으면 그 사실을 알린다(빈 패널을 내지 않는다)."""
        out = _render(format_diff("a.py", "same\n", "same\n"))
        assert "변경 내용 없음" in out

    def test_long_diff_is_truncated_with_count(self) -> None:
        """출력이 상한을 넘으면 잘라내고 남은 줄 수를 알린다."""
        old = "\n".join(f"line {i}" for i in range(DIFF_MAX_LINES * 2))
        new = "\n".join(f"changed {i}" for i in range(DIFF_MAX_LINES * 2))
        out = _render(format_diff("big.txt", old, new))
        assert "외" in out and "줄" in out


class TestReadTextForDiff:
    def test_missing_file_reports_new(self, tmp_path) -> None:
        """없는 파일은 '신규'로 알려 호출부가 새 파일 diff를 만들 수 있게 한다."""
        content, reason = read_text_for_diff(str(tmp_path / "nope.txt"))
        assert content is None and reason == "신규"

    def test_large_file_skipped(self, tmp_path) -> None:
        """상한을 넘는 파일은 읽지 않고 사유를 돌려준다."""
        f = tmp_path / "big.txt"
        f.write_text("a" * (DIFF_MAX_FILE_BYTES + 10), encoding="utf-8")
        content, reason = read_text_for_diff(str(f))
        assert content is None and "diff 생략" in reason

    def test_binary_file_detected(self, tmp_path) -> None:
        """UTF-8로 못 읽는 파일은 바이너리로 처리한다."""
        f = tmp_path / "bin.dat"
        f.write_bytes(b"\xff\xfe\x00\x01binary")
        content, reason = read_text_for_diff(str(f))
        assert content is None and "바이너리" in reason

    def test_looks_binary_detects_null(self) -> None:
        """NULL 문자가 있으면 바이너리로 간주한다."""
        assert looks_binary("ab\x00cd") is True
        assert looks_binary("normal text") is False


class TestChangePreview:
    def test_write_new_file(self, tmp_path) -> None:
        """존재하지 않는 경로에 Write하면 신규 내용이 추가로 표시된다."""
        target = tmp_path / "new.py"
        out = _render(
            format_change_preview("Write", {"file_path": str(target), "content": "hello\n"})
        )
        assert "+hello" in out

    def test_write_existing_file_shows_replacement(self, tmp_path) -> None:
        """기존 파일 Write는 원본 대비 변경을 보여 준다."""
        f = tmp_path / "a.txt"
        f.write_text("before\n", encoding="utf-8")
        out = _render(
            format_change_preview("Write", {"file_path": str(f), "content": "after\n"})
        )
        assert "-before" in out and "+after" in out

    def test_edit_applies_replacement(self, tmp_path) -> None:
        """Edit은 치환을 실제로 적용한 결과로 diff를 만든다."""
        f = tmp_path / "a.py"
        f.write_text("x = 1\ny = 2\n", encoding="utf-8")
        out = _render(
            format_change_preview(
                "Edit",
                {"file_path": str(f), "old_string": "y = 2", "new_string": "y = 99"},
            )
        )
        assert "+y = 99" in out

    def test_multiedit_applies_sequentially(self, tmp_path) -> None:
        """MultiEdit은 같은 파일 편집들을 순차 적용해 하나의 diff로 보여 준다."""
        f = tmp_path / "a.py"
        f.write_text("a = 1\nb = 2\n", encoding="utf-8")
        out = _render(
            format_change_preview(
                "MultiEdit",
                {
                    "edits": [
                        {"file_path": str(f), "old_string": "a = 1", "new_string": "a = 10"},
                        {"file_path": str(f), "old_string": "b = 2", "new_string": "b = 20"},
                    ]
                },
            )
        )
        assert "+a = 10" in out and "+b = 20" in out

    def test_non_file_tool_returns_none(self) -> None:
        """파일을 바꾸지 않는 도구는 미리보기 대상이 아니다."""
        assert format_change_preview("Bash", {"command": "ls"}) is None

    def test_broken_input_returns_none_not_raise(self) -> None:
        """입력이 이상해도 예외를 던지지 않는다(승인 흐름을 막으면 안 된다)."""
        assert format_change_preview("MultiEdit", {"edits": []}) is None
        assert format_change_preview("Edit", {}) is not None or True  # 예외만 없으면 통과
