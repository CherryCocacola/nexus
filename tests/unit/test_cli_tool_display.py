# 도구 표시 축약(C1)·/verbose 토글(D8)·슬래시 자동완성(D9) 검증.
"""
대화 흐름을 해치지 않는 도구 표시와, 필요할 때 전문을 보는 토글을 고정한다
(2026-08-05).

  - 기본: `⏺ Read(config.yaml)` 한 줄 + `⎿ 요약 (N줄)`
  - /verbose: 종전처럼 입력 JSON·결과 전문 패널
  - 에러 결과는 축약 모드에서도 전문을 보여 준다(원인을 봐야 대응 가능)
  - 자동완성은 `/`로 시작할 때만, 명령 목록은 REPL에서 동적으로 가져온다
"""

from __future__ import annotations

import io

from rich.console import Console

from cli.formatters import OutputFormatter, summarize_tool_input, summarize_tool_output
from cli.repl import NovaCompleter


def _render(renderable) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=200).print(renderable)
    return buf.getvalue()


class TestSummarizers:
    def test_path_arguments_shortened_to_filename(self) -> None:
        """경로 인자는 파일명만 남겨 한 줄에 들어가게 한다."""
        out = summarize_tool_input("Read", {"file_path": "/very/long/path/to/config.yaml"})
        assert out == "config.yaml"

    def test_bash_uses_command(self) -> None:
        """Bash는 명령이 핵심 인자다."""
        assert summarize_tool_input("Bash", {"command": "git status"}) == "git status"

    def test_multiedit_summarized_by_count(self) -> None:
        """MultiEdit은 편집 개수로 요약한다."""
        out = summarize_tool_input("MultiEdit", {"edits": [{"a": 1}, {"b": 2}]})
        assert out == "2건"

    def test_unknown_tool_falls_back_to_first_scalar(self) -> None:
        """분류표에 없는 도구는 첫 스칼라 값으로 폴백한다."""
        assert summarize_tool_input("SomeTool", {"query": "hello"}) == "hello"

    def test_long_value_is_truncated(self) -> None:
        """긴 값은 말줄임으로 한 줄을 유지한다."""
        out = summarize_tool_input("Bash", {"command": "echo " + "x" * 200})
        assert len(out) <= 61 and out.endswith("…")

    def test_empty_input_returns_empty(self) -> None:
        """인자가 없으면 빈 요약(괄호를 붙이지 않는다)."""
        assert summarize_tool_input("Read", {}) == ""

    def test_output_summary_includes_line_count(self) -> None:
        """결과 요약은 첫 내용 줄과 전체 줄 수를 함께 알린다."""
        out = summarize_tool_output("first line\nsecond\nthird")
        assert "first line" in out and "3줄" in out

    def test_empty_output_summary(self) -> None:
        """빈 결과도 안전하게 표기한다."""
        assert summarize_tool_output("") == "(빈 결과)"


class TestToolDisplayModes:
    def test_compact_tool_use_is_one_line(self) -> None:
        """기본 모드에서 도구 호출은 ⏺ 한 줄로 표시된다."""
        fmt = OutputFormatter()
        out = _render(fmt.format_tool_use("Read", {"file_path": "/tmp/config.yaml"}))
        assert "⏺" in out and "Read" in out and "config.yaml" in out
        assert "{" not in out  # JSON 전문이 아니다

    def test_verbose_tool_use_shows_json(self) -> None:
        """/verbose를 켜면 입력 인자 JSON 패널을 보여 준다."""
        fmt = OutputFormatter()
        fmt.verbose = True
        out = _render(fmt.format_tool_use("Read", {"file_path": "/tmp/config.yaml"}))
        assert "file_path" in out and "{" in out

    def test_compact_result_is_folded(self) -> None:
        """기본 모드에서 결과는 ⎿ 한 줄 요약으로 접힌다."""
        fmt = OutputFormatter()
        out = _render(fmt.format_tool_result("line1\nline2\nline3"))
        assert "⎿" in out and "3줄" in out
        assert "line3" not in out  # 전문은 접혀 있다

    def test_verbose_result_shows_full_panel(self) -> None:
        """/verbose를 켜면 결과 전문 패널을 보여 준다."""
        fmt = OutputFormatter()
        fmt.verbose = True
        out = _render(fmt.format_tool_result("line1\nline2\nline3"))
        assert "line3" in out

    def test_error_always_shown_in_full(self) -> None:
        """에러는 축약 모드에서도 전문을 보여 준다(원인 파악이 우선)."""
        fmt = OutputFormatter()
        out = _render(fmt.format_tool_result("권한 거부: 경로 정책 위반", is_error=True))
        assert "권한 거부" in out and "경로 정책 위반" in out

    def test_verbose_property_toggles(self) -> None:
        """verbose 프로퍼티가 실제로 토글된다(/verbose 명령이 쓰는 경로)."""
        fmt = OutputFormatter()
        assert fmt.verbose is False
        fmt.verbose = True
        assert fmt.verbose is True


class TestSlashCompleter:
    def _complete(self, text: str, commands: list[str]) -> list[str]:
        from prompt_toolkit.document import Document

        completer = NovaCompleter(lambda: commands)
        return [c.text for c in completer.get_completions(Document(text, len(text)), None)]

    def test_suggests_matching_commands(self) -> None:
        """접두어와 일치하는 슬래시 명령만 제안한다."""
        got = self._complete("/mo", ["/mode", "/model", "/help"])
        assert set(got) == {"/mode", "/model"}

    def test_no_suggestion_for_plain_text(self) -> None:
        """일반 대화 입력에는 제안하지 않는다(입력을 방해하지 않도록)."""
        assert self._complete("안녕하세요", ["/mode", "/help"]) == []

    def test_no_suggestion_after_first_token(self) -> None:
        """명령 인자를 입력하는 중에는 제안하지 않는다."""
        assert self._complete("/mode acc", ["/mode", "/model"]) == []

    def test_command_list_is_dynamic(self) -> None:
        """명령 목록을 콜러블로 받아, 나중에 추가된 명령도 자동으로 제안된다."""
        commands = ["/help"]
        completer = NovaCompleter(lambda: commands)
        commands.append("/verbose")
        from prompt_toolkit.document import Document

        got = [c.text for c in completer.get_completions(Document("/v", 2), None)]
        assert got == ["/verbose"]
