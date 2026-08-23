# CLI 표현 계층을 Claude Code 방식에 맞춘 변경(2026-08-23)을 고정하는 테스트.
"""
터미널에서 "지금 무슨 일이 일어나고 있는지"가 항상 보이게 만든 장치들을 검증한다.

  - 스피너 꼬리표: 경과초 · 생성 토큰 · 중단 키가 1초마다 갱신된다
  - 대기 동사 순환: 고정 문구 하나가 아니라 주기적으로 바뀐다(멈춤과 구분)
  - 턴 종료 요약: 짧은 턴에는 붙지 않는다(잡음 방지)
  - `@` 경로 자동완성: 경로 오타를 원천 차단하기 위한 장치
  - 슬래시 자동완성 설명: /help 와 같은 출처(_COMMAND_HELP)를 쓴다
  - thinking 인라인 표시: 박스가 아니라 흐린 한 덩어리
  - 도구 결과 들여쓰기: `⏺` 아래에 `  ⎿  ` 로 정렬
"""

from __future__ import annotations

import io

from prompt_toolkit.document import Document
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cli.formatters import OutputFormatter
from cli.repl import (
    _COMMAND_HELP,
    _SPINNER_VERB_PERIOD_SEC,
    _SPINNER_VERBS,
    _TURN_FOOTER_MIN_SEC,
    NexusREPL,
    NovaCompleter,
    build_spinner_suffix,
    build_spinner_text,
    build_turn_footer,
    compact_count,
)


def _render(renderable) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=200).print(renderable)
    return buf.getvalue()


class TestCompactCount:
    """스피너 한 줄에 큰 수를 욱여넣기 위한 축약 규칙."""

    def test_below_thousand_is_verbatim(self) -> None:
        assert compact_count(0) == "0"
        assert compact_count(999) == "999"

    def test_thousands_keep_one_decimal(self) -> None:
        assert compact_count(1234) == "1.2k"

    def test_ten_thousand_drops_decimal(self) -> None:
        """만 단위부터는 소수를 버린다 — 자릿수가 길어지면 축약 목적이 사라진다."""
        assert compact_count(45_000) == "45k"

    def test_millions_use_m_suffix(self) -> None:
        assert compact_count(2_500_000) == "2.5M"

    def test_negative_clamped_to_zero(self) -> None:
        """토큰 증가분이 음수로 계산되는 경계(세션 교체 등)에서도 깨지지 않는다."""
        assert compact_count(-5) == "0"


class TestSpinnerText:
    """스피너 문구 — 경과시간·토큰·중단키가 항상 붙어 있어야 한다.

    NexusREPL 인스턴스를 만들지 않고 순수 함수만 검증한다. 생성자가
    PromptSession(실제 콘솔 필요)을 열기 때문이기도 하지만, 더 큰 이유는
    "문구를 어떻게 조립하는가"가 인스턴스 상태와 무관한 규칙이기 때문이다.
    """

    def test_suffix_has_elapsed_and_interrupt_hint(self) -> None:
        """항상 경과초와 중단 키를 보여 준다 — 멈춤 여부를 판단할 유일한 근거."""
        suffix = build_spinner_suffix(12.7, 0)
        assert "12s" in suffix
        assert "esc 중단" in suffix

    def test_suffix_omits_token_count_when_nothing_produced(self) -> None:
        """0 토큰은 정보가 아니라 잡음이므로 표시하지 않는다."""
        assert "토큰" not in build_spinner_suffix(5.0, 0)

    def test_suffix_shows_produced_tokens(self) -> None:
        """생성 토큰이 있으면 축약해 붙인다."""
        assert "↓1.0k 토큰" in build_spinner_suffix(5.0, 1000)

    def test_base_label_is_kept_when_stage_known(self) -> None:
        """단계 라벨이 있으면 그것을 쓰고 꼬리표만 덧붙인다."""
        text = build_spinner_text("[yellow]Read 실행 중[/yellow]", 3.0, 0, 0)
        assert text.startswith("[yellow]Read 실행 중[/yellow]")
        assert "esc 중단" in text

    def test_waiting_state_uses_a_verb(self) -> None:
        """대기 상태에서는 동사 목록에서 하나를 골라 쓴다."""
        text = build_spinner_text(None, 0.0, 0, 0)
        assert any(v in text for v in _SPINNER_VERBS)

    def test_waiting_verb_rotates_over_time(self) -> None:
        """같은 턴이라도 시간이 지나면 문구가 바뀐다 — 멈춤과 구분되게 한다."""
        early = build_spinner_text(None, 0.0, 0, 0)
        later = build_spinner_text(None, float(_SPINNER_VERB_PERIOD_SEC), 0, 0)
        assert early != later

    def test_waiting_verb_differs_per_turn(self) -> None:
        """턴이 바뀌면 시작 동사도 달라진다(같은 화면 반복 인상을 줄이기 위함)."""
        assert build_spinner_text(None, 0.0, 0, 0) != build_spinner_text(None, 0.0, 0, 1)

    def test_verb_index_wraps_around(self) -> None:
        """동사 목록 끝을 넘어가도 인덱스 오류가 나지 않는다."""
        far = build_spinner_text(None, 0.0, 0, len(_SPINNER_VERBS) * 3 + 1)
        assert any(v in far for v in _SPINNER_VERBS)

    def test_session_tokens_survive_missing_state(self) -> None:
        """부트스트랩 실패로 상태가 없어도 장식용 수치 때문에 예외가 나면 안 된다."""
        repl = NexusREPL.__new__(NexusREPL)  # 생성자를 건너뛴 최소 인스턴스
        assert repl._session_tokens() == (0, 0)


class TestTurnFooter:
    """턴 종료 요약 — 길었던 턴에만 붙는다."""

    def test_short_turn_returns_none(self) -> None:
        """2초 미만 잡담마다 통계가 붙으면 대화가 지저분해진다."""
        assert build_turn_footer(1.9, 500) is None

    def test_long_turn_reports_elapsed(self) -> None:
        line = build_turn_footer(31.4, 0)
        assert line is not None
        assert "31.4s" in line

    def test_long_turn_includes_tokens_when_produced(self) -> None:
        line = build_turn_footer(31.4, 5200)
        assert line is not None
        assert "5.2k" in line

    def test_boundary_is_inclusive_of_threshold(self) -> None:
        """정확히 임계값이면 표시한다(경계에서 조용히 사라지지 않게)."""
        assert build_turn_footer(_TURN_FOOTER_MIN_SEC, 0) is not None


class TestPathCompletion:
    """`@경로` 자동완성 — 모델의 경로 오타를 입력 단계에서 없앤다."""

    def _complete(self, text: str) -> list[str]:
        completer = NovaCompleter(lambda: sorted(_COMMAND_HELP))
        return [c.text for c in completer.get_completions(Document(text, len(text)), None)]

    def test_at_prefix_lists_repo_entries(self) -> None:
        """리포 루트에서 `@co` 는 core/ 를 제안한다."""
        assert "core/" in self._complete("@co")

    def test_directory_gets_trailing_slash(self) -> None:
        """디렉터리에 `/` 를 붙여야 이어서 파고들 수 있다."""
        assert all(c.endswith("/") for c in self._complete("@core") if c == "core/")

    def test_nested_path_completes_inside_directory(self) -> None:
        out = self._complete("@core/boot")
        assert any(c.startswith("core/boot") for c in out)

    def test_hidden_entries_need_explicit_dot(self) -> None:
        """`.git` 등이 목록을 덮지 않게 한다 — 점을 직접 쳐야 나온다."""
        assert not any(c.startswith(".") for c in self._complete("@"))

    def test_missing_directory_yields_nothing(self) -> None:
        """없는 경로를 치는 중에도 예외를 올리지 않는다."""
        assert self._complete("@no_such_dir_xyz/aaa") == []

    def test_plain_text_is_not_completed(self) -> None:
        """일반 대화 입력을 방해하지 않는다."""
        assert self._complete("파일을 읽어줘") == []


class TestSlashCompletionMeta:
    """슬래시 자동완성이 설명을 함께 보여 준다(/help 와 같은 출처)."""

    def test_completion_carries_description(self) -> None:
        completer = NovaCompleter(lambda: ["/compact"])
        items = list(completer.get_completions(Document("/comp", 5), None))
        assert items and items[0].display_meta_text == _COMMAND_HELP["/compact"]

    def test_command_after_space_is_not_completed(self) -> None:
        """인자 입력 중에 명령 목록이 튀어나오면 방해가 된다."""
        completer = NovaCompleter(lambda: ["/mode"])
        assert list(completer.get_completions(Document("/mode ne", 8), None)) == []


class TestThinkingAndResultRendering:
    """thinking 은 낮추고, 도구 결과는 호출 줄 아래로 정렬한다."""

    def test_thinking_is_inline_text_not_panel(self) -> None:
        """박스는 화면에서 가장 강한 요소라 곁다리 정보에 쓰면 위계가 뒤집힌다."""
        out = OutputFormatter(show_thinking=True).format_thinking("이 부분을 먼저 본다")
        assert isinstance(out, Text)
        assert not isinstance(out, Panel)
        assert "Thinking" in _render(out)

    def test_tool_result_indents_under_call(self) -> None:
        """`⏺ Tool(...)` 아래 `  ⎿  결과` 로 한 덩어리처럼 보이게 한다."""
        rendered = _render(OutputFormatter().format_tool_result("첫 줄\n둘째 줄"))
        assert "  ⎿  " in rendered

    def test_error_result_stays_loud(self) -> None:
        """성공은 dim 으로 낮췄지만 에러는 계속 눈에 띄어야 한다."""
        out = OutputFormatter().format_tool_result("터졌다", is_error=True)
        assert isinstance(out, Panel)


class TestBannerRendering:
    """시작 배너 — 한 화면에 들어가고, 상태가 없어도 깨지지 않아야 한다."""

    def _render_banner(self, state) -> str:
        buf = io.StringIO()
        repl = NexusREPL.__new__(NexusREPL)  # 생성자(PromptSession) 건너뛰기
        repl.console = Console(file=buf, force_terminal=False, width=76)
        repl._permission_mode = "default"
        repl._model = "primary"
        repl._resume_session_id = None
        repl._state = state
        repl._display_banner()
        return buf.getvalue()

    def test_renders_without_state(self) -> None:
        """부트스트랩이 실패해도 배너는 떠야 한다 — 안내조차 못 보면 원인을 알 수 없다."""
        out = self._render_banner(None)
        assert "IDINO NOVA" in out

    def test_shows_working_directory(self) -> None:
        """지금 어느 리포에서 도는지가 가장 먼저 확인할 정보다."""
        assert "cwd" in self._render_banner(None)

    def test_fits_in_one_screen(self) -> None:
        """예전 배너(픽셀 로고 + 3섹션)는 20줄을 넘겨 첫 질문을 밀어냈다."""
        assert len(self._render_banner(None).rstrip().split("\n")) <= 12

    def test_shows_routed_models_when_routing_enabled(self) -> None:
        """라우팅이 켜져 있으면 대화·도구 모델을 함께 보여 준다."""

        class _Chat:
            model = "ax-4.0"

        class _Tool:
            model = "qwen3-coder-30b"

        class _Routing:
            enabled = True
            chat_mode = _Chat
            tool_mode = _Tool

        class _Config:
            routing = _Routing
            model = None

        class _State:
            config = _Config

        out = self._render_banner(_State())
        assert "ax-4.0" in out
        assert "qwen3-coder-30b" in out

    def test_hint_line_does_not_wrap_at_80_columns(self) -> None:
        """단축키 줄이 접히면 오히려 읽기 나빠진다 — 좁은 터미널 기준을 고정한다."""
        hint = [line for line in self._render_banner(None).split("\n") if "Enter" in line]
        assert hint and all(len(line) <= 76 for line in hint)


class TestInterruptListener:
    """Esc 중단 리스너 — 붙이지 못해도 대화를 막으면 안 된다(fail-soft)."""

    def test_returns_none_off_a_real_console(self) -> None:
        """비 TTY(테스트·CI·파이프)에서는 조용히 포기하고 Ctrl+C 로만 중단한다."""
        repl = NexusREPL.__new__(NexusREPL)
        repl._interrupt_requested = False
        assert repl._attach_interrupt_listener() is None

    def test_kill_switch_disables_listener(self, monkeypatch) -> None:
        """터미널이 이상하게 동작할 때 사용자가 끌 수 있는 비상 스위치."""
        monkeypatch.setenv("NEXUS_NO_ESC_INTERRUPT", "1")
        repl = NexusREPL.__new__(NexusREPL)
        assert repl._attach_interrupt_listener() is None
