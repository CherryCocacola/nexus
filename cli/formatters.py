"""
출력 포매터 — StreamEvent를 Rich 형식으로 변환한다.

4-Tier AsyncGenerator 체인의 최종 단계에서 StreamEvent를 수신하여
사용자에게 보기 좋은 터미널 출력으로 변환하는 역할을 한다.

의존성 방향: cli/ → core/ (단방향)
"""

from __future__ import annotations

import logging
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from core.message import StreamEvent, StreamEventType, TokenUsage

logger = logging.getLogger("nexus.cli.formatters")


class OutputFormatter:
    """
    StreamEvent를 Rich 렌더러블 객체로 변환한다.

    query_loop에서 yield된 StreamEvent를 받아서
    터미널에 표시할 수 있는 Rich Panel, Markdown, Text 등으로 변환한다.
    """

    def __init__(self, show_thinking: bool = False):
        """
        포매터를 초기화한다.

        Args:
            show_thinking: thinking 블록을 표시할지 여부.
                           디버깅 시에만 True로 설정한다.
        """
        self._show_thinking = show_thinking

    # ─── 텍스트 델타 ───

    # Qwen 3.5 thinking 필터링 상태
    _in_thinking: bool = False
    _thinking_buffer: str = ""

    def format_text_delta(self, text: str) -> str:
        """
        모델의 텍스트 조각(delta)을 반환한다.

        Qwen 3.5는 응답 전에 thinking 텍스트를 출력한다.
        </think> 태그 이전의 텍스트는 작게 표시하고,
        이후의 실제 응답만 일반 크기로 출력한다.
        """
        # </think> 감지 — thinking 종료
        if "</think>" in text:
            parts = text.split("</think>", 1)
            self._in_thinking = False
            # thinking 종료 후 실제 응답 부분만 반환
            after = parts[1].lstrip("\n")
            if self._thinking_buffer:
                # thinking 내용을 한 줄로 요약하여 작게 표시
                short = self._thinking_buffer[:80].replace("\n", " ")
                self._thinking_buffer = ""
                prefix = f"[dim italic]  thinking: {short}...[/dim italic]\n" if short.strip() else ""
                return prefix + after
            return after

        # thinking 중인지 판별
        # 첫 토큰에서 thinking 시작 감지 (소문자/대문자 다양)
        if not self._in_thinking and not self._thinking_buffer:
            lower = text.lower().strip()
            if any(lower.startswith(p) for p in [
                "thinking", "the user", "i need to", "let me", "i should",
                "분석", "사용자가", "먼저",
            ]):
                self._in_thinking = True
                self._thinking_buffer = text
                return ""

        if self._in_thinking:
            self._thinking_buffer += text
            return ""

        return text

    # ─── 도구 사용 ───

    def format_tool_use(self, tool_name: str, input_data: dict[str, Any]) -> Panel:
        """
        도구 호출 정보를 Panel로 포맷한다.

        도구 이름과 입력 데이터를 JSON 하이라이팅으로 표시하여
        사용자가 어떤 도구가 어떤 인자로 호출되었는지 쉽게 파악할 수 있다.
        """
        import json

        # 입력 데이터를 보기 좋은 JSON으로 변환한다
        input_json = json.dumps(input_data, indent=2, ensure_ascii=False)

        # Syntax 하이라이팅을 적용한다
        syntax = Syntax(input_json, "json", theme="monokai", line_numbers=False)

        return Panel(
            syntax,
            title=f"[bold cyan]Tool: {tool_name}[/bold cyan]",
            border_style="cyan",
            expand=False,
        )

    # ─── 도구 결과 ───

    def format_tool_result(self, content: str, is_error: bool = False) -> Panel:
        """
        도구 실행 결과를 Panel로 포맷한다.

        에러인 경우 빨간색 테두리, 정상이면 녹색 테두리를 사용한다.
        코드 블록이 포함된 경우 자동으로 syntax highlighting을 적용한다.
        """
        if is_error:
            # 에러 결과: 빨간색 테두리와 아이콘
            return Panel(
                Text(content, style="red"),
                title="[bold red]Error[/bold red]",
                border_style="red",
                expand=False,
            )

        # 정상 결과: 내용이 길면 축약하여 표시한다
        # 최대 표시 줄 수 제한 (터미널 가독성)
        max_lines = 50
        lines = content.split("\n")
        if len(lines) > max_lines:
            truncated = "\n".join(lines[:max_lines])
            truncated += f"\n... ({len(lines) - max_lines}줄 생략)"
        else:
            truncated = content

        return Panel(
            truncated,
            title="[bold green]Result[/bold green]",
            border_style="green",
            expand=False,
        )

    # ─── 사고(Thinking) ───

    def format_thinking(self, text: str) -> Panel:
        """
        모델의 사고 과정을 Panel로 포맷한다.

        show_thinking이 False이면 빈 Panel을 반환하지 않고
        호출 자체를 하지 않아야 한다 (호출자 책임).
        """
        return Panel(
            Markdown(text),
            title="[bold yellow]Thinking[/bold yellow]",
            border_style="yellow",
            expand=False,
        )

    # ─── 에러 ───

    def format_error(self, message: str) -> Panel:
        """
        시스템 에러 메시지를 Panel로 포맷한다.

        모델 에러, 네트워크 에러, 권한 거부 등 다양한 에러를 표시한다.
        """
        return Panel(
            Text(message, style="bold red"),
            title="[bold red]Error[/bold red]",
            border_style="red",
            expand=False,
        )

    # ─── 사용량 ───

    def format_usage(self, usage: TokenUsage) -> str:
        """
        토큰 사용량을 한 줄 문자열로 포맷한다.

        REPL 하단에 표시되는 간결한 형식이다.
        """
        parts = [
            f"입력: {usage.input_tokens:,}",
            f"출력: {usage.output_tokens:,}",
        ]
        # 캐시 정보가 있으면 추가한다
        if usage.cache_read_input_tokens > 0:
            parts.append(f"캐시읽기: {usage.cache_read_input_tokens:,}")
        if usage.cache_creation_input_tokens > 0:
            parts.append(f"캐시생성: {usage.cache_creation_input_tokens:,}")

        return f"[dim]토큰 | {' | '.join(parts)} | 합계: {usage.total_tokens:,}[/dim]"

    # ─── StreamEvent 라우터 ───

    @property
    def show_thinking(self) -> bool:
        """thinking 표시 여부를 반환한다."""
        return self._show_thinking

    @show_thinking.setter
    def show_thinking(self, value: bool) -> None:
        """thinking 표시 여부를 설정한다."""
        self._show_thinking = value

    def format_event(self, event: StreamEvent) -> Any | None:
        """
        StreamEvent를 적절한 Rich 렌더러블로 변환한다.

        각 이벤트 타입에 따라 적절한 format_* 메서드를 호출한다.
        표시할 필요가 없는 이벤트는 None을 반환한다.

        Args:
            event: 4-Tier 체인에서 yield된 StreamEvent

        Returns:
            Rich 렌더러블 객체 또는 None
        """
        event_type = event.type

        # 텍스트 델타 — 실시간 스트리밍 출력
        if event_type == StreamEventType.TEXT_DELTA and event.text:
            return self.format_text_delta(event.text)

        # 도구 사용 시작 — 도구 이름과 입력 표시
        if event_type == StreamEventType.TOOL_USE_START and event.tool_use:
            return self.format_tool_use(
                tool_name=event.tool_use.name,
                input_data=event.tool_use.input,
            )

        # 도구 결과 — 실행 결과 표시
        if event_type == StreamEventType.TOOL_RESULT and event.tool_result:
            return self.format_tool_result(
                content=event.tool_result.content,
                is_error=event.tool_result.is_error,
            )

        # 사고(Thinking) 델타 — 디버그 모드에서만 표시
        if event_type == StreamEventType.THINKING_DELTA and event.thinking_text:
            if self._show_thinking:
                return self.format_thinking(event.thinking_text)
            return None

        # 에러 — 에러 메시지 표시
        if event_type == StreamEventType.ERROR and event.message:
            return self.format_error(event.message)

        # 사용량 업데이트 — 토큰 사용량 표시
        if event_type == StreamEventType.USAGE_UPDATE and event.usage:
            return self.format_usage(event.usage)

        # 그 외 이벤트는 표시하지 않는다
        return None
