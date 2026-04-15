"""
Rich REPL — 터미널 기반 대화형 인터페이스.

Claude Code의 REPL.ts를 Rich/prompt-toolkit으로 재구현한다.
사용자 입력 → QueryEngine → StreamEvent 스트리밍 출력 루프를 제공한다.

의존성 방향: cli/ → core/ (단방향)

주요 기능:
  - Rich 배너 표시
  - 세션 명령어 (/help, /clear, /exit, /model, /config, /session)
  - StreamEvent 실시간 스트리밍 출력 (Rich Live)
  - 권한 프롬프트 (Y/N/Always)
  - Ctrl+C 안전 처리
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.formatters import OutputFormatter
from core.message import StreamEvent, StreamEventType

logger = logging.getLogger("nexus.cli.repl")

# ─── 버전 정보 ───
__version__ = "0.1.0"

# ─── 배너 텍스트 ───
_BANNER = r"""
 _   _
| \ | | _____  ___   _ ___
|  \| |/ _ \ \/ / | | / __|
| |\  |  __/>  <| |_| \__ \
|_| \_|\___/_/\_\\__,_|___/

Project Nexus v{version} — 에어갭 로컬 LLM 오케스트레이션
"""


class NexusREPL:
    """
    Rich 기반 터미널 REPL.

    메인 루프에서 사용자 입력을 받아 QueryEngine에 전달하고,
    AsyncGenerator로 돌아오는 StreamEvent를 실시간으로 표시한다.
    """

    def __init__(
        self,
        permission_mode: str = "default",
        model: str = "primary",
        resume_session_id: str | None = None,
    ):
        """
        REPL을 초기화한다.

        Args:
            permission_mode: 권한 모드 (default, auto, plan, trust, bypass)
            model: 사용할 모델 (primary, auxiliary)
            resume_session_id: 이어서 할 세션 ID (None이면 새 세션)
        """
        self.console = Console()
        self._formatter = OutputFormatter(show_thinking=False)
        self._permission_mode = permission_mode
        self._model = model
        self._resume_session_id = resume_session_id
        self._running = False

        # prompt-toolkit 세션 (히스토리 + 멀티라인 지원)
        self._prompt_session: PromptSession = PromptSession(
            history=InMemoryHistory(),
        )

        # 세션 명령어 맵 — 슬래시 명령어를 처리한다
        self._session_commands: dict[str, Any] = {
            "/help": self._cmd_help,
            "/clear": self._cmd_clear,
            "/exit": self._cmd_exit,
            "/model": self._cmd_model,
            "/config": self._cmd_config,
            "/session": self._cmd_session,
            "/thinking": self._cmd_thinking,
        }

        # 상태 변수
        self._state: Any = None  # GlobalState (bootstrap 후 설정)
        self._query_engine: Any = None  # QueryEngine (Phase 2 초기화 후 설정)

    # ─── 메인 루프 ───

    async def run(self) -> None:
        """
        메인 REPL 루프.

        1. 부트스트랩 (Phase 1 초기화)
        2. 배너 표시
        3. 입력 루프:
           - 세션 명령어이면 해당 핸들러 실행
           - 일반 텍스트이면 query_loop 실행 + 스트리밍 출력
        4. Ctrl+C → 현재 요청만 취소, Ctrl+D → 종료
        """
        # ① Phase 1 부트스트랩
        await self._bootstrap()

        # ② 배너 표시
        self._display_banner()

        # ③ 입력 루프
        self._running = True
        while self._running:
            try:
                # prompt-toolkit으로 사용자 입력을 받는다
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._prompt_session.prompt("nexus> "),
                )

                # 빈 입력은 무시한다
                if not user_input.strip():
                    continue

                # 세션 명령어 확인
                command_key = user_input.strip().split()[0].lower()
                if command_key in self._session_commands:
                    # 명령어 인자를 분리하여 전달한다
                    args = user_input.strip().split()[1:]
                    await self._session_commands[command_key](args)
                    continue

                # 일반 메시지 → query_loop 실행
                await self._process_message(user_input.strip())

            except KeyboardInterrupt:
                # Ctrl+C → 현재 요청 취소, REPL은 계속 실행
                self.console.print("\n[yellow]요청 취소됨[/yellow]")
                continue
            except EOFError:
                # Ctrl+D → 종료
                self.console.print("\n[dim]세션을 종료합니다.[/dim]")
                break

        # ④ 종료 처리
        await self._shutdown()

    # ─── 부트스트랩 ───

    async def _bootstrap(self) -> None:
        """
        Phase 1 + Phase 2 초기화를 수행한다.

        Phase 1: GlobalState와 설정을 로딩한다.
        Phase 2: ToolRegistry, MemoryManager, QueryEngine을 초기화한다.
        """
        try:
            from core.bootstrap import init, init_phase2

            # Phase 1: 환경 비의존 초기화
            self._state = await init()

            # Phase 2: QueryEngine + ToolRegistry + MemoryManager
            components = await init_phase2(self._state)
            self._query_engine = components.get("query_engine")
            logger.info("REPL 부트스트랩 완료 (Phase 1 + 2)")
        except Exception as e:
            # 부트스트랩 실패 시에도 기본 REPL은 동작하도록 한다
            logger.warning(f"부트스트랩 실패, 기본 모드로 시작: {e}")
            self._state = None

    # ─── 배너 표시 ───

    def _display_banner(self) -> None:
        """시작 배너와 도움말 힌트를 표시한다."""
        banner_text = _BANNER.format(version=__version__)
        self.console.print(
            Panel(
                Text(banner_text, style="bold blue"),
                border_style="blue",
                expand=False,
            )
        )
        self.console.print(
            "[dim]/help로 명령어를 확인하세요. "
            "Ctrl+C로 요청을 취소하고, Ctrl+D로 종료합니다.[/dim]\n"
        )

        # 모델 정보 표시
        model_info = f"모델: {self._model} | 권한: {self._permission_mode}"
        if self._resume_session_id:
            model_info += f" | 세션: {self._resume_session_id}"
        self.console.print(f"[dim]{model_info}[/dim]\n")

    # ─── 메시지 처리 ───

    async def _process_message(self, user_input: str) -> None:
        """
        사용자 메시지를 QueryEngine에 전달하고 결과를 스트리밍한다.

        QueryEngine이 아직 없으면 (Phase 3 미완성) 안내 메시지를 표시한다.
        """
        if self._query_engine is None:
            # QueryEngine이 없으면 placeholder 메시지를 표시한다
            # Phase 3 (Orchestrator) 완성 후 실제 query_loop 연동
            self.console.print(
                Panel(
                    "[yellow]QueryEngine이 아직 초기화되지 않았습니다.\n"
                    "Phase 3 (Orchestrator) 모듈이 완성되면 자동으로 연동됩니다.[/yellow]",
                    title="[bold yellow]안내[/bold yellow]",
                    border_style="yellow",
                )
            )
            return

        # QueryEngine에서 AsyncGenerator로 StreamEvent를 수신하여 표시한다
        try:
            async for event in self._query_engine.submit_message(user_input):
                self.display_stream_event(event)
        except asyncio.CancelledError:
            self.console.print("[yellow]요청이 취소되었습니다.[/yellow]")
        except Exception as e:
            self.console.print(self._formatter.format_error(str(e)))

    # ─── StreamEvent 표시 ───

    def display_stream_event(self, event: StreamEvent) -> None:
        """
        StreamEvent를 터미널에 표시한다.

        OutputFormatter를 사용하여 이벤트를 Rich 렌더러블로 변환하고,
        텍스트 델타는 실시간으로, Panel은 한 번에 출력한다.
        """
        result = self._formatter.format_event(event)
        if result is None:
            return

        # 텍스트 델타는 줄바꿈 없이 이어서 출력한다
        if event.type == StreamEventType.TEXT_DELTA:
            self.console.print(result, end="")
        # 사용량 정보는 줄바꿈 후 출력한다
        elif event.type == StreamEventType.USAGE_UPDATE:
            self.console.print()  # 텍스트 스트림 끝 줄바꿈
            self.console.print(result)
        # 나머지는 Panel 등 블록 단위로 출력한다
        else:
            self.console.print(result)

    # ─── 권한 프롬프트 ───

    async def prompt_permission(self, tool_name: str, message: str) -> bool:
        """
        사용자에게 도구 실행 권한을 요청한다.

        Layer 3 (CanUseToolHandler)에서 ASK 결정이 나오면 호출된다.

        Args:
            tool_name: 실행하려는 도구 이름
            message: 권한 요청 메시지 (왜 이 도구를 실행하려는지)

        Returns:
            True이면 허용, False이면 거부
        """
        self.console.print(
            Panel(
                f"[yellow]{message}[/yellow]",
                title=f"[bold yellow]권한 요청: {tool_name}[/bold yellow]",
                border_style="yellow",
            )
        )

        # 사용자 입력: Y(허용), N(거부), A(항상 허용)
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._prompt_session.prompt("(Y)허용 / (N)거부 / (A)항상허용: "),
            )
            choice = response.strip().upper()
            if choice in ("Y", "YES"):
                return True
            if choice in ("A", "ALWAYS"):
                # TODO(nexus): 항상 허용 정책을 세션에 저장
                return True
            return False
        except (KeyboardInterrupt, EOFError):
            return False

    # ─── 도구 결과 포맷 ───

    def _format_tool_result(self, content: str) -> str:
        """
        도구 결과를 포맷한다.

        코드 블록이 포함되어 있으면 syntax highlighting을 적용한다.
        """
        return content

    # ─── 세션 명령어 핸들러 ───

    async def _cmd_help(self, args: list[str]) -> None:
        """도움말을 표시한다."""
        table = Table(title="세션 명령어", border_style="blue")
        table.add_column("명령어", style="cyan", no_wrap=True)
        table.add_column("설명", style="white")

        commands = [
            ("/help", "이 도움말을 표시한다"),
            ("/clear", "화면을 지운다"),
            ("/exit", "세션을 종료한다"),
            ("/model [name]", "현재 모델을 표시하거나 변경한다"),
            ("/config", "현재 설정을 표시한다"),
            ("/session", "세션 정보를 표시한다"),
            ("/thinking", "thinking 표시를 토글한다"),
        ]
        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print(table)

    async def _cmd_clear(self, args: list[str]) -> None:
        """화면을 지운다."""
        self.console.clear()

    async def _cmd_exit(self, args: list[str]) -> None:
        """세션을 종료한다."""
        self.console.print("[dim]세션을 종료합니다.[/dim]")
        self._running = False

    async def _cmd_model(self, args: list[str]) -> None:
        """현재 모델을 표시하거나 변경한다."""
        if args:
            # 모델 변경
            new_model = args[0]
            if new_model in ("primary", "auxiliary"):
                self._model = new_model
                self.console.print(f"[green]모델이 '{new_model}'로 변경되었습니다.[/green]")
            else:
                self.console.print(
                    "[red]유효하지 않은 모델입니다. 'primary' 또는 'auxiliary'를 사용하세요.[/red]"
                )
        else:
            self.console.print(f"현재 모델: [bold]{self._model}[/bold]")

    async def _cmd_config(self, args: list[str]) -> None:
        """현재 설정을 표시한다."""
        if self._state and self._state.config:
            config = self._state.config
            table = Table(title="현재 설정", border_style="blue")
            table.add_column("항목", style="cyan")
            table.add_column("값", style="white")
            table.add_row("GPU 서버", config.gpu_server_url)
            table.add_row("에어갭 모드", str(config.air_gap_mode))
            table.add_row("로그 레벨", config.log_level)
            table.add_row("모델", self._model)
            table.add_row("권한 모드", self._permission_mode)
            self.console.print(table)
        else:
            self.console.print("[yellow]설정이 로드되지 않았습니다.[/yellow]")

    async def _cmd_session(self, args: list[str]) -> None:
        """세션 정보를 표시한다."""
        if self._state:
            summary = self._state.get_session_summary()
            table = Table(title="세션 정보", border_style="blue")
            table.add_column("항목", style="cyan")
            table.add_column("값", style="white")
            for key, value in summary.items():
                table.add_row(key, str(value))
            self.console.print(table)
        else:
            self.console.print("[yellow]세션이 초기화되지 않았습니다.[/yellow]")

    async def _cmd_thinking(self, args: list[str]) -> None:
        """thinking 표시를 토글한다."""
        self._formatter.show_thinking = not self._formatter.show_thinking
        status = "켜짐" if self._formatter.show_thinking else "꺼짐"
        self.console.print(f"[green]Thinking 표시: {status}[/green]")

    # ─── 종료 ───

    async def _shutdown(self) -> None:
        """종료 시 정리 작업을 수행한다."""
        if self._state:
            summary = self._state.get_session_summary()
            self.console.print(
                Panel(
                    f"턴: {summary['turns']} | "
                    f"입력 토큰: {summary['total_input_tokens']:,} | "
                    f"출력 토큰: {summary['total_output_tokens']:,} | "
                    f"도구 호출: {summary['total_tool_calls']}",
                    title="[bold]세션 요약[/bold]",
                    border_style="dim",
                )
            )
        self.console.print("[dim]Goodbye![/dim]")


def main():
    """CLI 진입점. pyproject.toml의 [project.scripts] nexus 엔트리포인트."""
    repl = NexusREPL()
    asyncio.run(repl.run())


if __name__ == "__main__":
    main()
