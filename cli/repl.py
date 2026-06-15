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
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.formatters import OutputFormatter
from core.message import StreamEvent, StreamEventType

logger = logging.getLogger("nexus.cli.repl")

# ─── 버전 정보 ───
__version__ = "0.1.0"

# ─── IDINO 회사 마크 (블록 ASCII 아트) ───
# 배너 좌측에 배치되는 IDINO 로고. 사용자 제공 원본 디자인을 정확히 유지한다.
# 블록(████)은 4칸씩, 블록 사이 공백도 4칸 — 가로 폭 20칸의 정사각 비율.
# 이 비율을 어기면 'i'/'d' 글자 형태가 깨지므로 변형 금지.
_IDINO_MARK = """\
████    ████
████    ████
        ████
        ████
████████████████████
████████████████████
████    ████    ████
████    ████    ████
████████████    ████
████████████    ████\
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
        log_level: str = "WARNING",
    ):
        """
        REPL을 초기화한다.

        Args:
            permission_mode: 권한 모드 (default, auto, plan, trust, bypass)
            model: 사용할 모델 (primary, auxiliary)
            resume_session_id: 이어서 할 세션 ID (None이면 새 세션)
            log_level: 채팅 화면에 표시할 nexus.* 로그 레벨 (v0.14.12).
                기본 WARNING — 채팅 흐름에 운영 INFO가 끼어들지 않게 한다.
                bootstrap이 _configure_logging으로 INFO로 reset하지만,
                부트스트랩 직후 _apply_log_level()이 다시 적용한다.
        """
        self.console = Console()
        self._formatter = OutputFormatter(show_thinking=False)
        self._permission_mode = permission_mode
        self._model = model
        self._resume_session_id = resume_session_id
        self._log_level = log_level
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
        finally:
            # v0.14.12 — bootstrap의 _configure_logging이 nexus.* 레벨을 INFO로
            # 재설정하므로 사용자 지정 log_level을 부트스트랩 직후 다시 적용.
            self._apply_log_level()

    def _apply_log_level(self) -> None:
        """nexus.* 로거 레벨을 self._log_level로 설정한다.

        대화 화면에 운영 INFO 로그가 흐르지 않도록 채팅 시작 직전 호출.
        """
        try:
            level = getattr(logging, self._log_level.upper(), logging.WARNING)
            logging.getLogger("nexus").setLevel(level)
        except Exception as e:  # noqa: BLE001
            logger.warning("로그 레벨 적용 실패: %s", e)

    # ─── 배너 표시 ───

    def _display_banner(self) -> None:
        """시작 배너 + 도움말 힌트를 표시한다 (v0.14.10 IDINO 코퍼레이트).

        IDINO 픽셀 마크(images.png 기반)를 좌측, 우측에 회사 카드·라우팅
        모델·인프라·슬래시 커맨드를 묶어 Claude Code 스타일 패널로 출력한다.
        라우팅이 활성이면 KNOWLEDGE/TOOL/CHAT 각 분기에 실제 사용 모델명
        (예: qwen3.5-27b, nexus-phase3)을 노출하여 alias("primary") 모호성을
        제거한다.
        """
        # ───── 좌측: IDINO 픽셀 마크 ─────
        # IDINO 코퍼레이트 컬러 rgb(0,71,157) — 배지/홈페이지/이미지와 일치
        mark = Text(_IDINO_MARK, style="bold rgb(0,71,157)")

        # ───── 우측: 회사 카드 + 시스템 정보 ─────
        info = Text()
        info.append("Project Nexus", style="bold rgb(0,71,157)")
        info.append(f"  v{__version__}\n", style="dim white")
        info.append("에어갭 로컬 LLM 오케스트레이션 플랫폼\n", style="white")
        info.append("Powered by ", style="dim white")
        info.append("IDINO Corp.", style="bold rgb(0,71,157)")
        info.append("  ·  Air-gapped AI for Enterprise\n", style="dim white")

        # 구분선
        info.append("─" * 44 + "\n", style="dim rgb(0,71,157)")

        # 라우팅 활성 여부에 따른 모델 표시 — alias 대신 실제 served-model-name
        routing = getattr(getattr(self._state, "config", None), "routing", None)
        if routing is not None and getattr(routing, "enabled", False):
            chat_model = getattr(
                getattr(routing, "chat_mode", None), "model", "?"
            )
            know_model = getattr(
                getattr(routing, "knowledge_mode", None), "model", "?"
            )
            tool_model = getattr(
                getattr(routing, "tool_mode", None), "model", "?"
            )
            info.append("모델 라우팅 (질의별 자동 분기)\n", style="bold white")
            info.append(f"  CHAT       {chat_model}\n", style="cyan")
            info.append(f"  KNOWLEDGE  {know_model}\n", style="green")
            info.append(f"  TOOL       {tool_model}\n", style="yellow")
        else:
            # 라우팅 비활성 — primary/auxiliary 모델 표시
            model_cfg = getattr(
                getattr(self._state, "config", None), "model", None
            )
            primary = getattr(model_cfg, "primary_model", self._model)
            aux = getattr(model_cfg, "auxiliary_model", "-")
            info.append("모델\n", style="bold white")
            info.append(f"  Primary    {primary}\n", style="cyan")
            info.append(f"  Auxiliary  {aux}\n", style="dim")

        # 인프라 한 줄
        gpu_url = getattr(
            getattr(self._state, "config", None), "gpu_server_url", "(?)"
        )
        scout_cfg = getattr(
            getattr(self._state, "config", None), "scout", None
        )
        scout_url = getattr(scout_cfg, "base_url", "-") if scout_cfg else "-"
        info.append("인프라\n", style="bold white")
        info.append(f"  Worker     {gpu_url}\n", style="dim")
        info.append(f"  Scout      {scout_url}\n", style="dim")

        # 세션 컨텍스트
        info.append("세션\n", style="bold white")
        info.append(f"  권한 모드   {self._permission_mode}\n", style="dim")
        if self._resume_session_id:
            info.append(f"  복원 세션   {self._resume_session_id}\n", style="dim")
        else:
            info.append("  복원 세션   (신규)\n", style="dim")

        # ───── 좌·우 grid 합치기 ─────
        layout = Table.grid(padding=(0, 4))
        layout.add_column(justify="left", vertical="top")
        layout.add_column(justify="left", vertical="top")
        layout.add_row(mark, info)

        # Claude Code 스타일 — 둥근 모서리 + 좌측 정렬 타이틀
        self.console.print(
            Panel(
                layout,
                title="[bold]✻ Welcome to Nexus[/bold]",
                title_align="left",
                subtitle="[dim]IDINO Corp. · 2026[/dim]",
                subtitle_align="right",
                border_style="rgb(0,71,157)",
                box=ROUNDED,
                padding=(1, 2),
                expand=False,
            )
        )
        # 슬래시 커맨드 단축키 한 줄
        self.console.print(
            "[dim]✻ 슬래시 커맨드: [/dim]"
            "[cyan]/help[/cyan][dim] · [/dim]"
            "[cyan]/model[/cyan][dim] · [/dim]"
            "[cyan]/config[/cyan][dim] · [/dim]"
            "[cyan]/session[/cyan][dim] · [/dim]"
            "[cyan]/clear[/cyan][dim] · [/dim]"
            "[cyan]/exit[/cyan]"
        )
        self.console.print(
            "[dim]   단축키:    [/dim]"
            "[white]Enter[/white][dim] 전송 · [/dim]"
            "[white]Ctrl+C[/white][dim] 요청 취소 · [/dim]"
            "[white]Ctrl+D[/white][dim] 종료[/dim]\n"
        )

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

        # QueryEngine에서 AsyncGenerator로 StreamEvent를 수신하여 표시한다.
        # v0.14.8 — Console.status() spinner로 단계별 진행을 시각화.
        # 첫 TEXT_DELTA가 도착하면 spinner를 닫고 본문 출력으로 넘어간다.
        # 도구 호출이 시작되면 다시 spinner로 전환해 "Reading X.py..." 같은
        # 단계 표시를 띄운다 — 사용자가 cold start 60초도 "분석 중"으로 인지.
        status_ctx = self.console.status(
            "[cyan]요청 분석 중...[/cyan]", spinner="dots"
        )
        status_ctx.__enter__()
        status_active = True
        try:
            async for event in self._query_engine.submit_message(user_input):
                # spinner를 켤지/끌지/문구를 갱신할지 결정
                next_label = self._stage_label_for(event)
                if next_label == "_TEXT_":
                    # 첫 본문 토큰이 도착 — spinner를 닫고 print로 전환
                    if status_active:
                        status_ctx.__exit__(None, None, None)
                        status_active = False
                elif next_label == "_TOOL_DONE_":
                    # 도구 완료 — Tool Panel을 곧 출력해야 하므로 spinner 닫기.
                    # spinner와 Panel이 동시에 활성이면 Panel이 깨져 보인다.
                    # 후속 이벤트(next LLM 응답·다음 도구 호출)에서 spinner는
                    # 아래 `elif next_label is not None` 분기가 자동 재개.
                    if status_active:
                        status_ctx.__exit__(None, None, None)
                        status_active = False
                elif next_label is not None:
                    # 단계 텍스트 갱신 (TURN_START/TOOL_USE_START 등)
                    if status_active:
                        status_ctx.update(next_label)
                    else:
                        # spinner가 닫혀 있으면 다시 연다 (도구 호출 시작 등)
                        status_ctx = self.console.status(
                            next_label, spinner="dots"
                        )
                        status_ctx.__enter__()
                        status_active = True

                self.display_stream_event(event)
        except asyncio.CancelledError:
            if status_active:
                status_ctx.__exit__(None, None, None)
                status_active = False
            self.console.print("[yellow]요청이 취소되었습니다.[/yellow]")
        except Exception as e:
            if status_active:
                status_ctx.__exit__(None, None, None)
                status_active = False
            self.console.print(self._formatter.format_error(str(e)))
        finally:
            # 안전하게 닫기 — TEXT가 안 도착했거나 예외 직후
            if status_active:
                status_ctx.__exit__(None, None, None)

    def _stage_label_for(self, event: Any) -> str | None:
        """StreamEvent를 보고 spinner 상태/문구를 결정한다.

        반환값:
          - "_TEXT_"        : TEXT_DELTA 도착 — spinner 닫고 본문 출력 모드
          - "_TOOL_DONE_"   : 도구 종료 — 다음 LLM 응답 대기 spinner
          - "[..]label[..]" : spinner 문구를 이 텍스트로 갱신
          - None            : 변동 없음 (현재 spinner 그대로)

        StreamEvent가 아닌 객체(Message 등)는 None을 돌려 변경 없음.
        """
        if not isinstance(event, StreamEvent):
            return None
        et = event.type
        if et == StreamEventType.TEXT_DELTA:
            return "_TEXT_"
        if et == StreamEventType.STREAM_REQUEST_START:
            # GPU 서버에 요청을 막 보낸 시점 — KB 검색이 끝난 직후
            return "[cyan]모델 추론 중...[/cyan]"
        if et == StreamEventType.MESSAGE_START:
            return "[cyan]응답 생성 중...[/cyan]"
        if et == StreamEventType.THINKING_START:
            return "[magenta]생각 정리 중...[/magenta]"
        if et == StreamEventType.TOOL_USE_START:
            tool_name = getattr(event, "tool_name", None) or "도구"
            return f"[yellow]{tool_name} 실행 중...[/yellow]"
        if et == StreamEventType.TOOL_USE_STOP:
            return "_TOOL_DONE_"
        return None

    # ─── StreamEvent 표시 ───

    def display_stream_event(self, event: StreamEvent) -> None:
        """
        StreamEvent를 터미널에 표시한다.

        OutputFormatter를 사용하여 이벤트를 Rich 렌더러블로 변환하고,
        텍스트 델타는 실시간으로, Panel은 한 번에 출력한다.

        QueryEngine.submit_message()는 StreamEvent와 Message를 모두 yield한다.
        Message 객체는 대화 히스토리용이므로 표시하지 않는다.
        """
        # Message 객체는 표시 대상이 아님 — StreamEvent만 처리
        if not isinstance(event, StreamEvent):
            return

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
