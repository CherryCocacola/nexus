"""
CLI 명령어 — Click 기반 커맨드라인 인터페이스.

'nexus' CLI의 하위 명령어를 정의한다.
  - chat: 대화형 세션 시작
  - ask: 단일 질문 (비대화형)
  - version: 버전 정보 표시
  - health: GPU 서버 상태 확인

의존성 방향: cli/ → core/ (단방향)
"""

from __future__ import annotations

import asyncio
import logging
import sys

import click

logger = logging.getLogger("nexus.cli.commands")


@click.group()
def cli():
    """Project Nexus — 에어갭 로컬 LLM 오케스트레이션 플랫폼."""
    pass


@cli.command()
@click.option(
    "--model",
    default="primary",
    type=click.Choice(["primary", "auxiliary"]),
    help="사용할 모델 (primary: Gemma 4 31B, auxiliary: ExaOne 7.8B)",
)
@click.option(
    "--permission-mode",
    default="default",
    type=click.Choice(["default", "auto", "plan", "trust", "bypass"]),
    help="권한 모드",
)
@click.option(
    "--resume",
    default=None,
    help="이어서 할 세션 ID",
)
def chat(model: str, permission_mode: str, resume: str | None) -> None:
    """대화형 채팅 세션을 시작한다."""
    from cli.repl import NexusREPL

    repl = NexusREPL(
        permission_mode=permission_mode,
        model=model,
        resume_session_id=resume,
    )
    asyncio.run(repl.run())


@cli.command()
def version() -> None:
    """버전 정보를 표시한다."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Project Nexus 버전 정보", border_style="blue")
    table.add_column("항목", style="cyan")
    table.add_column("값", style="white")

    table.add_row("Nexus 버전", "0.1.0")
    table.add_row("Python 버전", sys.version.split()[0])

    # 설정에서 모델 정보를 가져온다
    try:
        from core.config import load_and_validate_config

        config = load_and_validate_config()
        table.add_row("Primary 모델", config.model.primary_model)
        table.add_row("Auxiliary 모델", config.model.auxiliary_model)
        table.add_row("GPU 서버", config.gpu_server_url)
        table.add_row("에어갭 모드", str(config.air_gap_mode))
    except Exception:
        # 설정 로드 실패 시 기본 정보만 표시한다
        table.add_row("설정", "로드 실패")

    console.print(table)


@cli.command()
@click.argument("query")
@click.option(
    "--model",
    default="primary",
    type=click.Choice(["primary", "auxiliary"]),
    help="사용할 모델",
)
def ask(query: str, model: str) -> None:
    """
    단일 질문을 보낸다 (비대화형).

    스크립트나 파이프라인에서 사용할 수 있는 비대화형 모드이다.
    QueryEngine에 한 번만 메시지를 보내고 결과를 출력한 뒤 종료한다.
    """
    from rich.console import Console

    console = Console()

    async def _run_ask():
        """비대화형 질문을 실행한다."""
        try:
            from core.bootstrap import init

            await init()
        except Exception as e:
            console.print(f"[red]부트스트랩 실패: {e}[/red]")
            sys.exit(1)

        # QueryEngine이 없으면 안내 메시지를 표시한다
        # TODO(nexus): Phase 3 완성 후 QueryEngine 연동
        console.print(
            "[yellow]QueryEngine이 아직 초기화되지 않았습니다.\n"
            "Phase 3 (Orchestrator) 모듈이 완성되면 사용할 수 있습니다.[/yellow]"
        )

    asyncio.run(_run_ask())


@cli.command()
def health() -> None:
    """GPU 서버 상태를 확인한다."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    async def _check_health():
        """GPU 서버 헬스체크를 수행한다."""
        try:
            from core.config import load_and_validate_config

            config = load_and_validate_config()
            gpu_url = config.gpu_server_url
        except Exception:
            gpu_url = "http://localhost:8000"

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{gpu_url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    table = Table(title="GPU 서버 상태", border_style="green")
                    table.add_column("항목", style="cyan")
                    table.add_column("값", style="white")
                    table.add_row("URL", gpu_url)
                    table.add_row("상태", "[green]정상[/green]")
                    for key, value in data.items():
                        table.add_row(key, str(value))
                    console.print(table)
                else:
                    console.print(f"[red]GPU 서버 응답 이상: HTTP {resp.status_code}[/red]")
        except httpx.ConnectError:
            console.print(
                f"[red]GPU 서버에 연결할 수 없습니다: {gpu_url}\n"
                f"서버가 실행 중인지 확인하세요.[/red]"
            )
        except Exception as e:
            console.print(f"[red]헬스체크 실패: {e}[/red]")

    asyncio.run(_check_health())


if __name__ == "__main__":
    cli()
