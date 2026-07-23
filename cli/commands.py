"""
CLI 명령어 정의 모듈 — Click 기반의 'nexus' 커맨드라인 인터페이스.

이 파일은 사용자가 터미널에서 `nexus <명령어>` 형태로 실행하는 진입점(entry point)들을
한곳에 모아 정의한다. Click 라이브러리의 `@click.group()`으로 최상위 그룹 `cli`를 만들고,
그 아래에 각 하위 명령어(subcommand)를 `@cli.command()`로 붙이는 구조다.

제공하는 하위 명령어:
  - chat    : 대화형 채팅 세션을 시작한다 (REPL 실행).
              실제 화면/입력 루프는 cli.repl.NexusREPL이 담당.
  - ask     : 단일 질문을 한 번만 보내고 답을 출력한 뒤 종료한다
              (스크립트/파이프라인용 비대화형 모드).
  - version : Nexus·Python 버전과 설정에 잡힌 모델/GPU 서버 정보를 표 형태로 보여준다.
  - health  : GPU 서버(Machine B)의 /health 엔드포인트를 호출해 살아있는지 확인한다.

설계 메모(초보자용):
  - 무거운 모듈(core.bootstrap, cli.repl, httpx 등)은 파일 상단에서 한꺼번에 import하지 않고,
    각 명령어 함수 안에서 필요할 때 lazy import 한다. 이렇게 하면 `nexus version` 같은 가벼운
    명령을 실행할 때 채팅 엔진 전체를 불러오는 비용을 피할 수 있고, 순환 import 위험도 줄어든다.
  - 비동기 로직(질문 전송, 헬스체크)은 내부에 async 함수를 정의한 뒤 asyncio.run()으로 돌린다.

의존성 방향: cli/ → core/ (단방향). CLI는 core를 호출하지만 core는 cli를 절대 import하지 않는다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import logging
import sys

import click

# 이 모듈 전용 로거. 프로젝트 규칙(P7)에 따라 "nexus.{module}" 네이밍을 사용한다.
# 현재 이 파일에서 직접 쓰지는 않지만, 향후 명령어 실행 로깅을 붙일 자리를 미리 마련해 둔 것.
logger = logging.getLogger("nexus.cli.commands")


# @click.group() 은 여러 하위 명령어를 담는 "최상위 명령 그룹"을 만든다.
# 즉 `nexus` 자체는 아무 동작도 하지 않고, 그 아래 chat/ask/version/health 로
# 분기시키는 허브 역할만 한다. 아래 각 함수에 붙은 @cli.command() 가 이 그룹에 등록된다.
@click.group()
def cli():
    """IDINO NOVA — 에어갭 로컬 LLM 오케스트레이션 플랫폼."""
    # 그룹 함수 자체는 실행할 로직이 없으므로 pass. 실제 일은 하위 명령어들이 처리한다.
    pass


@cli.command()
@click.option(
    "--model",
    default="primary",
    type=click.Choice(["primary", "auxiliary"]),
    help="사용할 모델 (primary: Qwen 3.5 27B, auxiliary: ExaOne 7.8B)",
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
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help=(
        "로그 표시 레벨 — 채팅 화면에 보이는 nexus.* 로그 임계. "
        "기본 WARNING(채팅에 안 보임). 디버깅 시 INFO/DEBUG로 올려서 본다."
    ),
)
def chat(
    model: str,
    permission_mode: str,
    resume: str | None,
    log_level: str,
) -> None:
    """
    대화형 채팅 세션을 시작한다 (`nexus chat`).

    역할:
        터미널 기반의 대화형 REPL(Read-Eval-Print Loop)을 띄운다. 사용자가 프롬프트에
        질문을 입력하면 모델이 스트리밍으로 답하고, 계속 이어서 대화할 수 있다.
        이 함수 자체는 "설정을 모아 REPL 객체를 만들고 실행"하는 얇은 진입점이며,
        실제 화면 렌더링·입력 처리·에이전트 루프는 cli.repl.NexusREPL 이 담당한다.

    매개변수(모두 위의 @click.option / 인자에서 주입됨):
        model           : 사용할 모델. "primary"(Qwen 3.5 27B) 또는 "auxiliary"(ExaOne 7.8B).
        permission_mode : 권한 모드. default/auto/plan/trust/bypass 중 하나.
                          도구 실행을 얼마나 자동 허용할지를 결정한다.
        resume          : 이어서 진행할 이전 세션 ID. None이면 새 세션으로 시작한다.
        log_level       : 채팅 화면에 노출할 nexus.* 로그의 최소 레벨(임계값).

    반환:
        없음(None). 내부에서 asyncio.run()으로 REPL이 끝날 때까지 블로킹한다.

    호출 대상:
        cli.repl.NexusREPL — REPL 본체. 여기에 위 설정들을 그대로 넘긴다.
    """
    # v0.14.12 — 채팅 중 nexus.* INFO 로그(라우팅/RAG 주입 등)가 출력에 섞여
    # 거슬리는 문제 해결. 기본 WARNING으로 낮춰 채팅 화면을 깨끗이 유지하고,
    # 디버깅이 필요할 때만 --log-level INFO/DEBUG로 끌어올린다.
    # 실제 적용은 NexusREPL._apply_log_level()이 부트스트랩 직후 한 번 더 강제
    # (bootstrap._configure_logging이 INFO로 reset하기 때문).

    # NexusREPL은 무거운 의존성을 함께 끌고 오므로, chat 명령을 실제로 실행할 때만
    # lazy import 한다. (version/health 같은 가벼운 명령의 기동 비용을 낮추기 위함)
    from cli.repl import NexusREPL

    # CLI에서 받은 옵션들을 그대로 REPL 생성자에 전달한다.
    # resume 옵션 이름은 REPL 쪽 매개변수명(resume_session_id)에 맞춰 매핑한다.
    repl = NexusREPL(
        permission_mode=permission_mode,
        model=model,
        resume_session_id=resume,
        log_level=log_level,
    )
    # REPL의 비동기 실행 루프를 시작한다. 사용자가 종료할 때까지 이 줄에서 대기한다.
    asyncio.run(repl.run())


@cli.command()
def version() -> None:
    """
    버전 및 환경 정보를 표 형태로 표시한다 (`nexus version`).

    Nexus 자체 버전과 실행 중인 Python 버전을 먼저 보여주고, 이어서 설정 파일에서
    읽어온 모델·GPU 서버·에어갭 모드 정보를 덧붙인다. 설정 로드에 실패해도 명령이
    죽지 않도록, 실패 시에는 "로드 실패" 한 줄만 표시하고 넘어간다(그레이스풀 폴백).

    출력은 rich 라이브러리의 Table로 예쁘게 렌더링한다. 반환값은 없다.
    """
    # 표 출력용 rich 컴포넌트는 이 명령에서만 필요하므로 지역 import.
    from rich.console import Console
    from rich.table import Table

    # Console: 터미널 출력 담당 / Table: 항목-값 2열 표를 구성.
    console = Console()
    table = Table(title="Project Nexus 버전 정보", border_style="blue")
    table.add_column("항목", style="cyan")
    table.add_column("값", style="white")

    # 설정이 없어도 항상 보여줄 수 있는 기본 정보 두 줄.
    table.add_row("Nexus 버전", "0.1.0")
    # sys.version 예: "3.11.5 (main, ...)". split()[0]으로 "3.11.5" 부분만 뽑는다.
    table.add_row("Python 버전", sys.version.split()[0])

    # 설정에서 모델/서버 정보를 가져온다. 설정 파일이 없거나 검증 실패할 수 있으므로
    # try/except로 감싸고, 실패 시에도 명령 전체가 실패하지 않게 한다.
    try:
        from core.config import load_and_validate_config

        # YAML 설정을 로드 + Pydantic 검증까지 수행한 config 객체를 돌려받는다.
        config = load_and_validate_config()
        table.add_row("Primary 모델", config.model.primary_model)
        table.add_row("Auxiliary 모델", config.model.auxiliary_model)
        table.add_row("GPU 서버", config.gpu_server_url)
        table.add_row("에어갭 모드", str(config.air_gap_mode))
    except Exception:
        # 설정 로드 실패 시 기본 정보만 표시한다 (버전 확인 자체는 계속 가능하도록).
        table.add_row("설정", "로드 실패")

    # 완성된 표를 터미널에 출력한다.
    console.print(table)


@cli.command()
@click.option("--limit", default=20, help="표시할 최근 세션 수")
def sessions(limit: int) -> None:
    """
    저장된 대화 세션 목록을 보여준다 (`nexus sessions`).

    `chat --resume <세션ID>`로 이어받을 수 있도록, 최근 트랜스크립트 세션의
    ID·마지막 수정 시각·발화 수·첫 질문 미리보기를 표로 나열한다. 세션 ID를
    복사해 `nexus chat --resume <ID>`에 넣으면 그 대화가 복원된다.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    try:
        from core.config import load_and_validate_config
        from core.memory.transcript import list_transcript_sessions

        config = load_and_validate_config()
        rows = list_transcript_sessions(config.sessions_dir, limit=limit)
    except Exception as e:
        console.print(f"[red]세션 목록을 불러오지 못했습니다: {e}[/red]")
        return

    if not rows:
        console.print("[yellow]저장된 세션이 없습니다.[/yellow]")
        return

    table = Table(title="대화 세션", border_style="blue")
    table.add_column("세션 ID", style="cyan", no_wrap=True)
    table.add_column("마지막 수정", style="dim")
    table.add_column("발화", justify="right")
    table.add_column("첫 질문", style="white")
    for r in rows:
        table.add_row(
            r.get("session_id", "?"),
            (r.get("last_modified") or "")[:19].replace("T", " "),
            str(r.get("entries", 0)),
            (r.get("title_hint") or "")[:50],
        )
    console.print(table)
    console.print(
        "[dim]이어받기: [/dim][cyan]nexus chat --resume <세션 ID>[/cyan]"
    )


@cli.command()
@click.argument("query")
@click.option(
    "--model",
    default="primary",
    type=click.Choice(["primary", "auxiliary"]),
    help="사용할 모델",
)
@click.option(
    "--log-level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help=(
        "로그 표시 레벨 — 기본 WARNING. 비대화형 모드라 INFO 로그가 답변 텍스트와 "
        "섞이면 파이프·스크립트에서 파싱이 깨지므로 기본값을 조용하게 둔다."
    ),
)
def ask(query: str, model: str, log_level: str) -> None:
    """
    단일 질문을 보낸다 — 비대화형 1회성 모드 (`nexus ask "<질문>"`).

    역할:
        REPL을 띄우지 않고, 질문 한 번을 QueryEngine에 던진 뒤 스트리밍 답변을
        표준출력에 흘려보내고 즉시 종료한다. 셸 스크립트, 파이프라인, cron 등
        "대화가 필요 없는 자동화" 상황에서 쓰기 좋은 모드다.

    매개변수:
        query : 사용자가 넘긴 질문 문자열 (@click.argument 로 위치 인자).
        model : 사용할 모델 별칭("primary"/"auxiliary").

    흐름:
        부트스트랩(init → init_phase2)으로 엔진을 준비 → submit_message()가 내보내는
        StreamEvent를 순회 → 텍스트 조각(TEXT_DELTA)만 골라 화면에 이어붙여 출력.
        어느 단계든 실패하면 붉은 메시지를 찍고 종료 코드 1로 빠져나간다.

    반환:
        없음(None). 실패 시 sys.exit(1)로 프로세스를 종료한다.
    """
    from rich.console import Console

    console = Console()

    async def _run_ask():
        """
        실제 비동기 질문 처리를 담당하는 내부 코루틴.

        asyncio 이벤트 루프 안에서 돌아야 하므로 async로 정의하고,
        바깥에서 asyncio.run(_run_ask())으로 실행한다.
        """
        # 1) 부트스트랩: 전역 상태(init) → Phase2 컴포넌트(init_phase2) 초기화.
        #    여기서 QueryEngine을 포함한 핵심 컴포넌트들이 조립된다.
        try:
            from core.bootstrap import init, init_phase2

            state = await init()
            components = await init_phase2(state)
            engine = components.get("query_engine")
        except Exception as e:
            # 부트스트랩 도중 예외 → 사용자에게 알리고 실패 종료.
            console.print(f"[red]부트스트랩 실패: {e}[/red]")
            sys.exit(1)

        # bootstrap의 _configure_logging이 nexus.* 를 INFO로 되돌려 놓으므로,
        # 부트스트랩 직후 사용자가 지정한 레벨을 다시 씌운다(repl._apply_log_level 대칭).
        # 이렇게 하지 않으면 INFO 로그가 답변 텍스트 중간에 섞여 나온다.
        import logging as _logging

        _logging.getLogger("nexus").setLevel(
            getattr(_logging, log_level.upper(), _logging.WARNING)
        )

        # 2) 부트스트랩은 됐지만 엔진이 안 잡힌 경우(컴포넌트 누락)도 방어.
        if engine is None:
            console.print("[yellow]QueryEngine 초기화 실패[/yellow]")
            sys.exit(1)

        # 3) QueryEngine에 메시지를 보내고 텍스트 응답을 출력한다.
        #    submit_message()는 4-Tier 체인을 통과하며 StreamEvent를 순차적으로 yield한다.
        from core.message import StreamEvent, StreamEventType

        async for event in engine.submit_message(query):
            # 여러 종류의 StreamEvent 중, 실제 답변 텍스트 조각(TEXT_DELTA)만 골라 출력한다.
            # 도구 실행/사고 과정 등 다른 이벤트는 비대화형 모드에서 화면에 찍지 않는다.
            if (
                isinstance(event, StreamEvent)
                and event.type == StreamEventType.TEXT_DELTA
                and event.text
            ):
                # end="" 로 개행 없이 조각들을 이어붙여, 스트리밍이 자연스럽게 흐르도록 한다.
                console.print(event.text, end="")

    # 위 코루틴을 이벤트 루프에서 실행한다.
    asyncio.run(_run_ask())


@cli.command()
def health() -> None:
    """
    GPU 서버(Machine B)의 상태를 확인한다 (`nexus health`).

    설정에서 GPU 서버 URL을 읽어 `{url}/health` 로 HTTP GET을 보내고, 200 응답이면
    서버가 돌려준 상태 JSON을 표로 펼쳐 보여준다. 200이 아니거나 연결 자체가 안 되면
    그에 맞는 안내 메시지를 출력한다. 설정 로드에 실패하면 localhost:8000을 기본값으로 쓴다.

    반환값은 없으며, 네트워크 호출은 LAN 내부(에어갭 규칙 준수)로만 이뤄진다.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    async def _check_health():
        """
        실제 헬스체크를 수행하는 내부 코루틴.

        httpx 비동기 클라이언트로 GPU 서버 /health 를 호출하므로 async로 정의하고,
        바깥에서 asyncio.run()으로 실행한다.
        """
        # 1) 설정에서 GPU 서버 URL을 읽는다. 설정 로드 실패 시에는 개발용 기본값으로 폴백.
        try:
            from core.config import load_and_validate_config

            config = load_and_validate_config()
            gpu_url = config.gpu_server_url
        except Exception:
            # 설정을 못 읽어도 헬스체크는 시도할 수 있도록 로컬 기본 주소 사용.
            gpu_url = "http://localhost:8000"

        # 2) GPU 서버 /health 엔드포인트에 HTTP GET 요청.
        try:
            import httpx

            # timeout=10초: 서버가 죽었을 때 무한 대기하지 않도록 상한을 둔다.
            # async with 로 클라이언트를 열고, 블록을 벗어나면 자동으로 닫힌다.
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{gpu_url}/health")
                if resp.status_code == 200:
                    # 200이면 서버는 정상이다. 본문 파싱 실패가 "헬스체크 실패"가 되면 안 된다.
                    #   vLLM의 /health는 **본문이 빈 200**을 돌려준다(content-type 없음).
                    #   기존 코드는 resp.json()을 무조건 불러 JSONDecodeError로 떨어졌고,
                    #   그래서 vLLM 환경에서는 `nexus health`가 항상 실패했다.
                    try:
                        data = resp.json()
                    except ValueError:
                        data = {}
                    table = Table(title="GPU 서버 상태", border_style="green")
                    table.add_column("항목", style="cyan")
                    table.add_column("값", style="white")
                    table.add_row("URL", gpu_url)
                    table.add_row("상태", "[green]정상[/green]")
                    if isinstance(data, dict) and data:
                        # 서버가 상세 JSON을 준 경우에만 키/값을 펼친다.
                        for key, value in data.items():
                            table.add_row(key, str(value))
                    else:
                        # vLLM처럼 본문이 없는 경우 — 그 사실을 있는 그대로 알린다.
                        table.add_row("응답 본문", "(없음 — vLLM /health는 빈 200을 반환)")
                    console.print(table)
                else:
                    # 서버가 응답은 했지만 200이 아님(예: 503) → 상태 코드를 알려준다.
                    console.print(f"[red]GPU 서버 응답 이상: HTTP {resp.status_code}[/red]")
        except httpx.ConnectError:
            # 연결 자체가 실패(서버 미기동/네트워크 문제) → 원인 파악용 안내 메시지.
            console.print(
                f"[red]GPU 서버에 연결할 수 없습니다: {gpu_url}\n"
                f"서버가 실행 중인지 확인하세요.[/red]"
            )
        except Exception as e:
            # 그 밖의 예기치 못한 오류는 메시지로 노출해 디버깅을 돕는다.
            console.print(f"[red]헬스체크 실패: {e}[/red]")

    # 위 코루틴을 이벤트 루프에서 실행한다.
    asyncio.run(_check_health())


# 이 파일을 `python -m cli.commands` 처럼 직접 실행했을 때 Click 그룹을 기동한다.
# (평소에는 패키지의 콘솔 스크립트 진입점으로 cli 그룹이 호출된다.)
if __name__ == "__main__":
    cli()
