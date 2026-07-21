"""
Rich REPL — 터미널 기반 대화형 인터페이스 (Nexus CLI의 얼굴).

이 파일은 사용자가 터미널에서 Nexus와 직접 대화하는 진입 화면을 담당한다.
Claude Code의 REPL.ts를 파이썬 생태계(Rich + prompt-toolkit)로 재구현한 것으로,
전체 흐름은 아래 한 줄로 요약된다:

    사용자 입력 → QueryEngine.submit_message() → StreamEvent 스트리밍 출력

즉, 이 파일 자체는 "추론"을 하지 않는다. 실제 LLM 오케스트레이션은 core/의
QueryEngine이 담당하고, 이 파일은 (1) 입력을 받아 넘기고 (2) 돌아오는
StreamEvent를 사람이 보기 좋게 터미널에 그려주는 표현(presentation) 계층이다.

── 구성 요소 ──
  - NexusREPL 클래스: REPL의 모든 상태와 동작을 담은 메인 클래스.
      · run()              : 메인 입력 루프 (부트스트랩 → 배너 → while 루프)
      · _bootstrap()       : Phase 1/2 초기화로 QueryEngine을 준비
      · _process_message() : 한 번의 사용자 메시지를 QueryEngine에 흘려보내고
                             spinner(진행 표시)와 함께 스트리밍 출력
      · display_stream_event() : StreamEvent 한 개를 실제로 화면에 그림
      · _cmd_* 핸들러들    : /help, /model 등 슬래시 명령어 처리
  - _IDINO_MARK          : 배너 좌측에 그려지는 IDINO 픽셀 로고 문자열
  - main()               : pyproject.toml의 `nexus` 콘솔 스크립트 진입점

── 의존성 방향 ──
  cli/ → core/ (단방향). 이 파일은 core.bootstrap(초기화)과
  core.message(StreamEvent/StreamEventType 타입)만 core에서 가져오고,
  실제 로직은 QueryEngine 인스턴스를 통해 호출한다. core가 cli를 import하는
  역방향은 절대 없어야 한다(아키텍처 규칙 P2).

── 외부에서 이 파일을 호출하는 경로 ──
  콘솔에서 `nexus` 명령 → main() → NexusREPL().run(). 별도의 웹/HTTP API를
  노출하지는 않는다(웹 진입점은 web/ 패키지에 따로 존재).

작성자: 이현수 / 작성일: 2026-07-05
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

# 모듈 전용 로거. 프로젝트 규칙상 "nexus.{모듈경로}" 네임스페이스를 사용한다.
# 이렇게 하면 logging.getLogger("nexus")로 상위 레벨을 한 번에 조절할 수 있고,
# 아래 _apply_log_level()이 바로 그 상위 로거의 레벨을 바꿔 하위에 전파시킨다.
logger = logging.getLogger("nexus.cli.repl")

# ─── 버전 정보 ───
# 배너와 세션 요약 등에 노출되는 REPL 버전 문자열. 릴리스 시 갱신한다.
__version__ = "0.1.0"

# ─── IDINO 회사 마크 (블록 ASCII 아트) ───
# 배너 좌측에 배치되는 IDINO 로고. 사용자 제공 원본 디자인을 정확히 유지한다.
# 블록(████)은 4칸씩, 블록 사이 공백도 4칸 — 가로 폭 20칸의 정사각 비율.
# 이 비율을 어기면 'i'/'d' 글자 형태가 깨지므로 변형 금지.
# (파이썬 여러 줄 문자열이라 줄 끝의 백슬래시 "\"는 불필요한 개행을 제거하기 위한
#  것 — 문자열 시작/끝에 빈 줄이 들어가지 않도록 맞춰 둔 것이다.)
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
    Rich 기반 터미널 REPL — 대화 세션 하나를 처음부터 끝까지 관리하는 클래스.

    역할:
      메인 루프(run)에서 사용자 입력을 한 줄씩 받아 QueryEngine에 전달하고,
      QueryEngine이 AsyncGenerator로 돌려주는 StreamEvent들을 실시간으로
      터미널에 그린다. 슬래시 명령어(/help 등)는 QueryEngine을 거치지 않고
      REPL 내부에서 직접 처리한다.

    왜 클래스인가:
      배너 설정, 권한 모드, 선택 모델, 로그 레벨, prompt-toolkit 세션,
      부트스트랩으로 만들어진 GlobalState/QueryEngine 등 "세션 동안 살아있는
      상태"가 많다. 이를 인스턴스 필드로 묶어두면 각 핸들러가 self로 공유한다.

    핵심 흐름(생명주기):
      __init__ (상태 준비) → run() 안에서
        _bootstrap()       : core 초기화 → self._query_engine 확보
        _display_banner()  : 시작 화면 출력
        while 루프         : 입력 → (명령어면 _cmd_*) 또는 _process_message()
        _shutdown()        : 세션 요약 출력 후 종료

    주의:
      이 클래스는 core를 "사용"만 한다. GPU/모델을 직접 호출하지 않으며,
      모든 추론은 self._query_engine.submit_message()를 통해서만 일어난다.
    """

    def __init__(
        self,
        permission_mode: str = "default",
        model: str = "primary",
        resume_session_id: str | None = None,
        log_level: str = "WARNING",
    ):
        """
        REPL 인스턴스를 초기화한다 (아직 core 부트스트랩은 하지 않는다).

        여기서는 "네트워크/파일에 접근하지 않는" 순수 상태 준비만 한다.
        실제 QueryEngine 준비는 run() → _bootstrap()에서 비동기로 이뤄진다.
        생성자를 가볍게 유지해야 테스트에서 인스턴스를 쉽게 만들 수 있다.

        Args:
            permission_mode: 권한 모드 (default, auto, plan, trust, bypass).
                도구 실행을 얼마나 자동 허용할지를 결정한다. 배너/세션 표시용으로도 쓰인다.
            model: 사용할 모델 별칭 (primary, auxiliary). /model 명령으로 바꿀 수 있다.
            resume_session_id: 이어서 할 세션 ID (None이면 새 세션으로 시작).
            log_level: 채팅 화면에 표시할 nexus.* 로그 레벨 (v0.14.12에서 추가).
                기본 WARNING — 채팅 흐름 사이에 운영용 INFO 로그가 끼어들어
                대화가 지저분해지는 것을 막는다.
                (문제 상황) bootstrap의 _configure_logging이 nexus.* 레벨을
                강제로 INFO로 되돌려 놓는다. 그래서 부트스트랩이 끝난 직후
                _apply_log_level()을 한 번 더 호출해 사용자가 원한 레벨을
                다시 씌운다.
        """
        # Rich 콘솔 — 모든 출력(배너/패널/텍스트/스피너)이 이 객체를 통해 나간다.
        self.console = Console()
        # 이벤트 → Rich 렌더러블 변환기. show_thinking=False면 생각(thinking) 블록은
        # 숨긴다. /thinking 명령으로 이 플래그를 토글한다.
        self._formatter = OutputFormatter(show_thinking=False)
        self._permission_mode = permission_mode
        self._model = model
        self._resume_session_id = resume_session_id
        self._log_level = log_level
        # 메인 while 루프의 실행 여부 플래그. /exit가 False로 바꿔 루프를 끝낸다.
        self._running = False

        # prompt-toolkit 세션 — 방향키로 이전 입력 재호출(히스토리)과
        # 멀티라인 편집을 지원한다. InMemoryHistory라 프로세스 종료 시 사라진다.
        self._prompt_session: PromptSession = PromptSession(
            history=InMemoryHistory(),
        )

        # 세션 명령어 맵 — "/로 시작하는 입력"을 어떤 핸들러로 보낼지 정의한다.
        # run() 루프가 입력 첫 토큰을 이 맵의 키와 비교해 매칭되면 그 핸들러를 부른다.
        self._session_commands: dict[str, Any] = {
            "/help": self._cmd_help,
            "/clear": self._cmd_clear,
            "/exit": self._cmd_exit,
            "/model": self._cmd_model,
            "/config": self._cmd_config,
            "/session": self._cmd_session,
            "/thinking": self._cmd_thinking,
        }

        # ── 부트스트랩 후에 채워지는 상태 변수 ──
        # 지금은 None이고, _bootstrap()이 성공하면 실제 객체가 들어간다.
        # None으로 시작하는 이유: 부트스트랩이 실패해도 REPL 껍데기는 떠서
        # 안내 메시지라도 보여줄 수 있도록(부분 실패 허용).
        self._state: Any = None  # GlobalState — 설정/세션/모델 상태 등 전역 컨테이너
        self._query_engine: Any = None  # QueryEngine — 실제 대화 오케스트레이터(Tier 1)

    # ─── 메인 루프 ───

    async def run(self) -> None:
        """
        메인 REPL 루프 — REPL이 살아있는 동안 계속 도는 심장부.

        전체 순서:
          1. 부트스트랩 (core 초기화 → QueryEngine 준비)
          2. 시작 배너 표시
          3. 입력 루프(while):
             - 슬래시 명령어이면 해당 _cmd_* 핸들러 실행
             - 일반 텍스트이면 _process_message()로 QueryEngine에 넘겨 스트리밍
          4. 루프를 빠져나오면 종료 처리(_shutdown)

        키 입력 처리:
          - Ctrl+C(KeyboardInterrupt) → "지금 처리 중인 요청"만 취소하고 루프는 계속
          - Ctrl+D(EOFError) → 루프를 완전히 빠져나가 세션 종료

        반환값: 없음(None). 종료 시점까지 블로킹된다.
        """
        # ① 부트스트랩: core.bootstrap을 호출해 GlobalState/QueryEngine을 만든다.
        await self._bootstrap()

        # ② 배너 표시: 회사 로고 + 모델/인프라/세션 정보 패널을 출력한다.
        self._display_banner()

        # ③ 입력 루프 시작. _running이 False가 되면(=/exit) 루프가 끝난다.
        self._running = True
        while self._running:
            try:
                # prompt-toolkit의 prompt()는 동기(블로킹) 함수다. 그대로 await하면
                # 이벤트 루프가 멈춰 스트리밍/타이머가 막히므로, run_in_executor로
                # 별도 스레드에서 입력을 기다리게 하고 그 완료를 await한다.
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._prompt_session.prompt("nova> "),
                )

                # 공백만 입력한 경우는 아무 것도 하지 않고 다음 입력을 기다린다.
                if not user_input.strip():
                    continue

                # 입력의 첫 토큰(공백 기준 첫 단어)을 소문자로 만들어 명령어인지 확인.
                # 예: "/model auxiliary" → command_key = "/model"
                command_key = user_input.strip().split()[0].lower()
                if command_key in self._session_commands:
                    # 명령어라면 첫 토큰을 뺀 나머지를 인자 리스트로 만들어 핸들러에 전달.
                    # 예: "/model auxiliary" → args = ["auxiliary"]
                    args = user_input.strip().split()[1:]
                    await self._session_commands[command_key](args)
                    continue

                # 명령어가 아니면 일반 대화 메시지로 간주하고 QueryEngine에 넘긴다.
                await self._process_message(user_input.strip())

            except KeyboardInterrupt:
                # Ctrl+C → 방금 입력/처리 중이던 요청만 취소. REPL 자체는 살아있다.
                self.console.print("\n[yellow]요청 취소됨[/yellow]")
                continue
            except EOFError:
                # Ctrl+D(입력 스트림 종료) → 루프를 벗어나 정상 종료 절차로 간다.
                self.console.print("\n[dim]세션을 종료합니다.[/dim]")
                break

        # ④ 종료 처리: 세션 요약을 보여주고 마무리 인사를 출력한다.
        await self._shutdown()

    # ─── 부트스트랩 ───

    async def _bootstrap(self) -> None:
        """
        core를 2단계(Phase 1 + Phase 2)로 초기화해 QueryEngine을 준비한다.

        Phase 1: GlobalState와 설정(config)을 로딩한다(환경에 크게 의존하지 않는 초기화).
        Phase 2: ToolRegistry, MemoryManager, QueryEngine 등 실제 대화에 필요한
                 무거운 컴포넌트를 만든다. 반환된 dict에서 query_engine을 꺼내 보관한다.

        실패 정책(fail-soft):
          부트스트랩이 어떤 이유로든 실패해도 예외를 위로 던지지 않는다. 대신
          self._state를 None으로 두어 REPL 껍데기만 뜨게 한다. 이렇게 하면
          _process_message()가 "아직 초기화되지 않았다"는 안내를 보여줄 수 있어,
          완전히 죽는 것보다 디버깅에 유리하다.
        """
        try:
            # import를 함수 안에서 하는 이유: core.bootstrap이 무겁고, 순환 import
            # 위험을 피하기 위해 실제로 필요한 시점에 지연(lazy) 로딩한다.
            from core.bootstrap import init, init_phase2

            # Phase 1: 설정/전역 상태 로딩.
            self._state = await init()

            # Phase 2: 도구 레지스트리 + 메모리 + QueryEngine 구성.
            # 반환은 컴포넌트 dict이며, 이 중 query_engine만 REPL이 직접 쓴다.
            components = await init_phase2(self._state)
            self._query_engine = components.get("query_engine")
            logger.info("REPL 부트스트랩 완료 (Phase 1 + 2)")
        except Exception as e:
            # 구체 예외를 특정하기 어려운 최상위 초기화 단계라 광범위하게 잡되,
            # 조용히 삼키지 않고 경고 로그로 남긴 뒤 기본 모드로 계속 진행한다.
            logger.warning(f"부트스트랩 실패, 기본 모드로 시작: {e}")
            self._state = None
        finally:
            # v0.14.12 — bootstrap 내부의 _configure_logging이 nexus.* 로거 레벨을
            # INFO로 되돌려 놓기 때문에, 성공/실패와 무관하게(finally) 사용자가 지정한
            # log_level을 여기서 한 번 더 덮어써서 채팅 화면을 깔끔하게 유지한다.
            self._apply_log_level()

    def _apply_log_level(self) -> None:
        """nexus.* 루트 로거의 레벨을 self._log_level 문자열대로 설정한다.

        왜 필요한가:
          대화 화면에 운영용 INFO 로그가 흘러 대화가 지저분해지는 것을 막기 위해,
          부트스트랩 직후(채팅 시작 직전) 이 함수를 불러 원하는 레벨을 강제한다.
          "nexus" 상위 로거 하나만 바꿔도 "nexus.cli.repl" 등 하위로 전파된다.
        """
        try:
            # 문자열("WARNING")을 logging 모듈의 상수(logging.WARNING)로 변환.
            # 잘못된 문자열이면 getattr의 기본값 logging.WARNING으로 안전하게 폴백.
            level = getattr(logging, self._log_level.upper(), logging.WARNING)
            logging.getLogger("nexus").setLevel(level)
        except Exception as e:  # noqa: BLE001
            # 로그 레벨 조정 실패가 REPL 실행 자체를 막아서는 안 되므로 광범위하게
            # 잡아 경고만 남긴다. (noqa: BLE001 — 의도적 광범위 except임을 린터에 명시)
            logger.warning("로그 레벨 적용 실패: %s", e)

    # ─── 배너 표시 ───

    def _display_banner(self) -> None:
        """시작 배너 + 도움말 힌트를 화면에 출력한다 (v0.14.10 IDINO 코퍼레이트).

        레이아웃:
          하나의 둥근 패널(Panel) 안에 2열 grid를 넣는다.
            · 좌측 열: IDINO 픽셀 로고(_IDINO_MARK)
            · 우측 열: 회사 카드 + 모델 라우팅/인프라/세션 정보
          패널 아래에 슬래시 커맨드와 단축키 힌트 두 줄을 덧붙인다.

        모델 표시 규칙:
          config.routing.enabled가 True면 CHAT/KNOWLEDGE/TOOL 세 분기 각각에
          "실제 served-model-name"(예: qwen3.5-27b)을 보여준다. 별칭("primary")만
          보이면 어떤 모델이 도는지 헷갈리므로, 실모델명을 노출해 모호성을 없앤다.
          라우팅이 꺼져 있으면 대신 primary/auxiliary 모델을 표시한다.

        구현 메모:
          self._state가 None(부트스트랩 실패)일 수 있으므로, 아래 정보 조회는
          전부 getattr(..., 기본값) 패턴으로 방어적으로 접근한다. 즉 값이 없어도
          "?"나 "-" 같은 자리표시자로 대체되어 배너가 깨지지 않는다.
        """
        # ───── 좌측: IDINO 픽셀 마크 ─────
        # IDINO 코퍼레이트 컬러 rgb(0,71,157) — 배지/홈페이지/이미지와 색을 맞춘다.
        mark = Text(_IDINO_MARK, style="bold rgb(0,71,157)")

        # ───── 우측: 회사 카드 + 시스템 정보 ─────
        # Rich의 Text 객체에 append로 조각조각 스타일을 입혀 한 덩어리로 쌓는다.
        info = Text()
        info.append("IDINO NOVA", style="bold rgb(0,71,157)")
        info.append(f"  v{__version__}\n", style="dim white")
        info.append("에어갭 로컬 LLM 오케스트레이션 플랫폼\n", style="white")
        info.append("Powered by ", style="dim white")
        info.append("IDINO Corp.", style="bold rgb(0,71,157)")
        info.append("  ·  Air-gapped AI for Enterprise\n", style="dim white")

        # 회사 카드와 시스템 정보 사이를 나누는 가로 구분선.
        info.append("─" * 44 + "\n", style="dim rgb(0,71,157)")

        # 라우팅 설정을 안전하게 꺼낸다: state.config.routing 경로 중 하나라도
        # 없으면 None이 된다(중첩 getattr 방어 패턴).
        routing = getattr(getattr(self._state, "config", None), "routing", None)
        # 라우팅이 존재하고 켜져 있으면 → 질의 유형별(CHAT/KNOWLEDGE/TOOL) 실모델명 표시.
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
            # 라우팅 비활성 — 단순히 primary/auxiliary 두 모델만 표시한다.
            model_cfg = getattr(
                getattr(self._state, "config", None), "model", None
            )
            primary = getattr(model_cfg, "primary_model", self._model)
            aux = getattr(model_cfg, "auxiliary_model", "-")
            info.append("모델\n", style="bold white")
            info.append(f"  Primary    {primary}\n", style="cyan")
            info.append(f"  Auxiliary  {aux}\n", style="dim")

        # 인프라 정보(Worker=GPU 추론 서버, Scout=경량 라우팅/사전판단 서버) 표시.
        # 두 URL 모두 없을 수 있으므로 기본값("(?)", "-")으로 방어한다.
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

        # 세션 컨텍스트 — 현재 권한 모드와, 새 세션인지/이어받은 세션인지 표시.
        info.append("세션\n", style="bold white")
        info.append(f"  권한 모드   {self._permission_mode}\n", style="dim")
        if self._resume_session_id:
            info.append(f"  복원 세션   {self._resume_session_id}\n", style="dim")
        else:
            info.append("  복원 세션   (신규)\n", style="dim")

        # ───── 좌·우 열을 하나의 grid로 합쳐 나란히 배치 ─────
        # Table.grid는 테두리 없는 표. padding=(0,4)로 두 열 사이 좌우 간격을 준다.
        layout = Table.grid(padding=(0, 4))
        layout.add_column(justify="left", vertical="top")
        layout.add_column(justify="left", vertical="top")
        layout.add_row(mark, info)  # 왼쪽=로고, 오른쪽=정보

        # Claude Code 스타일 패널 — 둥근 모서리(ROUNDED) + 좌측 정렬 타이틀로 감싼다.
        self.console.print(
            Panel(
                layout,
                title="[bold]✻ Welcome to IDINO NOVA[/bold]",
                title_align="left",
                subtitle="[dim]IDINO Corp. · 2026[/dim]",
                subtitle_align="right",
                border_style="rgb(0,71,157)",
                box=ROUNDED,
                padding=(1, 2),
                expand=False,
            )
        )
        # 패널 아래 첫 번째 힌트 줄 — 사용 가능한 슬래시 커맨드 목록.
        self.console.print(
            "[dim]✻ 슬래시 커맨드: [/dim]"
            "[cyan]/help[/cyan][dim] · [/dim]"
            "[cyan]/model[/cyan][dim] · [/dim]"
            "[cyan]/config[/cyan][dim] · [/dim]"
            "[cyan]/session[/cyan][dim] · [/dim]"
            "[cyan]/clear[/cyan][dim] · [/dim]"
            "[cyan]/exit[/cyan]"
        )
        # 두 번째 힌트 줄 — 키보드 단축키 안내(Enter/Ctrl+C/Ctrl+D).
        self.console.print(
            "[dim]   단축키:    [/dim]"
            "[white]Enter[/white][dim] 전송 · [/dim]"
            "[white]Ctrl+C[/white][dim] 요청 취소 · [/dim]"
            "[white]Ctrl+D[/white][dim] 종료[/dim]\n"
        )

    # ─── 메시지 처리 ───

    async def _process_message(self, user_input: str) -> None:
        """
        사용자 메시지 한 개를 QueryEngine에 넘기고, 돌아오는 이벤트를 스트리밍 출력한다.

        이 메서드가 REPL과 core를 잇는 핵심 다리다. 흐름은 다음과 같다:
          1) QueryEngine이 준비됐는지 확인(없으면 안내만 하고 종료).
          2) "요청 분석 중..." 스피너를 띄운다(cold start 동안 사용자를 안심시킴).
          3) submit_message()가 yield하는 이벤트마다:
             - _stage_label_for()로 스피너를 켤지/끌지/문구를 바꿀지 판단,
             - display_stream_event()로 실제 화면 출력.
          4) 취소(Ctrl+C)·예외·정상 종료 어느 경우든 스피너를 반드시 닫는다.

        Args:
            user_input: 사용자가 입력한 순수 텍스트(슬래시 명령어가 아닌 일반 대화).
        """
        if self._query_engine is None:
            # 부트스트랩이 QueryEngine을 만들지 못한 경우. 조용히 실패하지 않고
            # 왜 응답이 안 나오는지 사용자에게 안내 패널로 알려준다.
            self.console.print(
                Panel(
                    "[yellow]QueryEngine이 아직 초기화되지 않았습니다.\n"
                    "Phase 3 (Orchestrator) 모듈이 완성되면 자동으로 연동됩니다.[/yellow]",
                    title="[bold yellow]안내[/bold yellow]",
                    border_style="yellow",
                )
            )
            return

        # ── 스트리밍 + 진행 스피너 ──
        # QueryEngine.submit_message()는 AsyncGenerator로 이벤트를 하나씩 흘려준다.
        # v0.14.8부터 Console.status() 스피너로 "지금 무슨 단계인지"를 시각화한다.
        # 아이디어:
        #   · 처음엔 "요청 분석 중..." 스피너를 띄운다.
        #   · 첫 본문 토큰(TEXT_DELTA)이 오면 스피너를 닫고 본문을 print로 흘린다.
        #   · 도구 호출이 시작되면 다시 스피너로 전환해 "X 실행 중..."을 보여준다.
        # 이렇게 하면 cold start로 60초가 걸려도 사용자는 "멈춘 게 아니라 분석 중"으로 인지한다.
        #
        # status_ctx는 컨텍스트 매니저지만, `async for` 도중 동적으로 열고 닫아야 해서
        # with 문 대신 __enter__/__exit__를 직접 호출하며 status_active 플래그로
        # "지금 스피너가 떠 있는가"를 추적한다.
        status_ctx = self.console.status(
            "[cyan]요청 분석 중...[/cyan]", spinner="dots"
        )
        status_ctx.__enter__()
        status_active = True
        try:
            async for event in self._query_engine.submit_message(user_input):
                # 이 이벤트를 근거로 스피너를 어떻게 할지 결정한다.
                # 반환 규약은 _stage_label_for()의 docstring 참고.
                next_label = self._stage_label_for(event)
                if next_label == "_TEXT_":
                    # 첫 본문 토큰 도착 → 스피너를 닫고 텍스트 출력 모드로 전환.
                    if status_active:
                        status_ctx.__exit__(None, None, None)
                        status_active = False
                elif next_label == "_TOOL_DONE_":
                    # 도구 실행 완료 → 곧 도구 결과 Panel을 출력해야 하므로 스피너를 닫는다.
                    # (스피너와 Panel이 동시에 활성이면 화면이 겹쳐 Panel이 깨져 보인다.)
                    # 다음 이벤트(다음 LLM 응답·다음 도구 호출)가 오면 아래
                    # `elif next_label is not None` 분기가 스피너를 자동으로 다시 연다.
                    if status_active:
                        status_ctx.__exit__(None, None, None)
                        status_active = False
                elif next_label is not None:
                    # 스피너 문구를 갱신해야 하는 단계 이벤트(TURN_START/TOOL_USE_START 등).
                    if status_active:
                        # 이미 스피너가 떠 있으면 문구만 바꾼다.
                        status_ctx.update(next_label)
                    else:
                        # 스피너가 닫혀 있었다면(예: 본문 출력 뒤 도구 호출 시작) 새로 연다.
                        status_ctx = self.console.status(
                            next_label, spinner="dots"
                        )
                        status_ctx.__enter__()
                        status_active = True

                # 스피너 상태 판단과 별개로, 이벤트 자체는 항상 화면에 그린다.
                self.display_stream_event(event)
        except asyncio.CancelledError:
            # Ctrl+C 등으로 스트림이 취소된 경우. 스피너를 닫고 취소 안내를 남긴다.
            if status_active:
                status_ctx.__exit__(None, None, None)
                status_active = False
            self.console.print("[yellow]요청이 취소되었습니다.[/yellow]")
        except Exception as e:
            # 그 밖의 예외는 포매터를 통해 사용자 친화적 에러 메시지로 출력한다.
            if status_active:
                status_ctx.__exit__(None, None, None)
                status_active = False
            self.console.print(self._formatter.format_error(str(e)))
        finally:
            # 어떤 경로로 끝나든(정상/취소/예외) 스피너가 남아있지 않도록 최종 안전 장치.
            # 예: TEXT_DELTA가 한 번도 안 와서 스피너가 계속 떠 있는 상태로 끝난 경우.
            if status_active:
                status_ctx.__exit__(None, None, None)

    def _stage_label_for(self, event: Any) -> str | None:
        """이벤트 종류를 보고 "스피너를 어떻게 할지"를 문자열/None으로 알려준다.

        이 함수는 순수 판단 함수다(화면을 직접 건드리지 않음). 실제 스피너 조작은
        호출부인 _process_message()가 반환값을 보고 수행한다. 이렇게 판단과 실행을
        나누면 스피너 전환 규칙을 한 곳(여기)에서만 읽으면 되어 이해가 쉽다.

        반환 규약:
          - "_TEXT_"        : 본문 토큰(TEXT_DELTA) 도착 → 스피너 닫고 본문 출력 모드로.
          - "_TOOL_DONE_"   : 도구 실행 종료 → 스피너 닫기(다음 응답 대기).
          - "[..]라벨[..]"  : 스피너 문구를 이 텍스트(Rich 마크업 포함)로 갱신.
          - None            : 이 이벤트로는 스피너를 바꾸지 않음(현 상태 유지).

        Args:
            event: submit_message()가 yield한 객체. StreamEvent가 아닐 수도 있다.

        Returns:
            위 규약에 따른 문자열, 또는 변동 없음을 뜻하는 None.
        """
        # submit_message()는 StreamEvent 외에 Message(히스토리용)도 흘려보낸다.
        # StreamEvent가 아니면 스피너와 무관하므로 None(변동 없음).
        if not isinstance(event, StreamEvent):
            return None
        et = event.type
        if et == StreamEventType.TEXT_DELTA:
            # 본문이 흘러나오기 시작 — 스피너를 걷어내야 텍스트가 깔끔히 보인다.
            return "_TEXT_"
        if et == StreamEventType.STREAM_REQUEST_START:
            # GPU(Worker) 서버에 추론 요청을 막 보낸 시점. 지식 검색(KB) 직후 단계.
            return "[cyan]모델 추론 중...[/cyan]"
        if et == StreamEventType.MESSAGE_START:
            # 모델이 응답 메시지를 만들기 시작한 단계.
            return "[cyan]응답 생성 중...[/cyan]"
        if et == StreamEventType.THINKING_START:
            # 모델이 내부 사고(thinking)를 시작한 단계 — 색을 달리해 구분.
            return "[magenta]생각 정리 중...[/magenta]"
        if et == StreamEventType.TOOL_USE_START:
            # 도구 호출 시작 — 어떤 도구인지 이름을 넣어 "Read 실행 중..." 식으로 보여준다.
            # tool_name 속성이 없거나 비어 있으면 일반명 "도구"로 폴백.
            tool_name = getattr(event, "tool_name", None) or "도구"
            return f"[yellow]{tool_name} 실행 중...[/yellow]"
        if et == StreamEventType.TOOL_USE_STOP:
            # 도구 실행 종료 신호.
            return "_TOOL_DONE_"
        # 그 외 이벤트(USAGE_UPDATE 등)는 스피너에 영향을 주지 않는다.
        return None

    # ─── StreamEvent 표시 ───

    def display_stream_event(self, event: StreamEvent) -> None:
        """
        StreamEvent 한 개를 실제로 터미널에 그린다(출력 담당의 최종 단계).

        동작:
          OutputFormatter가 이벤트를 Rich 렌더러블(Text/Panel 등)로 변환하고,
          이벤트 종류에 따라 출력 방식을 달리한다:
            · TEXT_DELTA   → 줄바꿈 없이 이어 붙여 실시간 타자기 효과.
            · USAGE_UPDATE → 텍스트 스트림을 한 줄 마감한 뒤 사용량 표를 출력.
            · 그 외         → Panel 등 완결된 블록 단위로 한 번에 출력.

        Args:
            event: 그릴 StreamEvent. (StreamEvent가 아니면 무시한다.)

        메모:
          submit_message()는 StreamEvent와 Message를 함께 yield하는데,
          Message는 대화 히스토리 적재용이라 화면에는 표시하지 않는다.
          또한 포매터가 None을 돌려주는(표시할 것이 없는) 이벤트도 조용히 건너뛴다.
        """
        # Message 등 StreamEvent가 아닌 객체는 화면 표시 대상이 아니다.
        if not isinstance(event, StreamEvent):
            return

        # 이벤트를 Rich 렌더러블로 변환. None이면 표시할 내용이 없다는 뜻.
        result = self._formatter.format_event(event)
        if result is None:
            return

        # 텍스트 델타는 토큰이 이어져 한 문단을 이루므로 end=""로 줄바꿈 없이 붙인다.
        if event.type == StreamEventType.TEXT_DELTA:
            self.console.print(result, end="")
        # 사용량(토큰 수 등) 정보는 스트리밍 텍스트 뒤에 오므로 먼저 줄을 바꿔 마감한다.
        elif event.type == StreamEventType.USAGE_UPDATE:
            self.console.print()  # 앞선 텍스트 스트림을 한 줄로 끝맺음
            self.console.print(result)
        # 나머지(도구 결과 Panel, 에러 등)는 완결된 블록이라 그대로 한 번에 출력.
        else:
            self.console.print(result)

    # ─── 권한 프롬프트 ───

    async def prompt_permission(self, tool_name: str, message: str) -> bool:
        """
        도구를 실행해도 되는지 사용자에게 대화형으로 물어보고 허용/거부를 돌려준다.

        언제 호출되나:
          권한 파이프라인의 Layer 3(CanUseToolHandler)에서 "ASK"(물어보기) 결정이
          나면 이 함수가 불려 사용자에게 최종 확인을 받는다.

        Args:
            tool_name: 실행하려는 도구 이름(패널 제목에 표시).
            message: 왜 이 도구를 실행하려는지 설명하는 메시지.

        Returns:
            True면 실행 허용, False면 거부. Ctrl+C/Ctrl+D로 취소해도 안전하게 거부(False).
        """
        # 무엇을 승인할지 노란 경고 패널로 강조해서 보여준다.
        self.console.print(
            Panel(
                f"[yellow]{message}[/yellow]",
                title=f"[bold yellow]권한 요청: {tool_name}[/bold yellow]",
                border_style="yellow",
            )
        )

        # 사용자 선택을 받는다: Y(허용) / N(거부) / A(항상 허용).
        # prompt()는 블로킹이므로 run_in_executor로 별도 스레드에서 대기시킨다.
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
                # 지금은 A도 이번 한 번만 허용과 동일하게 동작한다(정책 저장 미구현).
                return True
            # 그 외 입력은 모두 거부로 처리(fail-closed).
            return False
        except (KeyboardInterrupt, EOFError):
            # 확인 도중 취소하면 안전한 쪽(거부)으로 처리한다.
            return False

    # ─── 도구 결과 포맷 ───

    def _format_tool_result(self, content: str) -> str:
        """
        도구 실행 결과 문자열을 표시용으로 다듬는다.

        지금은 입력을 그대로 돌려주는 자리표시자(passthrough) 구현이다.
        원래 의도는 결과에 코드 블록이 있으면 문법 강조(syntax highlighting)를
        입히는 것 — 추후 확장을 위한 훅으로 남겨둔 메서드다.

        Args:
            content: 도구가 돌려준 원본 텍스트.

        Returns:
            (현재는) 가공하지 않은 동일한 텍스트.
        """
        return content

    # ─── 세션 명령어 핸들러 ───

    # 아래 _cmd_* 핸들러들은 모두 동일한 시그니처(self, args: list[str])를 가진다.
    # run() 루프가 `await handler(args)` 형태로 호출하므로 async여야 하며,
    # args에는 명령어 뒤에 이어진 토큰들이 리스트로 담겨 온다(없으면 빈 리스트).

    async def _cmd_help(self, args: list[str]) -> None:
        """/help — 사용 가능한 슬래시 명령어 목록을 표로 출력한다."""
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
        """/clear — 터미널 화면을 지운다(대화 히스토리 자체는 유지)."""
        self.console.clear()

    async def _cmd_exit(self, args: list[str]) -> None:
        """/exit — _running 플래그를 내려 run() 루프를 정상 종료시킨다."""
        self.console.print("[dim]세션을 종료합니다.[/dim]")
        self._running = False

    async def _cmd_model(self, args: list[str]) -> None:
        """/model [name] — 인자가 없으면 현재 모델을, 있으면 모델을 변경한다."""
        if args:
            # 인자가 주어졌으면 모델 변경 시도. 첫 번째 토큰만 사용한다.
            new_model = args[0]
            # 허용된 별칭(primary/auxiliary)만 받는다(fail-closed 검증).
            if new_model in ("primary", "auxiliary"):
                self._model = new_model
                self.console.print(f"[green]모델이 '{new_model}'로 변경되었습니다.[/green]")
            else:
                self.console.print(
                    "[red]유효하지 않은 모델입니다. 'primary' 또는 'auxiliary'를 사용하세요.[/red]"
                )
        else:
            # 인자가 없으면 현재 선택된 모델을 알려준다.
            self.console.print(f"현재 모델: [bold]{self._model}[/bold]")

    async def _cmd_config(self, args: list[str]) -> None:
        """/config — 현재 로딩된 주요 설정값을 표로 보여준다."""
        # 부트스트랩 성공(state.config 존재) 시에만 설정을 읽을 수 있다.
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
            # 부트스트랩 실패 등으로 설정이 없으면 안내만 한다.
            self.console.print("[yellow]설정이 로드되지 않았습니다.[/yellow]")

    async def _cmd_session(self, args: list[str]) -> None:
        """/session — 현재 세션 요약(턴 수, 토큰, 도구 호출 등)을 표로 보여준다."""
        if self._state:
            # GlobalState가 제공하는 요약 dict를 그대로 표로 펼친다.
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
        """/thinking — 모델의 사고(thinking) 블록을 화면에 표시할지 여부를 켜고 끈다."""
        # 포매터의 show_thinking 플래그를 반전시켜, 이후 이벤트 렌더링에 반영한다.
        self._formatter.show_thinking = not self._formatter.show_thinking
        status = "켜짐" if self._formatter.show_thinking else "꺼짐"
        self.console.print(f"[green]Thinking 표시: {status}[/green]")

    # ─── 종료 ───

    async def _shutdown(self) -> None:
        """세션 종료 시 마무리 작업 — 세션 사용량 요약을 출력하고 인사한다.

        run() 루프가 끝난 뒤 딱 한 번 호출된다. state가 있으면 이번 세션 동안의
        턴 수/토큰/도구 호출 통계를 요약 패널로 보여준다.
        """
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
    """CLI 진입점 — 콘솔에서 `nexus`를 치면 실행되는 함수.

    pyproject.toml의 [project.scripts]에 nexus = "cli.repl:main"으로 등록되어 있다.
    기본 옵션으로 NexusREPL을 만들고 asyncio 이벤트 루프에서 run()을 끝까지 돌린다.
    (권한 모드/모델/세션 복원 등 옵션 파싱이 필요하면 이 위에 인자 처리를 추가하면 된다.)
    """
    repl = NexusREPL()
    asyncio.run(repl.run())


# 이 파일을 스크립트로 직접 실행(`python cli/repl.py`)했을 때도 main()이 돌게 한다.
# 다른 모듈이 import할 때는 실행되지 않는다(파이썬 관용구).
if __name__ == "__main__":
    main()
