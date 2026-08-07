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
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.formatters import OutputFormatter, format_change_preview
from core.message import StreamEvent, StreamEventType

# 모듈 전용 로거. 프로젝트 규칙상 "nexus.{모듈경로}" 네임스페이스를 사용한다.
# 이렇게 하면 logging.getLogger("nexus")로 상위 레벨을 한 번에 조절할 수 있고,
# 아래 _apply_log_level()이 바로 그 상위 로거의 레벨을 바꿔 하위에 전파시킨다.
logger = logging.getLogger("nexus.cli.repl")


class SlashCommandCompleter(Completer):
    """`/`로 시작하는 입력에 슬래시 명령을 제안하는 자동완성기 (D9).

    명령 목록을 인자로 "고정"하지 않고 콜러블로 받는 이유: REPL이 명령을
    추가·제거해도 완성 목록이 자동으로 따라오게 하기 위함이다(손으로 두 벌
    관리하면 반드시 어긋난다).
    """

    def __init__(self, list_commands: Any) -> None:
        self._list_commands = list_commands

    def get_completions(self, document: Any, complete_event: Any) -> Any:
        text = document.text_before_cursor
        # 첫 토큰이 슬래시로 시작할 때만 제안한다(일반 대화 입력을 방해하지 않도록).
        if not text.startswith("/") or " " in text:
            return
        for command in self._list_commands():
            if command.startswith(text):
                yield Completion(command, start_position=-len(text))

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
            permission_mode: 권한 모드 (default, accept_edits, auto, plan, trust, bypass).
                도구 실행을 얼마나 자동 허용할지를 결정한다. 배너/세션 표시용으로도 쓰인다.
                accept_edits는 파일 수정(Write/Edit 등)만 자동 승인하고 Bash 등은 확인한다.
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
        # 도구 실행 컨텍스트 — _bootstrap이 채운다. 런타임 모드 전환(A2)이 이
        # 객체의 options·permission_mode를 갱신하므로 인스턴스에 보관한다.
        self._tool_ctx: Any = None
        # 마지막 어시스턴트 응답 본문(/copy용, D6). TEXT_DELTA를 누적해 둔다.
        self._last_response: str = ""
        # Bash "항상 허용" 명령 프리픽스 집합(A4). 도구명 단위로 Bash를 통째로
        # 풀면 이후 어떤 명령이든 통과하므로, 첫 토큰(예: "git", "ls") 단위로만
        # 등록한다. 등록·사용 시점 양쪽에서 CommandFilter 검사를 다시 통과해야 한다.
        self._bash_allow_prefixes: set[str] = set()
        # "이 세션 항상 허용"으로 승인한 도구 이름 집합(numbered 프롬프트 2번).
        # 세션 한정 메모리 저장 — 프로세스가 끝나면 사라진다(영속화 안 함).
        # Bash·DANGEROUS 부류는 등록 대상에서 제외한다(fail-closed).
        # TODO(nexus): A4에서 Bash 명령 프리픽스 단위·경로 정규화로 확장.
        self._session_allow: set[str] = set()
        # 메인 while 루프의 실행 여부 플래그. /exit가 False로 바꿔 루프를 끝낸다.
        self._running = False

        # 진행 스피너 상태 — _process_message가 스트리밍 중 열고 닫는다. 인스턴스
        # 필드로 두는 이유는 권한 확인 프롬프트가 스피너를 닫을 수 있어야 하기
        # 때문(_suspend_spinner 참고). 초기값은 스피너 없음.
        self._status_ctx: Any = None
        self._status_active: bool = False

        # prompt-toolkit 세션 — 방향키로 이전 입력 재호출(히스토리)과
        # 멀티라인 편집을 지원한다. InMemoryHistory라 프로세스 종료 시 사라진다.
        # Shift+Tab으로 권한 모드를 순환한다(A2). 키 콜백이 상태를 직접 바꾸지 않고
        # 입력 버퍼에 `/mode next`를 넣어 제출하는 이유: 전환을 메인 루프의 명령
        # 처리 경로 하나로 모아, 스트리밍 중 상태가 바뀌는 경합을 원천 차단하기 위함.
        # 터미널이 Shift+Tab을 보내지 못해도 `/mode next`를 직접 입력하면 동일하다.
        _kb = KeyBindings()

        @_kb.add("s-tab")
        def _cycle_permission_mode(event: Any) -> None:
            buf = event.app.current_buffer
            buf.text = "/mode next"
            buf.cursor_position = len(buf.text)
            buf.validate_and_handle()

        # Alt+Enter로 줄바꿈(D9). Enter는 그대로 "전송"이라 기존 사용감을 지키면서,
        # 코드·다단락을 여러 줄로 입력할 수 있다. 붙여넣기는 prompt_toolkit의
        # bracketed paste가 기본 처리하므로 여러 줄이 그대로 들어온다.
        @_kb.add("escape", "enter")
        def _insert_newline(event: Any) -> None:
            event.app.current_buffer.insert_text("\n")

        self._prompt_session: PromptSession = PromptSession(
            history=InMemoryHistory(),
            key_bindings=_kb,
            # 하단 상태줄(C2) — 권한 모드·토큰·세션을 항상 보이게 한다.
            # Rich Live 대신 prompt_toolkit의 bottom_toolbar를 쓰는 이유:
            # Live는 타자기처럼 흘리는 본문 출력과 화면 갱신이 서로 간섭한다.
            bottom_toolbar=self._bottom_toolbar,
            # `/`로 시작하면 슬래시 명령을 제안한다(D9).
            completer=SlashCommandCompleter(lambda: sorted(self._session_commands)),
            complete_while_typing=True,
            # 위/아래 화살표가 "입력한 접두어로 시작하는 과거 입력"을 찾게 한다.
            # (Ctrl+R 역방향 검색은 prompt_toolkit 기본 emacs 바인딩으로 동작한다.)
            enable_history_search=True,
            # 여러 줄 입력을 화면에 그대로 보여 준다(Alt+Enter로 만든 줄 포함).
            multiline=False,
        )

        # 세션 명령어 맵 — "/로 시작하는 입력"을 어떤 핸들러로 보낼지 정의한다.
        # run() 루프가 입력 첫 토큰을 이 맵의 키와 비교해 매칭되면 그 핸들러를 부른다.
        self._session_commands: dict[str, Any] = {
            "/help": self._cmd_help,
            "/clear": self._cmd_clear,
            "/exit": self._cmd_exit,
            "/model": self._cmd_model,
            "/mode": self._cmd_mode,
            "/config": self._cmd_config,
            "/session": self._cmd_session,
            "/thinking": self._cmd_thinking,
            "/verbose": self._cmd_verbose,
            "/cost": self._cmd_cost,
            "/save": self._cmd_save,
            "/diff": self._cmd_diff,
            "/copy": self._cmd_copy,
            "/compact": self._cmd_compact,
            "/resume": self._cmd_resume,
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

                # `!`로 시작하면 셸 명령 패스스루(D1). 슬래시 명령 판정보다 먼저
                # 처리해 명령 이름과 충돌할 여지를 없앤다.
                if user_input.lstrip().startswith("!"):
                    await self._run_bash_passthrough(user_input.lstrip()[1:])
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

            # --resume: 이전 세션 대화를 복원한다. 트랜스크립트가 있으면 그 세션 ID를
            # 그대로 이어받아(같은 transcript.jsonl에 append) Phase 2가 모든 컴포넌트를
            # 그 ID로 배선하게 하고, 복원한 메시지는 Phase 2 이후 엔진에 주입한다.
            # (읽기·주입 인프라는 이미 존재 — read_transcript_messages / bind_request.
            #  기존엔 배너 표시만 하고 실제 복원 배선이 빠져 있었다.)
            resumed_messages = self._load_resume_messages()

            # Phase 2: 도구 레지스트리 + 메모리 + QueryEngine 구성.
            # 반환은 컴포넌트 dict이며, 이 중 query_engine만 REPL이 직접 쓴다.
            components = await init_phase2(self._state)
            self._query_engine = components.get("query_engine")

            # A1/A3: ASK 확인 핸들러 배선 — v2 계약(numbered 프롬프트 + 피드백 +
            # accept-edits 모드 인지)을 주입한다. executor는 v2를 우선 사용하고,
            # v2가 없으면 구 ask_handler(bool)로 폴백한다(웹·비대화형 무회귀).
            # 자동 허용 모드(auto/bypass/trust)에서는 주입하지 않아 executor가
            # 통과시키게 한다 — 에이전트 자율 루프가 매번 멈추지 않도록.
            # accept_edits는 자동 허용 모드가 아니다 — 핸들러 "안"에서 파일 수정
            # 부류만 선별 자동 승인하고 Bash 등은 종전대로 묻는다(모드 인지형).
            #
            # A2(2026-08-05): 배선을 _apply_mode_change 한 곳으로 모았다. 기동 시에도
            # 같은 헬퍼를 태워, CLI 인자로 받은 모드가 파이프라인·도구 컨텍스트·
            # GlobalState까지 실제로 반영되게 한다(이전에는 self._permission_mode만
            # 바뀌고 GlobalState는 기본값 그대로인 미동기 상태였다).
            self._tool_ctx = components.get("tool_use_context")
            self._apply_mode_change(self._permission_mode, announce=False)

            # 진입점 채널을 'cli'로 고정 — 이 세션의 히스토리 저장(Redis 키·transcript
            # 폴더)을 cli 채널로 격리해 web/api 히스토리와 서로 안 보이게 한다. 엔진은
            # __init__ 때 채널을 context.options에서 읽지만, bootstrap 공용 컨텍스트를
            # 공유하므로 여기서 bind_request로 명시 주입한다. resume면 복원 메시지도 함께
            # 얹는다(빈 리스트는 None으로 넘겨 기존 히스토리를 지우지 않는다 — 무회귀).
            if self._query_engine is not None:
                self._query_engine.bind_request(
                    session_id=self._state.session_id,
                    restore_messages=resumed_messages or None,
                    channel="cli",
                )
                if resumed_messages:
                    logger.info(
                        "세션 복원: %d개 메시지 로드 (session=%s)",
                        len(resumed_messages),
                        self._resume_session_id,
                    )
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

    def _load_resume_messages(self) -> list:
        """--resume 세션의 트랜스크립트를 읽어 Message 리스트로 복원한다.

        반환:
          복원할 메시지 리스트(user/assistant 순서 보존). resume 미지정이거나
          해당 세션 파일이 없으면 빈 리스트를 돌려주고, 이어받기를 취소한다
          (state.session_id는 새 세션 그대로 두어 새 대화로 시작).

        부작용:
          유효한 트랜스크립트를 찾으면 state.session_id를 그 세션 ID로 덮어써,
          이후 Phase 2가 생성하는 트랜스크립트·엔진이 같은 파일에 이어 쓰게 한다.
        """
        if not self._resume_session_id or self._state is None:
            return []
        try:
            from core.memory.transcript import read_transcript_messages
            from core.message import Message

            sessions_dir = self._state.config.sessions_dir
            # cli 채널로 격리 저장된 세션에서 복원한다(web/api 세션은 대상 아님).
            raw = read_transcript_messages(
                sessions_dir, self._resume_session_id, channel="cli"
            )
            if not raw:
                logger.warning(
                    "복원할 세션을 찾지 못했습니다: %s — 새 세션으로 시작",
                    self._resume_session_id,
                )
                self._resume_session_id = None
                return []
            # 이어받기: 새 발화가 같은 트랜스크립트에 append되도록 세션 ID를 승계한다.
            self._state.session_id = self._resume_session_id
            messages: list = []
            for entry in raw:
                role = entry.get("role")
                content = entry.get("content") or ""
                if role == "user":
                    messages.append(Message.user(content))
                elif role == "assistant":
                    messages.append(Message.assistant(text=content))
            return messages
        except Exception as e:  # noqa: BLE001
            # 복원 실패가 REPL 기동 자체를 막아서는 안 된다 — 새 세션으로 폴백.
            logger.warning("세션 복원 실패: %s — 새 세션으로 시작", e)
            self._resume_session_id = None
            return []

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
        _ask_note = (
            "자동 허용"
            if self._permission_mode in {"auto", "bypass", "trust"}
            else "실행 전 확인"
        )
        info.append(
            f"  권한 모드   {self._permission_mode} (도구 ASK: {_ask_note})\n",
            style="dim",
        )
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
            "[cyan]/thinking[/cyan][dim] · [/dim]"
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

        # 새 응답 스트림 시작 — thinking 필터 상태를 초기화한다. 이전 응답이 사고
        # 구간을 닫지 못한 채 끝났더라도, 여기서 리셋해 다음 응답이 삼켜지지 않게 한다.
        self._formatter.reset_stream_state()
        # /copy가 "직전 답변"만 담도록 매 턴 초기화한다.
        self._last_response = ""
        # 사후 검증이 볼 구간의 시작점 — 이번 턴에 추가된 메시지만 근거로 삼는다.
        turn_start = len(getattr(self._query_engine, "_messages", []))

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
        # 스피너 상태는 인스턴스 필드로 둔다(지역변수 아님). 이유: 스트리밍 도중
        # 권한 확인(prompt_permission)이 executor 깊은 곳에서 호출되는데, 그때 입력
        # 프롬프트가 스피너와 겹치면 화면이 깨진다. 핸들러가 _suspend_spinner()로
        # 스피너를 닫으려면 이 상태에 접근할 수 있어야 하므로 self._status_* 로 공유한다.
        self._status_ctx = self.console.status(
            "[cyan]요청 분석 중...[/cyan] [dim](Ctrl+C로 중단)[/dim]", spinner="dots"
        )
        self._status_ctx.__enter__()
        self._status_active = True
        try:
            async for event in self._query_engine.submit_message(user_input):
                # 이 이벤트를 근거로 스피너를 어떻게 할지 결정한다.
                # 반환 규약은 _stage_label_for()의 docstring 참고.
                next_label = self._stage_label_for(event)
                if next_label == "_TEXT_":
                    # 첫 본문 토큰 도착 → 스피너를 닫고 텍스트 출력 모드로 전환.
                    self._suspend_spinner()
                elif next_label == "_TOOL_DONE_":
                    # 도구 실행 완료 → 곧 도구 결과 Panel을 출력해야 하므로 스피너를 닫는다.
                    # (스피너와 Panel이 동시에 활성이면 화면이 겹쳐 Panel이 깨져 보인다.)
                    # 다음 이벤트(다음 LLM 응답·다음 도구 호출)가 오면 아래
                    # `elif next_label is not None` 분기가 스피너를 자동으로 다시 연다.
                    self._suspend_spinner()
                elif next_label is not None:
                    # 스피너 문구를 갱신해야 하는 단계 이벤트(TURN_START/TOOL_USE_START 등).
                    if self._status_active:
                        # 이미 스피너가 떠 있으면 문구만 바꾼다.
                        self._status_ctx.update(next_label)
                    else:
                        # 스피너가 닫혀 있었다면(예: 본문 출력·권한 프롬프트 뒤 도구 호출
                        # 시작) 새로 연다.
                        self._status_ctx = self.console.status(
                            next_label, spinner="dots"
                        )
                        self._status_ctx.__enter__()
                        self._status_active = True

                # 스피너 상태 판단과 별개로, 이벤트 자체는 항상 화면에 그린다.
                self.display_stream_event(event)
        except asyncio.CancelledError:
            # Ctrl+C 등으로 스트림이 취소된 경우. 스피너를 닫고 취소 안내를 남긴다.
            self._suspend_spinner()
            self.console.print("[yellow]요청이 취소되었습니다.[/yellow]")
        except Exception as e:
            # 그 밖의 예외는 포매터를 통해 사용자 친화적 에러 메시지로 출력한다.
            self._suspend_spinner()
            self.console.print(self._formatter.format_error(str(e)))
        finally:
            # 어떤 경로로 끝나든(정상/취소/예외) 스피너가 남아있지 않도록 최종 안전 장치.
            # 예: TEXT_DELTA가 한 번도 안 와서 스피너가 계속 떠 있는 상태로 끝난 경우.
            self._suspend_spinner()

        # ── 사후 검증 ──
        # 이미 흘려보낸 본문은 고치지 않고, 확인이 필요한 것만 뒤에 덧붙인다.
        #   왜 CLI 에도 필요한가: 숫자 자릿수 오류와 "실행하지 않고 통과했다고 단정"이
        #   실측된 곳이 바로 CLI 였다. 검증기가 웹에만 있으면 정작 문제가 나는 표면이
        #   무방비다.
        #   _last_response 는 /copy 용으로 이미 이번 턴 본문만 담고 있어 그대로 쓴다.
        if self._last_response:
            from core.verification.post_check import build_answer_warnings

            warning = build_answer_warnings(
                self._last_response,
                getattr(self._query_engine, "_messages", [])[turn_start:],
            )
            if warning:
                self.console.print(warning)

    def _suspend_spinner(self) -> None:
        """진행 스피너가 떠 있으면 닫는다(멱등). 없으면 아무것도 하지 않는다.

        [왜 필요한가] 스트리밍 도중 권한 확인 프롬프트(prompt_permission)가
        executor 깊은 곳에서 호출된다. 스피너가 돌고 있는 상태에서 prompt-toolkit
        입력을 받으면 화면이 겹쳐 깨진다. 프롬프트 직전 이 메서드로 스피너를 닫고,
        스피너 재개는 다음 단계 이벤트에서 _process_message 루프가 자동 처리한다.
        """
        if getattr(self, "_status_active", False) and self._status_ctx is not None:
            self._status_ctx.__exit__(None, None, None)
            self._status_active = False

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
            # 화면에는 렌더된 값을, /copy용으로는 원문 텍스트를 따로 모은다.
            if event.text:
                self._last_response += event.text
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
        # 스트리밍 중 스피너가 떠 있으면 입력 프롬프트와 겹쳐 화면이 깨진다.
        # 프롬프트 전에 스피너를 닫는다(재개는 다음 단계 이벤트에서 루프가 처리).
        self._suspend_spinner()

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

    # ─── 권한 모드 런타임 전환 (A2) ───

    # /mode 인자로 받을 수 있는 값 — core의 세션 모드(PermissionModeValue)와 일치.
    _MODE_CHOICES = ("default", "accept_edits", "auto", "plan", "trust", "bypass")
    # Shift+Tab 순환 순서 — 실제로 자주 오가는 3단만 돈다(나머지는 /mode로 지정).
    _MODE_CYCLE = ("default", "accept_edits", "plan")
    # 확인 프롬프트를 아예 띄우지 않는 자동 허용 모드 — 핸들러를 주입하지 않는다.
    _AUTO_ALLOW_MODES = frozenset({"auto", "bypass", "trust"})

    def _apply_mode_change(self, new_mode: str, announce: bool = True) -> bool:
        """권한 모드를 바꾸고 관련 4개 지점을 한 번에 갱신한다.

        [왜 헬퍼 하나로 모으나]
          모드는 서로 다른 네 곳이 각자 들고 있다. 한 곳만 바꾸면 화면에 보이는
          모드와 실제 판정이 어긋난다(실제로 기동 시 GlobalState가 갱신되지 않아
          CLI 인자와 어긋나 있었다). 그래서 전환 경로를 이 함수 하나로 고정한다.

          ① 권한 파이프라인의 PermissionContext.mode (실제 5계층 판정 기준)
          ② tool_use_context.options["ask_handler_v2"] (확인 프롬프트 주입 여부)
          ③ tool_use_context.permission_mode (도구가 읽는 문자열 모드)
          ④ GlobalState.permission_mode + self._permission_mode (배너·/config 표시)

        [언제 호출되나]
          기동 직후(_bootstrap)와 `/mode` 명령 처리 시점뿐이다. 두 경우 모두
          "턴과 턴 사이"라, 스트리밍 중 모드가 바뀌어 판정이 뒤섞일 여지가 없다.

        Args:
            new_mode: 바꿀 모드 문자열. _MODE_CHOICES 밖의 값이면 거부한다.
            announce: True면 변경 결과를 화면에 한 줄로 알린다.

        Returns:
            변경(또는 초기 배선)에 성공하면 True, 모르는 모드면 False.
        """
        if new_mode not in self._MODE_CHOICES:
            if announce:
                self.console.print(
                    f"[red]알 수 없는 권한 모드: {new_mode}[/red] "
                    f"(가능: {', '.join(self._MODE_CHOICES)})"
                )
            return False

        # ④-a REPL 표시용 상태 — 아래 핸들러 주입 판단에도 쓰이므로 먼저 갱신한다.
        self._permission_mode = new_mode

        # ① 파이프라인 컨텍스트: frozen이라 model_copy로 mode만 갈아끼운다.
        #    (working_directory·session_id 등 나머지 필드는 그대로 보존)
        tool_ctx = self._tool_ctx
        if tool_ctx is not None:
            pipeline = tool_ctx.options.get("permission_pipeline")
            if pipeline is not None:
                try:
                    from core.permission.mode_mapping import (
                        map_mode_value_to_permission_mode,
                    )

                    pipeline.update_context(
                        pipeline.context.model_copy(
                            update={"mode": map_mode_value_to_permission_mode(new_mode)}
                        )
                    )
                except Exception as e:  # noqa: BLE001 — 표시 모드는 이미 바뀌었으므로 계속
                    logger.warning("권한 파이프라인 모드 갱신 실패: %s", e)

            # ② 확인 핸들러: 자동 허용 모드에서는 제거해 프롬프트가 뜨지 않게 한다.
            #    (accept_edits는 자동 허용이 아니다 — 핸들러 안에서 선별 승인한다)
            if new_mode in self._AUTO_ALLOW_MODES:
                tool_ctx.options.pop("ask_handler_v2", None)
            else:
                tool_ctx.options["ask_handler_v2"] = self.prompt_permission_v2

            # ③ 도구가 읽는 문자열 모드
            tool_ctx.permission_mode = new_mode

        # ④-b 전역 상태 — 배너·/config·요약이 같은 값을 보게 한다.
        if self._state is not None:
            try:
                from core.state import PermissionModeValue

                self._state.permission_mode = PermissionModeValue(new_mode)
            except ValueError:
                logger.warning("GlobalState 권한 모드 변환 실패: %s", new_mode)

        if announce:
            _note = (
                "자동 허용"
                if new_mode in self._AUTO_ALLOW_MODES
                else ("파일 수정 자동 승인" if new_mode == "accept_edits" else "실행 전 확인")
            )
            self.console.print(f"[green]권한 모드 → {new_mode}[/green] [dim]({_note})[/dim]")
        return True

    async def _cmd_mode(self, args: list[str]) -> None:
        """`/mode [모드|next]` — 권한 모드를 확인하거나 바꾼다.

        인자 없이 부르면 현재 모드와 선택지를 보여준다. `next`는 Shift+Tab과
        같은 순환(기본 → 편집 자동승인 → 계획)을 한 칸 돌린다.
        """
        if not args:
            table = Table(title="권한 모드", box=ROUNDED, show_header=True)
            table.add_column("모드", style="cyan")
            table.add_column("동작")
            table.add_column("현재", justify="center")
            descriptions = {
                "default": "쓰기·실행 전 확인",
                "accept_edits": "파일 수정 자동 승인, Bash 등은 확인",
                "auto": "대부분 자동 진행",
                "plan": "읽기·계획만, 쓰기 거부",
                "trust": "확인 없이 진행",
                "bypass": "권한 확인 우회",
            }
            for m in self._MODE_CHOICES:
                table.add_row(m, descriptions[m], "●" if m == self._permission_mode else "")
            self.console.print(table)
            self.console.print(
                "[dim]사용법: /mode <모드>  ·  Shift+Tab으로 "
                f"{' → '.join(self._MODE_CYCLE)} 순환[/dim]"
            )
            return

        target = args[0].strip().lower()
        if target == "next":
            # 순환 목록에 없는 모드에서 눌렀다면 첫 칸으로 보낸다.
            try:
                idx = self._MODE_CYCLE.index(self._permission_mode)
                target = self._MODE_CYCLE[(idx + 1) % len(self._MODE_CYCLE)]
            except ValueError:
                target = self._MODE_CYCLE[0]
        self._apply_mode_change(target)

    # ─── Bash 세션 allow-list (A4) ───

    # 프리픽스만으로 안전을 보장할 수 없게 만드는 셸 메타문자.
    # 예) "git status && rm -rf /" 는 첫 토큰이 git이지만 뒤에 위험 명령이 붙는다.
    # 이런 명령은 등록도, 자동 승인도 하지 않는다(fail-closed).
    _SHELL_METACHARS = (";", "|", "&", "`", "$(", ">", "<", "\n", "\r")

    def _command_filter(self) -> Any:
        """위험 명령 판정기를 얻는다(파이프라인 것을 재사용, 없으면 새로 생성).

        파이프라인이 쓰는 것과 같은 필터를 쓰는 이유: 규칙이 두 벌이 되면
        "권한 검사는 막는데 allow-list는 통과" 같은 불일치가 생긴다.
        """
        tool_ctx = self._tool_ctx
        if tool_ctx is not None:
            pipeline = tool_ctx.options.get("permission_pipeline")
            existing = getattr(pipeline, "_command_filter", None)
            if existing is not None:
                return existing
        from core.security.command_filter import CommandFilter

        return CommandFilter()

    def _bash_prefix_if_registrable(self, command: str) -> str | None:
        """Bash 명령에서 "항상 허용"으로 등록 가능한 프리픽스(첫 토큰)를 뽑는다.

        등록 거부 조건(하나라도 걸리면 None):
          - 셸 메타문자 포함 — 프리픽스만으로는 뒤에 붙는 명령을 통제할 수 없다
          - CommandFilter가 안전하다고 판정하지 않음(위험 패턴·미지 명령)
          - 첫 토큰이 비었거나 경로 구분자를 포함(`./x`, `/bin/x` 같은 우회)

        Returns:
            등록 가능한 프리픽스 문자열, 불가하면 None.
        """
        if not command or not isinstance(command, str):
            return None
        if any(ch in command for ch in self._SHELL_METACHARS):
            return None
        tokens = command.split()
        if not tokens:
            return None
        prefix = tokens[0]
        # 경로가 섞인 실행(./script, /usr/bin/x)은 이름만으로 동일성을 보장할 수
        # 없으므로 등록 대상에서 제외한다.
        if "/" in prefix or "\\" in prefix:
            return None
        try:
            safe, _severity, _reason = self._command_filter().check_command(command)
        except Exception as e:  # noqa: BLE001 — 판정 실패는 등록 거부로 흡수
            logger.warning("명령 안전성 판정 실패(등록 거부): %s", e)
            return None
        return prefix if safe else None

    def _is_bash_autoallowed(self, command: str) -> bool:
        """이미 등록된 프리픽스로 자동 승인할 수 있는 명령인지 판단한다.

        등록 여부만 보지 않고 **매 호출마다** 메타문자·위험 패턴을 다시 검사한다.
        등록 시점에 안전했던 프리픽스라도 이번 명령이 안전하다는 보장은 없기 때문이다
        (예: `git status`로 등록 → `git push --force` 호출).
        """
        if not self._bash_allow_prefixes:
            return False
        prefix = self._bash_prefix_if_registrable(command)
        return prefix is not None and prefix in self._bash_allow_prefixes

    async def prompt_permission_v2(
        self,
        tool_name: str,
        message: str,
        tool_input: dict | None = None,
    ) -> dict:
        """
        ask_handler v2 계약 구현 — 모드 인지형 numbered 권한 프롬프트 (A1/A3).

        구 prompt_permission(Y/N/A → bool)과의 차이:
          1) 모드 인지: accept_edits 모드에서 파일 수정(FILE_WRITE) 부류 도구는
             프롬프트 없이 즉시 승인하고 1줄 안내만 남긴다. Bash 등은 종전대로 묻는다.
          2) numbered 선택: 1) 예  2) 예(이 세션 항상 허용)  3) 아니오+피드백.
          3) 거부 피드백: 3번 선택 시 이유/지시를 입력받아 executor가 모델에
             다음 턴으로 전달한다(모델이 사용자 의도를 반영해 재시도 가능).

        Args:
            tool_name: 실행하려는 도구 이름.
            message: 왜 이 도구를 실행하려는지 설명하는 메시지.
            tool_input: 도구 입력(dict). 지금은 미사용 — Phase B에서 diff
                미리보기 렌더링에 쓰기 위해 계약에 미리 포함해 둔다.

        Returns:
            dict: {"approved": bool, "feedback": str, "always_allow": bool}.
            취소(Ctrl+C/Ctrl+D)는 안전한 쪽(거부, 피드백 없음)으로 처리한다.
        """
        # 분류는 core의 공개 함수를 재사용한다(CLI에 분류표 복제 금지 — 드리프트 방지).
        # lazy import: repl 기동 시 core.permission을 미리 끌고 오지 않기 위함.
        from core.permission.pipeline import categorize_tool_name
        from core.permission.types import ToolCategory

        category = categorize_tool_name(tool_name)

        # ① accept_edits 모드 — 파일 수정 부류만 자동 승인.
        #    categorize_tool_name이 None(미지의 도구)이면 자동 승인하지 않는다
        #    (fail-closed — 모르는 도구를 쓰기로 간주해 무확인 통과시키면 위험).
        if self._permission_mode == "accept_edits" and category == ToolCategory.FILE_WRITE:
            self.console.print(f"[dim]⏺ 자동 승인(accept edits): {tool_name}[/dim]")
            return {"approved": True, "feedback": "", "always_allow": False}

        # ② "이 세션 항상 허용"으로 이미 등록된 도구면 즉시 승인.
        if tool_name in self._session_allow:
            self.console.print(f"[dim]⏺ 자동 승인(세션 항상 허용): {tool_name}[/dim]")
            return {"approved": True, "feedback": "", "always_allow": False}

        # ②-b Bash는 도구가 아니라 "명령 프리픽스" 단위로 등록된다(A4).
        #     등록 여부와 별개로 이번 명령의 안전성을 매번 다시 검사한다.
        _command = str((tool_input or {}).get("command", ""))
        if category == ToolCategory.BASH and self._is_bash_autoallowed(_command):
            _prefix = _command.split()[0]
            self.console.print(f"[dim]⏺ 자동 승인(세션 항상 허용): {_prefix} …[/dim]")
            return {"approved": True, "feedback": "", "always_allow": False}

        # ③ numbered 프롬프트 — 스피너가 떠 있으면 입력과 겹치므로 먼저 닫는다.
        self._suspend_spinner()
        self.console.print(
            Panel(
                f"[yellow]{message}[/yellow]",
                title=f"[bold yellow]권한 요청: {tool_name}[/bold yellow]",
                border_style="yellow",
            )
        )
        # ③-b 파일을 바꾸는 도구면 변경 내용을 diff로 먼저 보여 준다(B2).
        #     경로만 보고 승인하는 것과 실제 변경을 보고 승인하는 것은 다르다.
        #     미리보기 생성이 실패해도(None) 승인 흐름은 그대로 진행한다.
        if category == ToolCategory.FILE_WRITE and tool_input:
            preview = format_change_preview(tool_name, tool_input)
            if preview is not None:
                self.console.print(preview)

        try:
            # prompt()는 블로킹이므로 run_in_executor로 별도 스레드에서 대기시킨다.
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._prompt_session.prompt(
                    "1) 예  2) 예(이 세션 항상 허용)  3) 아니오(피드백 입력): "
                ),
            )
            choice = response.strip().lower()

            # 1번(예) — 이번 한 번만 허용. 구 습관(y/yes)도 받아준다.
            if choice in ("1", "y", "yes"):
                return {"approved": True, "feedback": "", "always_allow": False}

            # 2번(항상 허용) — 세션 allow 목록에 등록 후 허용. 구 습관(a/always) 겸용.
            if choice in ("2", "a", "always"):
                # Bash는 도구명이 아니라 **명령 프리픽스**로 등록한다(A4).
                # 셸 메타문자가 있거나 CommandFilter가 안전하다고 보지 않으면
                # 등록하지 않고 이번 1회만 허용한다(fail-closed).
                if category == ToolCategory.BASH:
                    prefix = self._bash_prefix_if_registrable(_command)
                    if prefix is None:
                        self.console.print(
                            "[dim]이 명령은 프리픽스 단위로 안전하게 등록할 수 없어 "
                            "이번 한 번만 허용합니다.[/dim]"
                        )
                        return {"approved": True, "feedback": "", "always_allow": False}
                    self._bash_allow_prefixes.add(prefix)
                    self.console.print(
                        f"[dim]⏺ 이 세션 동안 `{prefix}` 명령을 항상 허용합니다"
                        "(위험 패턴·복합 명령은 매번 다시 확인).[/dim]"
                    )
                    return {"approved": True, "feedback": "", "always_allow": True}

                # DANGEROUS·미지의 도구는 등록 금지(fail-closed). 이번 1회만 허용.
                if category == ToolCategory.DANGEROUS or category is None:
                    self.console.print(
                        f"[dim]{tool_name}은(는) 항상 허용 대상이 아니라 "
                        "이번 한 번만 허용합니다.[/dim]"
                    )
                    return {"approved": True, "feedback": "", "always_allow": False}
                self._session_allow.add(tool_name)
                self.console.print(f"[dim]⏺ 이 세션 동안 {tool_name}을(를) 항상 허용합니다.[/dim]")
                return {"approved": True, "feedback": "", "always_allow": True}

            # 그 외(3 포함) — 거부. 이유/지시를 한 줄 받아 모델에 전달한다(선택 입력).
            feedback = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._prompt_session.prompt(
                    "거부 이유/지시 (모델에 전달, 비워도 됨): "
                ),
            )
            return {
                "approved": False,
                "feedback": feedback.strip(),
                "always_allow": False,
            }
        except (KeyboardInterrupt, EOFError):
            # 확인 도중 취소하면 안전한 쪽(거부)으로 처리한다.
            return {"approved": False, "feedback": "", "always_allow": False}

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
            ("/model", "현재 라우팅 모델을 표시한다"),
            ("/mode", "권한 모드 확인·변경 (Shift+Tab으로 순환)"),
            ("/config", "현재 설정을 표시한다"),
            ("/session", "세션 정보를 표시한다"),
            ("/thinking", "thinking 표시를 토글한다"),
            ("/verbose", "도구 표시를 축약↔전문으로 토글한다"),
            ("/cost", "세션 누적 토큰 사용량을 보여준다"),
            ("/save", "대화를 Markdown 파일로 저장한다"),
            ("/diff", "작업 트리의 git 변경사항을 보여준다"),
            ("/copy", "마지막 응답을 클립보드로 복사한다"),
            ("/compact", "대화 맥락을 강제로 압축한다"),
            ("/resume", "이전 세션 목록을 보고 이어받는다"),
            ("!<명령>", "셸 명령을 실행한다(표시 전용 — 대화 맥락에 미포함)"),
        ]
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        # 키 단축키 안내(D9/D10) — 명령 표만 보면 알 수 없는 조작을 함께 알린다.
        self.console.print(table)
        self.console.print(
            "[dim]단축키: Shift+Tab 권한 모드 순환 · Alt+Enter 줄바꿈 · "
            "Ctrl+R 히스토리 검색 · `/` 입력 시 명령 자동완성 · Ctrl+C 요청 취소[/dim]"
        )
        return

        self.console.print(table)

    async def _cmd_clear(self, args: list[str]) -> None:
        """/clear — 터미널 화면을 지운다(대화 히스토리 자체는 유지)."""
        self.console.clear()

    async def _cmd_exit(self, args: list[str]) -> None:
        """/exit — _running 플래그를 내려 run() 루프를 정상 종료시킨다."""
        self.console.print("[dim]세션을 종료합니다.[/dim]")
        self._running = False

    async def _cmd_model(self, args: list[str]) -> None:
        """/model — 라우팅이 질의 유형별로 쓰는 실제 모델을 표로 표시한다(표시 전용).

        [왜 표시 전용인가] 모델 선택은 라우팅 config(CHAT/KNOWLEDGE/TOOL)가 질의
        유형에 따라 자동으로 결정한다. 과거 이 명령은 인자로 모델을 "변경"하는 것처럼
        보였지만, self._model은 배너·표시에만 쓰일 뿐 엔진에 배선되지 않아 실제로는
        아무것도 바꾸지 못했다(장식용). 혼동을 없애기 위해 변경 기능을 제거하고,
        배너(_render_welcome)와 같은 소스인 config.routing에서 실모델명만 보여준다.
        """
        # 배너와 동일한 방어적 getattr 패턴으로 라우팅 설정을 꺼낸다.
        routing = getattr(getattr(self._state, "config", None), "routing", None)
        table = Table(title="모델 라우팅 (질의별 자동 분기)", border_style="blue")
        table.add_column("질의 유형", style="cyan")
        table.add_column("모델", style="white")
        if routing is not None and getattr(routing, "enabled", False):
            # 라우팅 활성 — 질의 유형별 실제 모델을 표시한다.
            table.add_row(
                "CHAT", getattr(getattr(routing, "chat_mode", None), "model", "?")
            )
            table.add_row(
                "KNOWLEDGE",
                getattr(getattr(routing, "knowledge_mode", None), "model", "?"),
            )
            table.add_row(
                "TOOL", getattr(getattr(routing, "tool_mode", None), "model", "?")
            )
        else:
            # 라우팅 비활성 — primary/auxiliary 두 모델만 표시한다.
            model_cfg = getattr(
                getattr(self._state, "config", None), "model", None
            )
            table.add_row("Primary", getattr(model_cfg, "primary_model", self._model))
            table.add_row("Auxiliary", getattr(model_cfg, "auxiliary_model", "-"))
        self.console.print(table)

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
            _ask_note = (
                "자동 허용"
                if self._permission_mode in {"auto", "bypass", "trust"}
                else "실행 전 확인"
            )
            table.add_row("권한 모드", f"{self._permission_mode} (도구 ASK: {_ask_note})")
            # 세션 allow-list 현황(A4) — 무엇이 확인 없이 통과하는지 사용자가 볼 수
            # 있어야 한다. 등록된 것이 없으면 행을 만들지 않는다.
            if self._session_allow:
                table.add_row("항상 허용(도구)", ", ".join(sorted(self._session_allow)))
            if self._bash_allow_prefixes:
                table.add_row(
                    "항상 허용(명령)",
                    ", ".join(f"`{p}`" for p in sorted(self._bash_allow_prefixes)),
                )
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

    # ─── 하단 상태줄 (C2) ───

    def _bottom_toolbar(self) -> str:
        """프롬프트 하단에 항상 보이는 한 줄 상태를 만든다 (C2).

        보여 주는 것: 권한 모드 · 누적 토큰(in/out) · 세션 ID 앞 8자.
        부트스트랩 전이거나 값을 못 읽어도 절대 예외를 내지 않는다 — 이 함수가
        실패하면 프롬프트 자체가 뜨지 않기 때문이다(입력 불가 상태가 된다).
        """
        try:
            mode = self._permission_mode
            if self._state is None:
                return f" 모드 {mode} · 초기화 중"
            s = self._state.get_session_summary()
            tin = s.get("total_input_tokens", 0)
            tout = s.get("total_output_tokens", 0)
            sid = str(s.get("session_id", ""))[:8]
            return f" 모드 {mode} · 토큰 {tin:,}/{tout:,} · 세션 {sid}"
        except Exception:  # noqa: BLE001 — 상태줄 실패로 입력을 막지 않는다
            return " "

    # ─── `!` bash 패스스루 (D1) ───

    async def _run_bash_passthrough(self, command: str) -> None:
        """`!<명령>` 입력을 셸 명령으로 직접 실행한다 (D1).

        [안전 규칙 — 모두 fail-closed]
          1. plan / deny_all 모드에서는 아예 실행하지 않는다(그 모드의 의미가
             "부작용 금지"이므로 우회 통로를 열어 주면 안 된다).
          2. 권한 파이프라인이 쓰는 것과 **같은 CommandFilter**로 먼저 검사한다.
             위험 판정이면 실행하지 않는다.
          3. 실행 여부와 무관하게 감사 로그(JSONL)에 남긴다.
          4. **v1은 표시 전용** — 결과를 대화 컨텍스트에 넣지 않는다. 모델이
             보지 못하므로 "방금 그 결과 봐줘"는 동작하지 않는다(문서화된 한계).
        """
        command = (command or "").strip()
        if not command:
            self.console.print("[yellow]실행할 명령이 없습니다. 예: !git status[/yellow]")
            return

        # ① 모드 게이트
        if self._permission_mode in ("plan", "deny_all"):
            self.console.print(
                f"[yellow]{self._permission_mode} 모드에서는 `!` 명령을 실행하지 않습니다.[/yellow]"
            )
            self._audit_bash(command, allowed=False, reason=f"mode:{self._permission_mode}")
            return

        # ② 위험 명령 필터(권한 파이프라인과 같은 규칙)
        try:
            safe, severity, reason = self._command_filter().check_command(command)
        except Exception as e:  # noqa: BLE001 — 판정 실패는 차단으로 흡수(fail-closed)
            logger.warning("`!` 명령 안전성 판정 실패(차단): %s", e)
            safe, severity, reason = False, "unknown", str(e)
        if not safe:
            self.console.print(
                Text(f"차단됨({severity}): {reason}", style="red")
            )
            self._audit_bash(command, allowed=False, reason=f"{severity}:{reason}")
            return

        # ③ 실행 — async 루프를 막지 않도록 비동기 서브프로세스로 돌린다.
        self._audit_bash(command, allowed=True, reason="")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._state.cwd if self._state else None,
            )
            out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        except TimeoutError:
            self.console.print("[red]명령 시간 초과(60초)[/red]")
            return
        except OSError as e:
            self.console.print(Text(f"실행 실패: {e}", style="red"))
            return

        out = out_b.decode("utf-8", "replace").rstrip()
        # 외부 출력은 반드시 Text로 감싼다(대괄호가 Rich markup으로 해석되는 것 방지).
        body = Text(out or "(출력 없음)")
        self.console.print(
            Panel(
                body,
                title=f"$ {command[:60]}",
                subtitle=f"exit={proc.returncode}",
                border_style="green" if proc.returncode == 0 else "red",
                expand=False,
            )
        )
        self.console.print("[dim]※ 이 결과는 대화 맥락에 포함되지 않습니다(표시 전용).[/dim]")

    def _audit_bash(self, command: str, allowed: bool, reason: str) -> None:
        """`!` 명령 실행 시도를 감사 로그에 남긴다(실패해도 흐름을 막지 않는다)."""
        try:
            tool_ctx = self._tool_ctx
            audit = tool_ctx.options.get("audit_logger") if tool_ctx else None
            if audit is None:
                return
            from core.permission.types import PermissionAuditEntry

            audit.log_decision(
                PermissionAuditEntry(
                    session_id=self._state.session_id if self._state else "",
                    tool_name="!bash",
                    tool_category="bash",
                    tool_input_summary=command[:200],
                    decision="allow" if allowed else "deny",
                    reason=reason,
                    source="cli_passthrough",
                    mode=self._permission_mode,
                    message="CLI `!` 패스스루",
                )
            )
        except Exception as e:  # noqa: BLE001 — 감사 실패가 명령 실행을 막지 않는다
            logger.debug("`!` 감사 기록 실패(무시): %s", e)

    async def _cmd_compact(self, args: list[str]) -> None:
        """/compact — 지금 대화 맥락을 강제로 압축한다 (D2).

        평소에는 임계치를 넘을 때만 자동 압축되지만, 긴 작업 뒤에 맥락을 미리
        정리하고 싶을 때가 있다. ContextManager의 강제 압축 경로(force=True)를
        그대로 쓰며, 전/후 토큰 추정치를 보여 준다.
        """
        engine = self._query_engine
        if engine is None:
            self.console.print("[yellow]세션이 초기화되지 않았습니다.[/yellow]")
            return
        cm = getattr(engine, "_context_manager", None)
        if cm is None:
            self.console.print("[yellow]컨텍스트 관리자가 없어 압축할 수 없습니다.[/yellow]")
            return
        messages = getattr(engine, "_messages", None)
        if not messages:
            self.console.print("[dim]압축할 대화가 없습니다.[/dim]")
            return

        before_msgs = len(messages)
        before_tokens = cm._estimate_tokens(messages)
        try:
            compacted = await cm.auto_compact_if_needed(messages, force=True)
        except Exception as e:  # noqa: BLE001 — 압축 실패가 세션을 끊으면 안 된다
            self.console.print(Text(f"압축 실패: {e}", style="red"))
            return

        # 엔진의 실제 히스토리를 압축 결과로 교체한다(같은 리스트 객체를 유지해
        # 다른 곳이 들고 있는 참조가 어긋나지 않게 in-place로 바꾼다).
        messages[:] = compacted
        after_tokens = cm._estimate_tokens(messages)
        saved = max(0, before_tokens - after_tokens)
        table = Table(title="컨텍스트 압축", border_style="blue", box=ROUNDED)
        table.add_column("항목", style="cyan")
        table.add_column("이전", justify="right")
        table.add_column("이후", justify="right")
        table.add_row("메시지", f"{before_msgs:,}", f"{len(messages):,}")
        table.add_row("토큰(추정)", f"{before_tokens:,}", f"{after_tokens:,}")
        self.console.print(table)
        self.console.print(f"[green]약 {saved:,} 토큰을 절약했습니다.[/green]")

    async def _cmd_resume(self, args: list[str]) -> None:
        """/resume [번호] — 이전 CLI 세션을 골라 이어서 대화한다 (D3).

        [왜 session_id까지 재바인드하나]
          메시지만 복원하고 세션 ID를 그대로 두면, 이어서 한 대화가 "새 세션"
          트랜스크립트에 쌓여 원본과 갈라진다. 그래서 엔진의 세션 ID를 선택한
          세션으로 갈아끼워 같은 기록에 이어 쓰게 한다.
        """
        if not self._state or self._query_engine is None:
            self.console.print("[yellow]세션이 초기화되지 않았습니다.[/yellow]")
            return
        try:
            from core.memory.transcript import (
                list_transcript_sessions,
                read_transcript_messages,
            )

            sessions_dir = self._state.config.sessions_dir
            rows = list_transcript_sessions(sessions_dir, limit=20, channel="cli")
        except Exception as e:  # noqa: BLE001
            self.console.print(Text(f"세션 목록을 읽을 수 없습니다: {e}", style="red"))
            return

        # 지금 세션은 이어받을 대상이 아니므로 목록에서 뺀다.
        rows = [r for r in rows if r.get("session_id") != self._state.session_id]
        if not rows:
            self.console.print("[dim]이어받을 이전 세션이 없습니다.[/dim]")
            return

        # 인자가 없으면 목록만 보여 준다(번호를 보고 다시 부르게).
        if not args:
            table = Table(title="이전 CLI 세션", border_style="blue", box=ROUNDED)
            table.add_column("#", justify="right", style="cyan")
            table.add_column("세션 ID")
            table.add_column("최근 수정")
            table.add_column("제목/첫 메시지")
            for i, r in enumerate(rows, 1):
                table.add_row(
                    str(i),
                    str(r.get("session_id", ""))[:8],
                    str(r.get("modified", ""))[:19],
                    str(r.get("title_hint", ""))[:40],
                )
            self.console.print(table)
            self.console.print("[dim]이어받기: /resume <번호>[/dim]")
            return

        try:
            idx = int(args[0])
            target = rows[idx - 1]
        except (ValueError, IndexError):
            self.console.print(f"[yellow]1~{len(rows)} 사이의 번호를 지정하세요.[/yellow]")
            return

        session_id = str(target.get("session_id", ""))
        try:
            raw = read_transcript_messages(sessions_dir, session_id, channel="cli")
            from core.message import Message

            restored: list = []
            for entry in raw:
                role, content = entry.get("role"), (entry.get("content") or "")
                if role == "user":
                    restored.append(Message.user(content))
                elif role == "assistant":
                    restored.append(Message.assistant(text=content))
            # 세션 ID까지 갈아끼워 같은 트랜스크립트에 이어 쓰게 한다.
            self._state.session_id = session_id
            self._query_engine.bind_request(
                session_id=session_id,
                restore_messages=restored or None,
                channel="cli",
            )
        except Exception as e:  # noqa: BLE001
            self.console.print(Text(f"세션 복원 실패: {e}", style="red"))
            return

        self.console.print(
            f"[green]세션을 이어받았습니다:[/green] {session_id[:8]} "
            f"[dim]({len(restored)}개 메시지)[/dim]"
        )

    async def _cmd_cost(self, args: list[str]) -> None:
        """/cost — 이 세션의 누적 토큰 사용량을 보여준다 (D5).

        [왜 '비용'이 아니라 토큰인가] 에어갭 온프레미스라 API 과금이 없다.
        대신 컨텍스트 예산 관리에 실제로 쓸모 있는 토큰 수를 보여 준다.
        """
        if not self._state:
            self.console.print("[yellow]세션이 초기화되지 않았습니다.[/yellow]")
            return
        s = self._state.get_session_summary()
        table = Table(title="토큰 사용량", border_style="blue", box=ROUNDED)
        table.add_column("항목", style="cyan")
        table.add_column("값", justify="right")
        table.add_row("턴", f"{s.get('turns', 0):,}")
        table.add_row("입력 토큰", f"{s.get('total_input_tokens', 0):,}")
        table.add_row("출력 토큰", f"{s.get('total_output_tokens', 0):,}")
        table.add_row(
            "합계",
            f"{s.get('total_input_tokens', 0) + s.get('total_output_tokens', 0):,}",
        )
        table.add_row("도구 호출", f"{s.get('total_tool_calls', 0):,}")
        table.add_row("경과(초)", f"{s.get('total_duration_seconds', 0)}")
        self.console.print(table)
        self.console.print("[dim]온프레미스 실행이라 과금은 없습니다(토큰만 표시).[/dim]")

    async def _cmd_save(self, args: list[str]) -> None:
        """/save [파일명] — 이번 세션 대화를 Markdown으로 저장한다 (D7).

        트랜스크립트(JSONL)를 그대로 두고, 사람이 읽고 공유할 수 있는 형태로
        따로 내보낸다. 경로를 주지 않으면 현재 폴더에 세션 ID로 만든다.
        """
        if not self._state:
            self.console.print("[yellow]세션이 초기화되지 않았습니다.[/yellow]")
            return
        try:
            from pathlib import Path

            from core.memory.transcript import read_transcript_messages

            session_id = self._state.session_id
            rows = read_transcript_messages(
                self._state.config.sessions_dir, session_id, channel="cli"
            )
            if not rows:
                self.console.print("[yellow]저장할 대화가 없습니다.[/yellow]")
                return

            target = Path(args[0]) if args else Path(f"nexus_{session_id[:8]}.md")
            lines = [f"# Nexus 대화 기록 ({session_id})", ""]
            for entry in rows:
                role = entry.get("role", "")
                content = (entry.get("content") or "").strip()
                if not content:
                    continue
                label = {"user": "사용자", "assistant": "NOVA"}.get(role, role)
                lines.append(f"## {label}")
                lines.append("")
                lines.append(content)
                lines.append("")
            target.write_text("\n".join(lines), encoding="utf-8")
            self.console.print(
                f"[green]대화를 저장했습니다:[/green] {target} ({len(rows)}개 메시지)"
            )
        except Exception as e:  # noqa: BLE001 — 저장 실패가 세션을 끊으면 안 된다
            self.console.print(f"[red]저장 실패:[/red] {e}")

    async def _cmd_diff(self, args: list[str]) -> None:
        """/diff — 현재 작업 트리의 git 변경사항을 보여준다 (D4).

        로컬 git만 호출하므로 에어갭에서도 안전하다. 리포가 아니면 안내만 한다.
        """
        # 비동기 서브프로세스로 실행한다 — REPL은 async 루프 위에서 도므로
        # blocking subprocess.run을 쓰면 그 동안 입력·스트리밍이 모두 멈춘다.
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",  # noqa: S607 — PATH의 git을 쓰는 것이 의도된 동작
                "diff",
                "--stat",
                "--",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._state.cwd if self._state else None,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=30)
        except (OSError, TimeoutError) as e:
            self.console.print(f"[red]git 실행 실패:[/red] {e}")
            return
        if proc.returncode != 0:
            msg = stderr_b.decode("utf-8", "replace").strip() or "git 저장소가 아닙니다."
            # git 출력에는 대괄호가 섞일 수 있다. 그대로 넘기면 Rich가 markup으로
            # 해석해 MarkupError로 죽으므로, 외부 문자열은 항상 Text로 감싼다.
            self.console.print(Text(msg, style="yellow"))
            return
        out = stdout_b.decode("utf-8", "replace").strip()
        if not out:
            self.console.print("[dim]변경사항이 없습니다.[/dim]")
            return
        self.console.print(
            Panel(Text(out), title="git diff --stat", border_style="cyan", expand=False)
        )

    async def _cmd_copy(self, args: list[str]) -> None:
        """/copy — 마지막 NOVA 응답을 클립보드로 복사한다 (D6).

        pyperclip이 있으면 그것으로, 없으면 OSC52 이스케이프로 터미널에 맡긴다
        (원격 SSH 세션에서도 로컬 클립보드로 복사되는 표준 방식).
        """
        text = (self._last_response or "").strip()
        if not text:
            self.console.print("[yellow]복사할 응답이 없습니다.[/yellow]")
            return
        try:
            import pyperclip  # type: ignore[import-not-found]

            pyperclip.copy(text)
            self.console.print(f"[green]복사했습니다[/green] [dim]({len(text)}자)[/dim]")
            return
        except Exception as e:  # noqa: BLE001 — 미설치·환경 미지원이면 OSC52로 폴백
            logger.debug("pyperclip 복사 실패 — OSC52로 폴백: %s", e)
        try:
            import base64
            import sys

            b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
            sys.stdout.write(f"\033]52;c;{b64}\a")
            sys.stdout.flush()
            self.console.print(
                f"[green]복사 요청을 보냈습니다[/green] [dim]({len(text)}자, OSC52)[/dim]"
            )
        except Exception as e:  # noqa: BLE001
            self.console.print(f"[red]복사 실패:[/red] {e}")

    async def _cmd_thinking(self, args: list[str]) -> None:
        """/thinking — 모델의 사고(thinking) 블록을 화면에 표시할지 여부를 켜고 끈다."""
        # 포매터의 show_thinking 플래그를 반전시켜, 이후 이벤트 렌더링에 반영한다.
        self._formatter.show_thinking = not self._formatter.show_thinking
        status = "켜짐" if self._formatter.show_thinking else "꺼짐"
        self.console.print(f"[green]Thinking 표시: {status}[/green]")

    async def _cmd_verbose(self, args: list[str]) -> None:
        """/verbose — 도구 호출·결과를 한 줄 축약으로 볼지, 전문으로 볼지 토글한다 (D8).

        평소에는 축약(⏺/⎿)이 대화 흐름을 해치지 않아 좋고, 도구가 정확히 어떤
        인자로 불렸는지·결과 전문이 필요할 때만 켜면 된다.
        """
        self._formatter.verbose = not self._formatter.verbose
        status = "전문(verbose)" if self._formatter.verbose else "축약"
        self.console.print(f"[green]도구 표시: {status}[/green]")

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
                    f"턴: {summary.get('turns', 0)} | "
                    f"입력 토큰: {summary.get('total_input_tokens', 0):,} | "
                    f"출력 토큰: {summary.get('total_output_tokens', 0):,} | "
                    f"도구 호출: {summary.get('total_tool_calls', 0)}",
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
