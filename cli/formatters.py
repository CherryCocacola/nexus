"""
출력 포매터(OutputFormatter) — 스트리밍 이벤트를 터미널 화면용 Rich 객체로 변환한다.

[이 파일이 하는 일]
Nexus는 모델의 응답을 한 번에 완성해서 주는 게 아니라, 4-Tier AsyncGenerator
체인을 통해 잘게 쪼갠 조각(StreamEvent)을 실시간으로 흘려보낸다.
이 파일은 그 체인의 "가장 바깥(최종 출력) 단계"에서 StreamEvent 하나하나를 받아,
사람이 터미널에서 보기 좋은 형태(Rich의 Panel / Markdown / Text / 색이 입혀진 문자열)로
바꿔주는 번역기 역할을 한다. 즉, "내부 데이터 → 화면에 보일 모양"을 담당한다.

[주요 구성]
- OutputFormatter 클래스: 유일한 공개 클래스. 아래 메서드들로 이벤트 종류별 출력을 만든다.
  - format_text_delta : 모델이 흘려보내는 텍스트 조각 처리(+ Qwen thinking 필터링)
  - format_tool_use   : 도구 호출(이름 + 입력 인자)을 JSON 패널로 표시
  - format_tool_result: 도구 실행 결과를 성공/에러 색상 패널로 표시
  - format_thinking   : 모델의 사고(thinking) 과정을 패널로 표시(디버그용)
  - format_error      : 시스템/네트워크/권한 에러 메시지를 패널로 표시
  - format_usage      : 토큰 사용량을 REPL 하단용 한 줄 문자열로 표시
  - format_event      : StreamEvent 타입을 보고 위 메서드 중 알맞은 것으로 라우팅하는 진입점

[의존성 방향] cli/ → core/ (단방향). 이 파일은 core.message의 StreamEvent 등
데이터 타입을 "읽기만" 하며, core 쪽이 cli를 거꾸로 import 하지 않는다.
화면 렌더링은 외부 라이브러리 Rich(rich.panel/markdown/syntax/text)에 의존한다.

[호출 관계] CLI REPL(예: cli/repl.py)이 query_loop에서 yield된 StreamEvent를
받아 이 OutputFormatter.format_event()에 넘기고, 반환된 Rich 객체를 콘솔에 출력한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from core.message import StreamEvent, StreamEventType, TokenUsage

# 이 모듈 전용 로거. "nexus.cli.formatters" 네임스페이스로 통일해
# 로그를 모듈별로 필터링/추적하기 쉽게 한다.
logger = logging.getLogger("nexus.cli.formatters")


class OutputFormatter:
    """
    StreamEvent(스트리밍 이벤트)를 Rich 렌더러블 객체로 변환하는 포매터.

    [역할] query_loop 등 4-Tier 체인에서 yield된 StreamEvent를 하나씩 받아서,
    터미널에 그대로 출력할 수 있는 Rich 객체(Panel, Markdown, Text)나
    색상 마크업이 포함된 문자열로 바꿔준다.

    [왜 필요한가] "무슨 데이터인가(core 계층의 StreamEvent)"와
    "어떻게 보여줄까(cli 계층의 화면 표현)"를 분리하기 위함이다.
    이렇게 분리하면 출력 모양을 바꿔도 core 로직은 건드릴 필요가 없다.

    [상태] thinking 필터링을 위해 인스턴스가 약간의 상태(_in_thinking,
    _thinking_buffer)를 들고 있다. 하나의 대화 스트림을 처리하는 동안
    이 포매터 인스턴스를 재사용한다는 전제로 동작한다.
    """

    def __init__(self, show_thinking: bool = False):
        """
        포매터를 초기화한다.

        Args:
            show_thinking: 모델의 thinking(사고) 블록을 화면에 표시할지 여부.
                           평소에는 사고 과정이 지저분하므로 False(숨김)가 기본이며,
                           디버깅으로 모델의 추론을 들여다볼 때만 True로 켠다.
        """
        # 내부 보관 필드. 외부에서는 show_thinking 프로퍼티(getter/setter)로 접근한다.
        self._show_thinking = show_thinking
        # thinking 필터링 상태 — 반드시 인스턴스 필드로 둔다(클래스 변수로 두면
        # 스트림 간 상태가 누출된다). _in_thinking: 사고 구간 통과 중 여부,
        # _thinking_buffer: 사고 구간 동안 누적한 텍스트(요약용).
        self._in_thinking = False
        self._thinking_buffer = ""

    def reset_stream_state(self) -> None:
        """새 응답 스트림 시작 시 thinking 필터 상태를 초기화한다.

        [왜 필요한가] 포매터 인스턴스는 REPL 전체 수명 동안 재사용된다. 만약 한
        응답이 사고 구간을 닫지 못한 채 끝나면(_in_thinking=True로 남으면), 리셋이
        없을 경우 다음 턴의 정상 응답까지 계속 삼켜 버린다. 매 응답 시작 시 이
        메서드를 호출해 상태를 깨끗이 비운다(repl._process_message 진입부에서 호출).
        """
        self._in_thinking = False
        self._thinking_buffer = ""

    # ─── 텍스트 델타 ───

    def format_text_delta(self, text: str) -> str:
        """
        모델이 흘려보내는 텍스트 조각(delta) 하나를 받아, 화면에 실제로 보여줄
        문자열로 가공해 반환한다.

        [배경] Qwen 3.5는 응답 앞부분에 사고 과정(thinking)을 먼저 출력하는데,
        이건 사용자에게 그대로 보이면 지저분하다. 그래서 여기서 사고 부분을
        감지해 걸러내고, 실제 답변만 통과시킨다. (사고는 한 줄 요약만 흐리게 표시)

        [핵심 흐름]
          1) '</think>' 태그가 보이면 → 사고 구간이 끝난 것. 태그 뒤의 실제 답변만 반환.
          2) 아직 아무것도 못 봤는데 조각이 사고처럼 시작하면 → 사고 시작으로 판단해 버퍼링.
          3) 사고 구간 중이면 → 화면에 내보내지 않고 버퍼에만 쌓음(빈 문자열 반환).
          4) 사고와 무관한 일반 텍스트면 → 그대로 반환.

        Args:
            text: 모델이 보낸 텍스트 조각 하나(스트리밍의 한 delta).

        Returns:
            화면에 출력할 문자열. 사고 구간처럼 숨겨야 하면 빈 문자열("")을 반환한다.
        """
        # (1) '</think>' 감지 — 사고 구간이 여기서 종료된다.
        if "</think>" in text:
            # 태그를 기준으로 앞(사고 잔여분)/뒤(실제 답변)로 한 번만 분리한다.
            parts = text.split("</think>", 1)
            self._in_thinking = False
            # 태그 뒤가 실제 응답. 맨 앞의 개행은 보기 싫으므로 제거한다.
            after = parts[1].lstrip("\n")
            if self._thinking_buffer:
                # 그동안 쌓아둔 사고 내용을 앞 80자만 잘라 한 줄로 요약한다.
                short = self._thinking_buffer[:80].replace("\n", " ")
                self._thinking_buffer = ""
                # 요약에 실제 내용이 있을 때만 흐린 이탤릭체 'thinking:' 한 줄을 앞에 붙인다.
                prefix = (
                    f"[dim italic]  thinking: {short}...[/dim italic]\n"
                    if short.strip() else ""
                )
                return prefix + after
            return after

        # (2) 스트림 맨 앞에서 리터럴 '<think>' 태그로 시작할 때만 사고 구간으로 진입한다.
        #     [변경 근거] 과거에는 "먼저"·"분석"·"사용자가"·"let me" 같은 접두어로
        #     사고 시작을 추측했으나, 이는 한국어 정상 응답(대부분 "먼저 …"로 시작)을
        #     통째로 삼키는 심각한 버그였다. 태그 기반은 태그가 실제로 나타날 때만
        #     작동하므로 정상 응답 오탐이 0이다.
        if not self._in_thinking and not self._thinking_buffer:
            if text.lstrip().startswith("<think>"):
                self._in_thinking = True
                self._thinking_buffer = text
                # 사고는 화면에 바로 내보내지 않는다.
                return ""

        # (3) 사고 구간을 지나는 중이면, 화면 출력 없이 버퍼에만 계속 쌓는다.
        if self._in_thinking:
            self._thinking_buffer += text
            return ""

        # (4) 사고와 무관한 평범한 답변 조각 → 그대로 화면에 흘려보낸다.
        return text

    # ─── 도구 사용 ───

    def format_tool_use(self, tool_name: str, input_data: dict[str, Any]) -> Panel:
        """
        도구 호출 정보(도구 이름 + 입력 인자)를 Rich Panel로 만들어 반환한다.

        [왜] 모델이 어떤 도구를(tool_name) 어떤 인자로(input_data) 호출했는지
        사용자가 한눈에 확인할 수 있게 JSON을 예쁘게(하이라이팅) 보여준다.

        Args:
            tool_name: 호출된 도구의 이름(예: "Read", "Bash").
            input_data: 도구에 전달된 입력 인자 딕셔너리.

        Returns:
            시안(cyan) 테두리에 JSON 하이라이팅이 들어간 Rich Panel.
        """
        # json은 이 메서드에서만 필요하므로 지역 import로 둔다.
        import json

        # 입력 인자를 사람이 읽기 좋은 들여쓰기 JSON으로 변환한다.
        # ensure_ascii=False로 한글이 유니코드 이스케이프 없이 그대로 보이게 한다.
        input_json = json.dumps(input_data, indent=2, ensure_ascii=False)

        # 위 JSON 문자열에 'json' 문법 하이라이팅(monokai 테마)을 입힌다. 줄 번호는 생략.
        syntax = Syntax(input_json, "json", theme="monokai", line_numbers=False)

        # 제목에 도구 이름을 넣고, 시안색 테두리 패널로 감싼다.
        # expand=False → 내용 폭에 맞춰 패널이 딱 붙게(터미널 폭 전체로 늘어나지 않게) 한다.
        return Panel(
            syntax,
            title=f"[bold cyan]Tool: {tool_name}[/bold cyan]",
            border_style="cyan",
            expand=False,
        )

    # ─── 도구 결과 ───

    def format_tool_result(self, content: str, is_error: bool = False) -> Panel:
        """
        도구 실행 결과를 Rich Panel로 만들어 반환한다.

        [규칙] 에러면 빨간색 테두리, 정상이면 녹색 테두리로 구분한다.
        정상 결과가 너무 길면 터미널 가독성을 위해 앞부분만 남기고 축약한다.

        Args:
            content: 도구가 돌려준 결과 텍스트.
            is_error: 결과가 에러인지 여부(True면 빨간색 에러 패널).

        Returns:
            성공/에러에 따라 색이 다른 Rich Panel.
        """
        if is_error:
            # 에러 결과: 빨간 글씨 + 빨간 테두리로 눈에 띄게 표시한다.
            return Panel(
                Text(content, style="red"),
                title="[bold red]Error[/bold red]",
                border_style="red",
                expand=False,
            )

        # 정상 결과: 결과가 길면 잘라서 보여준다(터미널이 도배되는 것을 방지).
        # 화면에 남길 최대 줄 수. 이보다 길면 뒷부분을 잘라내고 생략 안내를 붙인다.
        max_lines = 50
        lines = content.split("\n")
        if len(lines) > max_lines:
            # 앞에서 max_lines 줄만 남기고, 몇 줄을 생략했는지 안내 문구를 덧붙인다.
            truncated = "\n".join(lines[:max_lines])
            truncated += f"\n... ({len(lines) - max_lines}줄 생략)"
        else:
            # 짧으면 그대로 사용한다.
            truncated = content

        # 정상 결과는 녹색 테두리 패널로 감싼다.
        return Panel(
            truncated,
            title="[bold green]Result[/bold green]",
            border_style="green",
            expand=False,
        )

    # ─── 계획 체크리스트 (TodoWrite/TodoRead) ───

    @staticmethod
    def _looks_like_checklist(content: str) -> bool:
        """도구 결과 본문이 계획 체크리스트인지 판별한다(TodoWrite/TodoRead 결과 감지).

        TOOL_RESULT StreamEvent에는 도구명·metadata가 실리지 않으므로, 렌더된 본문의
        서명(체크박스 마커 또는 갱신 요약 머리말)으로 판별한다.
        """
        if content.startswith(("체크리스트 갱신됨", "(체크리스트가 비어")):
            return True
        # 마커 줄([x]/[~]/[ ])이 하나라도 있으면 체크리스트로 본다.
        return any(
            line.startswith(("[x] ", "[~] ", "[ ] ")) for line in content.split("\n")
        )

    def format_todo_list(self, content: str) -> Panel:
        """계획 체크리스트 본문을 색상 있는 Rich Panel로 렌더한다.

        [x] 완료(취소선/흐림), [~] 진행 중(강조), [ ] 대기 로 아이콘·색을 입힌다.
        요약 머리말(체크리스트 갱신됨 …)이나 경고 줄은 그대로 통과시킨다.
        """
        rendered = Text()
        for i, line in enumerate(content.split("\n")):
            if i:
                rendered.append("\n")
            if line.startswith("[x] "):
                rendered.append("✔ ", style="green")
                rendered.append(line[4:], style="dim strike")
            elif line.startswith("[~] "):
                rendered.append("◐ ", style="yellow")
                rendered.append(line[4:], style="bold")
            elif line.startswith("[ ] "):
                rendered.append("○ ", style="dim")
                rendered.append(line[4:])
            else:
                # 요약 머리말·경고·빈 목록 안내 등은 흐리게 그대로 표시.
                rendered.append(line, style="dim")
        return Panel(
            rendered,
            title="[bold cyan]체크리스트[/bold cyan]",
            border_style="cyan",
            expand=False,
        )

    # ─── 사고(Thinking) ───

    def format_thinking(self, text: str) -> Panel:
        """
        모델의 사고(thinking) 과정을 Rich Panel로 만들어 반환한다.

        [주의] show_thinking이 False일 때는 이 메서드가 빈 Panel을 만들어 반환하는 게
        아니라, 애초에 이 메서드를 "호출하지 않는" 것이 규칙이다. 즉 표시 여부 판단은
        호출자(아래 format_event) 책임이며, 이 메서드는 표시가 확정된 경우에만 불린다.

        Args:
            text: 모델의 사고 과정 텍스트(Markdown으로 렌더링됨).

        Returns:
            노란색 테두리의 Rich Panel.
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
        시스템 레벨 에러 메시지를 Rich Panel로 만들어 반환한다.

        [용도] 모델 에러, 네트워크(GPU 서버 통신) 에러, 권한 거부 등
        다양한 종류의 에러를 사용자에게 빨간색으로 눈에 띄게 알린다.

        Args:
            message: 표시할 에러 메시지.

        Returns:
            굵은 빨간 글씨 + 빨간 테두리의 Rich Panel.
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
        토큰 사용량을 REPL 하단에 표시할 한 줄 문자열로 만든다.

        [왜 문자열인가] 다른 포매터들은 Panel을 반환하지만, 사용량은 상태표시줄처럼
        간결하게 한 줄로 보여주는 게 낫기 때문에 색 마크업이 들어간 문자열을 반환한다.

        Args:
            usage: 입력/출력/캐시 토큰 수를 담은 TokenUsage 객체.

        Returns:
            "[dim]토큰 | 입력: ... | 출력: ... | 합계: ...[/dim]" 형태의 흐린 한 줄 문자열.
        """
        # 필수로 항상 보여줄 입력/출력 토큰 수. ','는 천 단위 구분 기호.
        parts = [
            f"입력: {usage.input_tokens:,}",
            f"출력: {usage.output_tokens:,}",
        ]
        # 프롬프트 캐시를 사용한 경우에만 캐시 관련 항목을 덧붙인다(값이 0이면 노이즈이므로 생략).
        if usage.cache_read_input_tokens > 0:
            parts.append(f"캐시읽기: {usage.cache_read_input_tokens:,}")
        if usage.cache_creation_input_tokens > 0:
            parts.append(f"캐시생성: {usage.cache_creation_input_tokens:,}")

        # 각 항목을 ' | '로 이어 붙이고, 흐린(dim) 스타일로 감싸 한 줄로 반환한다.
        return f"[dim]토큰 | {' | '.join(parts)} | 합계: {usage.total_tokens:,}[/dim]"

    # ─── StreamEvent 라우터 ───

    @property
    def show_thinking(self) -> bool:
        """현재 thinking(사고) 표시 여부를 반환한다."""
        return self._show_thinking

    @show_thinking.setter
    def show_thinking(self, value: bool) -> None:
        """thinking(사고) 표시 여부를 설정한다(런타임에 켜고 끌 수 있게 setter 제공)."""
        self._show_thinking = value

    def format_event(self, event: StreamEvent) -> Any | None:
        """
        StreamEvent 하나를 받아, 그 종류(type)에 맞는 format_* 메서드로 넘겨
        Rich 렌더러블(또는 문자열)로 변환하는 "라우터"이자 이 클래스의 진입점.

        [흐름] CLI REPL이 스트림에서 이벤트를 뽑을 때마다 이 메서드를 호출한다.
        이벤트 타입을 순서대로 검사해 알맞은 포맷 메서드를 부르고, 화면에 보여줄
        필요가 없는 이벤트(예: 내부용 이벤트, 사고 숨김 상태 등)는 None을 반환한다.

        Args:
            event: 4-Tier 체인에서 yield된 StreamEvent.

        Returns:
            화면에 출력할 Rich 렌더러블 객체 또는 문자열. 표시할 게 없으면 None.
        """
        # 이벤트의 종류를 먼저 꺼내 둔다(아래에서 반복 비교).
        event_type = event.type

        # 텍스트 델타 — 모델 답변 조각을 실시간으로 흘려 출력한다.
        if event_type == StreamEventType.TEXT_DELTA and event.text:
            return self.format_text_delta(event.text)

        # 도구 사용 — TOOL_USE_STOP 시점(=입력 인자가 완성된 순간)에 한 번만 표시한다 (v0.14.11).
        #   · TOOL_USE_START: 도구 이름만 도착한 시점이라 input이 아직 {}(빈 값)이다.
        #   · TOOL_USE_DELTA: 인자(arguments)가 조각조각 도착하며 누적된다.
        #   · TOOL_USE_STOP : 조각들이 합쳐져 완성된 input이 채워진다 → 이때 표시해야 정확하다.
        # 그래서 START에서 표시하면 항상 빈 {}만 보이는 문제가 있어, STOP에서 한 번만 표시한다.
        if event_type == StreamEventType.TOOL_USE_STOP and event.tool_use:
            return self.format_tool_use(
                tool_name=event.tool_use.name,
                input_data=event.tool_use.input,
            )

        # 도구 결과 — 도구가 실제로 실행되고 나온 결과를 표시한다.
        if event_type == StreamEventType.TOOL_RESULT and event.tool_result:
            content = event.tool_result.content
            # 계획 체크리스트(TodoWrite/TodoRead) 결과는 전용 체크박스 패널로 표시한다
            # (일반 결과 패널 대신 — 진행 상황을 한눈에 보이게).
            if not event.tool_result.is_error and self._looks_like_checklist(content):
                return self.format_todo_list(content)
            return self.format_tool_result(
                content=content,
                is_error=event.tool_result.is_error,
            )

        # 사고(Thinking) 델타 — show_thinking이 켜진 디버그 상황에서만 패널로 표시한다.
        if event_type == StreamEventType.THINKING_DELTA and event.thinking_text:
            if self._show_thinking:
                return self.format_thinking(event.thinking_text)
            # 사고 표시가 꺼져 있으면 화면에 아무것도 내보내지 않는다.
            return None

        # 에러 — 에러 메시지를 빨간 패널로 표시한다.
        if event_type == StreamEventType.ERROR and event.message:
            return self.format_error(event.message)

        # 사용량 업데이트 — 토큰 사용량을 한 줄로 표시한다.
        if event_type == StreamEventType.USAGE_UPDATE and event.usage:
            return self.format_usage(event.usage)

        # 위에서 걸리지 않은 그 외 이벤트(내부용 등)는 화면에 표시할 필요가 없으므로 None.
        return None
