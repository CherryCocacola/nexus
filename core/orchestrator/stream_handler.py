"""
스트리밍 도구 실행기 — 모델이 응답을 스트리밍하는 "도중에" 도구를 미리 실행하는
최적화 엔진.

이 파일이 하는 일 (한눈에):
  LLM(모델)은 답변을 한 글자씩 흘려보내며(SSE 스트리밍), 그 안에 "이 도구를
  써 줘"라는 tool_use 블록을 여러 개 만들어 낼 수 있다. 보통은 스트리밍이 전부
  끝난 뒤에야 도구를 하나씩 실행하지만, 그러면 모델이 말하는 동안 도구는 놀고
  있게 된다. 이 클래스는 "이미 완성된" tool_use 블록 중 동시 실행이 안전한
  것부터 백그라운드에서 곧바로 실행해 두어, 전체 턴(turn)의 지연 시간을 줄인다.

  원본 참고: Claude Code의 StreamingToolExecutor.ts를 파이썬(asyncio)으로 재구현.

핵심 아이디어:
  모델이 여러 tool_use 블록을 생성하는 동안, 이미 완성된 블록의 도구를
  백그라운드 asyncio Task로 미리 실행해 둔다. 모델 스트리밍이 끝나면
  아직 실행하지 못한(또는 순차 실행해야 하는) 나머지를 drain(마저 배출)한다.

주요 클래스:
  - StreamingToolExecutor: 아래 3단계 수명주기로 도구 실행을 관리한다.

동작 흐름 (한 턴 기준):
  1. add_tool(): 스트리밍 중 tool_use 블록이 하나 완성될 때마다 호출한다.
     - is_concurrency_safe=True  → 즉시 백그라운드 실행 시작(선행 실행)
     - is_concurrency_safe=False → 대기 큐에 보관(스트림 완료 후 순차 실행)
  2. get_completed(): 이미 끝난 도구 결과를 "기다리지 않고" 꺼낸다(비차단).
  3. drain_remaining(): 스트림 완료 후, 남은 모든 작업을 끝내고 결과를 yield한다.

왜 안전한 도구만 미리 실행하나:
  Read 같은 읽기 전용 도구는 서로 순서가 바뀌거나 동시에 돌아도 결과가
  달라지지 않는다(is_concurrency_safe=True). 반면 Edit/Write 같은 쓰기 도구는
  실행 순서가 결과를 바꿀 수 있어(파일을 덮어쓰는 등) 반드시 순서를 지켜
  스트림이 끝난 뒤 하나씩 실행한다. 이것이 fail-closed 기본값(P6)의 취지다.

시간 절약 예시:
  모델 출력: [Read(A), Read(B), Edit(C)]
  기존: 스트림 완료 → Read(A) → Read(B) → Edit(C) = 전체 시간
  개선: 스트리밍 중 Read(A) 시작 → Read(B) 시작 → 스트림 완료 → Edit(C)
       (스트림이 끝난 시점에 Read(A)/Read(B) 결과는 이미 준비되어 있음!)

호출 위치:
  core/orchestrator의 query_loop이 모델 스트리밍(Phase 2)과 도구 배출(Phase 4)
  단계에서 이 클래스를 사용한다. 실제 도구 실행은 core.tools.executor의
  run_tool_use()에 위임한다(권한 파이프라인·에러 래핑은 그쪽이 책임).

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.message import Message, StreamEvent
from core.tools.base import BaseTool, ToolUseContext
from core.tools.executor import run_tool_use

logger = logging.getLogger("nexus.orchestrator.stream_handler")

# 취소한 도구 Task 가 정리를 끝낼 때까지 기다려 주는 상한(초).
#   왜 상한이 필요한가: cancel_all()은 에러 복구 경로에서 불린다. 취소에 반응하지
#   않는 도구가 하나 있다고 해서 재시도 전체가 멈추면 안 된다. 정리를 기다리되
#   복구를 볼모로 잡지 않는 타협점이다.
_CANCEL_DRAIN_TIMEOUT = 5.0


class StreamingToolExecutor:
    """
    스트리밍 도구 실행 관리자 — "한 턴 동안만" 살아 있는 도구 실행 조율자.

    query_loop의 Phase 2(모델 스트리밍)에서 add_tool()로 도구를 접수하고,
    Phase 4(스트림 종료 후)에서 drain_remaining()으로 남은 것을 마저 실행한다.
    모델이 도구 호출을 생성하는 동안 동시 실행이 안전한 도구를 미리 실행하여
    전체 턴(turn)의 레이턴시(응답까지 걸리는 시간)를 줄이는 것이 목적이다.

    상태 관리 관점:
      - 백그라운드 실행 중인 것: _running_tasks (asyncio.Task 리스트)
      - 순차 실행 대기 중인 것: _deferred_calls (큐)
      - 이미 끝나 소비를 기다리는 결과: _completed
      한 턴이 끝나면 drain_remaining()/cancel_all()이 이 상태들을 비운다.

    동시성 주의:
      _completed는 여러 백그라운드 Task가 함께 쓰므로 _lock으로 보호한다.
      동시에 도는 도구 개수는 _semaphore(max_concurrent)로 상한을 건다.
    """

    def __init__(
        self,
        tools: list[BaseTool],
        context: ToolUseContext,
        tool_registry: Any | None = None,
        max_concurrent: int = 10,
    ):
        """
        실행기를 초기화한다. 보통 query_loop이 한 턴을 시작할 때 새로 만든다.

        Args:
            tools: 이번 턴에 사용 가능한 도구 객체 리스트(BaseTool 구현체들)
            context: 도구 실행 컨텍스트 — cwd, session_id, abort 시그널 등을 담음
            tool_registry: 도구 레지스트리. alias(별칭) 조회 등에 쓰이며 선택 사항
            max_concurrent: 백그라운드에서 동시에 돌릴 도구의 최대 개수(기본 10)
        """
        self._tools = tools
        self._context = context
        self._tool_registry = tool_registry
        self._max_concurrent = max_concurrent

        # 도구 이름 → 도구 객체로 빠르게 찾기 위한 조회 맵(dict).
        # 먼저 정식 이름(name)으로 채운 뒤,
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in tools}
        # 각 도구의 별칭(alias)들도 소문자로 등록해 둔다.
        # 모델이 별칭으로 도구를 부르더라도 찾을 수 있게 하기 위함이다.
        for t in tools:
            for alias in t.aliases:
                self._tool_map[alias.lower()] = t

        # ── 이 실행기의 실행 상태(모두 "한 턴" 동안만 유효) ──
        # 백그라운드에서 선행 실행 중인 도구 Task들.
        self._running_tasks: list[asyncio.Task] = []
        # 동시 실행이 불안전해 스트림 종료 후 순서대로 실행할 도구들의 대기 큐.
        self._deferred_calls: list[dict[str, Any]] = []  # 순차 실행 대기 큐
        # 이미 끝나 아직 상위(query_loop)로 배출되지 않은 결과들.
        self._completed: list[StreamEvent | Message] = []
        # _completed는 여러 백그라운드 Task가 동시에 append 하므로 락으로 보호.
        self._lock = asyncio.Lock()  # _completed 접근 동기화
        # 동시에 도는 도구 개수의 상한선. 자원 고갈(과도한 동시 실행)을 막는다.
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def add_tool(self, tool_use: dict[str, Any]) -> None:
        """
        스트리밍 도중 tool_use 블록이 "완성"될 때마다 query_loop이 호출한다.

        도구의 동시성 안전 플래그를 보고 두 갈래로 분기한다:
          - is_concurrency_safe=True  → 지금 즉시 백그라운드 Task로 선행 실행
          - is_concurrency_safe=False → 대기 큐에 넣어 두고 스트림 종료 후 순차 실행
        이렇게 나누는 이유는 모듈 상단 docstring의 "왜 안전한 도구만" 참고.

        Args:
            tool_use: 모델이 만든 도구 호출 한 건. {"id","name","input"} 형태의 dict.

        Returns:
            없음. 결과는 나중에 get_completed()/drain_remaining()으로 회수한다.
        """
        # 도구 이름을 꺼내 조회 맵에서 실제 도구 객체를 찾는다.
        # 이름이 없거나(빈 문자열) 미등록 도구면 tool 은 None 이 된다.
        name = tool_use.get("name", "")
        tool = self._tool_map.get(name)

        # 도구를 찾았고 동시 실행이 안전하면 → 지금 바로 백그라운드로 실행 시작.
        # (tool 이 None 이면 이 조건은 거짓이 되어 아래 else 로 빠져 안전하게 처리)
        if tool and tool.is_concurrency_safe:
            # 동시 실행 안전 → 즉시 백그라운드 실행 시작.
            # create_task 는 코루틴을 이벤트 루프에 등록만 하고 곧바로 반환하므로,
            # 여기서 도구가 끝나기를 기다리지 않는다(그래서 "선행 실행"이 된다).
            task = asyncio.create_task(
                self._execute_and_collect(tool_use),
                name=f"streaming_tool_{name}_{tool_use.get('id', '')}",
            )
            # 나중에 완료 여부를 확인하고 회수할 수 있도록 Task 를 보관해 둔다.
            self._running_tasks.append(task)
            logger.debug(f"StreamingToolExecutor: '{name}' 백그라운드 실행 시작")
        else:
            # 동시 실행 불안전(또는 미등록 도구) → 스트림 완료 후 순차 실행 예정.
            # 순서를 보존해야 하므로 append 로 큐 뒤에 쌓아 둔다.
            self._deferred_calls.append(tool_use)
            logger.debug(f"StreamingToolExecutor: '{name}' 순차 실행 대기열에 추가")

    def get_completed(self) -> list[StreamEvent | Message]:
        """
        이미 완료된 도구 결과를 "기다리지 않고" 한꺼번에 꺼낸다(비차단).

        스트리밍이 진행되는 동안 query_loop이 이 메서드를 주기적으로 불러,
        먼저 끝난 도구 결과를 그때그때 사용자에게 흘려보낼 수 있게 한다.
        (아직 안 끝난 도구를 기다리지 않으므로 스트리밍이 막히지 않는다.)

        Returns:
            완료된 StreamEvent/Message 리스트. 아직 끝난 게 없으면 빈 리스트.
        """
        # 완료분이 없으면 곧바로 빈 리스트를 돌려준다(복사 비용 회피).
        if not self._completed:
            return []
        # 스냅샷을 복사해 반환하고, 원본 버퍼는 비운다.
        # → 같은 결과를 다음 호출에서 중복으로 넘겨주는 일을 막는다.
        completed = self._completed.copy()
        self._completed.clear()
        return completed

    @property
    def pending_count(self) -> int:
        """
        아직 마무리되지 않은 작업 수(진행 지표).

        아직 안 끝난 백그라운드 Task 수 + 순차 실행 대기 큐 길이를 더한 값.
        query_loop이 "이 턴에 더 배출할 도구가 남았는지" 판단할 때 참고한다.
        """
        # 백그라운드 Task 중 아직 done() 이 아닌(=실행 중인) 것만 센다.
        running = sum(1 for t in self._running_tasks if not t.done())
        return running + len(self._deferred_calls)

    @property
    def has_deferred(self) -> bool:
        """순차 실행 대기 큐에 아직 처리할 도구가 남아 있는지 여부(True/False)."""
        return len(self._deferred_calls) > 0

    async def drain_remaining(self) -> asyncio.AsyncGenerator[StreamEvent | Message, None]:
        """
        스트림이 끝난 뒤, 남은 도구를 전부 마무리하고 결과를 순서대로 yield한다.

        이 메서드는 AsyncGenerator라서 `async for event in drain_remaining()`
        형태로 소비한다. query_loop의 Phase 4(스트림 종료 후 처리)에서 호출한다.

        실행 순서(3단계):
          1. 백그라운드에서 선행 실행 중이던 도구들이 모두 끝날 때까지 대기
          2. 그 결과(_completed에 쌓인 것)를 하나씩 yield
          3. 순차 실행이 필요한(지연된) 도구를 큐 순서대로 실행하며 yield

        모두 마치면 다음 턴을 위해 내부 상태를 깨끗이 비운다.
        """
        # 1단계: 백그라운드 작업이 모두 끝날 때까지 기다린다.
        #         asyncio.wait(ALL_COMPLETED)는 성공/실패와 무관하게
        #         전부 종료될 때까지 블록한다.
        if self._running_tasks:
            done, _ = await asyncio.wait(
                self._running_tasks,
                return_when=asyncio.ALL_COMPLETED,
            )
            # 실패한 Task 는 예외가 안에 삼켜져 있으므로, 여기서 꺼내 로그로 남긴다.
            # (도구 실행 실패 자체는 _execute_and_collect 에서 tool_use_error 로
            #  이미 결과화되지만, 그 밖의 예기치 못한 예외를 놓치지 않기 위함이다.)
            for task in done:
                if task.exception():
                    logger.error(f"StreamingToolExecutor 작업 실패: {task.exception()}")

        # 2단계: 선행 실행으로 이미 쌓인 완료 결과를 순서대로 배출한 뒤 버퍼를 비운다.
        for event in self._completed:
            yield event
        self._completed.clear()

        # 3단계: 순차 실행 대기 큐를 앞에서부터 하나씩 실행한다.
        # is_concurrency_safe=False 도구(Edit/Write 등)는 순서가 결과에 영향을
        # 주므로, 반드시 큐에 담긴 순서대로 한 건씩 끝내며 이벤트를 흘려보낸다.
        for tool_use in self._deferred_calls:
            async for event in run_tool_use(
                tool_use,
                self._tools,
                self._context,
                self._tool_registry,
            ):
                yield event

        # 정리 — 다음 턴에서 이 실행기를 재사용해도 깨끗하도록 상태를 초기화.
        self._running_tasks.clear()
        self._deferred_calls.clear()

    async def cancel_all(self) -> None:
        """
        진행 중인 모든 도구 실행을 취소하고 내부 상태를 전부 비운다.

        언제 부르나:
          모델 스트리밍이 도중에 실패해 되돌려야 할 때(query_loop의 에러 복구
          Transition 1~2)나, 컨텍스트 압축 후 턴을 다시 시작해야 할 때 호출한다.
          이때 선행 실행해 둔 도구 결과는 더 이상 쓸모가 없으므로 함께 폐기한다.
        """
        # 아직 안 끝난 백그라운드 Task 에 취소 신호를 보낸다.
        pending = [t for t in self._running_tasks if not t.done()]
        for task in pending:
            task.cancel()

        # ★취소가 실제로 끝날 때까지 기다린다(2026-08-24).
        #   예전에는 신호만 보내고 곧바로 참조를 비웠다. 그러면 아직 정리 중인
        #   도구 코루틴(과 그 안의 async generator)이 그대로 GC 로 넘어간다.
        #   그 상태에서 파이썬의 asyncgen finalizer 가 닫으려 들면
        #   `aclose(): asynchronous generator is already running` 이 날 수 있고,
        #   취소된 Task 를 아무도 회수하지 않아 "Task exception was never retrieved"
        #   경고도 남는다. 기다렸다 버리면 둘 다 사라진다.
        #
        #   ※상한을 두는 이유: 이 함수는 **에러 복구 경로**에서 불린다. 취소에
        #     반응하지 않는 도구(예: 블로킹 서브프로세스)가 하나라도 있으면
        #     재시도가 통째로 멈춘다. 정리를 기다리되 복구를 볼모로 잡지는 않는다.
        if pending:
            try:
                await asyncio.wait(pending, timeout=_CANCEL_DRAIN_TIMEOUT)
            except Exception:  # noqa: BLE001 — 정리 실패가 복구를 막으면 안 된다
                logger.debug("도구 취소 대기 중 예외(무시)", exc_info=True)
        # 세 버퍼를 모두 비워 이 실행기를 초기 상태로 되돌린다.
        self._running_tasks.clear()
        self._deferred_calls.clear()
        self._completed.clear()
        logger.info("StreamingToolExecutor: 모든 작업 취소됨")

    # ─── 내부 메서드(클래스 외부에서 직접 부르지 않음) ───

    async def _execute_and_collect(self, tool_use: dict[str, Any]) -> None:
        """
        백그라운드 Task 본체 — 도구 하나를 실행하고 결과를 _completed에 모은다.

        add_tool()이 create_task()로 이 코루틴을 띄운다. 실제 도구 실행은
        run_tool_use()에 위임하며, 그 도중 나오는 이벤트/메시지를 모아 두었다가
        마지막에 한 번에 _completed로 옮긴다.

        핵심 설계 두 가지:
          - 세마포어로 동시 실행 개수를 상한(기본 10)으로 묶어 자원 고갈을 막는다.
          - 예외가 나도 그냥 죽지 않고 tool_use_error 메시지로 감싸 결과에 넣는다.
            → 상위(query_loop)는 실패도 "정상적인 도구 결과"로 일관되게 처리한다.
        """
        # 세마포어 획득 → 동시에 이 블록에 들어올 수 있는 Task 수를 제한.
        async with self._semaphore:
            results: list[StreamEvent | Message] = []
            try:
                # 도구 실행 스트림을 소비하며 나오는 이벤트를 임시 리스트에 모은다.
                async for event in run_tool_use(
                    tool_use,
                    self._tools,
                    self._context,
                    self._tool_registry,
                ):
                    results.append(event)
            except Exception as e:
                # 예상치 못한 예외도 삼키지 않고 오류 결과로 변환해 흐름을 잇는다.
                logger.error(f"StreamingToolExecutor 실행 에러: {e}")
                results.append(
                    Message.tool_result(
                        tool_use.get("id", ""),
                        f"<tool_use_error>실행 실패: {e}</tool_use_error>",
                        is_error=True,
                    )
                )

            # 여러 Task 가 동시에 _completed 에 쓰면 리스트가 깨질 수 있으므로,
            # 락을 잡은 상태에서만 결과를 옮겨 담는다(동시 접근 보호).
            async with self._lock:
                self._completed.extend(results)
