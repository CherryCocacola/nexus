"""
스트리밍 도구 실행기 — 모델 스트리밍 중 도구를 병렬 실행하는 최적화 엔진.

Claude Code의 StreamingToolExecutor.ts를 재구현한다.

핵심 아이디어:
  모델이 여러 tool_use 블록을 생성하는 동안, 이미 완성된 블록의 도구를
  백그라운드에서 미리 실행한다. 모델 스트리밍이 끝나면 나머지를 drain한다.

동작 흐름:
  1. add_tool(): 스트리밍 중 tool_use 블록이 감지되면 호출
     - is_concurrency_safe=True → 즉시 백그라운드 실행 시작
     - is_concurrency_safe=False → 큐에 보관 (스트림 완료 후 순차 실행)
  2. get_completed(): 완료된 도구 결과를 비차단으로 꺼낸다
  3. drain_remaining(): 스트림 완료 후 모든 미완료 작업을 완료하고 yield

시간 절약 예시:
  모델 출력: [Read(A), Read(B), Edit(C)]
  기존: 스트림 완료 → Read(A) → Read(B) → Edit(C) = 전체 시간
  개선: 스트리밍 중 Read(A) 시작 → Read(B) 시작 → 스트림 완료 → Edit(C)
       Read 결과는 이미 완료되어 있음!
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.message import Message, StreamEvent
from core.tools.base import BaseTool, ToolUseContext
from core.tools.executor import run_tool_use

logger = logging.getLogger("nexus.orchestrator.stream_handler")


class StreamingToolExecutor:
    """
    스트리밍 도구 실행 관리자.

    query_loop의 Phase 2(모델 스트리밍)에서 사용한다.
    모델이 도구 호출을 생성하는 동안 읽기 전용 도구를 미리 실행하여
    전체 턴의 레이턴시를 줄인다.
    """

    def __init__(
        self,
        tools: list[BaseTool],
        context: ToolUseContext,
        tool_registry: Any | None = None,
        max_concurrent: int = 10,
    ):
        """
        Args:
            tools: 사용 가능한 도구 리스트
            context: 도구 실행 컨텍스트 (cwd, session_id 등)
            tool_registry: 도구 레지스트리 (alias 조회용, 선택)
            max_concurrent: 최대 동시 실행 수 (기본: 10)
        """
        self._tools = tools
        self._context = context
        self._tool_registry = tool_registry
        self._max_concurrent = max_concurrent

        # 도구 이름 → 도구 객체 매핑 (빠른 조회용)
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in tools}
        for t in tools:
            for alias in t.aliases:
                self._tool_map[alias.lower()] = t

        # 실행 상태
        self._running_tasks: list[asyncio.Task] = []
        self._deferred_calls: list[dict[str, Any]] = []  # 순차 실행 대기 큐
        self._completed: list[StreamEvent | Message] = []
        self._lock = asyncio.Lock()  # _completed 접근 동기화
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def add_tool(self, tool_use: dict[str, Any]) -> None:
        """
        스트리밍 중 tool_use 블록이 감지되면 호출한다.

        동시 실행이 안전한 도구(is_concurrency_safe=True)는 즉시
        백그라운드 실행을 시작하고, 그렇지 않은 도구는 큐에 보관한다.

        Args:
            tool_use: {"id": "...", "name": "...", "input": {...}}
        """
        name = tool_use.get("name", "")
        tool = self._tool_map.get(name)

        if tool and tool.is_concurrency_safe:
            # 동시 실행 안전 → 즉시 백그라운드 실행 시작
            task = asyncio.create_task(
                self._execute_and_collect(tool_use),
                name=f"streaming_tool_{name}_{tool_use.get('id', '')}",
            )
            self._running_tasks.append(task)
            logger.debug(f"StreamingToolExecutor: '{name}' 백그라운드 실행 시작")
        else:
            # 동시 실행 불안전 → 스트림 완료 후 순차 실행 예정
            self._deferred_calls.append(tool_use)
            logger.debug(f"StreamingToolExecutor: '{name}' 순차 실행 대기열에 추가")

    def get_completed(self) -> list[StreamEvent | Message]:
        """
        완료된 도구 결과를 비차단으로 꺼낸다.

        스트리밍 중에 주기적으로 호출하여 이미 완료된
        도구 결과를 소비할 수 있다.

        Returns:
            완료된 이벤트/메시지 리스트 (비어 있을 수 있음)
        """
        if not self._completed:
            return []
        completed = self._completed.copy()
        self._completed.clear()
        return completed

    @property
    def pending_count(self) -> int:
        """아직 완료되지 않은 작업 수."""
        running = sum(1 for t in self._running_tasks if not t.done())
        return running + len(self._deferred_calls)

    @property
    def has_deferred(self) -> bool:
        """순차 실행을 대기 중인 도구가 있는지 여부."""
        return len(self._deferred_calls) > 0

    async def drain_remaining(self) -> asyncio.AsyncGenerator[StreamEvent | Message, None]:
        """
        스트림 완료 후: 모든 미완료 작업을 완료하고 결과를 yield한다.

        실행 순서:
          1. 백그라운드 동시 실행 작업 완료 대기
          2. 완료된 결과 yield
          3. 지연된 도구를 순차 실행 + yield

        이 메서드는 query_loop의 Phase 4에서 호출된다.
        """
        # 1단계: 백그라운드 작업 완료 대기
        if self._running_tasks:
            done, _ = await asyncio.wait(
                self._running_tasks,
                return_when=asyncio.ALL_COMPLETED,
            )
            # 실패한 작업의 예외를 로깅
            for task in done:
                if task.exception():
                    logger.error(f"StreamingToolExecutor 작업 실패: {task.exception()}")

        # 2단계: 완료된 결과를 yield
        for event in self._completed:
            yield event
        self._completed.clear()

        # 3단계: 지연된 도구를 순차 실행
        # is_concurrency_safe=False인 도구들은 여기서 하나씩 실행한다
        for tool_use in self._deferred_calls:
            async for event in run_tool_use(
                tool_use,
                self._tools,
                self._context,
                self._tool_registry,
            ):
                yield event

        # 정리 — 다음 턴을 위해 상태를 초기화
        self._running_tasks.clear()
        self._deferred_calls.clear()

    async def cancel_all(self) -> None:
        """
        모든 진행 중인 도구 실행을 취소한다.

        query_loop에서 에러 복구(Transition 1~2)나
        컨텍스트 압축 재시도 시 호출한다.
        """
        for task in self._running_tasks:
            if not task.done():
                task.cancel()
        self._running_tasks.clear()
        self._deferred_calls.clear()
        self._completed.clear()
        logger.info("StreamingToolExecutor: 모든 작업 취소됨")

    # ─── 내부 메서드 ───

    async def _execute_and_collect(self, tool_use: dict[str, Any]) -> None:
        """
        백그라운드에서 도구를 실행하고 결과를 _completed에 추가한다.

        세마포어로 동시 실행 수를 제한한다 (기본 10개).
        에러 발생 시에도 tool_use_error를 _completed에 추가하여
        query_loop에서 정상적으로 처리할 수 있게 한다.
        """
        async with self._semaphore:
            results: list[StreamEvent | Message] = []
            try:
                async for event in run_tool_use(
                    tool_use,
                    self._tools,
                    self._context,
                    self._tool_registry,
                ):
                    results.append(event)
            except Exception as e:
                logger.error(f"StreamingToolExecutor 실행 에러: {e}")
                results.append(
                    Message.tool_result(
                        tool_use.get("id", ""),
                        f"<tool_use_error>실행 실패: {e}</tool_use_error>",
                        is_error=True,
                    )
                )

            # 락으로 _completed 동시 접근을 보호
            async with self._lock:
                self._completed.extend(results)
