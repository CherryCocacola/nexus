"""
TaskManager — 비동기 태스크 라이프사이클 관리.

Claude Code의 TaskManager를 Python으로 재구현한다.
로컬/원격 에이전트, Bash, 학습, 모니터링 등 배경 작업의
생성 → 실행 → 완료/실패/강제종료 라이프사이클을 관리한다.

핵심 역할:
  1. 태스크 생성 및 ID 부여 (타입별 prefix + uuid[:8])
  2. 코루틴을 asyncio.Task로 실행 (상태 자동 전환)
  3. fire-and-forget 배경 실행
  4. 강제 종료 (asyncio.Task.cancel())
  5. 진행률 추적 (0.0~1.0)
  6. 완료 콜백 통지
  7. 오래된 완료 태스크 정리

의존성 방향:
  core/task.py는 독립 모듈 — core/ 내 다른 모듈을 import하지 않는다.
  task_tools.py, bootstrap.py 등에서 이 모듈을 import한다.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("nexus.task")


# ─────────────────────────────────────────────
# TaskType — 태스크 유형 (7가지)
# ─────────────────────────────────────────────
class TaskType(str, Enum):
    """
    태스크 유형을 정의한다.
    Claude Code의 TaskType에 training을 추가한 7가지 유형이다.

    각 유형은 고유한 prefix를 가지며,
    태스크 ID 생성 시 prefix + uuid[:8] 형식으로 사용된다.
    """

    LOCAL_BASH = "local_bash"  # 로컬 Bash 명령 실행
    LOCAL_AGENT = "local_agent"  # 로컬 서브 에이전트
    REMOTE_AGENT = "remote_agent"  # 원격 에이전트
    TEAMMATE = "in_process_teammate"  # 프로세스 내 팀원
    WORKFLOW = "local_workflow"  # 로컬 워크플로우
    MONITOR = "monitor"  # 모니터링 태스크
    TRAINING = "training"  # 학습 태스크 (Nexus 전용)

    @property
    def prefix(self) -> str:
        """
        태스크 ID에 사용할 접두어를 반환한다.
        왜 prefix를 쓰는가: ID만 보고도 태스크 유형을 즉시 알 수 있다.
        예) "b3a7f2c1" → LOCAL_BASH, "tr9e1d4a2" → TRAINING
        """
        _prefix_map = {
            "local_bash": "b",
            "local_agent": "a",
            "remote_agent": "r",
            "in_process_teammate": "t",
            "local_workflow": "w",
            "monitor": "m",
            "training": "tr",
        }
        return _prefix_map[self.value]


# ─────────────────────────────────────────────
# TaskStatus — 태스크 상태 (5가지)
# ─────────────────────────────────────────────
class TaskStatus(str, Enum):
    """
    태스크의 현재 상태를 나타낸다.
    상태 전환: PENDING → RUNNING → COMPLETED / FAILED / KILLED
    """

    PENDING = "pending"  # 생성됨, 아직 실행 전
    RUNNING = "running"  # 실행 중
    COMPLETED = "completed"  # 정상 완료
    FAILED = "failed"  # 에러로 실패
    KILLED = "killed"  # 강제 종료됨


# ─────────────────────────────────────────────
# TaskState — 태스크 상태 모델
# ─────────────────────────────────────────────
class TaskState(BaseModel):
    """
    하나의 태스크에 대한 전체 상태 정보.
    Pydantic BaseModel로 정의하여 직렬화/검증이 자동으로 된다.

    왜 BaseModel인가: frozen dataclass 대신 BaseModel을 쓰는 이유는
    run() 메서드에서 status, progress, error_message 등을
    동적으로 업데이트해야 하기 때문이다.
    """

    id: str  # 고유 ID (prefix + uuid[:8])
    type: TaskType  # 태스크 유형
    status: TaskStatus = TaskStatus.PENDING  # 현재 상태
    description: str  # 태스크 설명
    start_time: datetime = Field(default_factory=datetime.utcnow)  # 생성 시각
    end_time: datetime | None = None  # 종료 시각 (완료/실패/강제종료 시 설정)
    progress: float = 0.0  # 진행률 (0.0 ~ 1.0)
    error_message: str | None = None  # 실패 시 에러 메시지
    result: str | None = None  # 최종 결과 요약 (1000자 제한)


# ─────────────────────────────────────────────
# TaskManager — 태스크 라이프사이클 관리
# ─────────────────────────────────────────────
class TaskManager:
    """
    비동기 태스크의 라이프사이클을 관리한다.

    사용 흐름:
      1. create() — 태스크 생성, ID 반환
      2. run() 또는 run_background() — 코루틴 실행
      3. update_progress() — 진행률 업데이트 (선택)
      4. kill() — 강제 종료 (필요 시)
      5. cleanup_old() — 오래된 완료 태스크 정리 (주기적)

    왜 asyncio.Task를 래핑하는가:
      - 상태 추적 (PENDING/RUNNING/COMPLETED/FAILED/KILLED)
      - 결과/에러 저장
      - 완료 콜백 지원
      - 태스크 목록 조회/관리
    """

    def __init__(self) -> None:
        # 모든 태스크 상태 저장소 — ID로 조회
        self.tasks: dict[str, TaskState] = {}
        # 현재 실행 중인 asyncio.Task — kill() 시 cancel() 호출용
        self._running: dict[str, asyncio.Task[None]] = {}
        # 태스크 완료 시 호출할 콜백 — on_complete()로 등록
        self._callbacks: dict[str, list[Callable[..., Any]]] = {}

    def create(self, task_type: str | TaskType, description: str) -> str:
        """
        새 태스크를 생성하고 고유 ID를 반환한다.

        Args:
            task_type: 태스크 유형 (문자열 또는 TaskType enum)
            description: 태스크 설명

        Returns:
            생성된 태스크 ID (예: "b3a7f2c1", "tr9e1d4a2")
        """
        # 문자열이면 TaskType으로 변환, 실패 시 WORKFLOW로 폴백
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                task_type = TaskType.WORKFLOW

        # ID 생성: prefix + uuid 앞 8자
        tid = f"{task_type.prefix}{uuid.uuid4().hex[:8]}"

        self.tasks[tid] = TaskState(
            id=tid,
            type=task_type,
            description=description,
        )
        logger.info("태스크 생성: %s - %s", tid, description)
        return tid

    async def run(self, tid: str, coro: Awaitable[Any]) -> None:
        """
        코루틴을 태스크로 실행한다.

        실행 흐름:
          1. 상태를 RUNNING으로 전환
          2. 코루틴을 asyncio.Task로 래핑하여 실행
          3. 정상 완료 → COMPLETED + result 저장
          4. CancelledError → KILLED
          5. 기타 예외 → FAILED + error_message 저장
          6. 완료 콜백 통지

        Args:
            tid: 태스크 ID (create()에서 반환된 값)
            coro: 실행할 코루틴
        """
        if tid not in self.tasks:
            raise ValueError(f"태스크 {tid}를 찾을 수 없습니다")

        self.tasks[tid].status = TaskStatus.RUNNING

        async def _wrapper() -> None:
            """코루틴을 래핑하여 상태 전환과 에러 처리를 수행한다."""
            try:
                result = await coro
                self.tasks[tid].status = TaskStatus.COMPLETED
                # 결과가 있으면 최대 1000자까지 저장
                self.tasks[tid].result = str(result)[:1000] if result else None
                self.tasks[tid].progress = 1.0
            except asyncio.CancelledError:
                # kill()에 의한 취소
                self.tasks[tid].status = TaskStatus.KILLED
            except Exception as e:
                # 예기치 않은 에러
                self.tasks[tid].status = TaskStatus.FAILED
                self.tasks[tid].error_message = str(e)
                logger.error("태스크 %s 실패: %s", tid, e)
            finally:
                # 종료 시각 기록 + 실행 목록에서 제거
                self.tasks[tid].end_time = datetime.utcnow()
                self._running.pop(tid, None)
                # 등록된 콜백 통지
                await self._notify(tid)

        self._running[tid] = asyncio.create_task(_wrapper())

    def run_background(self, tid: str, coro: Awaitable[Any]) -> None:
        """
        fire-and-forget 방식으로 태스크를 배경 실행한다.
        호출 즉시 반환되며, 태스크는 비동기로 실행된다.

        왜 별도 메서드인가:
          run()은 await 가능한 코루틴을 생성하지만,
          run_background()는 호출 즉시 반환되어
          GPU 사전연결, 세션 메모리 초기화 등에 적합하다.

        Args:
            tid: 태스크 ID
            coro: 실행할 코루틴
        """
        asyncio.create_task(self.run(tid, coro))

    async def kill(self, tid: str) -> None:
        """
        실행 중인 태스크를 강제 종료한다.

        asyncio.Task.cancel()을 호출하여 CancelledError를 발생시키고,
        _wrapper()의 except 절에서 KILLED 상태로 전환된다.

        Args:
            tid: 종료할 태스크 ID
        """
        if tid in self._running:
            # 실행 중인 asyncio.Task를 취소
            self._running[tid].cancel()
            try:
                await self._running[tid]
            except asyncio.CancelledError:
                pass
            self.tasks[tid].status = TaskStatus.KILLED
            self.tasks[tid].end_time = datetime.utcnow()
            logger.info("태스크 %s 강제 종료", tid)
        elif tid in self.tasks:
            # 실행 전(PENDING) 태스크를 종료
            self.tasks[tid].status = TaskStatus.KILLED
            self.tasks[tid].end_time = datetime.utcnow()

    def update_progress(self, tid: str, progress: float) -> None:
        """
        태스크 진행률을 업데이트한다.
        0.0~1.0 범위로 클램프한다.

        Args:
            tid: 태스크 ID
            progress: 진행률 (0.0 ~ 1.0)
        """
        if tid in self.tasks:
            self.tasks[tid].progress = min(max(progress, 0.0), 1.0)

    def get_active(self) -> list[TaskState]:
        """PENDING 또는 RUNNING 상태인 태스크만 반환한다."""
        return [
            t
            for t in self.tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        ]

    def get_all(self) -> list[TaskState]:
        """모든 태스크를 반환한다."""
        return list(self.tasks.values())

    def on_complete(self, tid: str, callback: Callable[..., Any]) -> None:
        """
        태스크 완료 시 호출할 콜백을 등록한다.
        콜백은 동기 또는 비동기 함수 모두 가능하다.

        Args:
            tid: 태스크 ID
            callback: 완료 시 호출할 함수 (인자: TaskState)
        """
        self._callbacks.setdefault(tid, []).append(callback)

    async def _notify(self, tid: str) -> None:
        """
        등록된 콜백을 호출한다.
        콜백 실행 중 에러가 발생해도 다른 콜백에 영향을 주지 않는다.
        """
        for cb in self._callbacks.pop(tid, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(self.tasks[tid])
                else:
                    cb(self.tasks[tid])
            except Exception as e:
                logger.warning("태스크 콜백 에러: %s", e)

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """
        오래된 완료 태스크를 정리한다.

        COMPLETED, FAILED, KILLED 상태이고
        end_time으로부터 max_age_hours 이상 경과한 태스크를 삭제한다.

        Args:
            max_age_hours: 보존 기간 (시간 단위, 기본: 24)

        Returns:
            삭제된 태스크 수
        """
        cutoff = datetime.utcnow()
        to_remove: list[str] = []

        for tid, task in self.tasks.items():
            # 완료/실패/강제종료된 태스크만 대상
            if task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.KILLED,
            ):
                if (
                    task.end_time
                    and (cutoff - task.end_time).total_seconds()
                    > max_age_hours * 3600
                ):
                    to_remove.append(tid)

        for tid in to_remove:
            del self.tasks[tid]

        if to_remove:
            logger.info("오래된 태스크 %d개 정리 완료", len(to_remove))

        return len(to_remove)
