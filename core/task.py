"""
TaskManager — 비동기(백그라운드) 태스크 라이프사이클 관리 모듈.

이 파일은 Nexus에서 "오래 걸리거나 뒤에서 돌아가는 일"을 하나의 태스크로
추적하기 위한 중앙 관리자다. Claude Code의 TaskManager를 Python asyncio로
재구현한 것으로, 로컬/원격 에이전트 실행, Bash 명령, 모델 학습(training),
백그라운드 모니터링 같은 작업을 "생성 → 실행 → 완료/실패/강제종료" 흐름으로
일관되게 다룬다.

왜 필요한가:
  asyncio.create_task()만 쓰면 태스크가 지금 어떤 상태인지, 결과가 무엇인지,
  누가 취소했는지 추적할 수 없다. TaskManager는 각 태스크에 고유 ID와
  상태(TaskState)를 붙여, 목록 조회·진행률 표시·강제 종료·완료 알림을
  가능하게 한다. CLI/웹 UI가 "지금 돌고 있는 작업" 목록을 보여줄 수 있는 것도
  이 관리자 덕분이다.

이 파일에 정의된 주요 구성요소:
  - TaskType   : 태스크 유형(7가지). ID 접두어(prefix)를 제공한다.
  - TaskStatus : 태스크 상태(5가지). 상태 전환의 어휘를 고정한다.
  - TaskState  : 태스크 하나의 전체 상태를 담는 Pydantic 모델.
  - TaskManager: 위 요소를 묶어 라이프사이클 전체를 관리하는 핵심 클래스.

TaskManager의 핵심 역할:
  1. 태스크 생성 및 ID 부여 (타입별 prefix + uuid[:8])
  2. 코루틴을 asyncio.Task로 실행 (상태 자동 전환)
  3. fire-and-forget 배경 실행 (호출 즉시 반환)
  4. 강제 종료 (asyncio.Task.cancel())
  5. 진행률 추적 (0.0~1.0)
  6. 완료 콜백 통지 (동기/비동기 콜백 모두 지원)
  7. 오래된 완료 태스크 정리 (메모리 누적 방지)

의존성 방향(중요):
  core/task.py는 독립 모듈이다 — core/ 내 다른 모듈을 import하지 않는다.
  반대로 task_tools.py, bootstrap.py 등 상위 모듈이 이 파일을 import한다.
  이렇게 해두면 순환 import가 생기지 않고, 어디서든 안전하게 재사용된다.

작성자: 이현수 / 작성일: 2026-07-05
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
    태스크 유형을 정의하는 열거형(Enum)이다.
    Claude Code의 TaskType에 training(학습)을 더한 7가지 유형을 가진다.

    str을 함께 상속하므로 각 멤버는 문자열처럼도 쓸 수 있다
    (예: JSON 직렬화, 비교, dict 키 등에서 값이 그대로 문자열로 나온다).

    각 유형은 고유한 prefix(접두어)를 가지며,
    태스크 ID 생성 시 prefix + uuid[:8] 형식으로 조합된다.
    """

    LOCAL_BASH = "local_bash"  # 로컬에서 Bash 명령을 실행하는 태스크
    LOCAL_AGENT = "local_agent"  # 같은 머신에서 도는 서브 에이전트
    REMOTE_AGENT = "remote_agent"  # 원격 머신에서 도는 에이전트
    TEAMMATE = "in_process_teammate"  # 같은 프로세스 안에서 협업하는 팀원 태스크
    WORKFLOW = "local_workflow"  # 여러 단계를 묶은 로컬 워크플로우
    MONITOR = "monitor"  # 상태를 주기적으로 감시하는 모니터링 태스크
    TRAINING = "training"  # 모델 학습 태스크 (Nexus 전용으로 추가된 유형)

    @property
    def prefix(self) -> str:
        """
        태스크 ID 앞에 붙일 접두어(prefix) 문자열을 반환한다.

        왜 prefix를 쓰는가:
          태스크 ID만 눈으로 봐도 어떤 유형인지 바로 알 수 있게 하려는 것이다.
          로그나 UI에서 ID 첫 글자만 보고 유형을 구분할 수 있어 디버깅이 쉽다.
          예) "b3a7f2c1" → LOCAL_BASH, "tr9e1d4a2" → TRAINING

        구현 메모:
          아래 _prefix_map은 이 enum의 '값(문자열)'을 키로 접두어를 찾는다.
          self.value가 곧 "local_bash" 같은 문자열이므로 이를 조회에 쓴다.
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
    태스크의 현재 상태를 나타내는 열거형(Enum)이다.

    상태는 다음 흐름으로만 전환된다:
      PENDING → RUNNING → (COMPLETED | FAILED | KILLED)
    즉 처음엔 대기(PENDING), 실행되면 RUNNING, 끝나면 셋 중 하나로 확정된다.
    COMPLETED/FAILED/KILLED는 종료 상태라서 그 뒤로는 바뀌지 않는다.
    """

    PENDING = "pending"  # 생성만 된 상태 — 아직 실행 시작 전
    RUNNING = "running"  # 코루틴이 현재 실행 중
    COMPLETED = "completed"  # 예외 없이 정상적으로 끝남
    FAILED = "failed"  # 실행 도중 예외가 발생해 실패함
    KILLED = "killed"  # kill()로 강제 취소되어 종료됨


# ─────────────────────────────────────────────
# TaskState — 태스크 상태 모델
# ─────────────────────────────────────────────
class TaskState(BaseModel):
    """
    태스크 하나에 대한 전체 상태 정보를 담는 Pydantic v2 모델이다.

    이 객체 하나가 "그 태스크의 현재 스냅샷" 역할을 한다. TaskManager는
    태스크 ID를 키로 이 객체들을 dict에 저장해두고, 실행 진행에 따라
    필드 값을 갱신한다. Pydantic BaseModel이라서 타입 검증과 JSON 직렬화가
    자동으로 되어 UI/로그로 내보내기 편하다.

    왜 frozen dataclass가 아니라 BaseModel(가변)인가:
      run() 실행 중에 status, progress, error_message, result 등을
      "제자리에서" 계속 바꿔야 하기 때문이다. 불변 객체였다면 매번 새
      객체를 만들어 dict를 교체해야 해 번거롭다. 그래서 가변 모델을 택했다.
    """

    id: str  # 태스크 고유 ID (형식: prefix + uuid[:8], 예: "b3a7f2c1")
    type: TaskType  # 태스크 유형 (TaskType enum 중 하나)
    status: TaskStatus = TaskStatus.PENDING  # 현재 상태 (기본값: 대기 중)
    description: str  # 사람이 읽을 태스크 설명 (로그·UI 표시용)
    start_time: datetime = Field(default_factory=datetime.utcnow)  # 생성 시각(UTC)
    end_time: datetime | None = None  # 종료 시각 — 완료/실패/강제종료 때만 채워짐
    progress: float = 0.0  # 진행률 0.0~1.0 (update_progress로 갱신)
    error_message: str | None = None  # 실패(FAILED) 시 남기는 에러 문자열
    result: str | None = None  # 정상 완료 시 결과 요약 (최대 1000자로 잘라 저장)


# ─────────────────────────────────────────────
# TaskManager — 태스크 라이프사이클 관리
# ─────────────────────────────────────────────
class TaskManager:
    """
    비동기 태스크의 라이프사이클 전체를 관리하는 핵심 클래스다.

    이 매니저 인스턴스 하나가 여러 태스크를 동시에 추적한다. 각 태스크는
    ID로 식별되며, 상태 스냅샷(TaskState)과 실제로 도는 asyncio.Task가
    각각 dict에 저장된다.

    전형적인 사용 흐름:
      1. create()              — 태스크를 등록하고 고유 ID를 받는다.
      2. run() / run_background() — 코루틴을 그 ID에 연결해 실행한다.
      3. update_progress()     — (선택) 진행률을 0~1로 갱신한다.
      4. kill()                — (필요 시) 실행 중 태스크를 강제 종료한다.
      5. cleanup_old()         — (주기적) 오래된 종료 태스크를 메모리에서 지운다.

    왜 asyncio.Task를 그냥 쓰지 않고 이렇게 래핑하는가:
      - 상태 추적 (PENDING/RUNNING/COMPLETED/FAILED/KILLED)을 붙이기 위해
      - 결과/에러 문자열을 저장해 나중에 조회하기 위해
      - 완료 시점에 콜백을 호출(알림)하기 위해
      - 실행 중/전체 태스크 목록을 조회·관리하기 위해
    """

    def __init__(self) -> None:
        # 모든 태스크의 상태 스냅샷 저장소. 키=태스크 ID, 값=TaskState.
        # 종료된 태스크도 cleanup_old()로 지우기 전까지는 여기에 남아 조회된다.
        self.tasks: dict[str, TaskState] = {}
        # 현재 "실제로 돌고 있는" asyncio.Task 저장소. 키=태스크 ID.
        # kill() 때 여기서 Task를 찾아 cancel()을 호출한다. 끝나면 제거된다.
        self._running: dict[str, asyncio.Task[None]] = {}
        # 태스크 완료 시 호출할 콜백 목록. on_complete()로 등록하고,
        # _notify()에서 꺼내 실행한다. 한 태스크에 여러 콜백을 걸 수 있다.
        self._callbacks: dict[str, list[Callable[..., Any]]] = {}

    def create(self, task_type: str | TaskType, description: str) -> str:
        """
        새 태스크를 등록하고 고유 ID를 만들어 반환한다.

        이 단계에서는 아직 코루틴을 실행하지 않는다. 상태 스냅샷(TaskState)만
        PENDING(대기) 상태로 self.tasks에 넣어둔다. 실제 실행은 이후
        run()/run_background()에 이 ID를 넘겨야 시작된다.

        Args:
            task_type: 태스크 유형. 문자열("local_bash" 등) 또는 TaskType enum.
            description: 사람이 읽을 태스크 설명 (로그·UI에 표시된다).

        Returns:
            생성된 태스크 ID (예: "b3a7f2c1", "tr9e1d4a2").
        """
        # 인자가 문자열이면 TaskType enum으로 변환한다.
        # 알 수 없는 값이면 예외 대신 WORKFLOW로 안전하게 폴백한다
        # (호출부에서 오타/미지원 값이 와도 태스크 생성 자체는 실패하지 않게).
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                task_type = TaskType.WORKFLOW

        # ID 조합: 유형 접두어 + uuid4 16진수 문자열의 앞 8자.
        # uuid로 충돌 가능성을 낮추고, 접두어로 유형을 식별 가능하게 한다.
        tid = f"{task_type.prefix}{uuid.uuid4().hex[:8]}"

        # 대기(PENDING) 상태의 스냅샷을 저장소에 등록한다.
        self.tasks[tid] = TaskState(
            id=tid,
            type=task_type,
            description=description,
        )
        logger.info("태스크 생성: %s - %s", tid, description)
        return tid

    async def run(self, tid: str, coro: Awaitable[Any]) -> None:
        """
        create()로 만들어 둔 태스크에 코루틴을 연결해 실제로 실행한다.

        주의: 이 메서드 자체는 코루틴을 끝까지 기다리지 않는다.
        내부에서 asyncio.create_task()로 백그라운드 실행을 '예약'하고
        곧바로 반환한다. 상태 전환과 결과 저장은 아래 _wrapper()가
        태스크가 끝나는 시점에 알아서 처리한다.

        실행 흐름(_wrapper 기준):
          1. 상태를 RUNNING으로 전환한다.
          2. 넘겨받은 coro를 await 하며 실행한다.
          3. 정상 완료 → COMPLETED + result 저장 + progress 1.0.
          4. CancelledError(=kill 호출) → KILLED.
          5. 그 외 예외 → FAILED + error_message 저장 + 에러 로그.
          6. 성공/실패와 무관하게 종료 시각 기록 + 실행목록 정리 + 콜백 통지.

        Args:
            tid: 태스크 ID (create()에서 반환된 값이어야 한다).
            coro: 실행할 코루틴(awaitable).

        Raises:
            ValueError: 등록되지 않은(존재하지 않는) tid가 들어온 경우.
        """
        # 존재하지 않는 ID면 조용히 넘어가지 않고 명시적으로 실패시킨다
        # (호출부의 버그를 빨리 드러내기 위한 fail-fast).
        if tid not in self.tasks:
            raise ValueError(f"태스크 {tid}를 찾을 수 없습니다")

        # 실행 시작을 알리기 위해 먼저 상태를 RUNNING으로 바꾼다.
        self.tasks[tid].status = TaskStatus.RUNNING

        async def _wrapper() -> None:
            """
            실제 코루틴을 감싸 상태 전환·결과 저장·에러 처리를 담당하는 내부 함수.

            try/except/finally로 "어떻게 끝나든" 상태가 반드시 종료 상태로
            확정되고 콜백이 호출되도록 보장한다. 클로저라서 바깥의 tid와 coro를
            그대로 참조한다.
            """
            try:
                # 본 작업 실행. 여기서 오래 걸리는 일이 벌어진다.
                result = await coro
                self.tasks[tid].status = TaskStatus.COMPLETED
                # 결과가 있으면 문자열로 바꿔 최대 1000자까지만 저장한다.
                # (거대한 결과가 메모리에 그대로 쌓이는 것을 막기 위한 상한.)
                self.tasks[tid].result = str(result)[:1000] if result else None
                self.tasks[tid].progress = 1.0
            except asyncio.CancelledError:
                # kill()이 cancel()을 부르면 여기로 온다 = 사용자/시스템의 취소.
                self.tasks[tid].status = TaskStatus.KILLED
            except Exception as e:
                # 예상치 못한 실행 오류. 상태를 FAILED로 두고 메시지를 남긴다.
                self.tasks[tid].status = TaskStatus.FAILED
                self.tasks[tid].error_message = str(e)
                logger.error("태스크 %s 실패: %s", tid, e)
            finally:
                # 성공/취소/실패 어느 쪽이든 공통으로 처리할 마무리 작업.
                # 1) 종료 시각을 남기고, 2) 실행 중 목록에서 빼고,
                # 3) 등록된 완료 콜백들에게 알린다.
                self.tasks[tid].end_time = datetime.utcnow()
                self._running.pop(tid, None)
                await self._notify(tid)

        # _wrapper를 백그라운드 asyncio.Task로 띄우고 핸들을 보관한다.
        # 이 핸들이 있어야 나중에 kill()에서 cancel()을 부를 수 있다.
        self._running[tid] = asyncio.create_task(_wrapper())

    def run_background(self, tid: str, coro: Awaitable[Any]) -> None:
        """
        fire-and-forget(쏘고 잊기) 방식으로 태스크를 배경 실행한다.
        호출하면 즉시 반환되고, 실제 실행은 비동기로 뒤에서 진행된다.

        run()과의 차이:
          run()은 async 메서드라 호출부가 'await self.run(...)'로 살짝
          기다렸다가(=create_task 등록까지) 넘어간다. 반면 이 메서드는
          동기 함수라서 await 없이도 부를 수 있고, run() 코루틴 자체를
          또 하나의 create_task로 감싸 완전히 비동기로 흘려보낸다.

        언제 쓰나:
          호출 결과를 그 자리에서 기다릴 필요가 없는 초기화성 작업에 적합하다.
          예: GPU 서버 사전 연결(warm-up), 세션 메모리 초기화 등.

        Args:
            tid: 태스크 ID (create()에서 받은 값).
            coro: 실행할 코루틴.
        """
        # self.run(...) 코루틴을 태스크로 감싸 즉시 스케줄링만 하고 반환한다.
        asyncio.create_task(self.run(tid, coro))

    async def kill(self, tid: str) -> None:
        """
        태스크를 강제 종료(취소)한다. 두 가지 경우를 모두 처리한다.

        (1) 이미 실행 중인 태스크:
            asyncio.Task.cancel()로 CancelledError를 발생시킨다. 그러면
            run()의 _wrapper() except 절이 상태를 KILLED로 바꾼다. 여기서는
            취소가 실제로 끝날 때까지 await로 기다린 뒤, 확실히 하기 위해
            상태/종료시각을 한 번 더 KILLED로 확정한다.

        (2) 아직 실행 전(PENDING)인 태스크:
            돌고 있는 asyncio.Task가 없으므로 cancel할 대상이 없다.
            상태만 바로 KILLED로 바꾸고 종료 시각을 남긴다.

        Args:
            tid: 종료할 태스크 ID.
        """
        if tid in self._running:
            # 실행 중인 asyncio.Task에 취소 신호를 보낸다.
            self._running[tid].cancel()
            try:
                # 취소가 실제로 마무리될 때까지 기다린다. 취소된 태스크를
                # await하면 CancelledError가 다시 올라오므로 여기서 삼킨다.
                await self._running[tid]
            except asyncio.CancelledError:
                pass
            # _wrapper의 finally보다 뒤라는 보장이 없으므로 상태를 명시 확정한다.
            self.tasks[tid].status = TaskStatus.KILLED
            self.tasks[tid].end_time = datetime.utcnow()
            logger.info("태스크 %s 강제 종료", tid)
        elif tid in self.tasks:
            # 아직 실행되지 않은 대기(PENDING) 태스크는 상태만 바꿔 종료 처리한다.
            self.tasks[tid].status = TaskStatus.KILLED
            self.tasks[tid].end_time = datetime.utcnow()

    def update_progress(self, tid: str, progress: float) -> None:
        """
        태스크의 진행률을 갱신한다. 진행 중인 작업이 스스로 호출해
        "지금 몇 %쯤 됐다"를 알리는 용도다.

        입력값은 min/max로 0.0~1.0 범위 안으로 강제 보정(클램프)한다.
        범위를 벗어난 값이 들어와도 안전하게 처리하기 위함이다.

        Args:
            tid: 태스크 ID.
            progress: 진행률 (0.0 ~ 1.0). 벗어나면 경계값으로 잘린다.
        """
        # 존재하는 태스크에만 반영한다. 없으면 조용히 무시한다.
        if tid in self.tasks:
            self.tasks[tid].progress = min(max(progress, 0.0), 1.0)

    def get_active(self) -> list[TaskState]:
        """
        아직 끝나지 않은 태스크(PENDING 또는 RUNNING)만 골라 반환한다.
        UI에서 "현재 진행 중인 작업" 목록을 보여줄 때 쓴다.
        """
        return [
            t
            for t in self.tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        ]

    def get_all(self) -> list[TaskState]:
        """저장소에 있는 모든 태스크(종료된 것 포함)를 리스트로 반환한다."""
        return list(self.tasks.values())

    def on_complete(self, tid: str, callback: Callable[..., Any]) -> None:
        """
        태스크가 끝났을 때 호출할 콜백을 등록한다.

        콜백은 동기 함수든 async 함수든 상관없다(_notify에서 알아서 구분해
        비동기면 await로 호출한다). 한 태스크에 여러 콜백을 걸 수 있으며,
        등록된 순서대로 호출된다. 콜백은 인자로 해당 TaskState를 받는다.

        Args:
            tid: 태스크 ID.
            callback: 완료 시 호출할 함수 (인자로 TaskState를 받음).
        """
        # 해당 tid의 콜백 리스트가 없으면 새로 만들고, 있으면 이어붙인다.
        self._callbacks.setdefault(tid, []).append(callback)

    async def _notify(self, tid: str) -> None:
        """
        해당 태스크에 등록된 완료 콜백들을 차례로 호출하는 내부 메서드.
        run()의 _wrapper() finally에서 태스크 종료 직후 호출된다.

        pop으로 콜백 리스트를 꺼내면서 동시에 저장소에서 제거하므로
        같은 콜백이 두 번 불리지 않는다. 콜백 하나가 예외를 던져도 잡아서
        로그만 남기고 넘어가, 나머지 콜백과 매니저 전체에 영향을 주지 않는다.
        """
        for cb in self._callbacks.pop(tid, []):
            try:
                # async 콜백이면 await로, 일반 함수면 그대로 호출한다.
                if asyncio.iscoroutinefunction(cb):
                    await cb(self.tasks[tid])
                else:
                    cb(self.tasks[tid])
            except Exception as e:
                # 콜백 오류는 치명적이지 않으므로 경고만 남기고 계속 진행한다.
                logger.warning("태스크 콜백 에러: %s", e)

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """
        오래된 '종료된' 태스크를 저장소에서 삭제해 메모리 누적을 막는다.

        self.tasks에는 끝난 태스크도 계속 남기 때문에, 오래 켜두면 dict가
        무한정 커진다. 이 메서드를 주기적으로 호출해 오래된 것들을 비운다.

        삭제 대상 조건(둘 다 만족):
          - 상태가 종료 상태(COMPLETED / FAILED / KILLED)일 것.
          - end_time으로부터 max_age_hours 시간 이상 지났을 것.
        아직 실행 중이거나 대기 중인 태스크는 절대 지우지 않는다.

        Args:
            max_age_hours: 종료 후 보존할 시간(시간 단위). 기본 24시간.

        Returns:
            실제로 삭제한 태스크 개수.
        """
        # 기준 시각(현재)을 한 번만 구해 반복 중 일관되게 비교에 쓴다.
        cutoff = datetime.utcnow()
        # 순회 도중 dict를 바로 지우면 위험하므로, 지울 ID를 먼저 모은다.
        to_remove: list[str] = []

        for tid, task in self.tasks.items():
            # 1차: 종료된(더 이상 안 바뀌는) 상태의 태스크만 후보로 본다.
            if task.status in (
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.KILLED,
            ):
                # 2차: 종료 시각이 있고, 경과 시간이 보존 기간을 넘었는지 확인.
                # max_age_hours * 3600으로 시간을 초로 바꿔 비교한다.
                if (
                    task.end_time
                    and (cutoff - task.end_time).total_seconds()
                    > max_age_hours * 3600
                ):
                    to_remove.append(tid)

        # 모아둔 ID들을 실제로 저장소에서 제거한다.
        for tid in to_remove:
            del self.tasks[tid]

        # 뭔가 지웠을 때만 로그를 남긴다(빈 정리에는 로그를 남기지 않음).
        if to_remove:
            logger.info("오래된 태스크 %d개 정리 완료", len(to_remove))

        return len(to_remove)
