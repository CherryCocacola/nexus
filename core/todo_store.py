# 세션·에이전트별 계획 체크리스트(TodoWrite) 상태를 보관하는 인메모리 저장소
"""
TodoStore — 계획 체크리스트(TodoWrite식) 상태 저장소.

[이 파일이 하는 일]
Claude Code의 TodoWrite처럼, 모델이 다단계 작업 중 스스로 "계획 체크리스트"를
만들고 갱신할 때 그 목록을 (세션, 에이전트) 단위로 보관한다. TurnStateStore
(core/orchestrator/turn_state.py)와 똑같은 "세션 키 인메모리 dict" 패턴을 따른다.

[용어 주의 — domain-model.md]
여기서 다루는 TodoItem/TodoStatus는 "계획 진행 상태"(pending/in_progress/
completed)다. core/task.py의 TaskState/TaskStatus(실행 상태 — pending/running/
completed/failed/killed)와는 이름부터 구분한다. 절대 혼용하지 않는다.
  - TodoStatus  : 계획 항목의 진행 상태 (이 파일)
  - TaskStatus  : 백그라운드 실행체의 상태 (core/task.py)

[배치 근거 — 의존성 방향 P2]
도구(core/tools)와 오케스트레이터(core/orchestrator) 양쪽이 이 저장소 타입을
import해야 한다. core/orchestrator에 두면 tools→orchestrator 역방향 import가
생겨 순환이 된다(orchestrator는 이미 tools를 import). 그래서 어느 쪽에서도
안전하게 import할 수 있는 최상위 leaf 모듈(core/todo_store.py)에 둔다.
이 모듈은 pydantic 외에 core 내부 모듈을 전혀 import하지 않는다.
(설계 문서는 core/task/todo_store.py를 제안했으나, core/task는 패키지가 아니라
단일 모듈 파일이라 core/todo_store.py로 배치한다 — 의존성 의도는 동일.)

[영속성]
1차 구현은 인메모리(프로세스 수명)다. 계획 체크리스트는 본질적으로 세션 수명
데이터라 Redis 승격은 불필요하다.
# TODO(nexus): 웹 세션 복원(재접속) 요구가 생기면 TurnStateStore와 함께
#   short_term(Redis) 직렬화를 후속 과제로 검토한다.

작성자: Nexus / 작성일: 2026-07-08
"""

from __future__ import annotations

import time
from enum import Enum

from pydantic import BaseModel

# 체크리스트가 무한정 커지는 것을 막는 상한(컨텍스트 폭주 방지). 도구의
# validate_input에서도 참조한다.
MAX_TODO_ITEMS = 50
# 항목 content 길이 상한(글자). 초과 시 도구가 에러를 반환한다.
MAX_CONTENT_LEN = 500


class TodoStatus(str, Enum):
    """체크리스트 항목의 진행 상태. TaskStatus(실행 상태)와 별개의 enum이다."""

    PENDING = "pending"  # 아직 시작 전
    IN_PROGRESS = "in_progress"  # 지금 진행 중 (원칙적으로 동시 1개)
    COMPLETED = "completed"  # 완료


class TodoItem(BaseModel):
    """
    체크리스트 항목 1개.

    frozen으로 두어, 목록을 교체할 때 항상 새 객체만 생성하도록 강제한다
    (StreamEvent/TurnState와 동일한 불변 원칙).
    """

    model_config = {"frozen": True}

    content: str  # 할 일 내용 (명령형, 예: "권한 파이프라인 테스트 실행")
    status: TodoStatus = TodoStatus.PENDING  # 항목 상태
    active_form: str = ""  # 진행 중 표시용 현재진행형 문구(예: "... 실행 중")


class TodoListState(BaseModel):
    """한 (세션, 에이전트) 쌍의 체크리스트 스냅샷. 역시 불변이다."""

    model_config = {"frozen": True}

    items: tuple[TodoItem, ...] = ()  # 항목들(순서 = 계획 순서)
    revision: int = 0  # 교체 횟수 (UI 증분 갱신·중복 프레임 무시·디버깅용)
    updated_at: float = 0.0  # 마지막 교체 시각(epoch 초)


# ─────────────────────────────────────────────
# 렌더링 헬퍼 — 도구 결과 에코 + 프롬프트 주입에 공통으로 쓴다
# ─────────────────────────────────────────────
def render_checklist(items: tuple[TodoItem, ...]) -> str:
    """
    체크리스트를 사람이(그리고 모델이) 읽기 좋은 여러 줄 문자열로 만든다.

    형식:
      [x] 완료된 항목
      [~] 진행 중 항목 ← 진행 중
      [ ] 대기 항목

    빈 목록이면 "(체크리스트가 비어 있습니다)"를 돌려준다.
    """
    if not items:
        return "(체크리스트가 비어 있습니다)"

    # 상태 → 체크박스 아이콘 매핑. 모르는 값은 방어적으로 "[ ]".
    icon = {
        TodoStatus.COMPLETED: "[x]",
        TodoStatus.IN_PROGRESS: "[~]",
        TodoStatus.PENDING: "[ ]",
    }
    lines: list[str] = []
    for item in items:
        mark = icon.get(item.status, "[ ]")
        line = f"{mark} {item.content}"
        # 진행 중 항목은 active_form(있으면)을 꼬리표로 붙여 UI/모델이 현재 단계를
        # 바로 알 수 있게 한다.
        if item.status == TodoStatus.IN_PROGRESS:
            tail = item.active_form or "진행 중"
            line += f" ← {tail}"
        lines.append(line)
    return "\n".join(lines)


class TodoStore:
    """
    체크리스트 저장소 — TurnStateStore와 동일한 '세션 키 인메모리 dict' 패턴.

    키는 (session_id, agent_id or "main")다. agent_id가 None(메인 에이전트)이면
    "main"으로 정규화한다. 서브에이전트는 자기 agent_id 키에만 쓰므로 메인 목록을
    덮어쓸 수 없다(격리).
    """

    def __init__(self) -> None:
        # 핵심 자료구조: (session_id, agent_key) → 최신 TodoListState.
        self._lists: dict[tuple[str, str], TodoListState] = {}

    @staticmethod
    def _key(session_id: str, agent_id: str | None) -> tuple[str, str]:
        """(session_id, agent_id) → 내부 dict 키. 메인 에이전트는 'main'으로 정규화."""
        return (session_id, agent_id or "main")

    def replace(
        self, session_id: str, agent_id: str | None, items: list[TodoItem]
    ) -> TodoListState:
        """
        해당 (세션, 에이전트)의 체크리스트를 통째로 새 목록으로 교체한다(원자적).

        Claude Code TodoWrite와 동일한 전체-교체 시맨틱이다. 이전 목록의 revision을
        1 올린 새 스냅샷을 만들어 저장하고 반환한다.
        """
        key = self._key(session_id, agent_id)
        prev = self._lists.get(key)
        new_revision = (prev.revision + 1) if prev is not None else 1
        state = TodoListState(
            items=tuple(items),
            revision=new_revision,
            updated_at=time.time(),
        )
        self._lists[key] = state
        return state

    def get(self, session_id: str, agent_id: str | None) -> TodoListState:
        """
        해당 (세션, 에이전트)의 현재 체크리스트를 반환한다.

        아직 아무것도 쓰지 않았으면 빈 TodoListState(revision=0)를 돌려준다.
        """
        return self._lists.get(self._key(session_id, agent_id), TodoListState())

    def clear_session(self, session_id: str) -> None:
        """
        세션 삭제 시 호출 — 그 세션에 속한 모든 (에이전트 포함) 목록을 제거한다.

        웹의 delete_session에서 함께 호출해, 세션이 사라지면 체크리스트도 정리한다.
        """
        # 순회 중 삭제 위험을 피하려고 지울 키를 먼저 모은다.
        to_remove = [k for k in self._lists if k[0] == session_id]
        for k in to_remove:
            del self._lists[k]

    def to_context_string(self, session_id: str, agent_id: str | None = None) -> str:
        """
        시스템 프롬프트 주입용 문자열을 만든다(prompt_assembler가 호출).

        메인 에이전트(agent_id=None)의 목록을 렌더한다. 목록이 비어 있으면 빈
        문자열을 돌려줘, 호출자가 섹션 자체를 생략하게 한다. 토큰 예산 상한은
        호출자(prompt_assembler)가 적용한다 — 여기서는 원본만 렌더한다.
        """
        state = self.get(session_id, agent_id)
        if not state.items:
            return ""
        return render_checklist(state.items)


# ─────────────────────────────────────────────
# 폴백 저장소 — TodoStore가 주입되지 않은 환경(단위 테스트 등)용
# ─────────────────────────────────────────────
# task_tools._fallback_tasks 관례를 따르되, 세션·에이전트 키를 유지하는 단일
# 프로세스 전역 TodoStore를 폴백으로 쓴다. 주입이 있으면 절대 이 폴백을 쓰지 않는다.
_fallback_store = TodoStore()


def get_todo_store(context: object) -> TodoStore:
    """
    도구 실행 컨텍스트에서 TodoStore를 꺼내오는 헬퍼(task_tools._get_task_manager 패턴).

    context.options["todo_store"]에 TodoStore가 주입돼 있으면 그것을, 없거나 타입이
    맞지 않으면 모듈 전역 폴백 저장소를 반환한다. 폴백도 세션·에이전트 키를 유지하므로
    격리 동작은 동일하다.
    """
    options = getattr(context, "options", None)
    if isinstance(options, dict):
        store = options.get("todo_store")
        if isinstance(store, TodoStore):
            return store
    return _fallback_store
