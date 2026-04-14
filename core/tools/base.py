"""
BaseTool ABC — 모든 도구의 기본 인터페이스.

Claude Code의 Tool<Input, Output, Progress> 인터페이스를 Python ABC로 재구현한다.
~40개 멤버를 7개 카테고리로 정리:
  1. Identity (name, description, aliases)
  2. Schema (input_schema)
  3. Behavior Flags (fail-closed 기본값)
  4. Limits (max_result_size, timeout)
  5. Lifecycle (validate_input, check_permissions, call, map_result)
  6. Observable (backfill_observable_input)
  7. UI Hints (get_progress_label 등)

핵심 설계 원칙 — fail-closed 기본값:
  - is_read_only = False (쓰기 도구로 가정)
  - is_concurrency_safe = False (순차 실행)
  - is_destructive = False
  - requires_confirmation = False
새 도구를 추가할 때 명시적으로 완화하지 않으면 가장 제한적으로 동작한다.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("nexus.tools")


# ─────────────────────────────────────────────
# 지원 타입
# ─────────────────────────────────────────────
class PermissionBehavior(str, Enum):
    """권한 결정 결과."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"  # 사용자에게 확인 요청 (CLI 인터랙티브)


class PermissionResult(BaseModel):
    """도구의 check_permissions() 반환값."""

    behavior: PermissionBehavior = PermissionBehavior.ALLOW
    message: str | None = None
    details: dict[str, Any] | None = None


class ToolResult(BaseModel):
    """
    도구 실행 결과.
    Claude Code의 ToolResultBlockParam에 대응한다.
    data가 직렬화되어 tool_result 메시지의 content가 된다.
    """

    data: Any
    is_error: bool = False
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(cls, data: Any, **metadata) -> ToolResult:
        """성공 결과를 생성한다."""
        return cls(data=data, metadata=metadata)

    @classmethod
    def error(cls, message: str, **metadata) -> ToolResult:
        """에러 결과를 생성한다."""
        return cls(data=message, is_error=True, error_message=message, metadata=metadata)


class ToolUseContext(BaseModel):
    """
    도구 실행 컨텍스트.
    도구가 실행될 때 필요한 환경 정보를 전달한다.
    """

    model_config = {"arbitrary_types_allowed": True}

    cwd: str  # 현재 작업 디렉토리
    session_id: str = ""
    agent_id: str | None = None  # 서브 에이전트 구별용
    tool_use_id: str = ""  # 이 도구 호출의 고유 ID
    read_file_timestamps: dict[str, float] = Field(default_factory=dict)
    abort_signal: Any = None  # asyncio.Event (취소 시그널)
    permission_mode: str = "default"
    parent_tool_use_id: str | None = None  # 부모 도구 (중첩 실행)
    options: dict[str, Any] = Field(default_factory=dict)


class ToolProgressEvent(BaseModel):
    """도구 실행 진행 이벤트 (장시간 실행 도구용)."""

    tool_use_id: str
    progress: float  # 0.0 ~ 1.0
    message: str = ""


# ─────────────────────────────────────────────
# BaseTool ABC
# ─────────────────────────────────────────────
class BaseTool(ABC):
    """
    모든 도구의 기본 클래스.
    이 클래스를 상속하고 name, description, input_schema, check_permissions, call을 구현한다.
    """

    # ═══ 1. Identity ═══

    @property
    @abstractmethod
    def name(self) -> str:
        """도구 고유 이름. 모델이 tool_use에서 참조하는 이름이다."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """모델에 제공되는 도구 설명."""
        ...

    @property
    def aliases(self) -> list[str]:
        """대체 이름 목록. 모델이 잘못된 이름을 사용해도 매칭할 수 있다."""
        return []

    @property
    def group(self) -> str:
        """도구 그룹. UI 카테고리 표시에 사용한다."""
        return "default"

    # ═══ 2. Schema ═══

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """도구 입력 JSON Schema. vLLM/OpenAI function_calling의 parameters로 전달된다."""
        ...

    # ═══ 3. Behavior Flags (fail-closed 기본값) ═══

    @property
    def is_read_only(self) -> bool:
        """True면 시스템 상태를 변경하지 않는다. 기본: False (쓰기 도구 가정)."""
        return False

    @property
    def is_destructive(self) -> bool:
        """True면 되돌리기 어려운 변경을 수행한다. 기본: False."""
        return False

    @property
    def is_concurrency_safe(self) -> bool:
        """True면 다른 도구와 병렬 실행이 안전하다. 기본: False (순차 실행)."""
        return False

    @property
    def should_defer(self) -> bool:
        """True면 모델 스트리밍 완료 후 실행을 지연할 수 있다. 기본: False."""
        return False

    @property
    def requires_confirmation(self) -> bool:
        """True면 항상 사용자 확인이 필요하다. 기본: False."""
        return False

    @property
    def is_enabled(self) -> bool:
        """False면 현재 비활성 (도구 풀에서 제외). 기본: True."""
        return True

    @property
    def is_user_facing(self) -> bool:
        """True면 실행/결과가 사용자에게 표시된다. 기본: True."""
        return True

    @property
    def has_side_effects(self) -> bool:
        """True면 외부 시스템에 부수 효과를 일으킨다. 기본: not is_read_only."""
        return not self.is_read_only

    # ═══ 4. Limits ═══

    @property
    def max_result_size(self) -> int:
        """최대 결과 크기 (문자 수). 초과 시 디스크에 저장. 기본: 100,000."""
        return 100_000

    @property
    def timeout_seconds(self) -> float:
        """실행 타임아웃 (초). 기본: 120초."""
        return 120.0

    @property
    def max_retries(self) -> int:
        """자동 재시도 횟수. 기본: 0."""
        return 0

    # ═══ 5. Lifecycle Methods ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        도메인 검증 (JSON Schema 이후 추가 검증).
        None이면 유효, str이면 에러 메시지.
        """
        return None

    @abstractmethod
    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """도구별 권한 확인. 5계층 파이프라인의 도구 고유 검사 단계."""
        ...

    @abstractmethod
    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """도구 실행 (핵심 로직). 모든 검증/권한 확인 후 호출된다."""
        ...

    def map_result(self, result: ToolResult) -> str:
        """도구 결과를 tool_result 메시지 content로 변환한다."""
        if result.is_error:
            return f"<tool_use_error>{result.error_message or result.data}</tool_use_error>"
        return str(result.data)

    # ═══ 6. Observable ═══

    def backfill_observable_input(
        self, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Hook에 전달할 관찰 가능 입력을 생성한다. 민감 정보를 제거할 수 있다."""
        return input_data.copy()

    # ═══ 7. UI Hints ═══

    def get_user_facing_name(self) -> str:
        """사용자에게 표시되는 이름."""
        return self.name

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        """진행 상태 표시 라벨."""
        return f"Running {self.name}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        """입력의 간단 요약 (로깅/UI용)."""
        if not input_data:
            return ""
        first_value = str(next(iter(input_data.values()), ""))
        return first_value[:100]

    # ═══ Schema Export ═══

    def to_schema(self) -> dict[str, Any]:
        """모델에 전달할 도구 스키마를 생성한다."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        flags = []
        if self.is_read_only:
            flags.append("RO")
        if self.is_concurrency_safe:
            flags.append("CS")
        if self.is_destructive:
            flags.append("DESTRUCTIVE")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"<Tool: {self.name}{flag_str}>"
