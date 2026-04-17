"""
GlobalState 싱글톤 — 프로세스 전체에서 하나만 존재하는 전역 상태.

Claude Code의 bootstrap/state.ts를 Python으로 재구현한다.
이 모듈은 core/ 내부의 다른 모듈을 import하지 않는다 (DAG leaf 격리).
이렇게 하면 순환 의존이 원천 차단된다.

주요 설계 원칙:
  1. 이 모듈은 bootstrap 외부(core/orchestrator, core/tools 등)를 import하지 않는다
  2. dataclass로 ~30개 필드를 타입 안전하게 정의한다
  3. threading.Lock으로 토큰 카운터 등 동시성 안전을 보장한다
  4. 모듈 레벨 싱글톤 (_STATE)으로 전역 접근을 제공한다
"""

from __future__ import annotations

import os
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────
# 권한 모드 열거형 (Permission Layer와 독립적인 bootstrap 수준 정의)
# ─────────────────────────────────────────────
class PermissionModeValue(str, Enum):
    """
    권한 모드 값.
    core/permission/에서 정의하는 PermissionMode와 1:1 대응하지만,
    순환 import를 피하기 위해 여기서 별도 정의한다.
    """

    DEFAULT = "default"
    AUTO = "auto"
    PLAN = "plan"
    TRUST = "trust"
    BYPASS = "bypass"
    HEADLESS = "headless"
    DENY_ALL = "deny_all"


# ─────────────────────────────────────────────
# GlobalState 데이터클래스
# ─────────────────────────────────────────────
@dataclass
class GlobalState:
    """
    프로세스 전역 싱글톤 상태.
    Claude Code의 bootstrap/state.ts ~100개 필드 State에 대응한다.

    카테고리:
    - 작업 디렉토리 (cwd, project_root)
    - 세션 (session_id, 시작 시간)
    - 설정 (config — NexusConfig 객체)
    - 모델 상태 (active_model)
    - 사용량 추적 (토큰, API 호출, 도구 호출)
    - 세션 플래그 (interactive, headless, permission_mode)
    - 컨텍스트 압축 (compact_boundary)
    - 플랫폼 정보
    """

    # === 작업 디렉토리 ===
    cwd: str = ""
    original_cwd: str = ""
    project_root: str = ""

    # === 세션 ===
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_session_id: str | None = None
    session_start_time: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    # === 설정 (Phase 1에서 로드) ===
    # Any 타입인 이유: NexusConfig를 import하면 순환 의존이 생길 수 있다
    config: Any = None

    # === 모델 상태 ===
    active_model: str = "primary"  # "primary" 또는 "auxiliary"
    model_override: str | None = None
    initial_model: str = "primary"

    # === 사용량 추적 (세션 누적) ===
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_api_calls: int = 0
    total_api_errors: int = 0
    total_duration_seconds: float = 0.0
    total_tool_calls: int = 0
    total_turns: int = 0

    # === 턴별 카운터 (매 턴 시작 시 리셋) ===
    current_turn_input_tokens: int = 0
    current_turn_output_tokens: int = 0
    current_turn_tool_calls: int = 0
    current_turn_start_time: float = 0.0

    # === 세션 플래그 ===
    is_interactive: bool = True
    is_headless: bool = False
    is_bare: bool = False
    permission_mode: PermissionModeValue = PermissionModeValue.DEFAULT

    # === 컨텍스트 압축 상태 ===
    # compact_boundary 이후의 메시지만 API로 전송한다
    compact_boundary: int = 0
    last_compact_turn: int = 0
    auto_compact_count: int = 0

    # === 플랫폼 정보 ===
    platform: dict = field(default_factory=dict)

    # === v7.0: 하드웨어 적응 ===
    hardware_tier: str = "small"             # "small" | "medium" | "large"
    scout_enabled: bool = False              # Scout(CPU 4B) 활성 여부
    orchestration_mode: str = "multi_model"  # "multi_model" | "single_model"

    # === 캐시 래치 (한번 설정되면 세션 내에서 변경되지 않는 값) ===
    fast_mode_latched: bool = False
    thinking_enabled_latched: bool | None = None

    # === 내부 Lock (repr에서 제외) ===
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def increment_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """스레드 안전하게 토큰 카운터를 증가시킨다."""
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.current_turn_input_tokens += input_tokens
            self.current_turn_output_tokens += output_tokens
            self.total_api_calls += 1

    def increment_tool_calls(self, count: int = 1) -> None:
        """도구 호출 카운터를 증가시킨다."""
        with self._lock:
            self.total_tool_calls += count
            self.current_turn_tool_calls += count

    def start_new_turn(self) -> None:
        """새 턴을 시작한다. 턴별 카운터를 리셋하고 총 턴 수를 증가시킨다."""
        with self._lock:
            self.current_turn_input_tokens = 0
            self.current_turn_output_tokens = 0
            self.current_turn_tool_calls = 0
            self.current_turn_start_time = time.monotonic()
            self.total_turns += 1

    def get_session_summary(self) -> dict:
        """세션 요약 정보를 반환한다. 로깅이나 메트릭스 보고에 사용한다."""
        return {
            "session_id": self.session_id,
            "turns": self.total_turns,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_api_calls": self.total_api_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "active_model": self.active_model,
            "permission_mode": self.permission_mode.value,
        }


# ─────────────────────────────────────────────
# 모듈 레벨 싱글톤 + 스레드 안전 접근자
# Claude Code: const STATE: State = getInitialState()
# ─────────────────────────────────────────────
_STATE: GlobalState | None = None
_STATE_LOCK = threading.Lock()


def get_initial_state(cwd: str | None = None) -> GlobalState:
    """
    전역 상태를 초기화한다.
    프로세스 생명주기에서 한 번만 호출해야 한다.
    두 번째 호출 시에는 기존 상태를 반환한다 (싱글톤 보장).

    왜 싱글톤인가: CLI, 웹 서버, SDK 등 어떤 진입점이든
    동일한 상태 객체를 공유해야 토큰 추적, 세션 관리가 일관된다.
    """
    global _STATE
    with _STATE_LOCK:
        if _STATE is not None:
            return _STATE

        # cwd를 절대 경로로 해석하고 Unicode NFC 정규화한다
        # Claude Code: realpathSync(cwd()).normalize('NFC')
        resolved_cwd = os.path.realpath(cwd or os.getcwd())
        resolved_cwd = unicodedata.normalize("NFC", resolved_cwd)

        _STATE = GlobalState(
            cwd=resolved_cwd,
            original_cwd=resolved_cwd,
        )
        return _STATE


def get_state() -> GlobalState:
    """현재 전역 상태를 반환한다. 초기화 전에 호출하면 RuntimeError를 발생시킨다."""
    if _STATE is None:
        raise RuntimeError(
            "GlobalState가 초기화되지 않았습니다. get_initial_state()를 먼저 호출하세요."
        )
    return _STATE


def reset_state_for_testing() -> None:
    """테스트용 상태 리셋. 프로덕션에서는 호출하지 않는다."""
    global _STATE
    with _STATE_LOCK:
        _STATE = None
