"""
GlobalState 싱글톤 — 프로세스 전체에서 "단 하나만" 존재하는 전역 상태 저장소.

이 파일은 무슨 일을 하나요? (온보딩용 개요)
  - Nexus 프로세스(=CLI, 웹 서버, SDK 등 어떤 진입점이든)가 실행되는 동안
    공유해야 하는 핵심 상태를 한 곳에 모아 둡니다.
  - 예: 지금 어느 디렉토리에서 작업 중인지, 세션 ID는 무엇인지, 이번 세션에서
    토큰을 몇 개 썼는지, 현재 권한 모드가 무엇인지 등.
  - 이 값들은 여러 모듈이 동시에 읽고 쓰기 때문에, 서로 다른 사본을 들고 있으면
    통계가 어긋납니다. 그래서 "싱글톤"(전역 단일 인스턴스)으로 관리합니다.

원본 대응:
  Claude Code의 bootstrap/state.ts(약 100개 필드의 State)를 Python으로 재구현한
  것입니다. 여기서는 실제로 쓰는 ~30개 필드만 추립니다.

주요 구성 요소:
  - PermissionModeValue : 권한 모드 열거형(문자열 Enum)
  - GlobalState         : 전역 상태를 담는 dataclass 본체
  - get_initial_state() : 최초 1회 상태를 만들어 주는 초기화 함수
  - get_state()         : 이미 만들어진 상태를 꺼내오는 접근자
  - reset_state_for_testing() : 테스트에서만 쓰는 상태 초기화 함수

의존성 규칙 (매우 중요):
  - 이 모듈은 core/ 내부의 다른 모듈(orchestrator, tools, permission 등)을
    "절대 import 하지 않습니다". 의존성 그래프에서 잎(leaf) 노드로 고립시켜
    순환 import를 원천 차단하기 위함입니다.
  - 그래서 설정 객체(NexusConfig)조차 타입을 Any로 두어 import를 피합니다.

설계 원칙 요약:
  1. bootstrap 바깥 모듈을 import 하지 않는다 (DAG leaf 격리)
  2. dataclass로 약 30개 필드를 타입 안전하게 정의한다
  3. threading.Lock으로 토큰 카운터 등 동시성 안전을 보장한다
  4. 모듈 레벨 싱글톤(_STATE)으로 전역 접근을 제공한다

작성자: 이현수 / 작성일: 2026-07-05
"""

# ── 표준 라이브러리만 사용 ──
# 순환 의존을 피하려고 프로젝트 내부 모듈은 하나도 import 하지 않는다.
from __future__ import annotations

import os  # cwd 절대 경로 해석(os.path.realpath, os.getcwd)에 사용
import threading  # 동시성 보호용 Lock 제공
import time  # 턴 시작 시각 측정(time.monotonic)에 사용
import unicodedata  # 경로 문자열의 유니코드 NFC 정규화에 사용
import uuid  # 세션 ID(고유값) 생성에 사용
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any  # NexusConfig import를 피하려고 config 타입을 Any로 둔다


# ─────────────────────────────────────────────
# 권한 모드 열거형 (Permission Layer와 독립적인 bootstrap 수준 정의)
# ─────────────────────────────────────────────
class PermissionModeValue(str, Enum):
    """
    권한 모드를 나타내는 문자열 Enum.

    왜 여기서 다시 정의하나요?
      실제 권한 로직은 core/permission/의 PermissionMode에 있습니다. 값은 1:1로
      대응하지만, 이 state 모듈이 core/permission을 import 하면 순환 의존이
      생깁니다. 그래서 "문자열 값"만 담은 가벼운 사본을 여기에 둡니다.

    str을 함께 상속하는 이유:
      멤버가 곧 문자열이라 로깅/직렬화(JSON) 시 그대로 "default" 같은 값으로
      다뤄져 편리합니다.

    각 모드 의미(요약):
      - DEFAULT      : 기본. 위험한 작업은 사용자에게 확인을 요청
      - ACCEPT_EDITS : 파일 수정(Write/Edit 등)은 자동 승인, Bash 등은 여전히 확인
      - AUTO         : 자동 진행 위주
      - PLAN         : 계획만 세우고 실제 쓰기 작업은 막음
      - TRUST        : 신뢰 모드(확인 완화)
      - BYPASS       : 권한 확인 우회
      - HEADLESS     : 사용자 상호작용이 없는 무인 실행 환경
      - DENY_ALL     : 전부 거부(가장 제한적)
    """

    DEFAULT = "default"
    ACCEPT_EDITS = "accept_edits"
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
    프로세스 전역 싱글톤 상태를 담는 dataclass.

    역할:
      한 번의 Nexus 실행(세션) 동안 여러 모듈이 공유해야 하는 값들을 한 객체에
      모아 둡니다. 이 객체는 get_initial_state()로 최초 1회 생성되고, 이후에는
      get_state()로 어디서든 동일한 인스턴스를 꺼내 씁니다.

    Claude Code 대응:
      bootstrap/state.ts의 약 100개 필드짜리 State에 대응합니다.

    필드 카테고리(아래 코드에 === 구분선으로 그룹핑되어 있음):
      - 작업 디렉토리 : cwd, original_cwd, project_root
      - 세션         : session_id, parent_session_id, 시작 시각
      - 설정         : config (NexusConfig 객체를 Any로 보관)
      - 모델 상태    : active_model, model_override 등
      - 사용량 추적  : 토큰/ API 호출/ 도구 호출 누적 카운터
      - 턴별 카운터  : 매 턴 시작 시 리셋되는 임시 카운터
      - 세션 플래그  : interactive/ headless/ permission_mode 등
      - 컨텍스트 압축: compact_boundary 등
      - 플랫폼 정보  : platform dict
      - 하드웨어 적응: hardware_tier 등 (v7.0)
      - MCP 가시성   : mcp_servers, mcp_connected
      - 캐시 래치    : 한 번 정해지면 세션 내내 유지되는 값

    동시성 주의:
      웹 서버 등에서는 여러 스레드가 이 객체를 함께 만집니다. 카운터를 증가시키는
      메서드들은 내부 _lock으로 감싸서 값이 꼬이지 않도록 보호합니다.
    """

    # === 작업 디렉토리 ===
    # cwd          : 현재 작업 디렉토리(작업 중 바뀔 수 있음)
    # original_cwd : 프로세스 시작 시점의 원래 디렉토리(변하지 않는 기준값)
    # project_root : 프로젝트 루트 경로(설정/탐색의 기준)
    cwd: str = ""
    original_cwd: str = ""
    project_root: str = ""

    # === 세션 ===
    # session_id        : 이 세션을 식별하는 고유 UUID. 기본값으로 매번 새로 생성.
    # parent_session_id : 서브 에이전트처럼 다른 세션에서 파생됐다면 부모 세션 ID.
    # session_start_time: 세션 시작 시각(UTC). 총 소요 시간 계산 등에 사용.
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_session_id: str | None = None
    session_start_time: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    # === 설정 (Phase 1에서 로드) ===
    # 실제로는 NexusConfig 객체가 들어오지만 타입을 Any로 둔다.
    # 이유: NexusConfig를 import하면 이 leaf 모듈에 순환 의존이 생길 수 있다.
    config: Any = None

    # === 모델 상태 ===
    # active_model  : 지금 사용 중인 모델 슬롯. "primary"(Qwen) / "auxiliary"(ExaOne).
    # model_override: 특정 요청에서 강제로 다른 모델을 쓰도록 덮어쓴 값(없으면 None).
    # initial_model : 세션 시작 시 기본 모델. override를 해제할 때 되돌아갈 기준값.
    active_model: str = "primary"  # "primary" 또는 "auxiliary"
    model_override: str | None = None
    initial_model: str = "primary"

    # === 사용량 추적 (세션 누적) ===
    # 세션이 시작된 뒤로 계속 더해지는 누적 통계. 로깅/과금/예산 강제에 쓴다.
    # cache_read/ cache_write 토큰은 프롬프트 캐시 적중/기록 분량을 따로 집계한다.
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
    # "이번 턴에서만" 얼마나 썼는지 보기 위한 임시 카운터.
    # start_new_turn()에서 0으로 리셋되며, current_turn_start_time에는 턴 시작
    # 시각(monotonic)이 기록되어 턴 소요 시간을 재는 데 쓰인다.
    current_turn_input_tokens: int = 0
    current_turn_output_tokens: int = 0
    current_turn_tool_calls: int = 0
    current_turn_start_time: float = 0.0

    # === 세션 플래그 ===
    # is_interactive : 사용자와 실시간 상호작용(질문/응답)이 가능한 세션인지.
    # is_headless    : 사람 개입 없이 자동 실행되는 무인 세션인지.
    # is_bare        : 부가 기능을 뺀 최소 실행 모드인지.
    # permission_mode: 현재 권한 모드(위 PermissionModeValue). 기본은 DEFAULT.
    is_interactive: bool = True
    is_headless: bool = False
    is_bare: bool = False
    permission_mode: PermissionModeValue = PermissionModeValue.DEFAULT

    # === 컨텍스트 압축 상태 ===
    # 대화가 길어지면 오래된 메시지를 압축(compact)한다. 그 경계를 여기서 추적한다.
    # compact_boundary  : 이 인덱스 "이후"의 메시지만 API로 전송한다(앞은 요약 대체).
    # last_compact_turn : 마지막으로 압축을 수행한 턴 번호.
    # auto_compact_count: 자동 압축이 몇 번 일어났는지 카운트.
    compact_boundary: int = 0
    last_compact_turn: int = 0
    auto_compact_count: int = 0

    # === 플랫폼 정보 ===
    # OS/아키텍처 등 실행 환경 메타데이터를 담는 dict(부트스트랩에서 채움).
    platform: dict = field(default_factory=dict)

    # === v7.0: 하드웨어 적응 ===
    # 실행 하드웨어 등급에 따라 오케스트레이션 전략을 바꾸기 위한 값들.
    # hardware_tier      : "small" | "medium" | "large" (GPU 메모리/성능 등급)
    # scout_enabled      : 경량 Scout 모델(CPU 4B)을 보조로 켤지 여부
    # orchestration_mode : 여러 모델 협업 vs 단일 모델 운영
    hardware_tier: str = "small"             # "small" | "medium" | "large"
    scout_enabled: bool = False              # Scout(CPU 4B) 활성 여부
    orchestration_mode: str = "multi_model"  # "multi_model" | "single_model"

    # === MCP 서버 가시성 (Phase 2 부트스트랩에서 채움) ===
    # mcp_servers: 서버명 → 등록 결과(등록된 도구 목록/개수 등)를 담는 dict.
    #   예: {"kowiki": {"tools": ["search"], "tool_count": 1}, ...}
    # mcp_connected: 도구가 1개 이상 등록되어 "연결 성공"으로 간주된 서버명 집합.
    # 왜 분리하는가: mcp_servers 는 상세 진단/표시용이고, mcp_connected 는
    #   "몇 개 서버가 실제로 살아 있는가" 를 빠르게 판단하기 위한 요약이다.
    mcp_servers: dict = field(default_factory=dict)
    mcp_connected: set = field(default_factory=set)

    # === 캐시 래치 (한번 설정되면 세션 내에서 변경되지 않는 값) ===
    # "래치(latch)"란 한 번 값이 정해지면 세션 동안 그대로 고정되는 스위치를 뜻한다.
    # fast_mode_latched      : 빠른 응답 모드가 세션에 고정됐는지.
    # thinking_enabled_latched: 사고(thinking) 활성 여부 고정값. 아직 미결정이면 None.
    fast_mode_latched: bool = False
    thinking_enabled_latched: bool | None = None

    # === 내부 Lock (repr에서 제외) ===
    # 카운터 증가 등 동시성 민감한 갱신을 감싸는 스레드 락.
    # repr=False: print(state) 등에서 락 객체가 노출되지 않도록 표시에서 뺀다.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def increment_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """
        토큰 사용량 카운터를 스레드 안전하게 증가시킨다.

        언제 호출하나:
          모델 API를 한 번 호출해 응답을 받은 직후. 이번 호출에서 소비한 입력/출력
          토큰 수를 넘겨주면 누적 통계와 이번 턴 통계에 함께 더해진다.

        매개변수:
          input_tokens  : 이번 API 호출의 입력(프롬프트) 토큰 수
          output_tokens : 이번 API 호출의 출력(생성) 토큰 수

        동작:
          세션 누적값과 이번 턴 값 양쪽을 갱신하고, API 호출 횟수도 1 증가시킨다.
          여러 스레드가 동시에 불러도 값이 꼬이지 않도록 _lock으로 보호한다.
        """
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.current_turn_input_tokens += input_tokens
            self.current_turn_output_tokens += output_tokens
            self.total_api_calls += 1

    def increment_tool_calls(self, count: int = 1) -> None:
        """
        도구 호출 횟수 카운터를 증가시킨다.

        매개변수:
          count: 이번에 실행한 도구 호출 개수(기본 1). 여러 도구를 한 번에 처리했다면
                 그 개수만큼 한 번에 더할 수 있다.

        세션 누적값과 이번 턴 값을 함께 갱신하며, _lock으로 동시성 안전을 보장한다.
        """
        with self._lock:
            self.total_tool_calls += count
            self.current_turn_tool_calls += count

    def start_new_turn(self) -> None:
        """
        새 턴을 시작할 때 호출한다.

        하는 일:
          - 이번 턴 전용 카운터(입력/출력 토큰, 도구 호출)를 모두 0으로 리셋한다.
          - current_turn_start_time에 현재 시각(time.monotonic)을 기록해 이 턴의
            소요 시간을 잴 수 있게 한다. monotonic을 쓰는 이유는 시스템 시계가
            바뀌어도 영향을 받지 않는 단조 증가 시간이기 때문이다.
          - 세션 누적 턴 수(total_turns)를 1 증가시킨다.

        모든 갱신은 _lock으로 감싸 동시성 안전을 보장한다.
        """
        with self._lock:
            self.current_turn_input_tokens = 0
            self.current_turn_output_tokens = 0
            self.current_turn_tool_calls = 0
            self.current_turn_start_time = time.monotonic()
            self.total_turns += 1

    def get_session_summary(self) -> dict:
        """
        세션의 핵심 통계를 dict로 요약해 반환한다.

        용도:
          로깅이나 메트릭 보고에서 "이 세션이 지금까지 무엇을 했는지"를 한눈에
          보여줄 때 쓴다. 반환 dict는 그대로 JSON 로그로 남기기 좋다.

        반환 키:
          session_id, turns, total_input_tokens, total_output_tokens,
          total_api_calls, total_tool_calls, total_duration_seconds(소수 2자리 반올림),
          active_model, permission_mode(문자열 값).
        """
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
# _STATE      : 실제 전역 상태 인스턴스를 담는 모듈 전역 변수. 초기화 전에는 None.
# _STATE_LOCK : 초기화/리셋 시 경쟁 조건을 막기 위한 전용 락.
_STATE: GlobalState | None = None
_STATE_LOCK = threading.Lock()


def get_initial_state(cwd: str | None = None) -> GlobalState:
    """
    전역 상태를 최초 1회 초기화하고 반환한다.

    호출 규칙:
      프로세스 생명주기에서 딱 한 번 부트스트랩 단계에서 호출하는 것을 전제로 한다.
      이미 초기화됐다면 새로 만들지 않고 기존 상태를 그대로 돌려준다(싱글톤 보장).

    매개변수:
      cwd: 시작 작업 디렉토리. None이면 현재 프로세스의 os.getcwd()를 사용한다.

    왜 싱글톤인가:
      CLI, 웹 서버, SDK 등 어떤 진입점으로 실행되든 "같은 상태 객체"를 공유해야
      토큰 추적과 세션 관리가 일관되게 유지되기 때문이다.

    경로 정규화:
      전달된 cwd를 절대 경로로 해석(realpath)한 뒤 유니코드 NFC로 정규화한다.
      NFC 정규화는 한글 등에서 자모가 분해/결합된 표현을 하나로 통일해, 같은 경로가
      서로 다른 문자열로 취급되는 문제를 막는다.
      (Claude Code 대응: realpathSync(cwd()).normalize('NFC'))
    """
    global _STATE
    with _STATE_LOCK:
        # 이미 만들어져 있으면 그대로 재사용한다(두 번째 이후 호출).
        if _STATE is not None:
            return _STATE

        # cwd를 절대 경로로 해석하고 유니코드 NFC로 정규화한다.
        # Claude Code: realpathSync(cwd()).normalize('NFC')
        resolved_cwd = os.path.realpath(cwd or os.getcwd())
        resolved_cwd = unicodedata.normalize("NFC", resolved_cwd)

        # 최초 상태 생성: 현재 디렉토리와 원래 디렉토리를 동일 값으로 시작한다.
        _STATE = GlobalState(
            cwd=resolved_cwd,
            original_cwd=resolved_cwd,
        )
        return _STATE


def get_state() -> GlobalState:
    """
    이미 초기화된 전역 상태를 반환한다.

    주의:
      get_initial_state()로 초기화하기 전에 호출하면 RuntimeError를 발생시킨다.
      "아직 준비되지 않은 상태를 몰래 만들어 쓰는" 실수를 조기에 드러내기 위한
      fail-fast 설계다.
    """
    if _STATE is None:
        raise RuntimeError(
            "GlobalState가 초기화되지 않았습니다. get_initial_state()를 먼저 호출하세요."
        )
    return _STATE


def reset_state_for_testing() -> None:
    """
    전역 상태를 초기화 이전 상태(None)로 되돌린다. 테스트 전용.

    각 테스트가 깨끗한 상태에서 시작하도록 _STATE를 비운다. 프로덕션 코드 경로에서는
    절대 호출하지 않는다. 초기화/리셋 경쟁을 막기 위해 _STATE_LOCK으로 감싼다.
    """
    global _STATE
    with _STATE_LOCK:
        _STATE = None
