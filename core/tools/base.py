"""
BaseTool ABC — Nexus의 모든 "도구(Tool)"가 공통으로 따르는 기본 인터페이스 정의.

이 파일은 Nexus 도구 시스템의 뼈대다. Claude Code의 제네릭 Tool<Input, Output, Progress>
인터페이스를 Python의 ABC(추상 기반 클래스)로 재구현한 것으로, Read/Write/Bash 같은
개별 도구들은 모두 여기 정의된 `BaseTool`을 상속해서 만든다. 즉 "새 도구를 어떻게
만들어야 하는가"의 규격서 역할을 한다.

파일 구성 (크게 두 덩어리):
  A. 지원 타입들 — 도구 실행에 쓰이는 작은 데이터 모델(Pydantic v2 / Enum)
     - PermissionBehavior : 권한 결정 결과(허용/거부/확인) 열거형
     - PermissionResult    : check_permissions()가 돌려주는 권한 판정 결과
     - ToolResult          : 도구 실행 결과(성공/에러) 컨테이너
     - ToolUseContext      : 도구가 실행될 때 필요한 주변 환경 정보
     - ToolProgressEvent   : 장시간 실행 도구의 진행률 이벤트
  B. BaseTool(ABC) — 실제 도구들이 상속하는 추상 기반 클래스.
     약 40개의 멤버를 아래 7개 카테고리로 정리해 둔다:
       1. Identity       (name, description, aliases, group)
       2. Schema         (input_schema)
       3. Behavior Flags (is_read_only 등 — fail-closed 기본값)
       4. Limits         (max_result_size, timeout_seconds, max_retries)
       5. Lifecycle      (validate_input → check_permissions → call → map_result)
       6. Observable     (backfill_observable_input)
       7. UI Hints       (get_progress_label 등 표시용 라벨)

핵심 설계 원칙 — "Fail-Closed(안전 우선) 기본값":
  도구는 기본적으로 가장 제한적인 상태로 동작한다. 새 도구를 만들 때
  개발자가 명시적으로 완화하지 않으면 아래처럼 가장 보수적으로 취급된다.
  - is_read_only          = False  → 시스템을 바꾸는 쓰기 도구로 가정
  - is_concurrency_safe   = False  → 병렬 실행 금지, 순차 실행만
  - is_destructive        = False  → (선언 안 하면) 파괴적 아님으로 두되 기본은 안전측
  - requires_confirmation = False
  이렇게 해두면 "깜빡하고 안전장치를 안 켠" 실수가 곧바로 위험으로 이어지지 않는다.

의존/노출:
  - Pydantic v2(BaseModel, Field)로 데이터 모델을 정의한다(프로젝트 표준 P5).
  - 개별 도구 구현체는 core/tools/implementations/ 에 위치하며 이 BaseTool을 상속한다.
  - 권한 판정(PermissionResult)은 core/permission/ 의 5계층 파이프라인과 맞물린다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# 이 모듈 전용 로거. 프로젝트 규칙에 따라 "nexus.{module}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.tools")


# ─────────────────────────────────────────────
# 지원 타입 — 도구 실행 과정에서 오가는 작은 데이터 모델들
# ─────────────────────────────────────────────
class PermissionBehavior(str, Enum):
    """
    권한 판정의 최종 결과를 나타내는 열거형(Enum).

    도구를 실제로 실행해도 되는지 결정할 때 이 세 값 중 하나로 표현한다.
    str을 함께 상속하므로 값 자체가 문자열("allow" 등)이라 JSON 직렬화/로그에 편하다.
    """

    ALLOW = "allow"  # 그대로 실행 허용
    DENY = "deny"  # 실행 차단 (권한 위반 등)
    ASK = "ask"  # 사용자에게 실행 여부를 되물음 (CLI 인터랙티브 확인)


class PermissionResult(BaseModel):
    """
    도구의 check_permissions()가 돌려주는 권한 판정 결과 객체.

    behavior 하나만 봐도 되지만, DENY/ASK인 경우 왜 그런지 설명(message)이나
    추가 상세정보(details)를 함께 담아 상위 권한 파이프라인/UI가 활용하도록 한다.

    필드:
      behavior : ALLOW/DENY/ASK 중 하나 (기본값 ALLOW — 특별한 제약이 없으면 허용)
      message  : 판정 사유를 사람이 읽을 수 있는 문장으로 (선택)
      details  : 판정에 쓰인 부가 데이터(예: 차단된 경로 등)를 담는 자유형 dict (선택)
    """

    behavior: PermissionBehavior = PermissionBehavior.ALLOW
    message: str | None = None
    details: dict[str, Any] | None = None


class ToolResult(BaseModel):
    """
    도구 한 번의 실행 결과를 담는 컨테이너.

    Claude Code의 ToolResultBlockParam에 대응하며, 여기 담긴 data가 직렬화되어
    최종적으로 tool_result 메시지의 content(모델이 다시 읽는 내용)가 된다.

    직접 생성하기보다 아래 팩토리 메서드 success()/error()를 쓰는 것을 권장한다
    (프로젝트 표준 P5의 Factory method 패턴).

    필드:
      data          : 실행 산출물(성공 시 결과, 에러 시 에러 문자열이 들어가기도 함)
      is_error      : 에러 결과인지 여부. True면 map_result()에서 에러 태그로 감싼다
      error_message : 에러 사유 문자열 (is_error=True일 때 사용)
      metadata      : 로깅/후처리에 쓰는 부가 정보(자유형 dict)
    """

    data: Any
    is_error: bool = False
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(cls, data: Any, **metadata) -> ToolResult:
        """
        성공 결과를 만드는 팩토리.

        키워드 인자로 넘긴 값들은 모두 metadata로 모여 들어간다.
        예) ToolResult.success(내용, bytes_read=1024)
        """
        return cls(data=data, metadata=metadata)

    @classmethod
    def error(cls, message: str, **metadata) -> ToolResult:
        """
        에러 결과를 만드는 팩토리.

        message는 data와 error_message 양쪽에 함께 담아, 결과를 문자열로 볼 때나
        에러 사유로 볼 때나 동일한 메시지를 얻도록 한다. is_error는 True로 고정.
        """
        return cls(data=message, is_error=True, error_message=message, metadata=metadata)


class ToolUseContext(BaseModel):
    """
    도구가 실행되는 "상황(context)" 정보를 한데 모아 전달하는 객체.

    도구의 call()/check_permissions()에 항상 함께 넘겨, 도구가 현재 작업 디렉토리,
    세션/에이전트 식별자, 취소 시그널 같은 주변 환경을 알 수 있게 한다.

    arbitrary_types_allowed=True 를 켠 이유:
      abort_signal에 Pydantic이 모르는 asyncio.Event 같은 임의 타입을 담아야 하기 때문.
    """

    model_config = {"arbitrary_types_allowed": True}

    cwd: str  # 현재 작업 디렉토리(경로 검증·상대경로 해석의 기준점)
    session_id: str = ""  # 대화 세션 식별자
    agent_id: str | None = None  # 서브 에이전트 구별용(메인이면 None)
    tool_use_id: str = ""  # 이 도구 호출 한 건의 고유 ID
    read_file_timestamps: dict[str, float] = Field(default_factory=dict)  # 파일별 최근 읽은 시각
    abort_signal: Any = None  # asyncio.Event — 실행 중단(취소) 시그널
    permission_mode: str = "default"  # 현재 권한 모드(default/accept_edits 등)
    parent_tool_use_id: str | None = None  # 부모 도구 호출 ID(중첩 실행 추적용)
    options: dict[str, Any] = Field(default_factory=dict)  # 도구별 추가 옵션 자유 저장소


class ToolProgressEvent(BaseModel):
    """
    장시간 실행되는 도구가 중간 진행 상황을 알릴 때 쓰는 이벤트.

    예를 들어 오래 걸리는 검색/빌드 도구가 진행률을 UI에 보여줄 때 사용한다.

    필드:
      tool_use_id : 어떤 도구 호출의 진행인지 식별
      progress    : 진행률 0.0 ~ 1.0 (0=시작, 1=완료)
      message     : 현재 단계 설명 문구(선택)
    """

    tool_use_id: str
    progress: float  # 0.0 ~ 1.0
    message: str = ""


# ─────────────────────────────────────────────
# BaseTool ABC — 모든 도구가 상속하는 추상 기반 클래스
# ─────────────────────────────────────────────
class BaseTool(ABC):
    """
    모든 도구의 기본(추상) 클래스.

    새 도구를 만들 때 이 클래스를 상속하고, 최소한 아래 5개(@abstractmethod로 표시된 것)를
    반드시 구현해야 한다:
      - name           : 도구 고유 이름
      - description    : 모델에게 보여줄 설명
      - input_schema   : 입력 JSON Schema
      - check_permissions() : 도구별 권한 검사
      - call()         : 실제 실행 로직

    나머지 멤버들은 합리적인 기본값(대부분 fail-closed)이 이미 구현돼 있어,
    특별히 다르게 동작해야 할 때만 하위 클래스에서 오버라이드하면 된다.

    아래 멤버들은 주석의 ═══ 구분선으로 7개 카테고리로 묶여 있으니,
    비슷한 성격의 속성/메서드를 함께 파악하기 쉽다.
    """

    # ═══ 1. Identity — 도구를 식별하는 이름/설명 계열 ═══

    @property
    @abstractmethod
    def name(self) -> str:
        """
        도구의 고유 이름. 모델이 tool_use(도구 호출)에서 이 이름으로 도구를 지목한다.
        레지스트리 등록 키이자 사용자에게 노출되는 기본 이름이므로 중복 없이 유일해야 한다.
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        모델(LLM)에게 제공되는 도구 설명문.
        모델은 이 설명만 보고 "언제 이 도구를 쓸지"를 판단하므로 명확하게 작성해야 한다.
        """
        ...

    @property
    def aliases(self) -> list[str]:
        """
        도구의 대체 이름 목록. 모델이 정식 name 대신 비슷한 다른 이름을 부르더라도
        여기에 등록돼 있으면 매칭시켜 준다. 기본값은 빈 리스트(별칭 없음).
        """
        return []

    @property
    def group(self) -> str:
        """
        도구가 속한 그룹 이름. 주로 UI에서 도구들을 카테고리별로 묶어 보여줄 때 쓴다.
        기본값은 "default".
        """
        return "default"

    # ═══ 2. Schema — 입력 형식 정의 ═══

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """
        도구 입력을 기술하는 JSON Schema(dict).
        vLLM/OpenAI의 function calling에서 parameters 자리에 그대로 전달되어,
        모델이 어떤 인자를 어떤 형식으로 넘겨야 하는지 알 수 있게 한다.
        """
        ...

    # ═══ 3. Behavior Flags — 도구의 성격을 나타내는 플래그 (fail-closed 기본값) ═══

    @property
    def is_read_only(self) -> bool:
        """
        True면 시스템 상태를 전혀 바꾸지 않는 읽기 전용 도구라는 뜻.
        기본값 False — 명시하지 않으면 "쓰기 도구"로 보수적으로 가정한다(fail-closed).
        """
        return False

    @property
    def is_destructive(self) -> bool:
        """
        True면 되돌리기 어려운(파괴적) 변경을 수행한다는 표시(예: 파일 삭제).
        기본값 False.
        """
        return False

    @property
    def is_concurrency_safe(self) -> bool:
        """
        True면 다른 도구와 동시에 병렬 실행해도 안전하다는 뜻(주로 읽기 도구).
        기본값 False — 안전이 보장되지 않으면 순차 실행하도록 보수적으로 잡는다.
        """
        return False

    @property
    def should_defer(self) -> bool:
        """
        True면 모델 스트리밍이 끝난 뒤로 실행을 미룰 수 있는 도구라는 뜻.
        기본값 False(즉시 실행 대상).
        """
        return False

    @property
    def requires_confirmation(self) -> bool:
        """
        True면 어떤 모드에서든 실행 전 사용자 확인을 반드시 받아야 한다.
        기본값 False.
        """
        return False

    @property
    def is_enabled(self) -> bool:
        """
        False면 이 도구를 현재 비활성으로 취급해 도구 풀에서 제외한다.
        기본값 True(활성).
        """
        return True

    @property
    def is_user_facing(self) -> bool:
        """
        True면 이 도구의 실행/결과가 사용자에게 보이는 도구라는 뜻.
        기본값 True. False면 내부 전용으로 조용히 동작한다.
        """
        return True

    @property
    def has_side_effects(self) -> bool:
        """
        True면 외부 시스템에 부수 효과(side effect)를 일으킨다는 뜻.
        기본값은 not is_read_only — 즉 읽기 전용이 아니면 부수 효과가 있다고 본다.
        """
        return not self.is_read_only

    # ═══ 4. Limits — 실행 관련 한계값 ═══

    @property
    def max_result_size(self) -> int:
        """
        결과로 담을 수 있는 최대 크기(문자 수). 이 값을 넘으면 디스크로 흘려보내는 등
        별도 처리를 해 컨텍스트 폭주를 막는다. 기본값 100,000자.
        """
        return 100_000

    @property
    def timeout_seconds(self) -> float:
        """도구 실행 제한 시간(초). 이 시간을 넘기면 타임아웃 처리한다. 기본값 120초."""
        return 120.0

    @property
    def max_retries(self) -> int:
        """실행 실패 시 자동 재시도 횟수. 기본값 0(재시도 안 함)."""
        return 0

    # ═══ 5. Lifecycle Methods — 도구 실행의 생애주기 단계 ═══
    # 실행 순서: validate_input → check_permissions → call → map_result

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        JSON Schema 검증을 통과한 뒤 수행하는 추가 도메인 검증.

        스키마만으로는 표현하기 어려운 규칙(예: 두 인자의 상호 조건 등)을 여기서 확인한다.
        반환값 규칙:
          - None  : 입력이 유효함
          - str   : 검증 실패. 반환된 문자열이 에러 메시지가 된다
        기본 구현은 항상 None(추가 검증 없음).
        """
        return None

    @abstractmethod
    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        이 도구에 특화된 권한 검사(도구 고유 단계).

        전체 5계층 권한 파이프라인 중 "도구 자신이 판단하는" 부분에 해당한다.
        입력과 실행 컨텍스트를 보고 ALLOW/DENY/ASK를 담은 PermissionResult를 돌려준다.
        (추상 메서드이므로 각 도구가 반드시 구현해야 한다.)
        """
        ...

    @abstractmethod
    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        도구의 실제 실행 로직(핵심 본체).

        validate_input과 check_permissions 등 앞선 모든 검증/권한 확인을 통과한 뒤에만
        호출된다. 실행 결과는 ToolResult로 감싸서 반환한다.
        (추상 메서드이므로 각 도구가 반드시 구현해야 한다.)
        """
        ...

    def map_result(self, result: ToolResult) -> str:
        """
        ToolResult를 tool_result 메시지의 content(문자열)로 변환한다.

        에러인 경우 <tool_use_error>...</tool_use_error> 태그로 감싸,
        상위 계층/모델이 "이건 에러 결과"임을 명확히 인지하도록 한다.
        정상 결과는 data를 문자열로 바꿔 그대로 돌려준다.
        """
        if result.is_error:
            # 에러 메시지가 따로 없으면 data를 대신 사용한다.
            return f"<tool_use_error>{result.error_message or result.data}</tool_use_error>"
        return str(result.data)

    # ═══ 6. Observable — Hook에 넘길 관찰용 입력 가공 ═══

    def backfill_observable_input(
        self, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Hook 시스템 등에 넘길 "관찰 가능한(observable) 입력"을 만든다.

        원본을 그대로 노출하면 곤란한 민감 정보(토큰/비밀번호 등)를 이 지점에서
        가공·제거할 수 있다. 기본 구현은 입력을 얕은 복사(copy)해 그대로 돌려준다.
        """
        return input_data.copy()

    # ═══ 7. UI Hints — 사용자 표시용 라벨/요약 ═══

    def get_user_facing_name(self) -> str:
        """사용자 화면에 표시할 이름. 기본은 내부 name과 동일하다."""
        return self.name

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        """실행 중임을 알리는 진행 상태 라벨. 기본은 "Running {name}..." 형태."""
        return f"Running {self.name}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        """
        입력을 짧게 요약한 문자열(로깅/UI용).

        입력이 비어 있으면 빈 문자열, 아니면 첫 번째 값을 문자열로 바꿔
        앞 100자까지만 잘라서 돌려준다(과도한 로그 방지).
        """
        if not input_data:
            return ""
        # 입력 dict의 첫 번째 값을 뽑아 대표 요약으로 사용한다.
        first_value = str(next(iter(input_data.values()), ""))
        return first_value[:100]

    # ═══ Schema Export — 모델에게 넘길 도구 스키마 생성 ═══

    def to_schema(self) -> dict[str, Any]:
        """
        모델(LLM)에게 전달할 도구 스키마 dict를 조립한다.
        name/description/input_schema 세 가지를 묶어 반환한다.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        # 디버깅/로그에서 도구를 한눈에 알아보기 쉽게, 이름과 주요 플래그를 축약해 보여준다.
        # RO=읽기전용, CS=병렬안전, DESTRUCTIVE=파괴적.
        flags = []
        if self.is_read_only:
            flags.append("RO")
        if self.is_concurrency_safe:
            flags.append("CS")
        if self.is_destructive:
            flags.append("DESTRUCTIVE")
        # 켜진 플래그가 하나라도 있으면 대괄호로 묶어 덧붙인다.
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"<Tool: {self.name}{flag_str}>"
