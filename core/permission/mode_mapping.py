"""
권한 모드 매핑 — PermissionModeValue → PermissionMode 변환.

왜 이 파일이 필요한가 (배경):
  Nexus에는 "권한 모드"를 나타내는 enum이 두 군데에 따로 있다.
    1) core/state.py 의 PermissionModeValue
       (default/auto/plan/trust/bypass/headless/deny_all — 소문자 값)
       → GlobalState.permission_mode 가 쓰는, 세션/CLI 레벨의 모드
    2) core/permission/types.py 의 PermissionMode
       (DEFAULT/ACCEPT_EDITS/BYPASS_PERMISSIONS/DONT_ASK/PLAN/AUTO/BUBBLE)
       → 5계층 권한 파이프라인(PermissionPipeline)이 쓰는 모드

  두 enum은 1:1로 대응하지 않고(값 개수·이름도 다름) 변환 함수도 없었다.
  파이프라인을 실제로 배선하려면 "세션 모드(PermissionModeValue)"를
  "파이프라인 모드(PermissionMode)"로 옮기는 단 하나의 공식 변환점이 필요하다.
  이 파일이 그 단일 변환점이다.

Fail-closed 원칙:
  매핑에 없는(알 수 없는) 값이 들어오면 가장 보수적인 DEFAULT로 떨어뜨린다.
  즉 "모르는 모드는 가장 안전한 기본 모드로 취급"한다.
  이렇게 하면 오타·신규 값·잘못된 문자열이 흘러들어와도 권한이 과도하게
  풀리는 사고를 막을 수 있다(안전한 쪽으로 실패).

노출 API (이 모듈이 밖으로 제공하는 것):
  - map_mode_value_to_permission_mode(value): 유일한 공개 함수.
    세션 모드 → 파이프라인 모드 변환의 단일 진입점.
  - _MODE_VALUE_TO_PERMISSION_MODE: 내부 매핑 테이블(밑줄 접두사 = 비공개).

의존 모듈:
  - core.permission.types.PermissionMode  (변환 결과 타입)
  - core.state.PermissionModeValue        (변환 입력 타입)
  두 모듈만 참조하며 역방향 의존이나 순환 import는 없다.

사용처 (누가 이 함수를 부르나):
  권한 파이프라인을 배선하는 지점에서, GlobalState.permission_mode(세션 모드)를
  PermissionPipeline이 이해하는 PermissionMode로 바꿔 넘길 때 호출한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

from core.permission.types import PermissionMode
from core.state import PermissionModeValue

# ─────────────────────────────────────────────
# 매핑 테이블 — PermissionModeValue → PermissionMode
# ─────────────────────────────────────────────
# 각 항목 옆에 "왜 이 PermissionMode로 매핑하는지" 근거를 단다.
_MODE_VALUE_TO_PERMISSION_MODE: dict[PermissionModeValue, PermissionMode] = {
    # default → DEFAULT
    #   둘 다 "읽기만 자동 허용, 나머지는 물어봄"이라는 기본 정책이라 그대로 대응.
    PermissionModeValue.DEFAULT: PermissionMode.DEFAULT,
    # accept_edits → ACCEPT_EDITS
    #   둘 다 "파일 수정(FILE_WRITE)은 자동 허용, Bash 등 나머지는 확인" 정책이라
    #   이름·의미가 그대로 대응한다. (CLI Stage 1 A1에서 세션 값이 신설되어
    #   기존에 도달 불가였던 파이프라인 ACCEPT_EDITS로 가는 경로가 열렸다.)
    PermissionModeValue.ACCEPT_EDITS: PermissionMode.ACCEPT_EDITS,
    # auto → AUTO
    #   둘 다 "대부분 허용하되 위험한 것만 물어봄" 정책이라 그대로 대응.
    PermissionModeValue.AUTO: PermissionMode.AUTO,
    # plan → PLAN
    #   둘 다 "읽기/계획만 허용, 쓰기는 전부 거부" 정책이라 그대로 대응.
    PermissionModeValue.PLAN: PermissionMode.PLAN,
    # trust → BYPASS_PERMISSIONS
    #   trust(신뢰)는 "사용자가 이미 신뢰해 확인 없이 진행"을 뜻한다. 파이프라인
    #   쪽에서 이 의미에 가장 가까운 것은 ASK→ALLOW로 바꿔 확인을 건너뛰는
    #   BYPASS_PERMISSIONS다. (파이프라인에 별도 'trust' 모드는 없다.)
    PermissionModeValue.TRUST: PermissionMode.BYPASS_PERMISSIONS,
    # bypass → BYPASS_PERMISSIONS
    #   이름·의미가 그대로 대응(모든 확인 우회).
    PermissionModeValue.BYPASS: PermissionMode.BYPASS_PERMISSIONS,
    # headless → DONT_ASK
    #   headless(무인/비대화형)에서는 사용자에게 물어볼 수 없다. 물어봐야 할
    #   상황(ASK)을 거부(DENY)로 바꾸는 DONT_ASK가 정확히 그 의미다(CI/CD용).
    PermissionModeValue.HEADLESS: PermissionMode.DONT_ASK,
    # deny_all → DONT_ASK
    #   진짜 "전부 거부(all-deny)" 전용 모드는 파이프라인에 아직 없다. 현존
    #   모드 중 가장 보수적인 것이 DONT_ASK(읽기만 허용, 나머지 ASK→DENY)이므로
    #   그것으로 대응한다.
    #   TODO(nexus): 완전한 all-deny(읽기조차 거부) 모드는 후속 단계에서
    #     PermissionMode에 신규 추가하여 정확히 대응한다. 지금은 DONT_ASK로
    #     근사한다(읽기 전용 도구만 통과 — fail-closed 관점에서 안전한 근사).
    PermissionModeValue.DENY_ALL: PermissionMode.DONT_ASK,
}


def map_mode_value_to_permission_mode(
    value: PermissionModeValue | str,
) -> PermissionMode:
    """
    세션 모드(PermissionModeValue)를 파이프라인 모드(PermissionMode)로 변환한다.

    이 함수가 프로젝트 전체에서 두 enum을 잇는 "유일한 공식 변환점"이다.
    다른 곳에서 개별적으로 if/elif로 매핑을 흉내내지 말고 반드시 이 함수를 쓴다
    (매핑 규칙이 한 군데에만 있어야 유지보수·감사가 쉽기 때문).

    처리 흐름(3단계):
      1) 문자열이 들어오면 PermissionModeValue enum으로 정규화한다.
         정규화에 실패하면(모르는 문자열) 즉시 DEFAULT로 안전하게 반환.
      2) 정규화된 enum으로 매핑 테이블을 조회한다.
      3) 테이블에 없으면(방어적) DEFAULT로 떨어뜨린다.

    Args:
        value: PermissionModeValue enum 또는 그 문자열 값("default" 등).
               GlobalState.permission_mode 는 enum이지만, 문자열로 전달되는
               경로(예: ToolUseContext.permission_mode: str)도 있으므로 둘 다 받는다.
               enum과 문자열을 모두 받아 호출부에서 타입 변환 부담을 없앤다.

    Returns:
        대응하는 PermissionMode. 알 수 없는 값이면 fail-closed로 DEFAULT.
        (예외를 던지지 않고 항상 유효한 PermissionMode를 돌려주므로 호출부에서
         별도 예외 처리가 필요 없다.)
    """
    # 문자열로 들어오면 PermissionModeValue enum으로 정규화한다.
    # (enum(value)는 값이 유효하지 않으면 ValueError를 던지므로 감싼다.)
    if isinstance(value, str) and not isinstance(value, PermissionModeValue):
        try:
            value = PermissionModeValue(value)
        except ValueError:
            # 모르는 문자열 → 가장 안전한 기본 모드
            return PermissionMode.DEFAULT

    # 매핑 조회. 테이블에 없으면(방어적) DEFAULT로 떨어뜨린다.
    return _MODE_VALUE_TO_PERMISSION_MODE.get(value, PermissionMode.DEFAULT)
