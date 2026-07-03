"""
core/permission/mode_mapping.py 단위 테스트.

무엇을 고정하는가:
    세션 모드(PermissionModeValue, 소문자 문자열 값)를 파이프라인 모드
    (PermissionMode)로 옮기는 "단일 공식 변환점"인 map_mode_value_to_permission_mode를
    검증한다. 이 변환은 권한 배선(커밋 20f5e7c/cbd48e4)의 진입 계약이므로,
    7개 값의 매핑과 fail-closed(미지값→DEFAULT) 동작이 절대 흔들리면 안 된다.

왜 중요한가:
    누군가 매핑 테이블을 실수로 바꾸면(예: trust를 DEFAULT로) 배포 환경의 권한
    강도가 조용히 달라진다. 이 테스트가 그 회귀를 즉시 잡는다.
"""

from __future__ import annotations

import pytest

from core.permission.mode_mapping import map_mode_value_to_permission_mode
from core.permission.types import PermissionMode
from core.state import PermissionModeValue

# 7개 세션 모드 → 파이프라인 모드의 "정답표".
# (구현 사실: default→DEFAULT, auto→AUTO, plan→PLAN, trust/bypass→BYPASS_PERMISSIONS,
#  headless/deny_all→DONT_ASK)
_EXPECTED: dict[PermissionModeValue, PermissionMode] = {
    PermissionModeValue.DEFAULT: PermissionMode.DEFAULT,
    PermissionModeValue.AUTO: PermissionMode.AUTO,
    PermissionModeValue.PLAN: PermissionMode.PLAN,
    PermissionModeValue.TRUST: PermissionMode.BYPASS_PERMISSIONS,
    PermissionModeValue.BYPASS: PermissionMode.BYPASS_PERMISSIONS,
    PermissionModeValue.HEADLESS: PermissionMode.DONT_ASK,
    PermissionModeValue.DENY_ALL: PermissionMode.DONT_ASK,
}


class TestModeMappingEnumInput:
    """enum(PermissionModeValue)을 직접 넣었을 때의 7개 매핑을 고정한다."""

    @pytest.mark.parametrize(("value", "expected"), list(_EXPECTED.items()))
    def test_enum_value_maps_to_expected_permission_mode(
        self, value: PermissionModeValue, expected: PermissionMode
    ) -> None:
        """각 세션 모드 enum이 정확히 대응하는 파이프라인 모드로 변환돼야 한다."""
        assert map_mode_value_to_permission_mode(value) == expected


class TestModeMappingStringInput:
    """문자열 값("default" 등)을 넣어도 enum과 동일하게 변환돼야 한다."""

    @pytest.mark.parametrize(("value", "expected"), list(_EXPECTED.items()))
    def test_string_value_maps_same_as_enum(
        self, value: PermissionModeValue, expected: PermissionMode
    ) -> None:
        """문자열 입력(value.value)도 enum 입력과 같은 결과를 내야 한다(둘 다 허용)."""
        # PermissionModeValue는 str 서브클래스지만, ".value"로 순수 문자열을 넘겨
        # "문자열 경로"를 명시적으로 태운다.
        assert map_mode_value_to_permission_mode(str(value.value)) == expected


class TestModeMappingFailClosed:
    """알 수 없는 입력은 가장 보수적인 DEFAULT로 떨어져야 한다(fail-closed)."""

    def test_unknown_string_returns_default(self) -> None:
        """매핑 테이블에 없는 문자열은 DEFAULT로 처리된다."""
        assert map_mode_value_to_permission_mode("nonexistent_mode") == PermissionMode.DEFAULT

    def test_empty_string_returns_default(self) -> None:
        """빈 문자열도 알 수 없는 값이므로 DEFAULT로 떨어진다."""
        assert map_mode_value_to_permission_mode("") == PermissionMode.DEFAULT

    def test_garbage_string_returns_default(self) -> None:
        """형식이 전혀 다른 문자열도 예외 없이 DEFAULT로 흡수돼야 한다(방어적)."""
        assert map_mode_value_to_permission_mode("BYPASS_PERMISSIONS") == PermissionMode.DEFAULT
