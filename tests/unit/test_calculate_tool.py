# Calculate 도구 검증 — 정확성과 "셸을 되돌려주지 않는다"는 경계.
"""
`CalculateTool` 의 계약을 고정한다 (2026-08-07).

[왜 이 도구가 생겼나]
    웹 프롬프트는 원래 "여러 자리 산술은 Bash 로 `python -c` 실행" 을 지시했는데,
    같은 날 보안 사고(테넌트 키 하나로 다른 테넌트의 API 키가 읽힘)로 Bash 를
    웹에서 제거했다. 계산 수단이 함께 사라져 이 도구로 대체했다.

[이 테스트가 지키는 것 — 두 가지]
    1. 정확성 — 금액 계산에서 자릿수가 틀리지 않는다(원래 문제의 본질).
    2. **경계** — 계산을 되찾자고 임의 코드 실행을 되돌려주면 안 된다.
       변수·함수 호출·속성 접근·import 는 전부 거부돼야 한다.
"""

from __future__ import annotations

import pytest

from core.tools.base import PermissionBehavior, ToolUseContext
from core.tools.implementations.calculate_tool import (
    CalculateTool,
    CalculationError,
    evaluate,
    format_result,
)

CTX = ToolUseContext(cwd=".")


# ─────────────────────────────────────────────
# 정확성
# ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1+1", 2),
        ("18764*27", 506628),
        ("(100-30)*2", 140),
        ("2**10", 1024),
        ("7//2", 3),
        ("7%2", 1),
        ("-5+3", -2),
        # 금액 — 쉼표를 그대로 받아들인다(모델이 쉼표를 지우다 자릿수를 틀리는 것이 사고 원인)
        ("1,250,000 * 12", 15000000),
        ("150,000,000 - 1,500,000", 148500000),
    ],
)
def test_arithmetic_is_exact(expr, expected):
    assert evaluate(expr) == expected


def test_large_integers_stay_exact():
    """파이썬 int 라 자릿수 제한 없이 정확하다 — 부동소수 반올림이 끼지 않는다."""
    assert evaluate("999999999999 * 999999999999") == 999999999999 * 999999999999


def test_result_formatting_helps_copying():
    """지수 표기를 쓰지 않고 천단위 쉼표를 넣는다 — 모델이 그대로 옮겨 적어야 한다."""
    assert format_result(15000000) == "15,000,000"
    assert format_result(10000000.0) == "10,000,000"  # 지수 표기(1e+07) 금지
    assert format_result(2.5) == "2.5"


# ─────────────────────────────────────────────
# ★경계 — 계산기를 셸로 되돌리지 않는다
# ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",  # 임의 명령 실행 시도
        "open('/app/config/tenants.yaml').read()",  # 오늘 유출된 바로 그 파일
        "().__class__.__bases__",  # 속성 접근을 통한 우회
        "print(1)",  # 함수 호출 일반
        "abs(-3)",  # 화이트리스트에 없는 내장 함수도 거부
        "x + 1",  # 변수
        "'a' * 3",  # 문자열
        "[1,2,3]",  # 리터럴 컨테이너
        "1 if True else 2",  # 조건식
        "lambda: 1",  # 람다
    ],
)
def test_only_arithmetic_is_allowed(expr):
    """★숫자와 사칙연산 외에는 전부 거부한다 — 이 경계가 뚫리면 Bash 를 뺀 의미가 없다."""
    with pytest.raises(CalculationError):
        evaluate(expr)


def test_huge_power_is_refused_before_computing():
    """거대한 거듭제곱은 계산을 시작하기 전에 막는다(시작하면 되돌릴 수 없다)."""
    with pytest.raises(CalculationError, match="거듭제곱"):
        evaluate("2**999999")


def test_division_by_zero_is_a_clear_message():
    for expr in ("1/0", "1//0", "1%0"):
        with pytest.raises(CalculationError, match="0으로 나눌 수 없습니다"):
            evaluate(expr)


def test_overlong_expression_is_refused():
    with pytest.raises(CalculationError, match="너무 깁니다"):
        evaluate("1+" * 400 + "1")


# ─────────────────────────────────────────────
# 도구 계약
# ─────────────────────────────────────────────


def test_validate_rejects_empty():
    assert CalculateTool().validate_input({"expression": "  "}) is not None
    assert CalculateTool().validate_input({"expression": "1+1"}) is None


@pytest.mark.asyncio
async def test_permissions_allow_without_prompt():
    """부작용이 전혀 없으므로 확인 없이 허용한다(P6 명시적 완화)."""
    result = await CalculateTool().check_permissions({"expression": "1+1"}, CTX)
    assert result.behavior == PermissionBehavior.ALLOW


def test_behavior_flags_declare_no_side_effects():
    tool = CalculateTool()
    assert tool.is_read_only is True
    assert tool.is_concurrency_safe is True


@pytest.mark.asyncio
async def test_call_returns_formatted_result():
    result = await CalculateTool().call({"expression": "1,250,000 * 12"}, CTX)

    assert not result.is_error
    assert "15,000,000" in result.data
    assert result.metadata["result"] == "15,000,000"


@pytest.mark.asyncio
async def test_call_reports_reason_on_bad_input():
    """실패는 이유를 그대로 알려 준다 — 모델이 무엇을 고쳐야 할지 알아야 한다."""
    result = await CalculateTool().call({"expression": "__import__('os')"}, CTX)

    assert result.is_error
    assert "사칙연산" in result.error_message


@pytest.mark.asyncio
async def test_registered_on_web_pool_all_tiers():
    """웹 전 티어에 등록돼 있어야 한다 — Bash 제거로 잃은 계산 수단의 대체재다."""
    from core.bootstrap import _create_web_tool_registry
    from core.model.hardware_tier import HardwareTier

    for tier in (None, HardwareTier.TIER_S, HardwareTier.TIER_M, HardwareTier.TIER_L):
        names = {t.name for t in _create_web_tool_registry(tier).get_all_tools()}
        assert "Calculate" in names, f"tier={tier} 에 Calculate 가 없다"
