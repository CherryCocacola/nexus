# Calculate 도구 — 셸 없이 산술식만 정확히 계산한다.
"""
Calculate 도구 — 모델의 암산을 대신하는 순수 계산기.

[왜 필요한가 — 2026-08-07]
  원래 웹 프롬프트는 "여러 자리 산술은 암산하지 말고 Bash 로 `python -c "print(…)"`
  를 실행하라"고 지시했다. 그런데 같은 날 Bash 를 웹 표면에서 제거했다(테넌트 키
  하나로 다른 테넌트의 API 키가 읽혀서 — `_create_web_tool_registry` 주석 참조).
  그 결과 웹에서 **정확한 계산 수단이 함께 사라졌다.**

  계산을 위해 임의 명령 실행 권한을 되돌려 줄 이유는 없다. 필요한 것은 산술뿐이므로
  산술만 하는 도구를 만든다. 이 도구는 파일도 네트워크도 프로세스도 건드리지 않는다.

[왜 eval 이 아니라 AST 인가]
  `eval()` 은 문자열 하나로 임의 코드를 실행한다 — 모델이 만든 문자열을 그대로
  넘기면 Bash 를 없앤 의미가 사라진다. 그래서 파이썬 파서로 **식(expression)만**
  파싱한 뒤, 허용한 노드 종류만 직접 계산한다. 변수·속성 접근·함수 호출·문자열은
  전부 거부한다. 화이트리스트 밖은 파싱 단계에서 막힌다.

[정확도]
  정수 연산은 파이썬 int 라 자릿수 제한 없이 정확하다(금액 계산이 주 용도).
  나눗셈만 실수가 되며, 이때는 결과를 그대로 돌려주되 표기는 지수형을 쓰지 않는다.
  쉼표가 섞인 금액(`1,250,000`)을 그대로 받아들인다 — 모델이 쉼표를 지우다 자릿수를
  틀리는 것이 실제 사고 원인이었기 때문이다.

작성자: 이현수 / 작성일: 2026-08-07
"""

from __future__ import annotations

import ast
import re
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 계산을 허용하는 이항 연산자. 나머지(행렬곱·비트연산 등)는 전부 거부한다.
_BIN_OPS: dict[type, Any] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY_OPS: dict[type, Any] = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

# 거듭제곱 폭발 방지. 2**100000 같은 식은 메모리·CPU 를 통째로 먹는다.
# 지수와 밑을 함께 제한해 계산 전에 거절한다(계산을 시작하면 이미 늦다).
_MAX_EXPONENT = 1000
_MAX_POW_BASE = 10**15
# 입력 길이 상한 — 비정상적으로 긴 식은 파싱 자체를 하지 않는다.
_MAX_EXPR_CHARS = 500

# 숫자 사이의 쉼표만 제거한다(1,250,000 → 1250000).
# 앞뒤가 숫자인 쉼표만 지우므로 다른 쉼표는 그대로 남아 파싱 단계에서 거부된다.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d)")


class CalculationError(ValueError):
    """계산할 수 없는 식일 때 올린다(사용자에게 그대로 보여줄 한글 메시지 포함)."""


def _eval_node(node: ast.AST) -> int | float:
    """허용된 노드만 재귀적으로 계산한다. 그 외는 즉시 거부한다."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculationError("숫자가 아닌 값은 계산할 수 없습니다.")
        return node.value

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculationError("지원하지 않는 단항 연산자입니다.")
        return op(_eval_node(node.operand))

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise CalculationError("지원하지 않는 연산자입니다(+ - * / // % ** 만 가능).")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            # 계산을 시작하기 전에 막는다 — 거대한 거듭제곱은 시작하면 되돌릴 수 없다.
            if abs(right) > _MAX_EXPONENT or abs(left) > _MAX_POW_BASE:
                raise CalculationError(
                    f"거듭제곱이 너무 큽니다(지수는 {_MAX_EXPONENT} 이하, "
                    f"밑은 {_MAX_POW_BASE:,} 이하만 계산합니다)."
                )
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise CalculationError("0으로 나눌 수 없습니다.")
        return op(left, right)

    # 변수·함수 호출·속성 접근·문자열·리스트 등은 전부 여기로 떨어진다.
    raise CalculationError("숫자와 사칙연산만 사용할 수 있습니다.")


def evaluate(expression: str) -> int | float:
    """산술식 문자열을 계산한다. 허용 범위를 벗어나면 CalculationError."""
    expr = (expression or "").strip()
    if not expr:
        raise CalculationError("계산할 식이 비어 있습니다.")
    if len(expr) > _MAX_EXPR_CHARS:
        raise CalculationError(f"식이 너무 깁니다({_MAX_EXPR_CHARS}자 이하).")

    expr = _THOUSANDS.sub("", expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalculationError(f"식을 해석할 수 없습니다: {expr}") from e
    return _eval_node(tree.body)


def format_result(value: int | float) -> str:
    """사람이 읽고 그대로 옮겨 적을 수 있게 표기한다.

    프로젝트의 "숫자 표기" 규칙상 모델은 결과를 **그대로 복사**해야 한다. 그래서
    지수 표기(1e+07)를 쓰지 않고, 정수는 천단위 쉼표를 넣어 자릿수를 세기 쉽게 한다.
    """
    if isinstance(value, int):
        return f"{value:,}"
    if value == int(value) and abs(value) < 10**15:
        return f"{int(value):,}"
    # 부동소수는 유효자리를 남기되 지수 표기를 피한다.
    return f"{value:,.10f}".rstrip("0").rstrip(".")


class CalculateTool(BaseTool):
    """숫자와 사칙연산만으로 이루어진 식을 정확히 계산하는 도구."""

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "Calculate"

    @property
    def description(self) -> str:
        return (
            "Evaluate an arithmetic expression exactly. Use this instead of mental "
            "math for multi-digit arithmetic, especially money amounts. Supports "
            "+ - * / // % ** and parentheses; thousands separators are accepted "
            "(e.g. '1,250,000 * 12'). Numbers only — no variables or functions."
        )

    @property
    def group(self) -> str:
        return "compute"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Arithmetic expression, e.g. '1,250,000 * 12' or '(18764*27)-100'"
                    ),
                }
            },
            "required": ["expression"],
        }

    # ═══ 3. Behavior Flags ═══
    # 순수 계산이라 부작용이 없다. fail-closed 기본값을 명시적으로 완화한다(P6).

    @property
    def is_read_only(self) -> bool:
        # 파일·네트워크·프로세스를 전혀 건드리지 않는다.
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        # 상태를 공유하지 않으므로 병렬 실행해도 안전하다.
        return True

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        if not str(input_data.get("expression", "")).strip():
            return "expression은 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """확인 없이 허용한다 — 외부에 영향이 전혀 없는 순수 계산이다."""
        return PermissionResult(
            behavior=PermissionBehavior.ALLOW,
            message=f"Calculate: {input_data.get('expression', '')}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """식을 계산해 결과를 돌려준다. 실패는 이유를 그대로 알려 준다."""
        expression = str(input_data["expression"])
        try:
            value = evaluate(expression)
        except CalculationError as e:
            return ToolResult.error(str(e))
        except (OverflowError, MemoryError, RecursionError) as e:
            return ToolResult.error(f"계산이 너무 큽니다: {type(e).__name__}")

        formatted = format_result(value)
        return ToolResult.success(
            f"{expression} = {formatted}",
            expression=expression,
            result=formatted,
            raw=str(value),
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Calculating {input_data.get('expression', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("expression", "")
