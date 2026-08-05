# core/permission/pipeline.py categorize_tool_name 검증 — 이름 기반 분류 단일 출처.
"""
categorize_tool_name(공개 함수)의 분류 규칙을 고정한다 (CLI Stage 1 A1).

왜 중요한가:
    이 함수는 파이프라인 _categorize_tool의 1단계(이름 매칭)이자, CLI accept_edits
    모드의 "파일 수정 부류만 자동 승인" 판정의 공동 출처다. 분류가 흔들리면
    권한 강도(무엇이 자동 승인되는가)가 조용히 달라지므로 표를 테스트로 고정한다.

핵심 계약:
    - 표에 없는 이름은 None — CLI는 None을 "자동 승인 금지"로 해석한다(fail-closed).
      (파이프라인 쪽 FILE_WRITE 폴백은 tool 객체 기반 _categorize_tool에만 있다.)
"""

from __future__ import annotations

import pytest

from core.permission.pipeline import categorize_tool_name
from core.permission.types import ToolCategory


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        # FILE_WRITE — accept_edits 자동 승인 대상 4종(대소문자 무관)
        ("Write", ToolCategory.FILE_WRITE),
        ("Edit", ToolCategory.FILE_WRITE),
        ("MultiEdit", ToolCategory.FILE_WRITE),
        ("NotebookEdit", ToolCategory.FILE_WRITE),
        # BASH — accept_edits에서도 반드시 확인
        ("Bash", ToolCategory.BASH),
        # READONLY 대표값
        ("Read", ToolCategory.READONLY),
        ("Grep", ToolCategory.READONLY),
        ("TodoWrite", ToolCategory.READONLY),
        # NETWORK / AGENT / MCP
        ("WebFetch", ToolCategory.NETWORK),
        ("Agent", ToolCategory.AGENT),
        ("mcp__server__tool", ToolCategory.MCP),
    ],
)
def test_known_names_categorized(tool_name: str, expected: ToolCategory) -> None:
    """표에 있는 표준 도구 이름은 정확한 카테고리로 분류된다."""
    assert categorize_tool_name(tool_name) == expected


def test_unknown_name_returns_none() -> None:
    """표에 없는 이름은 None — 호출처가 fail-closed로 해석할 수 있게 한다."""
    assert categorize_tool_name("SomePluginTool") is None


def test_pipeline_and_public_function_agree() -> None:
    """파이프라인 _categorize_tool(1단계)이 공개 함수와 같은 결과를 내는지 확인(드리프트 방지)."""
    from core.permission.pipeline import PermissionPipeline
    from core.permission.types import PermissionContext, PermissionMode

    class _NamedTool:
        """이름만 있는 최소 도구 대역 — 1단계(이름 매칭)에서 끝나는 경우만 쓴다."""

        def __init__(self, name: str) -> None:
            self.name = name
            self.is_read_only = False
            self.is_destructive = False

    pipeline = PermissionPipeline(
        context=PermissionContext(
            mode=PermissionMode.DEFAULT, working_directory=".", session_id="t"
        )
    )
    for name in ("Write", "Edit", "MultiEdit", "Bash", "Read"):
        assert pipeline._categorize_tool(_NamedTool(name)) == categorize_tool_name(name)
