# 사후 검증 통합 진입점과 CLI 배선을 고정하는 테스트.
"""
`core/verification/post_check.py` 의 계약과, **CLI 에도 배선돼 있는지**를 고정한다.

[왜 생겼나 — 2026-08-07]
    숫자 인용 검증과 실행 주장 검증을 만들었는데 둘 다 `web/app.py` 에만 배선돼
    있었다. 그런데 정작 문제가 관측된 곳은 CLI 였다.

        · 테스트 기대값 암산 오류(5415 → 5115)       — CLI
        · "예상 출력 결과: 모든 테스트가 통과했습니다"  — CLI
        · Write 로 저장한 파일 손상                   — CLI

    보호가 필요한 표면에 보호가 없었다. 이 테스트는 그 배선이 다시 빠지는 것을 막는다.
"""

from __future__ import annotations

from types import SimpleNamespace

from core.verification.post_check import (
    build_answer_warnings,
    collect_tool_result_texts,
)


def _msg(role: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=content)


# ─────────────────────────────────────────────
# 근거 자료 수집
# ─────────────────────────────────────────────


def test_collects_only_tool_results():
    messages = [
        _msg("user", "질문"),
        _msg("assistant", "답변"),
        _msg("tool_result", "18764 * 27 = 506,628"),
    ]

    assert collect_tool_result_texts(messages) == ["18764 * 27 = 506,628"]


def test_handles_enum_like_roles():
    """role 이 Enum 이어도 값으로 비교한다(웹·CLI 가 서로 다른 형태를 넘긴다)."""
    messages = [SimpleNamespace(role=SimpleNamespace(value="tool_result"), content="x")]

    assert collect_tool_result_texts(messages) == ["x"]


# ─────────────────────────────────────────────
# 통합 진입점 — 두 검증이 한 번에 걸린다
# ─────────────────────────────────────────────


def test_runs_number_check():
    """도구 결과와 자릿수가 어긋난 숫자를 잡는다."""
    warning = build_answer_warnings(
        "18,764 곱하기 27은 500,662입니다.",
        [_msg("tool_result", "18764 * 27 = 506,628")],
    )

    assert "숫자 확인 필요" in warning
    assert "506,628" in warning


def test_runs_execution_check():
    """도구를 하나도 안 썼는데 통과했다고 단정하면 잡는다."""
    warning = build_answer_warnings("모든 테스트가 통과했습니다.", [])

    assert "실행 확인 필요" in warning


def test_clean_answer_gets_nothing():
    assert build_answer_warnings("안녕하세요! 무엇을 도와드릴까요?", []) == ""


def test_never_raises_on_bad_input():
    """검증 실패가 응답을 막으면 안 된다 — 통째로 fail-soft."""
    assert build_answer_warnings("답변", [object()]) == ""  # role 속성이 없는 객체


# ─────────────────────────────────────────────
# ★배선 — 웹뿐 아니라 CLI 두 표면 모두에 걸려 있어야 한다
# ─────────────────────────────────────────────


def _source_of(rel_path: str) -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    return (root / rel_path).read_text(encoding="utf-8")


def test_wired_into_cli_ask():
    """`nexus ask`(비대화형)에 배선돼 있어야 한다 — 실측 사례가 나온 표면이다."""
    src = _source_of("cli/commands.py")

    assert "build_answer_warnings" in src


def test_wired_into_cli_repl():
    """대화형 REPL 에도 배선돼 있어야 한다."""
    src = _source_of("cli/repl.py")

    assert "build_answer_warnings" in src


def test_wired_into_web():
    """웹은 4개 응답 경로가 모두 같은 진입점을 쓴다(경로마다 빠뜨리지 않도록)."""
    src = _source_of("web/app.py")

    assert src.count("_answer_warnings_for(") >= 5  # 정의 1 + 호출 4


def test_web_no_longer_has_duplicate_helpers():
    """검증기를 늘릴 때 호출부를 다시 손대지 않도록 진입점을 하나로 유지한다."""
    src = _source_of("web/app.py")

    assert "_number_warning_for" not in src
    assert "_execution_warning_for" not in src
