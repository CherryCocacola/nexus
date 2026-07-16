# 대용량 문서 보존을 위한 context_manager 절단 경로 2건(팽창 버그·micro_compact 면제) 회귀 테스트.
"""ContextManager 도구 결과 절단 경로 회귀 테스트 (대용량 문서 분석 품질 보호).

배경:
    DocumentProcess가 문서 전문을 통짜로 반환해도, TIER_M/L의 압축 파이프라인이
    다음 턴 apply_all에서 그 결과를 잘라 모델이 전문을 못 보는 문제가 있었다.
    두 경로를 수정했고 이 테스트가 회귀를 막는다.

    1) _apply_tool_result_budget 팽창 버그: 한글 토큰 추정(2자/토큰)과 글자 환산
       (3자/토큰)이 어긋나, "토큰은 예산 초과이나 글자 수는 예산 이하"인 한글 결과를
       잘못 축약하면 head+tail 중복으로 오히려 길어졌다 → 토큰·글자 양쪽 초과일 때만 축약.
    2) _micro_compact 최근성 면제: 100줄 초과 결과를 최근성 무관하게 90줄로 접어,
       방금 읽은 문서 본문까지 접혔다 → 최근 N개 도구 결과는 손실 압축을 건너뛴다.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from core.message import Message
from core.model.hardware_tier import HardwareTier
from core.orchestrator.context_manager import ContextManager


def _role(msg: Message) -> str:
    return msg.role if isinstance(msg.role, str) else msg.role.value


def _lines(content: str) -> int:
    return len(content.split("\n"))


# ─────────────────────────────────────────────
# 1) 팽창 버그 — 한글 결과가 절단으로 오히려 길어지지 않는다
# ─────────────────────────────────────────────


def test_tool_result_budget_korean_content_never_expands() -> None:
    """토큰은 예산 초과이나 글자 수는 예산 이하인 한글 결과는 절단해도 길어지지 않는다.

    budget=200 → char_budget=600. 한글 400자는 토큰 추정 ~800(>200)이라 옛 코드는
    축약을 시도했고, head(300)+tail(150)+마커로 원본(400)보다 길어졌다. 수정 후에는
    글자 수(400)가 char_budget(600) 이하라 축약을 건너뛰어 원본 길이를 유지한다.
    """
    cm = ContextManager(
        model_provider=MagicMock(),
        max_context_tokens=8192,
        tool_result_budget=200,
        preserve_recent_tool_results=1,  # 최근 1개 보존 → 첫 결과가 예산 적용 대상
        tier=HardwareTier.TIER_M,
    )
    korean = "가나다라마바사아자차카타파하" * 30  # 약 400자 한글, 개행 없음
    korean = korean[:400]
    msgs = [
        Message.user("q"),
        Message.tool_result("tu-old", korean),   # 예산 적용 대상(팽창 버그 후보)
        Message.tool_result("tu-recent", "짧음"),  # 최근 1개 보존
    ]
    out = cm._apply_tool_result_budget(msgs)
    old = [m for m in out if m.tool_use_id == "tu-old"][0]
    # 핵심: 절단이 원본보다 결과를 길게 만들면 안 된다(팽창 금지).
    assert len(str(old.content)) <= len(korean)


def test_tool_result_budget_large_content_still_truncates_and_shrinks() -> None:
    """글자·토큰 모두 예산을 크게 넘는 결과는 여전히 축약돼 원본보다 짧아진다(기능 보존)."""
    cm = ContextManager(
        model_provider=MagicMock(),
        max_context_tokens=8192,
        tool_result_budget=200,   # char_budget=600
        preserve_recent_tool_results=1,
        tier=HardwareTier.TIER_M,
    )
    big = "x" * 20_000  # 글자·토큰 모두 예산 초과
    msgs = [
        Message.user("q"),
        Message.tool_result("tu-old", big),
        Message.tool_result("tu-recent", "짧음"),
    ]
    out = cm._apply_tool_result_budget(msgs)
    old = [m for m in out if m.tool_use_id == "tu-old"][0]
    assert len(str(old.content)) < len(big)  # 실제로 줄어야 한다


# ─────────────────────────────────────────────
# 2) micro_compact 최근성 면제 — 최근 결과는 100줄 접기를 건너뛴다
# ─────────────────────────────────────────────


def _many_lines(n: int) -> str:
    """서로 다른 n개의 줄로 이뤄진 텍스트(중복 줄 압축에 걸리지 않게 각 줄을 구별)."""
    return "\n".join(f"line-{i} 내용 문단" for i in range(n))


def test_micro_compact_recent_tool_result_not_folded() -> None:
    """최근 도구 결과(방금 읽은 문서)는 100줄 초과여도 접히지 않고 원형 보존된다."""
    cm = ContextManager(
        model_provider=MagicMock(),
        max_context_tokens=8192,
        preserve_recent_tool_results=1,  # 마지막 1개 보존
        tier=HardwareTier.TIER_M,
    )
    recent_doc = _many_lines(150)  # 100줄 초과
    msgs = [
        Message.user("문서 분석해줘"),
        Message.tool_result("tu-doc", recent_doc),  # 유일=최근 → 보존돼야 함
    ]
    out = cm._micro_compact(msgs)
    doc = [m for m in out if m.tool_use_id == "tu-doc"][0]
    # 접혔다면 "N줄 생략" 마커가 있고 줄 수가 ~90으로 줄었을 것 — 없어야 정상.
    assert "줄 생략" not in str(doc.content)
    assert _lines(str(doc.content)) == 150


def test_micro_compact_old_tool_result_still_folded() -> None:
    """최근 보존 범위 밖의 오래된 도구 결과는 현행대로 100줄 접기가 적용된다(보호 기능 유지)."""
    cm = ContextManager(
        model_provider=MagicMock(),
        max_context_tokens=8192,
        preserve_recent_tool_results=1,  # 최근 1개만 보존
        tier=HardwareTier.TIER_M,
    )
    old_big = _many_lines(150)
    msgs = [
        Message.user("q1"),
        Message.tool_result("tu-old", old_big),     # 접힘 대상(오래됨)
        Message.user("q2"),
        Message.tool_result("tu-recent", "짧은 결과"),  # 최근 1개 보존
    ]
    out = cm._micro_compact(msgs)
    old = [m for m in out if m.tool_use_id == "tu-old"][0]
    # 오래된 결과는 접혀서 줄 수가 크게 줄고 생략 마커가 있어야 한다.
    assert "줄 생략" in str(old.content)
    assert _lines(str(old.content)) < 150


def test_micro_compact_zero_preserve_folds_all() -> None:
    """preserve_recent_tool_results=0이면 면제가 없어 최근 결과도 접힌다([-0:] 함정 회피 확인)."""
    cm = ContextManager(
        model_provider=MagicMock(),
        max_context_tokens=8192,
        preserve_recent_tool_results=0,  # 면제 없음
        tier=HardwareTier.TIER_M,
    )
    msgs = [
        Message.user("q"),
        Message.tool_result("tu-1", _many_lines(150)),
    ]
    out = cm._micro_compact(msgs)
    only = [m for m in out if m.tool_use_id == "tu-1"][0]
    assert "줄 생략" in str(only.content)
