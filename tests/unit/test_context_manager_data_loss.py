# 압축이 사용자 본문을 스스로 버리거나, 채택된 경계가 다음 턴을 또 자르지 않는지 검증한다.
"""
2026-08-26 코드 리뷰에서 나온 두 결함을 고정한다. 둘 다 08-25 의 세션 격리 수정
(`4c53966`)으로는 해소되지 않는 **별개 경로**였다. 격리는 요청 간 누수를 막았고,
아래 둘은 요청 하나 안에서 벌어진다.

■ C1 — 사용자 턴이 preserve_recent_turns 보다 적으면 현재 질문까지 사라진다

    `_extract_recent_turns` 는 user 메시지가 n 개 미만이면 **빈 리스트**를 돌려줬다.
    주석에 "호출 측이 이를 감안"이라 적혀 있었지만 감안하는 호출부가 하나도 없었다.

        recent = []  →  result = [요약] 하나뿐  →  boundary = len(messages)

    운영은 preserve_recent_turns=6 인데 플러그인 요청은 user 메시지가 1~2개다.
    즉 임계치를 넘는 큰 요청은 **자기 본문을 버리고 요약만 모델에 보냈다.** 그 요약도
    최근 20개 메시지를 각 200자로 잘라 만든 것이라, 8만 자 입력의 요약은 사실상
    앞부분 몇 천 자의 요약이다. 실측 로그에서 6~8만 토큰이 234~1,061 토큰으로
    떨어진 항목들이 이 경로로 보인다.

    새 가드(결과가 작아졌을 때만 커밋)는 이걸 못 막는다 — 요약만 남았으니 "작아진
    것"은 사실이기 때문이다.

■ C2 — 채택된 경계가 다음 턴을 한 번 더 자른다

    `_compact_boundary` 는 **교체 전** 리스트의 인덱스인데, 세 호출부가 압축 결과로
    리스트를 교체한다. 다음 턴 `apply_all` 이 짧아진 리스트에 옛 인덱스를 다시
    적용하고, `_compact_summary` 도 남아 있어 요약이 두 번 붙는다.
"""

from __future__ import annotations

import pytest

from core.message import Message
from core.orchestrator.context_manager import ContextManager


def _mgr(**kw) -> ContextManager:
    """운영과 같은 보존 설정(preserve_recent_turns=6)을 기본으로 쓴다."""
    defaults = dict(
        model_provider=None,
        max_context_tokens=8000,
        tool_result_budget=2048,
        preserve_recent_turns=6,
        preserve_recent_tool_results=3,
        tier="large",
    )
    defaults.update(kw)
    return ContextManager(**defaults)


# ── C1 ───────────────────────────────────────────────────────


def test_fewer_turns_than_requested_preserves_everything():
    """★핵심★ 턴이 요청보다 적으면 0개가 아니라 있는 것 전부를 남긴다."""
    mgr = _mgr()
    messages = [Message.user("질문 하나뿐")]

    assert mgr._extract_recent_turns(messages, 6) == messages


def test_no_user_message_still_preserves():
    """user 를 못 찾아도 버리지 않는다 — 턴 경계를 모를 뿐이다."""
    mgr = _mgr()
    messages = [Message.system("시스템"), Message.assistant("답변")]

    assert mgr._extract_recent_turns(messages, 6) == messages


def test_enough_turns_still_slices_normally():
    """턴이 충분하면 종전대로 마지막 n턴만 남긴다(무회귀)."""
    mgr = _mgr()
    messages: list[Message] = []
    for i in range(10):
        messages.append(Message.user(f"질문{i}"))
        messages.append(Message.assistant(f"답변{i}"))

    recent = mgr._extract_recent_turns(messages, 2)

    assert len(recent) == 4  # user/assistant 2턴
    assert recent[0].text_content == "질문8"


@pytest.mark.asyncio
async def test_single_large_message_is_not_discarded(monkeypatch):
    """★실측 재현★ 큰 단일 메시지가 자기 본문을 버리면 안 된다.

    이 케이스가 정확히 사고 형태다 — 사용자 메시지 1개, 임계치 초과,
    preserve_recent_turns=6.
    """
    mgr = _mgr(max_context_tokens=100)

    async def _summary(_messages):
        return "요약"

    monkeypatch.setattr(mgr, "_get_model_summary", _summary)

    original = Message.user("긴 질문 " * 3000)
    result = await mgr.auto_compact_if_needed([original], force=True)

    texts = [m.text_content for m in result]
    assert any("긴 질문" in (t or "") for t in texts), "사용자 본문이 통째로 사라졌다"
    assert mgr._compact_boundary < 1, "경계가 전체를 가리켜 다음 턴이 통째로 잘린다"


# ── C2 ───────────────────────────────────────────────────────


def test_mark_result_adopted_clears_state():
    """결과를 채택한 호출부가 경계를 되돌릴 수단을 가져야 한다."""
    mgr = _mgr()
    mgr._compact_boundary = 40
    mgr._compact_summary = "이전 요약"

    mgr.mark_result_adopted()

    assert mgr._compact_boundary == 0
    assert mgr._compact_summary is None


def test_stale_boundary_would_truncate_the_adopted_list():
    """왜 되돌려야 하는지를 명시적으로 기록한다.

    경계 40 을 남긴 채 2개짜리 리스트로 교체하면, 다음 턴 슬라이스가 빈 목록이 된다.
    """
    mgr = _mgr()
    mgr._compact_boundary = 40
    adopted = [Message.system("[대화 요약]…"), Message.user("현재 질문")]

    assert adopted[mgr._compact_boundary :] == [], "옛 경계가 채택 리스트를 통째로 자른다"

    mgr.mark_result_adopted()
    assert adopted[mgr._compact_boundary :] == adopted


def test_adopted_summary_is_not_prepended_twice():
    """요약은 채택 리스트 0번에 이미 있다 — 상태에 남기면 두 번 붙는다."""
    mgr = _mgr()
    mgr._compact_summary = "이전 요약"
    adopted = [Message.system("[대화 요약]\n이전 요약\n[요약 끝]"), Message.user("현재 질문")]

    mgr.mark_result_adopted()
    applied = mgr.apply_all(adopted)

    summaries = [m for m in applied if "[대화 요약]" in (m.text_content or "")]
    assert len(summaries) == 1, "요약이 중복으로 붙었다"
