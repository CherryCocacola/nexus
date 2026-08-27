# 요약이 '버려지는 부분'을 보고 만들어지는지, 이득 없는 압축을 미리 피하는지 검증한다.
"""
2026-08-27. 압축 로직 품질을 고정한다.

관측된 6회 중 정상 동작이 한 번도 없었다.

    08-19  64,077 → 234      (6만 토큰 소멸)
    08-20  61,212 → 360
    08-21  87,878 → 1,061
    08-24  60,953 → 61,215   (절약 -262, 오히려 증가)
    08-25  60,976 → 61,309   (절약 -333)
    08-25  55,376 → 465

원인은 둘이었다.

■ ① 요약이 **보존될 부분**을 요약했다

    압축은 앞쪽을 요약으로 바꾸고 최근 턴은 원본으로 지킨다. 그런데 요약 입력이
    `messages[-20:]` 이라 **어차피 원본으로 남는 구간과 겹쳤다.** 정작 사라지는
    앞부분에 대해서는 요약이 아무 말도 하지 않았다.

■ ② 요약 입력이 대화의 3% 였다

    압축은 55,296 토큰(≈13만 자 이상)에서 걸리는데 입력 상한이 20×200=4,000자였다.
    6~8만 토큰이 234~1,061 토큰으로 떨어진 것은 "요약이 성기다"가 아니라 **요약이
    볼 수 있는 내용 자체가 없었다**는 뜻이다.

그리고 -262/-333 은 세 번째 형태다 — 보존 대상이 전체라 버릴 것이 없는데도 요약을
만들어 붙여 오히려 커졌다.
"""

from __future__ import annotations

import pytest

from core.message import Message
from core.orchestrator.context_manager import ContextManager


def _mgr(**kw) -> ContextManager:
    defaults = dict(
        model_provider=None,
        max_context_tokens=100,
        preserve_recent_turns=2,
        tier="large",
    )
    defaults.update(kw)
    return ContextManager(**defaults)


def _long_conversation(turns: int, chars: int) -> list[Message]:
    out: list[Message] = []
    for i in range(turns):
        out.append(Message.user(f"질문{i} " + "가" * chars))
        out.append(Message.assistant(f"답변{i} " + "나" * chars))
    return out


# ── ① 요약 대상 ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summary_sees_the_dropped_part_not_the_kept_part(monkeypatch):
    """★핵심★ 요약 입력은 사라지는 앞부분이어야 한다.

    보존되는 최근 턴을 요약해 봐야 아무 정보도 보태지 않는다. 오히려 사라지는
    앞부분의 결정·요구사항이 통째로 유실된다.
    """
    mgr = _mgr(preserve_recent_turns=2)
    seen: dict = {}

    async def _capture(messages):
        seen["texts"] = [m.text_content for m in messages]
        return "요약"

    monkeypatch.setattr(mgr, "_get_model_summary", _capture)

    messages = _long_conversation(6, 300)  # 12개 메시지
    await mgr.auto_compact_if_needed(messages, force=True)

    joined = " ".join(seen["texts"])
    assert "질문0" in joined, "사라지는 앞부분이 요약 입력에 없다"
    assert "질문5" not in joined, "보존되는 최근 턴이 요약 입력에 섞였다"


@pytest.mark.asyncio
async def test_dropped_and_kept_do_not_overlap(monkeypatch):
    """요약 대상과 보존 구간이 겹치면 안 된다 — 겹치면 요약이 낭비다."""
    mgr = _mgr(preserve_recent_turns=3)
    seen: dict = {}

    async def _capture(messages):
        seen["n"] = len(messages)
        return "요약"

    monkeypatch.setattr(mgr, "_get_model_summary", _capture)

    messages = _long_conversation(8, 200)  # 16개
    result = await mgr.auto_compact_if_needed(messages, force=True)

    kept = len(result) - 1  # 요약 메시지 1개를 뺀 나머지
    assert seen["n"] + kept == len(messages), "요약 대상 + 보존 = 전체 여야 한다"


# ── ② 요약 입력 예산 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_summary_input_is_not_capped_at_4000_chars(monkeypatch):
    """★실측★ 예전 상한(20개×200자=4,000자)으로는 대화의 3% 만 봤다."""
    mgr = _mgr(preserve_recent_turns=1)
    seen: dict = {}

    async def _capture(messages):
        # 실제 프롬프트 조립을 거치도록 원본을 호출하지 않고, 넘어온 양만 잰다.
        seen["chars"] = sum(len(str(m.content)) for m in messages)
        return "요약"

    monkeypatch.setattr(mgr, "_get_model_summary", _capture)

    messages = _long_conversation(20, 1000)  # 약 4만 자
    await mgr.auto_compact_if_needed(messages, force=True)

    assert seen["chars"] > 20_000, f"요약 대상이 {seen['chars']}자뿐이다"


def test_long_messages_keep_head_and_tail():
    """긴 메시지는 앞만 자르지 않는다 — 결론이 끝에 오는 경우가 많다."""
    from core.orchestrator.context_manager import _SUMMARY_INPUT_BUDGET

    assert _SUMMARY_INPUT_BUDGET >= 20_000, "입력 예산이 너무 작다"


# ── ③ 이득 없는 압축을 미리 피한다 ───────────────────────────


@pytest.mark.asyncio
async def test_nothing_to_drop_skips_summary_entirely(monkeypatch):
    """★-262/-333 재현★ 버릴 것이 없으면 요약을 만들지 않는다.

    보존 대상이 전체인데 요약을 만들어 붙이면 결과가 원본보다 커진다.
    실측 두 건이 정확히 그 형태였다.
    """
    mgr = _mgr(preserve_recent_turns=10)  # 보존 요구가 대화보다 크다
    calls = {"n": 0}

    async def _summary(_messages):
        calls["n"] += 1
        return "요" * 3000

    monkeypatch.setattr(mgr, "_get_model_summary", _summary)

    messages = _long_conversation(2, 300)  # 4개 — 전부 보존 대상
    result = await mgr.auto_compact_if_needed(messages, force=True)

    assert calls["n"] == 0, "버릴 게 없는데 요약 모델을 불렀다"
    assert result == messages, "원본을 그대로 돌려줘야 한다"
    assert mgr._compact_boundary == 0
    assert mgr._compact_summary is None


@pytest.mark.asyncio
async def test_effective_compaction_still_works(monkeypatch):
    """정상 압축은 종전대로 동작해야 한다(무회귀)."""
    mgr = _mgr(preserve_recent_turns=2)

    async def _summary(_messages):
        return "짧은 요약"

    monkeypatch.setattr(mgr, "_get_model_summary", _summary)

    messages = _long_conversation(20, 500)
    result = await mgr.auto_compact_if_needed(messages, force=True)

    assert len(result) < len(messages)
    assert "[대화 요약]" in (result[0].text_content or "")
    assert mgr._compact_boundary > 0
    # 마지막 턴의 원문은 반드시 살아 있어야 한다.
    assert "질문19" in " ".join(m.text_content or "" for m in result)
