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

네 번째는 실모델 검증에서 나왔다 — 요약 프롬프트에 파이썬 repr 이 들어갔다(④).
다섯 번째부터는 코드 리뷰가 찾았다(⑤) — force 경로가 원본을 돌려주는 것, 총량
절단이 경계 직전 구간을 통째로 버리는 것, 성공 압축이 무효 latch 를 안 푸는 것.
"""

from __future__ import annotations

import pytest

from core.message import Message, StreamEvent, StreamEventType
from core.orchestrator.context_manager import ContextManager, _readable_text


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


@pytest.mark.asyncio
async def test_long_messages_keep_head_and_tail():
    """긴 메시지는 앞만 자르지 않는다 — 결론이 끝에 오는 경우가 많다.

    예전에는 이 테스트가 `_SUMMARY_INPUT_BUDGET >= 20_000` 상수 비교만 했다.
    이름은 head/tail 보존인데 `…(중략)…` 로직은 **한 줄도 실행되지 않았다**
    (2026-08-27 리뷰 지적). 실제 프롬프트를 받아 확인한다.
    """
    seen = {}

    class _Capture:
        async def stream(self, **kwargs):
            seen["prompt"] = kwargs["messages"][0].text_content
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="요약")

    mgr = _mgr(model_provider=_Capture(), max_context_tokens=61440, preserve_recent_turns=1)
    # 메시지 하나가 per_msg 예산을 크게 넘도록 만든다.
    messages = [
        Message.user("시작표식 " + "가" * 30_000 + " 끝표식"),
        Message.user("최근 질문"),
        Message.assistant("최근 답변"),
    ]
    await mgr.auto_compact_if_needed(messages, force=True)

    prompt = seen["prompt"]
    assert "시작표식" in prompt, "앞부분이 잘렸다"
    assert "끝표식" in prompt, "끝부분이 잘렸다 — 결론이 사라진다"
    # 메시지 단위 중략과 총량 단위 중략 둘 다 "중략"으로 표기한다. 어느 쪽이
    # 걸렸든 가운데가 접혔다는 사실만 확인하면 된다.
    assert "중략" in prompt, "중략 로직이 실행되지 않았다"


# ── ③ 이득 없는 압축을 미리 피한다 ───────────────────────────


@pytest.mark.asyncio
async def test_nothing_to_drop_skips_summary_entirely(monkeypatch):
    """★-262/-333 재현★ 버릴 것이 없으면 요약을 만들지 않는다.

    보존 대상이 전체인데 요약을 만들어 붙이면 결과가 원본보다 커진다.
    실측 두 건이 정확히 그 형태였다.

    force 는 쓰지 않는다 — 실측 사고 2건이 모두 force=False(임계치 자동 압축)였고,
    force 는 오류 복구 경로라 반대로 **반드시 줄여야** 한다
    (test_force_never_returns_the_original_list 가 그쪽을 본다).
    """
    mgr = _mgr(preserve_recent_turns=10)  # 보존 요구가 대화보다 크다
    calls = {"n": 0}

    async def _summary(_messages):
        calls["n"] += 1
        return "요" * 3000

    monkeypatch.setattr(mgr, "_get_model_summary", _summary)

    messages = _long_conversation(2, 300)  # 4개 — 전부 보존 대상
    result = await mgr.auto_compact_if_needed(messages)

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


# ── ④ 요약 프롬프트에 파이썬 repr 이 섞이지 않는다 ─────────────


@pytest.mark.asyncio
async def test_summary_prompt_has_no_python_repr():
    """★실모델 검증에서 발견★ str(msg.content) 는 repr 을 만든다.

    `Message.assistant()` 는 content 를 항상 list[ContentBlock] 로 정규화한다.
    그래서 `str(msg.content)` 가 다음을 만들어 요약 프롬프트에 그대로 들어갔다.

        [TextBlock(type='text', text='...')]

    실제 A.X-4.0 호출에서 모델이 요약 대신 이 구조를 되뱉었고, 앞부분의 파일
    경로·식별자가 요약에 하나도 살아남지 못했다.
    """
    seen = {}

    class _Capture:
        async def stream(self, **kwargs):
            seen["prompt"] = kwargs["messages"][0].text_content
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="요약")

    mgr = _mgr(model_provider=_Capture(), preserve_recent_turns=1)
    messages = [
        Message.user("파일은 backend/services/koje_stats.py 입니다"),
        Message.assistant("확인했습니다. 그 파일을 고치겠습니다"),
        Message.user("최근 질문"),
        Message.assistant("최근 답변"),
    ]
    await mgr.auto_compact_if_needed(messages, force=True)

    prompt = seen["prompt"]
    assert "TextBlock(" not in prompt, "파이썬 repr 이 요약 프롬프트에 들어갔다"
    assert "확인했습니다" in prompt, "assistant 본문이 프롬프트에서 사라졌다"
    assert "backend/services/koje_stats.py" in prompt


def test_readable_text_keeps_tool_names():
    """도구 호출은 이름만 남긴다 — '진행 중인 작업'을 요약이 알아야 한다."""
    msg = Message.assistant(
        text="파일을 읽겠습니다",
        tool_uses=[{"id": "t1", "name": "Read", "input": {"path": "a.py"}}],
    )

    out = _readable_text(msg)

    assert "TextBlock(" not in out
    assert "파일을 읽겠습니다" in out
    assert "Read" in out


def test_readable_text_plain_string_unchanged():
    """content 가 문자열이면 그대로 돌려준다(도구 결과 등)."""
    assert _readable_text(Message.tool_result("t1", "결과 본문")) == "결과 본문"


# ── ⑤ 리뷰가 찾은 것 (2026-08-27) ────────────────────────────


@pytest.mark.asyncio
async def test_force_never_returns_the_original_list(monkeypatch):
    """★H1 재현★ force 경로가 원본을 돌려주면 오류 복구가 영구 실패한다.

    force 호출부(query_loop 의 "prompt is too long" 복구, CLI /compact)는 반환값으로
    리스트를 교체하고 **무조건** mark_result_adopted() 를 부른다. 원본이 돌아오면
    압축은 안 된 채 이전 경계·요약만 지워져, 다음 apply_all 이 경계 이전 원문을
    전부 되살린다. 줄이러 온 자리에서 오히려 늘어난다.

    조건은 흔하다 — 질문 하나에 도구 결과가 길게 이어지면 user 메시지가
    preserve_recent_turns 보다 적어 _extract_recent_turns 가 전체를 돌려준다.
    컨텍스트가 터지는 것도 대개 그 형태다.
    """
    mgr = _mgr(preserve_recent_turns=6)

    async def _summary(_messages):
        return "짧은 요약"

    monkeypatch.setattr(mgr, "_get_model_summary", _summary)

    # user 1개 + 도구 결과 다수 — 턴 경계로는 자를 것이 없다.
    messages = [Message.user("질문 하나")]
    for i in range(20):
        messages.append(Message.tool_result(f"tu{i}", f"결과{i} " + "가" * 2000))

    before = mgr._estimate_tokens(messages)
    result = await mgr.auto_compact_if_needed(messages, force=True)

    assert result is not messages, "force 가 원본 객체를 그대로 돌려줬다"
    assert mgr._estimate_tokens(result) < before, "force 가 줄이지 못했다"
    # 마지막 메시지 원문은 살아 있어야 한다.
    assert "결과19" in " ".join(m.text_content or "" for m in result)


@pytest.mark.asyncio
async def test_auto_path_still_returns_original_when_nothing_to_drop(monkeypatch):
    """자동 경로는 종전대로 원본을 돌려준다(무회귀) — force 만 달라야 한다."""
    mgr = _mgr(preserve_recent_turns=10)
    calls = {"n": 0}

    async def _summary(_messages):
        calls["n"] += 1
        return "요" * 3000

    monkeypatch.setattr(mgr, "_get_model_summary", _summary)

    messages = _long_conversation(2, 300)
    result = await mgr.auto_compact_if_needed(messages)

    assert calls["n"] == 0
    assert result is messages


@pytest.mark.asyncio
async def test_summary_prompt_keeps_the_last_dropped_message():
    """★H2 재현★ 총량 절단이 앞만 남기면 경계 바로 앞 구간이 통째로 빠진다.

    개별 메시지에는 "앞만 자르지 않는다"를 적용해 놓고 합친 뒤에는 앞만 남기면,
    사라지는 것은 이후 턴이 가장 많이 참조할 구간이다. 요약을 버릴 구간으로 옮긴
    수정의 취지가 거기서 절반 무효화된다.
    """
    seen = {}

    class _Capture:
        async def stream(self, **kwargs):
            seen["prompt"] = kwargs["messages"][0].text_content
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="요약")

    mgr = _mgr(model_provider=_Capture(), max_context_tokens=61440, preserve_recent_turns=2)
    # 메시지 수를 충분히 늘려 per_msg 하한(400) × n 이 예산을 넘게 만든다.
    messages = _long_conversation(40, 800)
    await mgr.auto_compact_if_needed(messages, force=True)

    prompt = seen["prompt"]
    # 보존 2턴을 뺀 마지막 버림 대상은 질문37/답변37 이다.
    assert "질문37" in prompt or "답변37" in prompt, (
        "경계 바로 앞 구간이 프롬프트에서 빠졌다 — 앞에서만 잘랐다"
    )
    assert "질문0" in prompt, "앞부분도 남아 있어야 한다"


@pytest.mark.asyncio
async def test_successful_compaction_releases_the_latch(monkeypatch):
    """★M2 재현★ 성공 압축이 latch 를 안 풀면 이후 자동 압축이 계속 막힌다.

    성공 압축은 리스트를 크게 줄이므로 len(messages) 가 옛 latch 값 아래에 오래
    머문다. 그동안 토큰은 도구 결과로 얼마든지 늘 수 있다.
    """
    mgr = _mgr(max_context_tokens=100, preserve_recent_turns=2)

    async def _huge(_messages):
        return "요" * 5000  # 무효 판정을 유도한다

    monkeypatch.setattr(mgr, "_get_model_summary", _huge)
    messages = _long_conversation(10, 200)
    await mgr.auto_compact_if_needed(messages)
    assert mgr._useless_compact_at is not None, "무효 latch 가 안 걸렸다"

    async def _short(_messages):
        return "짧은 요약"

    monkeypatch.setattr(mgr, "_get_model_summary", _short)
    await mgr.auto_compact_if_needed(messages, force=True)

    assert mgr._useless_compact_at is None, "성공 압축이 latch 를 풀지 않았다"


def test_summary_input_budget_follows_the_window():
    """★M3★ 좁은 창에서는 요약 입력도 같이 줄어야 한다.

    고정 24,000자를 쓰면 h200(16,384)·h100(8,192) 프로파일에서 요약 호출 자체가
    창을 넘는다. 그 호출이 일어나는 자리가 컨텍스트 초과에서 회복하려는 자리다.
    """
    assert _mgr(max_context_tokens=61440)._summary_input_budget() == 24_000
    assert _mgr(max_context_tokens=16384)._summary_input_budget() == 8_192
    assert _mgr(max_context_tokens=8192)._summary_input_budget() == 4_096


def test_readable_text_falls_back_to_thinking():
    """★L1★ thinking 만 있는 메시지가 요약 입력에서 통째로 사라지면 안 된다."""
    from core.message import Role, ThinkingBlock

    msg = Message(role=Role.ASSISTANT, content=[ThinkingBlock(thinking="내부 추론 내용")])

    out = _readable_text(msg)

    assert out == "내부 추론 내용"
    assert "ThinkingBlock(" not in out


def test_readable_text_prefers_text_over_thinking():
    """text 가 있으면 thinking 은 쓰지 않는다 — 프롬프트가 두 배로 불면 안 된다."""
    from core.message import Role, TextBlock, ThinkingBlock

    msg = Message(
        role=Role.ASSISTANT,
        content=[ThinkingBlock(thinking="내부 추론"), TextBlock(text="사용자에게 한 말")],
    )

    assert _readable_text(msg) == "사용자에게 한 말"
