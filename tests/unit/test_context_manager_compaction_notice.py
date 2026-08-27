# ContextManager 압축 표시 훅(take_last_compaction) + query_loop CONTEXT_COMPACT emit 단위 테스트
"""
컨텍스트 압축 "화면 표시" 훅 단위 테스트 (2026-07-09).

검증 대상:
  1. take_last_compaction(): 실제 압축이 일어났을 때만 사람이 읽을 문구를 1회 반환
     하고(consume), no-op 통과에서는 None을 반환한다.
     - 도구 결과 예산 절단(1단계) → "긴 도구 결과 정리 ..."
     - 오래된 턴 스닙(2단계) → "이전 대화 N턴 요약"
     - 모델 요약(4단계, auto_compact force) → "대화 요약 생성(모델 호출)"
     - 긴급 압축(emergency_compact) → "긴급 압축 — 최근 대화만 보존"
     - TIER_S passthrough → 아무 압축도 없으므로 None
  2. query_loop의 _compaction_event(): 압축 발생 시 CONTEXT_COMPACT StreamEvent를
     만들고, no-op일 때는 None(→ query_loop이 yield하지 않음).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from core.message import Message, StreamEvent, StreamEventType
from core.model.hardware_tier import HardwareTier
from core.orchestrator.context_manager import ContextManager
from core.orchestrator.query_loop import _compaction_event


class _FakeProvider:
    """auto_compact의 모델 요약용 최소 mock — text_delta 하나만 흘려보낸다."""

    async def stream(self, **_kwargs) -> AsyncIterator[StreamEvent]:
        # _get_model_summary가 text_delta.text만 모아 요약을 만든다.
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="핵심 요약본")


def _mk(tier: HardwareTier | None = HardwareTier.TIER_M, **kwargs) -> ContextManager:
    """테스트용 ContextManager. 기본은 압축이 실동작하는 TIER_M."""
    params = {
        "model_provider": MagicMock(),
        "max_context_tokens": 8192,
        "tier": tier,
    }
    params.update(kwargs)
    return ContextManager(**params)  # type: ignore[arg-type]


# ─────────────────────────────────────────────
# take_last_compaction: 기본/no-op
# ─────────────────────────────────────────────
def test_take_last_compaction_none_initially() -> None:
    """생성 직후에는 압축 이력이 없으므로 None."""
    cm = _mk()
    assert cm.take_last_compaction() is None


def test_apply_all_noop_leaves_no_compaction_notice() -> None:
    """작은 대화(임계치 미달)는 apply_all이 아무것도 줄이지 않으므로 None."""
    cm = _mk()
    msgs = [Message.user("안녕"), Message.assistant("네 안녕하세요")]
    cm.apply_all(msgs)
    assert cm.take_last_compaction() is None


# ─────────────────────────────────────────────
# 1단계: 도구 결과 예산 절단
# ─────────────────────────────────────────────
def test_tool_result_budget_sets_compaction_notice() -> None:
    """오래된 도구 결과가 예산 초과로 잘리면 '긴 도구 결과 정리' 문구가 남는다."""
    cm = _mk(tool_result_budget=200, preserve_recent_tool_results=1)
    long_content = "x" * 20_000
    msgs = [
        Message.user("hi"),
        Message.assistant("ok"),
        Message.tool_result("tu-old", long_content),  # 예산 적용 대상
        Message.user("again"),
        Message.assistant("ok2"),
        Message.tool_result("tu-recent", long_content),  # 보존
    ]
    cm.apply_all(msgs)
    phrase = cm.take_last_compaction()
    assert phrase is not None
    assert "도구 결과" in phrase


# ─────────────────────────────────────────────
# 2단계: 오래된 턴 스닙
# ─────────────────────────────────────────────
def test_snip_compact_sets_turn_summary_notice() -> None:
    """전체가 스닙 임계치를 넘고 보존 턴보다 많으면 '이전 대화 N턴 요약' 문구."""
    # max_context를 작게 잡아 강제로 스닙을 유발한다(threshold=0.7 → 70토큰).
    cm = _mk(max_context_tokens=100, snip_threshold=0.7, preserve_recent_turns=2)
    # 한글이 섞인 6개 턴 — estimated_tokens가 임계치를 넉넉히 넘도록.
    msgs: list[Message] = []
    for i in range(6):
        msgs.append(Message.user(f"질문 번호 {i} 상세한 배경 설명 텍스트가 여기에 이어집니다"))
        msgs.append(Message.assistant(f"답변 번호 {i} 자세한 응답 내용이 길게 이어지는 문장입니다"))
    cm.apply_all(msgs)
    phrase = cm.take_last_compaction()
    assert phrase is not None
    assert "턴 요약" in phrase


# ─────────────────────────────────────────────
# 4단계: 모델 요약 (auto_compact)
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_auto_compact_force_sets_model_summary_notice() -> None:
    """force=True 모델 요약 성공 시 '대화 요약 생성(모델 호출)' 문구."""
    # preserve_recent_turns 보다 턴이 많아야 **버릴 앞부분**이 생긴다. 그래야
    # 요약이 실제로 만들어지고 알림 문구가 남는다(2026-08-27).
    # 버릴 것이 없으면 요약은 순수 추가라 어떤 경우에도 줄일 수 없으므로,
    # auto_compact 가 요약을 만들지 않고 조기 반환한다.
    # 대화가 요약보다 커야 한다. 짧으면 요약이 순수 추가가 되어 결과가 커지고,
    # force 라도 그 결과는 채택하지 않는다(2026-08-27) — 그러면 표시 문구도 없다.
    cm = _mk(model_provider=_FakeProvider(), preserve_recent_turns=1)
    msgs = []
    for i in range(4):
        msgs.append(Message.user(f"질문{i} " + "가" * 500))
        msgs.append(Message.assistant(f"답변{i} " + "나" * 500))
    await cm.auto_compact_if_needed(msgs, force=True)
    phrase = cm.take_last_compaction()
    assert phrase is not None
    assert "모델 호출" in phrase


# ─────────────────────────────────────────────
# 긴급 압축
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_emergency_compact_sets_notice() -> None:
    """긴급 압축(TIER_M)은 항상 축소하므로 '긴급 압축' 문구가 남는다."""
    cm = _mk()
    msgs = [
        Message.user("첫 질문"),
        Message.assistant("첫 답변"),
        Message.user("둘째 질문"),
        Message.assistant("둘째 답변"),
    ]
    await cm.emergency_compact(msgs)
    phrase = cm.take_last_compaction()
    assert phrase is not None
    assert "긴급 압축" in phrase


# ─────────────────────────────────────────────
# consume(꺼내면 지움) 보장
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_take_is_consumed_once() -> None:
    """take_last_compaction은 한 번 꺼내면 지운다 → 두 번째 호출은 None."""
    cm = _mk()
    await cm.emergency_compact([Message.user("q"), Message.assistant("a")])
    assert cm.take_last_compaction() is not None
    assert cm.take_last_compaction() is None


# ─────────────────────────────────────────────
# TIER_S passthrough → 표시 없음
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_tier_s_passthrough_emits_no_notice() -> None:
    """TIER_S는 실제 압축을 하지 않으므로 어떤 경로에서도 문구가 없다."""
    cm = _mk(tier=HardwareTier.TIER_S)
    long_content = "x" * 20_000
    cm.apply_all([Message.user("hi"), Message.tool_result("tu", long_content)])
    assert cm.take_last_compaction() is None
    await cm.auto_compact_if_needed([Message.user("q")], force=True)
    assert cm.take_last_compaction() is None


# ─────────────────────────────────────────────
# query_loop 헬퍼: _compaction_event
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_compaction_event_built_when_compacted() -> None:
    """압축이 실제로 일어났으면 _compaction_event가 CONTEXT_COMPACT를 만든다."""
    cm = _mk()
    await cm.emergency_compact([Message.user("q"), Message.assistant("a")])
    ev = _compaction_event(cm)
    assert ev is not None
    etype = ev.type if isinstance(ev.type, str) else ev.type.value
    assert etype == StreamEventType.CONTEXT_COMPACT.value
    assert ev.message and "긴급 압축" in ev.message


def test_compaction_event_none_on_noop() -> None:
    """no-op(압축 없음)이면 _compaction_event는 None → query_loop이 yield 안 함."""
    cm = _mk()
    cm.apply_all([Message.user("안녕"), Message.assistant("네")])
    assert _compaction_event(cm) is None


def test_compaction_event_none_for_missing_hook() -> None:
    """take_last_compaction 훅이 없는 객체(구버전/모의)면 조용히 None."""
    assert _compaction_event(object()) is None
    assert _compaction_event(None) is None
