"""
ContextManager 티어별 전략 단위 테스트 — Ch 6 (2026-04-21).

검증 대상:
  1. TIER_S 주입 → passthrough=True
     - apply_all: messages 변경 없이 반환 (tool_result도 축약하지 않음)
     - auto_compact_if_needed: 임계치 초과해도 요약 안 함
     - emergency_compact: 최근 1턴만 추출 (규칙 기반 요약 건너뜀)
  2. TIER_M/L/None 주입 → passthrough=False
     - 기존 4단계 파이프라인 유지
  3. stats 프로퍼티가 tier/passthrough 필드를 노출
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.message import Message
from core.model.hardware_tier import HardwareTier
from core.orchestrator.context_manager import ContextManager


def _mk_context_manager(tier: HardwareTier | None) -> ContextManager:
    """테스트용 ContextManager — ModelProvider는 mock으로 충분."""
    return ContextManager(
        model_provider=MagicMock(),
        max_context_tokens=8192,
        tier=tier,
    )


# ─────────────────────────────────────────────
# passthrough 동작 (TIER_S)
# ─────────────────────────────────────────────
def test_tier_s_passthrough_flag() -> None:
    cm = _mk_context_manager(HardwareTier.TIER_S)
    assert cm.passthrough is True
    assert cm.stats["tier"] == "TIER_S"
    assert cm.stats["passthrough"] is True


def test_tier_s_apply_all_returns_messages_unchanged() -> None:
    """TIER_S는 apply_all을 통과시켜야 한다 (TurnStateStore가 담당)."""
    cm = _mk_context_manager(HardwareTier.TIER_S)
    # 큰 tool_result — TIER_M이면 예산 초과로 축소될 크기
    long_content = "x" * 20_000
    msgs = [
        Message.user("hi"),
        Message.assistant("ok"),
        Message.tool_result("tu-1", long_content),
    ]
    out = cm.apply_all(msgs)
    # 같은 객체 리스트 — 길이와 마지막 메시지 내용이 그대로
    assert len(out) == 3
    assert len(str(out[2].content)) == len(long_content)


@pytest.mark.asyncio
async def test_tier_s_auto_compact_passthrough_even_when_force_true() -> None:
    """TIER_S에서는 force=True여도 요약을 만들지 않는다."""
    cm = _mk_context_manager(HardwareTier.TIER_S)
    msgs = [Message.user("q"), Message.assistant("a")]
    out = await cm.auto_compact_if_needed(msgs, force=True)
    assert out == msgs
    # 모델 프로바이더 호출 없어야 함
    cm.model_provider.stream.assert_not_called()


@pytest.mark.asyncio
async def test_tier_s_emergency_compact_keeps_last_turn_only() -> None:
    """TIER_S의 긴급 압축은 최근 1턴만 추출하여 반환 (요약 생성 안 함)."""
    cm = _mk_context_manager(HardwareTier.TIER_S)
    msgs = [
        Message.user("첫 질문"),
        Message.assistant("첫 답변"),
        Message.user("둘째 질문"),
        Message.assistant("둘째 답변"),
    ]
    out = await cm.emergency_compact(msgs)
    # 최근 1턴 = 마지막 user + assistant
    assert len(out) == 2
    assert _role(out[0]) == "user"
    assert "둘째" in str(out[0].content)


# ─────────────────────────────────────────────
# 기본 동작 (TIER_M / TIER_L / None)
# ─────────────────────────────────────────────
@pytest.mark.parametrize("tier", [HardwareTier.TIER_M, HardwareTier.TIER_L, None])
def test_non_tier_s_is_not_passthrough(tier: HardwareTier | None) -> None:
    cm = _mk_context_manager(tier)
    assert cm.passthrough is False
    # tier=None이어도 stats는 안전하게 반환
    stats = cm.stats
    assert "tier" in stats
    assert stats["passthrough"] is False


def test_tier_m_apply_all_activates_compression_pipeline() -> None:
    """TIER_M에서는 tool_result 예산이 적용된다 (오래된 결과가 잘린다).

    preserve_recent_tool_results=1 → 최근 1개는 보존, 그 이전 것만 예산 적용.
    """
    cm = ContextManager(
        model_provider=MagicMock(),
        max_context_tokens=8192,
        tool_result_budget=200,
        preserve_recent_tool_results=1,  # 마지막 1개 보존, 이전 것들은 예산 적용
        tier=HardwareTier.TIER_M,
    )
    long_content = "x" * 20_000
    msgs = [
        Message.user("hi"),
        Message.assistant("ok"),
        Message.tool_result("tu-old", long_content),  # 이건 예산 적용 대상
        Message.user("again"),
        Message.assistant("ok2"),
        Message.tool_result("tu-recent", long_content),  # 이건 보존
    ]
    out = cm.apply_all(msgs)
    tool_out = [m for m in out if _role(m) == "tool_result"]
    assert len(tool_out) == 2
    # 첫 tool_result(old)는 축소, 두 번째(recent)는 그대로
    assert len(str(tool_out[0].content)) < len(long_content)
    assert len(str(tool_out[1].content)) == len(long_content)


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def _role(msg: Message) -> str:
    return msg.role if isinstance(msg.role, str) else msg.role.value
