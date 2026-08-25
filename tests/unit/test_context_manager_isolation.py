# 압축 상태가 요청 간에 새지 않는지, 이득 없는 압축이 상태를 남기지 않는지 검증한다.
"""
2026-08-25 장애 재현.

    웹은 요청마다 세션 전용 엔진을 조립하면서 ContextManager 는 기동 시 만든
    **하나를 공유**했다. 그런데 이 객체에는 `_compact_boundary`(경계 이전 메시지는
    버린다)와 `_compact_summary`(요약을 맨 앞에 붙인다)라는 대화별 상태가 있다.

    어떤 요청 하나가 auto-compact 를 유발하면 그 경계가 인스턴스에 남고, 이후
    **모든 요청**이 `messages[_compact_boundary:]` 로 잘렸다. 플러그인 요청은
    메시지가 1~2개뿐이라 통째로 사라졌고, 모델에는 시스템 프롬프트만 남아
    prompt_tokens 가 입력 크기와 무관한 고정값이 됐다.

        08-24 03:07 압축 → 이후 +138 토큰 고정          (요약만 남은 경우)
        08-25 08:24 압축 → 이후 5,504·4,688 고정        (입력 유실)
        08-20      압축 → 이후 4,674 고정               (같은 형태)

    값이 그때 남은 경계·요약에 따라 달라져 클라이언트가 상수로 막을 수 없었고,
    재기동하면 사라져 원인 추적이 오래 걸렸다.

두 번째 결함도 함께 고정한다 — 상태를 **검증 전에** 커밋하고 있었다.

        08-24 03:07  60,953 → 61,215  (절약 -262)
        08-25 08:22  60,976 → 61,309  (절약 -333)

    줄지 않은 압축은 이득이 없으면서 경계만 남긴다. 그 경계가 곧 입력 유실이 된다.
"""

from __future__ import annotations

import pytest

from core.message import Message
from core.orchestrator.context_manager import ContextManager


def _mgr(**kw) -> ContextManager:
    """TIER_L(파이프라인 활성) 관리자. 모델 호출은 하지 않는 테스트용."""
    defaults = dict(
        model_provider=None,
        max_context_tokens=8000,
        tool_result_budget=2048,
        preserve_recent_turns=1,
        preserve_recent_tool_results=2,
        tier="large",
    )
    defaults.update(kw)
    return ContextManager(**defaults)


# ── 1. 세션 격리 ─────────────────────────────────────────────


def test_for_session_does_not_inherit_compaction_state():
    """★장애 재현★ 공유 인스턴스의 압축 상태가 새 세션으로 새면 안 된다."""
    shared = _mgr()
    # 어떤 요청 하나가 압축을 유발한 상황을 만든다.
    shared._compact_boundary = 20
    shared._compact_summary = "이전 대화 요약"

    fresh = shared.for_session()

    assert fresh._compact_boundary == 0, "경계가 새 세션으로 샜다 — 입력이 잘린다"
    assert fresh._compact_summary is None, "요약이 새 세션으로 샜다 — 고정 토큰이 붙는다"


def test_for_session_preserves_configuration():
    """설정은 그대로 와야 한다 — 호출부가 인자를 다시 조립하면 두 벌이 된다."""
    shared = _mgr(max_context_tokens=65536, tool_result_budget=8000,
                  preserve_recent_turns=6, preserve_recent_tool_results=3)
    fresh = shared.for_session()

    assert fresh.max_tokens == 65536
    assert fresh.tool_result_budget == 8000
    assert fresh.preserve_recent_turns == 6
    assert fresh.preserve_recent_tool_results == 3
    assert fresh.snip_threshold == shared.snip_threshold
    assert fresh.auto_compact_threshold == shared.auto_compact_threshold


def test_for_session_keeps_passthrough_tier():
    """TIER_S 의 pass-through 성질도 따라와야 한다.

    tier 는 반드시 HardwareTier enum 으로 넘긴다 — 구현이 `.name`/`.value` 로만
    티어명을 뽑으므로 평문 문자열("small")은 잡히지 않는다(별개 사안, 여기서는
    bootstrap 과 같은 방식으로 넘겨 검증한다).
    """
    from core.model.hardware_tier import HardwareTier

    assert _mgr(tier=HardwareTier.TIER_S).for_session().passthrough is True
    assert _mgr(tier=HardwareTier.TIER_L).for_session().passthrough is False


def test_leaked_boundary_would_drop_a_short_request():
    """경계가 남아 있으면 짧은 요청이 통째로 사라진다는 것을 명시적으로 고정한다.

    이 테스트는 '고치기 전 동작'이 왜 치명적이었는지를 기록한다. 플러그인 요청은
    메시지가 1~2개뿐이라, 경계 20 이 남아 있으면 남는 메시지가 0개가 된다.
    """
    leaked = _mgr()
    leaked._compact_boundary = 20
    one_turn = [Message.user("프로젝트 구조를 알려줘")]

    assert one_turn[leaked._compact_boundary :] == [], "잘려서 빈 목록이 된다"
    # 세션별 인스턴스에서는 그대로 남는다.
    assert one_turn[leaked.for_session()._compact_boundary :] == one_turn


# ── 2. 이득 없는 압축은 상태를 남기지 않는다 ────────────────


@pytest.mark.asyncio
async def test_useless_auto_compaction_commits_no_state(monkeypatch):
    """★실측★ 임계치 자동 압축이 이득 없으면 원본 유지, 경계도 남기지 않는다.

    실측 사고 2건(08-24 -262, 08-25 -333)은 모두 `force=False` 였다. 그래서
    가드도 이 경로에만 건다 — force=True 는 에러 복구용이라 원본을 돌려주면
    호출자가 같은 컨텍스트 초과로 다시 실패한다.
    """
    mgr = _mgr(max_context_tokens=100)

    # 요약이 원본보다 길어지는 상황을 만든다(실측: 60,953 → 61,215).
    async def _huge_summary(_messages):
        return "요" * 5000

    monkeypatch.setattr(mgr, "_get_model_summary", _huge_summary)

    # 임계치를 넘겨 자동 압축이 돌게 하되 force 는 쓰지 않는다.
    messages = [Message.user("질문 " * 200), Message.assistant("답 " * 200)]
    result = await mgr.auto_compact_if_needed(messages, force=False)

    assert result == messages, "이득 없는 압축인데 결과를 바꿨다"
    assert mgr._compact_boundary == 0, "이득 없는 압축인데 경계를 남겼다"
    assert mgr._compact_summary is None, "이득 없는 압축인데 요약을 남겼다"


@pytest.mark.asyncio
async def test_effective_compaction_still_commits(monkeypatch):
    """실제로 줄어드는 압축은 종전대로 동작해야 한다(무회귀)."""
    mgr = _mgr(max_context_tokens=100)

    async def _short_summary(_messages):
        return "요약"

    monkeypatch.setattr(mgr, "_get_model_summary", _short_summary)

    # 앞쪽에 긴 대화를 쌓고 **마지막 턴만 짧게** 둔다. preserve_recent_turns=1 이라
    # 보존되는 것은 짧은 마지막 턴뿐이라, 요약+보존이 원본보다 확실히 작아진다.
    messages: list[Message] = []
    for _ in range(10):
        messages.append(Message.user("긴 질문 " * 300))
        messages.append(Message.assistant("긴 답 " * 300))
    messages.append(Message.user("짧은 질문"))
    messages.append(Message.assistant("짧은 답"))

    result = await mgr.auto_compact_if_needed(messages, force=True)

    assert len(result) < len(messages), "압축이 실제로 줄이지 못했다"
    assert mgr._compact_summary == "요약"
    assert mgr._total_compactions == 1
    assert mgr._compact_boundary > 0
