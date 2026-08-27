# 격리 배선·OOM 축소 복원·무효 압축 반복 방지를 검증한다.
"""
2026-08-26~27. 리뷰가 지적한 세 가지를 고정한다.

■ ① 격리 배선 테스트가 없었다 — 가장 큰 구멍

    `test_context_manager_isolation.py` 는 `for_session()` 만 검증하고 **실제 장애
    지점인 `web/app.py` 의 배선은 건드리지 않는다.** 그래서 그 줄을 공유 인스턴스로
    되돌려도 기존 테스트가 전부 통과한다. 장애를 막는 것은 함수가 아니라 배선이다.

    실전 검증도 기대할 수 없다 — 배포 후 9,978건 요청 동안 auto-compact 이 0건이다.
    플러그인 요청이 10~20K 토큰이라 임계치 55,296 을 못 넘는다. 자연 발생을
    기다리는 대신 배선을 결정적으로 고정한다.

■ ② GPU OOM 축소에 복원이 없었다

    `query_loop` 의 OOM 복구가 `max_tokens *= 0.7` 을 하는데 되돌리는 코드가 없다.
    웹은 요청마다 새 인스턴스라 자연히 초기화되지만, CLI 는 관리자가 프로세스
    수명 내내 살아 0.7ⁿ 로 누적 축소됐다.

■ ③ 무효 압축이 매 턴 반복됐다

    자동 압축이 "줄지 않았다"로 끝나면 토큰 수는 임계치 위에 그대로 남는다.
    query_loop 은 매 턴 압축을 다시 시도하므로, 대화가 끝날 때까지 턴마다 모델
    요약 왕복이 한 번씩 버려진다.
"""

from __future__ import annotations

import pytest

from core.message import Message
from core.model.hardware_tier import HardwareTier
from core.orchestrator.context_manager import ContextManager


def _mgr(**kw) -> ContextManager:
    defaults = dict(
        model_provider=None,
        max_context_tokens=8000,
        preserve_recent_turns=6,
        tier="large",
    )
    defaults.update(kw)
    return ContextManager(**defaults)


def _conversation_with_droppable_head() -> list[Message]:
    """버릴 앞부분이 확실히 생기는 대화(preserve_recent_turns=6 보다 턴이 많다)."""
    out: list[Message] = []
    for i in range(10):
        out.append(Message.user(f"질문{i} " + "가" * 200))
        out.append(Message.assistant(f"답변{i} " + "나" * 200))
    return out


# ── ① 웹 배선 ────────────────────────────────────────────────


def test_session_engines_never_share_a_context_manager():
    """★핵심★ `_assemble_session_engine` 이 요청마다 다른 인스턴스를 줘야 한다.

    이 테스트가 없으면 `web/app.py` 의 그 한 줄을 공유로 되돌려도 아무도 모른다.
    장애의 실제 지점이 함수가 아니라 **배선**이기 때문이다.
    """
    from web.app import _assemble_session_engine

    shared = _mgr()
    parts = _fake_parts(shared)

    engine_a, _ = _assemble_session_engine(parts, "session-a", tenant=None)
    engine_b, _ = _assemble_session_engine(parts, "session-b", tenant=None)

    cm_a = engine_a._context_manager
    cm_b = engine_b._context_manager

    assert cm_a is not shared, "공유 인스턴스가 그대로 넘어갔다 — 압축 상태가 샌다"
    assert cm_b is not shared
    assert cm_a is not cm_b, "두 세션이 같은 인스턴스를 쓴다"


def test_compaction_state_does_not_leak_between_session_engines():
    """한 세션에서 압축이 일어나도 다음 세션이 잘리지 않아야 한다.

    실측 장애가 정확히 이 형태였다 — 어떤 요청 하나가 auto-compact 을 유발하면
    그 경계가 남아 이후 모든 요청의 메시지를 잘랐다.
    """
    from web.app import _assemble_session_engine

    shared = _mgr()
    parts = _fake_parts(shared)

    engine_a, _ = _assemble_session_engine(parts, "session-a", tenant=None)
    engine_a._context_manager._compact_boundary = 20  # 압축이 일어난 상황
    engine_a._context_manager._compact_summary = "이전 요약"

    engine_b, _ = _assemble_session_engine(parts, "session-b", tenant=None)

    assert engine_b._context_manager._compact_boundary == 0
    assert engine_b._context_manager._compact_summary is None
    # 원본도 오염되지 않아야 한다 — 다음 clone 의 기준값이기 때문이다.
    assert shared._compact_boundary == 0


def _fake_parts(cm: ContextManager) -> dict:
    """`_assemble_session_engine` 이 읽는 부품 묶음.

    키가 하나라도 빠지면 KeyError 로 죽는다 — 실제 소비 목록에 맞춰 둔다
    (`grep 'parts\\["' web/app.py`).
    """
    return {
        # tier 는 dispatcher 가 `.value` 를 읽으므로 enum 이어야 한다(bootstrap 과 동일).
        "tier": HardwareTier.TIER_L,
        "worker_provider": object(),
        "coder_provider": None,
        "scout_provider": None,
        "scout_tools": [],
        "web_tools": [],
        "context_manager": cm,
        "memory_manager": None,
        "knowledge_retriever": None,
        "system_prompt": "S",
        "routing_config": None,
        "budgets": None,
        "base_options": {},
        "permission_mode": "default",
        "base_cwd": ".",
        "pe_enabled": False,
        "sessions_dir": ".",
    }


# ── ② OOM 축소 복원 ──────────────────────────────────────────


def test_restore_returns_to_configured_value():
    """OOM 으로 줄인 상한이 설정값으로 되돌아와야 한다."""
    mgr = _mgr(max_context_tokens=61440)
    mgr.max_tokens = int(mgr.max_tokens * 0.7)  # OOM 복구가 하는 일
    assert mgr.max_tokens == 43008

    mgr.restore_max_tokens()

    assert mgr.max_tokens == 61440


def test_repeated_reduction_does_not_accumulate_across_requests():
    """★CLI 누적 재현★ 요청 경계에서 복원하지 않으면 0.7ⁿ 로 작아진다."""
    mgr = _mgr(max_context_tokens=61440)

    for _ in range(3):
        mgr.restore_max_tokens()          # query_loop 이 요청 시작에 부른다
        mgr.max_tokens = int(mgr.max_tokens * 0.7)  # 그 요청에서 OOM 발생

    # 복원이 없었다면 61440 * 0.7^3 = 21073 이 됐을 것이다.
    assert mgr.max_tokens == 43008, "축소가 요청을 넘어 누적됐다"


def test_for_session_clone_starts_from_configured_value():
    """줄어든 상태에서 clone 해도 새 세션은 설정값에서 시작한다."""
    mgr = _mgr(max_context_tokens=61440)
    mgr.max_tokens = 1000

    assert mgr.for_session().max_tokens == 61440


# ── ③ 무효 압축 latch ────────────────────────────────────────


@pytest.mark.asyncio
async def test_useless_compaction_is_not_retried_every_turn(monkeypatch):
    """★낭비 재현★ 무효로 끝난 뒤 대화가 그대로면 모델 요약을 다시 부르지 않는다."""
    mgr = _mgr(max_context_tokens=100)
    calls = {"n": 0}

    async def _huge_summary(_messages):
        calls["n"] += 1
        return "요" * 5000  # 원본보다 커서 무효 처리된다

    monkeypatch.setattr(mgr, "_get_model_summary", _huge_summary)
    # 버릴 구간이 있어야 latch 경로를 탄다. 보존 대상이 전체면 요약을 아예 부르지
    # 않고 조기 반환한다(그쪽은 test_compaction_quality.py 가 따로 본다).
    messages = _conversation_with_droppable_head()

    await mgr.auto_compact_if_needed(messages)   # 1회차 — 무효 판정
    await mgr.auto_compact_if_needed(messages)   # 2회차 — 건너뛰어야 한다
    await mgr.auto_compact_if_needed(messages)   # 3회차

    assert calls["n"] == 1, f"요약 왕복이 {calls['n']}회 돌았다 — latch 가 안 걸렸다"


@pytest.mark.asyncio
async def test_latch_releases_when_conversation_grows(monkeypatch):
    """대화가 늘면 요약 재료가 달라지므로 다시 시도해야 한다."""
    mgr = _mgr(max_context_tokens=100)
    calls = {"n": 0}

    async def _huge_summary(_messages):
        calls["n"] += 1
        return "요" * 5000

    monkeypatch.setattr(mgr, "_get_model_summary", _huge_summary)
    # 버릴 구간이 있어야 latch 경로를 탄다. 보존 대상이 전체면 요약을 아예 부르지
    # 않고 조기 반환한다(그쪽은 test_compaction_quality.py 가 따로 본다).
    messages = _conversation_with_droppable_head()

    await mgr.auto_compact_if_needed(messages)
    messages = [*messages, Message.user("새 질문 " * 200), Message.assistant("새 답 " * 200)]
    await mgr.auto_compact_if_needed(messages)

    assert calls["n"] == 2, "대화가 늘었는데도 건너뛰었다"


@pytest.mark.asyncio
async def test_force_ignores_the_latch(monkeypatch):
    """강제 압축은 에러 복구 경로다 — latch 가 막으면 복구가 죽는다."""
    mgr = _mgr(max_context_tokens=100)
    calls = {"n": 0}

    async def _summary(_messages):
        calls["n"] += 1
        return "요" * 5000

    monkeypatch.setattr(mgr, "_get_model_summary", _summary)
    messages = _conversation_with_droppable_head()

    await mgr.auto_compact_if_needed(messages)
    await mgr.auto_compact_if_needed(messages, force=True)

    assert calls["n"] == 2, "force 가 latch 에 막혔다"
