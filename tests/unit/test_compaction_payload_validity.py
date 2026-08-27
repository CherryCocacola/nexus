# 압축 결과가 그대로 API payload 가 되므로, 모델에 보낼 수 있는 형태인지 검증한다.
"""
2026-08-27. 압축 결함이 **두 커밋 연속** 같은 자리에서 났다.

    8fc4eb1  force 경로가 원본을 돌려줘 오류 복구가 영구 실패
    2707325  force 폴백이 짝을 깨 고아 tool_result 를 만듦

둘 다 원인이 같다 — `ContextManager` 의 반환값을 **호출부가 어떻게 쓰는지**를
보지 않았다. 반환 리스트는 `mark_result_adopted()` 를 거쳐 곧바로
`inference._convert_messages()` 로 들어가 API payload 가 된다. 즉 압축은
"메시지를 줄이는 일"이 아니라 **"모델에 보낼 수 있는 메시지 목록을 만드는 일"**이다.

그래서 형태 자체를 고정한다. 어떤 압축 경로를 타든 결과는 다음을 지켜야 한다.

  1. 모든 `role:"tool"` 항목 앞에 짝이 되는 `tool_call_id` 를 낸 assistant 가 있다
  2. payload 가 `role:"tool"` 로 시작하지 않는다
  3. 실제로 줄어든다(force 경로)

이 파일이 있으면 "줄이기는 했는데 보낼 수 없는 목록"이 조용히 통과하지 못한다.
"""

from __future__ import annotations

import pytest

from core.message import Message
from core.model.inference import LocalModelProvider
from core.orchestrator.context_manager import ContextManager


def _mgr(**kw) -> ContextManager:
    defaults = dict(
        model_provider=None,
        max_context_tokens=61440,
        preserve_recent_turns=6,
        tier="large",
    )
    defaults.update(kw)
    return ContextManager(**defaults)


def _payload(messages: list[Message]) -> list[dict]:
    """실제 전송 형태로 변환한다 — 여기까지 가야 짝 깨짐이 드러난다."""
    provider = LocalModelProvider(base_url="http://127.0.0.1:1", model_id="test")
    return provider._convert_messages(messages, "시스템 프롬프트")


def assert_sendable(payload: list[dict]) -> None:
    """OpenAI 규약상 보낼 수 있는 순서인지 확인한다."""
    issued: set[str] = set()
    first_non_system = next(
        (m for m in payload if m.get("role") != "system"), None
    )
    assert first_non_system is None or first_non_system.get("role") != "tool", (
        "payload 가 role:'tool' 로 시작한다 — 선행 assistant 가 없다"
    )
    for item in payload:
        for call in item.get("tool_calls") or []:
            issued.add(call.get("id"))
        if item.get("role") == "tool":
            assert item.get("tool_call_id") in issued, (
                f"고아 tool 항목({item.get('tool_call_id')}) — 짝이 되는 "
                "assistant tool_calls 가 앞에 없다"
            )


def _tool_conversation(n: int, chars: int) -> list[Message]:
    """질문 하나 + 도구 호출 + 거대한 도구 결과들 — 컨텍스트가 터지는 실제 형태."""
    calls = [{"id": f"tu{i}", "name": "Read", "input": {}} for i in range(n)]
    msgs = [
        Message.user("이 파일들을 분석해 주세요"),
        Message.assistant(text="파일을 읽겠습니다", tool_uses=calls),
    ]
    for i in range(n):
        msgs.append(Message.tool_result(f"tu{i}", f"결과{i} " + "가" * chars))
    return msgs


@pytest.mark.asyncio
@pytest.mark.parametrize("n_results", [1, 3, 12])
async def test_force_compaction_result_is_sendable(monkeypatch, n_results):
    """★핵심★ force 압축 결과는 그대로 보낼 수 있어야 한다."""
    mgr = _mgr()

    async def _summary(_messages):
        return "짧은 요약"

    monkeypatch.setattr(mgr, "_get_model_summary", _summary)

    messages = _tool_conversation(n_results, 40_000)
    before = mgr._estimate_tokens(messages)
    result = await mgr.auto_compact_if_needed(messages, force=True)

    assert_sendable(_payload(result))
    assert mgr._estimate_tokens(result) < before, "force 인데 줄지 않았다"


@pytest.mark.asyncio
async def test_emergency_compaction_result_is_sendable():
    """긴급 압축도 같은 계약을 지켜야 한다 — 여기도 곧바로 payload 가 된다."""
    mgr = _mgr()
    messages = _tool_conversation(4, 5_000)

    result = await mgr.emergency_compact(messages)

    assert_sendable(_payload(result))


@pytest.mark.asyncio
async def test_auto_compaction_result_is_sendable(monkeypatch):
    """임계치 자동 압축도 마찬가지다."""
    mgr = _mgr(max_context_tokens=2_000)

    async def _summary(_messages):
        return "짧은 요약"

    monkeypatch.setattr(mgr, "_get_model_summary", _summary)

    messages: list[Message] = []
    for i in range(10):
        messages.append(Message.user(f"질문{i} " + "가" * 300))
        messages.append(
            Message.assistant(
                text=f"답변{i}", tool_uses=[{"id": f"tu{i}", "name": "Read", "input": {}}]
            )
        )
        messages.append(Message.tool_result(f"tu{i}", f"결과{i} " + "나" * 300))

    result = await mgr.auto_compact_if_needed(messages)

    assert_sendable(_payload(result))


def test_apply_all_result_is_sendable():
    """apply_all(1~3단계) 결과도 payload 가 된다 — query_loop 이 매 턴 쓴다."""
    mgr = _mgr(max_context_tokens=1_000, tool_result_budget=100)
    messages = _tool_conversation(6, 20_000)

    assert_sendable(_payload(mgr.apply_all(messages)))
