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

from core.message import Message, StreamEvent, StreamEventType
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


@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [6, 12])
async def test_multi_turn_flow_stays_sendable(turns):
    """★핵심★ 경계가 설정된 뒤 여러 턴을 돌려도 매 턴 보낼 수 있어야 한다.

    `query_loop.py:837-838` 의 실제 흐름을 그대로 재현한다.

        api_messages = context_manager.apply_all(state.messages)
        api_messages = await context_manager.auto_compact_if_needed(api_messages)

    경계는 **apply_all 출력** 기준으로 잡히는데 다음 턴에는 **state.messages** 에
    적용된다. 두 리스트는 길이가 다르다 — apply_all 이 요약을 앞에 붙이고 snip 이
    턴을 접기 때문이다. 신선한 매니저 하나만 보는 테스트로는 이 구간이 안 잡힌다.
    """

    class _P:
        async def stream(self, **_kwargs):
            yield StreamEvent(
                type=StreamEventType.TEXT_DELTA, text="앞부분 요약: 요구사항과 결정"
            )

    mgr = _mgr(
        model_provider=_P(),
        max_context_tokens=3_000,
        preserve_recent_turns=2,
        tool_result_budget=200,
    )

    state: list[Message] = []
    for t in range(turns):
        state.append(Message.user(f"질문{t} " + "가" * 400))
        state.append(
            Message.assistant(
                text=f"답변{t}", tool_uses=[{"id": f"tu{t}", "name": "Read", "input": {}}]
            )
        )
        state.append(Message.tool_result(f"tu{t}", f"결과{t} " + "나" * 400))

        api = mgr.apply_all(state)
        api = await mgr.auto_compact_if_needed(api)

        assert_sendable(_payload(api))
        # 경계는 항상 **이번에 넘어온 리스트** 안의 실제 메시지를 가리켜야 한다.
        resolved = mgr._resolve_boundary(state)
        assert 0 <= resolved <= len(state), f"턴{t}: 경계가 범위를 벗어났다({resolved})"


@pytest.mark.asyncio
async def test_boundary_points_at_the_same_message_in_the_callers_list():
    """★불변식★ 경계는 호출부 리스트에서도 **같은 메시지**를 가리켜야 한다.

    `_compact_boundary` 는 `auto_compact_if_needed` 에 넘어온 리스트 기준 인덱스인데,
    query_loop 은 `apply_all(state.messages)` 의 출력을 넘기고 다음 턴에는 그 인덱스를
    `state.messages` 에 적용한다. 두 리스트는 길이가 다르다 — apply_all 이 요약을
    앞에 붙이고 snip 이 턴을 접기 때문이다.

    실측(운영과 같은 흐름). 저장된 인덱스가 가리키는 메시지가 매 턴 어긋났다.

        턴2  state= 9  apply_all→7  저장=1  올바른 위치=3
        턴5  state=18  apply_all→7  저장=2  올바른 위치=12

    지금은 보존한 첫 메시지의 id 를 같이 기억해 리스트가 달라도 같은 곳을 찾는다.
    이 테스트가 없으면 인덱스로 되돌려도 아무도 울지 않는다 — 다른 압축 단계가
    결과를 덮어써 증상이 가려지기 때문이다.
    """

    class _P:
        async def stream(self, **_kwargs):
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="앞부분 요약")

    mgr = _mgr(
        model_provider=_P(),
        max_context_tokens=3_000,
        preserve_recent_turns=2,
        tool_result_budget=200,
    )

    state: list[Message] = []
    checked = 0
    for t in range(6):
        state.append(Message.user(f"질문{t} " + "가" * 400))
        state.append(Message.assistant(text=f"답변{t}"))
        state.append(Message.tool_result(f"tu{t}", f"결과{t} " + "나" * 400))

        api = mgr.apply_all(state)
        api = await mgr.auto_compact_if_needed(api)

        if not mgr._compact_boundary_id:
            continue
        expected = next(
            (i for i, m in enumerate(state) if m.id == mgr._compact_boundary_id), None
        )
        if expected is None:
            continue
        checked += 1

        # ★핵심★ apply_all 이 **실제로** 그 자리에서 잘라야 한다. _resolve_boundary 를
        # 직접 부르면 apply_all 이 그것을 쓰는지 안 쓰는지를 못 본다.
        again = mgr.apply_all(state)
        body = [m for m in again if not (m.text_content or "").startswith("[대화 요약]")]
        dropped_before = {
            (m.text_content or "")[:8] for m in state[:expected] if m.text_content
        }
        leaked = [
            (m.text_content or "")[:8]
            for m in body
            if (m.text_content or "")[:8] in dropped_before
        ]
        assert not leaked, (
            f"턴{t}: 요약으로 대체된 구간이 본문에 다시 나왔다 {leaked} — "
            f"경계가 어긋났다(저장 {mgr._compact_boundary}, 실제 {expected})"
        )

    assert checked > 0, "경계가 한 번도 설정되지 않아 아무것도 검증하지 못했다"


@pytest.mark.asyncio
async def test_drifting_boundary_does_not_force_a_summary_every_turn():
    """★실측★ 경계가 표류하면 압축이 **매 턴** 돈다 — 모델 왕복이 3배가 된다.

    경계는 `apply_all` 출력 기준 인덱스인데 다음 턴에는 `state.messages` 에 적용된다.
    요약이 앞에 붙어 리스트 길이가 달라지므로 경계가 매 턴 위로 밀리고, 활성 구간이
    다시 자라지 못해 압축 조건이 계속 참으로 남는다. 14턴 실측.

        옛 방식(인덱스)  턴7부터 매 턴 압축, 경계 10→12→13→15→16→18→19
        id 방식          3턴마다 한 번, 경계 10 고정

    토큰이나 payload 로는 안 드러난다 — 결과 모양이 같기 때문이다. 드러나는 것은
    **요약 호출 횟수**다.
    """
    calls = {"n": 0}

    class _P:
        async def stream(self, **_kwargs):
            calls["n"] += 1
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="앞부분 요약")

    mgr = _mgr(
        model_provider=_P(),
        max_context_tokens=6_000,
        preserve_recent_turns=2,
        tool_result_budget=100_000,
    )
    mgr.snip_threshold = 10.0  # snip 을 끄고 경계 효과만 남긴다

    state: list[Message] = []
    turns = 14
    for t in range(turns):
        state.append(Message.user(f"질문{t} " + "가" * 300))
        state.append(
            Message.assistant(
                text=f"답변{t}", tool_uses=[{"id": f"tu{t}", "name": "Read", "input": {}}]
            )
        )
        state.append(Message.tool_result(f"tu{t}", f"결과{t} " + "나" * 300))

        api = mgr.apply_all(state)
        api = await mgr.auto_compact_if_needed(api)
        assert_sendable(_payload(api))

    # 경계가 표류하면 턴7부터 전부(=8회 이상) 돈다. 정상이면 3턴에 한 번꼴이다.
    assert calls["n"] <= turns // 2, (
        f"{turns}턴 동안 요약이 {calls['n']}회 돌았다 — 경계가 표류해 압축 조건이 "
        "계속 참으로 남는다"
    )
