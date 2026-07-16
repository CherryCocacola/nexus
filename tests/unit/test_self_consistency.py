# 자기일관성(Self-Consistency) 사실 검증 기능 단위 테스트 — Point 4.3
"""
자기일관성 기능의 4계층 전파와 합의 로직을 격리 검증한다.

검증 범위(설계 §7 Phase 1~4 테스트 계획):
  - 합의 순수 함수(self_consistency): 정규화·다수결·동점 실패·임베딩 medoid.
  - 라우팅 게이팅(routing): G1(설정)/G2(클래스)/G3(사실형 패턴) 3중 게이트.
  - Tier 3(inference): stream(n>1) choices index 디멀티플렉스 → SC_CANDIDATE.
  - Tier 2(query_loop): 후보 수집 → 합의 → 승자 의사-스트림, tool_calls 폴백,
    enabled=false/게이트 미통과 시 SC 미발동(무회귀).

실제 vLLM/GPU/Redis/PG는 호출하지 않는다. httpx는 monkeypatch로 캡처하고,
AsyncGenerator는 규칙대로 `async for`로 전량 소비한다.

★완료 조건 미충족 항목(설계 §4/§7 Phase 6)★:
  본 파일은 mock 기반 로직 검증만 한다. §4의 latency/토큰/정확도 추정치는 미실측이며,
  B200 실서버에서 kowiki 사실형 QA 셋으로 on/off 벤치해 추정치를 실측으로 교체하는
  것이 최종 완료 조건이다(사용자가 실서버로 수행). 여기서는 mock으로 대체하지 않는다.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from core.config import RoutingConfig, SelfConsistencyConfig
from core.message import (
    Message,
    StopReason,
    StreamEvent,
    StreamEventType,
    ToolUseBlock,
)
from core.model.inference import LocalModelProvider, ModelConfig, ModelProvider
from core.orchestrator import self_consistency as sc
from core.orchestrator.query_loop import query_loop
from core.orchestrator.routing import RoutingResolver


# ═════════════════════════════════════════════
# 1) 합의 순수 함수 — normalize / majority / tie / numeric / embedding
# ═════════════════════════════════════════════
def test_normalize_answer_strips_markdown_and_josa() -> None:
    """마크다운 장식·조사·문장부호 꼬리가 제거되어 같은 답이 같은 문자열이 된다."""
    assert sc.normalize_answer("**1443년**입니다.") == sc.normalize_answer("1443년")
    assert sc.normalize_answer("- 세종") == sc.normalize_answer("세종")


def test_normalize_answer_numeric_comma_and_approx() -> None:
    """천단위 콤마·'약'·공백이 제거되어 수치 표면형이 통일된다(§3.1)."""
    assert sc.normalize_answer("5,100만 명") == sc.normalize_answer("약 5100만명")


def test_majority_vote_two_of_three_wins() -> None:
    """['1443년','1443년','1446년'] → '1443년' 다수결 채택(설계 §3.2)."""
    result = sc.majority_vote(["1443년", "1443년", "1446년"], min_agreement=2)
    assert result.winner == "1443년"
    assert result.consensus_reached is True
    assert result.agreement == 2
    assert result.method == "majority"


def test_majority_vote_picks_longest_original_in_group() -> None:
    """승자 그룹의 '원문 중 가장 긴 후보'를 표출용으로 채택한다(§3.2)."""
    result = sc.majority_vote(["1443년", "**1443년**입니다", "1446년"], min_agreement=2)
    # 정규화 폼은 같지만 원문 중 가장 긴 것을 돌려준다.
    assert result.winner == "**1443년**입니다"
    assert result.consensus_reached is True


def test_majority_vote_numeric_normalization_same_vote() -> None:
    """['5,100만 명','약 5100만명','4800만'] → 5100만 그룹이 다수(2표)로 승리."""
    result = sc.resolve_consensus(
        ["5,100만 명", "약 5100만명", "4800만"],
        min_agreement=2,
        short_answer_max_chars=80,
    )
    assert result.consensus_reached is True
    assert result.agreement == 2
    assert sc.normalize_answer(result.winner) == sc.normalize_answer("5100만명")


def test_majority_vote_all_disagree_falls_back_first() -> None:
    """3표가 전부 다르면 합의 실패 → 후보 0번 채택 + consensus_reached=False(§3.4)."""
    result = sc.majority_vote(["1443년", "1446년", "1450년"], min_agreement=2)
    assert result.consensus_reached is False
    assert result.winner == "1443년"  # 후보 0번
    assert result.method == "fallback_first"


def test_majority_vote_empty_is_safe() -> None:
    """빈 후보 리스트도 크래시 없이 실패 결과를 돌려준다(방어)."""
    result = sc.majority_vote([], min_agreement=2)
    assert result.consensus_reached is False
    assert result.winner == ""


def test_cluster_by_embedding_medoid() -> None:
    """서술형 폴백 — 최대 클러스터의 medoid를 채택한다(§3.3).

    후보 0,1은 임베딩이 거의 같고(코사인~1), 2는 직교(탈선 표본).
    → 클러스터 {0,1}, medoid는 0 또는 1, 합의 성공(min_agreement=2).
    """
    candidates = ["긴 서술형 답변 A" * 10, "긴 서술형 답변 A 유사" * 8, "완전히 다른 주제" * 10]
    embeddings = [[1.0, 0.0], [1.0, 0.02], [0.0, 1.0]]
    result = sc.cluster_by_embedding(
        candidates, embeddings, min_agreement=2, similarity_threshold=0.90
    )
    assert result.consensus_reached is True
    assert result.method == "embedding"
    assert result.winner in (candidates[0], candidates[1])


def test_cluster_by_embedding_mismatch_falls_back_to_majority() -> None:
    """임베딩 개수가 후보와 안 맞으면 majority_vote로 안전 폴백한다."""
    result = sc.cluster_by_embedding(
        ["a", "a", "b"], embeddings=[[1.0]], min_agreement=2
    )
    # 임베딩 불일치 → majority (["a","a","b"] → a 2표)
    assert result.winner == "a"
    assert result.consensus_reached is True


def test_resolve_consensus_long_without_embeddings_uses_majority() -> None:
    """서술형인데 임베딩 미제공 → majority로 폴백(정규화 우연 일치 시 다수결)."""
    long_a = "동일한 긴 문장입니다 " * 10
    result = sc.resolve_consensus(
        [long_a, long_a, "다른 긴 문장 " * 10],
        min_agreement=2,
        short_answer_max_chars=20,  # 전부 초과 → 서술형 경로
        embeddings=None,
    )
    assert result.consensus_reached is True
    assert result.winner == long_a


# ═════════════════════════════════════════════
# 2) 라우팅 게이팅 — G1 / G2 / G3
# ═════════════════════════════════════════════
def _routing_with_sc(**sc_kwargs: Any) -> RoutingConfig:
    """SC가 켜진 RoutingConfig를 만든다(테스트용)."""
    defaults: dict[str, Any] = {"enabled": True}
    defaults.update(sc_kwargs)
    return RoutingConfig(self_consistency=SelfConsistencyConfig(**defaults))


def test_gate_all_pass_knowledge_factual_activates_sc() -> None:
    """G1~G3 통과(KNOWLEDGE + 사실형 짧은 질문) → sc_n=config.n, SC 전용 샘플링 치환."""
    resolver = RoutingResolver(_routing_with_sc(n=3))
    decision = resolver.resolve("세종대왕이 훈민정음을 언제 반포했나요?")

    assert decision.query_class == "KNOWLEDGE"
    assert decision.sc_n == 3
    assert decision.self_consistency_active is True
    # SC 전용 샘플링으로 치환(§5.3).
    assert decision.temperature == pytest.approx(0.7)
    assert decision.top_p == pytest.approx(0.95)
    assert decision.max_tokens_cap == 512


def test_gate_disabled_no_sc_no_regression() -> None:
    """G1 미통과(enabled=false, 기본값) → sc_n=1, 샘플링은 KNOWLEDGE 프로필 그대로(무회귀)."""
    resolver = RoutingResolver(RoutingConfig())  # SC 기본 비활성
    decision = resolver.resolve("세종대왕이 훈민정음을 언제 반포했나요?")

    assert decision.sc_n == 1
    assert decision.self_consistency_active is False
    # 무회귀 — KNOWLEDGE 프로필의 기존 샘플링 값이 그대로 유지되어야 한다.
    assert decision.temperature == pytest.approx(0.2)
    assert decision.top_p == pytest.approx(0.95)
    assert decision.max_tokens_cap == 2048


def test_gate_tool_class_never_applies() -> None:
    """G2 미통과 — TOOL 분류 질의는 SC enabled여도 sc_n=1(도구 경로 보호)."""
    resolver = RoutingResolver(_routing_with_sc(n=3))
    # 사실형 패턴("언제")이 있어도 TOOL 키워드("파일 읽어줘")가 우선 분류된다.
    decision = resolver.resolve("이 파일 읽어줘, 언제 만든 거야")
    assert decision.query_class == "TOOL"
    assert decision.sc_n == 1


def test_gate_non_factual_knowledge_no_sc() -> None:
    """G3 미통과 — 서술형(사실형 패턴 없음) KNOWLEDGE 질의는 SC 미발동."""
    resolver = RoutingResolver(_routing_with_sc(n=3))
    decision = resolver.resolve("니체 철학을 자세히 설명해줘")
    assert decision.query_class == "KNOWLEDGE"
    assert decision.sc_n == 1  # 사실형 패턴 미매치 → 게이트 차단


def test_gate_long_question_no_sc() -> None:
    """G3 길이 상한 — factual 패턴이 있어도 질문이 너무 길면 서술형으로 보고 제외."""
    resolver = RoutingResolver(_routing_with_sc(n=3, factual_max_question_chars=20))
    decision = resolver.resolve("대한민국의 수도는 어디이고 인구는 얼마이며 면적은 몇 인가요")
    assert decision.query_class == "KNOWLEDGE"
    assert decision.sc_n == 1  # 길이 초과 → 차단


def test_gate_factual_gate_off_applies_to_any_knowledge() -> None:
    """factual_gate=false면 G3를 끈다 — 서술형 KNOWLEDGE에도 SC가 발동한다."""
    resolver = RoutingResolver(_routing_with_sc(n=3, factual_gate=False))
    decision = resolver.resolve("니체 철학을 자세히 설명해줘")
    assert decision.query_class == "KNOWLEDGE"
    assert decision.sc_n == 3


# ═════════════════════════════════════════════
# 3) Tier 3 — stream(n>1) choices 디멀티플렉스 → SC_CANDIDATE
# ═════════════════════════════════════════════
class _MultiChoiceSSEResponse:
    """n=3 표본을 choices[i].index로 구분해 흘려보내는 가짜 vLLM SSE 응답."""

    status_code = 200

    def __init__(self, tool_in_choice0: bool = False) -> None:
        self._tool = tool_in_choice0

    async def aiter_lines(self) -> AsyncGenerator[str, None]:
        # 표본 0/1은 "1443년", 표본 2는 "1446년"을 흘려보낸다.
        yield 'data: {"choices":[{"index":0,"delta":{"content":"1443년"},"finish_reason":null}]}'
        yield 'data: {"choices":[{"index":1,"delta":{"content":"1443년"},"finish_reason":null}]}'
        yield 'data: {"choices":[{"index":2,"delta":{"content":"1446년"},"finish_reason":null}]}'
        if self._tool:
            # 표본 0에 tool_call 조각 → SC 포기(폴백) 유도.
            yield (
                'data: {"choices":[{"index":0,"delta":{"tool_calls":'
                '[{"index":0,"id":"call_1","function":{"name":"Read","arguments":"{}"}}]},'
                '"finish_reason":"tool_calls"}]}'
            )
        else:
            yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        yield 'data: {"usage":{"prompt_tokens":10,"completion_tokens":9}}'
        yield "data: [DONE]"

    async def aiter_bytes(self) -> AsyncGenerator[bytes, None]:
        if False:
            yield b""


class _FakeStreamCtx:
    """`async with client.stream(...)` 컨텍스트 — 지정한 응답을 내어준다."""

    def __init__(self, response: Any, capture: dict[str, Any], **kwargs: Any) -> None:
        capture.clear()
        capture.update(kwargs)
        self._response = response

    async def __aenter__(self) -> Any:
        return self._response

    async def __aexit__(self, *exc: Any) -> bool:
        return False


async def _run_stream(
    provider: LocalModelProvider, response: Any, **stream_kwargs: Any
) -> tuple[list[StreamEvent], dict[str, Any]]:
    """stream()을 전량 소비하며 이벤트 목록과 전송 payload를 반환한다."""
    from unittest.mock import patch

    captured: dict[str, Any] = {}

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamCtx:
        return _FakeStreamCtx(response, captured, **kwargs)

    events: list[StreamEvent] = []
    with patch.object(provider._client, "stream", side_effect=fake_stream):
        async for ev in provider.stream(
            messages=[Message.user("세종대왕 훈민정음 반포 연도?")],
            system_prompt="너는 도우미다.",
            **stream_kwargs,
        ):
            events.append(ev)
    return events, captured


def _make_provider() -> LocalModelProvider:
    return LocalModelProvider(base_url="http://192.168.21.112:8000")


async def test_stream_n3_injects_n_and_emits_sc_candidates() -> None:
    """n=3 → payload에 n=3 주입, 표본별 SC_CANDIDATE(sample_index) 방출, TEXT_DELTA 억제."""
    provider = _make_provider()
    events, payload = await _run_stream(
        provider, _MultiChoiceSSEResponse(), n=3
    )

    assert payload["json"]["n"] == 3
    # 라이브 TEXT_DELTA는 없어야 한다(합의 후 상위에서 의사-스트림).
    assert not any(e.type == StreamEventType.TEXT_DELTA.value for e in events)
    # 표본 3개가 SC_CANDIDATE로 올라와야 한다.
    cands = [e for e in events if e.type == StreamEventType.SC_CANDIDATE.value]
    assert len(cands) == 3
    by_idx = {e.sample_index: e.text for e in cands}
    assert by_idx == {0: "1443년", 1: "1443년", 2: "1446년"}
    # MESSAGE_STOP + usage(표본 합산이 반영된 vLLM 값)도 정직 보고.
    stop = [e for e in events if e.type == StreamEventType.MESSAGE_STOP.value]
    assert stop and stop[-1].usage.output_tokens == 9


async def test_stream_n1_regression_no_n_key_and_live_text() -> None:
    """n=1(기본) → payload에 'n' 키가 없고, 기존처럼 라이브 TEXT_DELTA를 낸다(무회귀)."""

    class _SingleSSE:
        status_code = 200

        async def aiter_lines(self) -> AsyncGenerator[str, None]:
            yield 'data: {"choices":[{"delta":{"content":"안녕"},"finish_reason":null}]}'
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":3,"completion_tokens":1}}'
            )
            yield "data: [DONE]"

        async def aiter_bytes(self) -> AsyncGenerator[bytes, None]:
            if False:
                yield b""

    provider = _make_provider()
    events, payload = await _run_stream(provider, _SingleSSE())  # n 생략 → 1

    assert "n" not in payload["json"]  # payload 바이트 동일성(무회귀)
    texts = [e for e in events if e.type == StreamEventType.TEXT_DELTA.value]
    assert texts and texts[0].text == "안녕"
    # SC_CANDIDATE는 절대 나오면 안 된다.
    assert not any(e.type == StreamEventType.SC_CANDIDATE.value for e in events)


async def test_stream_n3_tool_call_aborts_to_single_fallback() -> None:
    """n=3인데 표본0에 tool_call → SC 포기, choice 0만 단일 스트림(TEXT_DELTA+TOOL_USE)."""
    provider = _make_provider()
    events, _ = await _run_stream(
        provider, _MultiChoiceSSEResponse(tool_in_choice0=True), n=3
    )
    # 폴백이므로 SC_CANDIDATE는 없고, choice0 텍스트 + tool_use가 나와야 한다.
    assert not any(e.type == StreamEventType.SC_CANDIDATE.value for e in events)
    assert any(e.type == StreamEventType.TEXT_DELTA.value for e in events)
    assert any(e.type == StreamEventType.TOOL_USE_STOP.value for e in events)


# ═════════════════════════════════════════════
# 4) Tier 2 — query_loop SC 통합 (합의·폴백·무회귀)
# ═════════════════════════════════════════════
class SCMockProvider(ModelProvider):
    """Tier 3가 SC 모드에서 산출하는 이벤트를 흉내내는 mock 프로바이더.

    n>1로 호출되면 mode에 따라:
      - "candidates": 설정된 후보 텍스트들을 SC_CANDIDATE로 방출(순수 SC).
      - "tool": TEXT_DELTA + TOOL_USE(폴백) 방출.
    n==1로 호출되면 단일 텍스트 응답을 낸다(무회귀 경로 확인용).
    """

    def __init__(
        self,
        candidates: list[str] | None = None,
        mode: str = "candidates",
        single_text: str = "단일응답",
    ) -> None:
        self._candidates = candidates or []
        self._mode = mode
        self._single_text = single_text
        self.last_n = 1

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
        model_override: str | None = None,
        enable_thinking: bool | None = False,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        structured_output: Any = None,
        n: int = 1,
        force_tool_choice: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.last_n = n
        yield StreamEvent(type=StreamEventType.MESSAGE_START, model_id="mock")

        if n > 1 and self._mode == "candidates":
            for i, text in enumerate(self._candidates):
                yield StreamEvent(
                    type=StreamEventType.SC_CANDIDATE, text=text, sample_index=i
                )
            yield StreamEvent(
                type=StreamEventType.MESSAGE_STOP, stop_reason=StopReason.END_TURN
            )
            return

        if n > 1 and self._mode == "tool":
            # 폴백 — choice0 텍스트 + tool_use(단일 스트림처럼).
            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="파일을 읽겠습니다")
            tub = ToolUseBlock(name="Read", input={"file_path": "x.txt"})
            yield StreamEvent(type=StreamEventType.TOOL_USE_START, tool_use=tub)
            yield StreamEvent(type=StreamEventType.TOOL_USE_STOP, tool_use=tub)
            yield StreamEvent(
                type=StreamEventType.MESSAGE_STOP, stop_reason=StopReason.TOOL_USE
            )
            return

        # n==1 단일 경로.
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=self._single_text)
        yield StreamEvent(
            type=StreamEventType.MESSAGE_STOP, stop_reason=StopReason.END_TURN
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def health_check(self) -> bool:
        return True

    async def count_tokens(self, messages: list[Message]) -> int:
        return len(messages) * 10

    def get_config(self) -> ModelConfig:
        return ModelConfig(model_id="mock", max_context_tokens=8192, max_output_tokens=4096)


async def _collect(gen: AsyncGenerator[Any, None]) -> list[Any]:
    """AsyncGenerator를 규칙대로 전량 소비해 리스트로 모은다."""
    out: list[Any] = []
    async for item in gen:
        out.append(item)
    return out


async def test_query_loop_sc_majority_consensus_pseudo_streams_winner(
    tool_use_context: Any,
) -> None:
    """SC 모드: 표본 후보를 버퍼링→다수결→승자를 TEXT_DELTA로 의사-스트림한다."""
    provider = SCMockProvider(candidates=["1443년", "1443년", "1446년"])
    events = await _collect(
        query_loop(
            messages=[Message.user("훈민정음 반포 연도?")],
            system_prompt="s",
            model_provider=provider,
            tools=[],
            context=tool_use_context,
            sc_n=3,
            sc_min_agreement=2,
        )
    )
    assert provider.last_n == 3
    # SC 후보 원문은 UI로 새어나가면 안 된다(버퍼 전용).
    assert not any(
        isinstance(e, StreamEvent) and e.type == StreamEventType.SC_CANDIDATE.value
        for e in events
    )
    # SYSTEM_INFO(활동 표시)가 첫 TEXT_DELTA보다 앞에 나와야 한다(§6.2).
    info_idx = next(
        i for i, e in enumerate(events)
        if isinstance(e, StreamEvent) and e.type == StreamEventType.SYSTEM_INFO.value
    )
    text_idx = next(
        i for i, e in enumerate(events)
        if isinstance(e, StreamEvent) and e.type == StreamEventType.TEXT_DELTA.value
    )
    assert info_idx < text_idx
    # 승자("1443년")가 의사-스트림으로 나오고 assistant 메시지 본문이 된다.
    winner_text = "".join(
        e.text
        for e in events
        if isinstance(e, StreamEvent)
        and e.type == StreamEventType.TEXT_DELTA.value
        and e.text
    )
    assert winner_text == "1443년"
    assistant = [
        e for e in events if isinstance(e, Message) and e.role == "assistant"
    ]
    assert assistant and assistant[-1].text_content == "1443년"


async def test_query_loop_sc_consensus_failure_warns_and_uses_first(
    tool_use_context: Any,
) -> None:
    """3표 분열 → SYSTEM_WARNING + 후보 0번 채택(§3.4)."""
    provider = SCMockProvider(candidates=["1443년", "1446년", "1450년"])
    events = await _collect(
        query_loop(
            messages=[Message.user("연도?")],
            system_prompt="s",
            model_provider=provider,
            tools=[],
            context=tool_use_context,
            sc_n=3,
            sc_min_agreement=2,
        )
    )
    warnings = [
        e for e in events
        if isinstance(e, StreamEvent) and e.type == StreamEventType.SYSTEM_WARNING.value
    ]
    assert any("합의 실패" in (w.message or "") for w in warnings)
    winner_text = "".join(
        e.text for e in events
        if isinstance(e, StreamEvent)
        and e.type == StreamEventType.TEXT_DELTA.value and e.text
    )
    assert winner_text == "1443년"  # 후보 0번


async def test_query_loop_sc_tool_call_aborts_to_normal_turn(
    tool_use_context: Any,
    basic_tools: Any,
) -> None:
    """표본에 tool_calls가 섞이면(Tier3 폴백) SC 합의를 건너뛰고 일반 도구 실행 턴으로."""
    provider = SCMockProvider(mode="tool")
    events = await _collect(
        query_loop(
            messages=[Message.user("x.txt 읽어줘")],
            system_prompt="s",
            model_provider=provider,
            tools=basic_tools,
            context=tool_use_context,
            sc_n=3,
        )
    )
    # tool_use가 실제로 실행되어 tool_result Message가 히스토리에 들어간다.
    tool_results = [
        e for e in events if isinstance(e, Message) and e.role == "tool_result"
    ]
    assert tool_results, "SC 폴백 시 도구가 정상 실행되어야 한다"
    # 폴백 경로에서는 SC_CANDIDATE 소비/합의가 없다(그냥 단일 턴).
    assert not any(
        isinstance(e, StreamEvent) and e.type == StreamEventType.SC_CANDIDATE.value
        for e in events
    )


async def test_query_loop_sc_disabled_no_behavior_change(
    tool_use_context: Any,
) -> None:
    """sc_n=1(기본) → SC 로직 완전 우회, n=1로 stream 호출, SYSTEM_INFO 없음(무회귀)."""
    provider = SCMockProvider(single_text="일반 응답입니다")
    events = await _collect(
        query_loop(
            messages=[Message.user("아무 질문")],
            system_prompt="s",
            model_provider=provider,
            tools=[],
            context=tool_use_context,
            # sc_n 생략 → 1
        )
    )
    assert provider.last_n == 1
    # SC 활동 표시가 없어야 한다(무회귀).
    assert not any(
        isinstance(e, StreamEvent)
        and e.type == StreamEventType.SYSTEM_INFO.value
        and "교차 검증" in (e.message or "")
        for e in events
    )
    assistant = [
        e for e in events if isinstance(e, Message) and e.role == "assistant"
    ]
    assert assistant and assistant[-1].text_content == "일반 응답입니다"
