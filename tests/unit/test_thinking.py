"""
Thinking Engine 단위 테스트.

테스트 대상:
  - ComplexityAssessor: 키워드, 길이, 코드블록, 증폭 패턴, 컨텍스트 보정
  - ThinkingStrategy / select_strategy: 스코어 → 전략 매핑
  - ThinkingCache: LRU 동작, TTL 만료, 키 생성
  - HiddenCoTEngine: 2-pass 실행 (mock 모델)
  - SelfReflectionEngine: 3-pass 실행 (mock 모델)
  - ThinkingOrchestrator: 전략 선택 + 엔진 실행 + 캐시 통합

외부 서비스(vLLM)는 모두 mock 처리한다.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from core.message import Message, StreamEvent, StreamEventType
from core.model.inference import ModelProvider
from core.thinking.assessor import ComplexityAssessor
from core.thinking.cache import ThinkingCache
from core.thinking.hidden_cot import HiddenCoTEngine
from core.thinking.orchestrator import ThinkingOrchestrator, ThinkingResult
from core.thinking.self_reflection import SelfReflectionEngine
from core.thinking.strategy import (
    DEFAULT_CONFIGS,
    ThinkingStrategy,
    select_strategy,
)


# ─────────────────────────────────────────────
# 테스트용 Mock ModelProvider
# ─────────────────────────────────────────────
class MockModelProvider(ModelProvider):
    """
    테스트용 모델 프로바이더.
    stream()이 호출되면 미리 설정된 텍스트를 StreamEvent로 yield한다.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        """
        Args:
            responses: 순서대로 반환할 텍스트 목록.
                      각 stream() 호출마다 하나씩 소비된다.
        """
        self._responses = list(responses or ["mock response"])
        self._call_count = 0

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """미리 설정된 텍스트를 StreamEvent로 yield한다."""
        # 응답 텍스트 선택 (호출 순서대로)
        idx = min(self._call_count, len(self._responses) - 1)
        text = self._responses[idx]
        self._call_count += 1

        yield StreamEvent(type=StreamEventType.MESSAGE_START, model_id="mock-model")
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=text)
        yield StreamEvent(type=StreamEventType.MESSAGE_STOP)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]

    async def health_check(self) -> bool:
        return True

    async def count_tokens(self, messages: list[Message]) -> int:
        return sum(m.estimated_tokens() for m in messages)

    def get_config(self):
        from core.model.inference import ModelConfig

        return ModelConfig(model_id="mock-model")


# ─────────────────────────────────────────────
# ComplexityAssessor 테스트
# ─────────────────────────────────────────────
class TestComplexityAssessor:
    """ComplexityAssessor의 복잡도 평가 로직을 검증한다."""

    def setup_method(self) -> None:
        self.assessor = ComplexityAssessor()

    def test_assess_empty_message_returns_zero(self) -> None:
        """빈 메시지는 복잡도 0.0을 반환한다."""
        assert self.assessor.assess("") == 0.0
        assert self.assessor.assess("   ") == 0.0

    def test_assess_simple_message_low_score(self) -> None:
        """단순한 메시지는 낮은 스코어를 반환한다."""
        score = self.assessor.assess("hello")
        assert score < 0.3

    def test_assess_keyword_refactor_increases_score(self) -> None:
        """'refactor' 키워드가 포함되면 스코어가 올라간다."""
        base = self.assessor.assess("change this function")
        with_refactor = self.assessor.assess("refactor this function")
        assert with_refactor > base

    def test_assess_keyword_architecture_high_weight(self) -> None:
        """'architecture' 키워드는 높은 가중치를 가진다."""
        score = self.assessor.assess("design the architecture")
        # architecture(0.4) + design(0.3) = 0.7 기본
        assert score >= 0.5

    def test_assess_korean_keywords(self) -> None:
        """한글 키워드도 정상적으로 매칭된다."""
        score = self.assessor.assess("이 코드를 리팩토링해 주세요")
        assert score > 0.0

    def test_assess_long_message_bonus(self) -> None:
        """긴 메시지는 길이 보정으로 스코어가 올라간다."""
        short = self.assessor.assess("fix bug")
        # 200자 이상 메시지 생성
        long_msg = "fix bug " + "a" * 200
        long_score = self.assessor.assess(long_msg)
        assert long_score > short

    def test_assess_code_block_bonus(self) -> None:
        """코드 블록이 포함되면 +0.1 가산된다."""
        without_code = self.assessor.assess("fix this function")
        with_code = self.assessor.assess("fix this function\n```python\ndef foo(): pass\n```")
        assert with_code > without_code

    def test_assess_amplifier_multifile(self) -> None:
        """'multi-file' 증폭 패턴이 적용되면 스코어가 증폭된다."""
        base = self.assessor.assess("refactor this")
        amplified = self.assessor.assess("refactor this across multi-file")
        assert amplified > base

    def test_assess_amplifier_korean_pattern(self) -> None:
        """한글 증폭 패턴 '여러 파일'이 적용된다."""
        base = self.assessor.assess("수정해 주세요")
        amplified = self.assessor.assess("여러 파일에 걸쳐 수정해 주세요")
        assert amplified >= base  # 증폭 패턴 적용

    def test_assess_context_bonus(self) -> None:
        """긴 대화 컨텍스트가 있으면 스코어가 올라간다."""
        # 컨텍스트 없는 경우
        no_ctx = self.assessor.assess("fix this")
        # 10턴 컨텍스트
        context = [Message.user(f"turn {i}") for i in range(10)]
        with_ctx = self.assessor.assess("fix this", context=context)
        assert with_ctx > no_ctx

    def test_assess_context_20_turns_max_bonus(self) -> None:
        """20턴 이상 컨텍스트에서 최대 보정(+0.2)이 적용된다."""
        context_20 = [Message.user(f"turn {i}") for i in range(20)]
        context_25 = [Message.user(f"turn {i}") for i in range(25)]
        score_20 = self.assessor.assess("fix this", context=context_20)
        score_25 = self.assessor.assess("fix this", context=context_25)
        # 20턴과 25턴 모두 +0.2 보정이므로 동일해야 한다
        assert score_20 == score_25

    def test_assess_clamp_to_max_one(self) -> None:
        """스코어는 1.0을 초과하지 않는다."""
        # 모든 키워드 + 증폭 패턴 + 긴 메시지
        extreme = (
            "refactor debug architecture implement fix create "
            "multi-file across modules end-to-end from scratch "
            "전면 재작성 시스템 전체 " + "x" * 500 + "\n```python\ncode\n```"
        )
        context = [Message.user(f"t{i}") for i in range(30)]
        score = self.assessor.assess(extreme, context=context)
        assert score <= 1.0

    def test_assess_clamp_to_min_zero(self) -> None:
        """스코어는 0.0 미만이 되지 않는다."""
        score = self.assessor.assess("a")
        assert score >= 0.0


# ─────────────────────────────────────────────
# ThinkingStrategy / select_strategy 테스트
# ─────────────────────────────────────────────
class TestThinkingStrategy:
    """전략 선택 로직을 검증한다."""

    def test_select_strategy_direct_below_03(self) -> None:
        """스코어 < 0.3이면 DIRECT를 선택한다."""
        assert select_strategy(0.0) == ThinkingStrategy.DIRECT
        assert select_strategy(0.1) == ThinkingStrategy.DIRECT
        assert select_strategy(0.29) == ThinkingStrategy.DIRECT

    def test_select_strategy_hidden_cot_03_to_06(self) -> None:
        """0.3 <= 스코어 < 0.6이면 HIDDEN_COT를 선택한다."""
        assert select_strategy(0.3) == ThinkingStrategy.HIDDEN_COT
        assert select_strategy(0.45) == ThinkingStrategy.HIDDEN_COT
        assert select_strategy(0.59) == ThinkingStrategy.HIDDEN_COT

    def test_select_strategy_self_reflect_06_to_08(self) -> None:
        """0.6 <= 스코어 < 0.8이면 SELF_REFLECT를 선택한다."""
        assert select_strategy(0.6) == ThinkingStrategy.SELF_REFLECT
        assert select_strategy(0.7) == ThinkingStrategy.SELF_REFLECT
        assert select_strategy(0.79) == ThinkingStrategy.SELF_REFLECT

    def test_select_strategy_multi_agent_08_and_above(self) -> None:
        """스코어 >= 0.8이면 MULTI_AGENT를 선택한다."""
        assert select_strategy(0.8) == ThinkingStrategy.MULTI_AGENT
        assert select_strategy(0.9) == ThinkingStrategy.MULTI_AGENT
        assert select_strategy(1.0) == ThinkingStrategy.MULTI_AGENT

    def test_default_configs_all_strategies_present(self) -> None:
        """DEFAULT_CONFIGS에 4가지 전략 모두의 설정이 있다."""
        for strategy in ThinkingStrategy:
            assert strategy in DEFAULT_CONFIGS

    def test_strategy_config_frozen(self) -> None:
        """StrategyConfig는 frozen이므로 속성 변경이 불가능하다."""
        config = DEFAULT_CONFIGS[ThinkingStrategy.DIRECT]
        with pytest.raises(AttributeError):
            config.max_passes = 99  # type: ignore[misc]

    def test_direct_config_single_pass(self) -> None:
        """DIRECT 전략은 1회 pass다."""
        assert DEFAULT_CONFIGS[ThinkingStrategy.DIRECT].max_passes == 1

    def test_hidden_cot_config_two_passes(self) -> None:
        """HIDDEN_COT 전략은 2회 pass다."""
        assert DEFAULT_CONFIGS[ThinkingStrategy.HIDDEN_COT].max_passes == 2

    def test_self_reflect_config_three_passes(self) -> None:
        """SELF_REFLECT 전략은 3회 pass다."""
        assert DEFAULT_CONFIGS[ThinkingStrategy.SELF_REFLECT].max_passes == 3


# ─────────────────────────────────────────────
# ThinkingCache 테스트
# ─────────────────────────────────────────────
class TestThinkingCache:
    """ThinkingCache의 LRU + TTL 동작을 검증한다."""

    def _make_result(self, text: str = "test") -> ThinkingResult:
        """테스트용 ThinkingResult를 생성한다."""
        return ThinkingResult(
            strategy=ThinkingStrategy.DIRECT,
            response=text,
            thinking_text="",
            passes=1,
            elapsed_seconds=0.1,
            score=0.1,
        )

    def test_make_key_deterministic(self) -> None:
        """동일 메시지는 항상 같은 키를 반환한다."""
        key1 = ThinkingCache.make_key("hello world")
        key2 = ThinkingCache.make_key("hello world")
        assert key1 == key2

    def test_make_key_strip_whitespace(self) -> None:
        """앞뒤 공백이 정규화된다."""
        key1 = ThinkingCache.make_key("hello")
        key2 = ThinkingCache.make_key("  hello  ")
        assert key1 == key2

    def test_make_key_different_messages(self) -> None:
        """다른 메시지는 다른 키를 반환한다."""
        key1 = ThinkingCache.make_key("hello")
        key2 = ThinkingCache.make_key("world")
        assert key1 != key2

    def test_put_and_get(self) -> None:
        """저장한 항목을 조회할 수 있다."""
        cache = ThinkingCache(max_size=10)
        result = self._make_result("cached")
        cache.put("key1", result)
        retrieved = cache.get("key1")
        assert retrieved is not None
        assert retrieved.response == "cached"

    def test_get_miss_returns_none(self) -> None:
        """존재하지 않는 키는 None을 반환한다."""
        cache = ThinkingCache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self) -> None:
        """max_size 초과 시 가장 오래된 항목이 제거된다."""
        cache = ThinkingCache(max_size=3)
        for i in range(4):
            cache.put(f"key{i}", self._make_result(f"val{i}"))

        # key0이 제거되었어야 한다
        assert cache.get("key0") is None
        # key1~key3은 남아있다
        assert cache.get("key1") is not None
        assert cache.get("key3") is not None

    def test_ttl_expiration(self) -> None:
        """TTL이 만료되면 None을 반환한다."""
        # 매우 짧은 TTL (0.05초)
        cache = ThinkingCache(max_size=10, ttl_seconds=0.05)
        cache.put("key1", self._make_result())
        # TTL 만료 대기
        time.sleep(0.1)
        assert cache.get("key1") is None

    def test_clear_empties_cache(self) -> None:
        """clear()는 캐시를 완전히 비운다."""
        cache = ThinkingCache()
        cache.put("key1", self._make_result())
        cache.put("key2", self._make_result())
        cache.clear()
        assert cache.size == 0
        assert cache.get("key1") is None

    def test_stats_tracking(self) -> None:
        """hit/miss 통계가 정확히 추적된다."""
        cache = ThinkingCache()
        cache.put("key1", self._make_result())
        cache.get("key1")  # hit
        cache.get("key2")  # miss

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_size_property(self) -> None:
        """size 프로퍼티가 현재 항목 수를 반환한다."""
        cache = ThinkingCache()
        assert cache.size == 0
        cache.put("key1", self._make_result())
        assert cache.size == 1


# ─────────────────────────────────────────────
# HiddenCoTEngine 테스트
# ─────────────────────────────────────────────
class TestHiddenCoTEngine:
    """HiddenCoTEngine의 2-pass 실행을 검증한다."""

    @pytest.mark.asyncio
    async def test_run_returns_thinking_result(self) -> None:
        """run()이 ThinkingResult를 반환한다."""
        model = MockModelProvider(responses=["분석 결과", "최종 응답"])
        engine = HiddenCoTEngine()
        result = await engine.run("테스트 질문", None, model)

        assert isinstance(result, ThinkingResult)
        assert result.strategy == ThinkingStrategy.HIDDEN_COT
        assert result.passes == 2

    @pytest.mark.asyncio
    async def test_run_two_pass_execution(self) -> None:
        """2회 모델 호출이 실행된다 (Pass 1: 분석, Pass 2: 응답)."""
        model = MockModelProvider(responses=["분석 내용", "최종 답변"])
        engine = HiddenCoTEngine()
        await engine.run("복잡한 질문", None, model)

        # 모델이 2회 호출되었는지 확인
        assert model._call_count == 2

    @pytest.mark.asyncio
    async def test_run_thinking_text_contains_analysis(self) -> None:
        """thinking_text에 Pass 1 분석 내용이 포함된다."""
        model = MockModelProvider(responses=["심층 분석 결과입니다", "최종 응답"])
        engine = HiddenCoTEngine()
        result = await engine.run("질문", None, model)

        assert result.thinking_text == "심층 분석 결과입니다"

    @pytest.mark.asyncio
    async def test_run_response_from_pass_2(self) -> None:
        """response는 Pass 2의 결과다."""
        model = MockModelProvider(responses=["분석", "Pass 2 응답"])
        engine = HiddenCoTEngine()
        result = await engine.run("질문", None, model)

        assert result.response == "Pass 2 응답"

    @pytest.mark.asyncio
    async def test_run_with_context(self) -> None:
        """컨텍스트가 전달되어도 정상 동작한다."""
        model = MockModelProvider(responses=["분석", "응답"])
        engine = HiddenCoTEngine()
        context = [Message.user("이전 질문"), Message.assistant("이전 답변")]
        result = await engine.run("새 질문", context, model)

        assert result.passes == 2

    @pytest.mark.asyncio
    async def test_run_elapsed_seconds_tracked(self) -> None:
        """소요 시간이 기록된다."""
        model = MockModelProvider(responses=["분석", "응답"])
        engine = HiddenCoTEngine()
        result = await engine.run("질문", None, model)

        assert result.elapsed_seconds >= 0.0


# ─────────────────────────────────────────────
# SelfReflectionEngine 테스트
# ─────────────────────────────────────────────
class TestSelfReflectionEngine:
    """SelfReflectionEngine의 3-pass 실행을 검증한다."""

    @pytest.mark.asyncio
    async def test_run_returns_thinking_result(self) -> None:
        """run()이 ThinkingResult를 반환한다."""
        model = MockModelProvider(responses=["분석", "초기 응답", "검증 완료"])
        engine = SelfReflectionEngine()
        result = await engine.run("질문", None, model)

        assert isinstance(result, ThinkingResult)
        assert result.strategy == ThinkingStrategy.SELF_REFLECT
        assert result.passes == 3

    @pytest.mark.asyncio
    async def test_run_three_pass_execution(self) -> None:
        """3회 모델 호출이 실행된다."""
        model = MockModelProvider(responses=["분석", "응답", "검증"])
        engine = SelfReflectionEngine()
        await engine.run("질문", None, model)

        assert model._call_count == 3

    @pytest.mark.asyncio
    async def test_run_response_from_pass_3(self) -> None:
        """response는 Pass 3(검증) 결과다."""
        model = MockModelProvider(responses=["분석", "초기", "검증된 최종 응답"])
        engine = SelfReflectionEngine()
        result = await engine.run("질문", None, model)

        assert result.response == "검증된 최종 응답"

    @pytest.mark.asyncio
    async def test_run_thinking_text_contains_all_passes(self) -> None:
        """thinking_text에 3개 pass 정보가 모두 포함된다."""
        model = MockModelProvider(responses=["심층분석", "초기응답", "최종응답"])
        engine = SelfReflectionEngine()
        result = await engine.run("질문", None, model)

        assert "Pass 1" in result.thinking_text
        assert "Pass 2" in result.thinking_text
        assert "Pass 3" in result.thinking_text

    @pytest.mark.asyncio
    async def test_run_with_context(self) -> None:
        """컨텍스트와 함께 정상 동작한다."""
        model = MockModelProvider(responses=["분석", "응답", "검증"])
        engine = SelfReflectionEngine()
        context = [Message.user("이전"), Message.assistant("답변")]
        result = await engine.run("새 질문", context, model)

        assert result.passes == 3


# ─────────────────────────────────────────────
# ThinkingOrchestrator 테스트
# ─────────────────────────────────────────────
class TestThinkingOrchestrator:
    """ThinkingOrchestrator의 전략 선택 + 실행 통합을 검증한다."""

    @pytest.mark.asyncio
    async def test_think_simple_message_uses_direct(self) -> None:
        """단순 메시지는 DIRECT 전략을 사용한다."""
        model = MockModelProvider(responses=["간단한 답변"])
        orch = ThinkingOrchestrator(model)
        result = await orch.think("hi")

        assert result.strategy == ThinkingStrategy.DIRECT
        assert result.passes == 1

    @pytest.mark.asyncio
    async def test_think_moderate_message_uses_hidden_cot(self) -> None:
        """중간 복잡도 메시지는 HIDDEN_COT 이상의 전략을 사용한다."""
        model = MockModelProvider(responses=["분석", "응답", "검증"])
        orch = ThinkingOrchestrator(model)
        # "implement"(0.2) + "create"(0.1) = 0.3 → HIDDEN_COT
        result = await orch.think("implement and create a new module")

        assert result.strategy == ThinkingStrategy.HIDDEN_COT
        assert result.passes == 2

    @pytest.mark.asyncio
    async def test_think_cache_hit(self) -> None:
        """동일 메시지의 두 번째 호출은 캐시에서 반환한다."""
        model = MockModelProvider(responses=["응답1", "분석", "응답2"])
        orch = ThinkingOrchestrator(model)

        result1 = await orch.think("hello")
        result2 = await orch.think("hello")

        # 두 번째 호출은 캐시 히트이므로 같은 결과
        assert result1.response == result2.response
        # 모델은 1회만 호출되어야 한다 (캐시 히트)
        assert model._call_count == 1

    @pytest.mark.asyncio
    async def test_think_different_messages_no_cache(self) -> None:
        """다른 메시지는 캐시 미스로 별도 실행된다."""
        model = MockModelProvider(responses=["응답1", "응답2"])
        orch = ThinkingOrchestrator(model)

        await orch.think("hello")
        await orch.think("world")

        assert model._call_count == 2

    @pytest.mark.asyncio
    async def test_think_score_recorded(self) -> None:
        """결과에 복잡도 스코어가 기록된다."""
        model = MockModelProvider(responses=["응답"])
        orch = ThinkingOrchestrator(model)
        result = await orch.think("implement a complex system")

        assert result.score >= 0.0
        assert result.score <= 1.0

    @pytest.mark.asyncio
    async def test_cache_stats(self) -> None:
        """캐시 통계가 올바르게 반환된다."""
        model = MockModelProvider(responses=["응답"])
        orch = ThinkingOrchestrator(model)

        await orch.think("hello")
        await orch.think("hello")  # 캐시 히트

        stats = orch.cache_stats
        assert stats["hits"] == 1
        assert stats["size"] == 1

    @pytest.mark.asyncio
    async def test_clear_cache(self) -> None:
        """clear_cache()가 캐시를 비운다."""
        model = MockModelProvider(responses=["응답1", "응답2"])
        orch = ThinkingOrchestrator(model)

        await orch.think("hello")
        orch.clear_cache()
        assert orch.cache_stats["size"] == 0

        # 캐시 클리어 후 같은 메시지도 다시 실행
        await orch.think("hello")
        assert model._call_count == 2
