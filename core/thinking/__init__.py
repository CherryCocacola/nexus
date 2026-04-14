"""
Thinking Engine — 사용자 입력의 복잡도를 분석하고 적절한 사고 전략을 적용한다.

Phase 5.0a: Thinking Engine 구현.
사양서 Ch.11에 정의된 복잡도 기반 사고 전략 시스템.

주요 컴포넌트:
  - ComplexityAssessor: 입력 복잡도 0.0~1.0 스코어 평가
  - ThinkingStrategy: 4가지 사고 전략 (DIRECT, HIDDEN_COT, SELF_REFLECT, MULTI_AGENT)
  - ThinkingOrchestrator: 전략 선택 + 엔진 실행 오케스트레이터
  - HiddenCoTEngine: 2-pass Hidden Chain-of-Thought
  - SelfReflectionEngine: 3-pass 자기 성찰
  - ThinkingCache: 동일 문제 재사고 방지 LRU 캐시
"""

from core.thinking.assessor import ComplexityAssessor
from core.thinking.cache import ThinkingCache
from core.thinking.hidden_cot import HiddenCoTEngine
from core.thinking.orchestrator import ThinkingOrchestrator, ThinkingResult
from core.thinking.self_reflection import SelfReflectionEngine
from core.thinking.strategy import (
    DEFAULT_CONFIGS,
    StrategyConfig,
    ThinkingStrategy,
    select_strategy,
)

__all__ = [
    "ComplexityAssessor",
    "DEFAULT_CONFIGS",
    "HiddenCoTEngine",
    "SelfReflectionEngine",
    "StrategyConfig",
    "ThinkingCache",
    "ThinkingOrchestrator",
    "ThinkingResult",
    "ThinkingStrategy",
    "select_strategy",
]
