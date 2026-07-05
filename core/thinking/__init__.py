"""
Thinking Engine 패키지 — 사용자 입력의 복잡도를 분석하고, 그 결과에 맞는
"사고 전략(thinking strategy)"을 골라 적용하는 모듈들의 묶음이다.

이 파일(__init__.py)은 실제 로직을 담지 않는다. thinking 패키지 안에 흩어져 있는
여러 하위 모듈(assessor, cache, hidden_cot, orchestrator, self_reflection, strategy)의
핵심 클래스/함수를 한곳에 모아 재노출(re-export)하는 "현관(진입점)" 역할만 한다.
덕분에 바깥 코드는 `from core.thinking import ThinkingOrchestrator` 처럼
내부 파일 구조를 몰라도 짧은 경로로 필요한 것만 가져다 쓸 수 있다.

배경(사양서 근거):
  - Phase 5.0a에서 구현된 Thinking Engine.
  - 사양서 Ch.11에 정의된 "복잡도 기반 사고 전략 시스템"을 코드로 옮긴 것.
  - 쉬운 질문은 곧바로 답하고(비용 절약), 어려운 질문일수록 더 깊게
    생각하는 전략을 쓰도록(정확도 확보) 입력 난이도에 따라 처리 강도를 조절한다.

이 패키지가 바깥에 노출하는 주요 컴포넌트:
  - ComplexityAssessor  : 입력의 복잡도를 0.0~1.0 스코어로 평가하는 판정기.
  - ThinkingStrategy    : 4가지 사고 전략을 나타내는 열거형.
                          (DIRECT=바로 답, HIDDEN_COT=숨은 사고사슬,
                           SELF_REFLECT=자기 성찰, MULTI_AGENT=다중 에이전트)
  - select_strategy     : 복잡도 스코어를 받아 어떤 전략을 쓸지 골라주는 함수.
  - StrategyConfig      : 각 전략의 세부 설정값(pass 횟수 등)을 담는 설정 객체.
  - DEFAULT_CONFIGS     : 전략별 기본 설정값 모음(사전).
  - ThinkingOrchestrator: 전략 선택 → 해당 엔진 실행까지 총괄하는 오케스트레이터.
  - ThinkingResult      : 오케스트레이터가 돌려주는 사고 결과 데이터.
  - HiddenCoTEngine     : 2-pass Hidden Chain-of-Thought(숨은 사고사슬) 엔진.
  - SelfReflectionEngine: 3-pass 자기 성찰(스스로 답을 되짚어 개선) 엔진.
  - ThinkingCache       : 동일한 문제를 다시 사고하지 않도록 결과를 저장하는 LRU 캐시.

작성자: 이현수 / 작성일: 2026-07-05
"""

# ---------------------------------------------------------------------------
# 하위 모듈에서 공개 API를 끌어와 패키지 최상위 이름으로 재노출한다.
# 아래 import들은 "이 패키지가 무엇을 제공하는지"를 한눈에 보여주는 목록이기도 하다.
# ---------------------------------------------------------------------------

# 입력 복잡도(0.0~1.0)를 평가하는 판정기.
from core.thinking.assessor import ComplexityAssessor

# 같은 문제를 재사고하지 않도록 결과를 캐싱하는 LRU 캐시.
from core.thinking.cache import ThinkingCache

# 2-pass Hidden Chain-of-Thought(숨은 사고사슬) 전략을 수행하는 엔진.
from core.thinking.hidden_cot import HiddenCoTEngine

# 전략 선택과 엔진 실행을 총괄하는 오케스트레이터와, 그 실행 결과 타입.
from core.thinking.orchestrator import ThinkingOrchestrator, ThinkingResult

# 3-pass 자기 성찰(스스로 답을 검토·개선) 전략 엔진.
from core.thinking.self_reflection import SelfReflectionEngine

# 전략 정의 관련 심볼들: 열거형, 설정 객체, 기본 설정 모음, 전략 선택 함수.
from core.thinking.strategy import (
    DEFAULT_CONFIGS,
    StrategyConfig,
    ThinkingStrategy,
    select_strategy,
)

# `from core.thinking import *` 로 가져올 수 있는 공개 심볼 목록.
# (와일드카드 import 범위를 명시적으로 고정해, 의도치 않은 이름 노출을 막는다.)
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
