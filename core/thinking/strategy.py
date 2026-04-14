"""
사고 전략 정의 — 복잡도 스코어에 따라 적용할 사고 전략을 결정한다.

4가지 전략:
  - DIRECT: 스코어 < 0.3 — 단순 질문, 추가 사고 불필요
  - HIDDEN_COT: 0.3 <= 스코어 < 0.6 — 2-pass 분석 후 응답
  - SELF_REFLECT: 0.6 <= 스코어 < 0.8 — 3-pass 분석+응답+검증
  - MULTI_AGENT: 스코어 >= 0.8 — 다중 에이전트 협업 (Phase 5.0b에서 구현)

각 전략에는 기본 설정(StrategyConfig)이 정의되어 있어
엔진이 일관된 방식으로 동작한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThinkingStrategy(str, Enum):
    """
    사고 전략 열거형.
    복잡도 스코어에 따라 하나가 선택된다.
    """

    DIRECT = "direct"  # 단순: 추가 사고 없이 바로 응답
    HIDDEN_COT = "hidden_cot"  # 중간: 내부 분석 후 응답 (2-pass)
    SELF_REFLECT = "self_reflect"  # 복잡: 분석 → 응답 → 검증 (3-pass)
    MULTI_AGENT = "multi_agent"  # 매우 복잡: 다중 에이전트 분해


@dataclass(frozen=True)
class StrategyConfig:
    """
    각 사고 전략의 실행 설정.
    frozen=True로 불변 — 런타임 중 변경하지 않는다.

    Attributes:
        name: 적용할 사고 전략
        max_passes: 최대 LLM 호출 횟수 (DIRECT=1, HIDDEN_COT=2, SELF_REFLECT=3)
        temperature: 사고 단계의 temperature (낮을수록 정확, 높을수록 창의적)
        system_prompt_suffix: 시스템 프롬프트에 추가할 사고 지시 문자열
    """

    name: ThinkingStrategy
    max_passes: int
    temperature: float
    system_prompt_suffix: str


def select_strategy(score: float) -> ThinkingStrategy:
    """
    복잡도 스코어에 따라 사고 전략을 선택한다.

    Args:
        score: 0.0~1.0 범위의 복잡도 스코어

    Returns:
        선택된 ThinkingStrategy

    임계값:
      - < 0.3  → DIRECT
      - < 0.6  → HIDDEN_COT
      - < 0.8  → SELF_REFLECT
      - >= 0.8 → MULTI_AGENT
    """
    if score < 0.3:
        return ThinkingStrategy.DIRECT
    elif score < 0.6:
        return ThinkingStrategy.HIDDEN_COT
    elif score < 0.8:
        return ThinkingStrategy.SELF_REFLECT
    else:
        return ThinkingStrategy.MULTI_AGENT


# ─── 전략별 기본 설정 ───
# 각 전략에 매핑된 StrategyConfig를 정의한다.
# 설정 파일(config/nexus_config.yaml)에서 오버라이드할 수 있다.
DEFAULT_CONFIGS: dict[ThinkingStrategy, StrategyConfig] = {
    ThinkingStrategy.DIRECT: StrategyConfig(
        name=ThinkingStrategy.DIRECT,
        max_passes=1,
        temperature=0.7,
        system_prompt_suffix="",  # 추가 지시 없음 — 모델이 바로 응답
    ),
    ThinkingStrategy.HIDDEN_COT: StrategyConfig(
        name=ThinkingStrategy.HIDDEN_COT,
        max_passes=2,
        temperature=0.4,  # 분석 단계는 낮은 temperature로 정확도 우선
        system_prompt_suffix=(
            "\n\n[사고 모드: Hidden CoT]\n"
            "먼저 문제를 분석한 후 응답하세요. "
            "분석 내용은 사용자에게 노출되지 않습니다."
        ),
    ),
    ThinkingStrategy.SELF_REFLECT: StrategyConfig(
        name=ThinkingStrategy.SELF_REFLECT,
        max_passes=3,
        temperature=0.3,  # 검증 단계는 더 낮은 temperature
        system_prompt_suffix=(
            "\n\n[사고 모드: Self-Reflect]\n"
            "1단계: 심층 분석, 2단계: 초기 응답 생성, "
            "3단계: 응답 검증 및 수정. "
            "모든 사고 과정은 내부적으로 처리됩니다."
        ),
    ),
    ThinkingStrategy.MULTI_AGENT: StrategyConfig(
        name=ThinkingStrategy.MULTI_AGENT,
        max_passes=5,  # 에이전트 간 왕복 포함
        temperature=0.5,
        system_prompt_suffix=(
            "\n\n[사고 모드: Multi-Agent]\n"
            "복잡한 문제를 하위 작업으로 분해하여 "
            "각 전문 에이전트에게 위임합니다."
        ),
    ),
}
