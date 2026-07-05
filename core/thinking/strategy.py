"""
사고 전략(Thinking Strategy) 정의 모듈.

이 파일 하나가 하는 일은 딱 두 가지다.
  1) "복잡도 스코어(0.0~1.0)" 를 받아 어떤 사고 전략을 쓸지 결정한다.
  2) 각 전략이 실제로 어떻게 동작할지(호출 횟수, temperature, 프롬프트 지시문)를
     기본값으로 정의해 둔다.

여기서 말하는 "사고 전략"이란, 모델이 사용자 질문에 답하기 전에 내부적으로
얼마나 깊게 생각(분석·검증)할지를 정하는 정책이다. 쉬운 질문에 무겁게 생각하면
느리고 비싸고, 어려운 질문에 가볍게 답하면 품질이 떨어진다. 그래서 질문의
복잡도에 맞춰 사고량을 조절한다.

지원하는 4가지 전략(복잡도가 높아질수록 더 많이 생각한다):
  - DIRECT       : 스코어 < 0.3        — 단순 질문. 추가 사고 없이 바로 응답 (1-pass)
  - HIDDEN_COT   : 0.3 <= 스코어 < 0.6 — 내부에서 한번 분석한 뒤 응답 (2-pass)
  - SELF_REFLECT : 0.6 <= 스코어 < 0.8 — 분석 → 응답 → 검증까지 (3-pass)
  - MULTI_AGENT  : 스코어 >= 0.8       — 다중 에이전트로 분해 (Phase 5.0b에서 구현)

주요 구성 요소:
  - ThinkingStrategy : 위 4가지 전략을 나타내는 문자열 Enum
  - StrategyConfig   : 각 전략의 실행 파라미터를 담는 불변(frozen) dataclass
  - select_strategy(): 복잡도 스코어 → 전략을 매핑하는 순수 함수
  - DEFAULT_CONFIGS  : 전략별 기본 StrategyConfig 매핑(설정 파일에서 오버라이드 가능)

이 모듈은 외부 의존성이 표준 라이브러리(dataclasses, enum)뿐이라 매우 가볍고,
사고 오케스트레이터(core/thinking/orchestrator 등)가 이 값들을 읽어 실제
LLM 호출 흐름을 제어한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ThinkingStrategy(str, Enum):
    """
    사고 전략 열거형(Enum).

    복잡도 스코어에 따라 이 중 정확히 하나가 select_strategy()로 선택된다.
    str 을 함께 상속하므로 각 멤버는 문자열처럼 다룰 수 있다. 예를 들어
    ThinkingStrategy.DIRECT == "direct" 가 참이고, JSON/YAML 직렬화나 로그
    출력 시 "direct" 같은 사람이 읽기 쉬운 값이 그대로 쓰인다. (설정 파일에서
    문자열로 전략을 지정하고 다시 Enum으로 되돌릴 때 편리하다.)

    각 멤버의 값(오른쪽 문자열)은 설정 파일·로그에서 쓰이는 안정적인 식별자이며,
    함부로 바꾸면 저장된 설정과 호환이 깨지므로 주의한다.
    """

    # 단순한 질문 — 별도 사고 단계 없이 곧바로 답변 (LLM 호출 1회)
    DIRECT = "direct"
    # 중간 난이도 — 내부에서 한번 분석한 뒤 답변. 분석은 사용자에게 감춤 (2-pass)
    HIDDEN_COT = "hidden_cot"
    # 복잡한 질문 — 분석 → 초기 응답 → 자기 검증까지 수행 (3-pass)
    SELF_REFLECT = "self_reflect"
    # 매우 복잡 — 문제를 하위 작업으로 쪼개 여러 전문 에이전트에 분배
    MULTI_AGENT = "multi_agent"


@dataclass(frozen=True)
class StrategyConfig:
    """
    하나의 사고 전략을 "어떻게 실행할지" 담는 설정 묶음.

    select_strategy()가 어떤 전략을 쓸지 '결정'한다면, 이 StrategyConfig는 그
    전략이 실제로 몇 번 LLM을 부를지, 얼마나 보수적으로 생성할지, 시스템
    프롬프트에 어떤 지시를 덧붙일지 같은 '구체적 실행 파라미터'를 정의한다.

    frozen=True 로 선언해 생성 후 필드를 바꿀 수 없는 불변 객체로 만든다. 사고
    설정은 요청 처리 도중 바뀌면 동작을 추적하기 어렵고 캐시·재현성이 깨질 수
    있으므로, 실수로라도 런타임에 수정되지 않도록 막는 것이다. 값을 바꾸려면
    새 인스턴스를 만든다.

    Attributes:
        name: 이 설정이 적용되는 사고 전략(ThinkingStrategy)
        max_passes: 이 전략이 허용하는 최대 LLM 호출 횟수. 사고 단계가 이 횟수를
            넘지 않도록 상한을 건다. (DIRECT=1, HIDDEN_COT=2, SELF_REFLECT=3,
            MULTI_AGENT=5 — 에이전트 간 왕복까지 포함해 넉넉히 잡음)
        temperature: 사고/생성 단계에서 쓰는 샘플링 온도. 값이 낮을수록 결정적이고
            정확한 방향, 높을수록 다양하고 창의적인 방향으로 생성된다. 분석·검증이
            중요한 전략일수록 낮게 잡는다.
        system_prompt_suffix: 시스템 프롬프트 끝에 덧붙일 사고 지시문. 모델에게
            "이번엔 이런 방식으로 생각하라"고 알려 주는 문자열이며, 전략마다 다르다.
            DIRECT는 빈 문자열이라 아무 지시도 추가하지 않는다.
    """

    name: ThinkingStrategy
    max_passes: int
    temperature: float
    system_prompt_suffix: str


def select_strategy(score: float) -> ThinkingStrategy:
    """
    복잡도 스코어를 받아 어떤 사고 전략을 쓸지 결정하는 순수 함수.

    복잡도 판정 로직(질문을 얼마나 어렵게 볼지)은 다른 모듈(assessor 등)이
    맡고, 이 함수는 그 결과 스코어를 구간별로 나눠 전략에 매핑하는 역할만 한다.
    입력이 같으면 항상 같은 결과를 주는 순수 함수라 테스트하기도 쉽다.

    임계값이 커질수록(=질문이 어려울수록) 더 무겁게 생각하는 전략이 선택된다.
    경계는 '이상/미만'을 명확히 해 두었다. 예: 0.3은 DIRECT가 아니라 HIDDEN_COT,
    0.8은 SELF_REFLECT가 아니라 MULTI_AGENT다.

    Args:
        score: 0.0~1.0 범위의 복잡도 스코어. 클수록 더 복잡한 질문을 뜻한다.
            (범위를 벗어난 값에 대한 별도 검증은 하지 않으며, 음수는 DIRECT로,
            1.0 초과는 MULTI_AGENT로 자연스럽게 떨어진다.)

    Returns:
        구간에 대응하는 ThinkingStrategy.

    구간별 매핑:
      - score < 0.3          → DIRECT
      - 0.3 <= score < 0.6   → HIDDEN_COT
      - 0.6 <= score < 0.8   → SELF_REFLECT
      - score >= 0.8         → MULTI_AGENT
    """
    # 아래로 갈수록 임계값이 커지므로, 위에서부터 순서대로 비교해 처음 걸리는
    # 구간을 고른다. (0.3 미만이 먼저 걸러지고, 그 다음 0.6, 0.8 순서)
    if score < 0.3:
        # 가장 쉬운 구간 — 추가 사고 없이 바로 답한다.
        return ThinkingStrategy.DIRECT
    elif score < 0.6:
        # 중간 난이도 — 내부 분석 1회를 거친 뒤 답한다.
        return ThinkingStrategy.HIDDEN_COT
    elif score < 0.8:
        # 어려운 구간 — 분석·응답·검증의 3-pass를 돈다.
        return ThinkingStrategy.SELF_REFLECT
    else:
        # 가장 어려운 구간 — 다중 에이전트로 문제를 분해한다.
        return ThinkingStrategy.MULTI_AGENT


# ─── 전략별 기본 설정 ───
# 각 사고 전략에 대응하는 StrategyConfig의 '기본값' 테이블이다.
# 사고 오케스트레이터는 select_strategy()로 전략을 고른 뒤, 이 딕셔너리에서
# 해당 전략의 실행 파라미터(호출 횟수·temperature·프롬프트 지시문)를 꺼내 쓴다.
# 여기 값들은 어디까지나 기본값이며, 설정 파일(config/nexus_config.yaml)에서
# 배포 환경에 맞게 오버라이드할 수 있다. (코드 수정 없이 튜닝하기 위한 구조)
DEFAULT_CONFIGS: dict[ThinkingStrategy, StrategyConfig] = {
    # DIRECT: 가장 가벼운 전략. temperature 0.7로 자연스러운 대화체를 허용하되
    # 사고 지시문은 붙이지 않아 모델이 곧바로 답하게 한다.
    ThinkingStrategy.DIRECT: StrategyConfig(
        name=ThinkingStrategy.DIRECT,
        max_passes=1,  # LLM 호출 1회 — 사고 단계 없음
        temperature=0.7,
        system_prompt_suffix="",  # 빈 문자열 — 프롬프트에 아무 지시도 덧붙이지 않음
    ),
    # HIDDEN_COT: 답하기 전 내부에서 한번 분석(2-pass). 분석 결과는 사용자에게
    # 노출하지 않고 최종 답변만 보여 준다. 분석 정확도를 위해 온도를 낮춘다.
    ThinkingStrategy.HIDDEN_COT: StrategyConfig(
        name=ThinkingStrategy.HIDDEN_COT,
        max_passes=2,  # 분석 1회 + 응답 1회
        temperature=0.4,  # 분석 단계는 낮은 temperature로 정확도 우선
        system_prompt_suffix=(
            "\n\n[사고 모드: Hidden CoT]\n"
            "먼저 문제를 분석한 후 응답하세요. "
            "분석 내용은 사용자에게 노출되지 않습니다."
        ),
    ),
    # SELF_REFLECT: 분석 → 초기 응답 → 검증의 3-pass. 마지막에 스스로 답을
    # 다시 점검·수정하므로, 흔들림을 줄이려 온도를 가장 낮게(0.3) 잡는다.
    ThinkingStrategy.SELF_REFLECT: StrategyConfig(
        name=ThinkingStrategy.SELF_REFLECT,
        max_passes=3,  # 분석 + 응답 + 검증
        temperature=0.3,  # 검증 단계까지 있어 더 낮은 temperature로 안정성 확보
        system_prompt_suffix=(
            "\n\n[사고 모드: Self-Reflect]\n"
            "1단계: 심층 분석, 2단계: 초기 응답 생성, "
            "3단계: 응답 검증 및 수정. "
            "모든 사고 과정은 내부적으로 처리됩니다."
        ),
    ),
    # MULTI_AGENT: 문제를 하위 작업으로 나눠 여러 전문 에이전트에게 위임한다.
    # 에이전트 간 왕복이 있어 호출 상한을 5회로 넉넉히 두고, 분해·조합에는
    # 어느 정도 유연성이 필요해 온도를 중간값(0.5)으로 둔다.
    # (실제 다중 에이전트 실행 로직은 Phase 5.0b에서 구현 예정 — 여기선 설정만.)
    ThinkingStrategy.MULTI_AGENT: StrategyConfig(
        name=ThinkingStrategy.MULTI_AGENT,
        max_passes=5,  # 에이전트 간 왕복 포함해 최대 5회
        temperature=0.5,
        system_prompt_suffix=(
            "\n\n[사고 모드: Multi-Agent]\n"
            "복잡한 문제를 하위 작업으로 분해하여 "
            "각 전문 에이전트에게 위임합니다."
        ),
    ),
}
