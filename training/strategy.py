"""
5-Phase 학습 전략 관리자 — 학습 단계(Phase)의 전이를 제어하는 모듈.

[이 파일이 하는 일]
Nexus는 로컬 LLM을 한 번에 크게 파인튜닝하지 않고, 5개의 단계(Phase 0~4)로
나누어 점진적으로 개선한다. 이 파일은 "지금 몇 단계인지"를 추적하고,
"다음 단계로 넘어가도 되는지"를 평가 지표로 판정하는 상태 관리자다.
Phase 0(프롬프트 엔지니어링, 가중치 변경 없음)에서 시작해
Phase 4(도메인 특화 파인튜닝)까지 조건을 만족할 때마다 한 칸씩 전진한다.

[왜 단계를 나누는가]
  - 데이터가 부족한 초기에는 모델을 건드리지 않고 프롬프트만 다듬는 게 안전하다.
  - 실 사용 데이터가 충분히 쌓이면 그때 LoRA/QLoRA로 실제 학습에 들어간다.
  - 각 단계의 전이 조건(정확도·데이터 수·회귀율·승인)을 자동 검증하여,
    데이터도 없이 성급하게 학습해 모델을 망치는 일을 막는다(fail-closed).

[구성 요소]
  - TrainingPhase        : 5단계를 나타내는 IntEnum (0~4).
  - PhaseTransitionCriteria : 다음 단계로 넘어가기 위한 최소 조건(frozen dataclass).
  - PHASE_TRANSITIONS    : Phase별 전이 기준 상수 테이블.
  - _PHASE_CONFIGS       : Phase별 학습 하이퍼파라미터(method/rank/lr 등) 상수 테이블.
  - TrainingStrategy     : 현재 Phase를 들고 전이 판정/실행/이력을 담당하는 핵심 클래스.

[사용 흐름]
  can_advance(eval_results) 로 전진 가능 여부를 먼저 확인하고,
  True일 때만 advance() 를 호출해 실제로 다음 Phase로 넘어간다.
  get_config_for_phase() 로 해당 Phase에서 쓸 학습 설정을 꺼내 트레이너에 전달한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger("nexus.training.strategy")


# ─────────────────────────────────────────────
# 학습 Phase 열거형
# ─────────────────────────────────────────────
class TrainingPhase(IntEnum):
    """
    5단계 학습 전략을 나타내는 열거형.

    IntEnum을 쓰는 이유: 값이 정수(0~4)라서 `current_phase + 1` 처럼
    산술 연산으로 "다음 단계"를 쉽게 계산할 수 있다(advance()에서 사용).
    또한 대소 비교(<, >)도 가능해 단계의 진행 정도를 직관적으로 다룰 수 있다.

    각 Phase는 이전 Phase의 성과 위에 쌓인다(누적적). 숫자가 커질수록
    필요한 데이터 양·GPU 리소스·학습 강도가 모두 늘어난다.
    """

    PROMPT_ENGINEERING = 0  # Phase 0: 가중치 변경 없이 프롬프트만으로 성능 최적화
    BOOTSTRAP_LORA = 1  # Phase 1: 합성(synthetic) 데이터로 LoRA를 처음 붙여 부트스트랩
    SELF_DATA_QLORA = 2  # Phase 2: 실제 사용 로그 데이터로 QLoRA 학습
    REASONING_FINETUNE = 3  # Phase 3: 추론(reasoning) 능력을 집중 강화하는 파인튜닝
    DOMAIN_FINETUNE = 4  # Phase 4: 특정 도메인에 특화시키는 최종 파인튜닝(마지막 단계)


# ─────────────────────────────────────────────
# Phase 전이 기준
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class PhaseTransitionCriteria:
    """
    한 Phase에서 다음 Phase로 넘어가기 위한 최소 조건 묶음.

    frozen=True 로 불변(immutable)이며, 아래 상수 테이블(PHASE_TRANSITIONS)에
    Phase별로 하나씩 정의해 둔다. can_advance()가 이 값들과 실제 평가 결과를
    하나하나 비교해 전이 허용 여부를 판단한다.

    핵심 규칙: 5개 조건을 "모두 동시에" 만족해야만 전이가 허용된다.
    fail-closed 원칙에 따라, 하나라도 미달이면 전진하지 않고 현재 Phase에 머문다.
    """

    min_eval_accuracy: float  # 통과에 필요한 최소 평가 정확도 (0.0~1.0 사이 비율)
    min_data_count: int  # 학습에 쌓여 있어야 하는 최소 데이터 개수
    min_eval_samples: int  # 정확도가 신뢰할 만하려면 필요한 최소 평가 샘플 수
    required_approval: bool  # True면 사람이 명시적으로 승인해야 전이 가능(위험 단계 보호)
    max_regression_pct: float  # 이전 대비 성능 하락 허용 상한(%) — 이보다 나빠지면 차단


# ─────────────────────────────────────────────
# Phase별 전이 기준 상수
# ─────────────────────────────────────────────
# 각 딕셔너리 키는 "현재 Phase"이고, 값은 "그 다음 Phase로 넘어가기 위한 조건"이다.
# 단계가 올라갈수록 요구 정확도/데이터 수는 높아지고, 허용 회귀율은 낮아진다(더 엄격).
# 뒤쪽 위험한 단계(2→3, 3→4)는 required_approval=True 로 사람 승인을 강제한다.
PHASE_TRANSITIONS: dict[TrainingPhase, PhaseTransitionCriteria] = {
    # Phase 0 → 1: 프롬프트만으로 기본 성능(정확도 0.6)을 확보하면 LoRA 학습에 진입
    TrainingPhase.PROMPT_ENGINEERING: PhaseTransitionCriteria(
        min_eval_accuracy=0.6,
        min_data_count=500,
        min_eval_samples=50,
        required_approval=False,
        max_regression_pct=5.0,
    ),
    # Phase 1 → 2: 부트스트랩 LoRA 기본 학습 완료 후 실 데이터 전환
    TrainingPhase.BOOTSTRAP_LORA: PhaseTransitionCriteria(
        min_eval_accuracy=0.7,
        min_data_count=2000,
        min_eval_samples=100,
        required_approval=False,
        max_regression_pct=3.0,
    ),
    # Phase 2 → 3: 실 데이터 QLoRA로 충분한 성능 확보 후 추론 강화
    TrainingPhase.SELF_DATA_QLORA: PhaseTransitionCriteria(
        min_eval_accuracy=0.8,
        min_data_count=5000,
        min_eval_samples=200,
        required_approval=True,
        max_regression_pct=2.0,
    ),
    # Phase 3 → 4: 추론 강화 완료 후 도메인 특화로 진입 (가장 엄격 + 사용자 승인 필수)
    TrainingPhase.REASONING_FINETUNE: PhaseTransitionCriteria(
        min_eval_accuracy=0.85,
        min_data_count=10000,
        min_eval_samples=500,
        required_approval=True,
        max_regression_pct=1.0,
    ),
    # Phase 4(DOMAIN_FINETUNE)는 최종 단계라 "다음"이 없다. 그래서 이 테이블에
    # 항목을 넣지 않는다. get(Phase4)는 None을 돌려주고, can_advance()는 이를
    # 최종 단계 도달로 해석해 더 이상 전진시키지 않는다(유지 모드).
}


# ─────────────────────────────────────────────
# Phase별 기본 학습 설정
# ─────────────────────────────────────────────
# Phase별 "학습 설정 프리셋" 테이블. get_config_for_phase()가 여기서 값을 꺼내
# 트레이너에 넘긴다. 단계가 올라갈수록 lora_rank/alpha(표현력)와 seq_length는
# 커지고, learning_rate(학습률)는 작아진다 — 뒤로 갈수록 크게 배우기보다
# 이미 학습된 것을 조심스럽게 미세 조정하기 때문이다.
# method: "none"(학습 안 함) / "lora" / "qlora"(4bit 양자화 LoRA, VRAM 절약).
_PHASE_CONFIGS: dict[TrainingPhase, dict[str, Any]] = {
    # Phase 0: 모델 가중치를 전혀 건드리지 않으므로 하이퍼파라미터가 없다.
    TrainingPhase.PROMPT_ENGINEERING: {
        "method": "none",
        "description": "프롬프트 엔지니어링만 사용 (모델 가중치 변경 없음)",
    },
    # Phase 1: 가장 가벼운 LoRA(rank 8). 합성 데이터로 감을 잡는 부트스트랩 단계.
    TrainingPhase.BOOTSTRAP_LORA: {
        "method": "lora",
        "lora_rank": 8,
        "lora_alpha": 16,
        "learning_rate": 3e-4,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "description": "합성 데이터로 LoRA 부트스트랩 학습",
    },
    # Phase 2: 실 사용 데이터로 QLoRA(4bit) 학습. rank 16으로 표현력을 키운다.
    TrainingPhase.SELF_DATA_QLORA: {
        "method": "qlora",
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 4096,
        "description": "실 사용 데이터로 QLoRA 학습",
    },
    # Phase 3: 추론 강화. rank 32로 더 키우고 epoch 5로 오래 학습해 추론력을 끌어올린다.
    TrainingPhase.REASONING_FINETUNE: {
        "method": "qlora",
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-4,
        "num_epochs": 5,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 4096,
        "description": "추론 능력 강화 QLoRA 학습",
    },
    # Phase 4: 도메인 특화. learning_rate를 가장 낮게(5e-5) 두고 epoch도 2로 짧게 잡아,
    # 기존 능력을 해치지 않으면서 특정 도메인 지식만 조심스럽게 얹는다.
    # seq_length 8192로 가장 길어 긴 도메인 문서도 다룰 수 있다.
    TrainingPhase.DOMAIN_FINETUNE: {
        "method": "qlora",
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 5e-5,
        "num_epochs": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 8192,
        "description": "도메인 특화 QLoRA 파인튜닝",
    },
}


# ─────────────────────────────────────────────
# TrainingStrategy 클래스
# ─────────────────────────────────────────────
@dataclass
class TrainingStrategy:
    """
    5-Phase 학습 전략의 실제 상태를 들고 있는 핵심 클래스.

    하는 일 세 가지:
      1) 현재 어느 Phase인지 추적한다(current_phase).
      2) 평가 결과를 받아 다음 Phase로 넘어가도 되는지 판정한다(can_advance).
         조건을 만족하면 실제로 한 칸 전진시킨다(advance).
      3) 언제 어느 단계에서 어느 단계로 넘어갔는지 이력을 남긴다(_history).
         이력을 남기는 이유는 나중에 "왜 이 시점에 학습했는가"를 추적
         (traceability)하기 위해서다.

    보통 이 객체 하나가 학습 파이프라인 전체에서 Phase 상태의 단일 출처
    (single source of truth) 역할을 한다.
    """

    # 학습은 항상 Phase 0(프롬프트 엔지니어링)에서 시작한다.
    current_phase: TrainingPhase = TrainingPhase.PROMPT_ENGINEERING
    # 전이 이력 저장소. 앞에 밑줄(_)이 붙은 건 "내부용"이라는 관례 표시이며,
    # 외부에서는 아래 history 프로퍼티(복사본)를 통해서만 읽도록 유도한다.
    _history: list[dict[str, Any]] = field(default_factory=list)

    def can_advance(self, eval_results: dict[str, Any]) -> tuple[bool, str]:
        """
        지금 Phase에서 다음 Phase로 넘어가도 되는지만 "판정"한다(상태는 바꾸지 않음).

        실제 전진은 이 메서드가 True를 준 뒤 호출자가 advance()를 불러야 일어난다.
        이렇게 판정과 실행을 분리한 이유: 로그만 찍고 넘어가지 않는 등 호출부가
        결정을 유연하게 다룰 수 있고, 판정 로직을 부작용 없이 테스트하기 쉽다.

        eval_results(평가 결과 딕셔너리)에서 읽는 키 — 없으면 안전한 기본값 사용:
          - accuracy: float — 평가 정확도 (0.0 ~ 1.0), 없으면 0.0
          - data_count: int — 현재까지 쌓인 학습 데이터 수, 없으면 0
          - eval_samples: int — 평가에 사용된 샘플 수, 없으면 0
          - regression_pct: float — 이전 Phase 대비 성능 회귀율(%), 없으면 0.0
          - approved: bool — 사용자 승인 여부(required_approval=True인 단계에서만 필요)

        Returns:
            (진행 가능 여부: bool, 사유 메시지: str) 튜플.
            False일 때 메시지에 "무엇이 왜 부족한지"를 담아 로그/UI에 그대로 쓸 수 있게 한다.
        """
        # Phase 4는 최종 단계라 다음이 없다 — 여기서 먼저 걸러 낸다.
        if self.current_phase == TrainingPhase.DOMAIN_FINETUNE:
            return False, "이미 최종 Phase(DOMAIN_FINETUNE)에 도달했습니다."

        # 현재 Phase에 대응하는 전이 기준을 꺼낸다. 정의가 없으면(이론상 발생하면
        # 안 되는 상황) 안전하게 전진을 막는다(fail-closed).
        criteria = PHASE_TRANSITIONS.get(self.current_phase)
        if criteria is None:
            return False, f"Phase {self.current_phase.name}에 대한 전이 기준이 정의되지 않았습니다."

        # 아래부터 5개 조건을 순서대로 검증한다. 하나라도 미달이면 즉시 False를
        # 반환하고 멈춘다(fail-closed) — 모든 조건을 통과해야만 마지막에 True가 된다.
        # (1) 평가 정확도가 기준 이상인가.
        accuracy = eval_results.get("accuracy", 0.0)
        if accuracy < criteria.min_eval_accuracy:
            return (
                False,
                f"평가 정확도 부족: {accuracy:.3f} < {criteria.min_eval_accuracy:.3f}",
            )

        # (2) 학습 데이터가 최소 요구량만큼 쌓였는가.
        data_count = eval_results.get("data_count", 0)
        if data_count < criteria.min_data_count:
            return (
                False,
                f"학습 데이터 부족: {data_count} < {criteria.min_data_count}",
            )

        # (3) 정확도 수치를 믿을 만큼 평가 샘플이 충분한가(샘플이 적으면 정확도가
        #     우연일 수 있으므로 별도로 확인한다).
        eval_samples = eval_results.get("eval_samples", 0)
        if eval_samples < criteria.min_eval_samples:
            return (
                False,
                f"평가 샘플 부족: {eval_samples} < {criteria.min_eval_samples}",
            )

        # (4) 이전 대비 성능 하락(회귀)이 허용 한계를 넘지 않았는가. 정확도가
        #     기준을 넘더라도 예전보다 크게 나빠졌다면 전진을 막는다.
        regression_pct = eval_results.get("regression_pct", 0.0)
        if regression_pct > criteria.max_regression_pct:
            return (
                False,
                f"회귀율 초과: {regression_pct:.1f}% > {criteria.max_regression_pct:.1f}%",
            )

        # (5) 승인이 필요한 위험 단계라면 사람이 승인했는지 확인한다.
        #     approved가 없거나 False면 지표를 다 만족해도 전진하지 않는다.
        if criteria.required_approval and not eval_results.get("approved", False):
            return (
                False,
                f"Phase {self.current_phase.name} → {TrainingPhase(self.current_phase + 1).name} "
                f"전이에는 사용자 승인이 필요합니다.",
            )

        # 5개 조건을 전부 통과 — 다음 Phase 이름을 담아 진행 가능(True)을 알린다.
        next_phase = TrainingPhase(self.current_phase + 1)
        return True, f"Phase {next_phase.name}으로 진행 가능합니다."

    def advance(self) -> TrainingPhase:
        """
        실제로 다음 Phase로 한 칸 전진시키고(current_phase를 갱신) 이력을 남긴다.

        주의: 이 메서드는 조건 검사를 하지 않는다. 반드시 can_advance()로 먼저
        확인해 True를 받은 뒤에 호출해야 한다. can_advance 없이 부르면 지표가
        미달인 상태로도 넘어가 버릴 수 있다. 단, 최종 Phase에서 부르는 실수만은
        여기서 ValueError로 막는다(안전장치).

        Returns:
            전진 후의 새 current_phase.
        """
        # 최종 단계에서 또 전진하려 하면 명확한 에러로 알린다(무한 전진 방지).
        if self.current_phase == TrainingPhase.DOMAIN_FINETUNE:
            raise ValueError(
                "이미 최종 Phase(DOMAIN_FINETUNE)에 도달하여 더 이상 진행할 수 없습니다."
            )

        # 로그/이력에 남기기 위해 이전 Phase를 먼저 보관한 뒤, 정수 +1로 다음 단계로 옮긴다.
        prev_phase = self.current_phase
        self.current_phase = TrainingPhase(self.current_phase + 1)

        # 언제 어디서 어디로 넘어갔는지 이력에 한 줄 추가한다(추적성 확보).
        self._history.append(
            {
                "from_phase": prev_phase.name,
                "to_phase": self.current_phase.name,
            }
        )

        logger.info(
            "Phase 전이: %s → %s",
            prev_phase.name,
            self.current_phase.name,
        )

        return self.current_phase

    def get_config_for_phase(self, phase: TrainingPhase | None = None) -> dict[str, Any]:
        """
        해당 Phase의 학습 설정(하이퍼파라미터 프리셋)을 꺼내 돌려준다.

        트레이너가 "이번 단계에서 어떤 method/rank/learning_rate로 학습할지"를
        물어볼 때 사용한다.

        Args:
            phase: 조회할 Phase. None이면 현재 Phase(current_phase)의 설정을 준다.

        Returns:
            학습 설정 딕셔너리. _PHASE_CONFIGS의 원본을 훼손하지 않도록 복사본을
            돌려주며, 어느 단계인지 알기 쉽게 phase(이름)와 phase_value(정수)를
            추가로 넣어 준다.
        """
        # phase 인자가 있으면 그걸, 없으면 현재 Phase를 대상으로 삼는다.
        target = phase if phase is not None else self.current_phase
        # .copy() 로 얕은 복사본을 만든다 — 호출자가 값을 바꿔도 원본 상수 테이블은 안전하다.
        config = _PHASE_CONFIGS.get(target, {}).copy()
        config["phase"] = target.name  # 사람이 읽을 단계 이름(예: "BOOTSTRAP_LORA")
        config["phase_value"] = int(target)  # 기계가 다루기 쉬운 단계 정수(0~4)
        return config

    @property
    def history(self) -> list[dict[str, Any]]:
        """
        지금까지의 전이 이력을 "복사본"으로 반환한다.

        list(...)로 새 리스트를 만들어 주기 때문에, 호출자가 결과를 수정해도
        내부 _history 원본은 훼손되지 않는다(캡슐화 보호).
        """
        return list(self._history)

    def get_transition_criteria(self) -> PhaseTransitionCriteria | None:
        """
        현재 Phase에서 다음 단계로 넘어가기 위한 전이 기준을 돌려준다.

        UI/로그에서 "지금 무엇을 더 채워야 전진하는지" 보여줄 때 유용하다.
        최종 Phase(DOMAIN_FINETUNE)에는 다음이 없으므로 None을 반환한다.
        """
        return PHASE_TRANSITIONS.get(self.current_phase)
