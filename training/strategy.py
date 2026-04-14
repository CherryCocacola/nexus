"""
5-Phase 학습 전략 관리자 — 학습 단계 전이를 제어한다.

Nexus의 학습은 5단계로 구성되며, 각 단계를 넘어가려면
평가 정확도, 데이터 수, 회귀율 등의 기준을 충족해야 한다.
Phase 0(프롬프트 엔지니어링)부터 시작하여
Phase 4(도메인 특화 파인튜닝)까지 점진적으로 모델을 개선한다.

왜 Phase 단계를 두는가:
  - 데이터가 부족한 초기에는 프롬프트 엔지니어링만으로 충분
  - 데이터가 충분히 쌓이면 LoRA/QLoRA로 전환하여 성능 개선
  - 각 단계의 전이 조건을 자동 검증하여 성급한 학습을 방지
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
    5단계 학습 전략.

    각 Phase는 이전 Phase의 기반 위에 구축된다.
    숫자가 클수록 더 많은 데이터와 리소스가 필요하다.
    """

    PROMPT_ENGINEERING = 0  # Phase 0: 프롬프트 엔지니어링만으로 최적화
    BOOTSTRAP_LORA = 1  # Phase 1: 합성 데이터로 LoRA 부트스트랩
    SELF_DATA_QLORA = 2  # Phase 2: 실 사용 데이터로 QLoRA 학습
    REASONING_FINETUNE = 3  # Phase 3: 추론 능력 강화 파인튜닝
    DOMAIN_FINETUNE = 4  # Phase 4: 도메인 특화 파인튜닝


# ─────────────────────────────────────────────
# Phase 전이 기준
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class PhaseTransitionCriteria:
    """
    다음 Phase로 진행하기 위한 최소 조건.

    모든 조건을 동시에 만족해야 전이가 허용된다.
    fail-closed 원칙: 기준을 충족하지 못하면 현재 Phase에 머문다.
    """

    min_eval_accuracy: float  # 최소 평가 정확도 (0.0 ~ 1.0)
    min_data_count: int  # 최소 학습 데이터 수
    min_eval_samples: int  # 최소 평가 샘플 수
    required_approval: bool  # 사용자 승인 필요 여부
    max_regression_pct: float  # 허용 최대 회귀율 (%) — 이전 대비 성능 하락 한계


# ─────────────────────────────────────────────
# Phase별 전이 기준 상수
# ─────────────────────────────────────────────
PHASE_TRANSITIONS: dict[TrainingPhase, PhaseTransitionCriteria] = {
    # Phase 0 → 1: 프롬프트만으로 기본 성능 확보 후 LoRA 진입
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
    # Phase 3 → 4: 추론 강화 완료 후 도메인 특화 (사용자 승인 필수)
    TrainingPhase.REASONING_FINETUNE: PhaseTransitionCriteria(
        min_eval_accuracy=0.85,
        min_data_count=10000,
        min_eval_samples=500,
        required_approval=True,
        max_regression_pct=1.0,
    ),
    # Phase 4는 최종 단계이므로 전이 기준 없음 (유지 모드)
}


# ─────────────────────────────────────────────
# Phase별 기본 학습 설정
# ─────────────────────────────────────────────
_PHASE_CONFIGS: dict[TrainingPhase, dict[str, Any]] = {
    TrainingPhase.PROMPT_ENGINEERING: {
        "method": "none",
        "description": "프롬프트 엔지니어링만 사용 (모델 가중치 변경 없음)",
    },
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
    5-Phase 학습 전략을 관리한다.

    현재 Phase를 추적하고, 평가 결과에 따라 다음 Phase로
    전이할 수 있는지 판단한다. 전이 이력을 기록하여
    학습 과정의 추적 가능성(traceability)을 보장한다.
    """

    current_phase: TrainingPhase = TrainingPhase.PROMPT_ENGINEERING
    _history: list[dict[str, Any]] = field(default_factory=list)

    def can_advance(self, eval_results: dict[str, Any]) -> tuple[bool, str]:
        """
        다음 Phase로 진행 가능한지 확인한다.

        eval_results에 필요한 키:
          - accuracy: float — 평가 정확도 (0.0 ~ 1.0)
          - data_count: int — 현재 학습 데이터 수
          - eval_samples: int — 평가에 사용된 샘플 수
          - regression_pct: float — 이전 Phase 대비 회귀율 (%)
          - approved: bool — 사용자 승인 여부 (required_approval이 True인 경우)

        Returns:
            (진행 가능 여부, 사유 메시지) 튜플
        """
        # Phase 4는 최종 단계 — 더 이상 진행할 수 없다
        if self.current_phase == TrainingPhase.DOMAIN_FINETUNE:
            return False, "이미 최종 Phase(DOMAIN_FINETUNE)에 도달했습니다."

        criteria = PHASE_TRANSITIONS.get(self.current_phase)
        if criteria is None:
            return False, f"Phase {self.current_phase.name}에 대한 전이 기준이 정의되지 않았습니다."

        # 각 기준을 순서대로 검증한다 (fail-closed)
        accuracy = eval_results.get("accuracy", 0.0)
        if accuracy < criteria.min_eval_accuracy:
            return (
                False,
                f"평가 정확도 부족: {accuracy:.3f} < {criteria.min_eval_accuracy:.3f}",
            )

        data_count = eval_results.get("data_count", 0)
        if data_count < criteria.min_data_count:
            return (
                False,
                f"학습 데이터 부족: {data_count} < {criteria.min_data_count}",
            )

        eval_samples = eval_results.get("eval_samples", 0)
        if eval_samples < criteria.min_eval_samples:
            return (
                False,
                f"평가 샘플 부족: {eval_samples} < {criteria.min_eval_samples}",
            )

        regression_pct = eval_results.get("regression_pct", 0.0)
        if regression_pct > criteria.max_regression_pct:
            return (
                False,
                f"회귀율 초과: {regression_pct:.1f}% > {criteria.max_regression_pct:.1f}%",
            )

        # 사용자 승인이 필요한 Phase에서 승인 여부 확인
        if criteria.required_approval and not eval_results.get("approved", False):
            return (
                False,
                f"Phase {self.current_phase.name} → {TrainingPhase(self.current_phase + 1).name} "
                f"전이에는 사용자 승인이 필요합니다.",
            )

        next_phase = TrainingPhase(self.current_phase + 1)
        return True, f"Phase {next_phase.name}으로 진행 가능합니다."

    def advance(self) -> TrainingPhase:
        """
        다음 Phase로 진행한다.

        주의: can_advance()로 먼저 확인한 후 호출해야 한다.
        최종 Phase에서 호출하면 ValueError를 발생시킨다.

        Returns:
            새로운 현재 Phase
        """
        if self.current_phase == TrainingPhase.DOMAIN_FINETUNE:
            raise ValueError(
                "이미 최종 Phase(DOMAIN_FINETUNE)에 도달하여 더 이상 진행할 수 없습니다."
            )

        prev_phase = self.current_phase
        self.current_phase = TrainingPhase(self.current_phase + 1)

        # 전이 이력 기록
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
        Phase별 학습 설정을 반환한다.

        Args:
            phase: 조회할 Phase. None이면 현재 Phase의 설정을 반환한다.

        Returns:
            학습 설정 딕셔너리 (method, lora_rank, learning_rate 등)
        """
        target = phase if phase is not None else self.current_phase
        config = _PHASE_CONFIGS.get(target, {}).copy()
        config["phase"] = target.name
        config["phase_value"] = int(target)
        return config

    @property
    def history(self) -> list[dict[str, Any]]:
        """전이 이력의 복사본을 반환한다 (외부 수정 방지)."""
        return list(self._history)

    def get_transition_criteria(self) -> PhaseTransitionCriteria | None:
        """현재 Phase의 전이 기준을 반환한다. 최종 Phase면 None."""
        return PHASE_TRANSITIONS.get(self.current_phase)
