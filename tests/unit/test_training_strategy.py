"""
training/strategy.py 단위 테스트 — 5-Phase 전략 관리자 검증.

Phase 전이 조건, can_advance 판단, advance 동작, Phase별 설정 반환을 테스트한다.
"""

from __future__ import annotations

import pytest

from training.strategy import (
    _PHASE_CONFIGS,
    PHASE_TRANSITIONS,
    PhaseTransitionCriteria,
    TrainingPhase,
    TrainingStrategy,
)


# ─────────────────────────────────────────────
# 기본 상태 테스트
# ─────────────────────────────────────────────
class TestTrainingPhase:
    """TrainingPhase 열거형 테스트."""

    def test_phase_values_are_sequential(self):
        """Phase 값이 0~4 순서대로 정의되어 있는지 확인한다."""
        assert TrainingPhase.PROMPT_ENGINEERING == 0
        assert TrainingPhase.BOOTSTRAP_LORA == 1
        assert TrainingPhase.SELF_DATA_QLORA == 2
        assert TrainingPhase.REASONING_FINETUNE == 3
        assert TrainingPhase.DOMAIN_FINETUNE == 4

    def test_phase_count_is_five(self):
        """Phase가 정확히 5개인지 확인한다."""
        assert len(TrainingPhase) == 5


class TestPhaseTransitionCriteria:
    """PhaseTransitionCriteria 테스트."""

    def test_criteria_is_frozen(self):
        """PhaseTransitionCriteria가 불변(frozen)인지 확인한다."""
        criteria = PhaseTransitionCriteria(
            min_eval_accuracy=0.6,
            min_data_count=500,
            min_eval_samples=50,
            required_approval=False,
            max_regression_pct=5.0,
        )
        with pytest.raises(AttributeError):
            criteria.min_eval_accuracy = 0.9  # type: ignore[misc]

    def test_all_phases_except_last_have_transition_criteria(self):
        """마지막 Phase를 제외한 모든 Phase에 전이 기준이 정의되어 있는지."""
        for phase in TrainingPhase:
            if phase == TrainingPhase.DOMAIN_FINETUNE:
                assert phase not in PHASE_TRANSITIONS
            else:
                assert phase in PHASE_TRANSITIONS


# ─────────────────────────────────────────────
# TrainingStrategy 테스트
# ─────────────────────────────────────────────
class TestTrainingStrategy:
    """TrainingStrategy 클래스 테스트."""

    def test_initial_phase_is_prompt_engineering(self):
        """초기 Phase가 PROMPT_ENGINEERING인지 확인한다."""
        strategy = TrainingStrategy()
        assert strategy.current_phase == TrainingPhase.PROMPT_ENGINEERING

    def test_can_advance_accuracy_insufficient(self):
        """정확도가 기준에 미달하면 진행 불가."""
        strategy = TrainingStrategy()
        can, reason = strategy.can_advance(
            {
                "accuracy": 0.3,  # 기준: 0.6
                "data_count": 1000,
                "eval_samples": 100,
                "regression_pct": 0.0,
            }
        )
        assert can is False
        assert "정확도 부족" in reason

    def test_can_advance_data_count_insufficient(self):
        """데이터 수가 기준에 미달하면 진행 불가."""
        strategy = TrainingStrategy()
        can, reason = strategy.can_advance(
            {
                "accuracy": 0.8,
                "data_count": 100,  # 기준: 500
                "eval_samples": 100,
                "regression_pct": 0.0,
            }
        )
        assert can is False
        assert "데이터 부족" in reason

    def test_can_advance_eval_samples_insufficient(self):
        """평가 샘플 수가 기준에 미달하면 진행 불가."""
        strategy = TrainingStrategy()
        can, reason = strategy.can_advance(
            {
                "accuracy": 0.8,
                "data_count": 1000,
                "eval_samples": 10,  # 기준: 50
                "regression_pct": 0.0,
            }
        )
        assert can is False
        assert "샘플 부족" in reason

    def test_can_advance_regression_too_high(self):
        """회귀율이 기준을 초과하면 진행 불가."""
        strategy = TrainingStrategy()
        can, reason = strategy.can_advance(
            {
                "accuracy": 0.8,
                "data_count": 1000,
                "eval_samples": 100,
                "regression_pct": 10.0,  # 기준: 5.0%
            }
        )
        assert can is False
        assert "회귀율 초과" in reason

    def test_can_advance_all_criteria_met(self):
        """모든 기준을 충족하면 진행 가능."""
        strategy = TrainingStrategy()
        can, reason = strategy.can_advance(
            {
                "accuracy": 0.8,
                "data_count": 1000,
                "eval_samples": 100,
                "regression_pct": 1.0,
            }
        )
        assert can is True
        assert "진행 가능" in reason

    def test_can_advance_requires_approval(self):
        """사용자 승인이 필요한 Phase에서 미승인 시 진행 불가."""
        strategy = TrainingStrategy(current_phase=TrainingPhase.SELF_DATA_QLORA)
        can, reason = strategy.can_advance(
            {
                "accuracy": 0.95,
                "data_count": 50000,
                "eval_samples": 1000,
                "regression_pct": 0.0,
                "approved": False,
            }
        )
        assert can is False
        assert "승인" in reason

    def test_can_advance_with_approval(self):
        """사용자 승인이 있으면 진행 가능."""
        strategy = TrainingStrategy(current_phase=TrainingPhase.SELF_DATA_QLORA)
        can, _ = strategy.can_advance(
            {
                "accuracy": 0.95,
                "data_count": 50000,
                "eval_samples": 1000,
                "regression_pct": 0.0,
                "approved": True,
            }
        )
        assert can is True

    def test_can_advance_final_phase_returns_false(self):
        """최종 Phase에서는 항상 진행 불가."""
        strategy = TrainingStrategy(current_phase=TrainingPhase.DOMAIN_FINETUNE)
        can, reason = strategy.can_advance(
            {
                "accuracy": 1.0,
                "data_count": 999999,
                "eval_samples": 99999,
                "regression_pct": 0.0,
                "approved": True,
            }
        )
        assert can is False
        assert "최종 Phase" in reason

    def test_advance_increments_phase(self):
        """advance()가 Phase를 1 증가시키는지 확인한다."""
        strategy = TrainingStrategy()
        new_phase = strategy.advance()
        assert new_phase == TrainingPhase.BOOTSTRAP_LORA
        assert strategy.current_phase == TrainingPhase.BOOTSTRAP_LORA

    def test_advance_records_history(self):
        """advance()가 전이 이력을 기록하는지 확인한다."""
        strategy = TrainingStrategy()
        strategy.advance()
        assert len(strategy.history) == 1
        assert strategy.history[0]["from_phase"] == "PROMPT_ENGINEERING"
        assert strategy.history[0]["to_phase"] == "BOOTSTRAP_LORA"

    def test_advance_final_phase_raises_error(self):
        """최종 Phase에서 advance()를 호출하면 ValueError가 발생한다."""
        strategy = TrainingStrategy(current_phase=TrainingPhase.DOMAIN_FINETUNE)
        with pytest.raises(ValueError, match="최종 Phase"):
            strategy.advance()

    def test_advance_through_all_phases(self):
        """Phase 0부터 4까지 순차적으로 진행할 수 있는지 확인한다."""
        strategy = TrainingStrategy()
        for expected in [
            TrainingPhase.BOOTSTRAP_LORA,
            TrainingPhase.SELF_DATA_QLORA,
            TrainingPhase.REASONING_FINETUNE,
            TrainingPhase.DOMAIN_FINETUNE,
        ]:
            result = strategy.advance()
            assert result == expected
        assert len(strategy.history) == 4

    def test_get_config_for_phase_returns_method(self):
        """get_config_for_phase()가 method 키를 포함하는지 확인한다."""
        strategy = TrainingStrategy()
        config = strategy.get_config_for_phase(TrainingPhase.SELF_DATA_QLORA)
        assert config["method"] == "qlora"
        assert config["phase"] == "SELF_DATA_QLORA"

    def test_get_config_for_current_phase(self):
        """인자 없이 호출하면 현재 Phase의 설정을 반환한다."""
        strategy = TrainingStrategy()
        config = strategy.get_config_for_phase()
        assert config["method"] == "none"
        assert config["phase"] == "PROMPT_ENGINEERING"

    def test_all_phases_have_configs(self):
        """모든 Phase에 설정이 정의되어 있는지 확인한다."""
        for phase in TrainingPhase:
            assert phase in _PHASE_CONFIGS

    def test_history_returns_copy(self):
        """history 프로퍼티가 원본이 아닌 복사본을 반환하는지 확인한다."""
        strategy = TrainingStrategy()
        strategy.advance()
        history = strategy.history
        history.clear()
        assert len(strategy.history) == 1  # 원본은 변경되지 않음

    def test_get_transition_criteria_returns_none_for_final(self):
        """최종 Phase의 전이 기준은 None이다."""
        strategy = TrainingStrategy(current_phase=TrainingPhase.DOMAIN_FINETUNE)
        assert strategy.get_transition_criteria() is None

    def test_get_transition_criteria_returns_criteria(self):
        """일반 Phase의 전이 기준이 올바르게 반환되는지."""
        strategy = TrainingStrategy()
        criteria = strategy.get_transition_criteria()
        assert criteria is not None
        assert criteria.min_eval_accuracy == 0.6
