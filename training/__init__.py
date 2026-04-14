"""
Training Pipeline — Phase 6.0 학습 파이프라인 모듈.

에어갭(폐쇄망) 환경에서 LoRA/QLoRA 기반 모델 학습을 관리한다.
5-Phase 전략(Prompt Engineering → Bootstrap LoRA → Self-Data QLoRA →
Reasoning Finetune → Domain Finetune)에 따라 점진적으로 모델을 개선한다.

의존성 방향: training → core (역방향 금지)
"""

from training.bootstrap_generator import BootstrapGenerator
from training.checkpoint_manager import CheckpointManager
from training.data_collector import DataCollector
from training.feedback_loop import FeedbackLoop
from training.strategy import TrainingPhase, TrainingStrategy
from training.trainer import LoRATrainer, TrainingConfig

__all__ = [
    "BootstrapGenerator",
    "CheckpointManager",
    "DataCollector",
    "FeedbackLoop",
    "LoRATrainer",
    "TrainingConfig",
    "TrainingPhase",
    "TrainingStrategy",
]
