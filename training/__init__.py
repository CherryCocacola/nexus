"""
Training Pipeline — Phase 6.0 학습 파이프라인 모듈 (패키지 진입점).

이 파일은 `training` 패키지의 __init__.py 로, 학습 관련 핵심 클래스들을
한곳에서 모아 외부에 노출하는 "공개 API 창구" 역할을 한다. 다른 모듈은
`from training import LoRATrainer` 처럼 내부 파일 경로를 몰라도 바로 가져올 수
있어, 패키지 내부 구조가 바뀌어도 사용하는 쪽 코드는 영향을 덜 받는다.

무엇을 하는 모듈인가:
- 에어갭(폐쇄망) 환경에서 LoRA/QLoRA 기반 모델 학습을 관리한다.
- 아래 5-Phase 전략에 따라 모델을 점진적으로 개선한다.
    1) Prompt Engineering  — 프롬프트만으로 성능 확보
    2) Bootstrap LoRA      — 초기 시드 데이터로 LoRA 학습
    3) Self-Data QLoRA     — 자체 수집 데이터로 QLoRA 학습
    4) Reasoning Finetune  — 추론 능력 강화 파인튜닝
    5) Domain Finetune     — 도메인 특화 파인튜닝

여기서 노출하는 주요 구성요소(각 클래스의 세부 구현은 동명의 하위 파일 참조):
- BootstrapGenerator — 초기 부트스트랩 학습 데이터 생성기
- CheckpointManager  — 학습 체크포인트 저장/복원 관리자
- DataCollector      — 실제 대화/실행 로그에서 학습 데이터 수집
- FeedbackLoop       — 피드백 기반 자동 재학습 루프
- LoRATrainer        — LoRA/QLoRA 실제 학습 실행기
- TrainingConfig     — 학습 하이퍼파라미터 등 설정 모델
- TrainingPhase      — 위 5-Phase 단계를 나타내는 열거형
- TrainingStrategy   — 단계 전환/전략 판단 로직

의존성 방향: training → core (core 는 training 을 절대 import 하지 않는다.
역방향 import 는 anti-patterns 규칙 #9 위반이므로 금지).

작성자: 이현수 / 작성일: 2026-07-05
"""

# 하위 모듈에서 공개할 클래스들을 끌어올려 `training` 네임스페이스에 등록한다.
# (이렇게 모아두면 사용하는 쪽은 파일 위치를 몰라도 패키지에서 바로 import 가능)
from training.bootstrap_generator import BootstrapGenerator
from training.checkpoint_manager import CheckpointManager
from training.data_collector import DataCollector
from training.feedback_loop import FeedbackLoop
from training.strategy import TrainingPhase, TrainingStrategy
from training.trainer import LoRATrainer, TrainingConfig

# __all__ — `from training import *` 시 노출할 공개 심볼의 명시적 목록.
# 여기에 적힌 이름만 와일드카드 import 대상이 되며, 이름순 정렬로 관리해
# 가독성과 diff 안정성을 확보한다.
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
