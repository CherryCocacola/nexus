"""
자동 학습 루프 — 수집→학습→평가→Phase 전이를 자동화한다.

FeedbackLoop는 Nexus의 자기 개선(self-improvement) 파이프라인의 핵심이다.
한 사이클은 다음 순서로 진행된다:
  1. DataCollector에서 수집된 데이터를 JSONL로 내보낸다
  2. LoRATrainer에게 학습을 요청한다
  3. 생성된 체크포인트를 평가한다
  4. TrainingStrategy에서 다음 Phase 전이를 판단한다

왜 자동 루프인가:
  - 에어갭 환경에서 수동 개입을 최소화한다
  - 평가 기반으로 학습 품질을 보장한다
  - 회귀 발생 시 자동으로 롤백할 수 있는 기반을 제공한다
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from training.data_collector import DataCollector
from training.strategy import TrainingStrategy
from training.trainer import LoRATrainer, TrainingJobStatus

logger = logging.getLogger("nexus.training.feedback_loop")

# 학습 상태 폴링 간격 (초)
_POLL_INTERVAL_SECONDS = 30.0
# 학습 최대 대기 시간 (초) — 2시간
_MAX_TRAINING_WAIT_SECONDS = 7200.0


class FeedbackLoop:
    """
    수집→학습→평가→승격 자동 루프.

    한 사이클을 실행하면 데이터 내보내기, 학습 요청, 평가, Phase 전이를
    순차적으로 수행한다. 각 단계의 결과를 딕셔너리로 반환하여
    호출자가 상황을 파악할 수 있도록 한다.
    """

    def __init__(
        self,
        eval_gpu_server_url: str | None = None,
        poll_interval: float = _POLL_INTERVAL_SECONDS,
        max_wait: float = _MAX_TRAINING_WAIT_SECONDS,
    ) -> None:
        """
        Args:
            eval_gpu_server_url: 평가용 GPU 서버 URL (None이면 trainer의 URL 사용)
            poll_interval: 학습 상태 폴링 간격 (초)
            max_wait: 학습 최대 대기 시간 (초)
        """
        self._eval_gpu_server_url = eval_gpu_server_url
        self._poll_interval = poll_interval
        self._max_wait = max_wait
        # 사이클 실행 이력
        self._cycle_history: list[dict[str, Any]] = []

    async def run_cycle(
        self,
        strategy: TrainingStrategy,
        trainer: LoRATrainer,
        collector: DataCollector,
    ) -> dict[str, Any]:
        """
        한 사이클: 데이터 내보내기→학습→평가→Phase 전이 판단.

        Args:
            strategy: 5-Phase 전략 관리자
            trainer: LoRA/QLoRA 트레이너
            collector: 상호작용 데이터 수집기

        Returns:
            사이클 결과:
              - phase: 현재 Phase 이름
              - exported_count: 내보낸 데이터 수
              - job_id: 학습 작업 ID
              - training_status: 학습 최종 상태
              - eval_results: 평가 결과
              - advanced: Phase 전이 여부
              - advance_reason: 전이 사유
        """
        cycle_start = datetime.now(UTC)
        result: dict[str, Any] = {
            "phase": strategy.current_phase.name,
            "cycle_start": cycle_start.isoformat(),
        }

        # ── Step 1: Phase별 설정 확인 ──
        phase_config = strategy.get_config_for_phase()
        if phase_config.get("method") == "none":
            # Phase 0(프롬프트 엔지니어링)은 학습을 수행하지 않는다
            result["skipped"] = True
            result["reason"] = "Phase 0(프롬프트 엔지니어링)은 학습이 필요하지 않습니다."
            logger.info("사이클 건너뜀: %s", result["reason"])
            return result

        # ── Step 2: 수집된 데이터를 JSONL로 내보내기 ──
        export_path = f"data/collected/export_{cycle_start.strftime('%Y%m%d_%H%M%S')}.jsonl"
        try:
            exported_count = await collector.export_jsonl(export_path)
            result["exported_count"] = exported_count
        except Exception as e:
            logger.error("데이터 내보내기 실패: %s", e)
            result["error"] = f"데이터 내보내기 실패: {e}"
            result["exported_count"] = 0
            self._cycle_history.append(result)
            return result

        if exported_count == 0:
            result["skipped"] = True
            result["reason"] = "내보낼 데이터가 없습니다."
            logger.info("사이클 건너뜀: 내보낼 데이터 없음")
            self._cycle_history.append(result)
            return result

        # ── Step 3: 학습 요청 ──
        try:
            job_id = await trainer.start_training(export_path)
            result["job_id"] = job_id
        except Exception as e:
            logger.error("학습 시작 실패: %s", e)
            result["error"] = f"학습 시작 실패: {e}"
            self._cycle_history.append(result)
            return result

        # ── Step 4: 학습 완료 대기 (폴링) ──
        training_status = await self._wait_for_training(trainer, job_id)
        result["training_status"] = training_status

        if training_status != TrainingJobStatus.COMPLETED.value:
            result["error"] = f"학습이 정상 완료되지 않았습니다: {training_status}"
            logger.warning("학습 비정상 종료: job_id=%s, status=%s", job_id, training_status)
            self._cycle_history.append(result)
            return result

        # ── Step 5: 체크포인트 평가 ──
        checkpoint_path = f"{trainer.config.output_dir}/{job_id}"
        eval_results = await self.evaluate(checkpoint_path)
        result["eval_results"] = eval_results

        # ── Step 6: Phase 전이 판단 ──
        can_advance, reason = strategy.can_advance(eval_results)
        result["can_advance"] = can_advance
        result["advance_reason"] = reason

        if can_advance:
            new_phase = strategy.advance()
            result["advanced"] = True
            result["new_phase"] = new_phase.name
            logger.info("Phase 전이 완료: → %s", new_phase.name)
        else:
            result["advanced"] = False
            logger.info("Phase 유지: %s (%s)", strategy.current_phase.name, reason)

        result["cycle_end"] = datetime.now(UTC).isoformat()
        self._cycle_history.append(result)
        return result

    async def _wait_for_training(
        self,
        trainer: LoRATrainer,
        job_id: str,
    ) -> str:
        """
        학습 완료를 대기한다 (폴링 방식).

        지정된 간격으로 상태를 확인하며, 최대 대기 시간을 초과하면
        타임아웃으로 처리한다.

        Returns:
            최종 학습 상태 문자열 (completed, failed, cancelled, timeout)
        """
        elapsed = 0.0

        while elapsed < self._max_wait:
            status_data = await trainer.get_status(job_id)
            status = status_data.get("status", TrainingJobStatus.FAILED.value)

            # 종료 상태 확인
            terminal_statuses = {
                TrainingJobStatus.COMPLETED.value,
                TrainingJobStatus.FAILED.value,
                TrainingJobStatus.CANCELLED.value,
            }
            if status in terminal_statuses:
                return status

            # 진행률 로깅
            progress = status_data.get("progress", 0.0)
            logger.info(
                "학습 진행 중: job_id=%s, progress=%.1f%%, elapsed=%.0fs",
                job_id,
                progress * 100,
                elapsed,
            )

            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

        # 타임아웃 — 학습 취소 시도
        logger.warning("학습 타임아웃: job_id=%s (%.0f초 초과)", job_id, self._max_wait)
        await trainer.cancel(job_id)
        return "timeout"

    async def evaluate(self, checkpoint_path: str) -> dict[str, Any]:
        """
        체크포인트를 평가한다.

        GPU 서버에 평가 요청을 보내고 결과를 반환한다.
        현재는 기본 평가 메트릭 구조를 반환하며,
        실제 평가 로직은 GPU 서버 측에서 구현한다.

        Args:
            checkpoint_path: 평가할 체크포인트 경로

        Returns:
            평가 결과: accuracy, data_count, eval_samples, regression_pct 등
        """
        # GPU 서버에 평가 요청
        eval_url = self._eval_gpu_server_url
        if eval_url:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{eval_url}/v1/training/evaluate",
                        json={"checkpoint_path": checkpoint_path},
                    )
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                logger.warning("GPU 서버 평가 요청 실패, 로컬 기본값 반환: %s", e)

        # 폴백: GPU 서버에 연결할 수 없을 때 기본 구조를 반환한다
        # 실제 운영에서는 GPU 서버가 반드시 평가 결과를 반환해야 한다
        logger.info("체크포인트 평가 (로컬 기본값): %s", checkpoint_path)
        return {
            "accuracy": 0.0,
            "data_count": 0,
            "eval_samples": 0,
            "regression_pct": 0.0,
            "approved": False,
            "checkpoint_path": checkpoint_path,
            "note": "GPU 서버 평가 미수행 — 기본값 반환",
        }

    @property
    def cycle_history(self) -> list[dict[str, Any]]:
        """사이클 실행 이력의 복사본을 반환한다."""
        return list(self._cycle_history)
