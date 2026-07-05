"""
자동 학습 루프 — 수집 → 학습 → 평가 → Phase 전이를 하나의 사이클로 자동화한다.

FeedbackLoop는 Nexus의 자기 개선(self-improvement) 파이프라인을 조율하는 오케스트레이터다.
사람이 매번 학습을 돌리고 결과를 확인하는 대신, 아래 한 사이클을 코드가 순서대로 진행한다:

  1. DataCollector가 그동안 모아둔 상호작용 데이터를 JSONL 파일로 내보낸다(export).
  2. LoRATrainer에게 그 JSONL을 넘겨 LoRA/QLoRA 학습을 시작하도록 요청한다.
  3. 학습이 끝날 때까지 상태를 폴링하며 기다린다.
  4. 학습으로 나온 체크포인트를 평가(evaluate)해 품질 지표를 얻는다.
  5. TrainingStrategy에게 지표를 넘겨 다음 Phase로 넘어갈지(전이) 판단한다.

왜 굳이 자동 루프로 만드는가:
  - 에어갭(폐쇄망) 환경이라 사람이 붙어서 수동 개입하기 어렵다 → 개입을 최소화한다.
  - "평가를 통과해야만 다음 단계로 간다"는 게이트를 두어 학습 품질을 보장한다.
  - 성능이 오히려 나빠지는 회귀(regression)가 나면 자동으로 되돌릴(롤백) 수 있는 토대가 된다.

주요 구성:
  - FeedbackLoop 클래스: run_cycle()이 진입점. 나머지는 그 하위 단계 도우미다.
  - 협력 대상(외부에서 주입받음): TrainingStrategy(단계 전략), LoRATrainer(실제 학습),
    DataCollector(데이터 수집). 이 파일은 이들을 직접 만들지 않고 인자로 받아 조립만 한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from training.data_collector import DataCollector
from training.strategy import TrainingStrategy
from training.trainer import LoRATrainer, TrainingJobStatus

# 이 모듈 전용 로거. 이름 규칙 "nexus.{module}"을 따라 로그 출처를 한눈에 알아볼 수 있게 한다.
logger = logging.getLogger("nexus.training.feedback_loop")

# 학습 상태를 다시 확인하기까지 기다리는 간격(초). 너무 짧으면 서버에 부담, 너무 길면 반응이 느리다.
_POLL_INTERVAL_SECONDS = 30.0
# 학습이 끝나기를 기다리는 최대 시간(초). 여기서는 2시간(7200초). 넘기면 타임아웃으로 간주한다.
_MAX_TRAINING_WAIT_SECONDS = 7200.0


class FeedbackLoop:
    """
    수집 → 학습 → 평가 → 승격(Phase 전이)을 한 번에 돌리는 자동 루프.

    run_cycle()를 호출하면 데이터 내보내기, 학습 요청, 완료 대기, 평가, Phase 전이 판단을
    순서대로 수행한다. 각 단계에서 무슨 일이 있었는지를 하나의 결과 딕셔너리에 차곡차곡 쌓아
    반환하므로, 호출자는 그 딕셔너리만 보고도 "어디까지 됐고 왜 멈췄는지"를 파악할 수 있다.

    상태 보존: 인스턴스는 지금까지 돌린 사이클 결과를 _cycle_history에 계속 누적한다.
    """

    def __init__(
        self,
        eval_gpu_server_url: str | None = None,
        poll_interval: float = _POLL_INTERVAL_SECONDS,
        max_wait: float = _MAX_TRAINING_WAIT_SECONDS,
    ) -> None:
        """
        FeedbackLoop를 초기화한다.

        Args:
            eval_gpu_server_url: 평가 요청을 보낼 GPU 서버 URL.
                None이면 별도 평가 서버 없이 로컬 기본값을 반환한다(evaluate() 참고).
            poll_interval: 학습 상태를 다시 확인하는 간격(초). 기본값은 모듈 상수.
            max_wait: 학습 완료를 기다리는 최대 시간(초). 초과하면 타임아웃 처리한다.
        """
        # 평가용 GPU 서버 주소. 없으면 평가 단계에서 폴백(기본값)으로 동작한다.
        self._eval_gpu_server_url = eval_gpu_server_url
        # 폴링 간격과 최대 대기 시간은 테스트나 운영 상황에 맞춰 주입 가능하도록 인자로 받는다.
        self._poll_interval = poll_interval
        self._max_wait = max_wait
        # 지금까지 실행한 사이클 결과를 순서대로 쌓아 두는 이력 리스트.
        # 외부에는 cycle_history 프로퍼티로 "복사본"만 노출해 내부 리스트가 오염되지 않게 한다.
        self._cycle_history: list[dict[str, Any]] = []

    async def run_cycle(
        self,
        strategy: TrainingStrategy,
        trainer: LoRATrainer,
        collector: DataCollector,
    ) -> dict[str, Any]:
        """
        한 사이클을 끝까지 실행한다: 데이터 내보내기 → 학습 → 완료 대기 → 평가 → Phase 전이 판단.

        중간 어느 단계에서든 실패하거나 건너뛸 조건이면, 그 시점까지의 정보를 담은 result를
        바로 반환한다(조기 반환). 그래서 반환 딕셔너리에 어떤 키가 들어있는지를 보면 어디까지
        진행됐는지 역추적할 수 있다.

        Args:
            strategy: 현재 Phase와 전이 규칙을 관리하는 5-Phase 전략 관리자.
            trainer: 실제 LoRA/QLoRA 학습을 수행하는 트레이너(GPU 서버와 통신).
            collector: 그동안의 상호작용을 모아 둔 데이터 수집기.

        Returns:
            사이클 결과 딕셔너리. 진행 정도에 따라 아래 키들이 채워진다:
              - phase: 사이클 시작 시점의 Phase 이름
              - exported_count: 내보낸 데이터 건수
              - job_id: 학습 작업 ID
              - training_status: 학습의 최종 상태 문자열
              - eval_results: 평가 결과 딕셔너리
              - can_advance / advanced / advance_reason: Phase 전이 가능 여부·실행 여부·사유
              - skipped / reason / error: 건너뛰거나 실패했을 때의 부가 정보
        """
        # 사이클 시작 시각(UTC). 아래 export 파일명과 결과 타임스탬프에 함께 사용한다.
        cycle_start = datetime.now(UTC)
        # 결과를 담을 딕셔너리. 단계가 진행될 때마다 키를 하나씩 추가해 나간다.
        result: dict[str, Any] = {
            "phase": strategy.current_phase.name,
            "cycle_start": cycle_start.isoformat(),
        }

        # ── Step 1: 현재 Phase가 학습을 필요로 하는지 확인 ──
        # Phase마다 학습 방법(method)이 다르다. method가 "none"이면 학습 자체를 하지 않는 단계다.
        phase_config = strategy.get_config_for_phase()
        if phase_config.get("method") == "none":
            # Phase 0은 순수 프롬프트 엔지니어링 단계라 모델 가중치 학습이 없다 → 조용히 건너뛴다.
            result["skipped"] = True
            result["reason"] = "Phase 0(프롬프트 엔지니어링)은 학습이 필요하지 않습니다."
            logger.info("사이클 건너뜀: %s", result["reason"])
            return result

        # ── Step 2: 수집된 데이터를 JSONL 파일로 내보내기 ──
        # 파일명에 시작 시각을 박아 사이클마다 고유한 파일이 생기도록 한다(덮어쓰기 방지).
        export_path = f"data/collected/export_{cycle_start.strftime('%Y%m%d_%H%M%S')}.jsonl"
        try:
            # collector가 실제로 파일을 쓰고, 내보낸 레코드 개수를 돌려준다.
            exported_count = await collector.export_jsonl(export_path)
            result["exported_count"] = exported_count
        except Exception as e:
            # 파일 쓰기·직렬화 등에서 실패하면 학습으로 넘어가지 않고 여기서 종료한다.
            logger.error("데이터 내보내기 실패: %s", e)
            result["error"] = f"데이터 내보내기 실패: {e}"
            result["exported_count"] = 0
            self._cycle_history.append(result)
            return result

        # 내보낼 데이터가 한 건도 없으면 학습할 재료가 없다는 뜻 → 사이클을 건너뛴다.
        if exported_count == 0:
            result["skipped"] = True
            result["reason"] = "내보낼 데이터가 없습니다."
            logger.info("사이클 건너뜀: 내보낼 데이터 없음")
            self._cycle_history.append(result)
            return result

        # ── Step 3: 트레이너에게 학습 시작 요청 ──
        try:
            # 방금 만든 JSONL 경로를 넘겨 학습을 시작하고, 추적용 작업 ID를 받는다.
            job_id = await trainer.start_training(export_path)
            result["job_id"] = job_id
        except Exception as e:
            # 학습 시작 자체가 실패(서버 연결 불가 등)하면 더 진행할 수 없으므로 종료한다.
            logger.error("학습 시작 실패: %s", e)
            result["error"] = f"학습 시작 실패: {e}"
            self._cycle_history.append(result)
            return result

        # ── Step 4: 학습이 끝날 때까지 폴링하며 대기 ──
        # 완료·실패·취소·타임아웃 중 하나의 최종 상태 문자열이 나올 때까지 블로킹된다.
        training_status = await self._wait_for_training(trainer, job_id)
        result["training_status"] = training_status

        # 정상 완료(COMPLETED)가 아니면 평가로 넘어가지 않는다.
        # 실패/취소/타임아웃 상태를 결과에 남기고 이번 사이클을 종료한다.
        if training_status != TrainingJobStatus.COMPLETED.value:
            result["error"] = f"학습이 정상 완료되지 않았습니다: {training_status}"
            logger.warning("학습 비정상 종료: job_id=%s, status=%s", job_id, training_status)
            self._cycle_history.append(result)
            return result

        # ── Step 5: 학습으로 나온 체크포인트를 평가 ──
        # 트레이너의 출력 디렉토리 아래 job_id 폴더에 체크포인트가 저장돼 있다고 약속돼 있다.
        checkpoint_path = f"{trainer.config.output_dir}/{job_id}"
        eval_results = await self.evaluate(checkpoint_path)
        result["eval_results"] = eval_results

        # ── Step 6: 평가 결과로 다음 Phase로 넘어갈 수 있는지 판단 ──
        # strategy가 지표(정확도·회귀율 등)를 보고 전이 가능 여부와 그 사유를 함께 돌려준다.
        can_advance, reason = strategy.can_advance(eval_results)
        result["can_advance"] = can_advance
        result["advance_reason"] = reason

        if can_advance:
            # 조건을 만족하면 실제로 Phase를 한 단계 올리고, 새 Phase 이름을 기록한다.
            new_phase = strategy.advance()
            result["advanced"] = True
            result["new_phase"] = new_phase.name
            logger.info("Phase 전이 완료: → %s", new_phase.name)
        else:
            # 조건 미달이면 현재 Phase를 그대로 유지한다(다음 사이클에서 다시 시도).
            result["advanced"] = False
            logger.info("Phase 유지: %s (%s)", strategy.current_phase.name, reason)

        # 사이클 종료 시각을 남기고, 전체 결과를 이력에 추가한 뒤 반환한다.
        result["cycle_end"] = datetime.now(UTC).isoformat()
        self._cycle_history.append(result)
        return result

    async def _wait_for_training(
        self,
        trainer: LoRATrainer,
        job_id: str,
    ) -> str:
        """
        학습이 끝날 때까지 일정 간격으로 상태를 확인(폴링)하며 기다린다.

        학습은 오래 걸리는 비동기 작업이라, 여기서는 self._poll_interval마다 상태를 물어보고,
        완료/실패/취소 같은 "종료 상태"가 나오면 즉시 그 값을 반환한다. self._max_wait를
        넘기도록 끝나지 않으면 무한 대기를 막기 위해 학습을 취소하고 "timeout"을 반환한다.

        Args:
            trainer: 상태 조회(get_status)와 취소(cancel)를 제공하는 트레이너.
            job_id: 기다릴 학습 작업의 ID.

        Returns:
            최종 상태 문자열. completed / failed / cancelled 또는 타임아웃 시 "timeout".
        """
        # 지금까지 기다린 누적 시간(초). max_wait에 도달하면 루프를 빠져나간다.
        elapsed = 0.0

        while elapsed < self._max_wait:
            # 현재 학습 상태를 조회한다. status 키가 없으면 안전하게 FAILED로 간주(fail-closed).
            status_data = await trainer.get_status(job_id)
            status = status_data.get("status", TrainingJobStatus.FAILED.value)

            # 더 이상 변하지 않는 "종료 상태"들의 집합. 이 중 하나면 대기를 끝내고 그대로 반환한다.
            terminal_statuses = {
                TrainingJobStatus.COMPLETED.value,
                TrainingJobStatus.FAILED.value,
                TrainingJobStatus.CANCELLED.value,
            }
            if status in terminal_statuses:
                return status

            # 아직 진행 중이면 현재 진행률을 로그로 남겨 운영자가 상황을 지켜볼 수 있게 한다.
            progress = status_data.get("progress", 0.0)
            logger.info(
                "학습 진행 중: job_id=%s, progress=%.1f%%, elapsed=%.0fs",
                job_id,
                progress * 100,
                elapsed,
            )

            # 다음 확인까지 대기하고, 누적 경과 시간을 갱신한다.
            await asyncio.sleep(self._poll_interval)
            elapsed += self._poll_interval

        # 여기까지 왔다면 max_wait를 초과한 것 → 매달린 학습을 취소하고 타임아웃으로 보고한다.
        logger.warning("학습 타임아웃: job_id=%s (%.0f초 초과)", job_id, self._max_wait)
        await trainer.cancel(job_id)
        return "timeout"

    async def evaluate(self, checkpoint_path: str) -> dict[str, Any]:
        """
        학습으로 만들어진 체크포인트의 품질을 평가한다.

        실제 평가(추론 돌려보고 정확도·회귀율 계산)는 무거운 GPU 작업이라 GPU 서버에 위임한다.
        eval_gpu_server_url이 설정돼 있으면 그 서버의 평가 엔드포인트로 요청을 보내고 결과를
        그대로 받아 반환한다. 서버가 없거나 요청이 실패하면, 사이클 자체가 죽지 않도록
        "승인되지 않음(approved=False)" 상태의 기본 구조를 폴백으로 반환한다.

        Args:
            checkpoint_path: 평가할 체크포인트의 경로.

        Returns:
            평가 결과 딕셔너리. accuracy, data_count, eval_samples, regression_pct,
            approved, checkpoint_path 등의 지표를 담는다.
        """
        # 평가를 위임할 GPU 서버 주소. 없으면(None) 아래 폴백으로 바로 넘어간다.
        eval_url = self._eval_gpu_server_url
        if eval_url:
            try:
                # httpx는 이 경로에서만 필요하므로 지연 import 한다(모듈 로딩 비용 절감).
                import httpx

                # 평가는 오래 걸릴 수 있어 넉넉히 120초 타임아웃을 준다.
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{eval_url}/v1/training/evaluate",
                        json={"checkpoint_path": checkpoint_path},
                    )
                    # HTTP 오류 코드면 예외로 만들어 아래 except에서 폴백 처리로 넘긴다.
                    response.raise_for_status()
                    return response.json()
            except Exception as e:
                # 네트워크 실패·오류 응답 등 무엇이든 여기서 잡고,
                # 사이클은 계속되게 폴백으로 넘어간다.
                logger.warning("GPU 서버 평가 요청 실패, 로컬 기본값 반환: %s", e)

        # 폴백: GPU 서버가 없거나 요청이 실패했을 때 반환하는 안전한 기본 구조.
        # approved=False이므로 이 결과로는 Phase 전이가 승인되지 않는다(fail-closed).
        # 실제 운영에서는 GPU 서버가 반드시 제대로 된 평가 결과를 반환해야 한다.
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
        """
        지금까지 실행한 사이클 이력의 "복사본"을 반환한다.

        내부 리스트(self._cycle_history)를 그대로 넘기면 호출자가 실수로 원본을 수정할 수 있어,
        list(...)로 얕은 복사를 만들어 반환한다. 방어적 복사(defensive copy) 패턴이다.
        """
        return list(self._cycle_history)
