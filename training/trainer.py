"""
LoRA/QLoRA 트레이너 — GPU 서버에 학습 요청을 보내고 상태를 추적한다.

Machine A(오케스트레이터)에서 Machine B(GPU 서버)로 학습 요청을 전달한다.
직접 GPU/CUDA를 호출하지 않고, HTTP API를 통해 학습을 관리한다.
이는 Nexus의 2-Machine 토폴로지(P4)를 준수하기 위함이다.

지원하는 학습 방법:
  - LoRA: Low-Rank Adaptation (Phase 1 부트스트랩)
  - QLoRA: Quantized LoRA (Phase 2~4 본격 학습)
  - full: Full fine-tuning (실험용, 메모리 집약적)

에어갭 준수: GPU 서버 URL은 LAN 주소만 허용한다.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("nexus.training.trainer")


# ─────────────────────────────────────────────
# 학습 작업 상태
# ─────────────────────────────────────────────
class TrainingJobStatus(str, Enum):
    """학습 작업의 상태를 나타낸다."""

    PENDING = "pending"  # 대기 중
    RUNNING = "running"  # 학습 진행 중
    COMPLETED = "completed"  # 완료
    FAILED = "failed"  # 실패
    CANCELLED = "cancelled"  # 취소됨


# ─────────────────────────────────────────────
# 학습 설정
# ─────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """
    LoRA/QLoRA 학습 하이퍼파라미터 설정.

    기본값은 RTX 5090 (32GB VRAM)에서 Gemma 4 31B의
    INT4 양자화 모델을 QLoRA로 학습하기 위한 최적값이다.
    """

    method: str = "qlora"  # "qlora", "lora", "full"
    model_path: str = "./models/gemma-4-31b-it"  # 기본 모델 경로
    output_dir: str = "./checkpoints/"  # 체크포인트 저장 경로
    lora_rank: int = 16  # LoRA rank — 클수록 표현력↑ 메모리↑
    lora_alpha: int = 32  # LoRA alpha — 보통 rank의 2배
    lora_dropout: float = 0.05  # LoRA dropout — 과적합 방지
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    batch_size: int = 1  # 배치 크기 (VRAM 제한으로 1 권장)
    gradient_accumulation_steps: int = 8  # 그래디언트 누적 — 실질적 배치 = 1 * 8 = 8
    learning_rate: float = 2e-4  # 학습률
    num_epochs: int = 3  # 에폭 수
    max_seq_length: int = 2048  # 최대 시퀀스 길이
    warmup_ratio: float = 0.03  # 웜업 비율 (전체 스텝의 3%)
    weight_decay: float = 0.01  # 가중치 감쇠
    save_steps: int = 100  # 체크포인트 저장 간격 (스텝)
    logging_steps: int = 10  # 로그 출력 간격 (스텝)

    def to_dict(self) -> dict[str, Any]:
        """학습 설정을 딕셔너리로 변환한다 (API 전송용)."""
        return {
            "method": self.method,
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_seq_length": self.max_seq_length,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
        }


def _validate_lan_url(url: str) -> None:
    """
    URL이 LAN 주소인지 검증한다.

    에어갭 환경에서 외부 네트워크 접근을 방지하기 위해
    GPU 서버 URL이 로컬/LAN 범위인지 확인한다.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    allowed_prefixes = ("localhost", "127.0.0.1", "10.", "172.", "192.168.")
    if not any(hostname.startswith(p) for p in allowed_prefixes):
        raise ValueError(
            f"에어갭 위반: GPU 서버 URL '{url}'이(가) LAN 주소가 아닙니다. "
            f"허용 범위: localhost, 127.0.0.1, 10.x, 172.x, 192.168.x"
        )


# ─────────────────────────────────────────────
# LoRA/QLoRA 트레이너
# ─────────────────────────────────────────────
class LoRATrainer:
    """
    LoRA/QLoRA 학습을 관리한다.

    Machine A에서 Machine B의 GPU 서버로 학습 요청을 HTTP API로 전송하고,
    학습 상태를 폴링하여 추적한다. 직접 GPU를 호출하지 않는다 (P4 준수).
    """

    def __init__(self, gpu_server_url: str, config: TrainingConfig) -> None:
        """
        Args:
            gpu_server_url: GPU 서버(Machine B)의 URL (LAN 주소만 허용)
            config: 학습 하이퍼파라미터 설정
        """
        _validate_lan_url(gpu_server_url)
        self._gpu_server_url = gpu_server_url.rstrip("/")
        self._config = config
        # 진행 중인 학습 작업 추적 (job_id → 상태 정보)
        self._active_jobs: dict[str, dict[str, Any]] = {}

    @property
    def config(self) -> TrainingConfig:
        """현재 학습 설정을 반환한다."""
        return self._config

    @property
    def gpu_server_url(self) -> str:
        """GPU 서버 URL을 반환한다."""
        return self._gpu_server_url

    async def start_training(self, data_path: str) -> str:
        """
        GPU 서버에 학습을 시작하도록 요청한다.

        학습 데이터 경로와 하이퍼파라미터를 전송하고,
        GPU 서버가 반환한 job_id를 돌려준다.

        Args:
            data_path: 학습 데이터(JSONL) 파일 경로

        Returns:
            job_id — 학습 작업 식별자

        Raises:
            FileNotFoundError: 학습 데이터 파일이 없을 때
            httpx.HTTPStatusError: GPU 서버 응답 에러
        """
        # 학습 데이터 파일 존재 확인
        if not Path(data_path).exists():
            raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {data_path}")

        # 학습 요청 페이로드 구성
        payload = {
            "data_path": data_path,
            "config": self._config.to_dict(),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._gpu_server_url}/v1/training/start",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                job_id = result.get("job_id", str(uuid.uuid4()))
        except httpx.ConnectError as e:
            # GPU 서버 연결 실패 시 로컬 job_id를 발급하여 추후 재시도 가능하게 한다
            job_id = f"local_{uuid.uuid4().hex[:12]}"
            logger.warning("GPU 서버 연결 실패, 로컬 job_id 발급: %s (에러: %s)", job_id, e)
            self._active_jobs[job_id] = {
                "job_id": job_id,
                "status": TrainingJobStatus.PENDING.value,
                "data_path": data_path,
                "config": self._config.to_dict(),
                "error": str(e),
            }
            return job_id

        # 활성 작업에 등록
        self._active_jobs[job_id] = {
            "job_id": job_id,
            "status": TrainingJobStatus.RUNNING.value,
            "data_path": data_path,
            "config": self._config.to_dict(),
        }

        logger.info(
            "학습 시작: job_id=%s, method=%s, data=%s",
            job_id,
            self._config.method,
            data_path,
        )
        return job_id

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """
        학습 작업의 현재 상태를 조회한다.

        GPU 서버에 상태를 폴링하고, 결과를 캐시에 반영한다.

        Args:
            job_id: 학습 작업 식별자

        Returns:
            상태 딕셔너리: job_id, status, progress, metrics 등
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._gpu_server_url}/v1/training/status/{job_id}",
                )
                response.raise_for_status()
                status_data = response.json()
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            # 연결 실패 시 캐시된 상태를 반환한다
            logger.warning("GPU 서버 상태 조회 실패: %s", e)
            cached = self._active_jobs.get(job_id)
            if cached:
                cached["error"] = str(e)
                return cached
            return {
                "job_id": job_id,
                "status": TrainingJobStatus.FAILED.value,
                "error": f"상태 조회 실패: {e}",
            }

        # 캐시 업데이트
        if job_id in self._active_jobs:
            self._active_jobs[job_id].update(status_data)

        return status_data

    async def cancel(self, job_id: str) -> bool:
        """
        진행 중인 학습 작업을 취소한다.

        Args:
            job_id: 취소할 학습 작업 식별자

        Returns:
            취소 성공 여부
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._gpu_server_url}/v1/training/cancel/{job_id}",
                )
                response.raise_for_status()
                success = response.json().get("cancelled", False)
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            logger.warning("학습 취소 요청 실패: %s", e)
            success = False

        # 로컬 상태 업데이트
        if job_id in self._active_jobs:
            if success:
                self._active_jobs[job_id]["status"] = TrainingJobStatus.CANCELLED.value
            logger.info("학습 취소 %s: job_id=%s", "성공" if success else "실패", job_id)

        return success

    @property
    def active_jobs(self) -> dict[str, dict[str, Any]]:
        """활성 학습 작업 목록의 복사본을 반환한다."""
        return dict(self._active_jobs)
