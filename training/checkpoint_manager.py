"""
체크포인트 관리자 — LoRA 어댑터 체크포인트의 목록 관리, 활성화, 롤백.

학습된 LoRA/QLoRA 어댑터를 체크포인트 단위로 관리한다.
각 체크포인트는 메타데이터(평가 결과, 학습 설정, 생성 시간)와 함께
파일 시스템에 저장되며, GPU 서버에 활성화/비활성화 요청을 보낼 수 있다.

체크포인트 디렉토리 구조:
  checkpoints/
    {checkpoint_name}/
      adapter_model.safetensors   — LoRA 가중치
      adapter_config.json         — LoRA 설정
      metadata.json               — 평가 결과, 학습 설정, 생성 시간

에어갭 준수: GPU 서버 URL은 LAN 주소만 허용한다.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("nexus.training.checkpoint_manager")

# 메타데이터 파일 이름 (각 체크포인트 디렉토리 내부)
_METADATA_FILENAME = "metadata.json"


class CheckpointManager:
    """
    체크포인트 목록 관리, 활성화, 롤백.

    파일 시스템에 저장된 체크포인트를 조회하고,
    GPU 서버에 특정 체크포인트를 활성화(LoRA hot-loading)하도록 요청한다.
    롤백 시에는 이전 체크포인트를 재활성화하거나 기본 모델로 복원한다.
    """

    def __init__(self, checkpoints_dir: str = "./checkpoints/") -> None:
        """
        Args:
            checkpoints_dir: 체크포인트가 저장된 디렉토리 경로
        """
        self._checkpoints_dir = Path(checkpoints_dir)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        # 현재 활성화된 체크포인트 이름 (None이면 기본 모델)
        self._active_checkpoint: str | None = None

    @property
    def checkpoints_dir(self) -> Path:
        """체크포인트 디렉토리 경로를 반환한다."""
        return self._checkpoints_dir

    @property
    def active_checkpoint(self) -> str | None:
        """현재 활성화된 체크포인트 이름을 반환한다."""
        return self._active_checkpoint

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """
        저장된 모든 체크포인트 목록을 반환한다.

        각 체크포인트 디렉토리에서 metadata.json을 읽어
        이름, 생성 시간, 평가 결과를 포함한 목록을 구성한다.
        생성 시간의 내림차순(최신 우선)으로 정렬한다.

        Returns:
            체크포인트 정보 목록:
              - name: 체크포인트 이름 (디렉토리 이름)
              - path: 전체 경로
              - created_at: 생성 시간 (ISO 8601)
              - metadata: 평가 결과 등 메타데이터
              - is_active: 현재 활성화 여부
        """
        checkpoints: list[dict[str, Any]] = []

        if not self._checkpoints_dir.exists():
            return checkpoints

        for ckpt_dir in self._checkpoints_dir.iterdir():
            if not ckpt_dir.is_dir():
                continue

            info: dict[str, Any] = {
                "name": ckpt_dir.name,
                "path": str(ckpt_dir),
                "is_active": ckpt_dir.name == self._active_checkpoint,
            }

            # 메타데이터 파일 읽기
            metadata_file = ckpt_dir / _METADATA_FILENAME
            if metadata_file.exists():
                try:
                    with open(metadata_file, encoding="utf-8") as f:
                        metadata = json.load(f)
                    info["created_at"] = metadata.get("created_at", "")
                    info["metadata"] = metadata
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("메타데이터 읽기 실패: %s (%s)", metadata_file, e)
                    info["created_at"] = ""
                    info["metadata"] = {}
            else:
                # 메타데이터 파일이 없으면 디렉토리 수정 시간 사용
                mtime = datetime.fromtimestamp(ckpt_dir.stat().st_mtime, tz=UTC)
                info["created_at"] = mtime.isoformat()
                info["metadata"] = {}

            checkpoints.append(info)

        # 최신 우선 정렬
        checkpoints.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        return checkpoints

    async def activate(
        self,
        name: str,
        gpu_server_url: str,
    ) -> dict[str, Any]:
        """
        특정 체크포인트를 GPU 서버에 활성화(LoRA hot-loading)한다.

        GPU 서버에 LoRA 어댑터 로딩을 요청하여,
        이후 추론 요청에 해당 어댑터가 적용되도록 한다.

        Args:
            name: 활성화할 체크포인트 이름
            gpu_server_url: GPU 서버 URL (LAN 주소만 허용)

        Returns:
            활성화 결과: name, status, message

        Raises:
            FileNotFoundError: 체크포인트 디렉토리가 없을 때
        """
        ckpt_path = self._checkpoints_dir / name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

        gpu_url = gpu_server_url.rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{gpu_url}/v1/lora/load",
                    json={
                        "checkpoint_name": name,
                        "checkpoint_path": str(ckpt_path),
                    },
                )
                response.raise_for_status()
                result = response.json()
        except httpx.ConnectError as e:
            logger.warning("GPU 서버 연결 실패: %s", e)
            result = {"status": "error", "message": f"GPU 서버 연결 실패: {e}"}
            return {"name": name, **result}
        except httpx.HTTPStatusError as e:
            logger.warning("GPU 서버 응답 에러: %s", e)
            result = {"status": "error", "message": f"GPU 서버 응답 에러: {e}"}
            return {"name": name, **result}

        # 활성화 성공 시 추적 업데이트
        self._active_checkpoint = name
        logger.info("체크포인트 활성화: %s", name)

        return {"name": name, "status": "activated", **result}

    async def rollback(self, gpu_server_url: str) -> dict[str, Any]:
        """
        현재 활성화된 LoRA 어댑터를 해제하고 기본 모델로 복원한다.

        GPU 서버에 LoRA 어댑터 언로드를 요청하여
        기본 모델만으로 추론이 이루어지도록 한다.

        Args:
            gpu_server_url: GPU 서버 URL (LAN 주소만 허용)

        Returns:
            롤백 결과: previous_checkpoint, status, message
        """
        previous = self._active_checkpoint
        gpu_url = gpu_server_url.rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{gpu_url}/v1/lora/unload",
                )
                response.raise_for_status()
                result = response.json()
        except httpx.ConnectError as e:
            logger.warning("GPU 서버 연결 실패 (롤백): %s", e)
            result = {"status": "error", "message": f"GPU 서버 연결 실패: {e}"}
            return {"previous_checkpoint": previous, **result}
        except httpx.HTTPStatusError as e:
            logger.warning("GPU 서버 응답 에러 (롤백): %s", e)
            result = {"status": "error", "message": f"GPU 서버 응답 에러: {e}"}
            return {"previous_checkpoint": previous, **result}

        # 활성 체크포인트 해제
        self._active_checkpoint = None
        logger.info("롤백 완료: %s → 기본 모델", previous or "(없음)")

        return {
            "previous_checkpoint": previous,
            "status": "rolled_back",
            **result,
        }

    def get_best(self, metric: str = "eval_accuracy") -> dict[str, Any] | None:
        """
        지정된 메트릭 기준으로 최고 성능의 체크포인트를 반환한다.

        각 체크포인트의 metadata.json에서 해당 메트릭을 읽어
        가장 높은 값을 가진 체크포인트를 선택한다.

        Args:
            metric: 비교할 메트릭 키 (metadata 내부 키)

        Returns:
            최고 성능 체크포인트 정보, 없으면 None
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        best: dict[str, Any] | None = None
        best_value = -float("inf")

        for ckpt in checkpoints:
            metadata = ckpt.get("metadata", {})
            value = metadata.get(metric, -float("inf"))
            if isinstance(value, (int, float)) and value > best_value:
                best_value = value
                best = ckpt

        return best

    def save_metadata(
        self,
        name: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        체크포인트에 메타데이터를 저장한다.

        학습 완료 후 평가 결과, 학습 설정 등을 metadata.json에 기록한다.

        Args:
            name: 체크포인트 이름
            metadata: 저장할 메타데이터 딕셔너리
        """
        ckpt_dir = self._checkpoints_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 생성 시간 자동 추가
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(UTC).isoformat()

        metadata_file = ckpt_dir / _METADATA_FILENAME
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info("메타데이터 저장: %s → %s", name, metadata_file)
