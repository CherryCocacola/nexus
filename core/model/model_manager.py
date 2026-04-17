"""
모델 매니저 — Machine A 측 모델 관리 및 hot-swap 조율.

이 모듈은 Machine A(오케스트레이터)에서 실행되며,
Machine B(GPU 서버)의 모델 상태를 원격으로 관리한다.

주요 책임:
  1. 현재 활성 모델 추적 (primary / auxiliary)
  2. Hot-swap 요청 조율 (primary ↔ auxiliary 전환)
  3. LoRA 체크포인트 로드/언로드 요청
  4. 모델 헬스 체크

왜 Machine A에 매니저가 필요한가:
  QueryEngine이 모델을 전환하거나 상태를 확인할 때,
  GPU 서버 API를 직접 호출하는 것보다 추상화된 인터페이스를 통하는 것이
  코드 결합도를 낮추고 테스트를 쉽게 한다.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger("nexus.model.manager")


class ActiveModel(str, Enum):
    """현재 활성 모델 식별자."""

    PRIMARY = "primary"  # Qwen 3.5 27B
    AUXILIARY = "auxiliary"  # ExaOne 7.8B


class ModelManager:
    """
    Machine A 측 모델 매니저.
    GPU 서버(Machine B)의 모델 상태를 원격으로 관리한다.

    사용 예:
        manager = ModelManager(gpu_server_url="http://192.168.22.28:8000")
        await manager.initialize()
        health = await manager.health_check()
        await manager.swap_model(ActiveModel.AUXILIARY)
    """

    def __init__(
        self,
        gpu_server_url: str = "http://localhost:8000",
        api_key: str = "local-key",
    ):
        self.gpu_server_url = gpu_server_url.rstrip("/")
        self.api_key = api_key
        self._active_model = ActiveModel.PRIMARY
        self._server_healthy = False
        self._active_loras: list[str] = []

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
        )

    @property
    def active_model(self) -> ActiveModel:
        """현재 활성 모델을 반환한다."""
        return self._active_model

    @property
    def is_healthy(self) -> bool:
        """GPU 서버가 정상인지 반환한다."""
        return self._server_healthy

    # ─── 초기화 ───

    async def initialize(self) -> dict[str, Any]:
        """
        GPU 서버 상태를 확인하고 초기 정보를 로드한다.
        부트스트랩 Phase 2에서 호출된다.
        """
        health = await self.health_check()
        if health:
            logger.info(
                f"모델 매니저 초기화 완료: "
                f"active_model={self._active_model.value}, "
                f"server={self.gpu_server_url}"
            )
        else:
            logger.warning(
                f"GPU 서버 연결 실패: {self.gpu_server_url}. "
                f"서버 시작 후 재시도가 필요합니다."
            )
        return {"healthy": health, "active_model": self._active_model.value}

    # ─── 헬스 체크 ───

    async def health_check(self) -> bool:
        """
        GPU 서버의 헬스 상태를 확인한다.
        성공 시 활성 모델, LoRA 정보를 업데이트한다.
        """
        try:
            resp = await self._client.get(
                f"{self.gpu_server_url}/health",
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._server_healthy = True

                # 서버에서 보고한 활성 모델로 동기화
                active = data.get("active_model", "primary")
                if active in ("primary", "auxiliary"):
                    self._active_model = ActiveModel(active)

                self._active_loras = data.get("active_loras", [])
                return True

            self._server_healthy = False
            return False
        except Exception as e:
            logger.debug(f"헬스 체크 실패: {e}")
            self._server_healthy = False
            return False

    # ─── 모델 전환 (hot-swap) ───

    async def swap_model(self, target: ActiveModel) -> dict[str, Any]:
        """
        Primary ↔ Auxiliary 모델을 전환한다.
        GPU 서버의 /v1/models/swap 엔드포인트를 호출한다.

        왜 hot-swap인가: RTX 5090(32GB)에서는 두 모델을 동시에 로딩할 수 없으므로,
        하나를 내리고 다른 하나를 올려야 한다. H200에서는 동시 로딩이 가능하다.
        """
        if target == self._active_model:
            return {"status": "already_active", "model": target.value}

        try:
            resp = await self._client.post(
                f"{self.gpu_server_url}/v1/models/swap",
                json={"target": target.value},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120.0,  # 모델 전환은 시간이 걸릴 수 있다
            )
            resp.raise_for_status()
            result = resp.json()
            self._active_model = target
            logger.info(f"모델 전환 완료: {target.value}")
            return result
        except Exception as e:
            logger.error(f"모델 전환 실패: {e}")
            return {"status": "error", "message": str(e)}

    # ─── LoRA 체크포인트 관리 ───

    async def load_lora(self, name: str, path: str) -> dict[str, Any]:
        """LoRA 체크포인트를 GPU 서버에 로드한다."""
        try:
            resp = await self._client.post(
                f"{self.gpu_server_url}/v1/lora/load",
                json={"name": name, "path": path},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,
            )
            resp.raise_for_status()
            result = resp.json()
            if name not in self._active_loras:
                self._active_loras.append(name)
            logger.info(f"LoRA 체크포인트 로드: {name}")
            return result
        except Exception as e:
            logger.error(f"LoRA 로드 실패: {e}")
            return {"status": "error", "message": str(e)}

    async def unload_lora(self, name: str) -> dict[str, Any]:
        """LoRA 체크포인트를 GPU 서버에서 언로드한다."""
        try:
            resp = await self._client.post(
                f"{self.gpu_server_url}/v1/lora/unload",
                json={"name": name},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            result = resp.json()
            if name in self._active_loras:
                self._active_loras.remove(name)
            logger.info(f"LoRA 체크포인트 언로드: {name}")
            return result
        except Exception as e:
            logger.error(f"LoRA 언로드 실패: {e}")
            return {"status": "error", "message": str(e)}

    # ─── 상태 정보 ───

    def get_status(self) -> dict[str, Any]:
        """현재 모델 매니저 상태를 반환한다."""
        return {
            "gpu_server_url": self.gpu_server_url,
            "active_model": self._active_model.value,
            "server_healthy": self._server_healthy,
            "active_loras": self._active_loras,
        }

    # ─── 정리 ───

    async def close(self) -> None:
        """HTTP 클라이언트를 정리한다."""
        await self._client.aclose()
