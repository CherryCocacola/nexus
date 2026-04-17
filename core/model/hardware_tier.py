"""
하드웨어 티어 감지 — GPU VRAM 기반으로 오케스트레이션 모드를 결정한다.

v7.0 핵심 모듈: 하드웨어 환경에 따라 Nexus의 동작 방식을 자동 조정한다.

왜 티어를 나누는가:
  - TIER_S (32GB): 8K 컨텍스트 → Scout(CPU) + Worker(GPU) 분리 필요
  - TIER_M (80GB): 32K 컨텍스트 → Worker 단독으로 충분
  - TIER_L (128GB+): 128K 컨텍스트 → v6.1 원래 설계 그대로 동작

핵심 원칙: 상위 티어는 하위 티어의 상위집합
  TIER_L ⊃ TIER_M ⊃ TIER_S
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger("nexus.model.hardware_tier")


class HardwareTier(str, Enum):
    """하드웨어 티어별 오케스트레이션 모드를 결정한다."""

    TIER_S = "small"    # RTX 5090 (32GB), 8K 컨텍스트
    TIER_M = "medium"   # H100 (80GB), 32K 컨텍스트
    TIER_L = "large"    # H200 (141GB) / GB10 (128GB), 128K 컨텍스트


# 티어별 오케스트레이션 설정
TIER_CONFIG = {
    HardwareTier.TIER_S: {
        "orchestration_mode": "multi_model",   # Scout(CPU) + Worker(GPU)
        "scout_enabled": True,
        "max_worker_tools": 11,                # 도구 수 제한
        "turn_state_enabled": True,            # 상태 외부화 활성
        "description": "RTX 5090 (32GB, 8K ctx)",
    },
    HardwareTier.TIER_M: {
        "orchestration_mode": "single_model",  # Worker 단독
        "scout_enabled": False,
        "max_worker_tools": 24,                # 도구 전체
        "turn_state_enabled": False,           # raw messages 누적 가능
        "description": "H100 (80GB, 32K ctx)",
    },
    HardwareTier.TIER_L: {
        "orchestration_mode": "single_model",  # Worker 단독
        "scout_enabled": False,
        "max_worker_tools": 24,
        "turn_state_enabled": False,
        "description": "H200/GB10 (128GB+, 128K ctx)",
    },
}


def detect_hardware_tier(config: Any = None) -> HardwareTier:
    """
    GPU VRAM을 기반으로 하드웨어 티어를 감지한다.

    감지 순서:
      1. config에 명시적 tier가 있으면 그대로 사용
      2. GPU VRAM을 확인하여 자동 분류
      3. GPU 정보를 가져올 수 없으면 TIER_S (가장 보수적)로 fallback

    Args:
        config: NexusConfig 객체 (선택)

    Returns:
        감지된 HardwareTier
    """
    # 1. config에서 명시적 tier 확인
    if config is not None:
        explicit_tier = None
        # config.hardware.tier 필드가 있으면 사용
        if hasattr(config, "hardware_tier") and config.hardware_tier:
            explicit_tier = config.hardware_tier
        elif hasattr(config, "hardware") and hasattr(config.hardware, "tier"):
            explicit_tier = config.hardware.tier

        if explicit_tier and explicit_tier != "auto":
            try:
                tier = HardwareTier(explicit_tier)
                logger.info("하드웨어 티어 (설정에서 지정): %s", tier.value)
                return tier
            except ValueError:
                logger.warning("알 수 없는 하드웨어 티어: %s, 자동 감지로 전환", explicit_tier)

    # 2. GPU VRAM 기반 자동 감지
    vram_gb = _detect_gpu_vram_gb()

    if vram_gb is None:
        # GPU 정보 없음 → 가장 보수적인 TIER_S
        logger.warning("GPU VRAM 감지 실패, TIER_S(보수적)로 fallback")
        return HardwareTier.TIER_S

    if vram_gb >= 120:
        tier = HardwareTier.TIER_L
    elif vram_gb >= 64:
        tier = HardwareTier.TIER_M
    else:
        tier = HardwareTier.TIER_S

    logger.info(
        "하드웨어 티어 자동 감지: %s (VRAM: %.1fGB)",
        tier.value,
        vram_gb,
    )
    return tier


def _detect_gpu_vram_gb() -> float | None:
    """
    GPU VRAM 용량을 GB 단위로 반환한다.

    에어갭 환경에서 torch/pynvml이 없을 수 있으므로
    여러 방법을 순차적으로 시도한다.
    """
    # 방법 1: torch.cuda (가장 정확)
    try:
        import torch

        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            return vram_bytes / (1024**3)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("torch.cuda VRAM 감지 실패: %s", e)

    # 방법 2: pynvml
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.total / (1024**3)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("pynvml VRAM 감지 실패: %s", e)

    # 방법 3: nvidia-smi CLI (Windows/Linux 공통)
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            vram_mb = float(result.stdout.strip().split("\n")[0])
            return vram_mb / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    except Exception as e:
        logger.debug("nvidia-smi VRAM 감지 실패: %s", e)

    return None


def get_tier_config(tier: HardwareTier) -> dict[str, Any]:
    """티어별 오케스트레이션 설정을 반환한다."""
    return dict(TIER_CONFIG.get(tier, TIER_CONFIG[HardwareTier.TIER_S]))
