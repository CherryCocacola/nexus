"""
core/model/hardware_tier.py 단위 테스트.

HardwareTier enum, detect_hardware_tier() 자동 감지,
_detect_gpu_vram_gb() GPU 감지 로직, get_tier_config() 설정 조회를 검증한다.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.model.hardware_tier import (
    TIER_CONFIG,
    HardwareTier,
    _detect_gpu_vram_gb,
    detect_hardware_tier,
    get_tier_config,
)


# ─────────────────────────────────────────────
# HardwareTier enum 테스트
# ─────────────────────────────────────────────
class TestHardwareTierEnum:
    """HardwareTier enum의 값과 속성을 검증한다."""

    def test_tier_values(self):
        """3가지 티어의 값이 올바른지 확인한다."""
        assert HardwareTier.TIER_S.value == "small"
        assert HardwareTier.TIER_M.value == "medium"
        assert HardwareTier.TIER_L.value == "large"

    def test_tier_is_str_enum(self):
        """HardwareTier는 str 기반이므로 문자열 비교가 가능하다."""
        assert HardwareTier.TIER_S == "small"
        assert HardwareTier.TIER_M == "medium"

    def test_tier_from_value(self):
        """문자열 값으로 enum 인스턴스를 생성할 수 있다."""
        assert HardwareTier("small") is HardwareTier.TIER_S
        assert HardwareTier("large") is HardwareTier.TIER_L


# ─────────────────────────────────────────────
# detect_hardware_tier() 테스트
# ─────────────────────────────────────────────
class TestDetectHardwareTier:
    """하드웨어 티어 감지 로직을 검증한다."""

    def test_explicit_tier_from_config_hardware_tier(self):
        """config.hardware_tier에 명시적 값이 있으면 그대로 사용한다."""
        config = SimpleNamespace(hardware_tier="medium")
        tier = detect_hardware_tier(config)
        assert tier is HardwareTier.TIER_M

    def test_explicit_tier_from_config_hardware_attr(self):
        """config.hardware.tier에 명시적 값이 있으면 그대로 사용한다."""
        config = SimpleNamespace(
            hardware_tier=None,
            hardware=SimpleNamespace(tier="large"),
        )
        tier = detect_hardware_tier(config)
        assert tier is HardwareTier.TIER_L

    def test_no_config_falls_back_to_tier_s(self):
        """config가 None이면 GPU 감지를 시도하고, 실패 시 TIER_S로 fallback한다."""
        # GPU 감지를 실패하도록 mock 처리
        with patch(
            "core.model.hardware_tier._detect_gpu_vram_gb", return_value=None
        ):
            tier = detect_hardware_tier(None)
        assert tier is HardwareTier.TIER_S

    def test_auto_config_detects_from_gpu_vram(self):
        """config.hardware_tier='auto'이면 GPU VRAM 기반으로 감지한다."""
        config = SimpleNamespace(hardware_tier="auto")
        with patch(
            "core.model.hardware_tier._detect_gpu_vram_gb", return_value=80.0
        ):
            tier = detect_hardware_tier(config)
        assert tier is HardwareTier.TIER_M

    def test_auto_vram_32gb_returns_tier_s(self):
        """VRAM 32GB → TIER_S (64GB 미만)."""
        with patch(
            "core.model.hardware_tier._detect_gpu_vram_gb", return_value=32.0
        ):
            tier = detect_hardware_tier(None)
        assert tier is HardwareTier.TIER_S

    def test_auto_vram_80gb_returns_tier_m(self):
        """VRAM 80GB → TIER_M (64GB 이상, 120GB 미만)."""
        with patch(
            "core.model.hardware_tier._detect_gpu_vram_gb", return_value=80.0
        ):
            tier = detect_hardware_tier(None)
        assert tier is HardwareTier.TIER_M

    def test_auto_vram_141gb_returns_tier_l(self):
        """VRAM 141GB → TIER_L (120GB 이상)."""
        with patch(
            "core.model.hardware_tier._detect_gpu_vram_gb", return_value=141.0
        ):
            tier = detect_hardware_tier(None)
        assert tier is HardwareTier.TIER_L

    def test_invalid_explicit_tier_falls_back_to_auto(self):
        """config에 잘못된 티어 값이 있으면 자동 감지로 전환된다."""
        config = SimpleNamespace(hardware_tier="invalid_tier")
        with patch(
            "core.model.hardware_tier._detect_gpu_vram_gb", return_value=32.0
        ):
            tier = detect_hardware_tier(config)
        assert tier is HardwareTier.TIER_S


# ─────────────────────────────────────────────
# _detect_gpu_vram_gb() 테스트
# ─────────────────────────────────────────────
class TestDetectGpuVramGb:
    """GPU VRAM 감지 로직을 검증한다."""

    def test_torch_cuda_returns_vram(self):
        """torch.cuda가 사용 가능하면 VRAM을 GB 단위로 반환한다."""
        # 32GB = 32 * 1024^3 bytes
        vram_bytes = 32 * (1024**3)
        mock_props = MagicMock()
        mock_props.total_mem = vram_bytes

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = _detect_gpu_vram_gb()

        assert result is not None
        assert abs(result - 32.0) < 0.01

    def test_no_torch_no_pynvml_no_nvidiasmi_returns_none(self):
        """torch, pynvml, nvidia-smi 모두 없으면 None을 반환한다."""
        # torch import를 실패시킨다
        with patch.dict("sys.modules", {"torch": None}), \
             patch.dict("sys.modules", {"pynvml": None}), \
             patch("subprocess.run", side_effect=FileNotFoundError):
            result = _detect_gpu_vram_gb()

        assert result is None

    def test_torch_import_error_tries_pynvml(self):
        """torch가 없으면 pynvml을 시도한다."""
        # pynvml로 80GB를 반환하도록 설정
        mock_pynvml = MagicMock()
        mock_info = MagicMock()
        mock_info.total = 80 * (1024**3)
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_info

        # torch를 ImportError로 실패시키고 pynvml은 성공하도록 한다
        def mock_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            if name == "pynvml":
                return mock_pynvml
            return original_import(name, *args, **kwargs)

        import builtins
        original_import = builtins.__import__

        with patch("builtins.__import__", side_effect=mock_import):
            result = _detect_gpu_vram_gb()

        # pynvml 경로로 값을 얻었어야 한다
        assert result is not None
        assert abs(result - 80.0) < 0.01


# ─────────────────────────────────────────────
# get_tier_config() 및 TIER_CONFIG 테스트
# ─────────────────────────────────────────────
class TestGetTierConfig:
    """티어별 오케스트레이션 설정 조회를 검증한다."""

    def test_tier_s_scout_enabled(self):
        """TIER_S는 scout_enabled=True이다."""
        cfg = get_tier_config(HardwareTier.TIER_S)
        assert cfg["scout_enabled"] is True

    def test_tier_m_scout_disabled(self):
        """TIER_M은 scout_enabled=False이다."""
        cfg = get_tier_config(HardwareTier.TIER_M)
        assert cfg["scout_enabled"] is False

    def test_tier_l_scout_disabled(self):
        """TIER_L은 scout_enabled=False이다."""
        cfg = get_tier_config(HardwareTier.TIER_L)
        assert cfg["scout_enabled"] is False

    def test_tier_s_multi_model_mode(self):
        """TIER_S는 multi_model 오케스트레이션 모드를 사용한다."""
        cfg = get_tier_config(HardwareTier.TIER_S)
        assert cfg["orchestration_mode"] == "multi_model"

    def test_tier_m_single_model_mode(self):
        """TIER_M/L은 single_model 오케스트레이션 모드를 사용한다."""
        cfg = get_tier_config(HardwareTier.TIER_M)
        assert cfg["orchestration_mode"] == "single_model"

    def test_tier_s_max_worker_tools_limited(self):
        """TIER_S는 도구 수가 11개로 제한된다."""
        cfg = get_tier_config(HardwareTier.TIER_S)
        assert cfg["max_worker_tools"] == 11

    def test_tier_m_max_worker_tools_full(self):
        """TIER_M은 도구 24개 전체를 사용한다."""
        cfg = get_tier_config(HardwareTier.TIER_M)
        assert cfg["max_worker_tools"] == 24

    def test_tier_s_turn_state_enabled(self):
        """TIER_S는 turn_state_enabled=True이다."""
        cfg = get_tier_config(HardwareTier.TIER_S)
        assert cfg["turn_state_enabled"] is True

    def test_tier_m_turn_state_disabled(self):
        """TIER_M은 turn_state_enabled=False이다."""
        cfg = get_tier_config(HardwareTier.TIER_M)
        assert cfg["turn_state_enabled"] is False

    def test_get_tier_config_returns_copy(self):
        """get_tier_config()는 원본이 아닌 복사본을 반환한다."""
        cfg1 = get_tier_config(HardwareTier.TIER_S)
        cfg2 = get_tier_config(HardwareTier.TIER_S)
        cfg1["scout_enabled"] = "MODIFIED"
        # 원본과 다른 호출 결과에 영향이 없어야 한다
        assert cfg2["scout_enabled"] is True

    def test_all_tiers_have_config(self):
        """모든 HardwareTier enum 값에 대해 TIER_CONFIG이 정의되어 있다."""
        for tier in HardwareTier:
            assert tier in TIER_CONFIG, f"{tier}의 설정이 TIER_CONFIG에 없습니다"
