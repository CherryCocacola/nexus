"""
core/model/gpu_detector.py 단위 테스트.

GPUTier, ModelSpec, get_tier_config()를 테스트한다.
실제 GPU는 필요하지 않다 — 설정 테이블만 검증한다.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from core.model.gpu_detector import (
    GPUTier,
    ModelSpec,
    QuantizationMethod,
    detect_gpu_tier,
    get_tier_config,
)


class TestGPUTierConfig:
    """GPU 티어 설정 테스트."""

    def test_rtx5090_uses_awq_quantization(self):
        """RTX 5090 설정이 AWQ 양자화를 사용하는지 확인한다."""
        config = get_tier_config(GPUTier.RTX_5090)
        assert config.tier == GPUTier.RTX_5090
        assert config.primary.quantization == QuantizationMethod.AWQ
        assert config.concurrent_models is False

    def test_rtx5090_enforce_eager(self):
        """RTX 5090에서 CUDA graph가 비활성화되는지 확인한다 (VRAM 절약)."""
        config = get_tier_config(GPUTier.RTX_5090)
        assert config.primary.enforce_eager is True

    def test_h100_no_quantization(self):
        """H100 설정이 양자화 없이 풀 프리시전인지 확인한다."""
        config = get_tier_config(GPUTier.H100)
        assert config.primary.quantization == QuantizationMethod.NONE
        assert config.primary.dtype == "bfloat16"

    def test_h200_concurrent_models(self):
        """H200에서 동시 모델 로딩이 가능한지 확인한다."""
        config = get_tier_config(GPUTier.H200)
        assert config.concurrent_models is True

    def test_multi_gpu_tensor_parallel(self):
        """Multi-GPU 설정에서 tensor parallelism이 적용되는지 확인한다."""
        config = get_tier_config(GPUTier.MULTI_GPU)
        assert config.primary.tensor_parallel_size == 2

    def test_all_tiers_have_primary(self):
        """모든 티어에 primary 모델이 설정되어 있는지 확인한다."""
        for tier in GPUTier:
            config = get_tier_config(tier)
            assert config.primary is not None
            assert config.primary.name == "qwen3.5-27b"

    def test_all_tiers_have_embedding(self):
        """모든 티어에 임베딩 모델이 설정되어 있는지 확인한다."""
        for tier in GPUTier:
            config = get_tier_config(tier)
            assert config.embedding is not None
            assert config.embedding.name == "multilingual-e5-large"

    def test_rtx5090_training_is_qlora(self):
        """RTX 5090의 학습 방식이 QLoRA인지 확인한다."""
        config = get_tier_config(GPUTier.RTX_5090)
        assert config.training.method == "qlora"
        assert config.training.gradient_checkpointing is True

    def test_h100_training_is_lora(self):
        """H100의 학습 방식이 LoRA인지 확인한다."""
        config = get_tier_config(GPUTier.H100)
        assert config.training.method == "lora"

    # ── A100 (80GB Ampere) 신규 티어 ──

    def test_a100_tier_enum_value(self):
        """GPUTier.A100 이 존재하고 값이 'a100' 인지 확인한다."""
        assert GPUTier.A100.value == "a100"

    def test_a100_config_bf16_no_quantization(self):
        """A100 은 Ampere 라 양자화 없이 BF16 풀 프리시전으로 서빙한다."""
        config = get_tier_config(GPUTier.A100)
        assert config.tier == GPUTier.A100
        assert config.primary.quantization == QuantizationMethod.NONE
        assert config.primary.dtype == "bfloat16"
        assert config.primary.enable_lora is True

    def test_all_tiers_have_config_without_keyerror(self):
        """모든 GPUTier 멤버에 대해 get_tier_config 가 KeyError 없이 동작해야 한다.

        새 티어(A100)를 추가하면서 configs 딕셔너리에 누락이 없는지 회귀 검증한다."""
        for tier in GPUTier:
            config = get_tier_config(tier)
            assert config.tier == tier


class TestDetectGpuTier:
    """detect_gpu_tier() — torch.cuda 를 mock 해 이름/VRAM 기반 분기를 검증한다.

    실제 GPU/torch 가 없어도 되도록, sys.modules 에 가짜 torch 를 주입한다.
    A100 80GB 와 H100 80GB 는 VRAM 이 동일해 '이름'으로만 구분되므로,
    이름에 'A100' 이 들어가면 A100, 아니면 H100 폴백을 확인하는 것이 핵심이다.
    """

    @staticmethod
    def _fake_torch(name: str, vram_gb: float, device_count: int = 1):
        """name/VRAM/장치수를 가진 가짜 torch 모듈을 만든다.

        detect_gpu_tier 는 함수 내부에서 `import torch` 하므로, patch.dict 로
        sys.modules['torch'] 를 이 가짜로 바꾸면 실제 CUDA 없이 분기만 검증된다.
        """
        props = SimpleNamespace(total_mem=int(vram_gb * (1024**3)), name=name)
        cuda = MagicMock()
        cuda.is_available.return_value = True
        cuda.device_count.return_value = device_count
        cuda.get_device_properties.return_value = props
        return SimpleNamespace(cuda=cuda)

    def test_detect_a100_by_name_at_80gb(self):
        """이름에 'A100' 이 있고 80GB 이면 A100 으로 판정한다."""
        fake = self._fake_torch("NVIDIA A100-SXM4-80GB", 80.0)
        with patch.dict("sys.modules", {"torch": fake}):
            assert detect_gpu_tier() == GPUTier.A100

    def test_detect_h100_by_name_at_80gb(self):
        """이름에 'H100' 이 있고 80GB 이면 H100 으로 판정한다."""
        fake = self._fake_torch("NVIDIA H100 80GB HBM3", 80.0)
        with patch.dict("sys.modules", {"torch": fake}):
            assert detect_gpu_tier() == GPUTier.H100

    def test_detect_80gb_unknown_name_falls_back_to_h100(self):
        """이름에 A100 단서가 없는 80GB 는 H100 으로 폴백한다(기존 동작 보존)."""
        fake = self._fake_torch("Generic Accelerator 80GB", 80.0)
        with patch.dict("sys.modules", {"torch": fake}):
            assert detect_gpu_tier() == GPUTier.H100

    def test_detect_rtx5090_at_32gb(self):
        """32GB 는 RTX 5090 티어로 판정한다(60GB 이하 분기)."""
        fake = self._fake_torch("NVIDIA GeForce RTX 5090", 32.0)
        with patch.dict("sys.modules", {"torch": fake}):
            assert detect_gpu_tier() == GPUTier.RTX_5090

    def test_detect_h200_above_120gb(self):
        """120GB 초과(141GB)는 H200 으로 판정한다."""
        fake = self._fake_torch("NVIDIA H200", 141.0)
        with patch.dict("sys.modules", {"torch": fake}):
            assert detect_gpu_tier() == GPUTier.H200

    def test_detect_multi_gpu_when_two_devices(self):
        """장치가 2개 이상이면 VRAM 무관하게 MULTI_GPU 로 판정한다."""
        fake = self._fake_torch("NVIDIA A100-80GB", 80.0, device_count=2)
        with patch.dict("sys.modules", {"torch": fake}):
            assert detect_gpu_tier() == GPUTier.MULTI_GPU


class TestModelSpec:
    """ModelSpec 테스트."""

    def test_default_values(self):
        """기본값이 fail-closed에 가까운지 확인한다."""
        spec = ModelSpec(name="test", path="./models/test")
        assert spec.enable_lora is True
        assert spec.enforce_eager is False
        assert spec.gpu_memory_utilization == 0.85
