"""
core/model/gpu_detector.py 단위 테스트.

GPUTier, ModelSpec, get_tier_config()를 테스트한다.
실제 GPU는 필요하지 않다 — 설정 테이블만 검증한다.
"""

from __future__ import annotations

from core.model.gpu_detector import (
    GPUTier,
    ModelSpec,
    QuantizationMethod,
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


class TestModelSpec:
    """ModelSpec 테스트."""

    def test_default_values(self):
        """기본값이 fail-closed에 가까운지 확인한다."""
        spec = ModelSpec(name="test", path="./models/test")
        assert spec.enable_lora is True
        assert spec.enforce_eager is False
        assert spec.gpu_memory_utilization == 0.85
