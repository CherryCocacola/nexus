"""
GPU 자동 감지 및 티어 시스템 — GPU VRAM 기반으로 최적 설정을 결정한다.

Claude Code에는 없는 Nexus 전용 시스템이다.
Claude Code는 클라우드 API를 사용하지만, Nexus는 로컬 GPU에서 직접 추론하므로
GPU 스펙에 따라 모델/양자화/배치/컨텍스트 설정을 자동 조정해야 한다.

설계 원칙:
  1. torch.cuda로 VRAM을 감지하여 GPUTier enum을 결정한다
  2. GPUTier별 최적 설정 테이블을 제공한다 (ModelSpec, TrainingSpec)
  3. 이 모듈은 Machine B(GPU 서버)에서 실행되지만,
     Machine A에서도 설정 참조용으로 사용할 수 있다
"""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger("nexus.gpu")


# ─────────────────────────────────────────────
# 열거형
# ─────────────────────────────────────────────
class GPUTier(str, Enum):
    """GPU 티어. VRAM 크기에 따라 자동 결정된다."""

    RTX_5090 = "rtx5090"  # 32GB — 기본 타겟
    H100 = "h100"  # 80GB
    H200 = "h200"  # 141GB
    MULTI_GPU = "multi_gpu"  # 2+ GPU


class QuantizationMethod(str, Enum):
    """양자화 방식."""

    NONE = "none"  # BF16 풀 프리시전
    AWQ = "awq"  # 4-bit AWQ (vLLM 네이티브)
    GPTQ = "gptq"  # 4-bit GPTQ
    BITSANDBYTES = "bnb"  # bitsandbytes NF4


# ─────────────────────────────────────────────
# 모델 스펙 (단일 모델의 서빙 설정)
# ─────────────────────────────────────────────
class ModelSpec(BaseModel):
    """
    단일 모델의 vLLM 서빙 설정.
    GPU 티어에 따라 양자화, 컨텍스트 길이, 배치 크기 등이 달라진다.
    """

    name: str
    path: str
    quantization: QuantizationMethod = QuantizationMethod.NONE
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.85
    max_num_batched_tokens: int | None = None
    max_num_seqs: int = 4
    tensor_parallel_size: int = 1
    enable_lora: bool = True
    max_lora_rank: int = 64
    max_loras: int = 4
    enforce_eager: bool = False  # True면 CUDA graph 비활성화 (VRAM 절약)


# ─────────────────────────────────────────────
# 임베딩 모델 스펙
# ─────────────────────────────────────────────
class EmbeddingSpec(BaseModel):
    """임베딩 모델 설정. e5-large는 항상 상주(always-on)한다."""

    name: str = "multilingual-e5-large"
    path: str = "./models/e5-large"
    dimension: int = 1024
    max_batch_size: int = 64
    device: str = "cuda:0"
    vram_gb: float = 0.7  # ~700MB


# ─────────────────────────────────────────────
# 학습 스펙
# ─────────────────────────────────────────────
class TrainingSpec(BaseModel):
    """LoRA/QLoRA 학습 설정."""

    method: str = "qlora"  # "qlora", "lora", "full"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    max_seq_length: int = 2048
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.1


# ─────────────────────────────────────────────
# GPU 티어별 전체 설정
# ─────────────────────────────────────────────
class GPUTierConfig(BaseModel):
    """
    GPU 티어별 전체 설정.
    primary(주 모델), auxiliary(보조 모델), embedding, training을 포함한다.
    concurrent_models가 False면 hot-swap으로 모델을 전환한다.
    """

    tier: GPUTier
    primary: ModelSpec
    auxiliary: ModelSpec | None = None
    embedding: EmbeddingSpec = Field(default_factory=EmbeddingSpec)
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    concurrent_models: bool = False  # True면 primary + auxiliary 동시 로딩 가능
    notes: str = ""


# ─────────────────────────────────────────────
# GPU 티어 감지 함수
# ─────────────────────────────────────────────
def detect_gpu_tier() -> GPUTier:
    """
    GPU VRAM 기반으로 티어를 자동 결정한다.
    torch.cuda가 필요하며, GPU가 없으면 RuntimeError를 발생시킨다.

    왜 자동 감지인가: 사용자가 GPU 스펙을 수동으로 입력하지 않아도
    최적 설정이 자동으로 적용되도록 하기 위해서이다.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch가 설치되지 않았습니다. GPU 서버에는 CUDA 지원 torch가 필요합니다."
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 감지할 수 없습니다. "
            "NVIDIA 드라이버와 CUDA 툴킷 설치를 확인하세요."
        )

    gpu_count = torch.cuda.device_count()
    device_props = torch.cuda.get_device_properties(0)
    vram_gb = device_props.total_mem / (1024**3)
    gpu_name = device_props.name

    logger.info(f"GPU 감지: {gpu_name}, VRAM={vram_gb:.1f}GB, 개수={gpu_count}")

    if gpu_count >= 2:
        total_vram = sum(
            torch.cuda.get_device_properties(i).total_mem / (1024**3)
            for i in range(gpu_count)
        )
        logger.info(f"멀티 GPU: {gpu_count}개, 총 VRAM={total_vram:.1f}GB")
        return GPUTier.MULTI_GPU
    elif vram_gb > 120:
        return GPUTier.H200
    elif vram_gb > 60:
        return GPUTier.H100
    else:
        return GPUTier.RTX_5090


def get_tier_config(tier: GPUTier) -> GPUTierConfig:
    """
    주어진 GPU 티어에 대한 최적 설정을 반환한다.
    각 티어별로 양자화, 컨텍스트 길이, 배치 크기 등이 미리 정의되어 있다.
    """
    configs: dict[GPUTier, GPUTierConfig] = {
        # ── RTX 5090 (32GB) — INT4 양자화 필수 ──
        GPUTier.RTX_5090: GPUTierConfig(
            tier=GPUTier.RTX_5090,
            primary=ModelSpec(
                name="qwen3.5-27b",
                path="./models/qwen3.5-27b",
                quantization=QuantizationMethod.AWQ,
                dtype="float16",
                max_model_len=4096,
                gpu_memory_utilization=0.80,
                max_num_seqs=1,
                enable_lora=True,
                max_lora_rank=32,
                max_loras=2,
                enforce_eager=True,  # CUDA graph 비활성화로 VRAM 절약
            ),
            auxiliary=ModelSpec(
                name="exaone-7.8b",
                path="./models/exaone-7.8b",
                quantization=QuantizationMethod.AWQ,
                dtype="float16",
                max_model_len=4096,
                gpu_memory_utilization=0.80,
                max_num_seqs=2,
                enable_lora=True,
                max_lora_rank=32,
                enforce_eager=True,
            ),
            training=TrainingSpec(
                method="qlora",
                lora_rank=16,
                lora_alpha=32,
                batch_size=1,
                gradient_accumulation_steps=8,
                gradient_checkpointing=True,
                max_seq_length=2048,
            ),
            concurrent_models=False,
            notes="Primary/Auxiliary 동시 로딩 불가. hot-swap 방식 사용.",
        ),
        # ── H100 (80GB) — BF16 풀 프리시전 ──
        GPUTier.H100: GPUTierConfig(
            tier=GPUTier.H100,
            primary=ModelSpec(
                name="qwen3.5-27b",
                path="./models/qwen3.5-27b",
                quantization=QuantizationMethod.NONE,
                dtype="bfloat16",
                max_model_len=8192,
                gpu_memory_utilization=0.85,
                max_num_seqs=4,
                enable_lora=True,
                max_lora_rank=64,
                max_loras=4,
                enforce_eager=False,
            ),
            auxiliary=ModelSpec(
                name="exaone-32b",
                path="./models/exaone-32b",
                quantization=QuantizationMethod.NONE,
                dtype="bfloat16",
                max_model_len=8192,
                gpu_memory_utilization=0.85,
                max_num_seqs=4,
                enable_lora=True,
                max_lora_rank=64,
            ),
            training=TrainingSpec(
                method="lora",
                lora_rank=64,
                lora_alpha=128,
                batch_size=4,
                gradient_accumulation_steps=4,
                gradient_checkpointing=False,
                max_seq_length=4096,
            ),
            concurrent_models=False,
            notes="58+32>80GB이므로 hot-swap 필요.",
        ),
        # ── H200 (141GB) — 동시 로딩 가능 ──
        GPUTier.H200: GPUTierConfig(
            tier=GPUTier.H200,
            primary=ModelSpec(
                name="qwen3.5-27b",
                path="./models/qwen3.5-27b",
                quantization=QuantizationMethod.NONE,
                dtype="bfloat16",
                max_model_len=16384,
                gpu_memory_utilization=0.90,
                max_num_seqs=8,
                enable_lora=True,
                max_lora_rank=128,
                max_loras=8,
                enforce_eager=False,
            ),
            auxiliary=ModelSpec(
                name="exaone-32b",
                path="./models/exaone-32b",
                quantization=QuantizationMethod.NONE,
                dtype="bfloat16",
                max_model_len=16384,
                gpu_memory_utilization=0.90,
                max_num_seqs=4,
                enable_lora=True,
                max_lora_rank=128,
            ),
            training=TrainingSpec(
                method="lora",
                lora_rank=128,
                lora_alpha=256,
                batch_size=8,
                gradient_accumulation_steps=2,
                gradient_checkpointing=False,
                max_seq_length=8192,
            ),
            concurrent_models=True,
            notes="58+32<141GB이므로 동시 로딩 가능. 최대 성능.",
        ),
        # ── Multi-GPU (2+) — Tensor Parallelism ──
        GPUTier.MULTI_GPU: GPUTierConfig(
            tier=GPUTier.MULTI_GPU,
            primary=ModelSpec(
                name="qwen3.5-27b",
                path="./models/qwen3.5-27b",
                quantization=QuantizationMethod.NONE,
                dtype="bfloat16",
                max_model_len=32768,
                gpu_memory_utilization=0.90,
                max_num_seqs=16,
                tensor_parallel_size=2,
                enable_lora=True,
                max_lora_rank=128,
                max_loras=8,
                enforce_eager=False,
            ),
            auxiliary=ModelSpec(
                name="exaone-32b",
                path="./models/exaone-32b",
                quantization=QuantizationMethod.NONE,
                dtype="bfloat16",
                max_model_len=32768,
                gpu_memory_utilization=0.90,
                max_num_seqs=8,
                tensor_parallel_size=2,
                enable_lora=True,
                max_lora_rank=128,
            ),
            training=TrainingSpec(
                method="lora",
                lora_rank=128,
                lora_alpha=256,
                batch_size=16,
                gradient_accumulation_steps=1,
                gradient_checkpointing=False,
                max_seq_length=16384,
            ),
            concurrent_models=True,
            notes="Tensor Parallelism 적용. Full fine-tune 가능.",
        ),
    }
    return configs[tier]
