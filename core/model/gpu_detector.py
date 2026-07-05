"""
GPU 자동 감지 및 티어(tier) 시스템 — 장착된 GPU의 VRAM 크기를 보고
그 하드웨어에 가장 잘 맞는 서빙/학습 설정을 자동으로 골라 주는 모듈.

[이 파일이 왜 필요한가]
Claude Code에는 없는 Nexus 전용 시스템이다. Claude Code는 클라우드 API를
쓰지만, Nexus는 폐쇄망(에어갭) 안에서 로컬 GPU로 직접 추론한다. 따라서
같은 코드라도 RTX 5090(32GB)에서는 4-bit 양자화가 필수이고, H200(141GB)
에서는 두 모델을 동시에 올릴 수 있는 것처럼, GPU마다 설정이 완전히 달라진다.
이 차이를 사람이 매번 손으로 맞추지 않도록 여기서 표준화한다.

[핵심 구성 요소]
  - GPUTier(enum)            : GPU를 5개 등급으로 분류(RTX 5090 / A100 / H100
                               / H200 / Multi-GPU)
  - QuantizationMethod(enum) : 양자화 방식(BF16 원본 / AWQ / GPTQ / bnb)
  - ModelSpec / EmbeddingSpec / TrainingSpec : 각각 추론 모델·임베딩 모델·
                               LoRA 학습의 세부 파라미터를 담는 Pydantic 모델
  - GPUTierConfig            : 위 스펙들을 티어 하나로 묶은 최종 설정 묶음
  - detect_gpu_tier()        : torch.cuda로 실제 GPU를 감지해 GPUTier 반환
  - get_tier_config(tier)    : 티어를 받아 미리 정의된 최적 설정을 반환

[실행 위치]
  이 모듈은 주로 Machine B(GPU 서버)에서 실행되지만, torch가 없어도 import
  자체는 되므로 Machine A(오케스트레이터)에서 설정값을 참조하는 용도로도
  쓸 수 있다. (torch는 detect_gpu_tier 안에서 지연 import 한다.)

작성자: 이현수 / 작성일: 2026-07-05
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
    """
    GPU 등급(티어). detect_gpu_tier()가 실제 VRAM 크기를 재서 이 중 하나로
    자동 분류한다. str을 상속하므로 값 자체가 "rtx5090" 같은 문자열이라
    로그 출력이나 YAML 설정과의 매칭이 편하다.

    각 항목 뒤의 주석은 대표 VRAM 용량과 특징이다. 용량이 겹치는 경우
    (A100 vs H100, 둘 다 80GB)는 이름으로 추가 구분한다.
    """

    RTX_5090 = "rtx5090"  # 32GB — Nexus 기본 개발/검증 타겟
    A100 = "a100"  # 80GB — H100과 VRAM은 같지만 Ampere 세대(BF16만, FP8 미지원)
    H100 = "h100"  # 80GB — Hopper 세대
    H200 = "h200"  # 141GB — 대용량, 두 모델 동시 로딩 가능
    MULTI_GPU = "multi_gpu"  # GPU 2장 이상 — Tensor Parallelism 사용


class QuantizationMethod(str, Enum):
    """
    모델 가중치를 어떤 방식으로 압축(양자화)해서 올릴지 나타내는 열거형.

    양자화는 가중치의 비트 수를 줄여 VRAM 사용량을 크게 낮추는 기법이다.
    VRAM이 부족한 RTX 5090에서는 AWQ(4-bit)가 필수지만, VRAM이 넉넉한
    A100/H100/H200에서는 굳이 압축하지 않고 원본 정밀도(NONE=BF16)로
    서빙해 품질을 최대한 유지한다.
    """

    NONE = "none"  # 양자화 없음 — BF16 풀 프리시전(원본 품질 유지)
    AWQ = "awq"  # 4-bit AWQ — vLLM이 기본 지원, RTX 5090에서 주로 사용
    GPTQ = "gptq"  # 4-bit GPTQ — 대안 4-bit 방식
    BITSANDBYTES = "bnb"  # bitsandbytes NF4 — 주로 QLoRA 학습 계열에서 사용


# ─────────────────────────────────────────────
# 모델 스펙 (단일 모델의 서빙 설정)
# ─────────────────────────────────────────────
class ModelSpec(BaseModel):
    """
    단일 모델 하나를 vLLM으로 서빙할 때 필요한 설정 묶음.

    여기 담긴 값들이 그대로 vLLM 엔진의 기동 인자로 이어진다. 같은 모델이라도
    GPU 티어가 달라지면 양자화 방식, 컨텍스트 길이, 동시 처리 개수 등을 다르게
    잡아야 하므로, 티어별로 이 스펙을 각각 다르게 채운다(get_tier_config 참조).

    필드 설명:
      name                    : 사람이 읽는 모델 식별 이름
      path                    : 로컬 가중치 경로(에어갭이라 항상 ./models/ 하위)
      quantization            : 양자화 방식(QuantizationMethod)
      dtype                   : 연산 자료형("bfloat16" / "float16" 등)
      max_model_len           : 최대 컨텍스트 길이(토큰). 클수록 VRAM을 더 씀
      gpu_memory_utilization  : vLLM이 점유할 VRAM 비율(0~1). 높을수록 공격적
      max_num_batched_tokens  : 한 배치에 묶는 최대 토큰 수(None이면 vLLM 기본값)
      max_num_seqs            : 동시에 처리할 시퀀스(요청) 최대 개수
      tensor_parallel_size    : 텐서 병렬 분할 수(멀티 GPU에서 2 이상)
      enable_lora             : LoRA 어댑터 핫로딩 허용 여부
      max_lora_rank           : 허용할 LoRA rank 상한
      max_loras               : 동시에 올릴 수 있는 LoRA 어댑터 개수
      enforce_eager           : True면 CUDA graph를 끔 → 속도는 조금 손해지만
                                VRAM을 아낀다(그래서 5090에서 True)
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
    """
    임베딩 모델 설정. RAG(장기 메모리 검색)에 쓰는 e5-large 전용이다.

    이 모델은 추론용 대형 모델과 달리 항상 GPU에 상주(always-on)한다.
    크기가 매우 작아(약 700MB) 상주해도 부담이 없고, 메모리 검색이 수시로
    일어나므로 매번 로드/언로드하면 오히려 느려지기 때문이다.

    필드 설명:
      dimension       : 임베딩 벡터 차원(pgvector 컬럼 차원과 일치해야 함)
      max_batch_size  : 한 번에 임베딩할 최대 문장 수
      device          : 상주시킬 GPU 장치("cuda:0")
      vram_gb         : 예상 VRAM 점유량(용량 계산 시 참고값)
    """

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
    """
    LoRA/QLoRA 파인튜닝(추가 학습) 설정.

    VRAM이 작을수록 메모리를 아끼는 방향(qlora + 작은 batch + gradient
    checkpointing)으로, 클수록 품질과 속도를 노리는 방향(lora/full + 큰 batch)
    으로 티어별 값이 달라진다.

    필드 설명:
      method                        : 학습 방식("qlora"=4bit 양자화 학습,
                                      "lora"=일반 LoRA, "full"=전체 파인튜닝)
      lora_rank / lora_alpha        : LoRA 어댑터 용량과 스케일 계수
      lora_dropout                  : LoRA 드롭아웃 비율(과적합 완화)
      batch_size                    : 스텝당 배치 크기
      gradient_accumulation_steps   : 그래디언트 누적 스텝(실효 배치 = batch ×
                                      이 값). 작은 GPU에서 크게 잡아 메모리 절약
      gradient_checkpointing        : 활성화 재계산으로 VRAM을 아끼는 옵션
                                      (속도는 손해)
      max_seq_length                : 학습 시퀀스 최대 길이
      learning_rate / num_epochs / warmup_ratio : 학습률·에폭 수·워밍업 비율
    """

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
    한 GPU 티어에 대한 "완성된" 설정 묶음. get_tier_config()가 이 객체를 만들어
    반환하며, 서빙·학습 코드가 이걸 그대로 참조한다.

    primary(주 추론 모델), auxiliary(보조 모델), embedding(임베딩 모델),
    training(학습 설정)을 한데 담는다.

    필드 설명:
      tier               : 어떤 GPUTier에 대한 설정인지
      primary            : 주 모델 스펙(예: qwen3.5-27b)
      auxiliary          : 보조 모델 스펙(한국어 특화 등). 없으면 None
      embedding          : 임베딩 모델 스펙(기본값 자동 생성)
      training           : 학습 스펙(기본값 자동 생성)
      concurrent_models  : True면 primary와 auxiliary를 동시에 VRAM에 올릴 수
                           있고, False면 VRAM이 부족해 필요할 때마다 한쪽을
                           내리고 다른 쪽을 올리는 hot-swap 방식으로 전환한다
      notes              : 사람이 읽는 메모(왜 이렇게 잡았는지 등)
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
    현재 머신에 장착된 GPU를 실제로 조사해서 알맞은 GPUTier를 돌려준다.

    반환: GPUTier enum 하나
    예외: torch 미설치 또는 CUDA GPU 부재 시 RuntimeError

    왜 자동 감지인가: 사용자가 GPU 스펙을 손으로 입력하지 않아도, 이 함수가
    감지한 티어를 get_tier_config()에 넘기면 최적 설정이 자동으로 적용된다.

    판정 흐름(위에서부터 순서대로 검사):
      1) GPU가 2장 이상  → MULTI_GPU
      2) VRAM > 120GB    → H200
      3) VRAM > 60GB     → 이름에 "a100"이 있으면 A100, 아니면 H100
      4) 그 외(작은 VRAM) → RTX_5090
    """
    # torch는 무거운 의존성이라 모듈 최상단이 아니라 여기서 지연 import 한다.
    # 덕분에 torch가 없는 Machine A에서도 이 파일 자체는 import 할 수 있다.
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch가 설치되지 않았습니다. GPU 서버에는 CUDA 지원 torch가 필요합니다."
        ) from e

    # CUDA로 접근 가능한 GPU가 하나도 없으면 티어 판정 자체가 불가능하다.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 감지할 수 없습니다. "
            "NVIDIA 드라이버와 CUDA 툴킷 설치를 확인하세요."
        )

    # 0번 GPU의 속성을 기준으로 VRAM 용량과 이름을 읽는다.
    gpu_count = torch.cuda.device_count()
    device_props = torch.cuda.get_device_properties(0)
    vram_gb = device_props.total_mem / (1024**3)  # 바이트 → GiB 변환
    gpu_name = device_props.name

    logger.info(f"GPU 감지: {gpu_name}, VRAM={vram_gb:.1f}GB, 개수={gpu_count}")

    # (1) GPU가 2장 이상이면 개별 용량과 무관하게 멀티 GPU 티어로 본다.
    #     모든 카드의 VRAM을 합산해 로그로 남겨 둔다(디버깅 참고용).
    if gpu_count >= 2:
        total_vram = sum(
            torch.cuda.get_device_properties(i).total_mem / (1024**3)
            for i in range(gpu_count)
        )
        logger.info(f"멀티 GPU: {gpu_count}개, 총 VRAM={total_vram:.1f}GB")
        return GPUTier.MULTI_GPU
    # (2) 단일 GPU인데 VRAM이 120GB를 넘으면 H200급 대용량으로 본다.
    elif vram_gb > 120:
        return GPUTier.H200
    # (3) 60GB 초과 ~ 120GB 이하는 80GB급(A100 또는 H100)이다.
    elif vram_gb > 60:
        # A100 80GB 와 H100 80GB 는 VRAM 이 동일해 용량만으로는 구분되지 않는다.
        # 따라서 GPU 이름에 "A100" 이 포함되면 A100 으로 판정하고,
        # 그렇지 않으면 기존대로 H100 으로 폴백한다(Ampere vs Hopper 구분).
        if "a100" in gpu_name.lower():
            return GPUTier.A100
        return GPUTier.H100
    # (4) 위 어디에도 해당하지 않는 작은 VRAM은 기본 타겟인 RTX 5090으로 처리.
    else:
        return GPUTier.RTX_5090


def get_tier_config(tier: GPUTier) -> GPUTierConfig:
    """
    GPUTier를 받아, 그 티어에 미리 튜닝해 둔 GPUTierConfig를 돌려준다.

    매개변수:
      tier : detect_gpu_tier()가 반환한(또는 설정에서 지정한) GPU 등급
    반환:
      해당 티어의 primary/auxiliary/embedding/training이 모두 채워진 설정

    아래 configs 딕셔너리에 티어별 값이 하드코딩되어 있다. 값이 티어마다
    다른 이유(VRAM 예산)를 각 블록 위 주석에 적어 두었으니 함께 참고할 것.
    새 GPU를 지원하려면 GPUTier에 항목을 추가하고 여기에 블록을 하나 더 넣으면 된다.
    """
    configs: dict[GPUTier, GPUTierConfig] = {
        # ── RTX 5090 (32GB) — VRAM이 가장 빡빡한 티어 ──
        # 27B 모델을 32GB에 욱여넣어야 하므로 AWQ 4-bit 양자화가 필수이고,
        # enforce_eager=True(CUDA graph off) + max_num_seqs=1로 메모리를 최대한
        # 아낀다. primary/auxiliary 동시 로딩은 불가라 hot-swap으로 전환한다.
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
        # ── A100 (80GB) — BF16 풀 프리시전(Ampere) ──
        # H100 80GB 와 VRAM 동일 → 프로파일을 거의 공유한다. 다만 A100 은
        # Hopper 가 아니라 Ampere 라 FP8/Transformer Engine 이 없으므로
        # 양자화 없이 BF16 으로만 서빙한다(H100 도 여기서는 NONE 이라 동일).
        # H100 대비 메모리 대역폭(HBM2e)이 낮아 max_num_seqs 를 보수적으로 둔다.
        GPUTier.A100: GPUTierConfig(
            tier=GPUTier.A100,
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
            notes="A100 80GB(Ampere). 58+32>80GB이므로 hot-swap 필요. FP8 미지원.",
        ),
        # ── H100 (80GB) — BF16 풀 프리시전(Hopper) ──
        # A100과 VRAM이 같아 서빙 프로파일은 사실상 동일하다. 다만 H100은
        # Hopper 세대라 대역폭·FP8 등에서 유리하다. 27B+32B 합이 80GB를 넘어
        # 여기서도 동시 로딩은 불가(concurrent_models=False, hot-swap).
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
        # ── H200 (141GB) — 두 모델 동시 로딩 가능 ──
        # VRAM이 커서 primary+auxiliary를 한꺼번에 올릴 수 있다
        # (concurrent_models=True). 컨텍스트 16K, max_num_seqs 8 등 설정을
        # 공격적으로 키워 최대 성능을 낸다.
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
        # ── Multi-GPU (2장 이상) — Tensor Parallelism ──
        # tensor_parallel_size=2로 한 모델을 여러 카드에 쪼개 올린다. 합산
        # VRAM이 넉넉해 컨텍스트 32K, full fine-tune까지 가능한 최상위 프로파일.
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
