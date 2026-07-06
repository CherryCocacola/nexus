#!/usr/bin/env python3
"""
Qwen 3.5 27B LoRA 학습 스크립트 — GPU 서버(Machine B)에서 직접 실행하는 배치 잡.

이 스크립트가 하는 일 (전체 흐름 한눈에):
  1) 원본(비양자화) Qwen 3.5 27B 모델을 4bit로 로드하고 그 위에 LoRA 어댑터를 붙인다.
  2) bootstrap_data.jsonl(대화 학습 데이터)을 읽어 Qwen ChatML 형식 텍스트로 변환한다.
  3) TRL의 SFTTrainer로 지도학습(SFT)을 3 epoch 수행한다.
  4) 학습된 LoRA 어댑터 + 토크나이저 + 메타데이터를 체크포인트 디렉토리에 저장한다.

주요 구성 요소:
  - format_conv(): 대화 1건을 Qwen ChatML(<|im_start|>/<|im_end|>) 텍스트로 변환하는 함수.
  - MODEL_PATH / DATA_PATH / OUTPUT_DIR: 서버상의 입출력 경로 상수.

의존 라이브러리:
  - unsloth(FastLanguageModel): 4bit 로딩 + LoRA 적용 + 학습 최적화.
  - datasets(Dataset), trl(SFTTrainer, SFTConfig): 데이터셋 구성과 SFT 학습 루프.

실행 환경:
  - Machine B(GPU 서버)에서 로컬로 실행하는 순차 스크립트(에어갭). CLI 인자는 없다.
  - RTX 5090 32GB VRAM 기준으로 4bit + LoRA + gradient checkpointing 조합으로 맞춰져 있다.

사용법:
  python3 train_qwen_lora.py

참고: Qwen 3.5는 ChatML 형식(<|im_start|>/<|im_end|>)의 프롬프트 규약을 사용한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

# 표준 라이브러리만 상단에서 import한다. 무거운 학습용 패키지(unsloth 등)는
# 로그를 먼저 남기기 위해 아래에서 필요한 시점에 늦게 import한다.
import json
import logging
import os
import time

# 학습 진행 상황을 시간·레벨과 함께 콘솔에 남기도록 기본 로깅을 설정한다.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train")

# 입출력 경로 상수 (GPU 서버 로컬 기준).
# 학습은 원본(비양자화) 모델을 사용한다 — AWQ 양자화본은 추론 전용이라 학습에 부적합.
MODEL_PATH = "/opt/nexus-gpu/models/qwen3.5-27b"       # 학습 대상 원본 모델 디렉토리
DATA_PATH = "/opt/nexus-gpu/training/bootstrap_data.jsonl"  # JSONL 학습 데이터
OUTPUT_DIR = "/opt/nexus-gpu/checkpoints/qwen35-phase1"      # 체크포인트 저장 위치

logger.info("Loading Qwen 3.5 27B + LoRA...")
# unsloth는 로딩이 무겁고 CUDA 초기화를 유발하므로 여기서 지연 import한다.
from unsloth import FastLanguageModel

# 원본 모델을 4bit로 로드한다 — 27B를 32GB VRAM 한 장에 올려 학습하기 위한 핵심 설정.
# max_seq_length=2048: 학습 시 최대 토큰 길이. 이보다 긴 샘플은 잘린다.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)
logger.info("Model loaded.")

# 로드한 모델에 LoRA 어댑터를 부착한다. 원본 가중치는 동결되고 저차원 어댑터만 학습된다.
#   - r=8: LoRA 랭크(어댑터 용량). 작을수록 파라미터·VRAM이 적게 든다.
#   - lora_alpha=16: 어댑터 출력 스케일 계수(보통 r의 2배로 둔다).
#   - lora_dropout=0.05: 과적합 방지를 위한 드롭아웃.
#   - target_modules: LoRA를 삽입할 트랜스포머 하위 모듈들
#     (어텐션 q/k/v/o + MLP gate/up/down 프로젝션).
#   - bias="none": 바이어스는 학습하지 않는다.
#   - use_gradient_checkpointing="unsloth": 메모리를 아끼려고 활성값을 재계산하는
#     unsloth 최적화 경로 사용(속도 약간 손해, VRAM 크게 절약).
model = FastLanguageModel.get_peft_model(
    model,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
)
logger.info("LoRA applied.")

# 데이터 로드 — JSONL을 한 줄씩 읽어 dict 리스트로 만든다. 빈 줄은 건너뛴다.
samples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

# 학습에 쓸 대화만 추린다: messages가 2턴 이상(최소 user+assistant)인 샘플만 사용한다.
conversations = [{"messages": s["messages"]} for s in samples if len(s.get("messages", [])) >= 2]
logger.info("Data: %d conversations", len(conversations))

# 데이터셋·학습 관련 패키지도 로깅 이후 시점에 지연 import한다.
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# Qwen ChatML 특수 토큰 — 각 발화의 시작/끝을 감싸는 구분자.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_conv(ex):
    """대화 1건(ex["messages"])을 Qwen ChatML 형식의 단일 학습 텍스트로 변환한다.

    각 메시지를 역할별로 <|im_start|>{role}\\n{content}<|im_end|> 블록으로 감싼 뒤
    줄바꿈으로 이어 붙인다. assistant 메시지가 tool_calls를 포함하면 본문 대신
    첫 번째 도구 호출을 {"name", "arguments"} JSON 문자열로 직렬화해 학습시킨다.
    (Nexus 내부 표준인 OpenAI tool_calls 형식을 모델이 출력하도록 가르치기 위함.)

    매개변수:
        ex: {"messages": [...]} 형태의 대화 한 건.
    반환:
        {"text": "..."} — 학습에 그대로 쓰는 ChatML 문자열을 담은 dict.
    """
    parts = []
    for msg in ex["messages"]:
        role = msg["role"]
        # content가 None이거나 없을 수 있으므로 안전하게 빈 문자열로 대체한다.
        content = msg.get("content") or ""
        if role == "user":
            # 사용자 발화는 그대로 user 블록으로 감싼다.
            parts.append(IM_START + "user\n" + content + IM_END)
        elif role == "assistant":
            # assistant가 도구를 호출한 경우: 텍스트 대신 도구 호출 JSON을 본문으로 사용한다.
            if "tool_calls" in msg and msg["tool_calls"]:
                tc = msg["tool_calls"][0]  # 첫 번째 도구 호출만 사용
                func = tc.get("function", {})
                # 도구 이름과 인자를 JSON 문자열로 직렬화(한글 보존을 위해 ensure_ascii=False).
                content = json.dumps(
                    {"name": func.get("name", ""), "arguments": func.get("arguments", "{}")},
                    ensure_ascii=False,
                )
            parts.append(IM_START + "assistant\n" + content + IM_END)
    # 모든 블록을 줄바꿈으로 연결해 하나의 학습 샘플 텍스트로 만든다.
    return {"text": "\n".join(parts)}


# 대화 리스트를 HuggingFace Dataset으로 만든 뒤, 위 format_conv로 text 컬럼을 생성한다.
# 원본 messages 컬럼은 학습에 불필요하므로 제거한다.
dataset = Dataset.from_list(conversations)
dataset = dataset.map(format_conv, remove_columns=["messages"])

# 체크포인트 출력 디렉토리를 미리 만든다(이미 있으면 그대로 둔다).
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SFT(지도 미세조정) 하이퍼파라미터 설정.
#   - num_train_epochs=3: 데이터 전체를 3회 반복 학습.
#   - per_device_train_batch_size=1 + gradient_accumulation_steps=8:
#     실질 배치 크기 8. VRAM 한계 때문에 물리 배치는 1로 두고 그래디언트를 누적한다.
#   - learning_rate=3e-4 / warmup_steps=10 / weight_decay=0.01: 학습률·워밍업·정규화.
#   - logging_steps=10 / save_steps=100 / save_total_limit=3: 로그·저장 주기와 보관 개수.
#   - bf16=True, fp16=False: RTX 5090에서 안정적인 bfloat16 혼합정밀 학습 사용.
#   - packing=False: 짧은 샘플을 한 시퀀스로 묶지 않는다(형식 왜곡 방지).
#   - report_to="none": 외부 리포팅(W&B 등) 비활성 — 에어갭 준수.
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    fp16=False,
    bf16=True,
    max_seq_length=2048,
    dataset_text_field="text",
    packing=False,
    report_to="none",
)

# 전체 학습 소요 시간을 재기 위한 시작 시각.
start = time.time()
# SFTTrainer가 실제 학습 루프(순전파·역전파·옵티마이저 스텝)를 담당한다.
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=sft_config)
result = trainer.train()  # 학습 실행 — 완료 후 손실 등 metrics를 반환한다.
logger.info("Training done: %s", result.metrics)

# 학습된 LoRA 어댑터 가중치와 토크나이저를 출력 디렉토리에 저장한다.
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 이번 학습을 추적·재현할 수 있도록 메타데이터를 함께 남긴다
# (모델·단계·epoch 수·샘플 수·소요 시간·metrics).
meta = {
    "model": "qwen3.5-27b",
    "phase": "phase1",
    "epochs": 3,
    "samples": len(conversations),
    "time_sec": time.time() - start,
    "metrics": result.metrics,
}
# 메타데이터를 사람이 읽기 쉬운 들여쓰기 JSON으로 저장(한글 보존).
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# 저장 위치와 총 소요 시간(분)을 마지막으로 로그에 남긴다.
logger.info("Saved to %s (%.1f min)", OUTPUT_DIR, (time.time() - start) / 60)
