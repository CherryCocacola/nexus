#!/usr/bin/env python3
"""
Qwen 3.5 27B LoRA Phase 2 학습 스크립트 — GPU 서버 전용.

Phase 1 대비 변경:
  - 데이터: 신규 bootstrap_data.jsonl (60/30/10 비율, 서브에이전트 시나리오 포함)
  - 출력: /opt/nexus-gpu/checkpoints/qwen35-phase2/
  - phase 레이블: "phase2"
  - 학습률과 에포크는 Phase 1 검증 설정 유지 (lr=3e-4, epochs=3)

실행 전 vLLM(8001)을 반드시 중단해야 한다 (VRAM 공유 불가).

사용법 (GPU 서버):
  python3 train_qwen_lora_phase2.py
"""

import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train")

# 학습은 원본(비양자화) 모델 사용 — AWQ는 추론 전용
MODEL_PATH = "/opt/nexus-gpu/models/qwen3.5-27b"
DATA_PATH = "/opt/nexus-gpu/training/bootstrap_data.jsonl"
OUTPUT_DIR = "/opt/nexus-gpu/checkpoints/qwen35-phase2"
PHASE_LABEL = "phase2"

logger.info("[%s] Loading Qwen 3.5 27B + LoRA...", PHASE_LABEL)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)
logger.info("Model loaded.")

model = FastLanguageModel.get_peft_model(
    model,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
)
logger.info("LoRA applied.")

# 데이터 로드 (신규 bootstrap: 도구 60% / 추론 30% / 서브에이전트 10%)
samples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

conversations = [{"messages": s["messages"]} for s in samples if len(s.get("messages", [])) >= 2]
logger.info("Data: %d conversations", len(conversations))

# 카테고리 분포 로깅
from collections import Counter

cat_counts = Counter(s.get("metadata", {}).get("category", "?") for s in samples)
for cat, cnt in sorted(cat_counts.items()):
    logger.info("  %-40s %d", cat, cnt)

from datasets import Dataset
from trl import SFTTrainer, SFTConfig

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_conv(ex):
    """Qwen ChatML 형식으로 변환한다 (tool_calls 포함)."""
    parts = []
    for msg in ex["messages"]:
        role = msg["role"]
        content = msg.get("content") or ""
        if role == "user":
            parts.append(IM_START + "user\n" + content + IM_END)
        elif role == "assistant":
            if "tool_calls" in msg and msg["tool_calls"]:
                tc = msg["tool_calls"][0]
                func = tc.get("function", {})
                content = json.dumps(
                    {"name": func.get("name", ""), "arguments": func.get("arguments", "{}")},
                    ensure_ascii=False,
                )
            parts.append(IM_START + "assistant\n" + content + IM_END)
    return {"text": "\n".join(parts)}


dataset = Dataset.from_list(conversations)
dataset = dataset.map(format_conv, remove_columns=["messages"])

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

start = time.time()
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=sft_config)
result = trainer.train()
logger.info("Training done: %s", result.metrics)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

meta = {
    "model": "qwen3.5-27b",
    "phase": PHASE_LABEL,
    "epochs": 3,
    "samples": len(conversations),
    "time_sec": time.time() - start,
    "metrics": result.metrics,
    "dataset_distribution": dict(cat_counts),
    "rationale": (
        "Phase 2 introduces subagent decision samples (10%) to teach the Worker "
        "when to invoke Agent(subagent_type='scout') vs. respond directly. "
        "Negative examples (direct_answer + single_tool) outweigh positives "
        "to suppress Scout overuse on trivial requests."
    ),
}
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

logger.info("Saved to %s (%.1f min)", OUTPUT_DIR, (time.time() - start) / 60)
