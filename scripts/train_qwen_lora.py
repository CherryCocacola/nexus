#!/usr/bin/env python3
"""
Qwen 3.5 27B LoRA 학습 스크립트 — GPU 서버에서 실행.

사용법:
  python3 train_qwen_lora.py

Qwen 3.5는 ChatML 형식(<|im_start|>/<|im_end|>)을 사용한다.
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
OUTPUT_DIR = "/opt/nexus-gpu/checkpoints/qwen35-phase1"

logger.info("Loading Qwen 3.5 27B + LoRA...")
from unsloth import FastLanguageModel

# 원본 모델을 4bit로 로드하여 32GB VRAM에서 학습 가능
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

# 데이터 로드
samples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

conversations = [{"messages": s["messages"]} for s in samples if len(s.get("messages", [])) >= 2]
logger.info("Data: %d conversations", len(conversations))

from datasets import Dataset
from trl import SFTTrainer, SFTConfig

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_conv(ex):
    """Qwen ChatML 형식으로 변환한다."""
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
    "phase": "phase1",
    "epochs": 3,
    "samples": len(conversations),
    "time_sec": time.time() - start,
    "metrics": result.metrics,
}
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

logger.info("Saved to %s (%.1f min)", OUTPUT_DIR, (time.time() - start) / 60)
