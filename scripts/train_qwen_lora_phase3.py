#!/usr/bin/env python3
"""
Qwen 3.5 27B LoRA Phase 3 학습 스크립트.

Phase 2 대비 개선점:
  1. tokenizer.apply_chat_template(...) 로 직렬화 — tool_calls가 Qwen3.5의
     공식 XML 포맷(<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...)
     으로 자동 변환된다. vLLM qwen3_xml 파서와 호환.
  2. 장문 지식 샘플(knowledge_explanation) 카테고리 학습 — 긴 설명 요청에
     구조화된 3~6단락 답변이 나오도록.
  3. tools 파라미터로 Agent 스키마를 apply_chat_template에 전달 — Worker가
     Agent 도구의 존재를 학습 시점부터 인지.

실행 전 vLLM(8001)을 반드시 중단해야 한다 (VRAM 공유 불가).
"""

import json
import logging
import os
import time
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train")

MODEL_PATH = "/opt/nexus-gpu/models/qwen3.5-27b"
DATA_PATH = "/opt/nexus-gpu/training/bootstrap_data.jsonl"
OUTPUT_DIR = "/opt/nexus-gpu/checkpoints/qwen35-phase3"
PHASE_LABEL = "phase3"


# ─── Agent 도구 스키마 (학습 시 apply_chat_template의 tools 인자로 전달) ───
# 이 스키마를 tokenizer가 chat_template.jinja에 주입하여 Worker가 학습 단계에서
# 도구의 존재를 학습한다. 실 서빙 때는 vLLM이 같은 경로로 도구를 주입한다.
AGENT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Agent",
        "description": (
            "Delegate a task to a sub-agent. Use subagent_type to select a "
            "specialized agent (e.g. 'scout' for read-only exploration)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Task for the sub-agent",
                },
                "subagent_type": {
                    "type": "string",
                    "description": "Sub-agent name (e.g. 'scout')",
                },
                "description": {
                    "type": "string",
                    "description": "Fallback role description for ad-hoc sub-agent",
                },
            },
            "required": ["prompt"],
        },
    },
}


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


# ─── 데이터 로드 ───
samples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

logger.info("Data: %d raw samples", len(samples))

# 카테고리 분포 로깅
cat_counts = Counter(s.get("metadata", {}).get("category", "?") for s in samples)
for cat, cnt in sorted(cat_counts.items()):
    logger.info("  %-40s %d", cat, cnt)


# ─── messages → Qwen3.5 공식 포맷 텍스트 변환 ───
def convert_tool_calls_for_template(messages):
    """
    bootstrap_generator가 만든 OpenAI 포맷 tool_calls를 tokenizer가 요구하는
    포맷으로 변환한다. 핵심: arguments를 JSON 문자열이 아닌 dict로 넘겨야
    chat_template이 <parameter=KEY>VALUE</parameter> 형태로 올바르게 전개한다.
    """
    converted = []
    for msg in messages:
        role = msg["role"]
        if role == "assistant" and msg.get("tool_calls"):
            new_tcs = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                new_tcs.append({
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args,
                    },
                })
            converted.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": new_tcs,
            })
        else:
            converted.append(msg)
    return converted


def format_conv(ex):
    """tokenizer.apply_chat_template으로 Qwen3.5 공식 포맷 텍스트 생성."""
    messages = convert_tool_calls_for_template(ex["messages"])

    # tool_calls가 있는 assistant가 포함된 경우에만 tools 스키마를 넘긴다.
    # 이렇게 하면 일반 샘플에는 시스템 프롬프트에 불필요한 도구 설명이 들어가지 않고,
    # 도구 호출 샘플에는 Worker가 Agent 도구의 존재를 인지한 상태로 학습된다.
    has_tool_call = any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in messages
    )
    tools_arg = [AGENT_TOOL_SCHEMA] if has_tool_call else None

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools_arg,
    )
    return {"text": text}


# ─── Dataset 구성 ───
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

conversations = [
    {"messages": s["messages"]} for s in samples if len(s.get("messages", [])) >= 2
]
dataset = Dataset.from_list(conversations)
dataset = dataset.map(format_conv, remove_columns=["messages"])

# 샘플 미리보기 (처음 1건)
logger.info("=== 첫 샘플 미리보기 (500자) ===")
logger.info(dataset[0]["text"][:500])
logger.info("=== 끝 ===")


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
        "Phase 3 adopts tokenizer.apply_chat_template with Qwen3.5 tool schema. "
        "tool_calls are serialized as the official XML block "
        "(<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>), "
        "matching the vLLM qwen3_xml parser. "
        "A new knowledge_explanation category provides long-form answers to "
        "compensate for Phase 2's short-answer regression."
    ),
}
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

logger.info("Saved to %s (%.1f min)", OUTPUT_DIR, (time.time() - start) / 60)
