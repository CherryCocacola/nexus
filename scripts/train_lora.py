#!/usr/bin/env python3
"""
LoRA 학습 스크립트 — GPU 서버에서 직접 실행한다.

사용법:
  python3 train_lora.py \
    --model-path /opt/nexus-gpu/models/gemma-4-31b-it-awq \
    --data-path /opt/nexus-gpu/training/bootstrap_data.jsonl \
    --output-dir /opt/nexus-gpu/checkpoints/phase1 \
    --lora-rank 8 \
    --epochs 3 \
    --lr 3e-4

이 스크립트는 GPU 서버(/opt/nexus-gpu/.venv/bin/python3.12)에서 실행된다.
vLLM을 먼저 중단한 후 실행해야 한다 (VRAM 공유 불가).
"""

import argparse
import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("nexus.train_lora")


def parse_args():
    parser = argparse.ArgumentParser(description="Nexus LoRA Training Script")
    parser.add_argument("--model-path", required=True, help="Base model path (AWQ)")
    parser.add_argument("--data-path", required=True, help="Training data JSONL path")
    parser.add_argument("--output-dir", default="./checkpoints/phase1", help="Checkpoint output directory")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    return parser.parse_args()


def load_training_data(data_path):
    """JSONL 학습 데이터를 로드한다."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info("학습 데이터 로드: %d개 샘플", len(samples))
    return samples


def convert_to_chat_format(samples):
    """
    Bootstrap JSONL을 Hugging Face chat 형식으로 변환한다.

    입력: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", ...}]}
    출력: 동일 형식 (이미 호환)
    """
    conversations = []
    for sample in samples:
        messages = sample.get("messages", [])
        if len(messages) >= 2:
            conversations.append({"messages": messages})
    logger.info("채팅 형식 변환: %d개 대화", len(conversations))
    return conversations


def main():
    args = parse_args()
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Nexus LoRA Training — Phase 1 Bootstrap")
    logger.info("=" * 60)
    logger.info("모델: %s", args.model_path)
    logger.info("데이터: %s", args.data_path)
    logger.info("출력: %s", args.output_dir)
    logger.info("LoRA: rank=%d, alpha=%d", args.lora_rank, args.lora_alpha)
    logger.info("학습: epochs=%d, lr=%s, batch=%d, grad_accum=%d",
                args.epochs, args.lr, args.batch_size, args.grad_accum)

    # 데이터 로드
    samples = load_training_data(args.data_path)
    conversations = convert_to_chat_format(samples)

    if not conversations:
        logger.error("학습 데이터가 비어 있습니다.")
        sys.exit(1)

    # 모델 + 토크나이저 로드 (unsloth — Gemma 4 전용 최적화)
    # unsloth은 Gemma4ClippableLinear 등 커스텀 레이어를 자동 처리한다.
    logger.info("모델 로드 중 (unsloth 4bit)...")
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_len,
            load_in_4bit=True,
            dtype=None,  # auto
        )
        logger.info("모델 로드 완료 (unsloth 4bit)")

    except Exception as e:
        logger.error("모델 로드 실패: %s", e)
        sys.exit(1)

    # LoRA 어댑터 적용 (unsloth 방식)
    logger.info("LoRA 어댑터 적용 중 (rank=%d)...", args.lora_rank)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # 데이터셋 준비
    logger.info("데이터셋 준비 중...")
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # 대화를 텍스트로 변환 (Gemma 4 chat template)
    def format_conversation(example):
        """대화를 Gemma 4 형식 텍스트로 변환한다."""
        text_parts = []
        for msg in example["messages"]:
            role = msg["role"]
            content = msg.get("content") or ""
            if role == "user":
                text_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    tc = msg["tool_calls"][0]
                    func = tc.get("function", {})
                    content = json.dumps({
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    }, ensure_ascii=False)
                text_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        return {"text": "\n".join(text_parts)}

    dataset = Dataset.from_list(conversations)
    dataset = dataset.map(format_conversation, remove_columns=["messages"])
    logger.info("데이터셋 준비 완료: %d개 샘플", len(dataset))

    # 학습 실행 (SFTTrainer — unsloth 최적화)
    logger.info("학습 시작...")
    os.makedirs(args.output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    logger.info("학습 완료: %s", metrics)

    # 어댑터 저장
    logger.info("LoRA 어댑터 저장 중...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 메타데이터 저장
    metadata = {
        "phase": "phase1_bootstrap",
        "model_path": args.model_path,
        "data_path": args.data_path,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "total_samples": len(conversations),
        "training_time_seconds": time.time() - start_time,
        "metrics": train_result.metrics,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("학습 완료! 소요 시간: %.1f분", elapsed / 60)
    logger.info("체크포인트: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
