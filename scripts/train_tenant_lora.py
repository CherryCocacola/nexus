#!/usr/bin/env python3
"""
테넌트별 LoRA 학습 CLI 래퍼 (M7, 2026-04-22).

기존 `train_qwen_lora_phase3.py`는 상수 경로(`qwen35-phase3`)에 체크포인트를
저장하여 default 테넌트만 지원했다. 멀티테넌시(Part 5 Ch 15)에서는 학교·기업별
LoRA를 분리 학습해야 하므로, 이 스크립트가 다음을 담당한다:

  1. `--tenant-id`와 `--phase` 인자로 어댑터 이름·출력 경로를 자동 해석
  2. 테넌트별 데이터 경로를 기본값으로 선택 (--data-path로 덮어쓰기 가능)
  3. 학습 후 metadata.json에 tenant_id·adapter_name을 기록

사용 예시:
  # 기존 default(Phase 3) 학습 — 경로/어댑터 기존과 동일
  python scripts/train_tenant_lora.py --phase 3

  # dongguk 테넌트용 LoRA 학습
  python scripts/train_tenant_lora.py --tenant-id dongguk --phase 3

  # 데이터 경로 명시적 지정
  python scripts/train_tenant_lora.py \\
      --tenant-id dongguk --phase 3 \\
      --data-path /opt/nexus-gpu/training/dongguk/bootstrap_data.jsonl

실행 전 GPU 서버(vLLM)는 VRAM 점유 때문에 반드시 중단해야 한다.

이 스크립트는 GPU 서버(Machine B)에서 직접 실행하는 것을 가정한다 —
Machine A(오케스트레이터)의 LoRATrainer는 HTTP로 이 스크립트를 트리거한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

# `training.*`과 `core.*`를 import하려면 리포지토리 루트가 PYTHONPATH에 있어야 한다.
# GPU 서버 시작 스크립트가 보장하지만, 단독 실행 시에도 동작하도록 여기서도 세팅.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.adapter_naming import (  # noqa: E402
    MAX_PHASE,
    MIN_PHASE,
    compose_adapter_name,
    compose_data_path,
    compose_output_dir,
    normalize_tenant_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_tenant")

MODEL_PATH_DEFAULT = "/opt/nexus-gpu/models/qwen3.5-27b"


# ─── Agent 도구 스키마 (phase3과 동일 — apply_chat_template 주입용) ───
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


# ─────────────────────────────────────────────
# CLI 인자 파싱
# ─────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="테넌트별 Qwen 3.5 27B LoRA 학습 러너 (M7)"
    )
    parser.add_argument(
        "--tenant-id",
        default=None,
        help="테넌트 식별자. 생략/'default'면 기존 default 테넌트 경로를 쓴다.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        required=True,
        help=f"학습 Phase 번호 ({MIN_PHASE}~{MAX_PHASE})",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help=(
            "학습 데이터 JSONL 경로. 생략 시 tenant_id를 기반으로 자동 해석:\n"
            "  default  → /opt/nexus-gpu/training/bootstrap_data.jsonl\n"
            "  {tenant} → /opt/nexus-gpu/training/{tenant}/bootstrap_data.jsonl"
        ),
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH_DEFAULT,
        help=f"베이스 모델 경로 (기본: {MODEL_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "체크포인트 출력 경로. 생략 시 자동 해석: "
            "/opt/nexus-gpu/checkpoints/qwen35[-{tenant}]-phaseN"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="학습 에폭 수 (기본 3)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (기본 8, phase3과 동일)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="경로/이름 해석만 출력하고 학습은 실행하지 않는다 (검증용).",
    )
    return parser.parse_args(argv)


# ─────────────────────────────────────────────
# 공용 헬퍼: messages → Qwen3.5 공식 포맷 변환
# ─────────────────────────────────────────────
def convert_tool_calls_for_template(messages: list[dict]) -> list[dict]:
    """OpenAI tool_calls 포맷을 tokenizer가 요구하는 dict arguments로 변환.

    Phase3 스크립트와 같은 로직 — 중복을 피하고자 같은 구현을 유지했다.
    arguments가 JSON 문자열이면 dict로 파싱해 chat_template가
    `<parameter=KEY>VALUE</parameter>`를 생성하게 한다.
    """
    converted: list[dict] = []
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


# ─────────────────────────────────────────────
# 메인 — 학습 실행
# ─────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    tenant_id = normalize_tenant_id(args.tenant_id)
    phase_label = f"phase{args.phase}"
    adapter_name = compose_adapter_name(tenant_id, args.phase)
    output_dir = args.output_dir or compose_output_dir(tenant_id, args.phase)
    data_path = args.data_path or compose_data_path(tenant_id)

    logger.info("=" * 60)
    logger.info("M7 테넌트 학습 러너")
    logger.info("  tenant_id     = %s", tenant_id)
    logger.info("  phase         = %d", args.phase)
    logger.info("  adapter_name  = %s", adapter_name)
    logger.info("  data_path     = %s", data_path)
    logger.info("  output_dir    = %s", output_dir)
    logger.info("  model_path    = %s", args.model_path)
    logger.info("  epochs / rank = %d / %d", args.epochs, args.lora_rank)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("--dry-run 모드 — 실제 학습 없이 종료")
        return 0

    if not Path(data_path).exists():
        logger.error("학습 데이터 파일이 없습니다: %s", data_path)
        return 2

    # ─── 모델 로드 ───
    # unsloth·datasets·trl은 GPU 서버에만 설치돼 있으므로 여기서 import.
    # Machine A(오케스트레이터)에서는 실행되지 않는다.
    logger.info("[%s] Loading Qwen 3.5 27B + LoRA...", phase_label)
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    logger.info("Model loaded.")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    logger.info("LoRA applied (r=%d).", args.lora_rank)

    # ─── 데이터 로드 ───
    samples: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    logger.info("Data: %d raw samples", len(samples))

    cat_counts = Counter(
        s.get("metadata", {}).get("category", "?") for s in samples
    )
    for cat, cnt in sorted(cat_counts.items()):
        logger.info("  %-40s %d", cat, cnt)

    def format_conv(ex):
        """Qwen3.5 공식 포맷으로 직렬화."""
        messages = convert_tool_calls_for_template(ex["messages"])
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

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    conversations = [
        {"messages": s["messages"]} for s in samples if len(s.get("messages", [])) >= 2
    ]
    dataset = Dataset.from_list(conversations)
    dataset = dataset.map(format_conv, remove_columns=["messages"])

    logger.info("=== 첫 샘플 미리보기 (500자) ===")
    logger.info(dataset[0]["text"][:500])
    logger.info("=== 끝 ===")

    os.makedirs(output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
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
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    result = trainer.train()
    logger.info("Training done: %s", result.metrics)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # M7: metadata에 tenant_id와 어댑터 이름을 기록 — 핫스왑/감사에 활용
    meta = {
        "model": "qwen3.5-27b",
        "tenant_id": tenant_id,
        "phase": phase_label,
        "adapter_name": adapter_name,
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "samples": len(conversations),
        "time_sec": time.time() - start,
        "metrics": result.metrics,
        "dataset_distribution": dict(cat_counts),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(
        "Saved to %s (%.1f min) — adapter=%s",
        output_dir, (time.time() - start) / 60, adapter_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
