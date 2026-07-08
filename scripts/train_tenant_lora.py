#!/usr/bin/env python3
"""
테넌트별 LoRA 학습 CLI 래퍼 (M7, 2026-04-22).

[이 파일이 하는 일]
테넌트(학교·기업 등 고객 단위)마다 별도의 LoRA 어댑터를 학습시키는
명령줄(CLI) 실행 스크립트다. 베이스 모델은 그대로 두고, 테넌트별
소량 데이터로 LoRA 가중치만 미세조정(fine-tuning)한 뒤 체크포인트로 저장한다.

[왜 필요한가]
기존 `train_qwen_lora_phase3.py`는 상수 경로(`qwen35-phase3`)에 체크포인트를
저장하여 default 테넌트 하나만 지원했다. 멀티테넌시(Part 5 Ch 15)에서는
학교·기업별 LoRA를 분리 학습해야 하므로, 이 스크립트가 다음을 담당한다:

  1. `--tenant-id`와 `--phase` 인자로 어댑터 이름·출력 경로를 자동 해석
  2. 테넌트별 데이터 경로를 기본값으로 선택 (--data-path로 덮어쓰기 가능)
  3. 학습 후 metadata.json에 tenant_id·adapter_name을 기록

[주요 구성 요소]
  - parse_args()                    : CLI 인자 정의·파싱
  - convert_tool_calls_for_template : OpenAI tool_calls → 토크나이저 포맷 변환
  - main()                          : 모델 로드→데이터 준비→학습→저장 전 과정

[의존 모듈]
  - core.adapter_naming : 테넌트 ID·경로·어댑터 이름 규칙을 한곳에 모은 모듈
  - unsloth / datasets / trl : 실제 4bit 로드·데이터셋·SFT 학습 (GPU 서버 전용)

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

작성자: 이현수 / 작성일: 2026-07-05
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

from core.adapter_naming import (  # noqa: E402
    MAX_PHASE,
    MIN_PHASE,
    compose_adapter_name,
    compose_data_path,
    compose_output_dir,
    normalize_tenant_id,
)

# 로그 포맷을 "시각 레벨: 메시지" 형태로 통일 — 학습 진행 상황을 콘솔에서 추적한다.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train_tenant")

# 베이스 모델(Qwen 3.5 27B)의 기본 경로. --model-path로 덮어쓸 수 있다.
MODEL_PATH_DEFAULT = "/opt/nexus-gpu/models/qwen3.5-27b"


# ─── Agent 도구 스키마 (phase3과 동일 — apply_chat_template 주입용) ───
# 학습 데이터에 도구 호출(tool_calls)이 있으면 이 스키마를 함께 넣어야
# 토크나이저가 도구 정의 블록까지 포함한 프롬프트를 정확히 재현한다.
# 즉, 실제 추론 시점의 프롬프트 형태와 학습 형태를 일치시키기 위한 장치다.
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
    """명령줄 인자를 정의하고 파싱한다.

    argv를 직접 넘길 수 있게 하여(기본값 None → sys.argv 사용) 테스트에서
    가짜 인자 리스트로 호출하기 쉽게 만들었다.

    반환: argparse.Namespace — tenant_id, phase, data_path, model_path,
          output_dir, epochs, lora_rank, dry_run 필드를 담는다.
    """
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

    [배경] OpenAI 표준 tool_calls는 arguments를 JSON "문자열"로 담는다.
    하지만 Qwen 채팅 템플릿은 arguments를 파이썬 dict로 받아야 각 인자를
    올바른 태그로 펼쳐준다. 그래서 문자열이면 dict로 풀어 다시 담는다.

    매개변수:
      messages : 한 대화의 메시지 리스트 (role/content/tool_calls 딕셔너리)
    반환:
      변환된 새 메시지 리스트 (원본은 수정하지 않고 새로 만들어 반환)
    """
    converted: list[dict] = []
    for msg in messages:
        role = msg["role"]
        # assistant가 도구를 호출한 메시지만 특별 처리한다.
        # 나머지 메시지(user·tool 결과 등)는 손대지 않고 그대로 통과시킨다.
        if role == "assistant" and msg.get("tool_calls"):
            new_tcs = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                # arguments가 JSON 문자열이면 dict로 파싱한다.
                # 파싱 실패 시엔 빈 dict로 안전하게 대체(fail-safe).
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
    """학습 전 과정을 순서대로 실행하는 진입점.

    흐름:
      1) 인자 파싱 후 tenant_id·경로·어댑터 이름을 확정한다.
      2) --dry-run이면 해석 결과만 로그로 남기고 즉시 종료(학습 없음).
      3) 베이스 모델을 4bit로 로드하고 LoRA 어댑터를 붙인다.
      4) JSONL 학습 데이터를 읽어 Qwen 포맷 텍스트로 변환한다.
      5) SFTTrainer로 학습을 돌린 뒤 체크포인트·metadata.json을 저장한다.

    반환: 프로세스 종료 코드(int). 0=성공, 2=데이터 파일 없음.
    """
    args = parse_args(argv)

    # adapter_naming 모듈의 규칙에 따라 테넌트 ID를 정규화하고,
    # 그로부터 어댑터 이름·출력 경로·데이터 경로를 일관되게 유도한다.
    # (--output-dir / --data-path를 직접 주면 그 값을 우선 사용한다.)
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

    # dry-run: 경로·이름이 의도대로 해석됐는지 확인하는 용도.
    # 무거운 모델 로드나 학습을 건너뛰고 바로 성공(0) 반환.
    if args.dry_run:
        logger.info("--dry-run 모드 — 실제 학습 없이 종료")
        return 0

    # 데이터가 없으면 학습이 무의미하므로 조기에 중단(종료코드 2)한다.
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

    # 베이스 모델에 LoRA 어댑터를 부착한다. 어텐션·MLP의 주요 선형층만
    # 대상으로 삼아 학습 파라미터를 최소화한다(메모리·속도 이점).
    # lora_alpha는 관례적으로 rank의 2배로 설정한다.
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
    # JSONL은 한 줄이 한 샘플(대화). 빈 줄은 건너뛰고 dict로 파싱해 모은다.
    samples: list[dict] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    logger.info("Data: %d raw samples", len(samples))

    # 카테고리별 샘플 수를 집계해 데이터 분포를 로그로 남긴다.
    # (특정 카테고리 편향을 눈으로 확인하기 위한 용도)
    cat_counts = Counter(
        s.get("metadata", {}).get("category", "?") for s in samples
    )
    for cat, cnt in sorted(cat_counts.items()):
        logger.info("  %-40s %d", cat, cnt)

    def format_conv(ex):
        """대화 하나를 Qwen3.5 공식 포맷 텍스트로 직렬화.

        데이터셋 map()에서 샘플마다 호출된다. 도구 호출이 포함된 대화면
        AGENT_TOOL_SCHEMA를 tools 인자로 함께 넘겨 프롬프트를 정확히 재현한다.
        반환: {"text": 직렬화된 문자열} — SFTTrainer가 학습할 최종 텍스트.
        """
        # tool_calls를 토크나이저가 이해하는 dict arguments 형태로 변환.
        messages = convert_tool_calls_for_template(ex["messages"])
        # 이 대화에 도구 호출이 하나라도 있으면 도구 스키마를 주입해야 한다.
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

    # 메시지가 2개 미만인 대화(사실상 학습 신호가 없는 것)는 걸러낸다.
    conversations = [
        {"messages": s["messages"]} for s in samples if len(s.get("messages", [])) >= 2
    ]
    # 리스트를 HuggingFace Dataset으로 만든 뒤 각 대화를 텍스트로 변환한다.
    # remove_columns로 원본 messages 열을 제거해 text 열만 남긴다.
    dataset = Dataset.from_list(conversations)
    dataset = dataset.map(format_conv, remove_columns=["messages"])

    logger.info("=== 첫 샘플 미리보기 (500자) ===")
    logger.info(dataset[0]["text"][:500])
    logger.info("=== 끝 ===")

    # 체크포인트를 저장할 디렉토리를 미리 만든다(이미 있어도 에러 없이 통과).
    os.makedirs(output_dir, exist_ok=True)

    # SFT(지도 미세조정) 하이퍼파라미터. batch=1 + 누적 8스텝으로
    # 실질 배치 8을 확보하고, bf16 + 시퀀스 길이 2048로 VRAM을 절약한다.
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

    # 학습 소요 시간 측정을 위해 시작 시각을 기록한다.
    start = time.time()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )
    result = trainer.train()
    logger.info("Training done: %s", result.metrics)

    # 학습된 LoRA 가중치와 토크나이저를 출력 경로에 저장한다.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # M7: metadata에 tenant_id와 어댑터 이름을 기록 — 핫스왑/감사에 활용
    # 나중에 어댑터를 로드·교체하거나 어떤 데이터로 학습됐는지 추적할 때
    # 이 파일 하나만 읽으면 되도록 학습 조건·결과를 함께 남긴다.
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
    # main()이 돌려준 종료 코드를 그대로 프로세스 exit code로 전달한다.
    # (HTTP 트리거 측이 성공/실패를 종료 코드로 판별할 수 있게 함)
    raise SystemExit(main())
