#!/usr/bin/env python3
"""
LoRA 학습 스크립트 — GPU 서버(Machine B)에서 직접 실행하는 단독 실행형 CLI.

무슨 일을 하나?
  1) JSONL 형식의 부트스트랩 학습 데이터를 읽어 들인다.
  2) 이를 Hugging Face 채팅(chat) 형식으로 정리한다.
  3) unsloth로 4bit 양자화된 베이스 모델(Gemma 4 계열)을 로드한다.
  4) 그 위에 LoRA 어댑터를 얹어 SFT(지도 미세조정) 학습을 돌린다.
  5) 학습된 LoRA 어댑터 + 토크나이저 + 메타데이터를 체크포인트로 저장한다.

주요 함수:
  - parse_args()            : 커맨드라인 인자 파싱
  - load_training_data()    : JSONL 데이터 로드
  - convert_to_chat_format(): 샘플을 chat 형식으로 정규화
  - main()                  : 전체 학습 파이프라인 실행(진입점)

의존 라이브러리(모두 GPU 서버 venv에 사전 설치되어 있어야 함):
  - unsloth  : Gemma 4 전용 최적화 + 4bit 로딩 + PEFT LoRA 래퍼
  - datasets : 학습용 Dataset 구성
  - trl      : SFTTrainer / SFTConfig (지도 미세조정 트레이너)

사용법:
  python3 train_lora.py \
    --model-path /opt/nexus-gpu/models/gemma-4-31b-it-awq \
    --data-path /opt/nexus-gpu/training/bootstrap_data.jsonl \
    --output-dir /opt/nexus-gpu/checkpoints/phase1 \
    --lora-rank 8 \
    --epochs 3 \
    --lr 3e-4

실행 환경 / 주의사항:
  - 이 스크립트는 GPU 서버(/opt/nexus-gpu/.venv/bin/python3.12)에서 실행된다.
  - vLLM 추론 서버와 VRAM을 공유할 수 없으므로, 반드시 vLLM을 먼저
    중단(stop)한 뒤에 학습을 시작해야 한다.
  - 에어갭(폐쇄망) 원칙: 외부 네트워크 접근 없이 로컬 모델/데이터만 사용한다.

작성자: 이현수 / 작성일: 2026-07-05
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
    """
    커맨드라인 인자를 정의하고 파싱한다.

    학습에 필요한 모든 설정값(모델 경로, 데이터 경로, 출력 경로,
    LoRA 하이퍼파라미터, 학습 하이퍼파라미터)을 CLI 인자로 받는다.
    하드코딩을 피하고, 실행할 때마다 다른 값을 넣어 여러 Phase를
    돌릴 수 있게 하기 위함이다.

    반환:
      argparse.Namespace — .model_path, .data_path 등 속성으로 접근.
    """
    parser = argparse.ArgumentParser(description="Nexus LoRA Training Script")
    # 필수 경로 인자: 베이스 모델과 학습 데이터 위치 (반드시 지정해야 함)
    parser.add_argument("--model-path", required=True, help="Base model path (AWQ)")
    parser.add_argument("--data-path", required=True, help="Training data JSONL path")
    # 체크포인트(학습 결과)를 저장할 디렉토리. 기본값은 phase1.
    parser.add_argument("--output-dir", default="./checkpoints/phase1", help="Checkpoint output directory")
    # LoRA 하이퍼파라미터:
    #   rank  = 저랭크 행렬의 차원. 클수록 표현력↑, 메모리/과적합 위험↑.
    #   alpha = LoRA 가중치 스케일. 보통 rank의 2배 정도로 둔다.
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    # 학습 하이퍼파라미터:
    #   epochs = 전체 데이터를 몇 번 반복 학습할지.
    #   lr     = 학습률(learning rate). 너무 크면 발산, 너무 작으면 학습 지연.
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    # 배치/시퀀스 설정:
    #   batch-size  = GPU당 한 번에 처리하는 샘플 수 (VRAM 한계로 보통 1).
    #   grad-accum  = 기울기를 몇 스텝 누적한 뒤 가중치를 갱신할지.
    #                 실질 배치 크기 = batch-size × grad-accum 이 된다.
    #   max-seq-len = 한 샘플의 최대 토큰 길이. 초과분은 잘린다.
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    return parser.parse_args()


def load_training_data(data_path):
    """
    JSONL(줄 단위 JSON) 학습 데이터를 읽어 파이썬 리스트로 반환한다.

    JSONL은 한 줄에 JSON 객체 하나가 들어 있는 형식이다. 파일을 한
    줄씩 읽으면서 비어 있지 않은 줄만 json.loads로 파싱해 담는다.
    빈 줄(공백만 있는 줄)은 건너뛰어 파싱 오류를 방지한다.

    매개변수:
      data_path — JSONL 파일 경로.
    반환:
      list[dict] — 각 원소가 한 개의 학습 샘플(dict).
    """
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            # 앞뒤 공백/개행 제거 후, 내용이 있는 줄만 파싱한다.
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info("학습 데이터 로드: %d개 샘플", len(samples))
    return samples


def convert_to_chat_format(samples):
    """
    부트스트랩 JSONL 샘플을 Hugging Face chat 형식으로 정규화한다.

    입력 샘플은 이미 {"messages": [...]} 구조라 형식 자체는 호환되지만,
    여기서는 "학습에 쓸 수 있는 유효한 대화"만 걸러 내는 역할을 한다.
    메시지가 2개 미만(예: user만 있고 assistant 응답이 없는 경우)이면
    학습쌍(질문-답변)이 성립하지 않으므로 제외한다.

    입력: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    출력: 동일 형식이되 messages 길이가 2개 이상인 대화만 담긴 리스트.

    매개변수:
      samples — load_training_data()가 반환한 원본 샘플 리스트.
    반환:
      list[dict] — 유효한 대화만 담은 리스트.
    """
    conversations = []
    for sample in samples:
        messages = sample.get("messages", [])
        # user+assistant 최소 한 쌍(2개) 이상이어야 학습 대상으로 채택.
        if len(messages) >= 2:
            conversations.append({"messages": messages})
    logger.info("채팅 형식 변환: %d개 대화", len(conversations))
    return conversations


def main():
    """
    학습 파이프라인 전체를 순서대로 실행하는 진입점(entry point).

    처리 흐름:
      1) 인자 파싱 및 설정값 로깅
      2) 데이터 로드 → chat 형식 변환 (비어 있으면 즉시 종료)
      3) unsloth로 4bit 베이스 모델 + 토크나이저 로드
      4) LoRA 어댑터 부착 (PEFT)
      5) 대화를 Gemma 4 텍스트 형식으로 변환해 Dataset 구성
      6) SFTTrainer로 학습 실행
      7) 어댑터 / 토크나이저 / 메타데이터 저장 후 소요 시간 로깅

    반환값은 없고, 실패 시 sys.exit(1)로 비정상 종료한다.
    """
    args = parse_args()
    # 전체 학습 소요 시간 측정을 위해 시작 시각을 기록해 둔다.
    start_time = time.time()

    # 실행에 사용된 주요 설정을 로그로 남겨, 나중에 어떤 조건으로
    # 학습했는지 추적할 수 있게 한다(재현성 확보).
    logger.info("=" * 60)
    logger.info("Nexus LoRA Training — Phase 1 Bootstrap")
    logger.info("=" * 60)
    logger.info("모델: %s", args.model_path)
    logger.info("데이터: %s", args.data_path)
    logger.info("출력: %s", args.output_dir)
    logger.info("LoRA: rank=%d, alpha=%d", args.lora_rank, args.lora_alpha)
    logger.info("학습: epochs=%d, lr=%s, batch=%d, grad_accum=%d",
                args.epochs, args.lr, args.batch_size, args.grad_accum)

    # 데이터 로드 후 chat 형식으로 정규화한다.
    samples = load_training_data(args.data_path)
    conversations = convert_to_chat_format(samples)

    # 유효한 대화가 하나도 없으면 학습할 것이 없으므로 즉시 중단한다.
    if not conversations:
        logger.error("학습 데이터가 비어 있습니다.")
        sys.exit(1)

    # 모델 + 토크나이저 로드 (unsloth — Gemma 4 전용 최적화)
    # unsloth은 Gemma4ClippableLinear 등 커스텀 레이어를 자동 처리한다.
    # 4bit 양자화로 로드해 32GB급 GPU에서도 대형 모델을 다룰 수 있게 한다.
    # unsloth import는 무겁고 실패 가능성이 있으므로 함수 안에서 지연 import.
    logger.info("모델 로드 중 (unsloth 4bit)...")
    try:
        from unsloth import FastLanguageModel

        # from_pretrained: 베이스 모델과 토크나이저를 함께 로드한다.
        #   dtype=None → 하드웨어에 맞는 자료형을 자동 선택(auto).
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_len,
            load_in_4bit=True,
            dtype=None,  # auto
        )
        logger.info("모델 로드 완료 (unsloth 4bit)")

    except Exception as e:
        # 경로 오타·라이브러리 부재·VRAM 부족 등으로 실패할 수 있다.
        # 이후 단계가 무의미하므로 에러를 남기고 비정상 종료한다.
        logger.error("모델 로드 실패: %s", e)
        sys.exit(1)

    # LoRA 어댑터 적용 (unsloth 방식)
    # 베이스 모델 가중치는 얼리고(freeze), 작은 저랭크 어댑터만 학습한다.
    # 덕분에 학습 파라미터 수와 VRAM 사용량이 크게 줄어든다.
    logger.info("LoRA 어댑터 적용 중 (rank=%d)...", args.lora_rank)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,               # 저랭크 차원
        lora_alpha=args.lora_alpha,     # 어댑터 스케일 계수
        lora_dropout=0.05,              # 과적합 완화를 위한 드롭아웃
        # target_modules: LoRA를 삽입할 레이어들.
        #   q/k/v/o_proj = 어텐션 투영, gate/up/down_proj = MLP 투영.
        #   트랜스포머의 핵심 선형 계층 전반에 어댑터를 붙인다.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",                            # bias는 학습하지 않음
        use_gradient_checkpointing="unsloth",   # 메모리 절약(재계산 방식)
    )

    # 데이터셋 준비
    # datasets / trl도 무거운 라이브러리라 필요한 시점에 지연 import.
    logger.info("데이터셋 준비 중...")
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # 대화를 텍스트로 변환 (Gemma 4 chat template)
    def format_conversation(example):
        """
        하나의 대화(messages 리스트)를 Gemma 4 채팅 템플릿 문자열로 변환한다.

        Gemma 4는 각 발화를 <start_of_turn>{role}\n{내용}<end_of_turn>
        형태로 감싼다. user는 그대로 role을 쓰고, assistant는 "model"로 쓴다.
        assistant가 도구 호출(tool_calls)을 포함하면, 첫 번째 도구 호출의
        함수 이름/인자를 JSON 문자열로 직렬화해 content 대신 사용한다.
        (모델이 도구 호출 형식을 학습하도록 하기 위함)

        반환:
          {"text": "..."} — 발화들을 개행으로 이어 붙인 최종 학습 텍스트.
        """
        text_parts = []
        for msg in example["messages"]:
            role = msg["role"]
            # content가 None인 경우(예: 도구 호출만 있는 메시지)를 대비해 "" 처리.
            content = msg.get("content") or ""
            if role == "user":
                text_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                # 도구 호출이 있으면 함수명+인자를 JSON으로 만들어 본문을 대체.
                if "tool_calls" in msg and msg["tool_calls"]:
                    tc = msg["tool_calls"][0]
                    func = tc.get("function", {})
                    content = json.dumps({
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    }, ensure_ascii=False)
                text_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        return {"text": "\n".join(text_parts)}

    # 대화 리스트를 HF Dataset으로 만든 뒤, 각 행을 텍스트로 변환한다.
    # remove_columns=["messages"]: 변환 후 원본 messages 열은 제거해
    # 최종적으로 "text" 열만 남긴다(트레이너가 이 열을 사용).
    dataset = Dataset.from_list(conversations)
    dataset = dataset.map(format_conversation, remove_columns=["messages"])
    logger.info("데이터셋 준비 완료: %d개 샘플", len(dataset))

    # 학습 실행 (SFTTrainer — unsloth 최적화)
    logger.info("학습 시작...")
    # 출력 디렉토리를 미리 생성(이미 있으면 그대로 둔다).
    os.makedirs(args.output_dir, exist_ok=True)

    # SFTConfig: 지도 미세조정에 필요한 모든 하이퍼파라미터 묶음.
    sft_config = SFTConfig(
        output_dir=args.output_dir,                          # 체크포인트 저장 위치
        num_train_epochs=args.epochs,                        # 학습 반복 횟수
        per_device_train_batch_size=args.batch_size,         # GPU당 배치 크기
        gradient_accumulation_steps=args.grad_accum,         # 기울기 누적 스텝
        learning_rate=args.lr,                               # 학습률
        warmup_ratio=0.03,      # 초반 3% 스텝은 학습률을 서서히 올림(워밍업)
        weight_decay=0.01,      # 가중치 감쇠(과적합 완화 정규화)
        logging_steps=10,       # 10스텝마다 학습 로그 출력
        save_steps=100,         # 100스텝마다 중간 체크포인트 저장
        save_total_limit=3,     # 체크포인트는 최근 3개까지만 보관(디스크 절약)
        fp16=False,             # fp16 비활성화
        bf16=True,              # bf16 사용(수치 안정성이 더 좋음)
        max_seq_length=args.max_seq_len,   # 최대 시퀀스 길이
        dataset_text_field="text",         # 학습에 사용할 데이터셋 열 이름
        packing=False,          # 여러 샘플을 한 시퀀스로 묶지 않음(패킹 비활성)
        report_to="none",       # 외부 실험추적 도구로 보고 안 함(에어갭 준수)
    )

    # SFTTrainer: 모델·토크나이저·데이터셋·설정을 묶어 실제 학습을 수행.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    # train(): 실제 미세조정 실행. 반환 객체의 .metrics에 손실 등 지표가 담김.
    train_result = trainer.train()
    metrics = train_result.metrics
    logger.info("학습 완료: %s", metrics)

    # 어댑터 저장
    # 베이스 모델 전체가 아니라 학습된 LoRA 어댑터 가중치만 저장된다.
    # 추론 시에는 베이스 모델에 이 어댑터를 얹어 사용한다.
    logger.info("LoRA 어댑터 저장 중...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 메타데이터 저장
    # 나중에 "이 체크포인트가 어떤 조건으로 학습됐는지" 알 수 있도록
    # 학습 조건과 결과 지표를 metadata.json 파일로 함께 남긴다.
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

    # 시작 시각과 비교해 전체 소요 시간을 분 단위로 로그에 남긴다.
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("학습 완료! 소요 시간: %.1f분", elapsed / 60)
    logger.info("체크포인트: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    # 이 파일을 직접 실행했을 때만 main()을 호출한다.
    # (다른 모듈에서 import될 때는 자동 실행되지 않도록 하는 파이썬 관용구)
    main()
