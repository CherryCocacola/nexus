#!/usr/bin/env python3
"""
Qwen 3.5 27B LoRA Phase 2 학습 스크립트 — GPU 서버(Machine B) 전용.

[이 파일의 역할]
Nexus의 메인 추론 모델인 Qwen 3.5 27B에 LoRA(저랭크 어댑터)를 붙여
지도학습(SFT)으로 미세조정하는 실행 스크립트다. Phase 1에서 검증한
학습 설정을 그대로 이어받되, 서브에이전트 판단 시나리오가 추가된
새 bootstrap 데이터로 재학습하는 것이 Phase 2의 목적이다.

[Phase 1 대비 변경점]
  - 데이터: 신규 bootstrap_data.jsonl (도구 60% / 추론 30% / 서브에이전트 10%)
  - 출력: /opt/nexus-gpu/checkpoints/qwen35-phase2/
  - phase 레이블: "phase2"
  - 학습률과 에포크는 Phase 1 검증 설정 유지 (lr=3e-4, epochs=3)

[전체 흐름]
  1) unsloth로 4bit 양자화 로드 후 LoRA 어댑터 부착
  2) JSONL 대화 데이터를 Qwen ChatML 형식 문자열로 변환
  3) trl의 SFTTrainer로 3 에포크 학습
  4) 어댑터·토크나이저 저장 + metadata.json 기록

[주의]
실행 전 vLLM(포트 8001)을 반드시 중단해야 한다. 학습과 추론이
동일 GPU의 VRAM을 공유할 수 없기 때문이다.

[의존 라이브러리] unsloth, datasets, trl (GPU 서버에 사전 설치되어 있음)

사용법 (GPU 서버):
  python3 train_qwen_lora_phase2.py

작성자: 이현수 / 작성일: 2026-07-05
"""

import json
import logging
import os
import time

# 로깅 기본 설정 — 시각/레벨/메시지를 한 줄로 출력한다.
# 학습은 수십 분 걸리므로 진행 상황을 콘솔로 계속 확인할 수 있어야 한다.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train")

# --- 경로 및 상수 설정 -------------------------------------------------------
# 학습은 원본(비양자화) 모델 사용 — AWQ는 추론 전용이라 미세조정에 부적합하다.
MODEL_PATH = "/opt/nexus-gpu/models/qwen3.5-27b"     # 베이스 모델 디렉토리
DATA_PATH = "/opt/nexus-gpu/training/bootstrap_data.jsonl"  # 학습 데이터(JSONL)
OUTPUT_DIR = "/opt/nexus-gpu/checkpoints/qwen35-phase2"      # 어댑터 저장 위치
PHASE_LABEL = "phase2"                                # 로그·메타데이터용 단계 레이블

# --- 모델 로드 + LoRA 부착 ---------------------------------------------------
logger.info("[%s] Loading Qwen 3.5 27B + LoRA...", PHASE_LABEL)
# unsloth는 로드 자체가 무겁고 GPU 서버에만 설치되므로 이 시점에 import 한다.
from unsloth import FastLanguageModel

# 27B 모델을 4bit로 로드해 VRAM 사용량을 줄인다(QLoRA 방식).
# max_seq_length=2048 — 학습 샘플의 최대 토큰 길이 상한.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)
logger.info("Model loaded.")

# 베이스 모델에 LoRA 어댑터를 얹는다. 원본 가중치는 동결되고
# 아래 target_modules의 저랭크 행렬만 학습되어 메모리·시간이 크게 절약된다.
#   r=8: LoRA 랭크(작을수록 파라미터↓), lora_alpha=16: 스케일 계수
#   lora_dropout=0.05: 과적합 방지용 드롭아웃
model = FastLanguageModel.get_peft_model(
    model,
    r=8, lora_alpha=16, lora_dropout=0.05,
    # 어텐션(q/k/v/o)과 MLP(gate/up/down) 투영층 모두를 학습 대상으로 지정
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",                              # 바이어스는 학습하지 않음
    use_gradient_checkpointing="unsloth",     # 메모리 절약용 체크포인팅
)
logger.info("LoRA applied.")

# --- 데이터 로드 -------------------------------------------------------------
# 신규 bootstrap 데이터를 한 줄씩 읽는다(JSONL: 한 줄 = 하나의 JSON 객체).
# 분포: 도구 60% / 추론 30% / 서브에이전트 10%.
samples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():                 # 빈 줄은 건너뛴다
            samples.append(json.loads(line))

# 학습에 쓸 대화만 추린다 — messages가 최소 2개(사용자+어시스턴트) 이상인 것만.
# 한 쪽만 있는 샘플은 SFT 학습쌍이 성립하지 않으므로 제외한다.
conversations = [{"messages": s["messages"]} for s in samples if len(s.get("messages", [])) >= 2]
logger.info("Data: %d conversations", len(conversations))

# 카테고리 분포 로깅 — 데이터가 의도한 비율대로 구성됐는지 눈으로 확인하기 위함.
from collections import Counter

cat_counts = Counter(s.get("metadata", {}).get("category", "?") for s in samples)
for cat, cnt in sorted(cat_counts.items()):
    logger.info("  %-40s %d", cat, cnt)

# --- ChatML 변환 준비 --------------------------------------------------------
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# Qwen의 대화 구분 특수 토큰 — 각 발화를 <|im_start|>role ... <|im_end|>로 감싼다.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_conv(ex):
    """대화 하나를 Qwen ChatML 형식의 단일 문자열로 변환한다 (tool_calls 포함).

    SFTTrainer는 "text" 필드의 평문을 학습하므로, 구조화된 messages 배열을
    모델이 실제로 보게 될 프롬프트 형식(ChatML)으로 직렬화해야 한다.

    처리 규칙:
      - user 발화: 내용을 그대로 user 블록으로 감싼다.
      - assistant 발화: tool_calls가 있으면 그 도구 호출을
        {"name", "arguments"} JSON으로 직렬화해 내용으로 사용한다.
        (모델에게 "이 상황에선 이렇게 도구를 호출하라"를 학습시키는 부분)

    매개변수:
      ex: {"messages": [...]} 형태의 대화 한 건.
    반환:
      {"text": "..."} — 발화들을 줄바꿈으로 이어붙인 학습용 문자열.
    """
    parts = []
    for msg in ex["messages"]:
        role = msg["role"]
        content = msg.get("content") or ""      # content가 None이면 빈 문자열로
        if role == "user":
            # 사용자 발화는 원문 그대로 user 블록으로 감싼다.
            parts.append(IM_START + "user\n" + content + IM_END)
        elif role == "assistant":
            # 어시스턴트가 도구를 호출한 경우: 첫 번째 tool_call을 JSON 문자열로 변환.
            # 이 JSON이 곧 모델이 생성하도록 학습되는 "정답 도구 호출"이 된다.
            if "tool_calls" in msg and msg["tool_calls"]:
                tc = msg["tool_calls"][0]
                func = tc.get("function", {})
                content = json.dumps(
                    {"name": func.get("name", ""), "arguments": func.get("arguments", "{}")},
                    ensure_ascii=False,          # 한글이 깨지지 않도록 유니코드 그대로
                )
            parts.append(IM_START + "assistant\n" + content + IM_END)
    return {"text": "\n".join(parts)}


# 리스트를 HuggingFace Dataset으로 만든 뒤 위 변환을 일괄 적용한다.
# 변환 후 원본 messages 컬럼은 제거하고 "text" 컬럼만 남긴다.
dataset = Dataset.from_list(conversations)
dataset = dataset.map(format_conv, remove_columns=["messages"])

# 체크포인트 저장 디렉토리 준비(이미 있으면 그대로 사용).
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 학습 설정 ---------------------------------------------------------------
# Phase 1에서 검증된 하이퍼파라미터를 그대로 유지한다.
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,                  # 전체 데이터 3회 반복 학습
    per_device_train_batch_size=1,       # 27B라 배치는 1로 최소화(VRAM 제약)
    gradient_accumulation_steps=8,       # 8스텝 누적 → 실질 배치 크기 8 효과
    learning_rate=3e-4,                  # Phase 1 검증 학습률
    warmup_steps=10,                     # 초반 10스텝 동안 학습률 점진 증가
    weight_decay=0.01,                   # 가중치 감쇠(과적합 억제)
    logging_steps=10,                    # 10스텝마다 손실 로깅
    save_steps=100,                      # 100스텝마다 체크포인트 저장
    save_total_limit=3,                  # 최근 3개 체크포인트만 보관(디스크 절약)
    fp16=False,
    bf16=True,                           # bfloat16 혼합정밀도(안정성↑)
    max_seq_length=2048,
    dataset_text_field="text",           # 학습 대상 컬럼명
    packing=False,                       # 여러 샘플을 한 시퀀스로 묶지 않음
    report_to="none",                    # 에어갭: 외부 트래킹(wandb 등) 비활성
)

# --- 학습 실행 ---------------------------------------------------------------
start = time.time()                       # 소요 시간 측정 시작
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=sft_config)
result = trainer.train()                  # 실제 학습 수행(가장 오래 걸리는 구간)
logger.info("Training done: %s", result.metrics)

# 학습된 LoRA 어댑터와 토크나이저를 저장한다.
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- 메타데이터 기록 ---------------------------------------------------------
# 이 체크포인트가 어떤 조건으로 만들어졌는지 추적하기 위한 정보.
# 나중에 여러 Phase 결과를 비교·감사할 때 근거 자료가 된다.
meta = {
    "model": "qwen3.5-27b",
    "phase": PHASE_LABEL,
    "epochs": 3,
    "samples": len(conversations),
    "time_sec": time.time() - start,        # 총 학습 소요(초)
    "metrics": result.metrics,
    "dataset_distribution": dict(cat_counts),
    # rationale: Phase 2의 핵심 의도 — 서브에이전트 판단 학습.
    # 서브에이전트 샘플 10%로 Worker가 Agent(subagent_type='scout') 호출과
    # 직접 응답을 언제 구분할지 가르친다. 사소한 요청에 Scout를 남용하지
    # 않도록, 부정 예시(직접 응답 + 단일 도구)를 긍정 예시보다 많이 넣었다.
    "rationale": (
        "Phase 2 introduces subagent decision samples (10%) to teach the Worker "
        "when to invoke Agent(subagent_type='scout') vs. respond directly. "
        "Negative examples (direct_answer + single_tool) outweigh positives "
        "to suppress Scout overuse on trivial requests."
    ),
}
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# 저장 위치와 총 소요 시간(분)을 마지막으로 알린다.
logger.info("Saved to %s (%.1f min)", OUTPUT_DIR, (time.time() - start) / 60)
