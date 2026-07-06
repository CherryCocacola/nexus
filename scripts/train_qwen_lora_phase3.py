#!/usr/bin/env python3
"""
Qwen 3.5 27B LoRA Phase 3 학습(파인튜닝) 스크립트.

이 파일이 하는 일(한눈에):
  - GPU 서버(Machine B)에 있는 Qwen 3.5 27B 베이스 모델을 4bit로 로드하고,
    그 위에 LoRA 어댑터를 얹어 bootstrap 학습 데이터로 SFT(지도 미세조정)를
    수행한 뒤, 학습된 LoRA 가중치와 메타데이터를 체크포인트로 저장한다.
  - 즉 "데이터 로드 → 대화 포맷 변환 → LoRA 학습 → 저장"의 1회성 배치
    스크립트다. 함수는 두 개(convert_tool_calls_for_template, format_conv)뿐이고,
    나머지는 위에서 아래로 순차 실행되는 절차형 코드다.

Phase 2 대비 개선점(왜 Phase 3를 새로 만들었나):
  1. tokenizer.apply_chat_template(...) 로 직렬화 — tool_calls가 Qwen3.5의
     공식 XML 포맷(<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...)
     으로 자동 변환된다. vLLM qwen3_xml 파서와 호환된다. Phase 2는 직접
     문자열을 조립했으나, 여기서는 tokenizer가 공식 템플릿으로 만들게 맡긴다.
  2. 장문 지식 샘플(knowledge_explanation) 카테고리 학습 — 긴 설명 요청에
     구조화된 3~6단락 답변이 나오도록. Phase 2에서 답변이 짧아지던
     회귀(short-answer regression)를 보정하려는 목적.
  3. tools 파라미터로 Agent 스키마를 apply_chat_template에 전달 — Worker가
     Agent 도구의 존재를 학습 시점부터 인지하도록 한다. 실제 서빙 때 vLLM이
     주입하는 도구 스키마와 학습 시점을 일치시켜 분포 차이를 줄인다.

주요 구성 요소:
  - AGENT_TOOL_SCHEMA : 학습에 주입할 Agent 도구의 OpenAI function 스키마.
  - convert_tool_calls_for_template() : OpenAI tool_calls → 템플릿용 포맷 변환.
  - format_conv() : 대화(messages)를 Qwen3.5 공식 포맷 텍스트로 직렬화.

주의(운영):
  실행 전 vLLM(8001)을 반드시 중단해야 한다. 학습과 추론 서버가 같은 GPU
  VRAM을 동시에 점유할 수 없기 때문이다(VRAM 공유 불가).

작성자: 이현수 / 작성일: 2026-07-05
"""

# 표준 라이브러리만 상단에서 import한다. unsloth/datasets/trl 같은 무거운
# 학습 라이브러리는 실제로 필요한 위치에서 지연 import한다(아래 참조). 무거운
# 라이브러리를 import하는 순간 GPU/CUDA 초기화가 일어나므로, 데이터 점검 전에
# 미리 로드해 두지 않으려는 의도다.
import json  # bootstrap_data.jsonl 파싱 및 metadata.json 저장용
import logging  # 진행 상황 로그 출력
import os  # 출력 디렉토리 생성, 경로 결합
import time  # 학습 소요 시간 측정
from collections import Counter  # 카테고리별 샘플 개수 집계

# 로그 포맷: "시각 레벨: 메시지". INFO 이상을 콘솔에 출력한다.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("train")

# ─── 경로/라벨 상수 (모두 Machine B(GPU 서버)의 절대 경로) ───
MODEL_PATH = "/opt/nexus-gpu/models/qwen3.5-27b"          # 베이스 모델 위치
DATA_PATH = "/opt/nexus-gpu/training/bootstrap_data.jsonl"  # 학습 데이터(JSONL)
OUTPUT_DIR = "/opt/nexus-gpu/checkpoints/qwen35-phase3"   # 체크포인트 저장 위치
PHASE_LABEL = "phase3"                                    # 로그/메타에 찍을 라벨


# ─── Agent 도구 스키마 (학습 시 apply_chat_template의 tools 인자로 전달) ───
# 이 스키마를 tokenizer가 chat_template.jinja에 주입하여 Worker가 학습 단계에서
# 도구의 존재를 학습한다. 실 서빙 때는 vLLM이 같은 경로로 도구를 주입한다.
#
# 형태는 OpenAI function-calling 스키마 그대로다(type=function + function{...}).
# Nexus의 표준 내부 계약이 OpenAI tool_calls 형식이므로 학습 데이터도 같은
# 계약을 따른다. 여기 실린 것은 대표 도구인 Agent 하나뿐인데, 도구 호출의
# "형식(문법)"을 학습시키는 것이 목적이라 전체 도구 목록을 넣을 필요는 없다.
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


# ─── 베이스 모델 + LoRA 어댑터 준비 ───
# unsloth는 여기서 지연 import한다(import 시 CUDA가 초기화되므로).
logger.info("[%s] Loading Qwen 3.5 27B + LoRA...", PHASE_LABEL)
from unsloth import FastLanguageModel

# 베이스 모델과 토크나이저를 로드한다.
#  - max_seq_length=2048 : 한 샘플의 최대 토큰 길이(이보다 길면 잘림).
#  - load_in_4bit=True    : 4bit 양자화 로드로 27B 모델을 단일 GPU VRAM에 적재.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)
logger.info("Model loaded.")

# 로드한 모델에 LoRA(저랭크 어댑터)를 붙인다. 베이스 가중치는 얼리고(freeze)
# 작은 어댑터 행렬만 학습하므로 VRAM과 시간이 크게 절약된다.
#  - r=8            : LoRA 랭크(어댑터 용량). 클수록 표현력↑, 메모리↑.
#  - lora_alpha=16  : 스케일링 계수(보통 r의 2배로 둔다).
#  - lora_dropout   : 과적합 방지용 드롭아웃.
#  - target_modules : 어댑터를 삽입할 선형 계층들(어텐션 q/k/v/o + MLP 3종).
#  - bias="none"    : 바이어스는 학습하지 않음.
#  - use_gradient_checkpointing="unsloth" : 활성값을 저장 대신 재계산해 VRAM 절약.
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
# bootstrap_data.jsonl은 한 줄에 JSON 하나(=학습 샘플 하나)인 JSONL 형식이다.
# 각 샘플은 최소한 "messages"(대화)와 선택적으로 "metadata"(카테고리 등)를 가진다.
samples = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():          # 빈 줄은 건너뛴다
            samples.append(json.loads(line))

logger.info("Data: %d raw samples", len(samples))

# 카테고리 분포 로깅 — 어떤 종류의 샘플이 몇 개씩 있는지 눈으로 확인한다.
# metadata나 category 키가 없으면 "?"로 집계한다(방어적 기본값).
cat_counts = Counter(s.get("metadata", {}).get("category", "?") for s in samples)
for cat, cnt in sorted(cat_counts.items()):
    logger.info("  %-40s %d", cat, cnt)


# ─── messages → Qwen3.5 공식 포맷 텍스트 변환 ───
def convert_tool_calls_for_template(messages):
    """OpenAI 포맷 tool_calls를 chat_template이 요구하는 포맷으로 변환한다.

    왜 필요한가:
      bootstrap_generator가 만든 원본 데이터는 tool_calls의 arguments를 JSON
      "문자열"로 담고 있다(OpenAI API 관례). 그런데 Qwen3.5 chat_template은
      arguments가 파이썬 dict일 때만 <parameter=KEY>VALUE</parameter> 형태로
      올바르게 전개한다. 문자열 그대로 넘기면 통째로 하나의 값처럼 깨져 나온다.
      그래서 여기서 문자열 arguments를 json.loads로 dict로 풀어준다.

    처리 흐름:
      - assistant이면서 tool_calls가 있는 메시지만 손본다.
      - 각 tool_call의 arguments가 문자열이면 dict로 파싱한다. 파싱 실패 시
        빈 dict로 대체한다(깨진 데이터가 학습 전체를 막지 않도록 방어).
      - 그 외 메시지(user/tool_result/도구 없는 assistant 등)는 원본 그대로 통과.

    매개변수:
      messages: 한 대화의 메시지 리스트(각 원소는 role/content 등을 가진 dict).
    반환:
      변환이 끝난 새 메시지 리스트(원본은 수정하지 않음).
    호출처:
      아래 format_conv()에서 apply_chat_template에 넘기기 직전에 사용한다.
    """
    converted = []
    for msg in messages:
        role = msg["role"]
        # 도구를 호출하는 assistant 메시지만 arguments 변환 대상이다.
        if role == "assistant" and msg.get("tool_calls"):
            new_tcs = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                # arguments 기본값은 빈 객체 문자열 "{}" (키가 없을 때 대비).
                args = func.get("arguments", "{}")
                # 문자열이면 dict로 파싱. 깨진 JSON은 빈 dict로 안전 처리.
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                # 템플릿이 기대하는 형태(arguments=dict)로 재조립한다.
                new_tcs.append({
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args,
                    },
                })
            # content가 None이면 빈 문자열로 통일(템플릿이 None을 싫어함).
            converted.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": new_tcs,
            })
        else:
            # 손댈 필요 없는 메시지는 그대로 넘긴다.
            converted.append(msg)
    return converted


def format_conv(ex):
    """대화 한 건을 Qwen3.5 공식 포맷 텍스트로 직렬화한다.

    dataset.map()에서 샘플마다 호출되며, {"text": ...} 형태를 반환한다.
    SFTTrainer가 이 "text" 필드를 그대로 학습 입력으로 사용한다.

    매개변수:
      ex: {"messages": [...]} 형태의 샘플 하나.
    반환:
      {"text": 직렬화된 전체 대화 문자열}.
    """
    # 1) tool_calls arguments를 dict로 정규화한다(위 함수 참조).
    messages = convert_tool_calls_for_template(ex["messages"])

    # 2) tool_calls가 있는 assistant가 포함된 경우에만 tools 스키마를 넘긴다.
    #    이렇게 하면 일반 샘플에는 시스템 프롬프트에 불필요한 도구 설명이 들어가지
    #    않고, 도구 호출 샘플에는 Worker가 Agent 도구의 존재를 인지한 상태로
    #    학습된다(학습 분포를 실제 서빙 상황과 맞추는 효과).
    has_tool_call = any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in messages
    )
    tools_arg = [AGENT_TOOL_SCHEMA] if has_tool_call else None

    # 3) 토크나이저의 공식 chat_template으로 문자열을 만든다.
    #    - tokenize=False           : 토큰 ID가 아닌 사람이 읽는 문자열로 반환.
    #    - add_generation_prompt=False : 학습용이므로 답변 유도 프롬프트를 붙이지
    #      않는다(추론 때와 달리 정답까지 포함된 완결 대화를 그대로 학습).
    #    - tools=tools_arg          : 위에서 정한 도구 스키마(없으면 None).
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools_arg,
    )
    return {"text": text}


# ─── Dataset 구성 ───
# datasets/trl도 무거우므로 이 시점에 지연 import한다.
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# 메시지가 2개 미만인 샘플(예: user만 있고 답변 없음)은 학습 가치가 없어 버린다.
# 최소 user+assistant 한 쌍(2개)이 있어야 유효한 대화로 본다.
conversations = [
    {"messages": s["messages"]} for s in samples if len(s.get("messages", [])) >= 2
]
# HuggingFace Dataset으로 감싼 뒤, 각 샘플을 format_conv로 "text" 문자열로 변환.
# remove_columns=["messages"]로 원본 messages 컬럼을 제거해 text만 남긴다.
dataset = Dataset.from_list(conversations)
dataset = dataset.map(format_conv, remove_columns=["messages"])

# 샘플 미리보기 (처음 1건) — 직렬화가 의도대로 됐는지 눈으로 검증하는 안전장치.
logger.info("=== 첫 샘플 미리보기 (500자) ===")
logger.info(dataset[0]["text"][:500])
logger.info("=== 끝 ===")


# 체크포인트 저장 디렉토리를 미리 만든다(이미 있으면 그대로 사용).
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 학습 하이퍼파라미터 설정 ───
# 각 값의 의미는 아래 인자별 주석 참조.
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,              # 체크포인트/로그 저장 경로
    num_train_epochs=3,                # 전체 데이터를 3회 반복 학습
    per_device_train_batch_size=1,     # GPU당 배치 크기(27B라 1로 최소화)
    gradient_accumulation_steps=8,     # 8스텝 누적 → 유효 배치 크기 8 효과
    learning_rate=3e-4,                # LoRA에 흔히 쓰는 학습률
    warmup_steps=10,                   # 초반 10스텝 동안 학습률을 서서히 올림
    weight_decay=0.01,                 # 가중치 감쇠(과적합 완화)
    logging_steps=10,                  # 10스텝마다 손실 등 로그 출력
    save_steps=100,                    # 100스텝마다 체크포인트 저장
    save_total_limit=3,                # 최근 체크포인트 3개만 유지(디스크 절약)
    fp16=False,                        # fp16 미사용
    bf16=True,                         # bf16 사용(수치 안정성 우수, Ampere+ 지원)
    max_seq_length=2048,               # 입력 최대 토큰 길이(모델 로드와 동일)
    dataset_text_field="text",         # 학습에 사용할 컬럼명(format_conv의 반환 키)
    packing=False,                     # 여러 샘플을 한 시퀀스로 합치지 않음
    report_to="none",                  # W&B 등 외부 리포팅 비활성(에어갭 준수)
)

# ─── 학습 실행 ───
start = time.time()  # 소요 시간 측정 시작점
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, args=sft_config)
result = trainer.train()  # 실제 파인튜닝 수행(가장 오래 걸리는 구간)
logger.info("Training done: %s", result.metrics)

# 학습된 LoRA 어댑터 가중치와 토크나이저를 출력 디렉토리에 저장한다.
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ─── 메타데이터 기록 ───
# 이 체크포인트가 어떤 조건으로 학습됐는지 나중에 추적할 수 있게 남긴다.
# (모델/phase/에폭/샘플 수/소요 시간/지표/카테고리 분포/설계 근거)
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
# metadata.json으로 저장한다. ensure_ascii=False로 한글이 깨지지 않게 한다.
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# 최종 저장 위치와 총 소요 시간(분)을 로그로 남기고 종료한다.
logger.info("Saved to %s (%.1f min)", OUTPUT_DIR, (time.time() - start) / 60)
