#!/usr/bin/env bash
# Devstral Small 2507 코딩 서브모델 vLLM (B200 GPU0, port 8005) — 2026-08-05
# 도구 호출: mistral 파서만 켠다. --tokenizer-mode mistral 은 쓰지 않는다 —
#   그 모드에서는 vLLM이 chat_template 파라미터를 거부해(400 "chat_template is not
#   supported for Mistral tokenizers") NOVA의 프롬프트 조립 경로가 통째로 실패한다.
#   HF 토크나이저 + mistral 파서 조합이면 [TOOL_CALLS] 출력도 정상 파싱된다.
# 원복: run_coder.sh.bak-noparser 복원 또는 이 파일 삭제 + start_all.sh coder 줄 제거.
set -e
export HF_HOME=/NHNHOME/nexus/hf_cache
export CUDA_VISIBLE_DEVICES=0
exec /NHNHOME/nexus/venv/bin/python -m vllm.entrypoints.openai.api_server   --model mistralai/Devstral-Small-2507 --served-model-name devstral-small   --dtype bfloat16 --quantization fp8   --max-model-len 32768 --gpu-memory-utilization 0.20   --port 8005 --host 127.0.0.1   --enable-auto-tool-choice --tool-call-parser mistral
