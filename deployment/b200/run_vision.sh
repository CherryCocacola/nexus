#!/usr/bin/env bash
# Gemma 3 27B 비전(VLM) vLLM 기동 — GPU1:8004 (임베딩과 공유). 재작성 2026-07-30
export HF_HOME=/NHNHOME/nexus/hf_cache
export CUDA_VISIBLE_DEVICES=1
exec /NHNHOME/nexus/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-27b-it --served-model-name gemma-3-27b \
  --max-model-len 8192 --gpu-memory-utilization 0.42 \
  --port 8004 --host 0.0.0.0 --trust-remote-code
