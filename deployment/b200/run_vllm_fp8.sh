#!/usr/bin/env bash
# A.X-4.0 FP8 vLLM 기동 (B200 GPU0, port 8001) — 소실분 재작성 2026-07-30
set -e
export HF_HOME=/NHNHOME/nexus/hf_cache
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export CUDA_VISIBLE_DEVICES=0
exec /NHNHOME/nexus/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model skt/A.X-4.0 --served-model-name ax-4.0 \
  --dtype bfloat16 --quantization fp8 \
  --max-model-len 65536 --gpu-memory-utilization 0.70 \
  --port 8001 --host 127.0.0.1 \
  --trust-remote-code --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes
