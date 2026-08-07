#!/usr/bin/env bash
export HF_HOME=/NHNHOME/nexus/hf_cache
export HF_HUB_DISABLE_XET=1
V=/NHNHOME/nexus/venv/bin
echo "[1/3] sentence-transformers 설치"; $V/pip install -q sentence-transformers
echo "[2/3] e5-large 다운로드"; $V/hf download intfloat/multilingual-e5-large
echo "[3/3] bge-reranker-v2-m3-ko 다운로드"; $V/hf download dragonkue/bge-reranker-v2-m3-ko
echo "EMB_DONE"
