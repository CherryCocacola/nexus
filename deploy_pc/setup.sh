#!/usr/bin/env bash
# =============================================================================
# Nexus 서비스 PC — 최초 1회 셋업 (Linux/macOS)
# 새 서버로 옮겼을 때 이 스크립트만 실행하면 오케스트레이터 실행 환경 준비.
# (GPU 불필요 — 추론/임베딩은 B200 백엔드 담당)
#   실행:  bash deploy_pc/setup.sh
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "[setup] 프로젝트 루트: $ROOT"
[ -d .venv_pc ] || python3 -m venv .venv_pc
.venv_pc/bin/python -m pip install --upgrade pip
.venv_pc/bin/python -m pip install -r deploy_pc/requirements-pc.txt
echo ""
echo "[setup] 완료. 확인/수정: .env / config/nexus_config.yaml / deploy_pc/tunnel.sh"
echo "실행 순서: (1) bash deploy_pc/tunnel.sh  (2) bash deploy_pc/start_web.sh (또는 start_cli.sh)"
