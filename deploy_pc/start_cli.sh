#!/usr/bin/env bash
# 서비스 PC에서 대화형 CLI(REPL) 기동 (Linux/macOS)
# 전제: tunnel.sh 가 다른 창에서 실행 중.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a; [ -f .env ] && . ./.env; set +a
export PYTHONIOENCODING=utf-8
# PC 전용 설정 사용 — 터널 로컬 포트(18001~18005)를 가리키는 사본.
# ★없으면 기본 nexus_config.yaml(8001~8005 = B200 컨테이너 내부용)로 떨어져
#   전 백엔드가 ConnectError 로 실패한다(2026-08-18 수정).
export NEXUS_CONFIG="${NEXUS_CONFIG:-config/nexus_config.pc.yaml}"
echo "[CLI] 대화형 REPL 시작 (백엔드=B200 터널). 종료: /exit"
exec .venv_pc/bin/python -m cli.repl
