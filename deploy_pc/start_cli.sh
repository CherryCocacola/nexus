#!/usr/bin/env bash
# 서비스 PC에서 대화형 CLI(REPL) 기동 (Linux/macOS)
# 전제: tunnel.sh 가 다른 창에서 실행 중.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a; [ -f .env ] && . ./.env; set +a
export PYTHONIOENCODING=utf-8
echo "[CLI] 대화형 REPL 시작 (백엔드=B200 터널). 종료: /exit"
exec .venv_pc/bin/python -m cli.repl
