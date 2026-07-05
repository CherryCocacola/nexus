#!/usr/bin/env bash
# 서비스 PC에서 웹(오케스트레이터) 기동 (Linux/macOS)
# 전제: tunnel.sh 가 다른 창에서 실행 중. 접속: http://localhost:8443
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a; [ -f .env ] && . ./.env; set +a
export PYTHONIOENCODING=utf-8
# 포트 8600 사용(8443 은 Docker 등 기존 서비스와 충돌 회피). 접속: http://127.0.0.1:8600
echo "[웹] 오케스트레이터 기동: http://127.0.0.1:8600  (백엔드=B200 터널)"
exec .venv_pc/bin/python -m uvicorn web.app:app --host 127.0.0.1 --port 8600 --log-level info
