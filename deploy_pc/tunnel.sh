#!/usr/bin/env bash
# =============================================================================
# Nexus 분리구조 — GPU 백엔드(B200) SSH 터널 (Linux/macOS)
# B200의 vLLM/임베딩/PG/Redis 를 이 서버의 localhost 로 안전하게 당겨온다.
# 이 창을 열어둔 채로 start_web.sh 를 다른 창에서 실행.
# 다른 백엔드로 이전 시 아래 3개 변수만 수정.
# =============================================================================
KEY="${NEXUS_SSH_KEY:-$HOME/.ssh/nexus_key}"   # SSH 개인키 경로
BASTION="${NEXUS_BASTION:-idino_user@59.150.33.1}"
PORT="${NEXUS_BASTION_PORT:-45702}"

echo "[터널] B200 백엔드로 연결 (passphrase 입력). 이 창을 닫지 마세요."
exec ssh -N \
  -L 8001:127.0.0.1:8001 \
  -L 8002:127.0.0.1:8002 \
  -L 5440:127.0.0.1:5440 \
  -L 6340:127.0.0.1:6340 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -i "$KEY" -p "$PORT" "$BASTION"
