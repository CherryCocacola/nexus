#!/usr/bin/env bash
# =============================================================================
# Nexus 분리구조 — GPU 백엔드(B200 "nova") SSH 터널 (Linux/macOS)
# =============================================================================
# B200의 모델 서버들을 이 서버의 localhost 로 안전하게 당겨온다.
# 이 창을 열어둔 채로 start_web.sh / start_cli.sh 를 다른 창에서 실행.
# 다른 백엔드로 이전 시 아래 3개 변수만 수정.
#
# ── 2026-08-18 수정 ────────────────────────────────────────────────────────
#   ① 로컬 바인드 포트를 8001/8002/5440/6340 → 18001~18005 로 교정.
#      config/nexus_config.pc.yaml 은 18xxx 를 보는데 이 스크립트만 옛 포트를
#      열고 있어 서로 맞지 않았다(포트 이동 이전 버전이 남아 있었음).
#   ② 이미지 18003 · 비전 18004 · 코딩 18005 추가 — 통째로 빠져 있었다.
#   ③ 기본 키를 nova_key 로 교체(07-30 재구축으로 nexus_key 는 폐기됨),
#      리포 안 user_mig/nova_key 를 먼저 찾는다. passphrase 없음.
# =============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

KEY="${NEXUS_SSH_KEY:-$ROOT/user_mig/nova_key}"   # SSH 개인키(passphrase 없음)
BASTION="${NEXUS_BASTION:-idino_user@59.150.33.1}"
PORT="${NEXUS_BASTION_PORT:-45702}"

if [ ! -f "$KEY" ]; then
  echo "[터널] SSH 키를 찾을 수 없습니다: $KEY" >&2
  echo "       이 키는 git 에 올라가지 않습니다(.gitignore). 담당자에게 받아 두십시오." >&2
  exit 1
fi
chmod 600 "$KEY" 2>/dev/null || true

echo "[터널] B200(nova) 백엔드로 연결. 이 창을 닫지 마세요."
echo "[터널] 추론 18001 · 임베딩 18002 · 이미지 18003 · 비전 18004 · 코딩 18005"

# 15440·16340 은 pc.yaml 이 DB 를 LAN 으로 직결하므로 실사용되지 않지만,
# 기존 점검 스크립트와의 호환을 위해 남긴다.
exec ssh -N \
  -L 18001:127.0.0.1:8001 \
  -L 18002:127.0.0.1:8002 \
  -L 18003:127.0.0.1:8003 \
  -L 18004:127.0.0.1:8004 \
  -L 18005:127.0.0.1:8005 \
  -L 15440:127.0.0.1:5440 \
  -L 16340:127.0.0.1:6340 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i "$KEY" -p "$PORT" "$BASTION"
