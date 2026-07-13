#!/usr/bin/env bash
# 112 호스트에서 실행 — B200 vLLM 상주 SSH 터널 설치·기동·검증 스크립트
# ---------------------------------------------------------------------------
# 선행: 이 스크립트와 같은 폴더에 nexus-b200-tunnel.service, 그리고 원본 SSH 키
#       nexus_key(passphrase = NHN 로그인 비밀번호)를 미리 옮겨둔다.
# 하는 일:
#   1) 키를 /home/idino/nexus-tunnel/ 로 옮기고 권한 600 설정
#   2) passphrase 없는 사본(nexus_key_nopass) 생성 (systemd 무인 기동용)
#   3) systemd 유닛 설치·enable·start
#   4) 18001 리슨 확인 + B200 /v1/models 로 served-model-name 실측(질문2 확인)
set -euo pipefail

TDIR=/home/idino/nexus-tunnel
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "== [1/3] 키 배치 (이 키는 passphrase 없음) =="
mkdir -p "$TDIR"
if [[ ! -f "$TDIR/nexus_key" ]]; then
  cp "$SRC_DIR/nexus_key" "$TDIR/nexus_key"
fi
chmod 600 "$TDIR/nexus_key"

echo "== [2/3] 기존 수동 터널 정리 (있으면) =="
pkill -f '18001:127.0.0.1:8001' 2>/dev/null || true
sleep 1

echo "== [3/3] systemd 유닛 설치·기동 =="
sudo cp "$SRC_DIR/nexus-b200-tunnel.service" /etc/systemd/system/nexus-b200-tunnel.service
sudo systemctl daemon-reload
sudo systemctl enable --now nexus-b200-tunnel.service
sleep 3
sudo systemctl --no-pager --full status nexus-b200-tunnel.service | head -n 12 || true

echo "== [4/4] 검증: 18001 리슨 + B200 served-model-name (질문2) =="
ss -tlnp 2>/dev/null | grep ':18001' || echo "  [경고] 18001 미개통 — journalctl -u nexus-b200-tunnel 확인"
echo "  --- B200 /v1/models ---"
curl -s --max-time 10 http://127.0.0.1:18001/v1/models || echo "  [경고] vLLM 응답 없음 — 터널/백엔드 상태 확인"
echo
echo "== 완료. 위 /v1/models 의 id 가 'ax-4.0' 인지 확인하세요."
echo "   ax-4.0 이 아니면 config/nexus_config.112.yaml 의 primary_model + 3개 모드 model 을 실제 값으로 재수정 후 nexus-web 재기동."
