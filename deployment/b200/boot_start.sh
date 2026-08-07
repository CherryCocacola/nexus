#!/usr/bin/env bash
# 재부팅 후 GPU 백엔드를 자동 기동하는 cron @reboot 진입점. 작성일 2026-08-07
#
# [왜 systemd 가 아닌가]
#   이 서버(B200)는 sudo 에 비밀번호가 필요해 시스템 유닛을 만들 수 없고,
#   사용자 systemd 버스도 없다("Failed to connect to bus: No medium found").
#   비root 로 재부팅 자동기동을 거는 유일한 수단이 사용자 crontab 의 @reboot 다.
#
# [왜 start_all.sh 를 곧바로 부르지 않는가]
#   @reboot 는 부팅 아주 이른 시점에 실행된다. 그때는 /NHNHOME 마운트나 NVIDIA
#   드라이버가 아직 준비되지 않았을 수 있다. 거기서 한 번 부르고 끝나면 아무 것도
#   뜨지 않는다 — watchdog 마저 start_all.sh 가 띄우는 구조라 자가복구조차
#   시작되지 않는다. 그래서 준비를 기다린 뒤 부르고, 그 뒤로도 몇 번 더 시도한다.
#   한 번만 성공하면 watchdog 이 60초 주기로 이어받는다.
#
# [로그]
#   /tmp/nexus_boot.log — /NHNHOME 이 아직 없을 수 있어 항상 쓸 수 있는 곳에 남긴다.
#   부팅이 잘못됐을 때 "왜 안 떴는지"를 여기서 본다.
set -u
LOG=/tmp/nexus_boot.log
N=/NHNHOME/nexus
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "===== 부팅 자동기동 시작 (uptime: $(cut -d' ' -f1 /proc/uptime)s) ====="

# 1) /NHNHOME 마운트 + 스크립트가 보일 때까지 최대 10분 대기.
waited=0
for _ in $(seq 1 60); do
  [ -x "$N/start_all.sh" ] && break
  sleep 10
  waited=$((waited + 10))
done
if [ ! -x "$N/start_all.sh" ]; then
  say "중단: $N/start_all.sh 를 찾지 못함(마운트 실패 의심) — 수동 확인 필요"
  exit 1
fi
say "start_all.sh 확인 (대기 ${waited}초)"

# 2) GPU 준비 대기 최대 5분. vLLM 은 드라이버 없이는 뜨지 못한다.
gpu_wait=0
for _ in $(seq 1 30); do
  nvidia-smi -L >/dev/null 2>&1 && break
  sleep 10
  gpu_wait=$((gpu_wait + 10))
done
if nvidia-smi -L >/dev/null 2>&1; then
  say "GPU 준비됨 (대기 ${gpu_wait}초)"
else
  say "경고: GPU 미확인 — 그래도 진행한다(watchdog 이 계속 재시도)"
fi

# 3) 기동 3회 시도. 이후는 watchdog 이 60초마다 이어받는다.
for k in 1 2 3; do
  out=$(bash "$N/start_all.sh" 2>&1)
  say "시도 $k: ${out//$'\n'/ | }"
  sleep 30
done

say "포트: $(ss -ltn 2>/dev/null | grep -oE ':800[0-9]' | sort -u | tr '\n' ' ')"
say "===== 부팅 자동기동 종료 ====="
