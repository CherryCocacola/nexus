#!/usr/bin/env bash
# 서비스 크래시 자동복구 워치독 — 60초마다 start_all.sh(멱등) 1회 실행, 재기동 발생 시 로그.
while true; do
  sleep 60
  out=$(bash /NHNHOME/nexus/start_all.sh 2>&1)
  if echo "$out" | grep -q "기동"; then
    echo "[$(date "+%F %T")] 복구: $(echo "$out" | grep 기동 | tr "\n" " ")" >> /NHNHOME/nexus/watchdog.log
  fi
done
