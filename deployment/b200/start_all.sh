#!/usr/bin/env bash
# Nexus B200 GPU백엔드 기동 + watchdog — 재작성 2026-07-31(정리·안정화).
# 멱등: 이미 떠있으면 건너뜀. web(8443)은 제거(112가 웹 정본). watchdog이 크래시 자동복구.
#
# 2026-08-07: 자격증명을 본문에서 빼고 .env(600)에서 읽도록 바꿨다.
#   이유는 보안만이 아니다 — 비밀번호가 본문에 있으면 이 스크립트 자체를 버전관리에
#   올릴 수 없고, 그러면 서버가 재구축될 때 GPU 백엔드를 살리는 수단이 사라진다.
set -u
N=/NHNHOME/nexus; V=$N/venv/bin; PGBIN=/usr/lib/postgresql/17/bin
export HF_HOME=$N/hf_cache

# 자격증명 로드 — set -a 로 읽는 동안만 자동 export 한다(자식 tmux 세션이 물려받아야 한다).
#   .env 가 없어도 **중단하지 않는다**: 이 스크립트가 곧 크래시 복구 수단이라,
#   Redis 인증 하나 때문에 vLLM·비전·이미지 복구까지 멈추면 손해가 더 크다.
#   대신 경고를 남겨 원인이 드러나게 한다.
if [ -r "$N/.env" ]; then
  set -a; . "$N/.env"; set +a
else
  echo "경고: $N/.env 없음 — Redis 인증이 실패할 수 있다(나머지는 계속 진행)" >&2
fi
export NEXUS_PG_PASSWORD="${NEXUS_PG_PASSWORD:-}"
export NEXUS_REDIS_PASSWORD="${NEXUS_REDIS_PASSWORD:-}"

pup(){ (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && { exec 3>&-; return 0; }; return 1; }
$PGBIN/pg_ctl -D $N/pgdata status >/dev/null 2>&1 || { echo "[PG] 기동"; $PGBIN/pg_ctl -D $N/pgdata -l $N/pg.log -w start; }
redis-cli -p 6340 -a "$NEXUS_REDIS_PASSWORD" ping >/dev/null 2>&1 || { echo "[Redis] 기동"; redis-server $N/redis.conf --daemonize yes; }
pup 8002 || { echo "[embed] 기동"; tmux new-session -d -s embed "cd $N && CUDA_VISIBLE_DEVICES=1 HF_HOME=$HF_HOME $V/python scripts/embed_server.py > $N/embed.log 2>&1"; }
pup 8001 || { echo "[vLLM] 기동"; tmux new-session -d -s vllm "bash $N/run_vllm_fp8.sh > $N/vllm.log 2>&1"; }
pup 8004 || { echo "[vision] 기동"; tmux new-session -d -s vision "bash $N/run_vision.sh > $N/vision.log 2>&1"; }
pup 8003 || { echo "[image] 기동"; tmux new-session -d -s image "cd $N && CUDA_VISIBLE_DEVICES=1 HF_HOME=$HF_HOME $V/python scripts/image_server.py > $N/image.log 2>&1"; }
pup 8005 || { echo "[coder] 기동"; tmux new-session -d -s coder "bash $N/run_coder.sh > $N/coder.log 2>&1"; }
# watchdog 자동기동(세션 있으면 skip → 재귀 방지)
tmux has-session -t watchdog 2>/dev/null || tmux new-session -d -s watchdog "bash $N/watchdog.sh"
echo "완료 — GPU백엔드: 8001 A.X-4.0 / 8002 임베딩 / 8003 이미지 / 8004 비전 (+watchdog)"
