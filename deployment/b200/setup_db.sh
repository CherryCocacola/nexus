#!/usr/bin/env bash
# B200 로컬 PostgreSQL 17 + Redis 최초 구성(1회성). 재구축 시 install_db.sh 다음에 실행.
#
# 2026-08-07: 비밀번호를 본문에서 빼고 .env 에서 읽도록 바꿨다.
#   여기서는 start_all.sh 와 달리 **fail-closed** 다 — .env 가 없으면 즉시 중단한다.
#   빈 비밀번호로 role 을 만들면 그 순간 DB가 무인증으로 열리고, 나중에 원인을
#   찾기도 어렵다. 기동 스크립트(복구 수단)와 프로비저닝 스크립트는 실패 방식이
#   달라야 한다.
set -e
N=/NHNHOME/nexus
PGBIN=/usr/lib/postgresql/17/bin
PGDATA=$N/pgdata

if [ ! -r "$N/.env" ]; then
  echo "중단: $N/.env 가 없다. .env.example 을 복사해 값을 채운 뒤 chmod 600 하고 다시 실행하라." >&2
  exit 1
fi
set -a; . "$N/.env"; set +a
: "${NEXUS_PG_PASSWORD:?NEXUS_PG_PASSWORD 가 .env 에 없다}"
: "${NEXUS_REDIS_PASSWORD:?NEXUS_REDIS_PASSWORD 가 .env 에 없다}"

if [ ! -f "$PGDATA/PG_VERSION" ]; then
  echo "[PG] initdb"
  $PGBIN/initdb -D "$PGDATA" -U idino_user --auth-local=trust --auth-host=scram-sha-256 -E UTF8 >/dev/null
  {
    echo "port = 5440"
    echo "listen_addresses = '127.0.0.1'"
    echo "unix_socket_directories = '/tmp'"
    echo "shared_buffers = 8GB"
  } >> "$PGDATA/postgresql.conf"
fi
echo "[PG] start"
$PGBIN/pg_ctl -D "$PGDATA" -l $N/pg.log -w start
sleep 2
echo "[PG] role/db"
# 비밀번호는 psql 변수로 넘긴다 — SQL 본문에 문자열을 끼워 넣으면
# 특수문자(!@#$)에서 따옴표가 깨지고, ps 출력에도 노출된다.
$PGBIN/psql -h /tmp -p 5440 -U idino_user -d postgres -v ON_ERROR_STOP=0 \
  -v pw="$NEXUS_PG_PASSWORD" <<'SQL'
CREATE ROLE nexus LOGIN PASSWORD :'pw';
CREATE DATABASE nexus OWNER nexus;
SQL
$PGBIN/psql -h /tmp -p 5440 -U idino_user -d nexus -c "CREATE EXTENSION IF NOT EXISTS vector;"
echo -n "[PG] pgvector: "; $PGBIN/psql -h /tmp -p 5440 -U idino_user -d nexus -tAc "SELECT extversion FROM pg_extension WHERE extname='vector';"

echo "[Redis] 설정+기동"
mkdir -p $N/redis_data
# redis.conf 는 비밀번호를 담으므로 버전관리에 올리지 않는다(redis.conf.example 참조).
# 생성 직후 600 으로 조인다 — 기본 umask 로 두면 644 가 된다.
umask 077
printf 'port 6340\nbind 127.0.0.1\nrequirepass %s\ndir %s/redis_data\nappendonly no\n' \
  "$NEXUS_REDIS_PASSWORD" "$N" > "$N/redis.conf"
umask 022
redis-server "$N/redis.conf" --daemonize yes
sleep 1
echo -n "[Redis] ping: "; redis-cli -p 6340 -a "$NEXUS_REDIS_PASSWORD" ping 2>/dev/null
echo "SETUP_DB_DONE"
