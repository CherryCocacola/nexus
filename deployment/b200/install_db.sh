#!/usr/bin/env bash
gcs(){ bash -lic "gcsudo $*"; }
echo "[1] 기본 패키지(redis 포함)"
gcs "apt-get install -y curl ca-certificates gnupg lsb-release redis-server postgresql-common"
echo "[2] PGDG 저장소 추가"
gcs "/usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y"
gcs "apt-get update -y"
echo "[3] PG17 + pgvector"
gcs "apt-get install -y postgresql-17 postgresql-17-pgvector"
echo "[4] 결과 확인"
ls -1 /usr/lib/postgresql/17/bin/postgres 2>/dev/null && echo PG17_OK
which redis-server >/dev/null 2>&1 && echo REDIS_OK
ls -1 /usr/lib/postgresql/17/lib/vector.so 2>/dev/null && echo PGVECTOR_OK
echo DB_INSTALL_DONE
