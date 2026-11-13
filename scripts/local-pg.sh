#!/usr/bin/env bash
# Start a throwaway PostgreSQL 17 + pgvector 0.8.x without root or Docker.
# Downloads the PGDG Debian 12 (bookworm, amd64) packages, unpacks them, runs initdb.
#   scripts/local-pg.sh start   -> server on a unix socket in $PTR_PG_DIR, port 5544
#   scripts/local-pg.sh stop
# Anything else works too: point PTR_DSN at any Postgres with pgvector >= 0.8.
set -euo pipefail

DIR="${PTR_PG_DIR:-/tmp/ptr-pg}"
PORT="${PTR_PG_PORT:-5544}"
POOL=https://apt.postgresql.org/pub/repos/apt/pool/main/p
PG_DEB="postgresql-17/postgresql-17_17.12-1.pgdg12+1_amd64.deb"
VEC_DEB="pgvector/postgresql-17-pgvector_0.8.7-1.pgdg12+2_amd64.deb"
BIN="$DIR/root/usr/lib/postgresql/17/bin"

case "${1:-start}" in
  start)
    mkdir -p "$DIR"
    if [ ! -x "$BIN/postgres" ]; then
      for deb in "$PG_DEB" "$VEC_DEB"; do
        curl -sSfL -o "$DIR/$(basename "$deb")" "$POOL/$deb"
        dpkg -x "$DIR/$(basename "$deb")" "$DIR/root"
      done
    fi
    if [ ! -f "$DIR/data/PG_VERSION" ]; then
      "$BIN/initdb" -D "$DIR/data" -U postgres --auth=trust -E UTF8 --locale=C >/dev/null
      cat >> "$DIR/data/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '$DIR'
listen_addresses = ''
shared_buffers = 512MB
maintenance_work_mem = 512MB
work_mem = 32MB
max_parallel_maintenance_workers = 1
jit = off
CONF
    fi
    "$BIN/pg_ctl" -D "$DIR/data" -l "$DIR/server.log" -w start >/dev/null
    "$DIR/root/usr/lib/postgresql/17/bin/postgres" --version
    echo "PTR_DSN='host=$DIR port=$PORT user=postgres dbname=postgres'"
    ;;
  stop)
    "$BIN/pg_ctl" -D "$DIR/data" -w stop >/dev/null && echo stopped
    ;;
  *)
    echo "usage: $0 start|stop" >&2
    exit 2
    ;;
esac
