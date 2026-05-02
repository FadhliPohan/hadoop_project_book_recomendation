#!/usr/bin/env bash
# fix_hdfs_permissions.sh — Set permissive HDFS directories for cluster users
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_FILE="$ROOT_DIR/config/cluster.yaml"

TARGET_USER="${1:-${USER:-fadhli}}"
TARGET_GROUP="${TARGET_GROUP:-supergroup}"
HDFS_SUPERUSER="${HDFS_SUPERUSER:-$TARGET_USER}"
HDFS_BIN="${HDFS_BIN:-/usr/local/hadoop/bin/hdfs}"

shift || true

if [[ $# -gt 0 ]]; then
  WORKERS=("$@")
else
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 tidak ditemukan, tidak bisa membaca config/cluster.yaml"
    exit 1
  fi

  mapfile -t WORKERS < <(
    python3 - "$CFG_FILE" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    print("")
    sys.exit(0)

cfg_file = Path(sys.argv[1])
if not cfg_file.exists():
    sys.exit(0)

cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
workers = cfg.get("cluster", {}).get("workers", []) or []
for worker in workers:
    if worker:
        print(str(worker))
PY
  )
fi

if [[ ${#WORKERS[@]} -eq 0 ]]; then
  WORKERS=("fadhli@worker1" "fadhli@worker2")
fi

echo "[INFO] === Fix HDFS Permissions ==="
echo "[INFO] Target user      : $TARGET_USER"
echo "[INFO] Target group     : $TARGET_GROUP"
echo "[INFO] HDFS superuser   : $HDFS_SUPERUSER"
echo "[INFO] Candidate workers: ${WORKERS[*]}"

for worker in "${WORKERS[@]}"; do
  echo "[INFO] --- Worker: $worker ---"
  ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no "$worker" 'bash -s' -- \
    "$TARGET_USER" "$TARGET_GROUP" "$HDFS_SUPERUSER" "$HDFS_BIN" <<'REMOTE'
set -euo pipefail

TARGET_USER="$1"
TARGET_GROUP="$2"
HDFS_SUPERUSER="$3"
HDFS_BIN="$4"

if [[ ! -x "$HDFS_BIN" ]]; then
  if command -v hdfs >/dev/null 2>&1; then
    HDFS_BIN="$(command -v hdfs)"
  elif [[ -x "/usr/local/hadoop/bin/hdfs" ]]; then
    HDFS_BIN="/usr/local/hadoop/bin/hdfs"
  else
    echo "[ERROR] Binary hdfs tidak ditemukan di node ini."
    exit 1
  fi
fi

run_hdfs() {
  if [[ -n "${HDFS_SUPERUSER:-}" ]]; then
    HADOOP_USER_NAME="$HDFS_SUPERUSER" "$HDFS_BIN" dfs "$@"
  else
    "$HDFS_BIN" dfs "$@"
  fi
}

run_hdfs -mkdir -p /tmp /user
run_hdfs -mkdir -p "/user/$TARGET_USER"
run_hdfs -mkdir -p "/user/$TARGET_USER/amazon_books"
run_hdfs -mkdir -p "/user/$TARGET_USER/output/amazon_books_ml"

run_hdfs -chmod 1777 /tmp
run_hdfs -chmod 777 /user
run_hdfs -chown -R "$TARGET_USER:$TARGET_GROUP" "/user/$TARGET_USER"
run_hdfs -chmod -R 777 "/user/$TARGET_USER"

echo "[INFO] HDFS permission summary:"
run_hdfs -ls -d / /tmp /user "/user/$TARGET_USER" "/user/$TARGET_USER/amazon_books" "/user/$TARGET_USER/output/amazon_books_ml"
REMOTE
done

echo "[INFO] Fix permission selesai."
