#!/usr/bin/env bash
set -euo pipefail

HOSTS_FILE="/etc/hosts"
MASTER_HOSTNAME="${MASTER_HOSTNAME:-fadhli}"
MASTER_IP="${MASTER_IP:-192.168.0.102}"
WORKER1_HOSTNAME="${WORKER1_HOSTNAME:-worker1}"
WORKER1_IP="${WORKER1_IP:-192.168.0.103}"
WORKER2_HOSTNAME="${WORKER2_HOSTNAME:-worker2}"
WORKER2_IP="${WORKER2_IP:-192.168.0.105}"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "[ERROR] Script ini harus dijalankan sebagai root."
    echo "[ERROR] Contoh: sudo bash scripts/fix_yarn_worker_network.sh"
    exit 1
  fi
}

backup_file() {
  local path="$1"
  local backup_path="${path}.codex.bak.$(date +%Y%m%d_%H%M%S)"
  cp "$path" "$backup_path"
  echo "[INFO] Backup dibuat: $backup_path"
}

fix_hosts_file() {
  python3 - "$HOSTS_FILE" "$MASTER_HOSTNAME" "$MASTER_IP" "$WORKER1_HOSTNAME" "$WORKER1_IP" "$WORKER2_HOSTNAME" "$WORKER2_IP" <<'PY'
from pathlib import Path
import sys

hosts_path = Path(sys.argv[1])
master_host = sys.argv[2]
master_ip = sys.argv[3]
worker1_host = sys.argv[4]
worker1_ip = sys.argv[5]
worker2_host = sys.argv[6]
worker2_ip = sys.argv[7]

managed_hosts = {master_host, worker1_host, worker2_host}

lines = hosts_path.read_text(encoding="utf-8").splitlines()
new_lines = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        new_lines.append(line)
        continue

    parts = stripped.split()
    if any(host in managed_hosts for host in parts[1:]):
        continue

    new_lines.append(line)

block = [
    f"{master_ip} {master_host}",
    f"{worker1_ip} {worker1_host}",
    f"{worker2_ip} {worker2_host}",
]

insert_at = 0
for idx, line in enumerate(new_lines):
    stripped = line.strip()
    if stripped and not stripped.startswith("#"):
        insert_at = idx + 1

new_lines[insert_at:insert_at] = block
hosts_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
PY
  echo "[INFO] /etc/hosts diperbarui."
}

restart_nodemanager() {
  local stop_cmd="/usr/local/hadoop/sbin/yarn-daemon.sh"
  if [[ ! -x "$stop_cmd" ]]; then
    echo "[ERROR] yarn-daemon.sh tidak ditemukan di /usr/local/hadoop/sbin/"
    exit 1
  fi

  echo "[INFO] Restart NodeManager..."
  "$stop_cmd" stop nodemanager || true
  sleep 2
  "$stop_cmd" start nodemanager
}

verify_worker() {
  echo "[INFO] Verifikasi hostname:"
  getent hosts "$MASTER_HOSTNAME" || true
  getent hosts "$WORKER1_HOSTNAME" || true
  getent hosts "$WORKER2_HOSTNAME" || true
}

main() {
  require_root

  if [[ ! -f "$HOSTS_FILE" ]]; then
    echo "[ERROR] File hosts tidak ditemukan: $HOSTS_FILE"
    exit 1
  fi

  backup_file "$HOSTS_FILE"
  fix_hosts_file
  restart_nodemanager
  verify_worker
}

main "$@"
