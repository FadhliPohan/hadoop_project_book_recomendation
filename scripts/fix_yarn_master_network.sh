#!/usr/bin/env bash
set -euo pipefail

HADOOP_CONF_DIR="${HADOOP_CONF_DIR:-/usr/local/hadoop/etc/hadoop}"
HOSTS_FILE="/etc/hosts"
YARN_SITE_FILE="$HADOOP_CONF_DIR/yarn-site.xml"
MASTER_HOSTNAME="${MASTER_HOSTNAME:-fadhli}"
MASTER_IP="${MASTER_IP:-192.168.0.102}"
WORKER1_HOSTNAME="${WORKER1_HOSTNAME:-worker1}"
WORKER1_IP="${WORKER1_IP:-192.168.0.103}"
WORKER2_HOSTNAME="${WORKER2_HOSTNAME:-worker2}"
WORKER2_IP="${WORKER2_IP:-192.168.0.105}"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "[ERROR] Script ini harus dijalankan sebagai root."
    echo "[ERROR] Contoh: sudo bash scripts/fix_yarn_master_network.sh"
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
    hosts = parts[1:]
    if any(host in managed_hosts for host in hosts):
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

fix_yarn_site() {
  python3 - "$YARN_SITE_FILE" "$MASTER_HOSTNAME" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

path = Path(sys.argv[1])
master_hostname = sys.argv[2]
tree = ET.parse(path)
root = tree.getroot()

properties = {}
for prop in root.findall("property"):
    name_el = prop.find("name")
    value_el = prop.find("value")
    if name_el is not None and value_el is not None and name_el.text:
        properties[name_el.text] = value_el

def set_property(name: str, value: str) -> None:
    if name in properties:
        properties[name].text = value
        return
    prop = ET.SubElement(root, "property")
    name_el = ET.SubElement(prop, "name")
    name_el.text = name
    value_el = ET.SubElement(prop, "value")
    value_el.text = value
    properties[name] = value_el

set_property("yarn.resourcemanager.hostname", master_hostname)
set_property("yarn.resourcemanager.bind-host", "0.0.0.0")

tree.write(path, encoding="utf-8", xml_declaration=True)
PY
  echo "[INFO] yarn-site.xml diperbarui."
}

restart_yarn() {
  echo "[INFO] Restart YARN..."
  local stop_cmd="/usr/local/hadoop/sbin/stop-yarn.sh"
  local start_cmd="/usr/local/hadoop/sbin/start-yarn.sh"
  if [[ -x "$stop_cmd" ]]; then
    "$stop_cmd" || true
  fi
  if [[ -x "$start_cmd" ]]; then
    "$start_cmd"
  else
    echo "[ERROR] start-yarn.sh tidak ditemukan."
    exit 1
  fi
}

verify_cluster() {
  echo "[INFO] Verifikasi hostname:"
  getent hosts "$MASTER_HOSTNAME" || true
  getent hosts "$WORKER1_HOSTNAME" || true
  getent hosts "$WORKER2_HOSTNAME" || true

  echo "[INFO] Verifikasi YARN:"
  if command -v yarn >/dev/null 2>&1; then
    yarn node -list || true
  else
    echo "[WARN] Command yarn tidak ditemukan."
  fi
}

main() {
  require_root

  if [[ ! -f "$HOSTS_FILE" ]]; then
    echo "[ERROR] File hosts tidak ditemukan: $HOSTS_FILE"
    exit 1
  fi
  if [[ ! -f "$YARN_SITE_FILE" ]]; then
    echo "[ERROR] File yarn-site.xml tidak ditemukan: $YARN_SITE_FILE"
    exit 1
  fi

  backup_file "$HOSTS_FILE"
  backup_file "$YARN_SITE_FILE"
  fix_hosts_file
  fix_yarn_site
  restart_yarn
  verify_cluster
}

main "$@"
