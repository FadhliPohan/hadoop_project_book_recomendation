#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ML_DIR="$ROOT_DIR/machine_learning"
CLUSTER_CFG="$ROOT_DIR/config/cluster.yaml"
HDFS_SITE_XML="${HDFS_SITE_XML:-/usr/local/hadoop/etc/hadoop/hdfs-site.xml}"
CORE_SITE_XML="${CORE_SITE_XML:-/usr/local/hadoop/etc/hadoop/core-site.xml}"

INCLUDE_HADOOP_STATE=0

usage() {
  cat <<'EOF'
Usage: ./scripts/reset_training_state.sh [--include-hadoop-state]

Options:
  --include-hadoop-state
      Hapus state Hadoop/HDFS/YARN di master dan semua worker, lalu format
      ulang NameNode agar cluster kembali kosong.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --include-hadoop-state)
      INCLUDE_HADOOP_STATE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Argumen tidak dikenali: $arg"
      usage
      exit 1
      ;;
  esac
done

PYTHON_BIN="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

remove_dir_contents() {
  local dir="$1"
  local preserve_gitkeep="${2:-0}"

  if [[ ! -d "$dir" ]]; then
    echo "[INFO] Skip, directory tidak ada: $dir"
    return
  fi

  echo "[INFO] Membersihkan isi directory: $dir"
  if [[ "$preserve_gitkeep" == "1" ]]; then
    find "$dir" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +
  else
    find "$dir" -mindepth 1 -exec rm -rf {} +
  fi
}

reset_local_training_artifacts() {
  echo "[INFO] Reset artifact training lokal dimulai..."
  echo "[INFO] Dataset mentah lokal tidak akan dihapus."

  remove_dir_contents "$ML_DIR/data/processed" 1
  remove_dir_contents "$ML_DIR/models/sentiment"
  remove_dir_contents "$ML_DIR/models/recommender"
  remove_dir_contents "$ML_DIR/reports" 1
  remove_dir_contents "$ML_DIR/logs" 1
  remove_dir_contents "$ML_DIR/mlruns" 1

  if [[ -f "$ML_DIR/models/model_registry.json" ]]; then
    echo "[INFO] Menghapus file registry model"
    rm -f "$ML_DIR/models/model_registry.json"
  fi

  if [[ -f "$ROOT_DIR/output/metrics.json" ]]; then
    echo "[INFO] Menghapus metrics lama di output/"
    rm -f "$ROOT_DIR/output/metrics.json"
  fi

  echo "[INFO] Reset artifact training lokal selesai."
}

declare -a WORKERS=()
declare -a NN_DIRS=()
declare -a DN_DIRS=()
declare -a TMP_DIRS=()
declare -a LOG_DIRS=()
SSH_TIMEOUT=5

load_hadoop_reset_metadata() {
  local line kind value
  while IFS=$'\t' read -r kind value; do
    case "$kind" in
      WORKER)
        [[ -n "$value" ]] && WORKERS+=("$value")
        ;;
      SSH_TIMEOUT)
        [[ -n "$value" ]] && SSH_TIMEOUT="$value"
        ;;
      NN_DIR)
        [[ -n "$value" ]] && NN_DIRS+=("$value")
        ;;
      DN_DIR)
        [[ -n "$value" ]] && DN_DIRS+=("$value")
        ;;
      TMP_DIR)
        [[ -n "$value" ]] && TMP_DIRS+=("$value")
        ;;
      LOG_DIR)
        [[ -n "$value" ]] && LOG_DIRS+=("$value")
        ;;
    esac
  done < <(
    "$PYTHON_BIN" - "$ROOT_DIR" "$CLUSTER_CFG" "$HDFS_SITE_XML" "$CORE_SITE_XML" <<'PY'
import getpass
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

root_dir = Path(sys.argv[1])
cluster_cfg = Path(sys.argv[2])
hdfs_site_xml = Path(sys.argv[3])
core_site_xml = Path(sys.argv[4])

try:
    import yaml
except Exception:
    yaml = None


def load_yaml(path: Path) -> dict:
    if not yaml or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_xml_properties(path: Path) -> dict:
    if not path.exists():
        return {}

    tree = ET.parse(path)
    root = tree.getroot()
    props = {}
    for prop in root.findall("property"):
        name = (prop.findtext("name") or "").strip()
        value = (prop.findtext("value") or "").strip()
        if name:
            props[name] = value
    return props


def split_local_paths(raw_value: str) -> list[str]:
    if not raw_value:
        return []

    paths: list[str] = []
    for part in raw_value.split(","):
        candidate = part.strip()
        if not candidate:
            continue

        parsed = urlparse(candidate)
        if parsed.scheme == "file":
            path_value = parsed.path.strip()
        elif parsed.scheme:
            continue
        else:
            path_value = candidate

        if path_value and path_value not in paths:
            paths.append(path_value)
    return paths


cluster_data = load_yaml(cluster_cfg)
cluster_cfg_data = cluster_data.get("cluster", {}) if isinstance(cluster_data, dict) else {}

workers = cluster_cfg_data.get("workers") or ["fadhli@worker1", "fadhli@worker2"]
ssh_timeout = cluster_cfg_data.get("ssh_timeout", 5)

hdfs_props = load_xml_properties(hdfs_site_xml)
core_props = load_xml_properties(core_site_xml)

nn_dirs = split_local_paths(hdfs_props.get("dfs.namenode.name.dir", "file:///data/hadoop/hdfs/namenode"))
dn_dirs = split_local_paths(hdfs_props.get("dfs.datanode.data.dir", "file:///data/hadoop/hdfs/datanode"))
tmp_dirs = split_local_paths(core_props.get("hadoop.tmp.dir", "/data/hadoop/tmp"))

hadoop_home = os.environ.get("HADOOP_HOME") or "/usr/local/hadoop"
log_dir = os.environ.get("HADOOP_LOG_DIR") or str(Path(hadoop_home) / "logs")

for worker in workers:
    print(f"WORKER\t{worker}")

print(f"SSH_TIMEOUT\t{ssh_timeout}")

for path in nn_dirs:
    print(f"NN_DIR\t{path}")

for path in dn_dirs:
    print(f"DN_DIR\t{path}")

for path in tmp_dirs:
    print(f"TMP_DIR\t{path}")

print(f"LOG_DIR\t{log_dir}")
PY
  )

  if [[ "${#WORKERS[@]}" -eq 0 ]]; then
    WORKERS=("fadhli@worker1" "fadhli@worker2")
  fi
}

build_remote_cleanup_command() {
  local cmd
  cmd='set -euo pipefail; remove_dir_contents(){ local dir="$1"; if [[ ! -d "$dir" ]]; then echo "[INFO] Skip, directory tidak ada: $dir"; return; fi; echo "[INFO] Membersihkan isi directory: $dir"; find "$dir" -mindepth 1 -exec rm -rf {} +; };'

  local dir
  for dir in "${DN_DIRS[@]}"; do
    cmd+=" remove_dir_contents $(printf '%q' "$dir");"
  done
  for dir in "${TMP_DIRS[@]}"; do
    cmd+=" remove_dir_contents $(printf '%q' "$dir");"
  done
  for dir in "${LOG_DIRS[@]}"; do
    cmd+=" remove_dir_contents $(printf '%q' "$dir");"
  done

  printf '%s' "$cmd"
}

cleanup_workers_hadoop_state() {
  local failures=0
  local remote_cmd
  remote_cmd="$(build_remote_cleanup_command)"

  local worker
  for worker in "${WORKERS[@]}"; do
    echo "[INFO] Membersihkan state Hadoop di worker: $worker"
    if ! printf '%s\n' "$remote_cmd" | ssh -o BatchMode=yes -o ConnectTimeout="${SSH_TIMEOUT}" "$worker" "bash -s"; then
      echo "[ERROR] Gagal membersihkan state Hadoop di worker: $worker"
      failures=$((failures + 1))
    fi
  done

  if (( failures > 0 )); then
    echo "[ERROR] Reset Hadoop dibatalkan karena ada $failures worker yang gagal dibersihkan."
    return 1
  fi
}

cleanup_master_hadoop_state() {
  local dir

  echo "[INFO] Membersihkan state Hadoop di master..."
  for dir in "${NN_DIRS[@]}"; do
    remove_dir_contents "$dir"
  done
  for dir in "${DN_DIRS[@]}"; do
    remove_dir_contents "$dir"
  done
  for dir in "${TMP_DIRS[@]}"; do
    remove_dir_contents "$dir"
  done
  for dir in "${LOG_DIRS[@]}"; do
    remove_dir_contents "$dir"
  done
}

format_namenode() {
  if ! command -v hdfs >/dev/null 2>&1; then
    echo "[ERROR] Command hdfs tidak ditemukan. Tidak bisa format NameNode."
    return 1
  fi

  echo "[INFO] Format ulang NameNode agar HDFS kembali kosong..."
  hdfs namenode -format -force -nonInteractive
}

reset_hadoop_cluster_state() {
  load_hadoop_reset_metadata

  echo "[WARN] Mode reset Hadoop aktif."
  echo "[WARN] Ini akan menghapus state HDFS/YARN/log Hadoop di master dan semua worker."
  echo "[WARN] Cluster akan berhenti dan HDFS harus diisi ulang dari awal."

  echo "[INFO] Menghentikan cluster sebelum cleanup..."
  if ! bash "$ROOT_DIR/scripts/stop_cluster.sh"; then
    echo "[ERROR] Gagal menghentikan cluster. Cleanup Hadoop dibatalkan demi keamanan."
    return 1
  fi

  cleanup_workers_hadoop_state
  cleanup_master_hadoop_state
  format_namenode

  echo "[INFO] Reset Hadoop selesai."
  echo "[INFO] Cluster saat ini dalam kondisi STOPPED dan NameNode sudah diformat ulang."
  echo "[INFO] Langkah berikutnya:"
  echo "[INFO]   1. Start Cluster"
  echo "[INFO]   2. Upload Dataset ke HDFS"
  echo "[INFO]   3. Jalankan Spark Submit / pipeline lagi dari awal"
}

echo "[INFO] Reset training state dimulai..."
reset_local_training_artifacts

if (( INCLUDE_HADOOP_STATE )); then
  reset_hadoop_cluster_state
fi

echo "[INFO] Reset training state selesai."
