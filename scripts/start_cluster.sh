#!/usr/bin/env bash
set -euo pipefail

HADOOP_WORKERS_FILE="${HADOOP_WORKERS_FILE:-/usr/local/hadoop/etc/hadoop/workers}"

run_if_exists() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "[INFO] Menjalankan: $cmd"
    "$cmd"
  else
    echo "[WARN] Command tidak ditemukan: $cmd"
  fi
}

count_expected_workers() {
  if [[ -f "$HADOOP_WORKERS_FILE" ]]; then
    awk 'NF && $1 !~ /^#/{count++} END{print count+0}' "$HADOOP_WORKERS_FILE"
  else
    echo "0"
  fi
}

show_hdfs_health() {
  if ! command -v hdfs >/dev/null 2>&1; then
    echo "[WARN] Command hdfs tidak ditemukan, skip validasi HDFS."
    return
  fi

  local report
  if command -v timeout >/dev/null 2>&1; then
    report="$(timeout 20s hdfs dfsadmin -report 2>&1)" || true
  else
    report="$(hdfs dfsadmin -report 2>&1)" || true
  fi
  if [[ -z "$report" ]]; then
    echo "[WARN] Gagal membaca hdfs dfsadmin -report"
    return
  fi

  local live_datanodes
  live_datanodes="$(
    sed -n 's/^Live datanodes (\([0-9]\+\)).*/\1/p' <<<"$report" | head -n1
  )"
  if [[ -z "$live_datanodes" ]]; then
    live_datanodes="$(awk -F': *' '/^Live datanodes/{print $2; exit}' <<<"$report" | tr -d '[:space:]')"
  fi
  if [[ -z "$live_datanodes" ]]; then
    echo "[WARN] Tidak bisa membaca jumlah Live datanodes."
    return
  fi

  echo "[INFO] Live DataNode: $live_datanodes (expected: $EXPECTED_WORKERS)"
  if [[ "$live_datanodes" =~ ^[0-9]+$ ]] && [[ "$EXPECTED_WORKERS" =~ ^[0-9]+$ ]] && (( live_datanodes < EXPECTED_WORKERS )); then
    echo "[WARN] DataNode aktif lebih sedikit dari worker yang dikonfigurasi."
    echo "[WARN] Cek UI NameNode dan log DataNode pada worker yang tidak muncul."
  fi
}

show_yarn_health() {
  if ! command -v yarn >/dev/null 2>&1; then
    echo "[WARN] Command yarn tidak ditemukan, skip validasi YARN."
    return
  fi

  local report
  if command -v timeout >/dev/null 2>&1; then
    report="$(timeout 20s yarn node -list 2>&1)" || true
  else
    report="$(yarn node -list 2>&1)" || true
  fi
  if [[ -z "$report" ]]; then
    echo "[WARN] Gagal membaca yarn node -list"
    return
  fi

  local running_nodes
  running_nodes="$(
    sed -n 's/^Total Nodes:[[:space:]]*\([0-9]\+\).*/\1/p' <<<"$report" | head -n1
  )"
  if [[ -n "$running_nodes" ]]; then
    echo "[INFO] YARN NodeManager aktif: $running_nodes (expected: $EXPECTED_WORKERS)"
    if [[ "$running_nodes" =~ ^[0-9]+$ ]] && [[ "$EXPECTED_WORKERS" =~ ^[0-9]+$ ]] && (( running_nodes < EXPECTED_WORKERS )); then
      echo "[WARN] NodeManager aktif lebih sedikit dari worker yang dikonfigurasi."
      echo "[WARN] Cek ResourceManager (port 8088/8032) dan service NodeManager di worker."
    fi
  else
    echo "[WARN] Tidak bisa membaca total node YARN."
  fi
}

EXPECTED_WORKERS="$(count_expected_workers)"
if [[ -z "$EXPECTED_WORKERS" || "$EXPECTED_WORKERS" -lt 1 ]]; then
  EXPECTED_WORKERS=2
fi

echo "[INFO] Start Hadoop cluster..."
run_if_exists start-dfs.sh
run_if_exists start-yarn.sh

echo "[INFO] Ringkasan service:"
if command -v jps >/dev/null 2>&1; then
  jps || true
else
  echo "[WARN] jps tidak tersedia."
fi

echo "[INFO] Validasi cluster:"
show_hdfs_health
show_yarn_health

echo "[INFO] Start cluster selesai."
