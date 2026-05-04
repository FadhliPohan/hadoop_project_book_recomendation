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

warn_loopback_hostname() {
  if ! command -v hostname >/dev/null 2>&1 || ! command -v getent >/dev/null 2>&1; then
    return 0
  fi

  local host resolved
  host="$(hostname 2>/dev/null || true)"
  [[ -z "$host" ]] && return 0

  resolved="$(getent hosts "$host" 2>/dev/null || true)"
  if [[ -n "$resolved" ]] && grep -Eq '(^|[[:space:]])127\.' <<<"$resolved"; then
    echo "[WARN] Hostname '$host' resolve ke loopback. Ini bisa membuat HDFS/YARN salah alamat."
    while IFS= read -r line; do
      echo "[WARN]   $line"
    done <<<"$resolved"
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

  local report running_nodes attempt
  for attempt in 1 2 3 4 5 6; do
    if command -v timeout >/dev/null 2>&1; then
      report="$(timeout 20s yarn node -list 2>&1)" || true
    else
      report="$(yarn node -list 2>&1)" || true
    fi
    if [[ -z "$report" ]]; then
      echo "[WARN] Gagal membaca yarn node -list"
      return
    fi

    running_nodes="$(
      sed -n 's/^Total Nodes:[[:space:]]*\([0-9]\+\).*/\1/p' <<<"$report" | head -n1
    )"
    if [[ -n "$running_nodes" ]]; then
      echo "[INFO] YARN NodeManager aktif: $running_nodes (expected: $EXPECTED_WORKERS, attempt: $attempt/6)"
      if [[ "$running_nodes" =~ ^[0-9]+$ ]] && (( running_nodes > 0 )); then
        if [[ "$EXPECTED_WORKERS" =~ ^[0-9]+$ ]] && (( running_nodes < EXPECTED_WORKERS )); then
          echo "[WARN] NodeManager aktif lebih sedikit dari worker yang dikonfigurasi."
          echo "[WARN] Cek ResourceManager (port 8088/8032) dan service NodeManager di worker."
        fi
        return
      fi
    else
      echo "[WARN] Tidak bisa membaca total node YARN."
      return
    fi

    if (( attempt < 6 )); then
      echo "[WARN] NodeManager belum terdaftar, tunggu 2 detik lalu cek lagi."
      sleep 2
    fi
  done

  echo "[WARN] Setelah 6 percobaan, belum ada NodeManager aktif di YARN."
  echo "[WARN] Cek ResourceManager (port 8088/8032) dan service NodeManager di worker."
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
warn_loopback_hostname
show_hdfs_health
show_yarn_health

echo "[INFO] Start cluster selesai."
