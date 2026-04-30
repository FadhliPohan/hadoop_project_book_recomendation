#!/usr/bin/env bash
set -euo pipefail

run_if_exists() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "[INFO] Menjalankan: $cmd"
    "$cmd"
  else
    echo "[WARN] Command tidak ditemukan: $cmd"
  fi
}

echo "[INFO] Start Hadoop cluster..."
run_if_exists start-dfs.sh
run_if_exists start-yarn.sh

echo "[INFO] Ringkasan service:"
if command -v jps >/dev/null 2>&1; then
  jps || true
else
  echo "[WARN] jps tidak tersedia."
fi

echo "[INFO] Start cluster selesai."
