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

echo "[INFO] Stop Hadoop cluster..."
run_if_exists stop-yarn.sh
run_if_exists stop-dfs.sh

echo "[INFO] Ringkasan service setelah stop:"
if command -v jps >/dev/null 2>&1; then
  jps || true
else
  echo "[WARN] jps tidak tersedia."
fi

echo "[INFO] Stop cluster selesai."
