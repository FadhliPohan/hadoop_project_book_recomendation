#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ML_DIR="$ROOT_DIR/machine_learning"

remove_dir_contents() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    echo "[INFO] Skip, directory tidak ada: $dir"
    return
  fi

  echo "[INFO] Membersihkan isi directory: $dir"
  find "$dir" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +
}

echo "[INFO] Reset training state dimulai..."
echo "[INFO] Dataset mentah tidak akan dihapus."

remove_dir_contents "$ML_DIR/data/processed"
remove_dir_contents "$ML_DIR/models/sentiment"
remove_dir_contents "$ML_DIR/models/recommender"
remove_dir_contents "$ML_DIR/reports"
remove_dir_contents "$ML_DIR/logs"
remove_dir_contents "$ML_DIR/mlruns"

if [[ -f "$ML_DIR/models/model_registry.json" ]]; then
  echo "[INFO] Menghapus file registry model"
  rm -f "$ML_DIR/models/model_registry.json"
fi

if [[ -f "$ROOT_DIR/output/metrics.json" ]]; then
  echo "[INFO] Menghapus metrics lama di output/"
  rm -f "$ROOT_DIR/output/metrics.json"
fi

echo "[INFO] Reset training state selesai."
