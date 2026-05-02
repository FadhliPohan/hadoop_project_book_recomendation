#!/usr/bin/env bash
# upload_to_hdfs.sh — Upload dataset lokal ke HDFS
set -euo pipefail

HDFS_TARGET="${1:-/data/amazon_books}"
LOCAL_DATASET_DIR="$(dirname "$0")/../machine_learning/dataset"
LOCAL_DATASET_DIR="$(realpath "$LOCAL_DATASET_DIR")"

echo "[INFO] === Upload Dataset ke HDFS ==="
echo "[INFO] Source : $LOCAL_DATASET_DIR"
echo "[INFO] Target : $HDFS_TARGET"

# Cek hdfs tersedia
if ! command -v hdfs > /dev/null 2>&1; then
  echo "[ERROR] Command 'hdfs' tidak ditemukan. Pastikan Hadoop sudah terinstall dan PATH sudah diset."
  exit 1
fi

# Buat direktori HDFS jika belum ada
echo "[INFO] Membuat direktori HDFS: $HDFS_TARGET"
hdfs dfs -mkdir -p "$HDFS_TARGET" || true

# Upload file CSV
UPLOADED=0
for csv_file in "$LOCAL_DATASET_DIR"/*.csv; do
  if [[ -f "$csv_file" ]]; then
    filename="$(basename "$csv_file")"
    echo "[INFO] Mengupload: $filename → $HDFS_TARGET/$filename"
    hdfs dfs -put -f "$csv_file" "$HDFS_TARGET/$filename"
    UPLOADED=$((UPLOADED + 1))
  fi
done

if [[ $UPLOADED -eq 0 ]]; then
  echo "[WARN] Tidak ada file CSV ditemukan di: $LOCAL_DATASET_DIR"
  echo "[WARN] Pastikan Books_rating.csv sudah ada di folder machine_learning/dataset/"
  exit 1
fi

echo "[INFO] Total file diupload: $UPLOADED"
echo "[INFO] Verifikasi isi HDFS:"
hdfs dfs -ls -h "$HDFS_TARGET"
echo "[INFO] Upload selesai."
