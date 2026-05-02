#!/usr/bin/env bash
# upload_to_hdfs.sh — Upload dataset lokal ke HDFS
set -euo pipefail

CURRENT_USER="${USER:-fadhli}"
DEFAULT_HDFS_TARGET="/user/${CURRENT_USER}/amazon_books"
HDFS_TARGET="${1:-$DEFAULT_HDFS_TARGET}"
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
if ! hdfs dfs -mkdir -p "$HDFS_TARGET"; then
  echo "[ERROR] Gagal membuat direktori HDFS: $HDFS_TARGET"
  echo "[ERROR] Kemungkinan besar user saat ini tidak punya izin tulis ke parent path."
  echo "[INFO] Coba gunakan path yang bisa ditulis user, contoh: $DEFAULT_HDFS_TARGET"
  echo "[INFO] Atau minta admin HDFS membuat dan memberi ownership:"
  echo "[INFO]   hdfs dfs -mkdir -p $HDFS_TARGET"
  echo "[INFO]   hdfs dfs -chown -R ${CURRENT_USER}:<group_hdfs_user> $HDFS_TARGET"
  exit 1
fi

# Upload file CSV
UPLOADED=0
if [[ ! -d "$LOCAL_DATASET_DIR" ]]; then
  echo "[ERROR] Directory dataset tidak ditemukan: $LOCAL_DATASET_DIR"
  exit 1
fi

# NOTE:
# Beberapa versi Hadoop CLI bermasalah saat source path lokal mengandung spasi.
# Karena root project ini mengandung "MY PROJECT", kita upload dari path relatif.
pushd "$LOCAL_DATASET_DIR" > /dev/null
shopt -s nullglob
csv_files=(*.csv)
shopt -u nullglob

for filename in "${csv_files[@]}"; do
  echo "[INFO] Mengupload: $filename → $HDFS_TARGET/$filename"
  hdfs dfs -put -f "./$filename" "$HDFS_TARGET/$filename"
  UPLOADED=$((UPLOADED + 1))
done
popd > /dev/null

if [[ $UPLOADED -eq 0 ]]; then
  echo "[WARN] Tidak ada file CSV ditemukan di: $LOCAL_DATASET_DIR"
  echo "[WARN] Pastikan Books_rating.csv sudah ada di folder machine_learning/dataset/"
  exit 1
fi

echo "[INFO] Total file diupload: $UPLOADED"
echo "[INFO] Verifikasi isi HDFS:"
hdfs dfs -ls -h "$HDFS_TARGET"
echo "[INFO] Upload selesai."
