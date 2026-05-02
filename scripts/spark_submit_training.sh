#!/usr/bin/env bash
# spark_submit_training.sh — Submit training job ke YARN cluster
set -euo pipefail

cd "$(dirname "$0")/.."

STEP="${1:-}"
NUM_EXECUTORS="${2:-2}"
EXECUTOR_CORES="${3:-2}"
EXECUTOR_MEMORY="${4:-2G}"
DRIVER_MEMORY="${5:-2G}"

if [[ -z "$STEP" ]]; then
  echo "Usage: ./scripts/spark_submit_training.sh <step> [num-executors] [executor-cores] [executor-memory] [driver-memory]"
  echo ""
  echo "Contoh:"
  echo "  ./scripts/spark_submit_training.sh preprocess_spark 2 2 2G 2G"
  echo ""
  echo "Steps tersedia untuk distributed (PySpark):"
  echo "  preprocess_spark  — Distributed preprocessing via Spark on YARN"
  echo ""
  echo "Steps training tetap berjalan di master (scikit-learn):"
  echo "  train_sentiment_baseline | train_recommender | evaluate"
  exit 1
fi

# Pilih Python bin
PYTHON_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

# Cek spark-submit tersedia
if ! command -v spark-submit > /dev/null 2>&1; then
  echo "[ERROR] spark-submit tidak ditemukan. Pastikan Apache Spark sudah terinstall dan PATH diset."
  exit 1
fi

echo "[INFO] Submitting ke YARN..."
echo "[INFO] Step             : $STEP"
echo "[INFO] num-executors    : $NUM_EXECUTORS"
echo "[INFO] executor-cores   : $EXECUTOR_CORES"
echo "[INFO] executor-memory  : $EXECUTOR_MEMORY"
echo "[INFO] driver-memory    : $DRIVER_MEMORY"

if [[ "$STEP" == "preprocess_spark" ]]; then
  SCRIPT_PATH="machine_learning/src/spark_preprocess.py"
  echo "[INFO] Script : $SCRIPT_PATH"

  spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors "$NUM_EXECUTORS" \
    --executor-cores "$EXECUTOR_CORES" \
    --executor-memory "$EXECUTOR_MEMORY" \
    --driver-memory "$DRIVER_MEMORY" \
    --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=$PYTHON_BIN" \
    --conf "spark.executorEnv.PYSPARK_PYTHON=$PYTHON_BIN" \
    "$SCRIPT_PATH"
else
  # Fallback: jalankan step biasa dengan --allow-training
  echo "[INFO] Step '$STEP' berjalan di master (scikit-learn, bukan distributed Spark)"
  "$PYTHON_BIN" machine_learning/main.py --step "$STEP" --allow-training
fi

echo "[INFO] Job selesai."
