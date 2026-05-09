#!/usr/bin/env bash
# spark_submit_training.sh — Submit training job ke YARN cluster
set -euo pipefail

cd "$(dirname "$0")/.."

STEP="${1:-}"
NUM_EXECUTORS="${2:-2}"
EXECUTOR_CORES="${3:-2}"
EXECUTOR_MEMORY="${4:-2G}"
DRIVER_MEMORY="${5:-2G}"
YARN_PREFLIGHT_TIMEOUT="${YARN_PREFLIGHT_TIMEOUT:-20}"
YARN_PREFLIGHT_RETRIES="${YARN_PREFLIGHT_RETRIES:-6}"
YARN_PREFLIGHT_RETRY_SLEEP="${YARN_PREFLIGHT_RETRY_SLEEP:-2}"

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

run_with_optional_timeout() {
  local timeout_seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout "${timeout_seconds}s" "$@"
  else
    "$@"
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
    echo "[WARN] Hostname '$host' resolve ke loopback. Ini sering membuat ResourceManager/worker salah alamat."
    while IFS= read -r line; do
      echo "[WARN]   $line"
    done <<<"$resolved"
    echo "[WARN] Perbaiki /etc/hosts agar hostname master resolve ke IP LAN, bukan 127.x.x.x."
    if [[ -z "${SPARK_LOCAL_IP:-}" ]]; then
      echo "[WARN] Jika perlu, set env SPARK_LOCAL_IP=<ip_lan_master> sebelum submit."
    fi
  fi
}

check_yarn_health() {
  if ! command -v yarn >/dev/null 2>&1; then
    echo "[WARN] Command yarn tidak ditemukan, skip preflight YARN."
    return 0
  fi

  local report rc running_nodes attempt
  attempt=1
  while (( attempt <= YARN_PREFLIGHT_RETRIES )); do
    rc=0
    report="$(run_with_optional_timeout "$YARN_PREFLIGHT_TIMEOUT" yarn node -list 2>&1)" || rc=$?

    if (( rc != 0 )); then
      echo "[ERROR] Gagal menghubungi YARN ResourceManager lewat 'yarn node -list'."
      while IFS= read -r line; do
        echo "[ERROR]   $line"
      done <<<"$report"
      echo "[ERROR] Periksa ResourceManager, NodeManager, firewall, dan resolusi hostname master."
      return 1
    fi

    running_nodes="$(
      sed -n 's/^Total Nodes:[[:space:]]*\([0-9]\+\).*/\1/p' <<<"$report" | head -n1
    )"
    if [[ -z "$running_nodes" ]]; then
      echo "[ERROR] Output preflight YARN tidak memuat jumlah node."
      while IFS= read -r line; do
        echo "[ERROR]   $line"
      done <<<"$report"
      return 1
    fi

    echo "[INFO] Preflight YARN: Total Nodes = $running_nodes (attempt $attempt/$YARN_PREFLIGHT_RETRIES)"
    if [[ "$running_nodes" =~ ^[0-9]+$ ]] && (( running_nodes > 0 )); then
      return 0
    fi

    if (( attempt < YARN_PREFLIGHT_RETRIES )); then
      echo "[WARN] NodeManager belum terdaftar. Menunggu ${YARN_PREFLIGHT_RETRY_SLEEP} detik..."
      sleep "$YARN_PREFLIGHT_RETRY_SLEEP"
    fi
    attempt=$((attempt + 1))
  done

  echo "[ERROR] Tidak ada NodeManager aktif setelah ${YARN_PREFLIGHT_RETRIES} percobaan."
  echo "[ERROR] Spark job biasanya akan berhenti di status ACCEPTED."
  return 1
}

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
echo "[INFO] sample-fraction  : ${SPARK_SAMPLE_FRACTION:-1.0}"
echo "[INFO] output-partitions: ${SPARK_OUTPUT_PARTITIONS:-0}"
echo "[INFO] max-rows         : ${SPARK_MAX_ROWS:-0}"
warn_loopback_hostname

if [[ "$STEP" == "preprocess_spark" ]]; then
  check_yarn_health
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
