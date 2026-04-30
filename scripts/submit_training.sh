#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
STEP="${1:-}"

if [[ -z "$STEP" ]]; then
  echo "Usage: ./scripts/submit_training.sh <step>"
  echo "Example: ./scripts/submit_training.sh train_sentiment_baseline"
  exit 1
fi

PYTHON_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" machine_learning/main.py --step "$STEP" --allow-training
