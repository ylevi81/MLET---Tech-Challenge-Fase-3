#!/bin/sh
set -eu

model_path="${MEDICAL_TRIAGE_MODEL_PATH:-/app/artifacts/model.joblib}"
metrics_path="${METRICS_PATH:-/app/artifacts/metrics.json}"

if [ ! -s "$model_path" ]; then
  if [ "${TRAIN_IF_MISSING:-0}" = "1" ]; then
    echo "Model artifact not found; downloading the Kaggle corpus and training once."
    python -m medical_triage.train \
      --model-path "$model_path" \
      --metrics-path "$metrics_path"
  else
    echo "Model artifact not found at $model_path." >&2
    echo "Run the training command or set TRAIN_IF_MISSING=1." >&2
    exit 1
  fi
fi

exec "$@"
