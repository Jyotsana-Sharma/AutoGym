#!/bin/sh
set -eu

mkdir -p /app/local/mlflow /app/local/mlartifacts

exec python -m mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
  --artifacts-destination "$MLFLOW_ARTIFACT_ROOT"
