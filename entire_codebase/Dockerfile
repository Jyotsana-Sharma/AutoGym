# syntax=docker/dockerfile:1
#
# SparkyFitness ML — unified multi-stage Dockerfile
#
# Stages (select with --target or docker-compose target:):
#   data      → data pipeline, drift monitor, soda checks
#   training  → XGBoost training, retrain API
#   serving   → FastAPI prediction server
#   mlflow    → MLflow tracking server (minimal, no shared base needed)
#
# Each stage gets only its own dependencies, so images stay small.
# The shared "base" stage carries OS packages + the safeguarding/ module,
# which is now required by both training (fairness gate) and serving (/explain).

# ── Shared base ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Safeguarding module needed by training (fairness gate) and serving (/explain)
COPY safeguarding/ ./safeguarding/

# ── Data stage ────────────────────────────────────────────────────────────────
FROM base AS data

COPY requirements/data.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/data/ ./src/data/
COPY configs/data/ ./configs/data/

CMD ["python", "--version"]

# ── Training stage ────────────────────────────────────────────────────────────
FROM base AS training

COPY requirements/training.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/training/ ./src/training/
COPY configs/training/ ./configs/training/

CMD ["python", "-m", "src.training.retrain_pipeline", "--help"]

# ── Serving stage ─────────────────────────────────────────────────────────────
FROM base AS serving

COPY requirements/serving.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/serving/ ./src/serving/

EXPOSE 8000
CMD ["uvicorn", "src.serving.app_production:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ── MLflow stage (self-contained, no shared deps needed) ──────────────────────
FROM python:3.11-slim AS mlflow

WORKDIR /app

RUN pip install --no-cache-dir mlflow==3.1.0

RUN mkdir -p /app/local/mlflow /app/local/mlartifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////app/local/mlflow/mlflow.db", \
     "--default-artifact-root", "/app/local/mlartifacts", \
     "--host", "0.0.0.0", "--port", "5000"]
