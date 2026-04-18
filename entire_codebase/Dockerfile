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

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Safeguarding module needed by training (fairness gate) and serving (/explain)
COPY safeguarding/ ./safeguarding/

# ── Shared Python dependencies ────────────────────────────────────────────────
# data/training/serving share heavy scientific packages. Installing the union
# once lets Docker reuse the SciPy/Pandas/XGBoost layers across those images.
FROM base AS python-deps

COPY requirements/data.txt requirements/data.txt
COPY requirements/training.txt requirements/training.txt
COPY requirements/serving.txt requirements/serving.txt
RUN cat requirements/data.txt requirements/training.txt requirements/serving.txt \
    > /tmp/requirements-all.txt \
    && pip install -r /tmp/requirements-all.txt \
    && rm -rf /tmp/requirements-all.txt

# ── Data stage ────────────────────────────────────────────────────────────────
FROM python-deps AS data

COPY src/data/ ./src/data/
COPY configs/data/ ./configs/data/

CMD ["python", "--version"]

# ── Training stage ────────────────────────────────────────────────────────────
FROM python-deps AS training

COPY src/training/ ./src/training/
COPY configs/training/ ./configs/training/

CMD ["python", "-m", "src.training.retrain_pipeline", "--help"]

# ── Serving stage ─────────────────────────────────────────────────────────────
FROM python-deps AS serving

COPY src/serving/ ./src/serving/

EXPOSE 8000
CMD ["uvicorn", "src.serving.app_production:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ── MLflow stage (self-contained, no shared deps needed) ──────────────────────
FROM python:3.11-slim AS mlflow

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1

RUN pip install mlflow==3.1.0

RUN mkdir -p /app/local/mlflow /app/local/mlartifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////app/local/mlflow/mlflow.db", \
     "--default-artifact-root", "/app/local/mlartifacts", \
     "--host", "0.0.0.0", "--port", "5000"]
