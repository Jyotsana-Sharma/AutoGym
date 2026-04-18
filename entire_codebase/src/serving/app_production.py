"""Production FastAPI serving app.

Key differences from app_baseline.py:
  1. Loads model from MLflow Model Registry (Production stage)
     → falls back to MODEL_FALLBACK_PATH if Registry unavailable
  2. Watches for new Production model versions and hot-reloads without downtime
  3. Logs every prediction to PostgreSQL for retraining feedback loop
  4. Reports model version in every response (enables A/B analysis)
  5. Exposes Prometheus metrics for monitoring and alerting
  6. Supports graceful model rollback via /admin/rollback endpoint

Environment variables:
  MLFLOW_TRACKING_URI      — MLflow server URL (default: http://localhost:5000)
  MLFLOW_MODEL_NAME        — registered model name (default: sparky-ranker)
  MODEL_FALLBACK_PATH      — local model file if Registry unavailable
  DATABASE_URL             — PostgreSQL DSN for prediction logging
  LOG_PREDICTIONS          — "true" to enable prediction logging (default: true)
  MODEL_POLL_INTERVAL_SEC  — how often to check for new model version (default: 60)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

from .prediction_logger import PredictionLogger
from .model_loader import ModelLoader
from .feature_contract import FEATURE_COLUMNS, ID_COLUMNS

# Safeguarding: explainability — import with graceful fallback
try:
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
    from safeguarding.explainability import Explainer
    _EXPLAINER_AVAILABLE = True
except Exception:
    _EXPLAINER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "sparky-ranker")
MODEL_FALLBACK_PATH = os.environ.get("MODEL_FALLBACK_PATH", "/models/xgboost_ranker.json")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
LOG_PREDICTIONS = os.environ.get("LOG_PREDICTIONS", "true").lower() == "true"
MODEL_POLL_INTERVAL = int(os.environ.get("MODEL_POLL_INTERVAL_SEC", "60"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------------------------------------------------------------------
# Prometheus custom metrics
# ---------------------------------------------------------------------------
prediction_score_hist = Histogram(
    "sparky_prediction_score",
    "Distribution of prediction scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
prediction_count_total = Counter(
    "sparky_predictions_total",
    "Total number of predictions served",
    ["model_version"],
)
model_version_gauge = Gauge(
    "sparky_model_version",
    "Currently loaded model version (numeric part)",
)
model_reload_counter = Counter(
    "sparky_model_reloads_total",
    "Number of times the model was reloaded due to a new version",
)
model_loaded_timestamp = Gauge(
    "sparky_model_loaded_timestamp_seconds",
    "Unix timestamp when the current model was loaded",
)
prediction_last_logged_timestamp = Gauge(
    "sparky_prediction_last_logged_timestamp_seconds",
    "Unix timestamp when prediction logging last succeeded",
)
request_batch_size_hist = Histogram(
    "sparky_request_batch_size",
    "Number of instances per prediction request",
    buckets=[1, 5, 10, 20, 50, 100, 200],
)

# ---------------------------------------------------------------------------
# Global model state (protected by a threading.RLock for hot-reload)
# ---------------------------------------------------------------------------
_model_lock = threading.RLock()
_model: xgb.Booster | None = None
_model_version: str = "unknown"
_model_source: str = "none"
_explainer: "Explainer | None" = None

loader = ModelLoader(
    model_name=MLFLOW_MODEL_NAME,
    fallback_path=MODEL_FALLBACK_PATH,
    tracking_uri=MLFLOW_TRACKING_URI,
)
prediction_logger = PredictionLogger(database_url=DATABASE_URL) if LOG_PREDICTIONS and DATABASE_URL else None


def _load_model_once():
    global _model, _model_version, _model_source, _explainer
    m, version, source = loader.load_production_model()
    with _model_lock:
        _model = m
        _model_version = version
        _model_source = source
        # Safeguarding: build explainer whenever model reloads
        if _EXPLAINER_AVAILABLE and m is not None:
            try:
                _explainer = Explainer(m, feature_columns=FEATURE_COLUMNS)
                logger.info("SHAP Explainer initialized for model version=%s", version)
            except Exception as exc:
                logger.warning("Explainer init failed (non-fatal): %s", exc)
                _explainer = None
    try:
        model_version_gauge.set(float(version.lstrip("v")) if version.lstrip("v").isdigit() else 0)
        model_loaded_timestamp.set(time.time())
    except Exception:
        pass
    logger.info("Model loaded: version=%s source=%s", version, source)


def _poll_for_new_model():
    """Background thread: poll MLflow Registry every MODEL_POLL_INTERVAL seconds."""
    while True:
        time.sleep(MODEL_POLL_INTERVAL)
        try:
            latest_version = loader.get_latest_production_version()
            if latest_version and latest_version != _model_version:
                logger.info(
                    "New Production model detected: %s → %s. Reloading...",
                    _model_version,
                    latest_version,
                )
                _load_model_once()
                model_reload_counter.inc()
        except Exception as exc:
            logger.warning("Model poll failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_once()
    poll_thread = threading.Thread(target=_poll_for_new_model, daemon=True)
    poll_thread.start()
    if prediction_logger:
        await prediction_logger.connect()
    logger.info("SparkyFitness serving started. model_version=%s", _model_version)
    yield
    if prediction_logger:
        await prediction_logger.close()
    logger.info("SparkyFitness serving stopped.")


app = FastAPI(
    title="SparkyFitness Meal Ranker API (Production)",
    description="XGBoost ranking model with MLflow Registry, prediction logging, and hot-reload",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    request_id: str
    model_name: str = "xgb_ranker"
    instances: list[dict]


class Prediction(BaseModel):
    user_id: int
    recipe_id: int
    score: float
    rank: int


class PredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    model_source: str
    generated_at: str
    predictions: list[Prediction]


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def assemble_features(instances: list[dict]) -> tuple[np.ndarray, list[int], list[int]]:
    df = pd.DataFrame(instances)
    user_ids = df["user_id"].astype(int).tolist()
    recipe_ids = df["recipe_id"].astype(int).tolist()

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            categories = sorted(df[col].fillna("unknown").astype(str).unique())
            cat_map = {v: i for i, v in enumerate(categories)}
            df[col] = df[col].fillna("unknown").astype(str).map(cat_map).fillna(-1).astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    return df[FEATURE_COLUMNS].values.astype(np.float32), user_ids, recipe_ids

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    with _model_lock:
        return {
            "status": "healthy" if _model is not None else "degraded",
            "model_version": _model_version,
            "model_source": _model_source,
            "model_name": MLFLOW_MODEL_NAME,
        }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(status_code=400, detail="instances must not be empty")

    with _model_lock:
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        model = _model
        version = _model_version
        source = _model_source

    request_batch_size_hist.observe(len(request.instances))

    try:
        feature_matrix, user_ids, recipe_ids = assemble_features(request.instances)
        dmatrix = xgb.DMatrix(feature_matrix, feature_names=FEATURE_COLUMNS)
        dmatrix.set_group([len(request.instances)])
        scores = model.predict(dmatrix)
    except Exception as exc:
        logger.exception("Feature assembly / inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")

    sorted_indices = np.argsort(scores)[::-1]
    predictions = []
    for rank, idx in enumerate(sorted_indices, start=1):
        score = round(float(scores[idx]), 4)
        predictions.append(Prediction(
            user_id=user_ids[int(idx)],
            recipe_id=recipe_ids[int(idx)],
            score=score,
            rank=rank,
        ))
        prediction_score_hist.observe(score)

    prediction_count_total.labels(model_version=version).inc(len(predictions))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    response = PredictResponse(
        request_id=request.request_id,
        model_name=request.model_name,
        model_version=version,
        model_source=source,
        generated_at=now,
        predictions=predictions,
    )

    # Async prediction logging for feedback loop
    if prediction_logger:
        try:
            await prediction_logger.log_batch(
                request_id=request.request_id,
                model_version=version,
                predictions=predictions,
                timestamp=now,
                features=request.instances,
            )
            prediction_last_logged_timestamp.set(time.time())
        except Exception as exc:
            logger.warning("Prediction logging failed (non-fatal): %s", exc)

        # Safeguarding: log raw inference features for drift monitoring
        try:
            await prediction_logger.log_features(
                request_id=request.request_id,
                model_version=version,
                instances=request.instances,
            )
        except Exception as exc:
            logger.warning("Feature logging failed (non-fatal): %s", exc)

    return response


@app.post("/feedback")
async def receive_feedback(payload: dict):
    """
    Accept user feedback (ratings, clicks) for a given request_id.
    Stores in PostgreSQL for the retraining feedback loop.
    """
    if prediction_logger:
        try:
            await prediction_logger.log_feedback(payload)
        except Exception as exc:
            logger.warning("Feedback logging failed: %s", exc)
            raise HTTPException(status_code=500, detail="Feedback storage failed")
    return {"accepted": True}


class ExplainRequest(BaseModel):
    instance: dict
    top_k: int = 10


@app.post("/explain")
async def explain_prediction(request: ExplainRequest):
    """
    Safeguarding — Explainability: return SHAP-based explanation for a single
    candidate recipe instance. Answers "why was this recipe recommended?".

    Pass the same feature dict you would send as one element of /predict instances.
    Returns top contributing features and a human-readable explanation string.
    """
    with _model_lock:
        explainer = _explainer

    if explainer is None:
        if not _EXPLAINER_AVAILABLE:
            raise HTTPException(
                status_code=501,
                detail="Explainability not available — shap package not installed",
            )
        raise HTTPException(status_code=503, detail="Explainer not initialized yet")

    try:
        result = explainer.explain_prediction(request.instance, top_k=request.top_k)
    except Exception as exc:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(exc)}")

    return result


@app.post("/admin/reload")
def reload_model():
    """Force a model reload from MLflow Registry."""
    _load_model_once()
    model_reload_counter.inc()
    return {"reloaded": True, "model_version": _model_version, "source": _model_source}


@app.get("/admin/model-info")
def model_info():
    with _model_lock:
        return {
            "model_version": _model_version,
            "model_source": _model_source,
            "model_name": MLFLOW_MODEL_NAME,
            "id_columns": ID_COLUMNS,
            "features": FEATURE_COLUMNS,
            "feature_count": len(FEATURE_COLUMNS),
        }


# Prometheus metrics endpoint
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
