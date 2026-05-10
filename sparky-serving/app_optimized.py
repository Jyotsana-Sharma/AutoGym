from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime, timezone
from functools import lru_cache
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SparkyFitness Meal Ranker API (Optimized)",
    description="XGBoost ranking model with infrastructure optimizations",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Feature schema — matches shared contract (contracts/recipe_ranker_input.sample.json)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "minutes", "n_ingredients", "n_steps", "avg_rating", "n_reviews",
    "cuisine",
    "calories", "total_fat", "sugar", "sodium", "protein",
    "saturated_fat", "carbohydrate",
    "total_fat_g", "sugar_g", "sodium_g", "protein_g",
    "saturated_fat_g", "carbohydrate_g",
    "has_egg", "has_fish", "has_milk", "has_nuts", "has_peanut",
    "has_sesame", "has_shellfish", "has_soy", "has_wheat",
    "daily_calorie_target", "protein_target_g", "carbs_target_g", "fat_target_g",
    "user_vegetarian", "user_vegan", "user_gluten_free", "user_dairy_free",
    "user_low_sodium", "user_low_fat",
    "history_pc1", "history_pc2", "history_pc3", "history_pc4",
    "history_pc5", "history_pc6",
]

N_FEATURES = len(FEATURE_COLUMNS)  # 44

# ---------------------------------------------------------------------------
# Request / response models — matches shared contract exactly
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
    generated_at: str
    predictions: list[Prediction]

# ---------------------------------------------------------------------------
# Model loading with LRU cache
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/xgboost_ranker.json")


@lru_cache(maxsize=1)
def _load_model(path: str) -> xgb.Booster:
    logger.info("Loading XGBoost model from %s", path)
    booster = xgb.Booster()
    booster.load_model(path)
    logger.info("Model loaded successfully")
    return booster


model = _load_model(MODEL_PATH)

# Pre-allocate reusable buffer for small batches
_MAX_PREALLOC = 128
_preallocated_buffer = np.zeros((_MAX_PREALLOC, N_FEATURES), dtype=np.float32)

# ---------------------------------------------------------------------------
# Feature assembly — optimized version matching training code
# ---------------------------------------------------------------------------

def assemble_features(instances: list[dict]) -> tuple[np.ndarray, list[int], list[int]]:
    """Build feature matrix with pre-allocation optimization."""
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

    n = len(df)
    if n <= _MAX_PREALLOC:
        features = _preallocated_buffer[:n]
        features[:] = df[FEATURE_COLUMNS].values.astype(np.float32)
        return features.copy(), user_ids, recipe_ids

    return df[FEATURE_COLUMNS].values.astype(np.float32), user_ids, recipe_ids

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "model": "xgb_ranker", "backend": "xgboost_optimized"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        if not request.instances:
            raise HTTPException(status_code=400, detail="instances must not be empty")

        feature_matrix, user_ids, recipe_ids = assemble_features(request.instances)

        dmatrix = xgb.DMatrix(feature_matrix)
        dmatrix.set_group([len(request.instances)])

        scores = model.predict(dmatrix)
        sorted_indices = np.argsort(scores)[::-1]

        predictions = [
            Prediction(
                user_id=user_ids[int(idx)],
                recipe_id=recipe_ids[int(idx)],
                score=round(float(scores[idx]), 4),
                rank=rank,
            )
            for rank, idx in enumerate(sorted_indices, start=1)
        ]

        return PredictResponse(
            request_id=request.request_id,
            model_name=request.model_name,
            model_version="v1_optimized",
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            predictions=predictions,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


Instrumentator().instrument(app).expose(app)
