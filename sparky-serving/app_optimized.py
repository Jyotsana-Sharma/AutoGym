from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import os
import logging
from typing import Optional
from functools import lru_cache
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SparkyFitness Meal Recommendation API (Optimized)",
    description="XGBoost ranking model with infrastructure optimizations",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

RECIPE_NUMERIC_FEATURES = [
    "minutes", "n_ingredients", "n_steps", "avg_rating", "n_reviews",
    "calories", "total_fat", "sugar", "sodium", "protein",
    "saturated_fat", "carbohydrate", "total_fat_g", "sugar_g", "sodium_g",
    "protein_g", "saturated_fat_g", "carbohydrate_g",
    "has_egg", "has_fish", "has_milk", "has_nuts", "has_peanut",
    "has_sesame", "has_shellfish", "has_soy", "has_wheat",
]

USER_FEATURES = [
    "daily_calorie_target", "protein_target_g", "carbs_target_g", "fat_target_g",
    "user_vegetarian", "user_vegan", "user_gluten_free", "user_dairy_free",
    "user_low_sodium", "user_low_fat",
    "history_pc1", "history_pc2", "history_pc3", "history_pc4",
    "history_pc5", "history_pc6",
]

CUISINE_CATEGORIES = [
    "african", "american", "asian", "chinese", "european", "french",
    "german", "greek", "indian", "latin_american", "mexican",
    "middle_eastern", "pacific", "scandinavian", "spanish", "unknown",
]

N_FEATURES = 59
N_RECIPE_FEATURES = len(RECIPE_NUMERIC_FEATURES)  # 27
N_USER_FEATURES = len(USER_FEATURES)  # 16
N_CUISINE_FEATURES = len(CUISINE_CATEGORIES)  # 16

# Pre-compute cuisine lookup for O(1) encoding
_CUISINE_INDEX = {c: i for i, c in enumerate(CUISINE_CATEGORIES)}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    user_id: Optional[int] = None
    user_features: dict
    candidate_recipes: list[dict]


class RankedRecipe(BaseModel):
    rank: int
    recipe_id: int
    name: str
    relevance_score: float
    calories: float
    protein_g: float
    carbohydrate_g: float
    total_fat_g: float
    minutes: float
    cuisine: str


class PredictResponse(BaseModel):
    user_id: Optional[int] = None
    ranked_recipes: list[RankedRecipe]
    model_version: str
    num_candidates_scored: int

# ---------------------------------------------------------------------------
# Model loading with LRU cache
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/xgboost_ranker.json")


@lru_cache(maxsize=1)
def _load_model(path: str) -> xgb.Booster:
    """Load model with caching so reloads are free."""
    logger.info("Loading XGBoost model from %s", path)
    booster = xgb.Booster()
    booster.load_model(path)
    logger.info("Model loaded successfully")
    return booster


# Pre-load model at module level for fast first request
model = _load_model(MODEL_PATH)

# Pre-allocate a reusable buffer for small batch sizes (common case)
_MAX_PREALLOC = 128
_preallocated_buffer = np.zeros((_MAX_PREALLOC, N_FEATURES), dtype=np.float32)

# ---------------------------------------------------------------------------
# Feature assembly (optimized)
# ---------------------------------------------------------------------------

def _encode_cuisine_fast(cuisine: str) -> int:
    """Return the index for one-hot encoding, defaulting to 'unknown'."""
    return _CUISINE_INDEX.get(cuisine.lower().strip(), _CUISINE_INDEX["unknown"])


def assemble_features(
    user_features: dict, candidate_recipes: list[dict]
) -> np.ndarray:
    """Build a (n_candidates, 59) feature matrix with pre-allocation."""
    n = len(candidate_recipes)

    # Reuse pre-allocated buffer when possible to avoid allocation overhead
    if n <= _MAX_PREALLOC:
        features = _preallocated_buffer[:n]
        features[:] = 0.0  # Reset to zero
    else:
        features = np.zeros((n, N_FEATURES), dtype=np.float32)

    # Pre-extract user feature values once (shared across all rows)
    user_vals = np.array(
        [float(user_features.get(f, 0.0)) for f in USER_FEATURES],
        dtype=np.float32,
    )
    user_start = N_RECIPE_FEATURES
    user_end = user_start + N_USER_FEATURES
    cuisine_start = user_end

    # Broadcast user features across all rows at once
    features[:, user_start:user_end] = user_vals

    for i, recipe in enumerate(candidate_recipes):
        # Recipe numeric features
        for j, f in enumerate(RECIPE_NUMERIC_FEATURES):
            features[i, j] = float(recipe.get(f, 0.0))

        # Cuisine one-hot (single index lookup)
        cuisine_idx = _encode_cuisine_fast(recipe.get("cuisine", "unknown"))
        features[i, cuisine_start + cuisine_idx] = 1.0

    # Return a copy if using pre-allocated buffer (DMatrix may hold reference)
    if n <= _MAX_PREALLOC:
        return features.copy()
    return features

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "model": "xgboost_ranker", "backend": "xgboost_optimized"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        candidates = request.candidate_recipes
        if not candidates:
            raise HTTPException(status_code=400, detail="candidate_recipes must not be empty")

        # Assemble feature matrix
        feature_matrix = assemble_features(request.user_features, candidates)
        dmatrix = xgb.DMatrix(feature_matrix)
        dmatrix.set_group([len(candidates)])

        # Score
        scores = model.predict(dmatrix)

        # Use numpy argsort for faster ranking (descending)
        sorted_indices = np.argsort(scores)[::-1]

        ranked_recipes = []
        for rank, idx in enumerate(sorted_indices, start=1):
            recipe = candidates[int(idx)]
            ranked_recipes.append(
                RankedRecipe(
                    rank=rank,
                    recipe_id=recipe["recipe_id"],
                    name=recipe.get("name", ""),
                    relevance_score=round(float(scores[idx]), 3),
                    calories=float(recipe.get("calories", 0.0)),
                    protein_g=float(recipe.get("protein_g", 0.0)),
                    carbohydrate_g=float(recipe.get("carbohydrate_g", 0.0)),
                    total_fat_g=float(recipe.get("total_fat_g", 0.0)),
                    minutes=float(recipe.get("minutes", 0.0)),
                    cuisine=recipe.get("cuisine", "unknown"),
                )
            )

        return PredictResponse(
            user_id=request.user_id,
            ranked_recipes=ranked_recipes,
            model_version="xgboost_ranker_v1_optimized",
            num_candidates_scored=len(candidates),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Prometheus metrics
Instrumentator().instrument(app).expose(app)
