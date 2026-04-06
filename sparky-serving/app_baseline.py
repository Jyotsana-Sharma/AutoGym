from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import json
import os
import logging
from typing import Optional
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SparkyFitness Meal Recommendation API",
    description="XGBoost ranking model for personalized meal recommendations",
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

# Total: 27 recipe numeric + 16 user + 16 cuisine one-hot = 59
N_FEATURES = 59

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RecipeCandidate(BaseModel):
    recipe_id: int
    name: str
    minutes: float = 0.0
    n_ingredients: float = 0.0
    n_steps: float = 0.0
    avg_rating: float = 0.0
    n_reviews: float = 0.0
    cuisine: str = "unknown"
    calories: float = 0.0
    total_fat: float = 0.0
    sugar: float = 0.0
    sodium: float = 0.0
    protein: float = 0.0
    saturated_fat: float = 0.0
    carbohydrate: float = 0.0
    total_fat_g: float = 0.0
    sugar_g: float = 0.0
    sodium_g: float = 0.0
    protein_g: float = 0.0
    saturated_fat_g: float = 0.0
    carbohydrate_g: float = 0.0
    has_egg: int = 0
    has_fish: int = 0
    has_milk: int = 0
    has_nuts: int = 0
    has_peanut: int = 0
    has_sesame: int = 0
    has_shellfish: int = 0
    has_soy: int = 0
    has_wheat: int = 0


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
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/xgboost_ranker.json")

logger.info("Loading XGBoost model from %s", MODEL_PATH)
model = xgb.Booster()
model.load_model(MODEL_PATH)
logger.info("Model loaded successfully")

# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------

def _encode_cuisine(cuisine: str) -> list[float]:
    """One-hot encode cuisine into a fixed-length vector."""
    vec = [0.0] * len(CUISINE_CATEGORIES)
    cuisine_lower = cuisine.lower().strip()
    if cuisine_lower in CUISINE_CATEGORIES:
        vec[CUISINE_CATEGORIES.index(cuisine_lower)] = 1.0
    else:
        vec[CUISINE_CATEGORIES.index("unknown")] = 1.0
    return vec


def assemble_features(
    user_features: dict, candidate_recipes: list[dict]
) -> np.ndarray:
    """Build a (n_candidates, 59) feature matrix."""
    n = len(candidate_recipes)
    features = np.zeros((n, N_FEATURES), dtype=np.float32)

    # Pre-extract user feature values (same for every row)
    user_vals = [float(user_features.get(f, 0.0)) for f in USER_FEATURES]

    for i, recipe in enumerate(candidate_recipes):
        idx = 0
        # Recipe numeric features (27)
        for f in RECIPE_NUMERIC_FEATURES:
            features[i, idx] = float(recipe.get(f, 0.0))
            idx += 1
        # User features (16)
        for v in user_vals:
            features[i, idx] = v
            idx += 1
        # Cuisine one-hot (16)
        cuisine_vec = _encode_cuisine(recipe.get("cuisine", "unknown"))
        for v in cuisine_vec:
            features[i, idx] = v
            idx += 1

    return features

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "model": "xgboost_ranker", "backend": "xgboost"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        candidates = request.candidate_recipes
        if not candidates:
            raise HTTPException(status_code=400, detail="candidate_recipes must not be empty")

        # Assemble feature matrix
        feature_matrix = assemble_features(request.user_features, candidates)
        dmatrix = xgb.DMatrix(feature_matrix)
        # For ranking models, set group so all candidates are in one query group
        dmatrix.set_group([len(candidates)])

        # Score
        scores = model.predict(dmatrix)

        # Build ranked results
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        ranked_recipes = []
        for rank, (idx, score) in enumerate(indexed_scores, start=1):
            recipe = candidates[idx]
            ranked_recipes.append(
                RankedRecipe(
                    rank=rank,
                    recipe_id=recipe["recipe_id"],
                    name=recipe.get("name", ""),
                    relevance_score=round(float(score), 3),
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
            model_version="xgboost_ranker_v1",
            num_candidates_scored=len(candidates),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# Prometheus metrics
Instrumentator().instrument(app).expose(app)
