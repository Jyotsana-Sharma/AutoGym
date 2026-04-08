"""
Generate a placeholder XGBoost ranking model for serving benchmarks.

Creates a valid model from synthetic data — no external CSVs needed.
The model accepts 44 float features and returns relevance scores,
matching the shared contract interface.

Usage:
    python scripts/train_placeholder_model.py
"""

import json
import os

import numpy as np
import xgboost as xgb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")

FEATURE_NAMES = [
    "minutes", "n_ingredients", "n_steps", "avg_rating", "n_reviews", "cuisine",
    "calories", "total_fat", "sugar", "sodium", "protein", "saturated_fat", "carbohydrate",
    "total_fat_g", "sugar_g", "sodium_g", "protein_g", "saturated_fat_g", "carbohydrate_g",
    "has_egg", "has_fish", "has_milk", "has_nuts", "has_peanut", "has_sesame",
    "has_shellfish", "has_soy", "has_wheat",
    "daily_calorie_target", "protein_target_g", "carbs_target_g", "fat_target_g",
    "user_vegetarian", "user_vegan", "user_gluten_free", "user_dairy_free",
    "user_low_sodium", "user_low_fat",
    "history_pc1", "history_pc2", "history_pc3", "history_pc4",
    "history_pc5", "history_pc6",
]

N_FEATURES = len(FEATURE_NAMES)  # 44


def main():
    rng = np.random.default_rng(42)

    n_users, recipes_per_user = 20, 10
    n_rows = n_users * recipes_per_user

    X = rng.standard_normal((n_rows, N_FEATURES)).astype(np.float32)
    labels = rng.integers(0, 5, size=n_rows).astype(np.float32)
    groups = np.full(n_users, recipes_per_user)

    dtrain = xgb.DMatrix(X, label=labels)
    dtrain.set_group(groups)

    model = xgb.train(
        {"objective": "rank:ndcg", "eta": 0.1, "max_depth": 4, "seed": 42},
        dtrain,
        num_boost_round=50,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "xgboost_ranker.json")
    model.save_model(model_path)
    print(f"Model saved: {model_path}")

    meta = {"feature_names": FEATURE_NAMES, "n_features": N_FEATURES}
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
