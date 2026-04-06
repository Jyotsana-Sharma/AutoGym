"""
Train a placeholder XGBoost ranking model from the data team's training table.
Produces: models/xgboost_ranker.json (native) for serving experiments.
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVING_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(SERVING_DIR, "data"))
MODEL_DIR = os.path.join(SERVING_DIR, "models")

# --- Feature columns (exclude identifiers, labels, and non-numeric) ---
ID_COLS = ["user_id", "recipe_id", "date", "rating", "label", "name"]
CUISINE_COL = "cuisine"

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    # One-hot encode cuisine
    df = pd.get_dummies(df, columns=[CUISINE_COL], prefix="cuisine")
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    return df, feature_cols

def build_group_sizes(df):
    """Build group sizes for ranking: group by user_id."""
    groups = df.groupby("user_id").size().values
    return groups

def main():
    print("Loading training data...")
    train_df, feature_cols = load_and_prepare(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    val_df = pd.get_dummies(val_df, columns=[CUISINE_COL], prefix="cuisine")

    # Align columns (val may have missing cuisine dummies)
    for col in feature_cols:
        if col not in val_df.columns:
            val_df[col] = 0
    val_df = val_df[[c for c in train_df.columns if c in val_df.columns]]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)
    groups_train = build_group_sizes(train_df)

    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)
    groups_val = build_group_sizes(val_df)

    print(f"Train: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Val:   {X_val.shape[0]} rows")
    print(f"Train groups: {len(groups_train)}, Val groups: {len(groups_val)}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(groups_train)

    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(groups_val)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "seed": 42,
    }

    print("Training XGBoost ranker...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train"), (dval, "val")],
        verbose_eval=20,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "xgboost_ranker.json")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Save feature names for serving
    meta = {
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "n_trees": model.num_boosted_rounds(),
    }
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model metadata saved to {meta_path}")

    # Quick sanity check
    preds = model.predict(dval)
    print(f"Sample predictions (first 10): {preds[:10]}")
    print(f"Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")

if __name__ == "__main__":
    main()
