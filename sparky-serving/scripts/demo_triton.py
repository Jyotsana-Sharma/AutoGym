"""
Demo script: sends the agreed contract request (model_input.json) to Triton.

Reads the shared contract JSON, encodes features to float arrays,
sends to Triton v2 API, and prints the response.

Usage:
    python scripts/demo_triton.py
    python scripts/demo_triton.py --model-name meal_ranker_batching
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAMPLE = os.path.join(SCRIPT_DIR, "..", "samples", "model_input.json")

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


def encode_instances(instances):
    df = pd.DataFrame(instances)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            categories = sorted(df[col].fillna("unknown").astype(str).unique())
            cat_map = {v: i for i, v in enumerate(categories)}
            df[col] = df[col].fillna("unknown").astype(str).map(cat_map).fillna(-1).astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    return df[FEATURE_COLUMNS].values.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Demo: send contract request to Triton")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model-name", default="meal_ranker_batching")
    parser.add_argument("--sample-input", default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    # Step 1: Load the agreed contract input
    with open(args.sample_input) as f:
        contract = json.load(f)

    print("=" * 60)
    print("STEP 1: Agreed contract input (model_input.json)")
    print("=" * 60)
    print(json.dumps(contract, indent=2)[:500])
    print(f"  ... ({len(contract['instances'])} instances, {len(FEATURE_COLUMNS)} features each)\n")

    # Step 2: Encode features to float array
    feature_matrix = encode_instances(contract["instances"])
    print("=" * 60)
    print("STEP 2: Encode features to float array for Triton")
    print("=" * 60)
    print(f"  Shape: {feature_matrix.shape}")
    print(f"  First row (first 10 values): {feature_matrix[0][:10].tolist()}\n")

    # Step 3: Send to Triton
    triton_payload = {
        "inputs": [{
            "name": "input__0",
            "shape": list(feature_matrix.shape),
            "datatype": "FP32",
            "data": feature_matrix.flatten().tolist(),
        }]
    }

    infer_url = f"{args.url}/v2/models/{args.model_name}/infer"
    print("=" * 60)
    print(f"STEP 3: Send to Triton ({args.model_name})")
    print("=" * 60)
    print(f"  URL: {infer_url}")

    resp = requests.post(infer_url, json=triton_payload, timeout=10)

    print(f"  Status: {resp.status_code}\n")

    # Step 4: Show response
    print("=" * 60)
    print("STEP 4: Triton response")
    print("=" * 60)
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
