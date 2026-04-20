"""Smoke test for the production serving endpoint.

Used by CI/CD to verify a deployment is healthy before finalizing promotion.
Tests:
  1. /health returns healthy
  2. /predict returns valid predictions for a sample request
  3. Prediction scores are in reasonable range
  4. Response time < 500ms

Usage:
  python scripts/smoke_test.py --url http://localhost:8000
  python scripts/smoke_test.py --url http://serving:8000 --strict
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests

SAMPLE_REQUEST = {
    "request_id": "smoke-test-001",
    "model_name": "xgb_ranker",
    "instances": [
        {
            "user_id": 1,
            "recipe_id": 100,
            "minutes": 30,
            "n_ingredients": 8,
            "n_steps": 5,
            "avg_rating": 4.2,
            "n_reviews": 50,
            "cuisine": "Italian",
            "calories": 450.0,
            "total_fat": 20.0,
            "sugar": 5.0,
            "sodium": 800.0,
            "protein": 25.0,
            "saturated_fat": 7.0,
            "carbohydrate": 45.0,
            "total_fat_g": 20.0,
            "sugar_g": 5.0,
            "sodium_g": 0.8,
            "protein_g": 25.0,
            "saturated_fat_g": 7.0,
            "carbohydrate_g": 45.0,
            "has_egg": 0,
            "has_fish": 0,
            "has_milk": 1,
            "has_nuts": 0,
            "has_peanut": 0,
            "has_sesame": 0,
            "has_shellfish": 0,
            "has_soy": 0,
            "has_wheat": 1,
            "daily_calorie_target": 2000.0,
            "protein_target_g": 100.0,
            "carbs_target_g": 250.0,
            "fat_target_g": 65.0,
            "user_vegetarian": 0,
            "user_vegan": 0,
            "user_gluten_free": 0,
            "user_dairy_free": 0,
            "user_low_sodium": 0,
            "user_low_fat": 0,
            "history_pc1": 0.1,
            "history_pc2": -0.2,
            "history_pc3": 0.05,
            "history_pc4": 0.3,
            "history_pc5": -0.1,
            "history_pc6": 0.0,
        },
        {
            "user_id": 1,
            "recipe_id": 200,
            "minutes": 15,
            "n_ingredients": 5,
            "n_steps": 3,
            "avg_rating": 3.8,
            "n_reviews": 20,
            "cuisine": "Mexican",
            "calories": 350.0,
            "total_fat": 15.0,
            "sugar": 2.0,
            "sodium": 600.0,
            "protein": 18.0,
            "saturated_fat": 4.0,
            "carbohydrate": 38.0,
            "total_fat_g": 15.0,
            "sugar_g": 2.0,
            "sodium_g": 0.6,
            "protein_g": 18.0,
            "saturated_fat_g": 4.0,
            "carbohydrate_g": 38.0,
            "has_egg": 0,
            "has_fish": 0,
            "has_milk": 0,
            "has_nuts": 0,
            "has_peanut": 0,
            "has_sesame": 0,
            "has_shellfish": 0,
            "has_soy": 0,
            "has_wheat": 1,
            "daily_calorie_target": 2000.0,
            "protein_target_g": 100.0,
            "carbs_target_g": 250.0,
            "fat_target_g": 65.0,
            "user_vegetarian": 0,
            "user_vegan": 0,
            "user_gluten_free": 0,
            "user_dairy_free": 0,
            "user_low_sodium": 0,
            "user_low_fat": 0,
            "history_pc1": 0.1,
            "history_pc2": -0.2,
            "history_pc3": 0.05,
            "history_pc4": 0.3,
            "history_pc5": -0.1,
            "history_pc6": 0.0,
        },
    ],
}


def run_smoke_test(base_url: str, strict: bool = False, timeout: float = 10.0) -> bool:
    results = []
    base_url = base_url.rstrip("/")

    # Test 1: Health check
    try:
        t0 = time.time()
        r = requests.get(f"{base_url}/health", timeout=timeout)
        latency_ms = (time.time() - t0) * 1000
        health = r.json()
        ok = r.status_code == 200 and health.get("status") == "healthy"
        results.append({
            "test": "health_check",
            "passed": ok,
            "latency_ms": round(latency_ms, 1),
            "status_code": r.status_code,
            "response": health,
        })
    except Exception as exc:
        results.append({"test": "health_check", "passed": False, "error": str(exc)})

    # Test 2: Prediction
    try:
        t0 = time.time()
        r = requests.post(f"{base_url}/predict", json=SAMPLE_REQUEST, timeout=timeout)
        latency_ms = (time.time() - t0) * 1000
        ok = r.status_code == 200
        if ok:
            body = r.json()
            predictions = body.get("predictions", [])
            ok = (
                len(predictions) == 2
                and all("score" in p and "rank" in p for p in predictions)
                and body.get("model_version") is not None
            )
            # Check latency SLA
            latency_ok = latency_ms < 500
            if strict and not latency_ok:
                ok = False
            results.append({
                "test": "prediction",
                "passed": ok,
                "latency_ms": round(latency_ms, 1),
                "latency_sla_ok": latency_ok,
                "n_predictions": len(predictions),
                "model_version": body.get("model_version"),
                "model_source": body.get("model_source"),
                "scores": [p["score"] for p in predictions],
            })
        else:
            results.append({
                "test": "prediction",
                "passed": False,
                "status_code": r.status_code,
                "body": r.text[:500],
            })
    except Exception as exc:
        results.append({"test": "prediction", "passed": False, "error": str(exc)})

    # Test 3: Score range sanity
    pred_result = next((r for r in results if r["test"] == "prediction"), None)
    if pred_result and pred_result.get("scores"):
        scores = pred_result["scores"]
        score_ok = all(-1e6 < s < 1e6 for s in scores)  # XGBoost raw scores can be unbounded
        results.append({
            "test": "score_range",
            "passed": score_ok,
            "scores": scores,
            "note": "Raw XGBoost scores — no strict [0,1] constraint for rankers",
        })

    all_passed = all(r["passed"] for r in results)
    print(json.dumps({"passed": all_passed, "results": results}, indent=2))
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--strict", action="store_true", help="Fail on latency > 500ms")
    args = parser.parse_args()

    passed = run_smoke_test(args.url, strict=args.strict)
    sys.exit(0 if passed else 1)
