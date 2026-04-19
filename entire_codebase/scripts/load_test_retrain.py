"""
load_test_retrain.py — End-to-end load test that triggers retraining.

Sends diverse, deliberately shifted feature distributions to /predict so
drift_monitor detects real distributional shift and fires the retrain webhook.
After sending traffic, forces a drift check and polls MLflow for the new run.

Usage:
    python scripts/load_test_retrain.py --serving-url http://localhost:8000 \
                                        --retrain-url http://localhost:8080 \
                                        --mlflow-url http://localhost:5000 \
                                        --n-requests 300
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---------------------------------------------------------------------------
# Feature generation — matches drift_monitor.py NUMERIC_FEATURES
# Baseline = Food.com typical ranges. Shifted = intentionally drifted values
# to guarantee KS statistic >= 0.1 across >30% of features.
# ---------------------------------------------------------------------------
def make_payload(shift: bool = False) -> dict:
    rng = random.Random()

    def v(base, spread, shift_by=0):
        return round(base + shift_by + rng.gauss(0, spread), 3)

    s = 2.5 if shift else 0.0  # standard deviation shift

    return {
        "instances": [{
            "user_id": rng.randint(1, 5000),
            "recipe_id": rng.randint(1, 50000),
            "minutes":          v(35,  20,  s * 25),
            "n_ingredients":    v(9,   3,   s * 3),
            "n_steps":          v(8,   3,   s * 3),
            "avg_rating":       min(5.0, v(4.2, 0.5, -s * 0.3)),
            "n_reviews":        max(1, int(v(25, 20,  s * 30))),
            "cuisine":          "unknown",
            "calories":         max(10, v(350, 120, s * 180)),
            "total_fat":        max(0,  v(32,  12,  s * 20)),
            "sugar":            max(0,  v(18,  12,  s * 20)),
            "sodium":           max(0,  v(22,  10,  s * 20)),
            "protein":          max(0,  v(18,  10,  s * 20)),
            "saturated_fat":    max(0,  v(11,  6,   s * 12)),
            "carbohydrate":     max(0,  v(38,  18,  s * 30)),
            "total_fat_g":      max(0,  v(16,  6,   s * 10)),
            "sugar_g":          max(0,  v(9,   6,   s * 10)),
            "sodium_g":         max(0,  v(0.5, 0.3, s * 0.5)),
            "protein_g":        max(0,  v(9,   5,   s * 10)),
            "saturated_fat_g":  max(0,  v(2.2, 1.2, s * 2.5)),
            "carbohydrate_g":   max(0,  v(10,  5,   s * 10)),
            "has_egg":          rng.randint(0, 1),
            "has_fish":         rng.randint(0, 1),
            "has_milk":         rng.randint(0, 1),
            "has_nuts":         rng.randint(0, 1),
            "has_peanut":       rng.randint(0, 1),
            "has_sesame":       rng.randint(0, 1),
            "has_shellfish":    rng.randint(0, 1),
            "has_soy":          rng.randint(0, 1),
            "has_wheat":        rng.randint(0, 1),
            "daily_calorie_target": v(2000, 200, s * 400),
            "protein_target_g":    max(0, v(50,  10,  s * 30)),
            "carbs_target_g":      max(0, v(250, 30,  s * 80)),
            "fat_target_g":        max(0, v(65,  10,  s * 25)),
            "user_vegetarian":  rng.randint(0, 1),
            "user_vegan":       rng.randint(0, 1),
            "user_gluten_free": rng.randint(0, 1),
            "user_dairy_free":  rng.randint(0, 1),
            "user_low_sodium":  rng.randint(0, 1),
            "user_low_fat":     rng.randint(0, 1),
            "history_pc1":      v(0, 1, s * 1.5),
            "history_pc2":      v(0, 1, s * 1.5),
            "history_pc3":      v(0, 1, s * 1.5),
            "history_pc4":      v(0, 1, 0),
            "history_pc5":      v(0, 1, 0),
            "history_pc6":      v(0, 1, 0),
        }]
    }


def send_request(serving_url: str, shift: bool) -> bool:
    try:
        r = requests.post(
            f"{serving_url}/predict",
            json=make_payload(shift=shift),
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


def send_traffic(serving_url: str, n: int, shift: bool, workers: int = 8):
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(send_request, serving_url, shift) for _ in range(n)]
        for f in as_completed(futures):
            if f.result():
                ok += 1
    return ok


def poll_retrain_status(retrain_url: str, timeout: int = 600) -> dict:
    deadline = time.time() + timeout
    print("\nPolling retrain-api status...")
    prev_status = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{retrain_url}/status", timeout=5)
            s = r.json()
            status = s.get("status", "unknown")
            if status != prev_status:
                print(f"  [{time.strftime('%H:%M:%S')}] status → {status}")
                prev_status = status
            if status in ("completed", "failed"):
                return s
        except Exception as e:
            print(f"  retrain-api unreachable: {e}")
        time.sleep(10)
    return {}


def get_latest_mlflow_run(mlflow_url: str, experiment_name: str) -> dict | None:
    try:
        r = requests.get(
            f"{mlflow_url}/api/2.0/mlflow/experiments/get-by-name",
            params={"experiment_name": experiment_name},
            timeout=5,
        )
        exp_id = r.json()["experiment"]["experiment_id"]
        r2 = requests.post(
            f"{mlflow_url}/api/2.0/mlflow/runs/search",
            json={"experiment_ids": [exp_id], "max_results": 1,
                  "order_by": ["start_time DESC"]},
            timeout=5,
        )
        runs = r2.json().get("runs", [])
        return runs[0] if runs else None
    except Exception as e:
        print(f"  MLflow query failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-url",  default="http://localhost:8000")
    parser.add_argument("--retrain-url",  default="http://localhost:8080")
    parser.add_argument("--mlflow-url",   default="http://localhost:5000")
    parser.add_argument("--n-requests",   type=int, default=300)
    parser.add_argument("--workers",      type=int, default=8)
    parser.add_argument("--experiment",   default="autogym-recipe-ranking-ray")
    parser.add_argument("--skip-drift-trigger", action="store_true",
                        help="Skip forcing a drift check (wait for scheduled 5-min check)")
    args = parser.parse_args()

    print("=" * 60)
    print("SparkyFitness — Load Test → Retrain Trigger")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{args.serving_url}/health", timeout=5)
        assert r.status_code == 200
        print(f"✓ Serving healthy: {r.json()}")
    except Exception as e:
        print(f"✗ Serving unreachable at {args.serving_url}: {e}")
        sys.exit(1)

    try:
        r = requests.get(f"{args.retrain_url}/health", timeout=5)
        assert r.status_code == 200
        print(f"✓ Retrain-api healthy: {r.json()}")
    except Exception as e:
        print(f"✗ Retrain-api unreachable at {args.retrain_url}: {e}")
        sys.exit(1)

    # Record baseline MLflow run before test
    print(f"\nChecking MLflow baseline run...")
    pre_run = get_latest_mlflow_run(args.mlflow_url, args.experiment)
    pre_run_id = pre_run["info"]["run_id"] if pre_run else None
    print(f"  Latest run before test: {pre_run_id or 'none'}")

    # Phase 1: warm-up with normal traffic (10 requests)
    print(f"\n[Phase 1] Sending 10 warm-up requests (normal distribution)...")
    ok = send_traffic(args.serving_url, 10, shift=False, workers=2)
    print(f"  {ok}/10 succeeded")
    time.sleep(1)

    # Phase 2: shifted traffic to create drift
    print(f"\n[Phase 2] Sending {args.n_requests} requests with SHIFTED distribution...")
    print(f"  (calories +450, protein +50g, history PCs shifted +3.75σ)")
    t0 = time.time()
    ok = send_traffic(args.serving_url, args.n_requests, shift=True, workers=args.workers)
    elapsed = time.time() - t0
    print(f"  {ok}/{args.n_requests} succeeded in {elapsed:.1f}s "
          f"({ok/elapsed:.1f} req/s)")

    if ok < 100:
        print(f"✗ Only {ok} requests succeeded — need ≥100 for drift check. Aborting.")
        sys.exit(1)

    # Force immediate drift check
    if not args.skip_drift_trigger:
        print(f"\n[Phase 3] Forcing immediate drift check...")
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "exec", "sparky-drift-monitor",
                 "python", "src/data/drift_monitor.py", "--once"],
                capture_output=True, text=True, timeout=60,
            )
            output = result.stdout + result.stderr
            if "DRIFT DETECTED" in output or "drift_detected" in output:
                print("  ✓ DRIFT DETECTED — retraining webhook fired")
            elif "overall_drift_detected" in output:
                print("  ✓ Drift check ran — check output for results")
            else:
                print("  Drift check output (last 10 lines):")
                for line in output.strip().split("\n")[-10:]:
                    print(f"    {line}")
        except Exception as e:
            print(f"  Could not exec drift monitor: {e}")
            print("  Triggering retrain directly via webhook...")
            try:
                r = requests.post(
                    f"{args.retrain_url}/trigger",
                    json={"reason": "load_test_drift", "auto_promote": True},
                    timeout=10,
                )
                print(f"  Webhook response: {r.json()}")
            except Exception as e2:
                print(f"  Webhook also failed: {e2}")

    # Poll retrain status
    status = poll_retrain_status(args.retrain_url, timeout=700)

    # Check MLflow for new run
    print(f"\nChecking MLflow for new training run...")
    post_run = get_latest_mlflow_run(args.mlflow_url, args.experiment)
    post_run_id = post_run["info"]["run_id"] if post_run else None

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Requests sent:       {args.n_requests} (shifted distribution)")
    print(f"  Retrain status:      {status.get('status', 'unknown')}")
    if status.get("result"):
        res = status["result"]
        print(f"  Training success:    {res.get('success')}")
        print(f"  NDCG@10:             {res.get('metrics', {}).get('ndcg_at_10', 'N/A')}")
        print(f"  Model registered:    {res.get('registered')}")
        print(f"  Auto-promoted:       {res.get('promoted')}")
        if res.get("baseline_metrics"):
            print(f"  Baseline NDCG@10:    {res['baseline_metrics'].get('ndcg_at_10', 'N/A')}")

    if post_run_id and post_run_id != pre_run_id:
        print(f"\n  ✓ NEW MLflow run created: {post_run_id}")
        run_name = post_run["data"]["tags"].get("mlflow.runName", "")
        print(f"    Run name: {run_name}")
    else:
        print(f"\n  ⚠ No new MLflow run detected yet (may still be in progress)")

    print(f"\n  → MLflow UI: {args.mlflow_url}")
    print(f"    Experiment: {args.experiment}")
    print(f"  → Retrain API status: {args.retrain_url}/status")
    print("=" * 60)


if __name__ == "__main__":
    main()
