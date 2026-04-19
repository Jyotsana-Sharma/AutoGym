"""
Load testing / benchmarking script for the SparkyFitness meal recommendation API.

Supports both FastAPI (/predict) and Triton (/v2/models/.../infer) backends.

Usage:
    # FastAPI
    python benchmark.py --url http://localhost:8000 --num-requests 500

    # Triton
    python benchmark.py --url http://localhost:8000 --triton --model-name meal_ranker

    # Triton with dynamic batching
    python benchmark.py --url http://localhost:8000 --triton --model-name meal_ranker_batching
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# Default path to sample input
DEFAULT_SAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "samples", "model_input.json"
)

WARMUP_REQUESTS = 10

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


def encode_instances(instances: list[dict]) -> np.ndarray:
    """Convert contract instances to a float32 feature matrix (same as app code)."""
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


def build_triton_payload(feature_matrix: np.ndarray) -> dict:
    """Build Triton v2 inference request from a feature matrix."""
    n_rows, n_cols = feature_matrix.shape
    return {
        "inputs": [{
            "name": "input__0",
            "shape": [n_rows, n_cols],
            "datatype": "FP32",
            "data": feature_matrix.flatten().tolist(),
        }]
    }


def send_request_fastapi(url: str, payload: dict, timeout: float = 30.0) -> dict:
    """Send a single FastAPI prediction request."""
    start = time.perf_counter()
    try:
        response = requests.post(f"{url}/predict", json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start
        return {"latency": elapsed, "status_code": response.status_code, "success": response.status_code == 200}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"latency": elapsed, "status_code": -1, "success": False, "error": str(e)}


def send_request_triton(url: str, payload: dict, model_name: str, timeout: float = 30.0) -> dict:
    """Send a single Triton v2 inference request."""
    start = time.perf_counter()
    try:
        response = requests.post(
            f"{url}/v2/models/{model_name}/infer", json=payload, timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        return {"latency": elapsed, "status_code": response.status_code, "success": response.status_code == 200}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"latency": elapsed, "status_code": -1, "success": False, "error": str(e)}


def run_warmup(send_fn, warmup_args: tuple):
    """Send warm-up requests to prime the server."""
    print(f"Warming up with {WARMUP_REQUESTS} requests...")
    for _ in range(WARMUP_REQUESTS):
        send_fn(*warmup_args)
    print("Warm-up complete.\n")


def run_benchmark(send_fn, send_args: tuple, num_requests: int, max_workers: int) -> dict:
    """Run a benchmark at a given concurrency level and return statistics."""
    latencies = []
    errors = 0

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(send_fn, *send_args) for _ in range(num_requests)]
        for future in as_completed(futures):
            result = future.result()
            latencies.append(result["latency"])
            if not result["success"]:
                errors += 1

    total_time = time.perf_counter() - start_time
    latencies_ms = np.array(latencies) * 1000

    return {
        "concurrency": max_workers,
        "num_requests": num_requests,
        "total_time_s": round(total_time, 3),
        "throughput_rps": round(num_requests / total_time, 2),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 2),
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "min_ms": round(float(np.min(latencies_ms)), 2),
        "max_ms": round(float(np.max(latencies_ms)), 2),
        "error_count": errors,
        "error_rate": round(errors / num_requests * 100, 2),
    }


def print_table(all_stats: list[dict]):
    """Print results as a formatted console table."""
    header = (
        f"{'Concurrency':>12} | {'Requests':>8} | {'Throughput':>12} | "
        f"{'p50 (ms)':>10} | {'p95 (ms)':>10} | {'p99 (ms)':>10} | "
        f"{'Mean (ms)':>10} | {'Errors':>8} | {'Total (s)':>10}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("BENCHMARK RESULTS")
    print(separator)
    print(header)
    print(separator)

    for s in all_stats:
        print(
            f"{s['concurrency']:>12} | {s['num_requests']:>8} | "
            f"{s['throughput_rps']:>10.2f}/s | "
            f"{s['p50_ms']:>10.2f} | {s['p95_ms']:>10.2f} | "
            f"{s['p99_ms']:>10.2f} | {s['mean_ms']:>10.2f} | "
            f"{s['error_count']:>8} | {s['total_time_s']:>10.3f}"
        )

    print(separator + "\n")


def save_results(all_stats: list[dict], output_path: str, args: argparse.Namespace):
    """Save results to a JSON file."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "target_url": args.url,
        "mode": "triton" if args.triton else "fastapi",
        "model_name": args.model_name if args.triton else None,
        "num_requests_per_level": args.num_requests,
        "concurrency_levels": args.max_workers,
        "warmup_requests": WARMUP_REQUESTS,
        "results": all_stats,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the SparkyFitness meal recommendation API"
    )
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                        help="Base URL of the serving API")
    parser.add_argument("--num-requests", type=int, default=500,
                        help="Number of requests per concurrency level")
    parser.add_argument("--max-workers", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Concurrency levels to test")
    parser.add_argument("--sample-input", type=str, default=DEFAULT_SAMPLE_PATH,
                        help="Path to sample input JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    parser.add_argument("--triton", action="store_true",
                        help="Use Triton v2 HTTP protocol instead of FastAPI")
    parser.add_argument("--model-name", type=str, default="meal_ranker",
                        help="Triton model name (default: meal_ranker)")
    args = parser.parse_args()

    # Load sample input
    sample_path = os.path.abspath(args.sample_input)
    if not os.path.exists(sample_path):
        print(f"ERROR: Sample input file not found: {sample_path}")
        sys.exit(1)

    with open(sample_path) as f:
        contract_payload = json.load(f)

    instances = contract_payload.get("instances", [])

    print(f"Target URL:      {args.url}")
    print(f"Mode:            {'Triton (' + args.model_name + ')' if args.triton else 'FastAPI'}")
    print(f"Requests/level:  {args.num_requests}")
    print(f"Concurrency:     {args.max_workers}")
    print(f"Sample input:    {sample_path}")
    print(f"Instances:       {len(instances)}")
    print()

    # Health check
    health_url = f"{args.url}/v2/health/ready" if args.triton else f"{args.url}/health"
    try:
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
            print(f"Health check:    OK")
        else:
            print(f"WARNING: Health check returned status {resp.status_code}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.url} - {e}")
        sys.exit(1)

    print()

    # Build payload and select send function
    if args.triton:
        feature_matrix = encode_instances(instances)
        payload = build_triton_payload(feature_matrix)
        send_fn = send_request_triton
        send_args = (args.url, payload, args.model_name)
    else:
        payload = contract_payload
        send_fn = send_request_fastapi
        send_args = (args.url, payload)

    # Warm-up
    run_warmup(send_fn, send_args)

    # Run benchmarks at each concurrency level
    all_stats = []
    for workers in args.max_workers:
        print(f"Running benchmark: concurrency={workers}, requests={args.num_requests}...")
        stats = run_benchmark(send_fn, send_args, args.num_requests, workers)
        all_stats.append(stats)
        print(
            f"  -> throughput={stats['throughput_rps']}/s, "
            f"p50={stats['p50_ms']}ms, p99={stats['p99_ms']}ms, "
            f"errors={stats['error_count']}"
        )

    # Display results
    print_table(all_stats)

    # Save results to JSON
    if args.output:
        output_path = args.output
    else:
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"benchmark_{timestamp}.json")

    save_results(all_stats, output_path, args)


if __name__ == "__main__":
    main()
