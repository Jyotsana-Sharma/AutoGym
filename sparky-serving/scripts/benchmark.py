"""
Load testing / benchmarking script for the SparkyFitness meal recommendation API.

Sends concurrent requests to the /predict endpoint and reports latency and
throughput statistics at multiple concurrency levels.

Usage:
    python benchmark.py --url http://localhost:8000 --num-requests 500
    python benchmark.py --url http://localhost:8000 --num-requests 500 --max-workers 1 4 8 16
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import requests

# Default path to sample input
DEFAULT_SAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "samples", "model_input.json"
)

WARMUP_REQUESTS = 10


def send_request(url: str, payload: dict, timeout: float = 30.0) -> dict:
    """Send a single prediction request and return timing info."""
    start = time.perf_counter()
    try:
        response = requests.post(
            f"{url}/predict",
            json=payload,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        return {
            "latency": elapsed,
            "status_code": response.status_code,
            "success": response.status_code == 200,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "latency": elapsed,
            "status_code": -1,
            "success": False,
            "error": str(e),
        }


def run_warmup(url: str, payload: dict):
    """Send warm-up requests to prime the server."""
    print(f"Warming up with {WARMUP_REQUESTS} requests...")
    for _ in range(WARMUP_REQUESTS):
        send_request(url, payload)
    print("Warm-up complete.\n")


def run_benchmark(
    url: str, payload: dict, num_requests: int, max_workers: int
) -> dict:
    """Run a benchmark at a given concurrency level and return statistics."""
    latencies = []
    errors = 0

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(send_request, url, payload)
            for _ in range(num_requests)
        ]
        for future in as_completed(futures):
            result = future.result()
            latencies.append(result["latency"])
            if not result["success"]:
                errors += 1

    total_time = time.perf_counter() - start_time
    latencies_ms = np.array(latencies) * 1000  # Convert to milliseconds

    stats = {
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
    return stats


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
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the serving API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=500,
        help="Number of requests per concurrency level (default: 500)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Concurrency levels to test (default: 1 4 8 16)",
    )
    parser.add_argument(
        "--sample-input",
        type=str,
        default=DEFAULT_SAMPLE_PATH,
        help="Path to sample input JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (default: ../results/benchmark_<timestamp>.json)",
    )
    args = parser.parse_args()

    # Load sample input
    sample_path = os.path.abspath(args.sample_input)
    if not os.path.exists(sample_path):
        print(f"ERROR: Sample input file not found: {sample_path}")
        sys.exit(1)

    with open(sample_path) as f:
        payload = json.load(f)

    print(f"Target URL:      {args.url}")
    print(f"Requests/level:  {args.num_requests}")
    print(f"Concurrency:     {args.max_workers}")
    print(f"Sample input:    {sample_path}")
    print(f"Candidates:      {len(payload.get('candidate_recipes', []))}")
    print()

    # Health check
    try:
        resp = requests.get(f"{args.url}/health", timeout=5)
        if resp.status_code == 200:
            print(f"Health check:    OK ({resp.json()})")
        else:
            print(f"WARNING: Health check returned status {resp.status_code}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.url} - {e}")
        sys.exit(1)

    print()

    # Warm-up
    run_warmup(args.url, payload)

    # Run benchmarks at each concurrency level
    all_stats = []
    for workers in args.max_workers:
        print(f"Running benchmark: concurrency={workers}, requests={args.num_requests}...")
        stats = run_benchmark(args.url, payload, args.num_requests, workers)
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
