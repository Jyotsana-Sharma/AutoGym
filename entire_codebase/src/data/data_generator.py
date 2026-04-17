"""Production traffic generator for the SparkyFitness ML system.

This script replays held-out interaction rows as synthetic production traffic by:
  1. building realistic multi-candidate /predict requests from feature rows
  2. calling the live serving API
  3. sending the observed outcome back through /feedback
  4. optionally writing a local audit CSV for demos

It is intentionally lightweight so teams can run it continuously during demos or
production emulation without human intervention.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


NON_FEATURE_COLUMNS = {
    "label",
    "date",
    "name",
    "submitted",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay production-like ML traffic")
    parser.add_argument(
        "--interactions-path",
        default=os.environ.get("PRODUCTION_TRAFFIC_CSV", "./output/test.csv"),
        help="CSV with held-out interactions and full feature rows",
    )
    parser.add_argument(
        "--features-csv",
        default=os.environ.get("PRODUCTION_FEATURES_CSV", ""),
        help="Optional CSV to sample additional candidate rows from",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("ML_RECOMMENDATION_URL", "http://localhost:8000"),
        help="Serving API base URL",
    )
    parser.add_argument("--days", type=int, default=7, help="Replay the last N days of data")
    parser.add_argument("--rate", type=float, default=1.0, help="Requests per second")
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=12,
        help="Number of candidate recipes per /predict request",
    )
    parser.add_argument(
        "--log-file",
        default="./output/generated_interactions.csv",
        help="Local audit CSV written during replay",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not call the serving API; only write the local audit log",
    )
    return parser.parse_args()


def simulate_action(rating: float) -> dict[str, Any]:
    if rating >= 4:
        return {"action": "cook", "rating": random.choice([4, 5])}
    if rating >= 3:
        return {"action": random.choice(["cook", "view", "view"]), "rating": random.choice([3, 4])}
    if rating >= 2:
        return {"action": random.choice(["view", "skip", "skip"]), "rating": random.choice([2, 3])}
    return {"action": "skip", "rating": random.choice([1, 2])}


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_replay_rows(path: str, days: int) -> pd.DataFrame:
    df = normalize_rows(pd.read_csv(path))
    if "user_id" not in df.columns or "recipe_id" not in df.columns:
        raise ValueError("Replay CSV must contain user_id and recipe_id columns")

    if "date" in df.columns and df["date"].notna().any():
        cutoff = df["date"].max() - pd.Timedelta(days=days)
        df = df[df["date"] >= cutoff].copy()
        df["day"] = df["date"].dt.date.astype(str)
    else:
        df["day"] = "undated"

    if df.empty:
        raise ValueError("No replay rows available after applying the date filter")
    return df


def load_candidate_pool(replay_df: pd.DataFrame, features_csv: str) -> pd.DataFrame:
    if features_csv and Path(features_csv).exists():
        pool = normalize_rows(pd.read_csv(features_csv))
        missing = {"user_id", "recipe_id"} - set(pool.columns)
        if missing:
            raise ValueError(f"Candidate CSV missing columns: {sorted(missing)}")
        return pool
    return replay_df.copy()


def row_to_instance(row: pd.Series) -> dict[str, Any]:
    instance = {}
    for key, value in row.items():
        if key in NON_FEATURE_COLUMNS:
            continue
        if pd.isna(value):
            instance[key] = 0.0
        elif key in {"user_id", "recipe_id"}:
            instance[key] = int(value)
        elif hasattr(value, "item"):
            instance[key] = value.item()
        elif isinstance(value, pd.Timestamp):
            instance[key] = value.isoformat()
        else:
            instance[key] = value
    return instance


def build_instances(
    target_row: pd.Series,
    user_pool: pd.DataFrame,
    global_pool: pd.DataFrame,
    candidate_pool_size: int,
) -> list[dict[str, Any]]:
    if len(user_pool) >= candidate_pool_size:
        sampled = user_pool.sample(n=candidate_pool_size, replace=False, random_state=None)
    else:
        sampled = user_pool.copy()
        needed = candidate_pool_size - len(sampled)
        if needed > 0:
            extra = global_pool[global_pool["recipe_id"] != target_row["recipe_id"]]
            if not extra.empty:
                sampled_extra = extra.sample(n=min(needed, len(extra)), replace=False, random_state=None)
                sampled = pd.concat([sampled, sampled_extra], ignore_index=True)

    if int(target_row["recipe_id"]) not in set(sampled["recipe_id"].astype(int).tolist()):
        sampled = pd.concat([sampled, target_row.to_frame().T], ignore_index=True)

    sampled = sampled.drop_duplicates(subset=["recipe_id"], keep="last").head(candidate_pool_size)
    return [row_to_instance(row) for _, row in sampled.iterrows()]


def call_predict(api_url: str, request_id: str, instances: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not HAS_REQUESTS:
        return None
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={
            "request_id": request_id,
            "model_name": "sparky-ranker",
            "instances": instances,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def call_feedback(
    api_url: str,
    request_id: str,
    user_id: int,
    recipe_id: int,
    action: str,
    rating: float,
) -> bool:
    if not HAS_REQUESTS:
        return False
    response = requests.post(
        f"{api_url.rstrip('/')}/feedback",
        json={
            "request_id": request_id,
            "user_id": user_id,
            "recipe_id": recipe_id,
            "action": action,
            "rating": rating,
        },
        timeout=10,
    )
    response.raise_for_status()
    return True


def ensure_log_file(path: str) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with log_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "request_id",
                    "user_id",
                    "recipe_id",
                    "action",
                    "rating",
                    "top_ranked_recipe_id",
                    "top_ranked_score",
                    "status",
                    "timestamp",
                ]
            )


def append_log(
    path: str,
    request_id: str,
    user_id: int,
    recipe_id: int,
    action: str,
    rating: float,
    top_recipe_id: int | None,
    top_score: float | None,
    status: str,
) -> None:
    with Path(path).open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                request_id,
                user_id,
                recipe_id,
                action,
                rating,
                top_recipe_id,
                top_score,
                status,
                datetime.now(timezone.utc).isoformat(),
            ]
        )


def main() -> None:
    args = parse_args()
    replay_df = load_replay_rows(args.interactions_path, args.days)
    candidate_pool = load_candidate_pool(replay_df, args.features_csv)
    ensure_log_file(args.log_file)

    replay_df = replay_df.sort_values("date") if "date" in replay_df.columns else replay_df
    user_groups = {uid: df for uid, df in candidate_pool.groupby("user_id", sort=False)}
    all_days = replay_df["day"].tolist()
    unique_days = list(dict.fromkeys(all_days))

    print("=" * 60)
    print("SparkyFitness — Production Traffic Generator")
    print("=" * 60)
    print(f"Replay rows:      {len(replay_df):,}")
    print(f"Candidate rows:   {len(candidate_pool):,}")
    print(f"API URL:          {args.api_url}")
    print(f"Offline mode:     {args.offline}")
    print(f"Audit log:        {args.log_file}")

    metrics: defaultdict[str, Any] = defaultdict(int)
    metrics["latency_ms"] = 0.0
    delay = 1 / args.rate if args.rate > 0 else 0

    for day in unique_days:
        day_rows = replay_df[replay_df["day"] == day]
        print(f"\nDay {day}: {len(day_rows):,} interactions")
        for _, row in day_rows.iterrows():
            user_id = int(row["user_id"])
            recipe_id = int(row["recipe_id"])
            rating = float(row.get("rating", 0) or 0)
            request_id = f"prod-{user_id}-{recipe_id}-{int(time.time() * 1000)}"
            action_payload = simulate_action(rating)
            instances = build_instances(
                target_row=row,
                user_pool=user_groups.get(user_id, candidate_pool[candidate_pool["user_id"] == user_id]),
                global_pool=candidate_pool,
                candidate_pool_size=args.candidate_pool_size,
            )

            top_recipe_id = None
            top_score = None
            status = "offline"
            metrics["requests"] += 1

            if not args.offline and HAS_REQUESTS:
                try:
                    started = time.perf_counter()
                    predict_response = call_predict(args.api_url, request_id, instances)
                    metrics["latency_ms"] += (time.perf_counter() - started) * 1000
                    predictions = predict_response.get("predictions", []) if predict_response else []
                    if predictions:
                        top_recipe_id = int(predictions[0]["recipe_id"])
                        top_score = float(predictions[0]["score"])
                    call_feedback(
                        args.api_url,
                        request_id=request_id,
                        user_id=user_id,
                        recipe_id=recipe_id,
                        action=action_payload["action"],
                        rating=float(action_payload["rating"]),
                    )
                    metrics["ok"] += 1
                    metrics[f"action_{action_payload['action']}"] += 1
                    status = "served"
                except Exception as exc:
                    metrics["fail"] += 1
                    status = f"error:{type(exc).__name__}"
            else:
                metrics["ok"] += 1
                metrics[f"action_{action_payload['action']}"] += 1

            append_log(
                args.log_file,
                request_id=request_id,
                user_id=user_id,
                recipe_id=recipe_id,
                action=action_payload["action"],
                rating=float(action_payload["rating"]),
                top_recipe_id=top_recipe_id,
                top_score=top_score,
                status=status,
            )

            if delay:
                time.sleep(delay)

        print(
            f"  cumulative requests={metrics['requests']:,} ok={metrics['ok']:,} "
            f"fail={metrics['fail']:,}"
        )

    avg_latency = metrics["latency_ms"] / metrics["ok"] if metrics["ok"] else 0.0
    print("\n" + "=" * 60)
    print("DONE: Production traffic replay complete")
    print(f"Requests:         {metrics['requests']:,}")
    print(f"Successful:       {metrics['ok']:,}")
    print(f"Failed:           {metrics['fail']:,}")
    print(f"Average latency:  {avg_latency:.1f} ms")
    print(f"Actions:          cook={metrics['action_cook']} view={metrics['action_view']} skip={metrics['action_skip']}")
    print(f"Audit log:        {args.log_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
