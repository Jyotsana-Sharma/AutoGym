from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import pandas as pd

from .metric import compute_ranking_metric
from .ray_training import run_xgb_ranker_ray
from .ranking_data import PreparedFrames, load_training_frames
from .mlflow_utils import log_run_metadata, read_config


def run_baseline_popularity(config: dict[str, Any], prepared: PreparedFrames) -> dict[str, float]:
    popularity = (
        prepared.train.groupby("recipe_id", as_index=False)["label"]
        .sum()
        .rename(columns={"label": "positive_label_count"})
        .sort_values("positive_label_count", ascending=False)
        .reset_index(drop=True)
    )
    popularity["score"] = popularity["positive_label_count"] / popularity["positive_label_count"].max()
    scored = prepared.test.copy()
    popularity_scores = popularity.set_index("recipe_id")["score"]
    scored["score"] = scored["recipe_id"].map(popularity_scores).fillna(0.0)

    metrics = compute_ranking_metric(
        scored,
        group_key=config["training"]["group_key"],
        label_column=config["training"]["label_column"],
        score_column="score"
    )
    metrics["unique_recipes_ranked"] = float(popularity["recipe_id"].nunique())
    return metrics


def run_training(config_path: Path):
    config = read_config(config_path)
    run_name = config["run_name"]
    start_time = time.perf_counter()
    candidate = config["candidate_name"]

    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact(str(config_path))
        mlflow.set_tag("trainer_backend", "ray_train")
        log_run_metadata(config)
        prepared = load_training_frames(config)
        mlflow.log_dict(
            {
                "feature_count": len(prepared.feature_columns),
                "feature_columns": prepared.feature_columns,
            },
            "dataset_metadata.json",
        )

        if candidate == "baseline_popularity":
            metrics = run_baseline_popularity(config, prepared)
        else:
            metrics = run_xgb_ranker_ray(config, prepared)

        elapsed = time.perf_counter() - start_time
        metrics["wall_time_seconds"] = round(elapsed, 4)
        mlflow.log_metrics(metrics)

        summary = {
            "run_name": run_name,
            "candidate_name": candidate,
            "metrics": metrics,
        }
        mlflow.log_dict(summary, "run_summary.json")
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    summary = run_training(Path(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
