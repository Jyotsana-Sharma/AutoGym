from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

from .metric import compute_ranking_metric
from .ray_training import run_xgb_ranker_ray
from .ranking_data import PreparedFrames, load_training_frames
from .mlflow_utils import log_run_metadata, read_config


def group_sizes(frame: pd.DataFrame, group_key: str) -> list[int]:
    return frame.groupby(group_key, sort=False).size().tolist()


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


def run_xgb_ranker(config: dict[str, Any], prepared: PreparedFrames) -> dict[str, float]:
    training_config = config["training"]
    group_key = training_config["group_key"]
    label_column = training_config["label_column"]
    feature_columns = prepared.feature_columns

    train_frame = prepared.train
    validation_frame = prepared.validation
    scored = prepared.test.copy()

    model = xgb.XGBRanker(
        objective=training_config["objective"],
        learning_rate=training_config["learning_rate"],
        n_estimators=training_config["n_estimators"],
        max_depth=training_config["max_depth"],
        min_child_weight=training_config["min_child_weight"],
        subsample=training_config["subsample"],
        colsample_bytree=training_config["colsample_bytree"],
        reg_lambda=training_config["reg_lambda"],
        eval_metric=training_config["eval_metric"],
        random_state=config["random_seed"],
        tree_method=training_config["tree_method"],
    )

    model.fit(
        train_frame[feature_columns],
        train_frame[label_column],
        group=group_sizes(train_frame, group_key),
        eval_set=[(validation_frame[feature_columns], validation_frame[label_column])],
        eval_group=[group_sizes(validation_frame, group_key)],
    )

    scored["score"] = model.predict(scored[feature_columns])
    mlflow.xgboost.log_model(
        model,
        name="model",
        input_example=train_frame[feature_columns].head(5),
        model_format="json",
    )

    metrics = compute_ranking_metric(
        scored,
        group_key=group_key,
        label_column=label_column,
        score_column="score"
    )
    return metrics


def run_training(config_path: Path):
    config = read_config(config_path)
    run_name = config["run_name"]
    start_time = time.perf_counter()
    candidate = config["candidate_name"]
    
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_artifact(str(config_path))
        if "ray" in config:
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
        elif "ray" in config:
            metrics = run_xgb_ranker_ray(config, prepared)
        else:
            metrics = run_xgb_ranker(config, prepared)

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
