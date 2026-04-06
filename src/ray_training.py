from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

from .metric import compute_ranking_metric
from .ranking_data import PreparedFrames

os.environ.setdefault("RAY_TRAIN_V2_ENABLED", "0")


class FailWorkerOnce(xgb.callback.TrainingCallback):
    def __init__(self, enabled: bool, failure_epoch: int, marker_path: str) -> None:
        self.enabled = enabled
        self.failure_epoch = failure_epoch
        self.marker_path = Path(marker_path)

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: dict) -> bool:
        if not self.enabled or self.marker_path.exists() or epoch + 1 != self.failure_epoch:
            return False
        self.marker_path.write_text("failed\n", encoding="utf-8")
        raise RuntimeError("Injected worker failure after checkpoint")


def write_shards(
    frame: pd.DataFrame,
    group_key: str,
    split_name: str,
    shard_dir: Path,
    num_workers: int,
) -> list[str]:
    frame = frame.sort_values([group_key, "recipe_id"]).reset_index(drop=True)
    paths: list[str] = []
    for i in range(num_workers):
        worker_frame = frame[frame[group_key] % num_workers == i].reset_index(drop=True)
        split_path = shard_dir / f"{split_name}_{i}.parquet"
        worker_frame.to_parquet(split_path, index=False)
        paths.append(str(split_path))
    return paths


def worker_loop(loop_config: dict[str, Any]) -> None:
    from ray import train
    from ray.train.xgboost import RayTrainReportCallback

    worker_rank = train.get_context().get_world_rank()
    training = loop_config["training"]
    ray_config = loop_config["ray"]

    train_frame = pd.read_parquet(loop_config["train_shards"][worker_rank])
    validation_frame = pd.read_parquet(loop_config["validation_shards"][worker_rank])
    train_matrix = xgb.DMatrix(
        train_frame[loop_config["feature_columns"]],
        label=train_frame[training["label_column"]],
        qid=train_frame[training["group_key"]],
        feature_names=loop_config["feature_columns"],
    )
    validation_matrix = xgb.DMatrix(
        validation_frame[loop_config["feature_columns"]],
        label=validation_frame[training["label_column"]],
        qid=validation_frame[training["group_key"]],
        feature_names=loop_config["feature_columns"],
    )

    checkpoint = train.get_checkpoint()
    starting_model = None if checkpoint is None else RayTrainReportCallback.get_model(checkpoint)
    starting_round = 0 if starting_model is None else starting_model.num_boosted_rounds()
    remaining_rounds = training["n_estimators"] - starting_round
    if remaining_rounds <= 0:
        train.report({"validation_ndcg": 0.0}, checkpoint=checkpoint)
        return

    xgb.train(
        params={
            "objective": training["objective"],
            "learning_rate": training["learning_rate"],
            "max_depth": training["max_depth"],
            "min_child_weight": training["min_child_weight"],
            "subsample": training["subsample"],
            "colsample_bytree": training["colsample_bytree"],
            "reg_lambda": training["reg_lambda"],
            "eval_metric": training["eval_metric"],
            "tree_method": training["tree_method"],
            "seed": loop_config["random_seed"],
            "nthread": ray_config["cpus_per_worker"],
        },
        dtrain=train_matrix,
        num_boost_round=remaining_rounds,
        evals=[(validation_matrix, "validation")],
        xgb_model=starting_model,
        callbacks=[
            RayTrainReportCallback(
                frequency=ray_config["checkpoint_frequency"],
                checkpoint_at_end=True,
            ),
            FailWorkerOnce(
                enabled=ray_config["simulate_worker_failure"] and worker_rank == 0,
                failure_epoch=ray_config["failure_epoch"],
                marker_path=loop_config["failure_marker_path"],
            ),
        ],
        verbose_eval=False,
    )


def run_xgb_ranker_ray(config: dict[str, Any], prepared: PreparedFrames) -> dict[str, float]:
    import ray
    from ray.air.config import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
    from ray.train.data_parallel_trainer import DataParallelTrainer
    from ray.train.xgboost import RayTrainReportCallback

    ray_config = config["ray"]
    training = config["training"]
    storage_path = Path(ray_config["storage_path"]).expanduser().resolve()
    storage_path.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(tempfile.mkdtemp(prefix="autogym-ray-shards-"))
    failure_marker_path = str(shard_dir / "worker_0_failed_once")
    train_shards = write_shards(
        prepared.train,
        training["group_key"],
        "train",
        shard_dir,
        ray_config["num_workers"],
    )
    validation_shards = write_shards(
        prepared.validation,
        training["group_key"],
        "validation",
        shard_dir,
        ray_config["num_workers"],
    )

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    trainer = DataParallelTrainer(
        train_loop_per_worker=worker_loop,
        train_loop_config={
            "feature_columns": prepared.feature_columns,
            "failure_marker_path": failure_marker_path,
            "random_seed": config["random_seed"],
            "ray": ray_config,
            "train_shards": train_shards,
            "training": training,
            "validation_shards": validation_shards,
        },
        scaling_config=ScalingConfig(
            num_workers=ray_config["num_workers"],
            use_gpu=False,
            resources_per_worker={"CPU": ray_config["cpus_per_worker"]},
        ),
        run_config=RunConfig(
            name=config.get("run_name", "xgb_ranker"),
            storage_path=str(storage_path),
            failure_config=FailureConfig(max_failures=ray_config["max_failures"]),
            checkpoint_config=CheckpointConfig(num_to_keep=1),
        ),
    )

    try:
        result = trainer.fit()
        booster = RayTrainReportCallback.get_model(result.checkpoint)
        scored = prepared.test.copy()
        scored["score"] = booster.predict(
            xgb.DMatrix(scored[prepared.feature_columns], feature_names=prepared.feature_columns)
        )
        
        mlflow.xgboost.log_model(
            booster,
            name="model",
            input_example=prepared.train[prepared.feature_columns].head(5),
            model_format="json",
        )
        
        return compute_ranking_metric(
            scored,
            group_key=training["group_key"],
            label_column=training["label_column"],
            score_column="score"
        )
    finally:
        ray.shutdown()
        shutil.rmtree(shard_dir, ignore_errors=True)
