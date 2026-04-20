"""MLflow Model Registry integration with quality gates and promotion logic.

Flow:
  1. Training run finishes → call evaluate_and_register()
  2. Model evaluated against NDCG@10 threshold and fairness checks
  3. If passing → registered to MLflow Registry as "Staging"
  4. promote_to_production() called manually OR by automation after canary period
  5. Previous production model is archived (not deleted — enables rollback)
  6. Serving picks up new "Production" version automatically
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "sparky-ranker")
NDCG_THRESHOLD = float(os.environ.get("NDCG_THRESHOLD", "0.79"))
# Minimum improvement over current production model to promote
IMPROVEMENT_THRESHOLD = float(os.environ.get("IMPROVEMENT_THRESHOLD", "0.01"))


def get_client() -> MlflowClient:
    return MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def get_production_model_ndcg(client: MlflowClient) -> float | None:
    """Return the NDCG@10 of the current Production model, or None if none exists."""
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        return run.data.metrics.get("ndcg_at_10")
    except Exception as exc:
        logger.warning("Could not retrieve production model metric: %s", exc)
        return None


def evaluate_and_register(
    run_id: str,
    metrics: dict[str, float],
    config_path: str | None = None,
    tags: dict[str, str] | None = None,
    fairness_passed: bool = True,
    fairness_summary: str = "not_run",
) -> dict[str, Any]:
    """
    Evaluate model quality gates and register to MLflow Registry if passing.

    Quality gates (all must pass):
      1. ndcg_at_10 >= NDCG_THRESHOLD
      2. ndcg_at_10 > current production - IMPROVEMENT_THRESHOLD  (or no prod exists)
      3. fairness_passed — per-group NDCG within 20% of overall, allergen safety <1%

    Returns a result dict with keys:
      - registered: bool
      - model_version: str | None
      - stage: str
      - gate_results: dict
      - reason: str
    """
    client = get_client()
    ndcg = metrics.get("ndcg_at_10", 0.0)
    prod_ndcg = get_production_model_ndcg(client)

    gate_results = {
        "ndcg_threshold": {
            "passed": ndcg >= NDCG_THRESHOLD,
            "value": ndcg,
            "threshold": NDCG_THRESHOLD,
        },
        "improvement_over_prod": {
            "passed": prod_ndcg is None or ndcg > (prod_ndcg - IMPROVEMENT_THRESHOLD),
            "value": ndcg,
            "prod_value": prod_ndcg,
            "threshold_delta": IMPROVEMENT_THRESHOLD,
        },
        "fairness": {
            "passed": fairness_passed,
            "summary": fairness_summary,
        },
    }

    all_passed = all(g["passed"] for g in gate_results.values())

    if not all_passed:
        failed = [k for k, v in gate_results.items() if not v["passed"]]
        reason = f"Quality gates FAILED: {failed}. NDCG={ndcg:.4f} (threshold={NDCG_THRESHOLD})"
        logger.warning(reason)
        # Log gate failure to MLflow run
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("quality_gate_status", "FAILED")
            mlflow.log_dict(gate_results, "quality_gate_results.json")
        return {
            "registered": False,
            "model_version": None,
            "stage": "None",
            "gate_results": gate_results,
            "reason": reason,
        }

    # All gates passed — register model
    model_uri = f"runs:/{run_id}/model"
    run_tags = {
        "ndcg_at_10": str(ndcg),
        "prod_ndcg_at_registration": str(prod_ndcg),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "quality_gate_status": "PASSED",
    }
    if tags:
        run_tags.update(tags)
    if config_path:
        run_tags["config_path"] = config_path

    mv = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
        tags=run_tags,
    )

    # Transition to Staging automatically; Production requires explicit promotion
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    client.update_model_version(
        name=MODEL_NAME,
        version=mv.version,
        description=f"NDCG@10={ndcg:.4f}. Registered {datetime.now(timezone.utc).isoformat()}",
    )

    # Tag the MLflow run
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("quality_gate_status", "PASSED")
        mlflow.set_tag("registry_version", mv.version)
        mlflow.set_tag("registry_stage", "Staging")
        mlflow.log_dict(gate_results, "quality_gate_results.json")

    reason = f"Registered as version {mv.version} in Staging. NDCG@10={ndcg:.4f}"
    logger.info(reason)
    return {
        "registered": True,
        "model_version": mv.version,
        "stage": "Staging",
        "gate_results": gate_results,
        "reason": reason,
    }


def promote_to_production(
    version: str | None = None,
    require_manual_approval: bool = False,
) -> dict[str, Any]:
    """
    Promote a Staging model version to Production.

    If version is None, promotes the latest Staging version.
    Archives the current Production version (for rollback).

    Returns dict with:
      - promoted: bool
      - version: str
      - previous_production_version: str | None
    """
    client = get_client()

    if version is None:
        staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        if not staging_versions:
            msg = "No model in Staging to promote"
            logger.warning(msg)
            return {"promoted": False, "reason": msg}
        version = staging_versions[0].version

    # Archive current production
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    prev_prod_version = None
    if prod_versions:
        prev_prod_version = prod_versions[0].version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=prev_prod_version,
            stage="Archived",
            archive_existing_versions=False,
        )
        logger.info("Archived previous production version %s", prev_prod_version)

    # Promote Staging → Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=False,
    )
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description=(
            f"Promoted to Production at {datetime.now(timezone.utc).isoformat()}. "
            f"Previous production: v{prev_prod_version}"
        ),
    )

    logger.info("Promoted model version %s to Production", version)
    return {
        "promoted": True,
        "version": version,
        "previous_production_version": prev_prod_version,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }


def rollback_production(target_version: str | None = None) -> dict[str, Any]:
    """
    Roll back Production to a previous (Archived) version.

    If target_version is None, rolls back to the most recent Archived version.
    """
    client = get_client()

    # Get current production
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    current_prod = prod_versions[0].version if prod_versions else None

    if target_version is None:
        # Find most recent archived version before current prod
        all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        archived = [
            v for v in all_versions
            if v.current_stage == "Archived"
        ]
        if not archived:
            return {"rolled_back": False, "reason": "No archived versions to roll back to"}
        # Sort by version number descending
        archived.sort(key=lambda v: int(v.version), reverse=True)
        target_version = archived[0].version

    # Archive current production
    if current_prod:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=current_prod,
            stage="Archived",
            archive_existing_versions=False,
        )

    # Restore target version to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=target_version,
        stage="Production",
        archive_existing_versions=False,
    )
    client.update_model_version(
        name=MODEL_NAME,
        version=target_version,
        description=f"Rolled back to Production at {datetime.now(timezone.utc).isoformat()}. Replaced v{current_prod}",
    )

    logger.info("Rolled back to model version %s", target_version)
    return {
        "rolled_back": True,
        "target_version": target_version,
        "replaced_version": current_prod,
        "rolled_back_at": datetime.now(timezone.utc).isoformat(),
    }


def export_production_model(output_path: str) -> str:
    """
    Download the current Production model from MLflow and save it locally.
    Returns the local path to the model file.
    """
    client = get_client()
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not prod_versions:
        raise RuntimeError("No Production model found in MLflow Registry")

    mv = prod_versions[0]
    model_uri = f"models:/{MODEL_NAME}/Production"
    local_dir = mlflow.artifacts.download_artifacts(model_uri)

    # Find the model JSON file
    model_file = None
    for p in Path(local_dir).rglob("*.json"):
        if "model" in p.name or "xgboost" in p.name:
            model_file = p
            break
    if model_file is None:
        # Fall back to any file in model/ subdirectory
        model_dir = Path(local_dir) / "model"
        if model_dir.exists():
            files = list(model_dir.iterdir())
            if files:
                model_file = files[0]

    if model_file is None:
        raise RuntimeError(f"Could not locate model file in {local_dir}")

    import shutil
    shutil.copy2(model_file, output_path)
    logger.info("Exported production model v%s to %s", mv.version, output_path)
    return output_path


def get_production_version_info() -> dict[str, Any] | None:
    """Return metadata about the current Production model version."""
    client = get_client()
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        mv = versions[0]
        run = client.get_run(mv.run_id)
        return {
            "version": mv.version,
            "run_id": mv.run_id,
            "stage": mv.current_stage,
            "creation_timestamp": mv.creation_timestamp,
            "last_updated_timestamp": mv.last_updated_timestamp,
            "description": mv.description,
            "ndcg_at_10": run.data.metrics.get("ndcg_at_10"),
            "tags": dict(mv.tags),
        }
    except Exception as exc:
        logger.warning("Could not get production version info: %s", exc)
        return None


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="MLflow Model Registry CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_promote = sub.add_parser("promote", help="Promote Staging → Production")
    p_promote.add_argument("--version", default=None)

    p_rollback = sub.add_parser("rollback", help="Rollback Production to previous version")
    p_rollback.add_argument("--version", default=None)

    p_export = sub.add_parser("export", help="Export Production model to local path")
    p_export.add_argument("--output", required=True)

    p_info = sub.add_parser("info", help="Show Production model info")

    args = parser.parse_args()

    if args.cmd == "promote":
        result = promote_to_production(version=args.version)
    elif args.cmd == "rollback":
        result = rollback_production(target_version=args.version)
    elif args.cmd == "export":
        result = {"path": export_production_model(args.output)}
    elif args.cmd == "info":
        result = get_production_version_info()
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, indent=2, default=str))
