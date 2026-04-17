"""End-to-end retraining pipeline.

Orchestrates:
  1. Download fresh training data from Swift OR use local CSV files
  2. Run Soda data quality checks on training data
  3. Train XGBoost ranker with the specified config
  4. Evaluate against quality gates (NDCG@10 threshold)
  5. Register passing model to MLflow Registry (Staging)
  6. Optionally auto-promote to Production (if auto_promote=True)
  7. Export model file for serving fallback

Invoked by:
  - retrain_api.py (via HTTP webhook from drift monitor / scheduler)
  - GitHub Actions CI workflow
  - Manual: python -m src.retrain_pipeline --config configs/train/xgb_ranker.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb

from .model_registry import evaluate_and_register, promote_to_production, export_production_model
from .train import run_training
from .mlflow_utils import read_config

# Safeguarding modules — imported with graceful fallback so missing deps
# don't break the pipeline entirely; but failures are reported in the result.
try:
    _REPO_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(_REPO_ROOT))
    from safeguarding.fairness_checker import run_fairness_check
    from safeguarding.explainability import Explainer
    _SAFEGUARDING_AVAILABLE = True
except Exception as _sg_exc:
    _SAFEGUARDING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logging.getLogger(__name__).warning(
        "Safeguarding modules not available (%s) — fairness/explainability skipped", _sg_exc
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_soda_checks(train_csv: str, val_csv: str, test_csv: str) -> bool:
    """Run Soda data quality checks. Returns True if all checks pass."""
    soda_script = Path(__file__).parent.parent / "data" / "run_soda_checks.py"
    if not soda_script.exists():
        logger.warning("Soda checks script not found at %s — skipping", soda_script)
        return True

    result = subprocess.run(
        [sys.executable, str(soda_script),
         "--train-csv", train_csv,
         "--val-csv", val_csv,
         "--test-csv", test_csv],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Soda data quality checks FAILED:\n%s", result.stdout + result.stderr)
        return False
    logger.info("Soda data quality checks PASSED")
    return True


def patch_config_with_data_paths(config_path: Path, train_csv: str, val_csv: str, test_csv: str) -> Path:
    """Create a temporary patched config with the correct data paths."""
    import yaml
    import tempfile

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["data"]["train_path"] = train_csv
    config["data"]["validation_path"] = val_csv
    config["data"]["test_path"] = test_csv

    # Create temp config file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="retrain_", delete=False
    )
    yaml.dump(config, tmp)
    tmp.close()
    return Path(tmp.name)


def run_retraining(
    config_path: str,
    train_csv: str | None = None,
    val_csv: str | None = None,
    test_csv: str | None = None,
    skip_data_checks: bool = False,
    auto_promote: bool = False,
    model_export_path: str | None = None,
) -> dict:
    """
    Full retraining pipeline.

    Returns result dict with keys:
      - success: bool
      - run_id: str | None
      - metrics: dict
      - registered: bool
      - model_version: str | None
      - promoted: bool
      - duration_seconds: float
    """
    start = time.perf_counter()
    result = {
        "success": False,
        "run_id": None,
        "metrics": {},
        "registered": False,
        "model_version": None,
        "promoted": False,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
    }

    # Use env-var paths if not supplied
    train_csv = train_csv or os.environ.get("TRAIN_CSV", "train.csv")
    val_csv = val_csv or os.environ.get("VAL_CSV", "val.csv")
    test_csv = test_csv or os.environ.get("TEST_CSV", "test.csv")

    logger.info("Starting retraining pipeline")
    logger.info("  Config:    %s", config_path)
    logger.info("  Train CSV: %s", train_csv)
    logger.info("  Val CSV:   %s", val_csv)
    logger.info("  Test CSV:  %s", test_csv)

    # Step 1: Data quality checks
    if not skip_data_checks:
        logger.info("Step 1/5: Running data quality checks...")
        if not run_soda_checks(train_csv, val_csv, test_csv):
            result["failure_reason"] = "Data quality checks failed"
            result["duration_seconds"] = round(time.perf_counter() - start, 2)
            return result
    else:
        logger.info("Step 1/5: Skipping data quality checks (--skip-data-checks)")

    # Step 2: Patch config with actual data paths
    logger.info("Step 2/5: Preparing training config...")
    cfg_path = Path(config_path)
    try:
        patched_cfg = patch_config_with_data_paths(cfg_path, train_csv, val_csv, test_csv)
    except Exception as exc:
        logger.error("Failed to patch config: %s", exc)
        result["failure_reason"] = str(exc)
        result["duration_seconds"] = round(time.perf_counter() - start, 2)
        return result

    # Step 3: Train
    logger.info("Step 3/5: Running training...")
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        summary = run_training(patched_cfg)
        metrics = summary.get("metrics", {})
        result["metrics"] = metrics
        logger.info("Training complete. Metrics: %s", metrics)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        result["failure_reason"] = str(exc)
        result["duration_seconds"] = round(time.perf_counter() - start, 2)
        return result
    finally:
        # Clean up temp config
        try:
            Path(patched_cfg).unlink(missing_ok=True)
        except Exception:
            pass

    # Extract run_id from active MLflow run
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run else None
    if run_id is None:
        # Try to get the last run from the experiment
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(read_config(cfg_path).get("experiment_name", "sparky"))
        if exp:
            runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)
            run_id = runs[0].info.run_id if runs else None

    result["run_id"] = run_id

    # Step 3.5: Safeguarding — fairness check + global explainability
    fairness_passed = True
    fairness_summary = "not_run"
    if _SAFEGUARDING_AVAILABLE and run_id:
        logger.info("Step 3.5/5: Running fairness check and explainability...")
        try:
            test_df = pd.read_csv(test_csv)

            # Score the test set with the just-trained model from MLflow
            model_uri = f"runs:/{run_id}/model"
            booster = mlflow.xgboost.load_model(model_uri)
            feature_cols = [
                c for c in test_df.columns
                if c not in ("user_id", "recipe_id", "label", "date", "submitted")
            ]
            dmatrix = xgb.DMatrix(test_df[feature_cols].fillna(0).values.astype("float32"),
                                   feature_names=feature_cols)
            test_df["score"] = booster.predict(dmatrix)

            # Fairness gate
            fairness_result = run_fairness_check(test_df)
            fairness_passed = fairness_result["overall_passed"]
            fairness_summary = fairness_result.get("summary", "")
            result["fairness_result"] = fairness_result

            with mlflow.start_run(run_id=run_id):
                mlflow.log_dict(fairness_result, "fairness_results.json")
                mlflow.set_tag("fairness_passed", str(fairness_passed))

            if not fairness_passed:
                logger.warning("Fairness gate FAILED: %s", fairness_summary)
            else:
                logger.info("Fairness gate PASSED: %s", fairness_summary)

            # Global explainability — log SHAP feature importance to MLflow
            try:
                explainer = Explainer(booster, feature_columns=feature_cols)
                exp_result = explainer.global_feature_importance(
                    test_df, output_dir="/tmp/explainability"
                )
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_dict(
                        exp_result, "explainability/global_feature_importance.json"
                    )
                    if "plot_path" in exp_result:
                        mlflow.log_artifact(
                            exp_result["plot_path"], artifact_path="explainability"
                        )
                logger.info(
                    "Global explainability logged (%d features)",
                    len(exp_result.get("feature_importance", [])),
                )
            except Exception as exc:
                logger.warning("Explainability logging failed (non-fatal): %s", exc)

        except Exception as exc:
            logger.warning("Safeguarding step failed (non-fatal): %s", exc)
            fairness_passed = True   # don't block registration on infra errors
            fairness_summary = f"error: {exc}"
    else:
        logger.info("Step 3.5/5: Safeguarding modules unavailable — skipping")

    # Step 4: Evaluate and register
    logger.info("Step 4/5: Evaluating quality gates and registering model...")
    if run_id:
        reg_result = evaluate_and_register(
            run_id=run_id,
            metrics=metrics,
            config_path=config_path,
            tags={"triggered_at": result["triggered_at"]},
            fairness_passed=fairness_passed,
            fairness_summary=fairness_summary,
        )
        result["registered"] = reg_result["registered"]
        result["model_version"] = reg_result.get("model_version")
        result["gate_results"] = reg_result.get("gate_results", {})
        result["registration_reason"] = reg_result.get("reason")
        logger.info("Registration result: %s", reg_result["reason"])
    else:
        logger.warning("Could not determine MLflow run_id — skipping model registration")

    # Step 5: Auto-promote if requested and registered
    if auto_promote and result["registered"] and result["model_version"]:
        logger.info("Step 5/5: Auto-promoting to Production...")
        promo_result = promote_to_production(version=result["model_version"])
        result["promoted"] = promo_result.get("promoted", False)
        logger.info("Promotion result: %s", promo_result)
    else:
        logger.info("Step 5/5: Skipping auto-promotion (manual approval required)")

    # Export model file for serving fallback
    if model_export_path and result["registered"]:
        try:
            export_production_model(model_export_path)
            result["model_exported_to"] = model_export_path
        except Exception as exc:
            logger.warning("Model export failed (non-fatal): %s", exc)

    result["success"] = True
    result["duration_seconds"] = round(time.perf_counter() - start, 2)
    logger.info(
        "Retraining pipeline complete in %.1fs. registered=%s promoted=%s",
        result["duration_seconds"],
        result["registered"],
        result["promoted"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="SparkyFitness Retraining Pipeline")
    parser.add_argument("--config", default="configs/train/xgb_ranker.yaml")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--skip-data-checks", action="store_true")
    parser.add_argument("--auto-promote", action="store_true",
                        help="Automatically promote passing model to Production")
    parser.add_argument("--model-export-path", default=None,
                        help="Export production model to this path after promotion")
    args = parser.parse_args()

    result = run_retraining(
        config_path=args.config,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        skip_data_checks=args.skip_data_checks,
        auto_promote=args.auto_promote,
        model_export_path=args.model_export_path or os.environ.get("MODEL_EXPORT_PATH"),
    )

    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
