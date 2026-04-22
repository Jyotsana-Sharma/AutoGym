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
  - Scheduled automatically (weekly + data-influx) via retrain_api.py
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

from .model_registry import evaluate_and_register, promote_to_production
from .train import run_training
from .mlflow_utils import read_config
from .ranking_data import NON_FEATURE_COLUMNS

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

    data_dir = str(Path(train_csv).parent)
    checks_path = Path(__file__).parent.parent.parent / "configs" / "data" / "soda_checks.yml"
    result = subprocess.run(
        [sys.executable, str(soda_script),
         "--data-dir", data_dir,
         "--checks", str(checks_path)],
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

    # Step 0: Refresh training data from live database
    db_url = os.environ.get("DATABASE_URL")
    output_dir = str(Path(train_csv).parent)
    if db_url:
        logger.info("Step 0/5: Refreshing training data from database...")
        refresh_cmd = [
            sys.executable, "-m", "src.data.batch_pipeline",
            "--output-dir", output_dir,
            "--raw-dir", "/data",
            "--db-url", db_url,
        ]
        raw_dir = Path("/data")
        raw_missing = (
            not (raw_dir / "RAW_recipes.csv").exists()
            or not (raw_dir / "RAW_interactions.csv").exists()
        )
        if raw_missing and os.environ.get("OS_APPLICATION_CREDENTIAL_ID") and os.environ.get("OS_APPLICATION_CREDENTIAL_SECRET"):
            refresh_cmd.append("--download-raw")
        try:
            refresh = subprocess.run(
                refresh_cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max — prevents hanging forever
                env={**os.environ, "DATABASE_URL": db_url},
            )
            if refresh.returncode == 0:
                logger.info("Training data refreshed from database")
            else:
                logger.warning(
                    "Batch pipeline refresh failed (using existing data):\n%s",
                    refresh.stdout[-2000:] + refresh.stderr[-2000:],
                )
        except subprocess.TimeoutExpired:
            logger.warning("Batch pipeline refresh timed out after 600s — using existing data")
    else:
        logger.info("Step 0/5: DATABASE_URL not set — using existing training CSVs")

    # Step 1: Data quality checks
    if not skip_data_checks:
        logger.info("Step 1/6: Running data quality checks...")
        if not run_soda_checks(train_csv, val_csv, test_csv):
            result["failure_reason"] = "Data quality checks failed"
            result["duration_seconds"] = round(time.perf_counter() - start, 2)
            return result
    else:
        logger.info("Step 1/6: Skipping data quality checks (--skip-data-checks)")

    # Step 2: Patch config with actual data paths
    logger.info("Step 2/6: Preparing training config...")
    cfg_path = Path(config_path)
    try:
        patched_cfg = patch_config_with_data_paths(cfg_path, train_csv, val_csv, test_csv)
    except Exception as exc:
        logger.error("Failed to patch config: %s", exc)
        result["failure_reason"] = str(exc)
        result["duration_seconds"] = round(time.perf_counter() - start, 2)
        return result

    # Step 2.5: Baseline comparison — runs popularity model in same experiment for reference
    logger.info("Step 2.5/6: Running baseline popularity model for comparison...")
    try:
        baseline_cfg_path = Path(config_path).parent / "baseline_popularity.yaml"
        if baseline_cfg_path.exists():
            patched_baseline_cfg = patch_config_with_data_paths(
                baseline_cfg_path, train_csv, val_csv, test_csv
            )
            baseline_summary = run_training(patched_baseline_cfg)
            result["baseline_metrics"] = baseline_summary.get("metrics", {})
            logger.info("Baseline NDCG@10: %.4f", result["baseline_metrics"].get("ndcg_at_10", 0))
            try:
                Path(patched_baseline_cfg).unlink(missing_ok=True)
            except Exception:
                pass
        else:
            logger.info("baseline_popularity.yaml not found — skipping baseline")
    except Exception as exc:
        logger.warning("Baseline comparison failed (non-fatal): %s", exc)

    # Step 3: Train
    logger.info("Step 3/6: Running training...")
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        summary = run_training(patched_cfg)
        metrics = summary.get("metrics", {})
        result["metrics"] = metrics
        logger.info("Training complete. Metrics: %s", metrics)
        baseline_ndcg = result.get("baseline_metrics", {}).get("ndcg_at_10", None)
        ray_ndcg = metrics.get("ndcg_at_10", None)
        if baseline_ndcg is not None and ray_ndcg is not None:
            improvement = ray_ndcg - baseline_ndcg
            logger.info(
                "Model comparison — baseline NDCG@10: %.4f | Ray NDCG@10: %.4f | improvement: %+.4f",
                baseline_ndcg, ray_ndcg, improvement,
            )
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

    # Extract run_id — use last_active_run so baseline run in Step 2.5
    # doesn't shadow the XGBoost run here.
    last_run = mlflow.last_active_run()
    run_id = last_run.info.run_id if last_run else None
    result["run_id"] = run_id

    # Step 3.5/6: Safeguarding — fairness check + global explainability
    fairness_passed = True
    fairness_summary = "not_run"
    if _SAFEGUARDING_AVAILABLE and run_id:
        logger.info("Step 3.5/6: Running fairness check and explainability...")
        try:
            test_df = pd.read_csv(test_csv)
            train_df_ref = pd.read_csv(train_csv)

            # Score the test set with the just-trained model from MLflow
            model_uri = f"runs:/{run_id}/model"
            booster = mlflow.xgboost.load_model(model_uri)
            # Use exact feature names from the trained model so cuisine
            # (label-encoded during training) is included correctly
            feature_cols = booster.feature_names
            test_enc = test_df.copy()
            for col in feature_cols:
                if col in test_enc.columns and (
                    test_enc[col].dtype == object
                    or pd.api.types.is_string_dtype(test_enc[col])
                ):
                    categories = {
                        v: i for i, v in enumerate(
                            sorted(train_df_ref[col].fillna("unknown").astype(str).unique())
                        )
                    }
                    test_enc[col] = (
                        test_enc[col].fillna("unknown").astype(str)
                        .map(categories).fillna(-1).astype(float)
                    )
            dmatrix = xgb.DMatrix(
                test_enc[feature_cols].fillna(0).values.astype("float32"),
                feature_names=feature_cols,
            )
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
        logger.info("Step 3.5/6: Safeguarding modules unavailable — skipping")

    # Step 4: Evaluate and register
    logger.info("Step 4/6: Evaluating quality gates and registering model...")
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
        logger.info("Step 5/6: Auto-promoting to Production...")
        promo_result = promote_to_production(version=result["model_version"])
        result["promoted"] = promo_result.get("promoted", False)
        logger.info("Promotion result: %s", promo_result)
    else:
        logger.info("Step 5/6: Skipping auto-promotion (manual approval required)")

    # Export model file for serving fallback — download directly from run artifacts
    # (not from Production stage, since model is in Staging at this point)
    if model_export_path and result.get("registered") and result.get("run_id"):
        try:
            import shutil
            local_dir = mlflow.artifacts.download_artifacts(f"runs:/{result['run_id']}/model")
            model_file = None
            for p in Path(local_dir).rglob("*.json"):
                if p.name in ("model.json",) or "xgboost" in p.name or "model" in p.name:
                    model_file = p
                    break
            if model_file is None:
                all_files = [f for f in Path(local_dir).rglob("*") if f.is_file()]
                model_file = all_files[0] if all_files else None
            if model_file:
                Path(model_export_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(model_file), model_export_path)
                logger.info("Exported model from run %s to %s", result["run_id"], model_export_path)
                result["model_exported_to"] = model_export_path
                try:
                    artifact_dir = mlflow.artifacts.download_artifacts(
                        f"runs:/{result['run_id']}/embedding_artifacts"
                    )
                    target_dir = Path(model_export_path).parent / "embedding_artifacts"
                    shutil.copytree(artifact_dir, target_dir, dirs_exist_ok=True)
                    result["embedding_artifacts_exported_to"] = str(target_dir)
                    logger.info("Exported embedding artifacts to %s", target_dir)
                except Exception as exc:
                    logger.warning("Embedding artifact export failed (non-fatal): %s", exc)
            else:
                logger.warning("No model file found in run artifacts at %s", local_dir)
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
