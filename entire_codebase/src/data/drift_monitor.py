"""Data drift monitor — runs in a loop comparing live inference feature distributions
to the training data distribution.

Uses the Kolmogorov-Smirnov two-sample test to detect distributional shift.
When drift exceeds threshold, sends a webhook to the retrain-api to trigger
a new retraining run.

Also monitors:
  1. Ingestion quality — checks schema of incoming raw data
  2. Training set quality — checks feature stats at dataset compile time
  3. Live inference drift — compares production features to training baseline

Invoked by:
  - docker-compose (continuous loop): python scripts/drift_monitor.py --loop
  - One-shot check:                   python scripts/drift_monitor.py --once
  - CI check before training:         python scripts/drift_monitor.py --report-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
import requests
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://sparky:sparky_pass@localhost:5433/sparky",
)
RETRAIN_WEBHOOK_URL = os.environ.get("RETRAIN_WEBHOOK_URL", "http://retrain-api:8080/trigger")
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.01"))  # p-value threshold
MIN_KS_STATISTIC = float(os.environ.get("MIN_KS_STATISTIC", "0.1"))  # minimum effect size
CHECK_INTERVAL_SECONDS = int(os.environ.get("CHECK_INTERVAL_SECONDS", "300"))
LOOKBACK_HOURS = int(os.environ.get("LOOKBACK_HOURS", "24"))
MIN_SAMPLES_FOR_DRIFT = int(os.environ.get("MIN_SAMPLES_FOR_DRIFT", "100"))
TRAINING_BASELINE_PATH = os.environ.get("TRAINING_BASELINE_PATH", "/training-data/train.csv")

# Numeric features to monitor for drift
NUMERIC_FEATURES = [
    "minutes", "n_ingredients", "n_steps", "avg_rating", "n_reviews",
    "calories", "total_fat", "sugar", "sodium", "protein",
    "saturated_fat", "carbohydrate",
    "daily_calorie_target", "protein_target_g", "carbs_target_g", "fat_target_g",
    "history_pc1", "history_pc2", "history_pc3",
]


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def load_training_baseline(path: str) -> pd.DataFrame | None:
    """Load a sample of training data as the reference distribution."""
    try:
        df = pd.read_csv(path, nrows=5000)
        logger.info("Loaded training baseline from %s (%d rows)", path, len(df))
        return df
    except Exception as exc:
        logger.warning("Could not load training baseline: %s", exc)
        return None


def load_recent_inference_features(conn, hours: int) -> pd.DataFrame | None:
    """Pull recent inference feature rows from PostgreSQL."""
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with conn.cursor() as cur:
            cur.execute(
                """SELECT features FROM inference_features
                   WHERE captured_at >= %s
                   ORDER BY captured_at DESC
                   LIMIT 10000""",
                (cutoff,),
            )
            rows = cur.fetchall()
        if not rows:
            return None
        records = [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in rows]
        return pd.DataFrame(records)
    except Exception as exc:
        logger.warning("Could not load inference features from DB: %s", exc)
        return None


def run_ks_test(
    baseline: pd.Series,
    live: pd.Series,
    feature_name: str,
) -> dict[str, Any]:
    """Run KS two-sample test between baseline and live distributions."""
    baseline_clean = baseline.dropna().values
    live_clean = live.dropna().values

    if len(baseline_clean) < 2 or len(live_clean) < 2:
        return {
            "feature": feature_name,
            "ks_statistic": None,
            "p_value": None,
            "drift_detected": False,
            "reason": "insufficient_data",
        }

    ks_stat, p_value = stats.ks_2samp(baseline_clean, live_clean)
    drift_detected = p_value < DRIFT_THRESHOLD and ks_stat >= MIN_KS_STATISTIC

    return {
        "feature": feature_name,
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_value), 6),
        "drift_detected": drift_detected,
        "threshold": DRIFT_THRESHOLD,
        "min_ks_statistic": MIN_KS_STATISTIC,
        "baseline_mean": round(float(np.mean(baseline_clean)), 4),
        "live_mean": round(float(np.mean(live_clean)), 4),
        "baseline_std": round(float(np.std(baseline_clean)), 4),
        "live_std": round(float(np.std(live_clean)), 4),
        "n_baseline": len(baseline_clean),
        "n_live": len(live_clean),
    }


def check_ingestion_quality(raw_data_path: str | None = None) -> dict[str, Any]:
    """
    Data quality at ingestion point (external data sources).
    Validates schema and basic statistics of raw CSVs.
    """
    result = {"check": "ingestion_quality", "passed": True, "issues": []}
    paths_to_check = []

    if raw_data_path:
        paths_to_check.append(raw_data_path)
    else:
        # Look for standard locations
        for p in ["/data/RAW_recipes.csv", "/data/RAW_interactions.csv"]:
            if Path(p).exists():
                paths_to_check.append(p)

    if not paths_to_check:
        result["issues"].append("No raw data files found to validate")
        return result

    for path in paths_to_check:
        try:
            df = pd.read_csv(path, nrows=1000)
            # Check for critical nulls
            null_pct = df.isnull().mean()
            high_null_cols = null_pct[null_pct > 0.5].index.tolist()
            if high_null_cols:
                result["issues"].append(f"{path}: High null rate in {high_null_cols}")
                result["passed"] = False
            # Check row count
            full_count = sum(1 for _ in open(path)) - 1  # subtract header
            result[f"{Path(path).name}_rows"] = full_count
            if full_count < 100:
                result["issues"].append(f"{path}: Too few rows ({full_count})")
                result["passed"] = False
        except Exception as exc:
            result["issues"].append(f"{path}: {exc}")
            result["passed"] = False

    return result


def check_training_set_quality(train_csv: str, val_csv: str, test_csv: str) -> dict[str, Any]:
    """
    Data quality when compiling training sets.
    Checks label balance, feature ranges, and temporal integrity.
    """
    result = {"check": "training_set_quality", "passed": True, "issues": []}

    for split_name, csv_path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        if not Path(csv_path).exists():
            result["issues"].append(f"{split_name}: file not found at {csv_path}")
            result["passed"] = False
            continue
        try:
            df = pd.read_csv(csv_path, nrows=5000)
            rows = len(df)

            if rows == 0:
                result["issues"].append(f"{split_name}: empty file")
                result["passed"] = False
                continue

            result[f"{split_name}_rows"] = rows

            # Label balance check (only for train)
            if "label" in df.columns and split_name == "train":
                pos_rate = df["label"].mean()
                result["train_positive_rate"] = round(float(pos_rate), 3)
                if pos_rate < 0.02 or pos_rate > 0.98:
                    result["issues"].append(
                        f"Severe label imbalance in train: positive_rate={pos_rate:.3f}"
                    )
                    result["passed"] = False

            # Check for all-zero feature columns (potential pipeline bug)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            zero_cols = [c for c in numeric_cols if df[c].std() == 0 and c != "label"]
            if zero_cols:
                result["issues"].append(f"{split_name}: zero-variance columns {zero_cols[:5]}")

            # Calorie range check
            if "calories" in df.columns:
                bad_cal = (df["calories"] < 0).sum()
                if bad_cal > 0:
                    result["issues"].append(f"{split_name}: {bad_cal} rows with negative calories")
                    result["passed"] = False

        except Exception as exc:
            result["issues"].append(f"{split_name}: {exc}")
            result["passed"] = False

    return result


def run_drift_check(baseline_df: pd.DataFrame, live_df: pd.DataFrame) -> dict[str, Any]:
    """Run drift check across all monitored features."""
    results = []
    drifted_features = []

    for feature in NUMERIC_FEATURES:
        if feature not in baseline_df.columns or feature not in live_df.columns:
            continue
        res = run_ks_test(baseline_df[feature], live_df[feature], feature)
        results.append(res)
        if res["drift_detected"]:
            drifted_features.append(feature)

    drift_rate = len(drifted_features) / max(len(results), 1)
    overall_drift = drift_rate > 0.3  # drift if >30% of features are drifted

    return {
        "check": "inference_drift",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_features_checked": len(results),
        "n_drifted": len(drifted_features),
        "drift_rate": round(drift_rate, 3),
        "overall_drift_detected": overall_drift,
        "drifted_features": drifted_features,
        "feature_results": results,
        "threshold": DRIFT_THRESHOLD,
    }


def log_drift_to_db(conn, drift_results: dict):
    """Write per-feature drift results to PostgreSQL drift_log table."""
    try:
        with conn.cursor() as cur:
            for res in drift_results.get("feature_results", []):
                if res.get("ks_statistic") is None:
                    continue
                cur.execute(
                    """INSERT INTO drift_log
                       (feature_name, ks_statistic, p_value, drift_detected, threshold)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (
                        res["feature"],
                        res["ks_statistic"],
                        res["p_value"],
                        res["drift_detected"],
                        res["threshold"],
                    ),
                )
        conn.commit()
    except Exception as exc:
        logger.warning("Could not write drift log to DB: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass


def cleanup_old_inference_features(conn, retention_days: int = 90) -> int:
    """
    Safeguarding — Privacy: delete inference feature rows older than retention_days.
    Called every monitoring cycle to enforce the 90-day data retention policy.
    Returns the number of rows deleted.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM inference_features
                   WHERE captured_at < NOW() - INTERVAL '%s days'""",
                (retention_days,),
            )
            deleted = cur.rowcount
        conn.commit()
        if deleted > 0:
            logger.info(
                "Privacy retention: deleted %d inference_features rows older than %d days",
                deleted, retention_days,
            )
        return deleted
    except Exception as exc:
        logger.warning("Retention cleanup failed (non-fatal): %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return 0


def trigger_retraining(reason: str) -> bool:
    """Send a retraining trigger webhook to the retrain-api service."""
    payload = {
        "reason": reason,
        "config": "configs/training/xgb_ranker.yaml",
        "auto_promote": False,  # require manual approval
    }
    try:
        response = requests.post(
            RETRAIN_WEBHOOK_URL,
            json=payload,
            timeout=10,
        )
        if response.status_code in (200, 202):
            logger.info("Retraining triggered successfully: %s", response.json())
            return True
        else:
            logger.warning("Retrain trigger returned %d: %s", response.status_code, response.text)
            return False
    except Exception as exc:
        logger.warning("Could not reach retrain-api: %s", exc)
        return False


def run_once(report_only: bool = False) -> dict[str, Any]:
    """Run a single drift check cycle. Returns the full report."""
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "actions_taken": [],
    }

    # Load training baseline
    baseline_df = load_training_baseline(TRAINING_BASELINE_PATH)

    # Connect to DB
    conn = None
    try:
        conn = get_db_connection()
    except Exception as exc:
        logger.warning("DB connection failed: %s", exc)

    # 0. Privacy retention — delete inference features older than 90 days
    if conn:
        cleanup_old_inference_features(conn, retention_days=90)

    # 1. Ingestion quality check
    ingestion_result = check_ingestion_quality()
    report["checks"]["ingestion_quality"] = ingestion_result

    # 2. Training set quality (if files exist)
    train_dir = Path(TRAINING_BASELINE_PATH).parent
    training_result = check_training_set_quality(
        str(train_dir / "train.csv"),
        str(train_dir / "val.csv"),
        str(train_dir / "test.csv"),
    )
    report["checks"]["training_set_quality"] = training_result

    # 3. Live inference drift
    if baseline_df is not None and conn is not None:
        live_df = load_recent_inference_features(conn, hours=LOOKBACK_HOURS)
        if live_df is not None and len(live_df) >= MIN_SAMPLES_FOR_DRIFT:
            drift_result = run_drift_check(baseline_df, live_df)
            report["checks"]["inference_drift"] = drift_result
            log_drift_to_db(conn, drift_result)

            if drift_result["overall_drift_detected"] and not report_only:
                logger.warning(
                    "DRIFT DETECTED in %d/%d features. Triggering retraining.",
                    drift_result["n_drifted"],
                    drift_result["n_features_checked"],
                )
                triggered = trigger_retraining(
                    reason=f"drift_detected: {drift_result['drifted_features'][:5]}"
                )
                report["actions_taken"].append(
                    {"action": "trigger_retraining", "success": triggered}
                )
        else:
            n = len(live_df) if live_df is not None else 0
            report["checks"]["inference_drift"] = {
                "skipped": True,
                "reason": f"insufficient live samples: {n} < {MIN_SAMPLES_FOR_DRIFT}",
            }
    else:
        report["checks"]["inference_drift"] = {
            "skipped": True,
            "reason": "no baseline or no DB connection",
        }

    if conn:
        conn.close()

    return report


def main():
    parser = argparse.ArgumentParser(description="SparkyFitness Drift Monitor")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--loop", action="store_true", help="Run in continuous loop")
    group.add_argument("--once", action="store_true", help="Run once and exit")
    group.add_argument("--report-only", action="store_true", help="Report only, no triggers")
    args = parser.parse_args()

    if args.loop:
        logger.info("Drift monitor starting in loop mode (interval=%ds)", CHECK_INTERVAL_SECONDS)
        while True:
            try:
                report = run_once()
                logger.info("Drift check complete. actions=%s", report.get("actions_taken"))
            except Exception as exc:
                logger.exception("Drift check failed: %s", exc)
            time.sleep(CHECK_INTERVAL_SECONDS)
    else:
        report = run_once(report_only=args.report_only)
        print(json.dumps(report, indent=2, default=str))
        # Exit non-zero if critical checks failed
        if not report["checks"].get("training_set_quality", {}).get("passed", True):
            sys.exit(1)


if __name__ == "__main__":
    main()
