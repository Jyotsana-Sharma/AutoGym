"""
run_soda_checks.py  —  Bonus: Data Quality Checks with Soda
=============================================================
Runs automated data quality validation on all pipeline outputs
using Soda Core. Catches bad data before it enters training.

Runs non-interactively:
  docker compose --profile soda run --rm soda-checks

Concrete example this solves:
  If a Kaggle dataset update introduces null calories or duplicate
  recipe IDs, Soda catches it BEFORE the XGBoost model trains on
  bad data — preventing silent model degradation in production.
"""
import os, sys, time
from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    from soda.scan import Scan
    HAS_SODA = True
except ImportError:
    HAS_SODA = False


def run_soda_scan(data_dir, checks_path=None):
    """Run Soda quality checks on all pipeline CSVs."""
    print("=" * 60)
    print("SparkyFitness — Soda Data Quality Checks")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    data_dir = Path(data_dir)

    # Define which files to check
    datasets = {
        "enriched_recipes": data_dir / "enriched_recipes.csv",
        "train": data_dir / "train.csv",
        "val": data_dir / "val.csv",
        "test": data_dir / "test.csv",
    }

    # Load all available datasets
    loaded = {}
    for name, path in datasets.items():
        if path.exists():
            loaded[name] = pd.read_csv(path)
            print(f"  Loaded {name}: {loaded[name].shape}")
        else:
            print(f"  SKIP: {path} not found")

    if not loaded:
        print("  ERROR: No data files found")
        sys.exit(1)

    if HAS_SODA:
        print("\n-- Running Soda Core checks --")
        return run_with_soda(loaded, checks_path)
    else:
        print("\n-- Soda not installed, running built-in checks --")
        return run_builtin_checks(loaded)


def run_with_soda(loaded, checks_path):
    """Run checks using Soda Core library."""
    checks_yaml = None
    if checks_path and os.path.exists(checks_path):
        with open(checks_path) as f:
            checks_yaml = f.read()
    else:
        checks_yaml = get_default_checks()

    scan = Scan()
    scan.set_data_source_name("dask")
    scan.set_scan_definition_name("sparky-quality-scan")

    for name, df in loaded.items():
        scan.add_pandas_dataframe(dataset_name=name, pandas_df=df)

    scan.add_sodacl_yaml_str(checks_yaml)
    scan.execute()

    results = scan.get_scan_results()
    passed = scan.get_checks_pass()
    warned = scan.get_checks_warn()
    failed = scan.get_checks_fail()

    print(f"\n-- Soda Results --")
    print(f"  Passed:  {len(passed)}")
    print(f"  Warned:  {len(warned)}")
    print(f"  Failed:  {len(failed)}")

    if failed:
        print("\n  FAILED CHECKS:")
        for check in failed:
            print(f"    ✗ {check}")

    if warned:
        print("\n  WARNINGS:")
        for check in warned:
            print(f"    ⚠ {check}")

    print(f"\n  Overall: {'PASS' if not failed else 'FAIL'}")
    return len(failed) == 0


def run_builtin_checks(loaded):
    """Fallback: run equivalent checks without Soda library."""
    total_checks = 0
    passed_checks = 0
    failed_details = []

    def check(name, condition, msg):
        nonlocal total_checks, passed_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f"    ✓ {msg}")
        else:
            failed_details.append(msg)
            print(f"    ✗ FAIL: {msg}")

    # --- enriched_recipes checks ---
    if "enriched_recipes" in loaded:
        df = loaded["enriched_recipes"]
        print(f"\n  Checking enriched_recipes ({len(df):,} rows):")
        check("row_count", len(df) > 50000,
              f"row_count > 50000 (actual: {len(df):,})")
        check("recipe_id_nulls", df["recipe_id"].notna().all(),
              f"no null recipe_ids (nulls: {df['recipe_id'].isna().sum()})")
        check("recipe_id_unique", df["recipe_id"].nunique() == len(df),
              f"no duplicate recipe_ids (dupes: {len(df) - df['recipe_id'].nunique()})")
        check("calories_nulls", df["calories"].notna().all() if "calories" in df.columns else False,
              f"no null calories (nulls: {df['calories'].isna().sum() if 'calories' in df.columns else 'N/A'})")
        if "calories" in df.columns:
            check("calories_min", df["calories"].min() >= 0,
                  f"min calories >= 0 (actual: {df['calories'].min():.1f})")
            check("calories_max", df["calories"].max() < 500000,
                  f"max calories < 500000 (actual: {df['calories'].max():.1f})")
            check("calories_avg", 200 <= df["calories"].mean() <= 800,
                  f"avg calories between 200-800 (actual: {df['calories'].mean():.1f})")
        if "minutes" in df.columns:
            check("minutes_min", df["minutes"].min() > 0,
                  f"min minutes > 0 (actual: {df['minutes'].min()})")
        if "n_ingredients" in df.columns:
            check("ingredients_min", df["n_ingredients"].min() >= 1,
                  f"min n_ingredients >= 1 (actual: {df['n_ingredients'].min()})")
        if "n_reviews" in df.columns:
            check("reviews_min", df["n_reviews"].min() >= 3,
                  f"min n_reviews >= 3 (actual: {df['n_reviews'].min()})")
        if "cuisine" in df.columns:
            check("cuisine_nulls", df["cuisine"].notna().all(),
                  f"no null cuisine (nulls: {df['cuisine'].isna().sum()})")

    # --- train/val/test checks ---
    for split_name in ["train", "val", "test"]:
        if split_name in loaded:
            df = loaded[split_name]
            print(f"\n  Checking {split_name} ({len(df):,} rows):")
            min_rows = 100000 if split_name == "train" else 1000
            check(f"{split_name}_rows", len(df) > min_rows,
                  f"row_count > {min_rows:,} (actual: {len(df):,})")
            for col in ["user_id", "recipe_id", "label"]:
                if col in df.columns:
                    check(f"{split_name}_{col}_null", df[col].notna().all(),
                          f"no null {col} (nulls: {df[col].isna().sum()})")
            if "label" in df.columns:
                bal = df["label"].mean()
                check(f"{split_name}_balance", 0.05 < bal < 0.95,
                      f"label balance between 0.1-0.9 (actual: {bal:.3f})")

    # --- Temporal leakage check across splits ---
    if "train" in loaded and "test" in loaded:
        if "date" in loaded["train"].columns and "date" in loaded["test"].columns:
            print(f"\n  Checking temporal leakage:")
            source_col = "data_source" if "data_source" in loaded["train"].columns and "data_source" in loaded["test"].columns else None
            sources = sorted(set(loaded["train"][source_col].dropna().astype(str)) | set(loaded["test"][source_col].dropna().astype(str))) if source_col else [None]
            for source in sources:
                train_part = loaded["train"][loaded["train"][source_col].astype(str) == source] if source_col else loaded["train"]
                test_part = loaded["test"][loaded["test"][source_col].astype(str) == source] if source_col else loaded["test"]
                if train_part.empty or test_part.empty:
                    continue
                train_max = pd.to_datetime(train_part["date"], errors="coerce", format="mixed", utc=True).max()
                test_min = pd.to_datetime(test_part["date"], errors="coerce", format="mixed", utc=True).min()
                label = f" for {source}" if source_col else ""
                check(f"temporal{label}", train_max < test_min,
                      f"train max{label} ({train_max}) < test min{label} ({test_min})")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SODA QUALITY REPORT")
    print(f"  Total checks:  {total_checks}")
    print(f"  Passed:        {passed_checks}")
    print(f"  Failed:        {total_checks - passed_checks}")
    if failed_details:
        print(f"\n  FAILURES:")
        for f in failed_details:
            print(f"    ✗ {f}")
    print(f"\n  Result: {'PASS ✓' if not failed_details else 'FAIL ✗'}")
    print(f"{'='*60}")

    return len(failed_details) == 0


def get_default_checks():
    """Return default Soda checks YAML."""
    return """
checks for enriched_recipes:
  - row_count > 50000
  - missing_count(recipe_id) = 0
  - duplicate_count(recipe_id) = 0
  - missing_count(calories) = 0
  - min(calories) >= 0
  - max(calories) < 10000
  - avg(calories) between 200 and 800
  - min(minutes) > 0
  - min(n_ingredients) >= 1
  - min(n_reviews) >= 3

checks for train:
  - row_count > 100000
  - missing_count(user_id) = 0
  - missing_count(recipe_id) = 0
  - missing_count(label) = 0
  - avg(label) between 0.05 and 0.95

checks for val:
  - row_count > 1000
  - missing_count(label) = 0

checks for test:
  - row_count > 1000
  - missing_count(label) = 0
"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--checks", default="./configs/soda_checks.yml")
    args = ap.parse_args()
    success = run_soda_scan(args.data_dir, args.checks)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
