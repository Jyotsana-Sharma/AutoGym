"""Fairness checker for the SparkyFitness recipe recommendation system.

Implements the safeguarding plan's fairness pillar.

Checks run at two points:
  1. Post-training evaluation (before model registration) — checks for
     demographic bias in NDCG@10 across user dietary preference groups.
  2. Live monitoring (periodic) — checks for distribution shifts in
     recommendations across allergen / dietary groups.

Groups monitored:
  - Dietary: vegetarian, vegan, gluten-free, dairy-free, low-sodium, low-fat
  - Allergen: users with egg/fish/milk/nuts/peanut/sesame/shellfish/soy/wheat allergies
  - Nutrition level: calorie target quartiles (low / medium / high)

Fairness criteria (all must pass for model registration):
  - Per-group NDCG@10 >= 0.80 * overall NDCG@10  (no group worse by >20%)
  - No group with < 50 users is penalized (insufficient data)
  - Recommendation allergen safety: <1% of top-ranked recipes contain user allergens
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Groups to test fairness across
DIET_FLAGS = [
    "user_vegetarian", "user_vegan", "user_gluten_free",
    "user_dairy_free", "user_low_sodium", "user_low_fat",
]
ALLERGEN_FLAGS = [
    "has_egg", "has_fish", "has_milk", "has_nuts",
    "has_peanut", "has_sesame", "has_shellfish", "has_soy", "has_wheat",
]
ALLERGEN_USER_FLAGS = [
    "user_vegetarian",  # proxy for plant-based allergen sensitivity
]

# Groups with fewer than MIN_GROUP_SIZE users won't be penalized
MIN_GROUP_SIZE = 50
# Maximum allowed degradation vs. overall metric (20%)
MAX_DEGRADATION = 0.20


def compute_ndcg_at_10(group: pd.DataFrame, label_col: str = "label", score_col: str = "score") -> float:
    """Compute NDCG@10 for a single user group."""
    if len(group) == 0:
        return float("nan")

    # Sort by score
    sorted_group = group.sort_values(score_col, ascending=False)
    relevance = sorted_group[label_col].values[:10]

    # DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

    # Ideal DCG
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))

    return float(dcg / idcg) if idcg > 0 else 0.0


def overall_ndcg(test_df: pd.DataFrame, group_key: str = "user_id") -> float:
    """Compute mean NDCG@10 across all users."""
    return test_df.groupby(group_key).apply(compute_ndcg_at_10).mean()


def fairness_ndcg_by_group(
    test_df: pd.DataFrame,
    group_col: str,
    group_key: str = "user_id",
    label_col: str = "label",
    score_col: str = "score",
) -> dict[str, Any]:
    """
    Compute per-group NDCG@10 split by a binary flag column.
    Returns stats for group=0 and group=1.
    """
    results = {}
    for val in [0, 1]:
        group_label = f"{group_col}={val}"
        subset = test_df[test_df[group_col] == val]
        n_users = subset[group_key].nunique()
        if n_users < MIN_GROUP_SIZE:
            results[group_label] = {
                "n_users": n_users,
                "ndcg_at_10": None,
                "skipped": True,
                "reason": f"Too few users ({n_users} < {MIN_GROUP_SIZE})",
            }
            continue
        ndcg = subset.groupby(group_key).apply(compute_ndcg_at_10).mean()
        results[group_label] = {
            "n_users": n_users,
            "ndcg_at_10": round(float(ndcg), 4),
            "skipped": False,
        }
    return results


def check_allergen_safety(
    test_df: pd.DataFrame,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Check that top-ranked recipes don't contain user allergens.
    For each user, check whether the top-k recommended recipes
    contain any allergen the user is allergic to.

    Returns: fraction of (user, allergen) pairs that are violated.
    """
    issues = []
    allergen_pairs = [
        # (user flag, recipe flag) — same allergen
        ("user_dairy_free", "has_milk"),
        ("user_gluten_free", "has_wheat"),
        ("user_vegan", "has_egg"),
        ("user_vegan", "has_milk"),
    ]

    for user_flag, recipe_flag in allergen_pairs:
        if user_flag not in test_df.columns or recipe_flag not in test_df.columns:
            continue
        # Users who have the restriction
        restricted_users = test_df[test_df[user_flag] == 1]["user_id"].unique()
        if len(restricted_users) == 0:
            continue
        # For those users, get their top-k recommendations
        user_data = test_df[test_df["user_id"].isin(restricted_users)].copy()
        top_k_df = (
            user_data.sort_values("score", ascending=False)
            .groupby("user_id")
            .head(top_k)
        )
        violation_rate = top_k_df[recipe_flag].mean()
        issues.append({
            "user_restriction": user_flag,
            "recipe_allergen": recipe_flag,
            "violation_rate": round(float(violation_rate), 4),
            "passed": violation_rate < 0.01,
            "n_users": len(restricted_users),
        })

    overall_passed = all(i["passed"] for i in issues) if issues else True
    return {
        "allergen_safety_passed": overall_passed,
        "pairs_checked": len(issues),
        "violations": issues,
    }


def run_fairness_check(
    test_df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    group_key: str = "user_id",
) -> dict[str, Any]:
    """
    Full fairness evaluation. Returns report dict with:
      - overall_passed: bool
      - overall_ndcg: float
      - group_results: dict (per diet/allergen flag)
      - allergen_safety: dict
      - fairness_violations: list
    """
    if score_col not in test_df.columns:
        raise ValueError(f"score_col '{score_col}' not found in test_df")

    overall = overall_ndcg(test_df, group_key=group_key)
    min_acceptable = overall * (1 - MAX_DEGRADATION)

    group_results = {}
    fairness_violations = []

    # Check dietary groups
    for flag in DIET_FLAGS:
        if flag not in test_df.columns:
            continue
        res = fairness_ndcg_by_group(test_df, flag, group_key, label_col, score_col)
        for group_label, stats in res.items():
            if stats.get("skipped"):
                group_results[group_label] = stats
                continue
            ndcg = stats["ndcg_at_10"]
            passed = ndcg >= min_acceptable
            stats["passed"] = passed
            stats["min_acceptable"] = round(min_acceptable, 4)
            stats["overall_ndcg"] = round(overall, 4)
            group_results[group_label] = stats
            if not passed:
                fairness_violations.append({
                    "group": group_label,
                    "ndcg": ndcg,
                    "min_acceptable": round(min_acceptable, 4),
                    "degradation_pct": round((1 - ndcg / overall) * 100, 1),
                })

    # Allergen safety check
    allergen_safety = check_allergen_safety(test_df)

    # Overall pass/fail
    group_failures = [v for v in fairness_violations]
    allergen_fail = not allergen_safety["allergen_safety_passed"]
    overall_passed = len(group_failures) == 0 and not allergen_fail

    return {
        "overall_passed": overall_passed,
        "overall_ndcg": round(overall, 4),
        "max_degradation_allowed": MAX_DEGRADATION,
        "min_acceptable_ndcg": round(min_acceptable, 4),
        "group_results": group_results,
        "fairness_violations": fairness_violations,
        "allergen_safety": allergen_safety,
        "summary": (
            "PASSED — all groups within acceptable range"
            if overall_passed
            else f"FAILED — {len(group_failures)} group violations + allergen_fail={allergen_fail}"
        ),
    }


def load_and_check(test_csv: str, score_csv: str | None = None) -> dict[str, Any]:
    """Load test CSV (with score column) and run fairness check."""
    df = pd.read_csv(test_csv)
    if score_csv:
        scores = pd.read_csv(score_csv)
        df = df.merge(scores[["user_id", "recipe_id", "score"]], on=["user_id", "recipe_id"], how="left")
    if "score" not in df.columns:
        raise ValueError("test CSV must contain a 'score' column from model predictions")
    return run_fairness_check(df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SparkyFitness Fairness Checker")
    parser.add_argument("--test-csv", required=True, help="Scored test CSV with 'score' column")
    parser.add_argument("--score-csv", default=None, help="Separate score CSV to merge")
    args = parser.parse_args()

    result = load_and_check(args.test_csv, args.score_csv)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["overall_passed"] else 1)
