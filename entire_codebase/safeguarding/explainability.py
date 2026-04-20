"""Explainability module — generates SHAP-based explanations for predictions.

Supports two modes:
  1. Global explanation — feature importance across the entire test set
     (run once after training, logged to MLflow as artifact)
  2. Per-request explanation — SHAP values for a specific prediction
     (served via /explain endpoint in the serving API)

Implements the explainability and transparency pillars of the safeguarding plan:
  - Users can request "why was this recipe recommended?"
  - Auditors can inspect global feature importance
  - Regulators can review fairness of feature usage (e.g., allergen flags used correctly)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("shap not installed — explainability features limited")

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


FEATURE_COLUMNS = [
    "minutes", "n_ingredients", "n_steps", "avg_rating", "n_reviews",
    "cuisine",
    "calories", "total_fat", "sugar", "sodium", "protein",
    "saturated_fat", "carbohydrate",
    "total_fat_g", "sugar_g", "sodium_g", "protein_g",
    "saturated_fat_g", "carbohydrate_g",
    "has_egg", "has_fish", "has_milk", "has_nuts", "has_peanut",
    "has_sesame", "has_shellfish", "has_soy", "has_wheat",
    "daily_calorie_target", "protein_target_g", "carbs_target_g", "fat_target_g",
    "user_vegetarian", "user_vegan", "user_gluten_free", "user_dairy_free",
    "user_low_sodium", "user_low_fat",
    "history_pc1", "history_pc2", "history_pc3", "history_pc4",
    "history_pc5", "history_pc6",
]

# Human-readable feature names for display
FEATURE_DISPLAY_NAMES = {
    "minutes": "Cooking time (min)",
    "n_ingredients": "Number of ingredients",
    "n_steps": "Number of steps",
    "avg_rating": "Average community rating",
    "n_reviews": "Number of reviews",
    "cuisine": "Cuisine type",
    "calories": "Calories",
    "protein": "Protein content",
    "carbohydrate": "Carbohydrates",
    "total_fat": "Total fat",
    "sugar": "Sugar",
    "sodium": "Sodium",
    "daily_calorie_target": "Your calorie target",
    "protein_target_g": "Your protein target",
    "carbs_target_g": "Your carbs target",
    "fat_target_g": "Your fat target",
    "user_vegetarian": "Vegetarian preference",
    "user_vegan": "Vegan preference",
    "user_gluten_free": "Gluten-free preference",
    "user_dairy_free": "Dairy-free preference",
    "user_low_sodium": "Low sodium preference",
    "user_low_fat": "Low fat preference",
    "has_egg": "Contains eggs",
    "has_fish": "Contains fish",
    "has_milk": "Contains dairy",
    "has_nuts": "Contains nuts",
    "has_wheat": "Contains gluten",
    "history_pc1": "Your cooking history (style 1)",
    "history_pc2": "Your cooking history (style 2)",
    "history_pc3": "Your cooking history (style 3)",
}


class Explainer:
    def __init__(self, model, feature_columns: list[str] = FEATURE_COLUMNS):
        self.model = model
        self.feature_columns = feature_columns
        self._shap_explainer = None

    def _get_shap_explainer(self, background_data: np.ndarray | None = None):
        if not _SHAP_AVAILABLE:
            raise ImportError("shap not installed. Run: pip install shap")
        if self._shap_explainer is None:
            if _XGB_AVAILABLE and isinstance(self.model, xgb.Booster):
                self._shap_explainer = shap.TreeExplainer(self.model)
            elif background_data is not None:
                self._shap_explainer = shap.KernelExplainer(
                    self.model.predict, background_data
                )
            else:
                raise ValueError("Need background_data for non-tree models")
        return self._shap_explainer

    def global_feature_importance(
        self,
        test_df: pd.DataFrame,
        max_display: int = 20,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute global SHAP feature importance over a test sample.
        Returns top features sorted by mean |SHAP|.
        Optionally saves plots to output_dir.
        """
        X = test_df[self.feature_columns].values.astype(np.float32)
        sample = X[:min(500, len(X))]  # subsample for speed

        explainer = self._get_shap_explainer()
        shap_values = explainer.shap_values(sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = sorted(
            zip(self.feature_columns, mean_abs_shap.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        result = {
            "feature_importance": [
                {
                    "feature": feat,
                    "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "mean_abs_shap": round(val, 6),
                }
                for feat, val in importance[:max_display]
            ],
            "n_samples": len(sample),
            "total_features": len(self.feature_columns),
        }

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / "global_feature_importance.json", "w") as f:
                json.dump(result, f, indent=2)
            # Save summary plot if matplotlib available
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                shap.summary_plot(
                    shap_values,
                    sample,
                    feature_names=self.feature_columns,
                    show=False,
                    max_display=max_display,
                )
                plt.savefig(
                    Path(output_dir) / "shap_summary.png",
                    dpi=100, bbox_inches="tight",
                )
                plt.close()
                result["plot_path"] = str(Path(output_dir) / "shap_summary.png")
            except Exception as exc:
                logger.warning("Could not generate SHAP plot: %s", exc)

        return result

    def explain_prediction(
        self,
        instance: dict,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Explain a single prediction for a user-facing "why was this recommended?" feature.

        Returns top contributing features in user-friendly language.
        """
        df = pd.DataFrame([instance])
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_columns].values.astype(np.float32)

        try:
            explainer = self._get_shap_explainer()
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            sv = shap_values[0]

            contributions = sorted(
                [
                    {
                        "feature": feat,
                        "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                        "value": float(df[feat].iloc[0]),
                        "shap_contribution": round(float(sv[i]), 4),
                        "direction": "positive" if sv[i] > 0 else "negative",
                    }
                    for i, feat in enumerate(self.feature_columns)
                ],
                key=lambda x: abs(x["shap_contribution"]),
                reverse=True,
            )[:top_k]

            # Generate human-readable explanation
            positive = [c for c in contributions if c["direction"] == "positive"][:3]
            negative = [c for c in contributions if c["direction"] == "negative"][:2]

            explanation_text = "This recipe was recommended because:"
            if positive:
                names = [c["display_name"] for c in positive]
                explanation_text += f" {', '.join(names)} match your preferences."
            if negative:
                names = [c["display_name"] for c in negative]
                explanation_text += f" (Minor factors against: {', '.join(names)}.)"

            return {
                "explanation": explanation_text,
                "top_contributing_features": contributions,
                "method": "SHAP TreeExplainer",
            }
        except Exception as exc:
            logger.warning("SHAP explanation failed: %s — using fallback", exc)
            return self._fallback_explanation(instance)

    def _fallback_explanation(self, instance: dict) -> dict[str, Any]:
        """Rule-based fallback when SHAP is unavailable."""
        reasons = []
        if instance.get("avg_rating", 0) >= 4.0:
            reasons.append("high community rating")
        if instance.get("calories", 9999) <= instance.get("daily_calorie_target", 9999) * 1.1:
            reasons.append("fits your calorie target")
        if instance.get("user_vegetarian") and not instance.get("has_meat", 0):
            reasons.append("matches your vegetarian preference")
        explanation = "This recipe was recommended because: " + (
            ", ".join(reasons) if reasons else "it matches your overall preferences."
        )
        return {
            "explanation": explanation,
            "top_contributing_features": [],
            "method": "rule_based_fallback",
        }


def run_global_explainability(
    model_path: str,
    test_csv: str,
    output_dir: str = "/tmp/explainability",
) -> dict[str, Any]:
    """
    Load model and test CSV, compute global feature importance, save artifacts.
    Called post-training before MLflow registration.
    """
    if not _XGB_AVAILABLE:
        return {"error": "xgboost not installed"}
    booster = xgb.Booster()
    booster.load_model(model_path)

    df = pd.read_csv(test_csv)
    explainer = Explainer(booster)
    result = explainer.global_feature_importance(df, output_dir=output_dir)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SparkyFitness Explainability")
    parser.add_argument("--model", required=True, help="Path to XGBoost model JSON")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", default="/tmp/explainability")
    args = parser.parse_args()

    result = run_global_explainability(args.model, args.test_csv, args.output_dir)
    print(json.dumps(result, indent=2, default=str))
