"""MLflow Model Registry loader with fallback to local file system.

Responsibilities:
  - Load Production model from MLflow Registry
  - Fall back to local model file if Registry is unavailable
  - Poll for new Production versions (used by app_production.py)
  - Report source of the currently loaded model for transparency
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import mlflow
import xgboost as xgb
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(
        self,
        model_name: str = "sparky-ranker",
        fallback_path: str = "/models/xgboost_ranker.json",
        tracking_uri: str = "http://localhost:5000",
    ):
        self.model_name = model_name
        self.fallback_path = fallback_path
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

    def _get_client(self) -> MlflowClient:
        return MlflowClient(tracking_uri=self.tracking_uri)

    def get_latest_production_version(self) -> str | None:
        """Return the latest Production version string, or None."""
        try:
            client = self._get_client()
            versions = client.get_latest_versions(self.model_name, stages=["Production"])
            if versions:
                return versions[0].version
        except Exception as exc:
            logger.warning("Could not fetch production version from Registry: %s", exc)
        return None

    def load_production_model(self) -> tuple[xgb.Booster, str, str]:
        """
        Load the Production model.

        Returns (booster, version_str, source) where source is one of:
          "mlflow_registry" | "local_fallback"
        """
        version = self.get_latest_production_version()
        if version is not None:
            try:
                model_uri = f"models:/{self.model_name}/Production"
                logger.info("Loading model from MLflow Registry: %s", model_uri)
                booster = self._load_from_registry(model_uri)
                logger.info("Loaded model version %s from MLflow Registry", version)
                return booster, f"v{version}", "mlflow_registry"
            except Exception as exc:
                logger.warning(
                    "Failed to load from MLflow Registry (v%s): %s. Falling back to local file.",
                    version,
                    exc,
                )

        # Fallback to local file
        return self._load_from_local(), "local", "local_fallback"

    def _load_from_registry(self, model_uri: str) -> xgb.Booster:
        """Load model via mlflow.xgboost — handles .xgb/.ubj/.json formats."""
        import mlflow.xgboost
        return mlflow.xgboost.load_model(model_uri)

    def _load_from_local(self) -> xgb.Booster:
        """Load model from local filesystem path."""
        if not Path(self.fallback_path).exists():
            raise RuntimeError(
                f"Neither MLflow Registry nor fallback path ({self.fallback_path}) available"
            )
        logger.info("Loading model from local fallback: %s", self.fallback_path)
        booster = xgb.Booster()
        booster.load_model(self.fallback_path)
        return booster

    @staticmethod
    def _find_model_file(directory: Path) -> Path:
        """Find the XGBoost model file inside a downloaded MLflow artifact directory."""
        # Try common locations
        for pattern in ["model/model.json", "model/*.json", "*.json"]:
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]
        # Last resort: any file
        files = [f for f in directory.rglob("*") if f.is_file()]
        if files:
            return files[0]
        raise RuntimeError(f"No model file found in {directory}")
