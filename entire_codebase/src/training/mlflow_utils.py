from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import yaml


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def flatten_params(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_params(value, flat_key))
        else:
            flat[flat_key] = value
    return flat


def log_run_metadata(config: dict[str, Any]) -> None:
    mlflow.log_params({key: str(value) for key, value in flatten_params(config).items()})
    mlflow.log_param("tracking_uri", mlflow.get_tracking_uri())
