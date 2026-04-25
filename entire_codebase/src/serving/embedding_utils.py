from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from .feature_contract import EMBEDDING_COMPONENTS, FEATURE_COLUMNS


def embedding_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(1, EMBEDDING_COMPONENTS + 1)]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(np.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def default_feature_value(name: str) -> Any:
    return "unknown" if name == "cuisine" else 0.0


def ensure_feature_columns(instance: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(instance)
    for col in FEATURE_COLUMNS:
        enriched.setdefault(col, default_feature_value(col))
    return enriched


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(sklearn_cosine_similarity([left], [right])[0, 0])


def pad_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] >= EMBEDDING_COMPONENTS:
        return matrix[:, :EMBEDDING_COMPONENTS]
    return np.pad(matrix, ((0, 0), (0, EMBEDDING_COMPONENTS - matrix.shape[1])))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
