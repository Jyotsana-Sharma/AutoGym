from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .feature_contract import EMBEDDING_COMPONENTS
from .embedding_utils import (
    cosine_similarity,
    embedding_columns,
    ensure_feature_columns,
    normalize_text,
    pad_embedding_matrix,
    safe_float,
)

logger = logging.getLogger(__name__)

MIN_HISTORY_ITEMS = 3
INGREDIENT_EMBEDDING_COLUMNS = embedding_columns("ingredient_emb")
TEXT_EMBEDDING_COLUMNS = embedding_columns("text_emb")
HISTORY_INGREDIENT_EMBEDDING_COLUMNS = embedding_columns("history_ingredient_emb")
HISTORY_TEXT_EMBEDDING_COLUMNS = embedding_columns("history_text_emb")


class EmbeddingPreprocessor:
    def __init__(self, ingredient_pipeline: Any | None, text_pipeline: Any | None):
        self.ingredient_pipeline = ingredient_pipeline
        self.text_pipeline = text_pipeline

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path | None) -> "EmbeddingPreprocessor | None":
        if not artifact_dir:
            return None
        root = Path(artifact_dir)
        ingredient_path = root / "ingredient_pipeline.pkl"
        text_path = root / "text_pipeline.pkl"
        if not ingredient_path.exists() or not text_path.exists():
            return None
        try:
            ingredient_pipeline = joblib.load(ingredient_path)
            text_pipeline = joblib.load(text_path)
            return cls(ingredient_pipeline, text_pipeline)
        except Exception as exc:
            logger.warning("Embedding artifact load failed: %s", exc)
            return None

    def enrich_instances(self, instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ingredient_candidate_cache: dict[str, np.ndarray] = {}
        text_candidate_cache: dict[str, np.ndarray] = {}
        ingredient_history_cache: dict[str, np.ndarray] = {}
        text_history_cache: dict[str, np.ndarray] = {}
        enriched_instances: list[dict[str, Any]] = []

        for raw_instance in instances:
            instance = dict(raw_instance)
            ingredient_doc = normalize_text(instance.get("candidate_ingredient_text"))
            text_doc = normalize_text(
                " ".join(
                    filter(
                        None,
                        [
                            instance.get("candidate_name_text"),
                            instance.get("candidate_description_text"),
                        ],
                    )
                )
            )

            ingredient_vector = ingredient_candidate_cache.setdefault(
                ingredient_doc,
                self._single_document_vector(ingredient_doc, self.ingredient_pipeline),
            )
            text_vector = text_candidate_cache.setdefault(
                text_doc,
                self._single_document_vector(text_doc, self.text_pipeline),
            )

            history_ingredient_key = str(instance.get("user_history_ingredient_text", ""))
            history_text_key = str(instance.get("user_history_name_text", ""))
            history_ingredient_docs = self._parse_history_documents(history_ingredient_key)
            history_text_docs = self._parse_history_documents(history_text_key)
            sufficient_history = max(
                len(history_ingredient_docs),
                len(history_text_docs),
            ) >= MIN_HISTORY_ITEMS

            if sufficient_history:
                history_ingredient_vector = ingredient_history_cache.setdefault(
                    history_ingredient_key,
                    self._history_vector_from_docs(
                        history_ingredient_docs,
                        self.ingredient_pipeline,
                    ),
                )
                history_text_vector = text_history_cache.setdefault(
                    history_text_key,
                    self._history_vector_from_docs(
                        history_text_docs,
                        self.text_pipeline,
                    ),
                )
            else:
                history_ingredient_vector = np.zeros(EMBEDDING_COMPONENTS, dtype=float)
                history_text_vector = np.zeros(EMBEDDING_COMPONENTS, dtype=float)

            self._assign_vector(instance, INGREDIENT_EMBEDDING_COLUMNS, ingredient_vector)
            self._assign_vector(instance, TEXT_EMBEDDING_COLUMNS, text_vector)
            self._assign_vector(
                instance, HISTORY_INGREDIENT_EMBEDDING_COLUMNS, history_ingredient_vector
            )
            self._assign_vector(instance, HISTORY_TEXT_EMBEDDING_COLUMNS, history_text_vector)
            instance["ingredient_embedding_cosine"] = cosine_similarity(
                ingredient_vector, history_ingredient_vector
            )
            instance["text_embedding_cosine"] = cosine_similarity(
                text_vector, history_text_vector
            )
            instance["history_macro_similarity"] = (
                self._macro_similarity(instance) if sufficient_history else 0.0
            )
            enriched_instances.append(ensure_feature_columns(instance))

        return enriched_instances

    def _assign_vector(self, instance: dict[str, Any], columns: list[str], vector: np.ndarray) -> None:
        for idx, column in enumerate(columns):
            instance[column] = float(vector[idx]) if idx < len(vector) else 0.0

    def _pipeline_transform(self, documents: list[str], pipeline: Any | None) -> np.ndarray:
        if pipeline is None:
            return np.zeros((len(documents), EMBEDDING_COMPONENTS), dtype=float)

        normalized = [normalize_text(doc) for doc in documents]
        if not any(normalized):
            return np.zeros((len(documents), EMBEDDING_COMPONENTS), dtype=float)

        matrix = pipeline.transform(normalized)
        return pad_embedding_matrix(matrix)

    def _single_document_vector(self, document: str, pipeline: Any | None) -> np.ndarray:
        matrix = self._pipeline_transform([document], pipeline)
        return matrix[0] if len(matrix) else np.zeros(EMBEDDING_COMPONENTS, dtype=float)

    def _history_vector_from_docs(
        self,
        docs: list[str],
        pipeline: Any | None,
    ) -> np.ndarray:
        if len(docs) < MIN_HISTORY_ITEMS:
            return np.zeros(EMBEDDING_COMPONENTS, dtype=float)
        matrix = self._pipeline_transform(docs, pipeline)
        if len(matrix) == 0:
            return np.zeros(EMBEDDING_COMPONENTS, dtype=float)
        return matrix.mean(axis=0)

    def _parse_history_documents(self, raw_value: str) -> list[str]:
        if not raw_value:
            return []
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [normalize_text(item) for item in parsed if normalize_text(item)]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        normalized = normalize_text(raw_value)
        return [normalized] if normalized else []

    def _macro_similarity(self, instance: dict[str, Any]) -> float:
        protein_g = safe_float(instance.get("protein_g"))
        carbohydrate_g = safe_float(instance.get("carbohydrate_g"))
        total_fat_g = safe_float(instance.get("total_fat_g"))
        candidate = np.array(
            [
                (protein_g * 4.0) + (carbohydrate_g * 4.0) + (total_fat_g * 9.0),
                protein_g,
                carbohydrate_g,
                total_fat_g,
            ],
            dtype=float,
        )
        history = np.array(
            [
                safe_float(instance.get("user_history_avg_calories")),
                safe_float(instance.get("user_history_avg_protein_g")),
                safe_float(instance.get("user_history_avg_carbohydrate_g")),
                safe_float(instance.get("user_history_avg_total_fat_g")),
            ],
            dtype=float,
        )
        return cosine_similarity(candidate, history)
