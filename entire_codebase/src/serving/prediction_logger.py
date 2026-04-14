"""Prediction logger — writes serving predictions and user feedback to PostgreSQL.

This closes the feedback loop:
  serving predictions → PostgreSQL → batch_pipeline → retraining

Tables written:
  prediction_log       — every /predict request + ranked outputs
  user_feedback        — explicit ratings / clicks from /feedback endpoint
  inference_features   — raw feature values for drift monitoring

The data_generator.py in sparky-data-pipeline reads user_interactions;
this logger also inserts synthetic "interaction" rows so the feedback
loop works even before real users provide explicit ratings.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

try:
    import asyncpg
    _ASYNCPG_AVAILABLE = True
except ImportError:
    _ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not installed — prediction logging disabled")


class PredictionLogger:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._pool = None

    async def connect(self):
        if not _ASYNCPG_AVAILABLE or not self.database_url:
            return
        try:
            self._pool = await asyncpg.create_pool(self.database_url, min_size=2, max_size=10)
            await self._ensure_tables()
            logger.info("PredictionLogger connected to PostgreSQL")
        except Exception as exc:
            logger.warning("PredictionLogger could not connect to DB: %s", exc)
            self._pool = None

    async def close(self):
        if self._pool:
            await self._pool.close()

    async def _ensure_tables(self):
        """Create logging tables if they don't exist."""
        if not self._pool:
            return
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_log (
                    id              BIGSERIAL PRIMARY KEY,
                    request_id      TEXT NOT NULL,
                    model_version   TEXT NOT NULL,
                    user_id         INTEGER NOT NULL,
                    recipe_id       INTEGER NOT NULL,
                    score           FLOAT NOT NULL,
                    rank            INTEGER NOT NULL,
                    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_prediction_log_user ON prediction_log(user_id);
                CREATE INDEX IF NOT EXISTS idx_prediction_log_ts ON prediction_log(predicted_at);

                CREATE TABLE IF NOT EXISTS user_feedback (
                    id              BIGSERIAL PRIMARY KEY,
                    request_id      TEXT,
                    user_id         INTEGER NOT NULL,
                    recipe_id       INTEGER NOT NULL,
                    rating          FLOAT,
                    action          TEXT,
                    feedback_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_user_feedback_user ON user_feedback(user_id);

                CREATE TABLE IF NOT EXISTS inference_features (
                    id              BIGSERIAL PRIMARY KEY,
                    request_id      TEXT NOT NULL,
                    model_version   TEXT NOT NULL,
                    user_id         INTEGER NOT NULL,
                    recipe_id       INTEGER NOT NULL,
                    features        JSONB NOT NULL,
                    captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_inf_features_ts ON inference_features(captured_at);
            """)

    async def log_batch(
        self,
        request_id: str,
        model_version: str,
        predictions: list[Any],
        timestamp: str,
        features: list[dict] | None = None,
    ):
        """Log a batch of predictions to prediction_log."""
        if not self._pool:
            return
        rows = [
            (request_id, model_version, p.user_id, p.recipe_id, p.score, p.rank)
            for p in predictions
        ]
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    """INSERT INTO prediction_log
                       (request_id, model_version, user_id, recipe_id, score, rank)
                       VALUES ($1, $2, $3, $4, $5, $6)""",
                    rows,
                )
                # Also write top-ranked prediction as a synthetic user interaction
                # so the retraining data pipeline can use it
                top = [p for p in predictions if p.rank == 1]
                if top:
                    p = top[0]
                    await conn.execute(
                        """INSERT INTO user_interactions
                           (user_id, recipe_id, rating, action, created_at)
                           VALUES ($1, $2, NULL, 'served', NOW())
                           ON CONFLICT DO NOTHING""",
                        p.user_id, p.recipe_id,
                    )
        except Exception as exc:
            logger.warning("Failed to log predictions: %s", exc)

    async def log_feedback(self, payload: dict):
        """
        Log explicit user feedback (rating, click) from the /feedback endpoint.
        Also updates user_interactions for the retraining pipeline.
        """
        if not self._pool:
            return
        request_id = payload.get("request_id", "")
        user_id = int(payload.get("user_id", 0))
        recipe_id = int(payload.get("recipe_id", 0))
        rating = payload.get("rating")
        action = payload.get("action", "rate")

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO user_feedback
                       (request_id, user_id, recipe_id, rating, action)
                       VALUES ($1, $2, $3, $4, $5)""",
                    request_id, user_id, recipe_id,
                    float(rating) if rating is not None else None,
                    action,
                )
                # Write back to user_interactions for retraining loop
                if rating is not None:
                    await conn.execute(
                        """INSERT INTO user_interactions
                           (user_id, recipe_id, rating, action, created_at)
                           VALUES ($1, $2, $3, $4, NOW())
                           ON CONFLICT DO NOTHING""",
                        user_id, recipe_id, float(rating), action,
                    )
        except Exception as exc:
            logger.warning("Failed to log feedback: %s", exc)

    async def log_features(
        self,
        request_id: str,
        model_version: str,
        instances: list[dict],
    ):
        """
        Log raw inference features for drift monitoring.
        Called by drift_monitor.py to build production feature distribution.
        """
        if not self._pool:
            return
        rows = [
            (
                request_id,
                model_version,
                int(inst.get("user_id", 0)),
                int(inst.get("recipe_id", 0)),
                json.dumps(inst),
            )
            for inst in instances
        ]
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    """INSERT INTO inference_features
                       (request_id, model_version, user_id, recipe_id, features)
                       VALUES ($1, $2, $3, $4, $5::jsonb)""",
                    rows,
                )
        except Exception as exc:
            logger.warning("Failed to log features: %s", exc)
