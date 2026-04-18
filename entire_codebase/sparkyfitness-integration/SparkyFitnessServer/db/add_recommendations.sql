-- =============================================================================
-- Migration: Personalized Recipe Recommendations
-- Run as the database owner (sparky), not the app user (sparkyapp).
-- Safe to re-run (idempotent).
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- recommendation_cache
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommendation_cache (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id        TEXT,
    user_id           TEXT NOT NULL,
    meal_id           TEXT NOT NULL,
    ml_user_id        INTEGER,
    ml_recipe_id      INTEGER,
    score             DOUBLE PRECISION NOT NULL,
    model_version     TEXT NOT NULL DEFAULT 'unknown',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at        TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 day')
);

CREATE INDEX IF NOT EXISTS idx_rec_cache_user_id
    ON recommendation_cache (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_rec_cache_request
    ON recommendation_cache (request_id);

-- ---------------------------------------------------------------------------
-- recommendation_interactions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommendation_interactions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_id UUID NOT NULL REFERENCES recommendation_cache (id) ON DELETE CASCADE,
    user_id           TEXT NOT NULL,
    action            TEXT NOT NULL CHECK (action IN ('viewed','logged','dismissed','saved')),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rec_interactions_user_id
    ON recommendation_interactions (user_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE recommendation_cache        ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendation_interactions ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'recommendation_cache'
          AND policyname = 'user_own_recommendations'
    ) THEN
        EXECUTE $pol$
            CREATE POLICY user_own_recommendations ON recommendation_cache
                USING (user_id = current_setting('app.current_user_id', true))
        $pol$;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'recommendation_interactions'
          AND policyname = 'user_own_interactions'
    ) THEN
        EXECUTE $pol$
            CREATE POLICY user_own_interactions ON recommendation_interactions
                USING (user_id = current_setting('app.current_user_id', true))
        $pol$;
    END IF;
END$$;

-- ---------------------------------------------------------------------------
-- Grants for the app user
-- (sparkyapp is created by SparkyFitness server on first run)
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sparkyapp') THEN
        EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON recommendation_cache TO sparkyapp';
        EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON recommendation_interactions TO sparkyapp';
    END IF;
END$$;
