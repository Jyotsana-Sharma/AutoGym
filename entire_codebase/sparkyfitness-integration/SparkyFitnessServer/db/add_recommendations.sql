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

DROP POLICY IF EXISTS user_own_recommendations ON recommendation_cache;
CREATE POLICY user_own_recommendations ON recommendation_cache
    USING (true)
    WITH CHECK (true);

DROP POLICY IF EXISTS user_own_interactions ON recommendation_interactions;
CREATE POLICY user_own_interactions ON recommendation_interactions
    USING (true)
    WITH CHECK (true);

-- ---------------------------------------------------------------------------
-- Grants for the app user
-- (sparkyapp is created by SparkyFitness server on first run)
-- ---------------------------------------------------------------------------
DO $$
DECLARE
    app_role TEXT;
BEGIN
    FOREACH app_role IN ARRAY ARRAY['sparkyapp', 'sparky_app'] LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = app_role) THEN
            EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON recommendation_cache TO %I', app_role);
            EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON recommendation_interactions TO %I', app_role);
        END IF;
    END LOOP;
END$$;
