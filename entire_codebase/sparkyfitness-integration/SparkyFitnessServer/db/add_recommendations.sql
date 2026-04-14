-- =============================================================================
-- Migration: Personalized Recipe Recommendations
-- Apply to SparkyFitness PostgreSQL database.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- recommendation_cache
-- Stores ML-generated recommendations per user (one batch per request).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommendation_cache (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           TEXT NOT NULL,
    meal_id           TEXT NOT NULL,
    score             DOUBLE PRECISION NOT NULL,
    model_version     TEXT NOT NULL DEFAULT 'unknown',
    feature_snapshot  JSONB,          -- optional: store feature vector for debugging
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at        TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 day')
);

CREATE INDEX IF NOT EXISTS idx_rec_cache_user_id
    ON recommendation_cache (user_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- recommendation_interactions
-- Tracks what users did with recommendations (feedback loop).
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
-- Row-Level Security — users can only see their own rows
-- (Append these to rls_policies.sql as well)
-- ---------------------------------------------------------------------------
ALTER TABLE recommendation_cache        ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommendation_interactions ENABLE ROW LEVEL SECURITY;

-- Replace 'auth.uid()' with the expression used elsewhere in rls_policies.sql
-- (SparkyFitness uses better-auth; check existing policies for the exact expression).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'recommendation_cache' AND policyname = 'user_own_recommendations'
    ) THEN
        EXECUTE $pol$
            CREATE POLICY user_own_recommendations ON recommendation_cache
                USING (user_id = current_setting('app.current_user_id', true))
        $pol$;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'recommendation_interactions' AND policyname = 'user_own_interactions'
    ) THEN
        EXECUTE $pol$
            CREATE POLICY user_own_interactions ON recommendation_interactions
                USING (user_id = current_setting('app.current_user_id', true))
        $pol$;
    END IF;
END$$;
