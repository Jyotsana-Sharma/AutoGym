-- Additional tables for serving feedback loop
-- Runs after init_db.sql

CREATE TABLE IF NOT EXISTS prediction_log (
    id              BIGSERIAL PRIMARY KEY,
    request_id      TEXT NOT NULL,
    recommendation_id TEXT,
    model_version   TEXT NOT NULL,
    user_id         INTEGER NOT NULL,
    recipe_id       INTEGER NOT NULL,
    score           FLOAT NOT NULL,
    rank            INTEGER NOT NULL,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prediction_log_user ON prediction_log(user_id);
CREATE INDEX IF NOT EXISTS idx_prediction_log_ts   ON prediction_log(predicted_at);
CREATE INDEX IF NOT EXISTS idx_prediction_log_ver  ON prediction_log(model_version);
CREATE INDEX IF NOT EXISTS idx_prediction_log_rec  ON prediction_log(recommendation_id);

CREATE TABLE IF NOT EXISTS user_feedback (
    id              BIGSERIAL PRIMARY KEY,
    request_id      TEXT,
    recommendation_id TEXT,
    user_id         INTEGER NOT NULL,
    recipe_id       INTEGER NOT NULL,
    rating          FLOAT,
    action          TEXT,    -- 'view', 'viewed', 'cook', 'rate', 'skip', 'served', 'logged', ...
    feedback_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_feedback_user ON user_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_ts   ON user_feedback(feedback_at);
CREATE INDEX IF NOT EXISTS idx_user_feedback_rec  ON user_feedback(recommendation_id);

CREATE TABLE IF NOT EXISTS inference_features (
    id              BIGSERIAL PRIMARY KEY,
    request_id      TEXT NOT NULL,
    recommendation_id TEXT,
    model_version   TEXT NOT NULL,
    user_id         INTEGER NOT NULL,
    recipe_id       INTEGER NOT NULL,
    features        JSONB NOT NULL,
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inf_features_ts  ON inference_features(captured_at);
CREATE INDEX IF NOT EXISTS idx_inf_features_ver ON inference_features(model_version);
CREATE INDEX IF NOT EXISTS idx_inf_features_rec ON inference_features(recommendation_id);

-- Drift log: records of drift detection runs
CREATE TABLE IF NOT EXISTS drift_log (
    id              BIGSERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    feature_name    TEXT NOT NULL,
    ks_statistic    FLOAT NOT NULL,
    p_value         FLOAT NOT NULL,
    drift_detected  BOOLEAN NOT NULL,
    threshold       FLOAT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_drift_log_ts ON drift_log(run_at);

-- Retraining log: records of triggered retraining runs
CREATE TABLE IF NOT EXISTS retraining_log (
    id              BIGSERIAL PRIMARY KEY,
    triggered_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger_reason  TEXT NOT NULL,  -- 'drift', 'scheduled', 'manual', 'ci'
    run_id          TEXT,
    model_version   TEXT,
    ndcg_at_10      FLOAT,
    registered      BOOLEAN,
    promoted        BOOLEAN,
    duration_sec    FLOAT,
    status          TEXT NOT NULL DEFAULT 'pending'  -- 'pending','running','completed','failed'
);
