CREATE TABLE IF NOT EXISTS recipes (
    recipe_id       INTEGER PRIMARY KEY,
    name            TEXT,
    minutes         INTEGER,
    cuisine         VARCHAR(50),
    calories        REAL,
    protein_g       REAL,
    carbohydrate_g  REAL,
    total_fat_g     REAL,
    sugar_g         REAL,
    sodium_mg       REAL,
    saturated_fat_g REAL,
    n_ingredients   INTEGER,
    n_steps         INTEGER,
    avg_rating      REAL,
    n_reviews       INTEGER,
    has_milk        SMALLINT DEFAULT 0,
    has_egg         SMALLINT DEFAULT 0,
    has_nuts        SMALLINT DEFAULT 0,
    has_peanut      SMALLINT DEFAULT 0,
    has_fish        SMALLINT DEFAULT 0,
    has_shellfish   SMALLINT DEFAULT 0,
    has_wheat       SMALLINT DEFAULT 0,
    has_soy         SMALLINT DEFAULT 0,
    has_sesame      SMALLINT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    user_id              INTEGER PRIMARY KEY,
    dietary_restrictions JSONB DEFAULT '{}',
    macro_targets        JSONB DEFAULT '{}',
    allergens            JSONB DEFAULT '[]',
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_interactions (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    recipe_id   INTEGER NOT NULL,
    rating      SMALLINT CHECK (rating BETWEEN 0 AND 5),
    action      VARCHAR(20) DEFAULT 'view'
                CHECK (action IN ('view','cook','rate','skip','served','logged','dismissed','saved')),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inter_user   ON user_interactions (user_id);
CREATE INDEX IF NOT EXISTS idx_inter_recipe ON user_interactions (recipe_id);
CREATE INDEX IF NOT EXISTS idx_inter_date   ON user_interactions (created_at);
CREATE INDEX IF NOT EXISTS idx_inter_user_date ON user_interactions (user_id, created_at);

CREATE TABLE IF NOT EXISTS user_features_cache (
    user_id              INTEGER PRIMARY KEY,
    daily_calorie_target REAL,
    protein_target_g     REAL,
    carbs_target_g       REAL,
    fat_target_g         REAL,
    user_vegetarian      SMALLINT DEFAULT 0,
    user_vegan           SMALLINT DEFAULT 0,
    user_gluten_free     SMALLINT DEFAULT 0,
    user_dairy_free      SMALLINT DEFAULT 0,
    user_low_sodium      SMALLINT DEFAULT 0,
    user_low_fat         SMALLINT DEFAULT 0,
    history_pc1 REAL DEFAULT 0, history_pc2 REAL DEFAULT 0,
    history_pc3 REAL DEFAULT 0, history_pc4 REAL DEFAULT 0,
    history_pc5 REAL DEFAULT 0, history_pc6 REAL DEFAULT 0,
    version_id  VARCHAR(32),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

DO $$ BEGIN
  CREATE TRIGGER trg_users_ts BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TRIGGER trg_feat_ts BEFORE UPDATE ON user_features_cache
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
