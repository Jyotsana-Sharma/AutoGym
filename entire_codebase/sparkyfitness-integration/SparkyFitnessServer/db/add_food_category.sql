-- Migration: Add food_category column for category-aware recommendation filtering
-- Safe to re-run (idempotent).

ALTER TABLE foods
  ADD COLUMN IF NOT EXISTS food_category TEXT DEFAULT 'other';

-- Partial indexes — only index rows that have a real category assigned
CREATE INDEX IF NOT EXISTS idx_foods_category
    ON foods(food_category)
    WHERE food_category IS NOT NULL AND food_category <> 'other';

CREATE INDEX IF NOT EXISTS idx_foods_category_public
    ON foods(food_category, shared_with_public)
    WHERE shared_with_public = TRUE;

COMMENT ON COLUMN foods.food_category IS
  'Broad food type used for category-aware recommendation pre-filtering. '
  'Values: fruit | vegetable | protein | dairy | grain | legume | snack | dessert | beverage | other';
