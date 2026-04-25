"""
import_foodcom_to_sparkyfitness.py  —  Food.com → SparkyFitness Public Foods

Imports a curated, fitness-relevant subset of enriched Food.com recipes into
the SparkyFitness database (foods + food_variants tables) so they appear as
public candidate items in the recommendation pipeline.

Why this matters:
  - The XGBoost model was TRAINED on Food.com data → it ranks these recipes
    with full feature fidelity (all 67 features populated)
  - SparkyFitness app items currently produce ~25 zeroed-out features
  - Importing Food.com recipes gives cold-start users a rich, high-quality
    candidate pool from day one
  - The feedback loop stays intact: users log/save/dismiss these items
    like any other food → feedback flows back to retraining

Usage:
  python scripts/import_foodcom_to_sparkyfitness.py \
      --enriched-csv ./output/enriched_recipes.csv \
      --raw-csv ./data/RAW_recipes.csv \
      --db-url "postgresql://sparky:sparky_fit_pass@localhost:5433/sparkyfitness_db" \
      --dry-run

Docker:
  docker compose run --rm batch-pipeline python scripts/import_foodcom_to_sparkyfitness.py \
      --enriched-csv /output/enriched_recipes.csv \
      --raw-csv /data/RAW_recipes.csv \
      --db-url "$SPARKY_FITNESS_DB_URL"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import psycopg2
    import psycopg2.extras

    HAS_PG = True
except ImportError:
    HAS_PG = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CURATION FILTERS — keep only fitness-relevant, high-quality recipes
# ─────────────────────────────────────────────────────────────
MIN_AVG_RATING = 4.0          # Only well-reviewed recipes
MIN_REVIEWS = 10              # Enough signal to trust the rating
MAX_CALORIES = 800            # Per serving — fitness-appropriate
MIN_PROTEIN_G = 5.0           # Minimum meaningful protein content
MAX_MINUTES = 90              # Reasonable prep time for a meal
MIN_INGREDIENTS = 2           # Not just "water" or "salt"
MAX_INGREDIENTS = 20          # Not absurdly complex

# Allergen keywords mapping (matches build_training_table.py)
ALLERGEN_KEYWORDS = {
    "milk":      ["milk", "butter", "cream", "cheese", "yogurt", "mozzarella",
                  "parmesan", "gouda", "cheddar", "buttermilk", "whey"],
    "egg":       ["egg", "egg yolk", "egg white", "mayonnaise"],
    "nuts":      ["almond", "walnut", "hazelnut", "cashew", "pistachio",
                  "pecan", "macadamia", "brazil nut"],
    "peanut":    ["peanut", "peanut butter"],
    "fish":      ["salmon", "tuna", "cod", "trout", "halibut", "anchovy", "sardine"],
    "shellfish": ["shrimp", "prawn", "lobster", "crab", "scallop",
                  "mussel", "clam", "oyster"],
    "wheat":     ["wheat", "flour", "bread", "pasta", "noodles", "semolina"],
    "soy":       ["soy", "soybean", "tofu", "tempeh", "edamame", "soy sauce"],
    "sesame":    ["sesame", "tahini"],
}

# PDV → absolute grams conversion (2000 kcal reference diet)
PDV_TO_ABS = {
    "total_fat": 78,        # g
    "sugar": 50,            # g
    "sodium": 2.3,          # g (stored as grams in feature contract)
    "protein": 50,          # g
    "saturated_fat": 20,    # g
    "carbohydrate": 275,    # g
}


def stable_uuid(recipe_id: int) -> str:
    """Generate a deterministic UUID-like string from a Food.com recipe_id.

    This ensures idempotency — re-running the import produces the same IDs
    and ON CONFLICT can safely skip duplicates.
    """
    digest = hashlib.sha256(f"foodcom-recipe-{recipe_id}".encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def stable_variant_uuid(recipe_id: int) -> str:
    """Deterministic variant UUID for the default food variant."""
    digest = hashlib.sha256(f"foodcom-variant-{recipe_id}".encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def parse_list_column(value: Any) -> list:
    """Parse a string representation of a Python list."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        import ast
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []


def detect_allergens_for_recipe(ingredients_raw: Any) -> dict[str, bool]:
    """Detect allergens from ingredient list, returning boolean flags."""
    ingredients = parse_list_column(ingredients_raw)
    ingredients_lower = [ing.lower() for ing in ingredients]
    flags = {}
    for allergen, keywords in ALLERGEN_KEYWORDS.items():
        found = False
        for ing in ingredients_lower:
            if any(kw in ing for kw in keywords):
                found = True
                break
        flags[f"has_{allergen}"] = found
    return flags


def build_ingredient_text(ingredients_raw: Any) -> str:
    """Build a space-separated, normalized ingredient string for text search."""
    ingredients = parse_list_column(ingredients_raw)
    return " ".join(
        ing.strip().lower() for ing in ingredients if ing.strip()
    )


def infer_food_category(name: str, ingredient_text: str) -> str:
    """Map a Food.com recipe into the broad categories used by recommendations."""
    text = f"{name} {ingredient_text}".lower()
    category_keywords = [
        ("beverage", ["smoothie", "juice", "tea", "coffee", "shake", "drink"]),
        ("dessert", ["cake", "cookie", "pie", "brownie", "pudding", "dessert", "candy"]),
        (
            "protein",
            [
                "chicken",
                "beef",
                "turkey",
                "pork",
                "fish",
                "salmon",
                "tuna",
                "omelet",
                "omelette",
                "tofu",
                "shrimp",
            ],
        ),
        ("legume", ["bean", "lentil", "chickpea", "hummus", "peas"]),
        ("dairy", ["yogurt", "cheese", "milk", "cottage cheese"]),
        (
            "vegetable",
            ["broccoli", "spinach", "kale", "carrot", "zucchini", "pepper", "salad", "vegetable"],
        ),
        ("fruit", ["apple", "banana", "berry", "strawberry", "blueberry", "orange", "peach", "fruit"]),
        ("grain", ["rice", "pasta", "oat", "bread", "quinoa", "cereal", "noodle"]),
        ("snack", ["snack", "bar", "chips", "popcorn", "cracker"]),
    ]
    for category, keywords in category_keywords:
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def curate_recipes(enriched: pd.DataFrame, raw: pd.DataFrame | None) -> pd.DataFrame:
    """Apply fitness-relevant curation filters to the enriched recipes."""
    logger.info("Starting curation from %d enriched recipes", len(enriched))

    df = enriched.copy()

    # Ensure required columns exist
    required = ["recipe_id", "name", "calories", "avg_rating", "n_reviews", "minutes"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Enriched CSV missing required columns: {missing}")

    # Join raw recipe data for ingredients and description if not already present
    if raw is not None and "ingredients" not in df.columns:
        raw = raw.copy()
        raw.columns = [c.strip().lower() for c in raw.columns]
        if "id" in raw.columns and "recipe_id" not in raw.columns:
            raw = raw.rename(columns={"id": "recipe_id"})
        join_cols = [c for c in ["recipe_id", "ingredients", "description", "steps"]
                     if c in raw.columns and c not in df.columns or c == "recipe_id"]
        df = df.merge(raw[join_cols].drop_duplicates("recipe_id"),
                       on="recipe_id", how="left", suffixes=("", "_raw"))

    # Compute absolute grams if only PDV columns exist
    for pdv_col, factor in PDV_TO_ABS.items():
        gram_col = f"{pdv_col}_g"
        if gram_col not in df.columns and pdv_col in df.columns:
            df[gram_col] = (df[pdv_col] / 100) * factor

    # Apply curation filters
    before = len(df)
    df = df[df["avg_rating"] >= MIN_AVG_RATING]
    logger.info("  After avg_rating >= %.1f: %d (dropped %d)", MIN_AVG_RATING, len(df), before - len(df))

    before = len(df)
    df = df[df["n_reviews"] >= MIN_REVIEWS]
    logger.info("  After n_reviews >= %d: %d (dropped %d)", MIN_REVIEWS, len(df), before - len(df))

    before = len(df)
    df = df[df["calories"] <= MAX_CALORIES]
    logger.info("  After calories <= %d: %d (dropped %d)", MAX_CALORIES, len(df), before - len(df))

    if "protein_g" in df.columns:
        before = len(df)
        df = df[df["protein_g"] >= MIN_PROTEIN_G]
        logger.info("  After protein_g >= %.1f: %d (dropped %d)", MIN_PROTEIN_G, len(df), before - len(df))

    before = len(df)
    df = df[(df["minutes"] > 0) & (df["minutes"] <= MAX_MINUTES)]
    logger.info("  After minutes <= %d: %d (dropped %d)", MAX_MINUTES, len(df), before - len(df))

    if "n_ingredients" in df.columns:
        before = len(df)
        df = df[(df["n_ingredients"] >= MIN_INGREDIENTS) & (df["n_ingredients"] <= MAX_INGREDIENTS)]
        logger.info("  After ingredients %d–%d: %d (dropped %d)",
                     MIN_INGREDIENTS, MAX_INGREDIENTS, len(df), before - len(df))

    # Remove recipes with missing names
    df = df[df["name"].notna() & (df["name"].str.strip() != "")]

    logger.info("Curation complete: %d fitness-relevant recipes selected", len(df))
    return df.reset_index(drop=True)


def build_import_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert curated DataFrame rows into SparkyFitness food + variant dicts."""
    rows = []
    for _, recipe in df.iterrows():
        recipe_id = int(recipe["recipe_id"])
        food_uuid = stable_uuid(recipe_id)
        variant_uuid = stable_variant_uuid(recipe_id)

        # Build ingredient text for search/embedding
        ingredient_text = ""
        if "ingredients" in recipe.index:
            ingredient_text = build_ingredient_text(recipe.get("ingredients"))

        # Detect allergens
        allergens = {}
        if "ingredients" in recipe.index:
            allergens = detect_allergens_for_recipe(recipe.get("ingredients"))

        # Build description with cuisine tag
        description = ""
        if "description" in recipe.index and pd.notna(recipe.get("description")):
            description = str(recipe["description"]).strip()
        cuisine = recipe.get("cuisine", "unknown")
        if cuisine and cuisine != "unknown":
            description = f"[{cuisine.title()}] {description}" if description else f"{cuisine.title()} recipe"

        # Map nutrition — convert PDV back to absolute where needed
        calories = float(recipe.get("calories", 0) or 0)
        protein_g = float(recipe.get("protein_g", 0) or 0)
        carbohydrate_g = float(recipe.get("carbohydrate_g", 0) or 0)
        total_fat_g = float(recipe.get("total_fat_g", 0) or 0)
        saturated_fat_g = float(recipe.get("saturated_fat_g", 0) or 0)
        sugar_g = float(recipe.get("sugar_g", 0) or 0)
        sodium_g = float(recipe.get("sodium_g", 0) or 0)

        rows.append({
            "food_uuid": food_uuid,
            "variant_uuid": variant_uuid,
            "foodcom_recipe_id": recipe_id,
            "name": str(recipe["name"]).strip()[:200],
            "description": description[:500] if description else None,
            "ingredient_text": ingredient_text[:1000] if ingredient_text else None,
            "food_category": infer_food_category(str(recipe["name"]), ingredient_text),
            "n_ingredients": int(recipe.get("n_ingredients", 1) or 1),
            "serving_size": 1,
            "serving_unit": "serving",
            "calories": round(calories, 1),
            "protein": round(protein_g, 1),
            "carbs": round(carbohydrate_g, 1),
            "fat": round(total_fat_g, 1),
            "saturated_fat": round(saturated_fat_g, 1),
            "sugars": round(sugar_g, 1),
            "sodium": round(sodium_g * 1000, 1),  # Convert g → mg for SparkyFitness
            "dietary_fiber": 0,  # Not available in Food.com data
            "cuisine": cuisine,
            "avg_rating": float(recipe.get("avg_rating", 0) or 0),
            "n_reviews": int(recipe.get("n_reviews", 0) or 0),
            "minutes": int(recipe.get("minutes", 0) or 0),
            **allergens,
        })

    return rows


def insert_into_sparkyfitness(db_url: str, rows: list[dict], dry_run: bool = False) -> int:
    """Insert curated recipes into SparkyFitness foods + food_variants tables."""
    if not HAS_PG:
        raise RuntimeError("psycopg2 is required for database import")

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cursor = conn.cursor()

    try:
        cursor.execute(
            "ALTER TABLE foods ADD COLUMN IF NOT EXISTS food_category TEXT DEFAULT 'other'"
        )
        # user_id = NULL means system-owned, shared_with_public = TRUE.
        psycopg2.extras.execute_batch(
            cursor,
            """
                INSERT INTO foods (id, name, brand, user_id, shared_with_public,
                                   is_custom, is_quick_food, provider_type,
                                   provider_external_id, food_category)
                VALUES (%(food_uuid)s, %(name)s, %(description)s, NULL, TRUE,
                        FALSE, FALSE, 'foodcom', %(foodcom_recipe_id)s::text,
                        %(food_category)s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    brand = EXCLUDED.brand,
                    shared_with_public = TRUE,
                    food_category = EXCLUDED.food_category
            """,
            rows,
        )
        psycopg2.extras.execute_batch(
            cursor,
            """
                INSERT INTO food_variants (id, food_id, serving_size, serving_unit,
                                           is_default,
                                           calories, protein, carbs, fat,
                                           saturated_fat, sugars, sodium,
                                           dietary_fiber)
                VALUES (%(variant_uuid)s, %(food_uuid)s, %(serving_size)s, %(serving_unit)s,
                        TRUE,
                        %(calories)s, %(protein)s, %(carbs)s, %(fat)s,
                        %(saturated_fat)s, %(sugars)s, %(sodium)s,
                        %(dietary_fiber)s)
                ON CONFLICT (id) DO UPDATE SET
                    calories = EXCLUDED.calories,
                    protein = EXCLUDED.protein,
                    carbs = EXCLUDED.carbs,
                    fat = EXCLUDED.fat,
                    saturated_fat = EXCLUDED.saturated_fat,
                    sugars = EXCLUDED.sugars,
                    sodium = EXCLUDED.sodium
            """,
            rows,
        )

        if dry_run:
            logger.info("DRY RUN — rolling back %d inserts", len(rows))
            conn.rollback()
        else:
            conn.commit()
            logger.info("Committed %d Food.com recipes to SparkyFitness DB", len(rows))

    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

    return len(rows)


def print_summary(rows: list[dict]) -> None:
    """Print a summary of what will be imported."""
    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("Food.com → SparkyFitness Import Summary")
    print("=" * 60)
    print(f"  Total recipes to import:  {len(rows):,}")
    print(f"  Avg calories:             {df['calories'].mean():.0f}")
    print(f"  Avg protein (g):          {df['protein'].mean():.1f}")
    print(f"  Avg prep time (min):      {df['minutes'].mean():.0f}")

    if "cuisine" in df.columns:
        print(f"\n  Top cuisines:")
        for cuisine, count in df["cuisine"].value_counts().head(8).items():
            print(f"    {cuisine:<20} {count:>5}")

    if "food_category" in df.columns:
        print(f"\n  Top food categories:")
        for category, count in df["food_category"].value_counts().head(8).items():
            print(f"    {category:<20} {count:>5}")

    # Allergen distribution
    allergen_cols = [c for c in df.columns if c.startswith("has_")]
    if allergen_cols:
        print(f"\n  Allergen prevalence:")
        for col in sorted(allergen_cols):
            pct = df[col].mean() * 100
            print(f"    {col:<20} {pct:>5.1f}%")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Import curated Food.com recipes into SparkyFitness as public foods"
    )
    parser.add_argument(
        "--enriched-csv", required=True,
        help="Path to enriched_recipes.csv from build_training_table.py"
    )
    parser.add_argument(
        "--raw-csv", default=None,
        help="Path to RAW_recipes.csv (for ingredients/description if not in enriched)"
    )
    parser.add_argument(
        "--db-url", default=os.environ.get("SPARKY_FITNESS_DB_URL"),
        help="SparkyFitness PostgreSQL connection URL. Defaults to SPARKY_FITNESS_DB_URL."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be imported without writing to DB"
    )
    parser.add_argument(
        "--max-recipes", type=int,
        default=int(os.environ["FOODCOM_IMPORT_MAX_RECIPES"])
        if os.environ.get("FOODCOM_IMPORT_MAX_RECIPES")
        else None,
        help="Cap the number of imported recipes (for testing)"
    )
    args = parser.parse_args()

    if not args.db_url:
        parser.error("--db-url is required unless SPARKY_FITNESS_DB_URL is set")

    # Load data
    logger.info("Loading enriched recipes from %s", args.enriched_csv)
    enriched = pd.read_csv(args.enriched_csv)
    enriched.columns = [c.strip().lower() for c in enriched.columns]

    raw = None
    if args.raw_csv and Path(args.raw_csv).exists():
        logger.info("Loading raw recipes from %s", args.raw_csv)
        raw = pd.read_csv(args.raw_csv)

    # Curate
    curated = curate_recipes(enriched, raw)

    if args.max_recipes:
        curated = curated.head(args.max_recipes)
        logger.info("Capped at %d recipes (--max-recipes)", args.max_recipes)

    # Build import rows
    rows = build_import_rows(curated)
    print_summary(rows)

    if not rows:
        logger.warning("No recipes passed curation filters — nothing to import")
        return

    # Import
    if args.dry_run:
        logger.info("DRY RUN — would insert %d recipes into SparkyFitness DB", len(rows))
        # Show sample
        print("\nSample records (first 3):")
        for row in rows[:3]:
            print(f"  {row['name'][:50]:<50} {row['calories']:>6.0f} cal  "
                  f"{row['protein']:>5.1f}g protein  {row['cuisine']}")
    else:
        imported = insert_into_sparkyfitness(args.db_url, rows, dry_run=False)
        logger.info("DONE: %d recipes imported successfully", imported)


if __name__ == "__main__":
    main()
