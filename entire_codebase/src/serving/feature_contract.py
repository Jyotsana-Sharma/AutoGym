from __future__ import annotations

"""Shared train/serve feature contract for the SparkyFitness ranker."""

ID_COLUMNS = ["user_id", "recipe_id"]
EMBEDDING_COMPONENTS = 8


def _embedding_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(1, EMBEDDING_COMPONENTS + 1)]


def _candidate_embedding_columns() -> list[str]:
    columns: list[str] = []
    for idx in range(1, EMBEDDING_COMPONENTS + 1):
        columns.extend([f"ingredient_emb_{idx}", f"text_emb_{idx}"])
    return columns


REQUEST_ONLY_FIELDS = [
    "candidate_name_text",
    "candidate_description_text",
    "candidate_ingredient_text",
    "user_history_name_text",
    "user_history_ingredient_text",
    "user_history_avg_calories",
    "user_history_avg_protein_g",
    "user_history_avg_carbohydrate_g",
    "user_history_avg_total_fat_g",
]

FEATURE_COLUMNS = [
    "minutes", "n_ingredients", "n_steps", "avg_rating", "n_reviews",
    "cuisine",
    "calories", "total_fat", "sugar", "sodium", "protein",
    "saturated_fat", "carbohydrate",
    "total_fat_g", "sugar_g", "sodium_g", "protein_g",
    "saturated_fat_g", "carbohydrate_g",
    *_candidate_embedding_columns(),
    "has_egg", "has_fish", "has_milk", "has_nuts", "has_peanut",
    "has_sesame", "has_shellfish", "has_soy", "has_wheat",
    "meal_type_breakfast", "meal_type_lunch", "meal_type_dinner", "meal_type_snack",
    "user_preference_sample_count",
    "user_vegetarian", "user_vegan", "user_gluten_free", "user_dairy_free",
    "user_low_sodium", "user_low_fat", "user_low_sugar", "user_high_protein",
    "history_pc1", "history_pc2", "history_pc3", "history_pc4",
    "history_pc5", "history_pc6",
    *_embedding_columns("history_ingredient_emb"),
    *_embedding_columns("history_text_emb"),
    "ingredient_embedding_cosine",
    "text_embedding_cosine",
    "history_macro_similarity",
    "category_affinity_score",
]
