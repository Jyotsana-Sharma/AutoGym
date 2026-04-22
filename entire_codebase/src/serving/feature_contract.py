"""Shared train/serve feature contract for the SparkyFitness ranker."""

ID_COLUMNS = ["user_id", "recipe_id"]
EMBEDDING_COMPONENTS = 8


def _embedding_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(1, EMBEDDING_COMPONENTS + 1)]


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
    "rating",
    "minutes", "n_ingredients", "n_steps", "avg_rating", "n_reviews",
    "cuisine",
    "calories", "total_fat", "sugar", "sodium", "protein",
    "saturated_fat", "carbohydrate",
    "total_fat_g", "sugar_g", "sodium_g", "protein_g",
    "saturated_fat_g", "carbohydrate_g",
    "has_egg", "has_fish", "has_milk", "has_nuts", "has_peanut",
    "has_sesame", "has_shellfish", "has_soy", "has_wheat",
    "daily_calorie_target", "protein_target_g", "carbs_target_g", "fat_target_g",
    "user_vegetarian", "user_vegan", "user_gluten_free", "user_dairy_free",
    "user_low_sodium", "user_low_fat",
    "history_pc1", "history_pc2", "history_pc3", "history_pc4",
    "history_pc5", "history_pc6",
    *_embedding_columns("ingredient_emb"),
    *_embedding_columns("text_emb"),
    *_embedding_columns("history_ingredient_emb"),
    *_embedding_columns("history_text_emb"),
    "ingredient_embedding_cosine",
    "text_embedding_cosine",
    "history_macro_similarity",
]
