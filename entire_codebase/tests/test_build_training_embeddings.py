from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_training_table import (
    add_recipe_embedding_features,
    assemble_training_table,
    derive_user_features,
)


class BuildTrainingEmbeddingsTest(unittest.TestCase):
    def test_offline_embeddings_and_similarity_features_are_materialized(self):
        recipes = pd.DataFrame(
            [
                {
                    "recipe_id": 1,
                    "name": "Blueberry Yogurt Pancakes",
                    "minutes": 15,
                    "tags": "['breakfast', 'vegetarian']",
                    "ingredients": "['blueberries', 'yogurt', 'oats', 'egg']",
                    "n_ingredients": 4,
                    "n_steps": 4,
                    "description": "sweet breakfast pancakes",
                    "avg_rating": 4.8,
                    "n_reviews": 10,
                    "cuisine": "american",
                    "allergens": ["milk", "egg"],
                    "calories": 320.0,
                    "total_fat": 12.0,
                    "sugar": 18.0,
                    "sodium": 4.0,
                    "protein": 22.0,
                    "saturated_fat": 6.0,
                    "carbohydrate": 25.0,
                    "total_fat_g": 9.0,
                    "sugar_g": 9.0,
                    "sodium_g": 92.0,
                    "protein_g": 11.0,
                    "saturated_fat_g": 1.2,
                    "carbohydrate_g": 41.0,
                },
                {
                    "recipe_id": 2,
                    "name": "Strawberry Yogurt Pancakes",
                    "minutes": 18,
                    "tags": "['breakfast', 'vegetarian']",
                    "ingredients": "['strawberries', 'yogurt', 'oats', 'egg']",
                    "n_ingredients": 4,
                    "n_steps": 4,
                    "description": "berry breakfast pancakes",
                    "avg_rating": 4.6,
                    "n_reviews": 9,
                    "cuisine": "american",
                    "allergens": ["milk", "egg"],
                    "calories": 315.0,
                    "total_fat": 11.0,
                    "sugar": 17.0,
                    "sodium": 4.0,
                    "protein": 21.0,
                    "saturated_fat": 6.0,
                    "carbohydrate": 24.0,
                    "total_fat_g": 8.5,
                    "sugar_g": 8.0,
                    "sodium_g": 90.0,
                    "protein_g": 10.5,
                    "saturated_fat_g": 1.2,
                    "carbohydrate_g": 40.0,
                },
                {
                    "recipe_id": 3,
                    "name": "Garlic Beef Rice Bowl",
                    "minutes": 30,
                    "tags": "['dinner']",
                    "ingredients": "['beef', 'rice', 'garlic', 'soy sauce']",
                    "n_ingredients": 4,
                    "n_steps": 5,
                    "description": "savory rice dinner bowl",
                    "avg_rating": 4.2,
                    "n_reviews": 8,
                    "cuisine": "asian",
                    "allergens": ["soy"],
                    "calories": 540.0,
                    "total_fat": 21.0,
                    "sugar": 4.0,
                    "sodium": 12.0,
                    "protein": 32.0,
                    "saturated_fat": 9.0,
                    "carbohydrate": 30.0,
                    "total_fat_g": 16.0,
                    "sugar_g": 2.0,
                    "sodium_g": 276.0,
                    "protein_g": 16.0,
                    "saturated_fat_g": 1.8,
                    "carbohydrate_g": 48.0,
                },
            ]
        )
        interactions = pd.DataFrame(
            [
                {"user_id": 1, "recipe_id": 1, "date": "2026-01-01", "rating": 5, "label": 1},
                {"user_id": 1, "recipe_id": 1, "date": "2026-01-02", "rating": 5, "label": 1},
                {"user_id": 1, "recipe_id": 1, "date": "2026-01-03", "rating": 5, "label": 1},
                {"user_id": 1, "recipe_id": 1, "date": "2026-01-04", "rating": 5, "label": 1},
                {"user_id": 1, "recipe_id": 1, "date": "2026-01-05", "rating": 5, "label": 1},
                {"user_id": 1, "recipe_id": 2, "date": "2026-01-06", "rating": 2, "label": 0},
                {"user_id": 1, "recipe_id": 3, "date": "2026-01-07", "rating": 2, "label": 0},
            ]
        )

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                recipes_with_embeddings = add_recipe_embedding_features(recipes)
                user_features = derive_user_features(interactions, recipes_with_embeddings)
                training = assemble_training_table(interactions, recipes_with_embeddings, user_features)
            finally:
                os.chdir(cwd)

            self.assertTrue((Path(tmpdir) / "embedding_artifacts" / "ingredient_pipeline.pkl").exists())
            self.assertTrue((Path(tmpdir) / "embedding_artifacts" / "text_pipeline.pkl").exists())
            self.assertIn("ingredient_emb_1", training.columns)
            self.assertIn("history_ingredient_emb_1", training.columns)
            self.assertIn("ingredient_embedding_cosine", training.columns)
            self.assertIn("text_embedding_cosine", training.columns)

            recipe2 = training[training["recipe_id"] == 2].iloc[0]
            recipe3 = training[training["recipe_id"] == 3].iloc[0]
            self.assertGreater(recipe2["ingredient_embedding_cosine"], recipe3["ingredient_embedding_cosine"])
            self.assertGreater(recipe2["text_embedding_cosine"], recipe3["text_embedding_cosine"])


if __name__ == "__main__":
    unittest.main()
