from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path
import sys

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.embedding_preprocessor import EmbeddingPreprocessor


def build_pipeline(docs: list[str]) -> dict:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=128)
    tfidf = vectorizer.fit_transform(docs)
    svd = TruncatedSVD(n_components=2, random_state=42)
    svd.fit(tfidf)
    return {
        "vectorizer": vectorizer,
        "svd": svd,
        "output_dim": 2,
        "target_dim": 8,
    }


class EmbeddingPreprocessorTest(unittest.TestCase):
    def test_similar_history_scores_higher_than_unrelated_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ingredient_docs = [
                "yogurt blueberry pancake oats",
                "yogurt blueberry smoothie oats",
                "chicken rice garlic",
            ]
            text_docs = [
                "blueberry yogurt pancakes breakfast",
                "blueberry yogurt parfait breakfast",
                "savory chicken rice bowl dinner",
            ]
            with open(root / "ingredient_pipeline.pkl", "wb") as handle:
                pickle.dump(build_pipeline(ingredient_docs), handle)
            with open(root / "text_pipeline.pkl", "wb") as handle:
                pickle.dump(build_pipeline(text_docs), handle)

            preprocessor = EmbeddingPreprocessor.from_artifact_dir(root)
            self.assertIsNotNone(preprocessor)

            history_names = '["blueberry yogurt pancakes","blueberry yogurt parfait","yogurt berry breakfast bowl"]'
            history_ingredients = '["yogurt blueberry oats","blueberry yogurt granola","oats yogurt blueberry"]'
            common = {
                "user_id": 1,
                "recipe_id": 10,
                "calories": 320,
                "protein_g": 18,
                "carbohydrate_g": 44,
                "total_fat_g": 9,
                "user_history_name_text": history_names,
                "user_history_ingredient_text": history_ingredients,
                "user_history_avg_calories": 300,
                "user_history_avg_protein_g": 17,
                "user_history_avg_carbohydrate_g": 41,
                "user_history_avg_total_fat_g": 8,
            }
            enriched = preprocessor.enrich_instances([
                {
                    **common,
                    "recipe_id": 10,
                    "candidate_name_text": "Blueberry Yogurt Pancakes",
                    "candidate_description_text": "Sweet berry breakfast",
                    "candidate_ingredient_text": "yogurt blueberry pancake oats",
                },
                {
                    **common,
                    "recipe_id": 11,
                    "candidate_name_text": "Garlic Chicken Rice Bowl",
                    "candidate_description_text": "Savory dinner",
                    "candidate_ingredient_text": "chicken rice garlic",
                },
            ])

            self.assertGreater(
                enriched[0]["ingredient_embedding_cosine"],
                enriched[1]["ingredient_embedding_cosine"],
            )
            self.assertGreater(
                enriched[0]["text_embedding_cosine"],
                enriched[1]["text_embedding_cosine"],
            )
            self.assertGreater(
                enriched[0]["history_macro_similarity"],
                0.0,
            )

    def test_short_history_zero_fills_history_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs = ["berry yogurt", "chicken rice", "oat pancake"]
            with open(root / "ingredient_pipeline.pkl", "wb") as handle:
                pickle.dump(build_pipeline(docs), handle)
            with open(root / "text_pipeline.pkl", "wb") as handle:
                pickle.dump(build_pipeline(docs), handle)

            preprocessor = EmbeddingPreprocessor.from_artifact_dir(root)
            enriched = preprocessor.enrich_instances([
                {
                    "user_id": 1,
                    "recipe_id": 22,
                    "candidate_name_text": "Berry Yogurt Bowl",
                    "candidate_description_text": "Breakfast",
                    "candidate_ingredient_text": "berry yogurt",
                    "user_history_name_text": '["berry yogurt bowl","banana oats"]',
                    "user_history_ingredient_text": '["berry yogurt","banana oats"]',
                    "protein_g": 12,
                    "carbohydrate_g": 28,
                    "total_fat_g": 6,
                    "user_history_avg_calories": 210,
                    "user_history_avg_protein_g": 11,
                    "user_history_avg_carbohydrate_g": 25,
                    "user_history_avg_total_fat_g": 5,
                }
            ])[0]

            self.assertEqual(enriched["ingredient_embedding_cosine"], 0.0)
            self.assertEqual(enriched["text_embedding_cosine"], 0.0)
            self.assertEqual(enriched["history_macro_similarity"], 0.0)
            self.assertEqual(enriched["history_ingredient_emb_1"], 0.0)
            self.assertEqual(enriched["history_text_emb_1"], 0.0)


if __name__ == "__main__":
    unittest.main()
