from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.batch_pipeline import build_online_training_rows


class BatchPipelineOnlineRowsTest(unittest.TestCase):
    def test_build_online_training_rows_maps_feedback_actions(self):
        records = [
            {
                "request_id": "req-1",
                "recommendation_id": "rec-1",
                "inf_user_id": 7,
                "inf_recipe_id": 101,
                "fb_user_id": 7,
                "fb_recipe_id": 101,
                "action": "logged",
                "date": "2026-04-01T00:00:00Z",
                "features_json": {
                    "minutes": 10,
                    "cuisine": "american",
                    "ingredient_embedding_cosine": 0.9,
                    "candidate_name_text": "ignored raw field",
                },
            },
            {
                "request_id": "req-2",
                "recommendation_id": "rec-2",
                "inf_user_id": 7,
                "inf_recipe_id": 102,
                "fb_user_id": 7,
                "fb_recipe_id": 102,
                "action": "saved",
                "date": "2026-04-02T00:00:00Z",
                "features_json": {
                    "minutes": 20,
                    "ingredient_embedding_cosine": 0.4,
                },
            },
            {
                "request_id": "req-3",
                "recommendation_id": "rec-3",
                "inf_user_id": 7,
                "inf_recipe_id": 103,
                "fb_user_id": 7,
                "fb_recipe_id": 103,
                "action": "dismissed",
                "date": "2026-04-03T00:00:00Z",
                "features_json": {
                    "minutes": 30,
                    "ingredient_embedding_cosine": 0.1,
                },
            },
            {
                "request_id": "req-4",
                "recommendation_id": "rec-4",
                "inf_user_id": 7,
                "inf_recipe_id": 104,
                "fb_user_id": 7,
                "fb_recipe_id": 104,
                "action": "viewed",
                "date": "2026-04-04T00:00:00Z",
                "features_json": {
                    "minutes": 40,
                },
            },
            {
                "request_id": "req-5",
                "recommendation_id": "rec-5",
                "inf_user_id": 8,
                "inf_recipe_id": 105,
                "fb_user_id": 9,
                "fb_recipe_id": 105,
                "action": "logged",
                "date": "2026-04-05T00:00:00Z",
                "features_json": {
                    "minutes": 50,
                },
            },
        ]

        frame = build_online_training_rows(records, upsample_factor=1)

        self.assertEqual(len(frame), 3)
        self.assertEqual(frame["label"].tolist(), [1, 1, 0])
        self.assertNotIn("candidate_name_text", frame.columns)
        self.assertEqual(frame.loc[0, "cuisine"], "american")
        self.assertEqual(frame.loc[1, "cuisine"], "unknown")


if __name__ == "__main__":
    unittest.main()
