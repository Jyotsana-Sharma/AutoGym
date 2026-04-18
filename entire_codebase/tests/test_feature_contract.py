from __future__ import annotations

import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.feature_contract import FEATURE_COLUMNS, ID_COLUMNS
from src.training.ranking_data import infer_feature_columns


class FeatureContractTest(unittest.TestCase):
    def test_serving_feature_contract_matches_training_inference(self):
        frame = pd.DataFrame(
            columns=[
                *ID_COLUMNS,
                "date",
                "label",
                "name",
                *FEATURE_COLUMNS,
            ]
        )

        self.assertEqual(infer_feature_columns(frame), FEATURE_COLUMNS)
        self.assertEqual(len(FEATURE_COLUMNS), 45)

    def test_non_model_request_metadata_is_not_a_feature(self):
        frame = pd.DataFrame(
            columns=[
                *ID_COLUMNS,
                "recommendation_id",
                "request_id",
                *FEATURE_COLUMNS,
            ]
        )

        inferred = infer_feature_columns(frame)
        self.assertNotIn("recommendation_id", FEATURE_COLUMNS)
        self.assertNotIn("request_id", FEATURE_COLUMNS)
        self.assertNotIn("recommendation_id", inferred)
        self.assertNotIn("request_id", inferred)


if __name__ == "__main__":
    unittest.main()
