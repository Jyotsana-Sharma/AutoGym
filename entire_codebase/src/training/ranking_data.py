from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from src.serving.feature_contract import REQUEST_ONLY_FIELDS


NON_FEATURE_COLUMNS = {
    "user_id",
    "recipe_id",
    "date",
    "label",
    "rating",
    "group_id",
    "name",
    "data_source",
    "request_id",
    "recommendation_id",
}
NON_FEATURE_COLUMNS.update(REQUEST_ONLY_FIELDS)

# Heuristic-only fields are useful for product logic, explanations, or
# fallback rankers, but they should not be learned by the offline model. Most
# of these are either target/remaining-day calculations or live-only signals
# that are constant in the offline recipe data, which creates train/serve skew.
HEURISTIC_ONLY_COLUMNS = {
    "daily_calorie_target",
    "protein_target_g",
    "carbs_target_g",
    "fat_target_g",
    "remaining_calorie_ratio",
    "remaining_protein_ratio",
    "remaining_carb_ratio",
    "remaining_fat_ratio",
    "goal_calorie_gap_ratio",
    "goal_protein_gap_ratio",
    "goal_carb_gap_ratio",
    "goal_fat_gap_ratio",
    "candidate_is_food",
    "candidate_is_user_owned",
    "candidate_is_public",
    "community_logged_count",
    "community_saved_count",
    "community_dismissed_count",
    "recently_logged_candidate",
    "recently_dismissed_candidate",
    "recently_seen_candidate",
    "basket_lift_score",
    "meal_type_match",
}
NON_FEATURE_COLUMNS.update(HEURISTIC_ONLY_COLUMNS)


@dataclass
class PreparedFrames:
    train: DataFrame
    validation: DataFrame
    test: DataFrame
    feature_columns: list[str]

def read_split(path: str) -> DataFrame:
    split_path = Path(path)
    frame = pd.read_csv(split_path)
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="raise").astype(int)
    frame["recipe_id"] = pd.to_numeric(frame["recipe_id"], errors="raise").astype(int)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(float)
    return frame


def infer_feature_columns(frame: DataFrame) -> list[str]:
    feature_columns = [column for column in frame.columns if column not in NON_FEATURE_COLUMNS]
    return feature_columns


def string_feature_columns(frame: DataFrame, feature_columns: list[str]) -> list[str]:
    return [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(frame[column])
    ]


def encode_string_columns(
    train: DataFrame,
    validation: DataFrame,
    test: DataFrame,
    feature_columns: list[str],
) -> tuple[DataFrame, DataFrame, DataFrame]:
    encoded_frames = [train.copy(), validation.copy(), test.copy()]
    for column in string_feature_columns(train, feature_columns):
        categories = {value: idx for idx, value in enumerate(sorted(train[column].fillna("unknown").astype(str).unique()))}
        for frame in encoded_frames:
            frame[column] = (
                frame[column]
                .fillna("unknown")
                .astype(str)
                .map(categories)
                .fillna(-1)
                .astype(float)
            )
    return tuple(encoded_frames)


def numeric_features(
    train: DataFrame,
    validation: DataFrame,
    test: DataFrame,
    feature_columns: list[str],
) -> tuple[DataFrame, DataFrame, DataFrame]:
    encoded_frames = [train.copy(), validation.copy(), test.copy()]
    for column in feature_columns:
        for frame in encoded_frames:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
    return tuple(encoded_frames)


def load_training_frames(config: dict) -> PreparedFrames:
    data_config = config["data"]
    train = read_split(data_config["train_path"])
    validation = read_split(data_config["validation_path"])
    test = read_split(data_config["test_path"])

    feature_columns = infer_feature_columns(train)
    train, validation, test = encode_string_columns(train, validation, test, feature_columns)
    train, validation, test = numeric_features(train, validation, test, feature_columns)

    return PreparedFrames(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        feature_columns=feature_columns,
    )
