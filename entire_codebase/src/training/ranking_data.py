from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas import DataFrame


NON_FEATURE_COLUMNS = {
    "user_id",
    "recipe_id",
    "date",
    "label",
    "name",
    "request_id",
    "recommendation_id",
}


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
