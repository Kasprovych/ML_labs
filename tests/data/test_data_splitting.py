"""Тести стратифікованого розбиття та схеми датасету."""

import pandas as pd
import pytest

from mllabs.data import KICK_SCHEMA, DatasetSchema, DatasetSplit, StratifiedSplitter


def test_split_preserves_class_shares_and_sizes(clean_kick_frame):
    split = StratifiedSplitter(test_size=0.25, random_state=0).split(clean_kick_frame, "IsBadBuy")

    assert len(split.x_test) == 300
    assert len(split.x_train) == 900
    overall = clean_kick_frame["IsBadBuy"].mean()
    assert split.y_train.mean() == pytest.approx(overall, abs=0.005)
    assert split.y_test.mean() == pytest.approx(overall, abs=0.005)
    assert "IsBadBuy" not in split.x_train.columns


def test_split_is_reproducible_with_same_seed(clean_kick_frame):
    first = StratifiedSplitter(0.2, random_state=7).split(clean_kick_frame, "IsBadBuy")
    second = StratifiedSplitter(0.2, random_state=7).split(clean_kick_frame, "IsBadBuy")

    assert first.x_test.equals(second.x_test)


def test_schema_survives_round_trip_through_dict():
    restored = DatasetSchema.from_dict(KICK_SCHEMA.to_dict())

    assert restored == KICK_SCHEMA


def test_schema_without_removes_columns_from_every_role():
    reduced = KICK_SCHEMA.without(["VehYear", "PRIMEUNIT", "VehBCost"])

    assert "VehYear" not in reduced.numeric
    assert "VehYear" not in reduced.collinear_columns
    assert "PRIMEUNIT" not in reduced.categorical
    assert "VehBCost" not in reduced.price_columns


def test_split_metrics_count_rows_and_positive_share_for_given_label():
    split = DatasetSplit(
        x_train=pd.DataFrame({"a": range(4)}),
        x_test=pd.DataFrame({"a": range(2)}),
        y_train=pd.Series([0, 0, 0, 1]),
        y_test=pd.Series([1, 0]),
    )

    metrics = split.to_metrics(positive_label=1)

    assert metrics == {
        "train_rows": 4,
        "test_rows": 2,
        "train_minority_share": 0.25,
        "test_minority_share": 0.5,
    }
