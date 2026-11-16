"""Тести власних трансформерів: FrequencyEncoder, GroupAggregateEncoder, RowFeatureEngineer."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from mllabs.preprocessing import (
    FrequencyEncoder,
    GroupAggregateEncoder,
    GroupAggregation,
    RowFeatureEngineer,
)


def test_frequency_encoder_uses_train_shares_only():
    train = pd.DataFrame({"model": ["A", "A", "A", "B"]})
    test = pd.DataFrame({"model": ["B", "B", "B", "B", "A"]})  # у тесті інший розподіл

    encoded = FrequencyEncoder().fit(train).transform(test)

    np.testing.assert_allclose(encoded[:, 0], [0.25, 0.25, 0.25, 0.25, 0.75])


def test_frequency_encoder_maps_unseen_category_and_missing_to_zero():
    encoder = FrequencyEncoder().fit(pd.DataFrame({"model": ["A", "B"]}))

    encoded = encoder.transform(pd.DataFrame({"model": ["C", np.nan]}))

    np.testing.assert_array_equal(encoded[:, 0], [0.0, 0.0])


def test_frequency_encoder_keeps_shape_and_feature_names():
    train = pd.DataFrame({"make": ["X", "Y"], "state": ["FL", "FL"]})

    encoder = FrequencyEncoder().fit(train)

    assert encoder.transform(train).shape == (2, 2)
    assert list(encoder.get_feature_names_out()) == ["make", "state"]
    assert clone(encoder).get_params() == {}  # сумісний із clone() крос-валідації


_AGGREGATION = GroupAggregation("Model", "VehBCost", "mean")


def test_group_aggregate_encoder_does_not_see_test_rows():
    train = pd.DataFrame({"Model": ["A", "A", "B"], "VehBCost": [100.0, 300.0, 50.0]})
    test = pd.DataFrame({"Model": ["A", "B"], "VehBCost": [10_000.0, 10_000.0]})

    transformed = GroupAggregateEncoder([_AGGREGATION]).fit(train).transform(test)

    # Середні — з train (200 і 50), величезні тестові ціни на них не впливають.
    assert transformed["VehBCost_dev_by_Model"].tolist() == [9_800.0, 9_950.0]


def test_group_aggregate_encoder_uses_overall_statistic_for_unseen_group():
    train = pd.DataFrame({"Model": ["A", "B"], "VehBCost": [100.0, 300.0]})
    test = pd.DataFrame({"Model": ["NEW"], "VehBCost": [250.0]})

    transformed = GroupAggregateEncoder([_AGGREGATION]).fit(train).transform(test)

    assert transformed["VehBCost_dev_by_Model"].iloc[0] == 50.0  # 250 − середнє train 200


def test_group_aggregate_encoder_adds_only_deviation_not_group_statistic():
    train = pd.DataFrame({"Model": ["A", "A"], "VehBCost": [100.0, 300.0]})

    transformed = GroupAggregateEncoder([_AGGREGATION]).fit_transform(train)

    # Статистика поруч із ціною та відхиленням дала б точну залежність
    # «ціна = статистика + відхилення», тому окремою колонкою вона не додається.
    assert set(transformed.columns) == {"Model", "VehBCost", "VehBCost_dev_by_Model"}
    assert GroupAggregateEncoder([_AGGREGATION]).output_features == ("VehBCost_dev_by_Model",)


def test_group_aggregate_encoder_does_not_modify_input():
    train = pd.DataFrame({"Model": ["A"], "VehBCost": [1.0]})
    before = train.copy()

    GroupAggregateEncoder([_AGGREGATION]).fit_transform(train)

    pd.testing.assert_frame_equal(train, before)


def test_row_feature_engineer_computes_ratios_dates_and_indicators(clean_kick_frame):
    rows = clean_kick_frame.head(50)

    result = RowFeatureEngineer().fit_transform(rows)

    expected_ratio = rows["VehBCost"] / rows["MMRAcquisitionAuctionAveragePrice"]
    pd.testing.assert_series_equal(result["cost_to_mmr"], expected_ratio, check_names=False)
    np.testing.assert_allclose(result["miles_per_year"], rows["VehOdo"] / (rows["VehicleAge"] + 1))
    assert result["wheel_type_missing"].tolist() == rows["WheelType"].isna().astype(int).tolist()
    assert result["purchase_month"].between(1, 12).all()
    new_columns = (
        RowFeatureEngineer.NUMERIC_FEATURES
        + RowFeatureEngineer.DATE_FEATURES
        + RowFeatureEngineer.INDICATOR_FEATURES
    )
    assert set(new_columns) <= set(result.columns)
    assert len(RowFeatureEngineer.describe()) == len(new_columns)


def test_row_feature_engineer_propagates_missing_price_as_missing_ratio():
    rows = pd.DataFrame(
        {
            "VehBCost": [5000.0],
            "MMRAcquisitionAuctionAveragePrice": [np.nan],
            "MMRCurrentAuctionAveragePrice": [4000.0],
            "MMRAcquisitionRetailAveragePrice": [6000.0],
            "VehOdo": [50_000.0],
            "VehicleAge": [4.0],
            "WarrantyCost": [1000.0],
            "PurchDate": pd.to_datetime(["2010-03-15"]),
            "WheelType": ["Alloy"],
        }
    )

    result = RowFeatureEngineer().transform(rows)

    assert np.isnan(result["cost_to_mmr"].iloc[0])
    assert result["mmr_missing"].iloc[0] == 1
    assert result["miles_per_year"].iloc[0] == pytest.approx(10_000.0)
