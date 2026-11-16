"""Тести детермінованого очищення сирої таблиці (RawDataCleaner)."""

import numpy as np
import pandas as pd

from mllabs.data import KICK_SCHEMA, RawDataCleaner


def _clean(raw: pd.DataFrame):
    return RawDataCleaner(KICK_SCHEMA, max_missing_share=0.5).clean(raw)


def test_cleaner_turns_zero_and_one_dollar_prices_into_missing(raw_kick_frame):
    raw = raw_kick_frame.copy()
    raw.loc[0, "VehBCost"] = 0.0
    raw.loc[1, "MMRCurrentRetailCleanPrice"] = 1.0
    raw.loc[2, "VehBCost"] = 2.0  # два долари — дивно, але правило стосується лише <= 1

    result = _clean(raw)

    assert np.isnan(result.data.loc[0, "VehBCost"])
    assert np.isnan(result.data.loc[1, "MMRCurrentRetailCleanPrice"])
    assert result.data.loc[2, "VehBCost"] == 2.0
    price_columns = list(KICK_SCHEMA.price_columns)
    prices = result.data[price_columns]
    assert (prices.isna() | (prices > 1)).all().all()  # лишилися або пропуски, або ціни > 1


def test_cleaner_counts_hidden_missing_values(raw_kick_frame):
    result = _clean(raw_kick_frame)

    prices = raw_kick_frame[list(KICK_SCHEMA.price_columns)]
    n_tokens = sum(
        int((raw_kick_frame[column] == "NOT AVAIL").sum()) for column in KICK_SCHEMA.categorical
    )
    assert result.hidden_missing_cells == int((prices <= 1).sum().sum()) + n_tokens


def test_cleaner_normalizes_case_and_text_tokens(raw_kick_frame):
    data = _clean(raw_kick_frame).data

    assert set(data["Transmission"].dropna()) == {"AUTO", "MANUAL"}
    assert "NOT AVAIL" not in set(data["Color"].dropna())
    assert data["Color"].isna().sum() == (raw_kick_frame["Color"] == "NOT AVAIL").sum()


def test_cleaner_drops_duplicate_and_sparse_columns_from_data_and_schema(raw_kick_frame):
    result = _clean(raw_kick_frame)

    for column in ("WheelTypeID", "PRIMEUNIT", "AUCGUART"):
        assert column not in result.data.columns
        assert column not in result.schema.features
    assert "WheelType" in result.schema.categorical_low


def test_cleaner_casts_target_categories_and_date(raw_kick_frame):
    data = _clean(raw_kick_frame).data

    assert data["IsBadBuy"].dtype == np.int64
    assert set(data["IsBadBuy"]) == {0, 1}
    assert pd.api.types.is_datetime64_any_dtype(data["PurchDate"])
    assert data["VNZIP1"].dtype == object
    assert isinstance(data["VNZIP1"].iloc[0], str)


def test_cleaner_does_not_modify_raw_table(raw_kick_frame):
    before = raw_kick_frame.copy()

    _clean(raw_kick_frame)

    pd.testing.assert_frame_equal(raw_kick_frame, before)
