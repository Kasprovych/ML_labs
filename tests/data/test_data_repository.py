"""Тести сховища підготовленого датасету (запис і читання з типами)."""

import pandas as pd
import pytest

from mllabs.data import (
    KICK_SCHEMA,
    ProcessedDatasetNotFoundError,
    ProcessedDatasetRepository,
    RawDataCleaner,
    StratifiedSplitter,
)


@pytest.fixture
def saved(tmp_path, raw_kick_frame):
    cleaning = RawDataCleaner(KICK_SCHEMA, 0.5).clean(raw_kick_frame)
    split = StratifiedSplitter(0.2, random_state=0).split(cleaning.data, "IsBadBuy")
    repository = ProcessedDatasetRepository(tmp_path, "kick")
    repository.save(split, cleaning.schema)
    return repository, split, cleaning.schema


def test_repository_round_trip_preserves_values_and_types(saved):
    repository, split, schema = saved

    loaded = repository.load()

    pd.testing.assert_frame_equal(loaded.x_train, split.x_train, check_dtype=False)
    pd.testing.assert_series_equal(loaded.y_test, split.y_test, check_dtype=False)
    assert loaded.x_train["VNZIP1"].dtype == object
    assert isinstance(loaded.x_train["VNZIP1"].iloc[0], str)  # індекс не став числом
    assert pd.api.types.is_datetime64_any_dtype(loaded.x_train["PurchDate"])
    assert loaded.x_train["VehOdo"].dtype == float
    assert loaded.y_train.dtype == int
    assert repository.load_schema() == schema


def test_repository_returns_stratified_train_subsample_and_full_test(saved):
    repository, split, _ = saved

    loaded = repository.load(train_sample_size=300, random_state=0)

    assert len(loaded.x_train) == 300
    assert len(loaded.x_test) == len(split.x_test)
    assert loaded.y_train.mean() == pytest.approx(split.y_train.mean(), abs=0.01)


def test_repository_raises_clear_error_before_lab1_was_run(tmp_path):
    repository = ProcessedDatasetRepository(tmp_path, "kick")

    with pytest.raises(ProcessedDatasetNotFoundError, match="mllabs run 1"):
        repository.load()


def test_repository_stores_preprocessing_settings_next_to_schema(tmp_path, raw_kick_frame):
    cleaning = RawDataCleaner(KICK_SCHEMA, 0.5).clean(raw_kick_frame)
    split = StratifiedSplitter(0.2, random_state=0).split(cleaning.data, "IsBadBuy")
    repository = ProcessedDatasetRepository(tmp_path, "kick")
    manifest = {"random_state": 42, "settings": {"polynomial_features": ["VehOdo"]}}

    repository.save(split, cleaning.schema, manifest)

    assert repository.load_preprocessing() == manifest


def test_repository_raises_clear_error_without_preprocessing_settings(tmp_path):
    with pytest.raises(ProcessedDatasetNotFoundError, match="mllabs run 1"):
        ProcessedDatasetRepository(tmp_path, "kick").load_preprocessing()
