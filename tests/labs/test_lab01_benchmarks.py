"""Тести порівняльних блоків ЛР1: опис датасету, імп'ютація, кодування, аугментація."""

import numpy as np
import pandas as pd
import pytest

from mllabs.data import KICK_SCHEMA
from mllabs.labs.lab01.augmentation import KIND_SYNTHETIC, AugmentationBenchmark
from mllabs.labs.lab01.encoding import CardinalityMismatchError, EncodingBenchmark
from mllabs.labs.lab01.imputation import ImputationBenchmark
from mllabs.labs.lab01.overview import DatasetProfiler
from mllabs.preprocessing import (
    PreprocessingScenario,
    PreprocessingSettings,
    PreprocessorFactory,
    SamplerFactory,
)


def test_profiler_counts_classes_and_missing_cells():
    schema = KICK_SCHEMA.without([c for c in KICK_SCHEMA.features if c not in ("VehOdo", "Model")])
    data = pd.DataFrame(
        {
            "VehOdo": [1.0, np.nan, 3.0, 4.0],
            "Model": ["A", "B", "B", "C"],
            "PurchDate": pd.to_datetime(["2010-01-01"] * 4),
            "IsBadBuy": [0, 0, 0, 1],
        }
    )

    profile = DatasetProfiler(schema, high_cardinality_threshold=3).profile(data)

    assert profile.minority_share == pytest.approx(0.25)
    assert profile.imbalance_ratio == pytest.approx(3.0)
    assert profile.rows_with_missing_share == pytest.approx(0.25)
    assert profile.n_high_cardinality == 1
    assert profile.max_cardinality == 3


@pytest.fixture
def correlated_numeric():
    rng = np.random.default_rng(0)
    base = rng.normal(size=600)
    return pd.DataFrame(
        {
            "a": base,
            "b": 2 * base + rng.normal(scale=0.1, size=600),
            "c": -base + rng.normal(scale=0.1, size=600),
        }
    )


def test_masking_experiment_shows_knn_and_mice_beat_median_on_correlated_data(
    correlated_numeric,
):
    benchmark = ImputationBenchmark(
        mask_share=0.1, sample_size=500, knn_neighbors=5, mice_max_iter=10, random_state=0
    )

    result = benchmark.run_masking(correlated_numeric)

    # Медіана стандартизованої ознаки ≈ 0, тому її RMSE ≈ 1 (одне σ).
    assert result.rmse_overall["median"] == pytest.approx(1.0, abs=0.15)
    assert result.rmse_overall["knn"] < 0.5
    assert result.rmse_overall["mice"] < 0.5
    assert list(result.rmse_by_feature.columns) == ["Медіана", "KNN", "MICE"]
    # Приховано ~10 % значень; кожне — в усіх трьох методах.
    assert len(result.restored) == pytest.approx(0.1 * 500 * 3 * 3, rel=0.2)


def test_missingness_rate_shows_informative_missing_values():
    column = pd.Series(["A", None, "B", None])
    is_positive = pd.Series([False, True, False, True])

    rates = ImputationBenchmark.positive_rate_by_missingness(column, is_positive)

    assert rates == {"missing": 1.0, "present": 0.0}


def test_encoding_plan_chooses_method_by_cardinality(clean_kick_frame):
    schema = KICK_SCHEMA.without(["PRIMEUNIT", "AUCGUART"])  # видалені очищенням
    benchmark = EncodingBenchmark(schema, high_cardinality_threshold=30, rare_category_size=5)

    plan = benchmark.plan(clean_kick_frame).set_index("Ознака")

    assert plan.loc["Model", "Метод"] == "Target Encoding"  # 40 категорій
    assert plan.loc["VNST", "Метод"] == "One-Hot"  # 6 категорій
    assert plan.loc["Transmission", "Метод"] == "Ordinal (0/1)"


def test_augmentation_illustration_adds_only_synthetic_minority_rows(clean_kick_frame):
    schema = KICK_SCHEMA.without(["WheelTypeID", "PRIMEUNIT", "AUCGUART"])
    factory = PreprocessorFactory(schema, 0, PreprocessingSettings(onehot_min_frequency=5))
    features = clean_kick_frame.drop(columns=["IsBadBuy"])
    target = clean_kick_frame["IsBadBuy"]
    benchmark = AugmentationBenchmark(
        SamplerFactory(0.5, 5, 0.05, 0),
        plot_sample_size=100,
        random_state=0,
        positive_label=schema.positive_label,
    )

    result = benchmark.illustrate(factory.build(PreprocessingScenario.FULL), features, target)
    balance = result.class_balance.set_index("Метод")

    n_majority = int((target == 0).sum())
    assert (balance["Клас 0"] == n_majority).all()  # мажоритарний клас не чіпається
    assert balance.loc["Без аугментації", "Синтетичних"] == 0
    for method in ("SMOTE", "Гаусів шум"):
        assert balance.loc[method, "Клас 1"] == pytest.approx(0.5 * n_majority, abs=1)
    no_augmentation = result.projection[result.projection["method"] == "Без аугментації"]
    assert (no_augmentation["kind"] != KIND_SYNTHETIC).all()


def test_cardinality_check_rejects_schema_that_disagrees_with_threshold(clean_kick_frame):
    schema = KICK_SCHEMA.without(["WheelTypeID", "PRIMEUNIT", "AUCGUART"])
    features = clean_kick_frame.drop(columns=["IsBadBuy"])

    # У синтетичних даних найменша високопотужна ознака має 6 категорій, низькі — до 4.
    EncodingBenchmark(schema, high_cardinality_threshold=5, rare_category_size=2).check_schema(
        features
    )
    with pytest.raises(CardinalityMismatchError, match="categorical_high"):
        EncodingBenchmark(schema, high_cardinality_threshold=30, rare_category_size=2).check_schema(
            features
        )
