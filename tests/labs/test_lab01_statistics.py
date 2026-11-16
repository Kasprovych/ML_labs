"""Тести кореляційного аналізу, VIF, тестів нормальності та масштабування (ЛР1, R5–R8, R13)."""

import numpy as np
import pandas as pd
import pytest

from mllabs.data import KICK_SCHEMA
from mllabs.labs.lab01.correlation import CorrelationAnalyzer, ExtraCollinearDropCheck
from mllabs.labs.lab01.normality import NormalityTester
from mllabs.labs.lab01.scaling import (
    METHOD_BOX_COX,
    METHOD_RAW,
    METHOD_ROBUST,
    METHOD_YEO_JOHNSON,
    ScalingBenchmark,
)
from mllabs.preprocessing import PreprocessingSettings


@pytest.fixture
def correlated():
    rng = np.random.default_rng(0)
    base = rng.normal(size=2000)
    return pd.DataFrame(
        {
            "base": base,
            "copy": base + rng.normal(scale=0.05, size=2000),  # майже дублікат
            "independent": rng.normal(size=2000),
        }
    )


def test_vif_is_near_one_for_independent_and_large_for_duplicate(correlated):
    vif = CorrelationAnalyzer(0.9).vif(correlated)

    assert vif["independent"] == pytest.approx(1.0, abs=0.05)
    assert vif["copy"] > 100


def test_vif_table_drops_collinear_feature_and_reduces_vif(correlated):
    table = CorrelationAnalyzer(0.9).vif_table(correlated, dropped=["copy"]).set_index("Ознака")

    assert np.isnan(table.loc["copy", "VIF після"])
    assert table.loc["base", "VIF після"] == pytest.approx(1.0, abs=0.05)


def test_vif_table_shows_extra_drop_variant_and_metrics_count_features_above_threshold(
    correlated,
):
    analyzer = CorrelationAnalyzer(0.9)
    table = analyzer.vif_table(correlated, dropped=[], extra_dropped=["copy"])
    pairs = analyzer.high_pairs(
        analyzer.matrix(correlated, "pearson"), analyzer.matrix(correlated, "spearman")
    )

    metrics = CorrelationAnalyzer.to_metrics(pairs, table, vif_threshold=10.0)

    assert metrics["n_vif_above_threshold_after"] == 2  # base і copy дублюють одна одну
    assert metrics["n_vif_above_threshold_extra_drop"] == 0
    assert metrics["max_vif_extra_drop"] == pytest.approx(1.0, abs=0.05)


def test_extra_drop_check_rejects_candidates_already_dropped_by_schema():
    with pytest.raises(ValueError, match="collinear_columns"):
        ExtraCollinearDropCheck(KICK_SCHEMA, ["VehYear"], PreprocessingSettings(), random_state=0)


def test_normality_metrics_name_most_skewed_feature():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"normal": rng.normal(size=800), "skewed": rng.exponential(size=800)})
    tester = NormalityTester(alpha=0.05, shapiro_sample_size=300, random_state=0)

    metrics = NormalityTester.to_metrics(tester.test(data))

    assert metrics["most_skewed_feature"] == "skewed"
    assert len(tester.qq_sample(data)) == 300


def test_high_pairs_lists_each_strong_pair_once(correlated):
    analyzer = CorrelationAnalyzer(0.9)

    pairs = analyzer.high_pairs(
        analyzer.matrix(correlated, "pearson"), analyzer.matrix(correlated, "spearman")
    )

    assert len(pairs) == 1
    assert {pairs.loc[0, "Ознака 1"], pairs.loc[0, "Ознака 2"]} == {"base", "copy"}


def test_spearman_sees_monotonic_nonlinear_link_better_than_pearson():
    x = np.linspace(0, 10, 200)
    data = pd.DataFrame({"x": x, "exp_x": np.exp(x)})
    analyzer = CorrelationAnalyzer(0.9)

    pearson = analyzer.matrix(data, "pearson").loc["x", "exp_x"]
    spearman = analyzer.matrix(data, "spearman").loc["x", "exp_x"]

    assert spearman == pytest.approx(1.0)
    assert pearson < 0.8


def test_normality_tests_accept_normal_and_reject_skewed_feature():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"normal": rng.normal(size=3000), "skewed": rng.exponential(size=3000)})

    table = NormalityTester(alpha=0.05, shapiro_sample_size=500, random_state=0).test(data)
    verdict = table.set_index("Ознака")["Нормальна"]

    assert verdict["normal"] == "так"
    assert verdict["skewed"] == "ні"
    assert table.set_index("Ознака").loc["skewed", "Асиметрія"] > 1.5
    assert len(table.columns) <= 7  # таблиця має вміститися на сторінку звіту


def test_robust_scaler_keeps_skewness_and_power_transform_removes_it():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {"price": rng.lognormal(8, 0.6, size=3000), "shifted": rng.normal(size=3000)}
    )

    result = ScalingBenchmark().run(data)
    skew = result.skewness

    assert skew.loc["price", METHOD_ROBUST] == pytest.approx(skew.loc["price", METHOD_RAW])
    assert abs(skew.loc["price", METHOD_YEO_JOHNSON]) < 0.1
    # Box-Cox можливий лише для строго додатних ознак.
    assert np.isnan(skew.loc["shifted", METHOD_BOX_COX])
    assert result.n_box_cox_features == 1
    assert ScalingBenchmark.most_skewed(result, 1) == ["price"]


def test_box_cox_skips_degenerate_feature_like_year():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "year": rng.integers(2001, 2011, size=3000).astype(float),
            "price": rng.lognormal(8, 0.6, size=3000),
        }
    )

    result = ScalingBenchmark().run(data)

    assert np.isfinite(result.mean_abs_skew[METHOD_BOX_COX])
    assert np.isfinite(result.skewness.loc["price", METHOD_BOX_COX])
