"""Тести ЛР5: подання таблиць звіту і зведені ключі ``metrics.json`` (без моделей і мережі)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mllabs.labs.lab05 import plots, summary
from mllabs.labs.lab05.errors_study import group_effects, overlap_metrics


def _variants() -> pd.DataFrame:
    """Два TFLite-варіанти й еталон Keras для двох категорій (числа вигадані)."""
    rows = []
    for category, auroc in (("alpha", 0.90), ("beta", 0.80)):
        rows.append({"category": category, "variant": "keras", "pixel_auroc": auroc})
        for variant, size, latency, drop in (("fp32", 4.0, 40.0, 0.0), ("int8", 1.0, 20.0, 0.02)):
            rows.append(
                {
                    "category": category,
                    "variant": variant,
                    "size_mb": size,
                    "pixel_auroc": auroc - drop,
                    "image_f1_recalibrated": 0.5,
                    "latency_ms_1": latency,
                    "latency_p90_ms_1": latency + 1,
                    "latency_ms_усі": latency / 2,
                }
            )
    return pd.DataFrame(rows)


def _resolution() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["alpha", "alpha"],
            "input_size": [64, 48],
            "pixel_auroc": [0.9, 0.8],
            "image_auroc": [0.9, 0.7],
            "latency_ms_1": [40.0, 30.0],
            "latency_ms_усі": [20.0, 15.0],
        }
    )


def test_formatted_fixes_digits_marks_missing_and_keeps_other_columns():
    frame = pd.DataFrame({"steps": [64.0, 128.0], "ms": [36.8168, np.nan], "auroc": [0.5, 0.25]})

    result = summary.formatted(frame, {"steps": 0, "ms": 1, "absent": 2})

    assert list(result["steps"]) == ["64", "128"]
    assert list(result["ms"]) == ["36.8", "—"]
    assert result["auroc"].dtype == float
    assert frame["steps"].dtype == float  # вхідну таблицю не змінено


def test_edge_table_shows_dash_for_keras_and_one_digit_latency():
    table = summary.edge_table(_variants())

    keras_row = table[table["Варіант"].str.startswith("Keras")].iloc[0]
    int8_row = table[table["Варіант"] == "int8"].iloc[0]
    assert keras_row["Розмір, МБ"] == keras_row["Latency 1 потік, мс"] == "—"
    assert keras_row["× fp32 (1 потік)"] == "—"
    assert int8_row["Latency 1 потік, мс"] == "20.0"
    assert int8_row["× fp32 (1 потік)"] == "0.50"
    assert int8_row["Розмір, МБ"] == "1.00"


def test_edge_metrics_store_means_ratios_to_fp32_and_postprocess_share():
    postprocess = {"latency_ms_1": 4.0, "latency_p90_ms_1": 5.0, "latency_ms_усі": 1.0}

    metrics = summary.edge_metrics(_variants(), _resolution(), postprocess)

    assert metrics["relative_to_fp32"]["int8"] == {
        "size_mb": 0.25,
        "latency_ms_1": 0.5,
        "latency_ms_усі": 0.5,
    }
    assert metrics["relative_to_fp32"]["fp32"]["latency_ms_1"] == 1.0
    assert "keras" not in metrics["relative_to_fp32"]
    assert metrics["mean"]["int8"]["pixel_auroc_delta"] == pytest.approx(-0.02)
    assert metrics["mean"]["int8"]["latency_ms_1"] == 20.0
    assert metrics["postprocess_ms"]["latency_ms_1"] == 4.0
    assert metrics["postprocess_ms"]["share_of_fp32_1"] == pytest.approx(0.1)
    assert metrics["postprocess_ms"]["share_of_fp32_усі"] == pytest.approx(0.05)
    assert metrics["int8"]["alpha"]["pixel_auroc_delta"] == pytest.approx(-0.02)


def test_similarity_metrics_keep_per_category_means():
    frame = pd.DataFrame(
        {
            "level": ["full"] * 4,
            "method": ["ig"] * 4,
            "category": ["alpha", "alpha", "beta", "beta"],
            "image": [0, 1, 0, 1],
            "spearman": [0.2, 0.4, 0.8, 1.0],
            "ssim": [0.1, 0.1, 0.5, 0.5],
        }
    )

    metrics = summary.similarity_metrics(frame)

    assert metrics["full"]["ig"]["spearman"] == pytest.approx(0.6)
    assert metrics["full"]["ig"]["spearman_by_category"] == pytest.approx(
        {"alpha": 0.3, "beta": 0.9}
    )
    assert metrics["full"]["ig"]["ssim_by_category"] == pytest.approx({"alpha": 0.1, "beta": 0.5})


def test_group_effects_are_addressable_and_skip_untested_comparisons():
    groups = pd.DataFrame(
        {
            "category": ["alpha", "alpha", "усі"],
            "comparison": ["FP vs TN", "FN vs TP", "FP vs TN"],
            "descriptor": ["locality", "locality", "brightness_std"],
            "r": [0.91, np.nan, -0.2],
            "p_holm": [0.001, np.nan, 0.5],
            "n_first": [42, 3, 63],
            "n_second": [8, 47, 68],
        }
    )

    effects = group_effects(groups)

    assert effects["alpha"] == {
        "fp_vs_tn": {"locality": {"r": 0.91, "p_holm": 0.001, "n_first": 42, "n_second": 8}}
    }
    assert effects["усі"]["fp_vs_tn"]["brightness_std"]["r"] == -0.2


def test_overlap_metrics_are_keyed_by_category_and_outcome():
    overlap = pd.DataFrame(
        {
            "category": ["alpha", "alpha"],
            "outcome": ["FP", "FN"],
            "n_fae": [42, 3],
            "also_PaDiM": [38, 0],
            "all_methods": [14, 0],
        }
    )

    metrics = overlap_metrics(overlap)

    assert metrics["alpha"]["FP"] == {"n_fae": 42, "also_PaDiM": 38, "all_methods": 14}
    assert metrics["alpha"]["FN"]["n_fae"] == 3


def test_tradeoff_labels_go_left_when_a_point_on_the_right_is_at_the_same_height():
    points = pd.DataFrame(
        {
            "group": ["a", "a", "a", "a"],
            "label": ["fp32", "fp16", "dynamic", "full"],
            "latency_ms": [36.8, 36.8, 30.2, 19.7],
            "pixel_auroc": [0.9725, 0.9725, 0.9718, 0.9536],
        }
    )

    figure = plots.plot_edge_tradeoff(points, (33.0, 100.0))
    alignment = {text.get_text(): text.get_ha() for text in figure.axes[0].texts}
    plt.close(figure)

    # fp32 і fp16 збігаються (розводяться по вертикалі), праворуч від них нічого немає.
    assert alignment["fp32"] == alignment["fp16"] == "left"
    assert alignment["dynamic"] == "right"  # праворуч на тій самій висоті — fp32/fp16
    assert alignment["full"] == "left"  # сусіди праворуч значно вище
