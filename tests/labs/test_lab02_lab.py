"""Наскрізний тест ЛР2 на синтетичних «даних ЛР1»: усі кроки, артефакти, ключі метрик."""

import json

import matplotlib
import pytest

matplotlib.use("Agg")  # малювання без вікна

from mllabs.core import LabContext
from mllabs.labs.lab02.config import Lab2Config
from mllabs.labs.lab02.lab import Lab2
from mllabs.modeling import ModelSettings

#: Зменшені параметри: на ~1000 рядках повні значення не мають сенсу.
_SMALL_CONFIG = Lab2Config(
    model_settings=ModelSettings(forest_estimators=10, boosting_max_iter=10),
    cv_folds=3,
    cv_n_jobs=1,
    ratio_sweep=(1.0,),
    learning_curve_sizes=(300,),
    bootstrap_repeats=20,
    curve_points=50,
)


@pytest.fixture(scope="module")
def finished_lab(lab1_project):
    context = LabContext(number=2, project=lab1_project, seed=0)
    Lab2(context, _SMALL_CONFIG).run()
    return context


def test_lab_writes_every_planned_figure_and_table(finished_lab):
    figures = {path.stem for path in finished_lab.paths.figures_dir.glob("*.png")}
    tables = {path.stem for path in finished_lab.paths.tables_dir.glob("*.csv")}

    assert figures == {
        "cv_pr_auc",
        "test_metrics_heatmap",
        "test_pr_curves",
        "test_roc_curves",
        "test_confusion_matrices",
        "threshold_shift",
        "augmentation_ratio_sweep",
        "synthetic_memorization",
        "learning_curves",
        "fit_time",
    }
    assert tables == {
        "data_state",
        "models",
        "cv_results",
        "cv_folds",
        "scenario_decomposition",
        "thresholds",
        "augmentation_ratio_sweep",
        "augmentation_ratio_calibration",
        "learning_curves",
        "test_results",
        "test_threshold_effect",
        "augmentation_effect",
        "augmentation_threshold_effect",
        "comparison",
        "model_profile",
    }


def test_lab_writes_all_metric_sections(finished_lab):
    metrics = json.loads(finished_lab.paths.metrics_file.read_text(encoding="utf-8"))

    assert set(metrics) == {
        "data",
        "scenarios",
        "models",
        "cv",
        "test",
        "augmentation",
        "profile",
        "lab3_handoff",
    }
    assert metrics["data"]["manifest_checked"] is True
    assert set(metrics["cv"]["logreg"]) == {"A", "B", "C_smote", "C_adasyn"}
    assert set(metrics["cv"]["decomposition"]["logreg"]) == {"A", "A_missing", "B", "B_mice"}
    assert {"cv_delta", "cv_spread", "test_ci_low", "test_ci_high"} <= set(
        metrics["augmentation"]["effect"]["hist_gradient_boosting"]["smote"]
    )
    overfitting = metrics["augmentation"]["overfitting"]["random_forest"]
    assert {"gap_by_ratio", "synthetic_mean_proba", "smallest_size_delta"} <= set(overfitting)
    assert set(metrics["lab3_handoff"]) >= {"scenario", "random_forest", "hist_gradient_boosting"}


def test_test_set_is_never_augmented_and_counts_are_consistent(finished_lab):
    metrics = json.loads(finished_lab.paths.metrics_file.read_text(encoding="utf-8"))
    scenarios = metrics["scenarios"]

    assert scenarios["A"]["synthetic_rows"] == 0
    assert scenarios["C_smote"]["synthetic_rows"] > 0
    assert scenarios["C_smote"]["minority_share_after_sampling"] == pytest.approx(1 / 3, abs=0.01)
    assert metrics["test"]["n_rows"] == metrics["data"]["test_rows"]


def test_comparison_table_is_compact(finished_lab):
    import pandas as pd

    comparison = pd.read_csv(finished_lab.paths.tables_dir / "comparison.csv")

    assert len(comparison) == 12  # 3 моделі × 4 сценарії
    assert len(comparison.columns) <= 7
