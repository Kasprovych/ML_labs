"""Наскрізний тест ЛР3 на синтетичних «даних ЛР1»: усі кроки, артефакти, таблиця завдання."""

import json

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # малювання без вікна

from mllabs.core import LabContext
from mllabs.labs.lab03.config import Lab3Config
from mllabs.labs.lab03.lab import Lab3
from mllabs.labs.lab03.search_space import ParameterRange, SearchSpace
from mllabs.labs.lab03.summary import METHOD_COLUMNS
from mllabs.modeling import ModelSettings

#: Мінімальний конфіг: сітка 2×2, N = 3, фолди 2/3, однопроцесне виконання.
_SMALL_CONFIG = Lab3Config(
    model_settings=ModelSettings(boosting_max_iter=5),
    dev_sample_size=600,
    inner_folds=2,
    outer_folds=3,
    search_space=SearchSpace(
        ranges=(
            ParameterRange(
                "learning_rate", "крок", 0.05, 0.2, (0.05, 0.2), log=True, integer=False
            ),
            ParameterRange("max_leaf_nodes", "листки", 4, 16, (4, 16), log=True, integer=True),
            # max_features немає в describe() фабрики — перевіряємо, що baseline його все одно має.
            ParameterRange(
                "max_features", "ознаки", 0.5, 1.0, (0.5, 1.0), log=False, integer=False
            ),
        )
    ),
    random_iterations=(2, 3),
    optuna_trials=3,
    optuna_startup_trials=1,
    search_n_jobs=1,
    optuna_fold_n_jobs=1,
    optuna_threads_per_worker=1,
    main_process_threads=1,
    leakage_folds=3,
    cv_n_jobs=1,
    cv_threads_per_worker=1,
    leakage_sample_sizes=(400,),
    bootstrap_repeats=20,
)


@pytest.fixture(scope="module")
def finished_lab(lab1_project):
    context = LabContext(number=3, project=lab1_project, seed=0)
    Lab3(context, _SMALL_CONFIG).run()
    return context


def test_lab_writes_every_planned_figure_and_table(finished_lab):
    figures = {path.stem for path in finished_lab.paths.figures_dir.glob("*.png")}
    tables = {path.stem for path in finished_lab.paths.tables_dir.glob("*.csv")}

    assert figures == {
        "quality_vs_time",
        "search_convergence",
        "nested_inner_vs_outer",
        "leakage_comparison",
        "leakage_by_size",
        "search_space_samples",
    }
    assert tables == {
        "data_state",
        "baseline",
        "search_space",
        "optimization_methods",
        "optimization_cost",
        "optimization_quality",
        "nested_pairwise",
        "best_params",
        "search_history",
        "nested_folds",
        "leakage_comparison",
        "leakage_folds",
        "leakage_holdout",
        "leakage_by_size",
        "test_results",
    }


def test_methods_table_has_exactly_the_assignment_columns(finished_lab):
    table = pd.read_csv(finished_lab.paths.tables_dir / "optimization_methods.csv")

    assert tuple(table.columns) == METHOD_COLUMNS
    assert len(table) == 5  # baseline + Grid + 2 × Random + Optuna
    assert table.loc[1, "Кількість ітерацій / оцінок"] == "8 / 16"


def test_report_tables_fit_seven_columns(finished_lab):
    for name in (
        "optimization_cost",
        "optimization_quality",
        "nested_folds",
        "leakage_comparison",
        "leakage_by_size",
        "test_results",
        "best_params",
    ):
        table = pd.read_csv(finished_lab.paths.tables_dir / f"{name}.csv")
        assert len(table.columns) <= 7, name


def test_lab_writes_all_metric_sections(finished_lab):
    metrics = json.loads(finished_lab.paths.metrics_file.read_text(encoding="utf-8"))

    assert {
        "data",
        "metric",
        "model",
        "search_space",
        "baseline",
        "config",
        "leakage",
        "methods",
        "methods_summary",
        "timing",
    } <= set(metrics)
    assert set(metrics["methods"]) == {"grid", "random_2", "random_3", "optuna_3"}
    assert "test_pr_auc" in metrics["baseline"]
    assert {"headline", "by_size", "correct", "leak_fit_all", "leak_smote"} <= set(
        metrics["leakage"]
    )
