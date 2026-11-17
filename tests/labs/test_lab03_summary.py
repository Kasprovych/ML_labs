"""Тести зведених таблиць ЛР3: оптимізм у двох формах, вартість, підсумок без «переможця»."""

import numpy as np
import pytest

from mllabs.labs.lab03.baseline import BaselineResult
from mllabs.labs.lab03.nested import NestedFoldResult, NestedResult
from mllabs.labs.lab03.search import EvaluationRecord, SearchOutcome
from mllabs.labs.lab03.summary import (
    BASELINE_LABEL,
    METHOD_COLUMNS,
    MethodReport,
    _methods_summary,
    cost_table,
    methods_table,
    quality_table,
)

_OUTER = (0.40, 0.50, 0.45)
_BASELINE_INNER = (0.38, 0.47, 0.44)  # зсув baseline inner − outer: −0.02, −0.03, −0.01


def _baseline() -> BaselineResult:
    return BaselineResult(
        inner_scores=(0.44, 0.45, 0.46),
        inner_wall_seconds=1.0,
        inner_compute_seconds=3.0,
        outer_scores=_OUTER,
        outer_roc_auc=(0.7, 0.7, 0.7),
        outer_train_scores=(0.9, 0.9, 0.9),
        inner_in_outer_scores=_BASELINE_INNER,
    )


def _report(key: str, inner: tuple[float, ...], outer: tuple[float, ...]) -> MethodReport:
    record = EvaluationRecord(0, {"learning_rate": 0.1}, 0.46, 0.01, (0.46,), 6.0)
    outcome = SearchOutcome(
        key, key, 20, 60, 10.0, 6.0, {"learning_rate": 0.1}, 0.46, 0.01, (record,), 3, 3
    )
    folds = tuple(
        NestedFoldResult(k, i, o, 0.7, {"learning_rate": 0.1}, 1.0, 0.1)
        for k, (i, o) in enumerate(zip(inner, outer, strict=True))
    )
    return MethodReport(outcome, NestedResult(key, key, folds, 5.0))


def test_baseline_fold_shift_is_inner_minus_outer_per_fold():
    np.testing.assert_allclose(_baseline().fold_shift, [-0.02, -0.03, -0.01])


def test_selection_optimism_subtracts_baseline_shift_on_the_same_fold():
    # Сирий оптимізм методу: 0.39−0.40, 0.49−0.50, 0.45−0.45 = −0.01, −0.01, 0.
    report = _report("m", (0.39, 0.49, 0.45), _OUTER)

    raw_mean, _ = report.nested.optimism()
    selection_mean, _ = report.nested.selection_optimism(_baseline().fold_shift)

    assert raw_mean == pytest.approx(-0.02 / 3)
    # Від'ємний сирий оптимізм стає додатним після вирахування зсуву baseline.
    assert selection_mean == pytest.approx((0.01 + 0.02 + 0.01) / 3)


def test_quality_table_has_baseline_row_and_both_optimism_forms():
    table = quality_table(_baseline(), [_report("m", (0.39, 0.49, 0.45), _OUTER)])

    assert table["Метод"].tolist() == [BASELINE_LABEL, "m"]
    assert {"Оптимізм сирий ± s", "Оптимізм відбору ± s"} <= set(table.columns)
    assert table.loc[0, "Оптимізм сирий ± s"].startswith("-0.0200")
    assert table.loc[1, "Оптимізм відбору ± s"].startswith("0.0133")
    assert len(table.columns) <= 7


def test_methods_table_keeps_exactly_the_five_assignment_columns():
    table = methods_table(_baseline(), [_report("m", (0.39, 0.49, 0.45), _OUTER)])

    assert tuple(table.columns) == METHOD_COLUMNS


def test_cost_table_shows_parallel_mode_of_each_method():
    table = cost_table([_report("m", (0.39, 0.49, 0.45), _OUTER)])

    assert table.loc[0, "Процесів × потоків"] == "3 × 3"


def test_methods_summary_names_no_quality_winner_and_counts_distinguishable_pairs():
    reports = [
        _report("a", (0.39, 0.49, 0.45), _OUTER),
        _report("b", (0.40, 0.50, 0.46), (0.41, 0.50, 0.46)),
    ]

    summary = _methods_summary(_baseline(), reports)

    assert not {"best_nested", "best_inner", "max_optimism"} & set(summary)
    assert summary["pairs_total"] == 3
    assert summary["pairs_vs_baseline_total"] == 2
    assert summary["pairs_between_methods_total"] == 1
    assert summary["time_basis"] == "wall_seconds"
