"""Тести ядра: шляхи, артефакти, шаблонний метод Lab.run()."""

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # малювання без вікна — обов'язково для тестів

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mllabs.core import Lab, LabContext, Step
from mllabs.core.paths import ProjectPaths, lab_slug


def test_lab_slug_pads_number():
    assert lab_slug(3) == "lab03"
    with pytest.raises(ValueError):
        lab_slug(0)


def test_artifact_store_saves_figure_table_and_merges_metrics(project: ProjectPaths):
    context = LabContext(number=1, project=project)
    figure, axes = plt.subplots()
    axes.plot([1, 2, 3])

    figure_file = context.artifacts.save_figure(figure, "line")
    table_file = context.artifacts.save_table(pd.DataFrame({"a": [0.123456]}), "numbers")
    context.artifacts.save_metrics({"f1": np.float64(0.5)})
    context.artifacts.save_metrics({"model": {"auc": 0.9}})

    assert figure_file.is_file()
    assert table_file.read_text(encoding="utf-8").splitlines() == ["a", "0.1235"]
    assert context.artifacts.load_metrics() == {"f1": 0.5, "model": {"auc": 0.9}}


def test_lab_runs_steps_in_declared_order(project: ProjectPaths):
    executed: list[str] = []

    class DemoLab(Lab):
        NUMBER = 1
        TITLE = "Демо"

        def steps(self) -> Sequence[Step]:
            return [
                Step("перший", lambda: executed.append("first")),
                Step("другий", lambda: executed.append("second")),
            ]

    DemoLab(LabContext(number=1, project=project)).run()

    assert executed == ["first", "second"]
