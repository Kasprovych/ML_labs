"""Наскрізний тест ЛР1 на синтетичних даних: усі кроки, артефакти, дані для ЛР2–3."""

import dataclasses
import json

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # малювання без вікна

from mllabs.core import LabContext
from mllabs.data import ProcessedDatasetRepository
from mllabs.labs.lab01.config import Lab1Config
from mllabs.labs.lab01.lab import Lab1
from mllabs.preprocessing import PreprocessingSettings

#: Зменшені параметри: на 1200 рядках повні значення не мають сенсу.
_SMALL_CONFIG = Lab1Config(
    high_cardinality_threshold=5,  # у синтетичних даних високопотужні ознаки мають ≥ 6 категорій
    shapiro_sample_size=500,
    isolation_forest_estimators=50,
    imputation_sample_size=400,
    preprocessing=PreprocessingSettings(onehot_min_frequency=5),
    mutual_information_sample_size=800,
    pca_plot_sample_size=200,
)


class _FakeSource:
    """Фальшиве джерело даних замість OpenML (без мережі)."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def load(self) -> pd.DataFrame:
        return self._frame.copy()


@pytest.fixture(scope="module")
def finished_lab(tmp_path_factory, raw_kick_frame):
    from mllabs.core.paths import ProjectPaths

    project = ProjectPaths(root=tmp_path_factory.mktemp("project"))
    context = LabContext(number=1, project=project, seed=0)
    Lab1(context, _SMALL_CONFIG, source=_FakeSource(raw_kick_frame)).run()
    return context


def test_lab_writes_every_planned_figure_and_table(finished_lab):
    figures = {path.stem for path in finished_lab.paths.figures_dir.glob("*.png")}
    tables = {path.stem for path in finished_lab.paths.tables_dir.glob("*.csv")}

    assert len(figures) == 15
    assert len(tables) == 23
    assert {"engineered_features_mi", "augmentation_pca", "qq_plots"} <= figures
    assert {
        "preprocessing_summary",
        "feature_engineering_comparison",
        "full_pipeline_vif",
    } <= tables


def test_lab_writes_all_metric_sections(finished_lab):
    metrics = json.loads(finished_lab.paths.metrics_file.read_text(encoding="utf-8"))

    assert set(metrics) == {
        "dataset",
        "split",
        "correlation",
        "normality",
        "outliers",
        "imputation",
        "encoding",
        "scaling",
        "augmentation",
        "feature_engineering",
        "selection",
    }
    decision = metrics["selection"]["categorical_imputation"]
    assert {"delta", "spread", "n_features_simple", "n_features_complex"} <= set(decision)
    assert {"chosen", "rule_verdict", "exception"} <= set(decision)
    assert metrics["split"]["test_rows"] == 240
    assert set(metrics["augmentation"]) == {"none", "smote", "adasyn", "gaussian_noise"}


def test_lab_saves_clean_untransformed_split_for_next_labs(finished_lab, raw_kick_frame):
    repository = ProcessedDatasetRepository(finished_lab.project.data_processed_dir, "kick")

    split = repository.load()

    assert len(split.x_train) + len(split.x_test) == len(raw_kick_frame)
    assert split.x_train["Model"].str.startswith("MODEL").all()  # категорії не закодовані
    assert split.x_train["VehBCost"].isna().any() or split.x_train["VehBCost"].gt(1).all()
    assert "WheelTypeID" not in split.x_train.columns


def test_lab_records_actual_preprocessing_settings_next_to_schema(finished_lab):
    repository = ProcessedDatasetRepository(finished_lab.project.data_processed_dir, "kick")

    manifest = repository.load_preprocessing()

    assert PreprocessingSettings.from_dict(manifest["settings"]) == _SMALL_CONFIG.preprocessing
    assert manifest["random_state"] == 0
    assert manifest["scenarios"]["full"]["scaler"] == "power"


def test_lab1_config_does_not_duplicate_preprocessing_settings():
    own_fields = {field.name for field in dataclasses.fields(Lab1Config)}
    settings_fields = {field.name for field in dataclasses.fields(PreprocessingSettings)}

    # Жодне поле налаштувань препроцесингу не продубльоване в конфігурації ЛР1 ...
    assert not own_fields & settings_fields
    assert not {"knn_imputer_neighbors", "onehot_min_frequency"} & own_fields
    # ... а типові значення ЛР1 — ті самі, з якими ЛР2–3 створять фабрику.
    assert Lab1Config().preprocessing == PreprocessingSettings()
