"""Тести каталогу сценаріїв ЛР2 і конфігурації."""

import pytest

from mllabs.data import KICK_SCHEMA
from mllabs.labs.lab01.config import Lab1Config
from mllabs.labs.lab02.config import Lab2Config
from mllabs.labs.lab02.scenarios import ScenarioCatalog, ScenarioId
from mllabs.preprocessing import (
    CategoricalImputation,
    ImputerKind,
    PreprocessingScenario,
    PreprocessorFactory,
    SamplerKind,
)


@pytest.fixture
def catalog():
    return ScenarioCatalog(
        PreprocessorFactory(KICK_SCHEMA, 0),
        sampling_strategy=0.5,
        k_neighbors=5,
        noise_scale=0.05,
        random_state=0,
    )


def test_main_scenarios_are_the_four_task_scenarios(catalog):
    main = [spec.scenario_id for spec in catalog.main()]

    assert main == [ScenarioId.A, ScenarioId.B, ScenarioId.C_SMOTE, ScenarioId.C_ADASYN]
    assert all(spec.on_test for spec in catalog.main())
    assert not catalog.spec(ScenarioId.A_MISSING).on_test
    assert not catalog.spec(ScenarioId.B_MICE).on_test


def test_scenarios_reuse_lab1_recipes(catalog):
    factory = PreprocessorFactory(KICK_SCHEMA, 0)

    assert catalog.spec(ScenarioId.A).options == factory.options_for(PreprocessingScenario.MINIMAL)
    assert catalog.spec(ScenarioId.B).options == factory.options_for(PreprocessingScenario.FULL)
    assert catalog.spec(ScenarioId.C_SMOTE).options == catalog.spec(ScenarioId.B).options
    assert catalog.spec(ScenarioId.C_SMOTE).sampler_kind is SamplerKind.SMOTE


def test_diagnostic_variants_change_exactly_one_brick(catalog):
    minimal = catalog.spec(ScenarioId.A).options
    a_missing = catalog.spec(ScenarioId.A_MISSING).options
    full = catalog.spec(ScenarioId.B).options
    b_mice = catalog.spec(ScenarioId.B_MICE).options

    assert a_missing.categorical_imputation is CategoricalImputation.MISSING_CATEGORY
    assert {**vars(a_missing), "categorical_imputation": None} == {
        **vars(minimal),
        "categorical_imputation": None,
    }
    assert b_mice.imputer is ImputerKind.MICE
    assert {**vars(b_mice), "imputer": None} == {**vars(full), "imputer": None}


def test_sampling_strategy_can_be_overridden_for_the_sweep(catalog):
    spec = catalog.spec(ScenarioId.C_SMOTE)

    assert catalog.sampler(spec).sampling_strategy == 0.5
    assert catalog.sampler(spec, sampling_strategy=1.0).sampling_strategy == 1.0
    assert catalog.sampler(catalog.spec(ScenarioId.B)) is None


def test_augmentation_settings_are_the_same_as_in_lab1():
    lab1, lab2 = Lab1Config(), Lab2Config()

    assert lab2.sampling_strategy == lab1.sampling_strategy
    assert lab2.smote_neighbors == lab1.smote_neighbors
    assert lab2.noise_scale == lab1.noise_scale
    assert lab2.preprocessing == lab1.preprocessing


def test_transformations_describe_augmentation(catalog):
    text = catalog.transformations(catalog.spec(ScenarioId.C_SMOTE))

    assert "SMOTE" in text
    assert "Yeo-Johnson" in text
    assert "SMOTE" not in catalog.transformations(catalog.spec(ScenarioId.B))
