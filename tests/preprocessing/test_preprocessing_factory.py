"""Тести фабрики пайплайнів препроцесингу та проксі-оцінювача."""

import json
from dataclasses import replace

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from mllabs.data import KICK_SCHEMA, RawDataCleaner, StratifiedSplitter
from mllabs.preprocessing import (
    MODEL_STEP,
    SAMPLER_STEP,
    CategoricalImputation,
    HighCardinalityEncoding,
    ImputerKind,
    PreprocessingMismatchError,
    PreprocessingOptions,
    PreprocessingScenario,
    PreprocessingSettings,
    PreprocessorFactory,
    ProxyModelEvaluator,
    SamplerFactory,
    SamplerKind,
    ScalerKind,
    build_model_pipeline,
)


@pytest.fixture(scope="module")
def prepared(raw_kick_frame):
    cleaning = RawDataCleaner(KICK_SCHEMA, 0.5).clean(raw_kick_frame)
    split = StratifiedSplitter(0.25, random_state=0).split(cleaning.data, "IsBadBuy")
    factory = PreprocessorFactory(
        cleaning.schema, random_state=0, settings=PreprocessingSettings(onehot_min_frequency=5)
    )
    return factory, split


@pytest.mark.parametrize("scenario", list(PreprocessingScenario))
def test_scenarios_produce_finite_matrix_with_same_width_on_train_and_test(prepared, scenario):
    factory, split = prepared
    pipeline = factory.build(scenario)

    train_matrix = pipeline.fit_transform(split.x_train, split.y_train)
    test_matrix = pipeline.transform(split.x_test)

    assert train_matrix.shape[0] == len(split.x_train)
    assert test_matrix.shape == (len(split.x_test), train_matrix.shape[1])
    assert np.isfinite(train_matrix).all()
    assert np.isfinite(test_matrix).all()


def test_factory_returns_unfitted_pipeline(prepared):
    factory, _ = prepared

    pipeline = factory.build(PreprocessingScenario.FULL)

    assert not hasattr(pipeline.named_steps["columns"], "transformers_")


@pytest.mark.parametrize(
    "change",
    [
        {"imputer": ImputerKind.KNN},
        {"imputer": ImputerKind.MICE},
        {"categorical_imputation": CategoricalImputation.MODE},
        {"high_cardinality_encoding": HighCardinalityEncoding.FREQUENCY},
        {"high_cardinality_encoding": HighCardinalityEncoding.ONEHOT},
        {"scaler": ScalerKind.ROBUST},
        {"add_polynomial_features": True},
    ],
)
def test_every_option_builds_a_working_pipeline(prepared, change):
    factory, split = prepared
    options = replace(PreprocessingOptions(), **change)

    matrix = factory.build_from_options(options).fit(split.x_train, split.y_train)

    assert matrix.transform(split.x_test).shape[0] == len(split.x_test)


def test_collinear_columns_are_dropped_only_in_full_scenario(prepared):
    factory, _ = prepared
    full = factory.numeric_columns(factory.options_for(PreprocessingScenario.FULL))
    minimal = factory.numeric_columns(factory.options_for(PreprocessingScenario.MINIMAL))

    assert "VehYear" not in full
    assert "VehYear" in minimal
    assert "MMRCurrentRetailCleanPrice" not in full


def test_full_scenario_does_not_duplicate_wheel_type_missing_indicator(prepared):
    factory, split = prepared
    options = factory.options_for(PreprocessingScenario.FULL)
    assert options.categorical_imputation is CategoricalImputation.MISSING_CATEGORY
    pipeline = factory.build_from_options(options).fit(split.x_train, split.y_train)

    names = list(pipeline.named_steps["columns"].get_feature_names_out())

    # Пропуск WheelType уже є колонкою One-Hot «WheelType_MISSING»; індикатор був би її копією.
    assert "WheelType_MISSING" in names
    assert "wheel_type_missing" not in names
    assert "mmr_missing" in names  # пропуск цін MMR іншою колонкою не представлений


def test_wheel_type_indicator_is_kept_when_missing_values_are_filled_by_mode(prepared):
    factory, _ = prepared
    options = replace(
        factory.options_for(PreprocessingScenario.FULL),
        categorical_imputation=CategoricalImputation.MODE,
    )

    # Без категорії MISSING інформацію про пропуск несе лише індикатор.
    assert "wheel_type_missing" in factory.indicator_columns(options)


def test_full_numeric_block_has_no_exact_linear_dependencies(prepared):
    factory, split = prepared
    options = factory.options_for(PreprocessingScenario.FULL)
    pipeline = factory.build(PreprocessingScenario.FULL).fit(split.x_train, split.y_train)
    names = list(pipeline.named_steps["columns"].get_feature_names_out())
    block = [*factory.numeric_columns(options), *factory.indicator_columns(options)]

    matrix = pipeline.transform(split.x_train)[:, [names.index(name) for name in block]]

    # Повний ранг: жодна числова колонка не є лінійною комбінацією інших
    # (як була б «AcqRetAvg − AcqAucAvg» чи «ціна − середня ціна моделі»).
    assert np.linalg.matrix_rank(matrix) == len(block)
    assert not {"retail_margin", "log_warranty", "VehBCost_mean_by_Model"} & set(names)


def test_settings_round_trip_through_plain_dict():
    settings = PreprocessingSettings(mice_max_iter=7, polynomial_features=("VehOdo",))

    assert PreprocessingSettings.from_dict(settings.to_dict()) == settings


def test_manifest_records_settings_and_both_scenarios(prepared):
    factory, _ = prepared

    manifest = factory.manifest()

    assert manifest["settings"]["onehot_min_frequency"] == 5
    assert set(manifest["scenarios"]) == {"minimal", "full"}
    assert manifest["scenarios"]["minimal"]["add_row_features"] is False


def test_full_pipeline_fitted_on_train_ignores_categories_seen_only_in_test(prepared):
    factory, split = prepared
    pipeline = factory.build(PreprocessingScenario.FULL).fit(split.x_train, split.y_train)
    test = split.x_test.copy()
    test["Model"] = "NEVER_SEEN_MODEL"
    test["Color"] = "NEVER_SEEN_COLOR"

    matrix = pipeline.transform(test)

    assert np.isfinite(matrix).all()


def test_fitting_on_train_does_not_depend_on_test_rows(prepared):
    factory, split = prepared
    pipeline = factory.build(PreprocessingScenario.FULL)
    fitted = clone(pipeline).fit(split.x_train, split.y_train)
    probe = split.x_test.head(5)
    before = fitted.transform(probe)

    # Навіть якщо тестові дані зіпсовані, трансформер, навчений на train, їх не «вчиться».
    corrupted = split.x_test.assign(VehBCost=split.x_test["VehBCost"] * 100)
    fitted.transform(corrupted)

    np.testing.assert_array_equal(fitted.transform(probe), before)


def test_proxy_evaluator_returns_metrics_in_valid_range_with_and_without_sampler(prepared):
    factory, split = prepared
    evaluator = ProxyModelEvaluator(n_splits=3, max_iter=500, random_state=0, positive_label=1)
    sampler = SamplerFactory(0.5, 5, 0.05, 0).create(SamplerKind.GAUSSIAN_NOISE)
    preprocessor = factory.build(PreprocessingScenario.FULL)

    plain = evaluator.evaluate(preprocessor, split.x_train, split.y_train)
    augmented = evaluator.evaluate(preprocessor, split.x_train, split.y_train, sampler=sampler)

    for scores in (plain, augmented):
        assert 0.0 <= scores.pr_auc <= 1.0
        assert 0.5 < scores.roc_auc <= 1.0  # синтетична ціль залежить від ознак
        assert scores.n_features > 0
    # Семплер додає рядки рідкісного класу -> модель позначає більше позитивних.
    assert augmented.recall >= plain.recall


def test_check_manifest_accepts_own_manifest_after_json_round_trip(prepared):
    factory, _ = prepared
    recorded = json.loads(json.dumps(factory.manifest()))

    factory.check_manifest(recorded)  # не кидає винятку


def test_check_manifest_reports_every_differing_setting(prepared):
    factory, _ = prepared
    recorded = json.loads(json.dumps(factory.manifest()))
    recorded["settings"]["mice_max_iter"] = 10
    recorded["scenarios"]["full"]["add_group_aggregates"] = False
    recorded["random_state"] = 7  # зерно не є рішенням препроцесингу

    with pytest.raises(PreprocessingMismatchError, match="settings") as error:
        factory.check_manifest(recorded)
    assert "scenarios" in str(error.value)


def test_model_pipeline_is_flat_with_sampler_before_model_and_leaves_inputs_unfitted(prepared):
    factory, split = prepared
    preprocessor = factory.build(PreprocessingScenario.FULL)
    model = LogisticRegression(max_iter=500)
    sampler = SamplerFactory(0.5, 5, 0.05, 0).create(SamplerKind.SMOTE)

    pipeline = build_model_pipeline(preprocessor, model, sampler)
    pipeline.fit(split.x_train, split.y_train)

    names = [name for name, _ in pipeline.steps]
    assert names == [*[name for name, _ in preprocessor.steps], SAMPLER_STEP, MODEL_STEP]
    assert pipeline.predict_proba(split.x_test).shape == (len(split.x_test), 2)
    assert not hasattr(model, "coef_")  # передані об'єкти клонуються, а не навчаються


def test_evaluator_uses_identical_folds_for_every_variant_and_reports_each_fold(prepared):
    factory, split = prepared
    evaluator = ProxyModelEvaluator(n_splits=4, max_iter=500, random_state=0, positive_label=1)

    first = evaluator.splits(split.x_train, split.y_train)
    second = evaluator.splits(split.x_train, split.y_train)
    full = evaluator.evaluate(
        factory.build(PreprocessingScenario.FULL), split.x_train, split.y_train
    )

    # Розбиття не залежить від препроцесора — пари фолдів у різних варіантів збігаються.
    for (train_a, test_a), (train_b, test_b) in zip(first, second, strict=True):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
    assert len(full.fold_pr_auc) == 4
    assert full.pr_auc == pytest.approx(np.mean(full.fold_pr_auc))
    assert full.pr_auc_std == pytest.approx(np.std(full.fold_pr_auc, ddof=1))


def test_full_scenario_keeps_retail_prices_and_adds_group_deviations(prepared):
    factory, _ = prepared
    options = factory.options_for(PreprocessingScenario.FULL)

    columns = factory.numeric_columns(options)

    assert options.add_group_aggregates
    # Роздрібні ціни лишаються: за правилом вибору з ними краще (Δ > s).
    assert "MMRAcquisitionRetailAveragePrice" in columns
    assert "MMRCurrentRetailAveragePrice" in columns
    assert "MMRAcquisitonRetailCleanPrice" not in columns
    assert "VehBCost_dev_by_Model" in columns
    assert "VehBCost_mean_by_Model" not in columns  # лише відхилення, без статистики групи
