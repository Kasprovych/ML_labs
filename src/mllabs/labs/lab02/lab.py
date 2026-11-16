"""Лабораторна робота №2: Порівняння традиційних ML-алгоритмів.

Цей модуль лише ОРКЕСТРУЄ кроки: що і в якому порядку виконується.
Самі обчислення живуть в окремих модулях цього пакета і в спільних пакетах
``mllabs.preprocessing`` (сценарії даних ЛР1) та ``mllabs.modeling`` (моделі,
метрики, крос-валідація), а параметри — у ``config.py``.

Головне правило: кроки 1–6 працюють лише з навчальною вибіркою. Тестова
вибірка з'являється тільки в кроці 7 — для заздалегідь заданого переліку
конфігурацій, і жодне рішення за нею не ухвалюється.
"""

from __future__ import annotations

from collections.abc import Sequence

from mllabs.core import Lab, LabContext, Step
from mllabs.data import DatasetSchema, DatasetSplit, ProcessedDatasetRepository
from mllabs.labs.lab02 import plots
from mllabs.labs.lab02.augmentation_analysis import (
    AugmentationEffect,
    LearningCurveStudy,
    RatioSweep,
    memorization_frame,
    merge_overfitting_metrics,
)
from mllabs.labs.lab02.comparison import GridResult, ScenarioComparison
from mllabs.labs.lab02.config import Lab2Config
from mllabs.labs.lab02.data_state import DataStateReporter
from mllabs.labs.lab02.runner import ConfigurationRunner
from mllabs.labs.lab02.scenarios import ScenarioCatalog, ScenarioId
from mllabs.labs.lab02.summary import (
    ComparisonTableBuilder,
    ModelProfileBuilder,
    lab3_handoff,
    models_table,
)
from mllabs.labs.lab02.test_evaluation import HoldoutEvaluator, HoldoutResult
from mllabs.modeling import (
    BootstrapInterval,
    CrossValidator,
    CurveBuilder,
    F1ThresholdSelector,
    MetricsCalculator,
    ModelFactory,
    ModelKind,
    PairedBootstrap,
)
from mllabs.preprocessing import PreprocessorFactory


class Lab2(Lab):
    """Порівняння традиційних ML-алгоритмів на сценаріях даних А, Б, В.

    Args:
        context: Контекст виконання (шляхи, артефакти, логер).
        config: Параметри експерименту; за замовчуванням — стандартні.
    """

    NUMBER = 2
    TITLE = "Порівняння традиційних ML-алгоритмів"

    # Стан, який кроки передають один одному (лише оголошення типів, як поля в Java).
    _schema: DatasetSchema
    _split: DatasetSplit
    _catalog: ScenarioCatalog
    _models: ModelFactory
    _calculator: MetricsCalculator
    _runner: ConfigurationRunner
    _comparison: ScenarioComparison
    _grid: GridResult
    _holdout: HoldoutResult
    _test_intervals: dict[tuple[ModelKind, ScenarioId], BootstrapInterval]
    _test_flat: dict[tuple[ModelKind, ScenarioId], dict[str, float]]
    _sweep_metrics: dict[str, dict]
    _curve_metrics: dict[str, dict]

    def __init__(self, context: LabContext, config: Lab2Config | None = None) -> None:
        super().__init__(context)
        self._config = config or Lab2Config()
        self._artifacts = context.artifacts
        self._seed = context.seed

    def steps(self) -> Sequence[Step]:
        """Кроки відповідають пунктам завдання з ASSIGNMENT.md."""
        return [
            Step("1. Дані ЛР1 і звірка препроцесингу", self._load_data),
            Step("2. Сценарії даних і стан даних", self._describe_scenarios),
            Step("3. Крос-валідація: моделі × сценарії", self._cross_validate),
            Step("4. Поріг класифікації на OOF-прогнозах", self._report_thresholds),
            Step("5. Аугментація: інтенсивність і синтетика", self._sweep_augmentation),
            Step("6. Аугментація: криві навчання", self._learning_curves),
            Step("7. Фінальна оцінка на тестовій вибірці", self._evaluate_on_test),
            Step("8. Порівняльна таблиця і профіль моделей", self._summarize),
        ]

    # ------------------------------------------------------------------ 1–2. Дані

    def _load_data(self) -> None:
        """Читає train/test ЛР1 і перевіряє, що препроцесинг той самий, що в ЛР1 (R1)."""
        repository = ProcessedDatasetRepository(
            self._context.project.data_processed_dir, self._config.dataset_name
        )
        self._schema = repository.load_schema()
        self._split = repository.load()
        preprocessors = PreprocessorFactory(self._schema, self._seed, self._config.preprocessing)
        preprocessors.check_manifest(repository.load_preprocessing())

        config = self._config
        self._catalog = ScenarioCatalog(
            preprocessors,
            sampling_strategy=config.sampling_strategy,
            k_neighbors=config.smote_neighbors,
            noise_scale=config.noise_scale,
            random_state=self._seed,
        )
        self._models = ModelFactory(config.model_settings, self._seed)
        self._calculator = MetricsCalculator(self._schema.positive_label)
        self._runner = ConfigurationRunner(
            self._catalog,
            self._models,
            CrossValidator(
                config.cv_folds, self._seed, self._schema.positive_label, config.cv_n_jobs
            ),
            self._calculator,
            F1ThresholdSelector(self._schema.positive_label),
            self._split.x_train,
            self._split.y_train,
        )
        self._artifacts.save_metrics(
            {
                "data": {
                    "train_rows": len(self._split.x_train),
                    "test_rows": len(self._split.x_test),
                    "n_raw_features": len(self._schema.features),
                    "train_minority_share": float(
                        self._calculator.is_positive(self._split.y_train).mean()
                    ),
                    "majority_baseline_accuracy_train": self._calculator.majority_accuracy(
                        self._split.y_train
                    ),
                    "manifest_checked": True,
                }
            }
        )

    def _describe_scenarios(self) -> None:
        """Стан даних у кожному сценарії: ознаки, рядки, частка класу 1 (R3–R5, R8)."""
        reporter = DataStateReporter(self._catalog, self._schema.positive_label)
        specs = [
            self._catalog.spec(scenario)
            for scenario in (
                ScenarioId.A,
                ScenarioId.A_MISSING,
                ScenarioId.B,
                ScenarioId.B_MICE,
                ScenarioId.C_SMOTE,
                ScenarioId.C_ADASYN,
            )
        ]
        states = reporter.describe(specs, self._split.x_train, self._split.y_train)
        self._artifacts.save_table(reporter.table(states), "data_state", float_format="%.2f")
        self._artifacts.save_metrics({"scenarios": reporter.to_metrics(states)})

    # ------------------------------------------------------------------ 3–4. CV

    def _cross_validate(self) -> None:
        """Три моделі × чотири сценарії + діагностика А_missing і Б_MICE (R2, R6)."""
        config = self._config
        self._comparison = ScenarioComparison(
            self._runner,
            self._catalog,
            config.models,
            config.decomposition_models,
            config.sampling_strategy,
        )
        self._grid = self._comparison.run()
        comparison, grid = self._comparison, self._grid

        self._artifacts.save_table(models_table(self._models, config.models), "models")
        self._artifacts.save_table(comparison.cv_results_table(grid), "cv_results")
        self._artifacts.save_table(comparison.folds_table(grid), "cv_folds")
        self._artifacts.save_table(comparison.decomposition_table(grid), "scenario_decomposition")
        self._artifacts.save_figure(
            plots.plot_cv_pr_auc(
                comparison.plot_frame(grid),
                self._model_labels(),
                self._scenario_labels(),
                chance_level=float(self._calculator.is_positive(self._split.y_train).mean()),
            ),
            "cv_pr_auc",
        )
        profile = ModelProfileBuilder(self._catalog, config.models)
        self._artifacts.save_figure(
            plots.plot_fit_time(
                profile.fit_time_frame(grid), self._model_labels(), self._scenario_labels()
            ),
            "fit_time",
        )
        self._artifacts.save_metrics(
            {
                "models": profile.models_metrics(self._models, grid),
                "cv": comparison.to_metrics(grid),
            }
        )

    def _report_thresholds(self) -> None:
        """Поріг з максимальним F1 на OOF-прогнозах train і його ефект (R6, R10)."""
        model = self._config.threshold_shift_model
        builder = CurveBuilder(self._config.curve_points, self._schema.positive_label)
        scenarios = (ScenarioId.B, ScenarioId.C_SMOTE)
        curves = {
            self._catalog.label(scenario): builder.by_threshold(
                self._grid.get(model, scenario).target, self._grid.get(model, scenario).oof_proba
            )
            for scenario in scenarios
        }
        thresholds = {
            self._catalog.label(scenario): self._grid.get(model, scenario).threshold
            for scenario in scenarios
        }
        self._artifacts.save_table(self._comparison.thresholds_table(self._grid), "thresholds")
        self._artifacts.save_figure(
            plots.plot_threshold_shift(curves, thresholds, ModelFactory.short_name(model)),
            "threshold_shift",
        )

    # ------------------------------------------------------------------ 5–6. Аугментація

    def _sweep_augmentation(self) -> None:
        """Частота SMOTE × місткість моделі; запам'ятовування синтетики (R11)."""
        config = self._config
        sweep_study = RatioSweep(
            self._runner, config.models, config.ratio_sweep, config.sampling_strategy
        )
        sweep = sweep_study.run(self._grid)
        self._artifacts.save_table(
            sweep_study.table(sweep, self._calculator), "augmentation_ratio_sweep"
        )
        self._artifacts.save_table(
            sweep_study.calibration_table(sweep), "augmentation_ratio_calibration"
        )
        self._artifacts.save_figure(
            plots.plot_ratio_sweep(sweep_study.plot_frame(sweep), self._model_labels()),
            "augmentation_ratio_sweep",
        )
        self._artifacts.save_figure(
            plots.plot_synthetic_memorization(
                memorization_frame(self._grid, config.models, self._calculator),
                self._model_labels(),
            ),
            "synthetic_memorization",
        )
        self._sweep_metrics = sweep_study.to_metrics(sweep, self._calculator)

    def _learning_curves(self) -> None:
        """Криві навчання Б проти В-SMOTE: ефект аугментації від обсягу даних (R11)."""
        config = self._config
        study = LearningCurveStudy(
            self._runner,
            config.learning_curve_models,
            config.learning_curve_sizes,
            config.sampling_strategy,
        )
        results = study.run(self._grid)
        self._artifacts.save_table(study.table(results, self._catalog), "learning_curves")
        self._artifacts.save_figure(
            plots.plot_learning_curves(
                study.plot_frame(results),
                self._model_labels(config.learning_curve_models),
                self._scenario_labels(LearningCurveStudy.SCENARIOS),
            ),
            "learning_curves",
        )
        self._curve_metrics = study.to_metrics(results)

    # ------------------------------------------------------------------ 7. Test

    def _evaluate_on_test(self) -> None:
        """Єдине використання test: 12 заздалегідь заданих конфігурацій (R6, R10, P1)."""
        config = self._config
        evaluator = HoldoutEvaluator(
            self._catalog, self._models, self._calculator, config.cv_n_jobs
        )
        self._holdout = evaluator.evaluate(
            self._grid,
            config.models,
            self._catalog.main(),
            train=(self._split.x_train, self._split.y_train),
            test=(self._split.x_test, self._split.y_test),
        )
        self._test_intervals = evaluator.bootstrap(
            self._holdout,
            config.models,
            PairedBootstrap(config.bootstrap_repeats, config.confidence_level, self._seed),
        )
        holdout = self._holdout
        pr_curves, roc_curves = evaluator.curves(
            holdout, CurveBuilder(config.curve_points, self._schema.positive_label)
        )
        # Модель для матриць помилок обрано за CV (найкращий PR-AUC), не за test.
        best_model = self._grid.best(spec.scenario_id for spec in self._catalog.main()).key.model
        test_share = float(self._calculator.is_positive(holdout.target).mean())

        self._artifacts.save_table(evaluator.results_table(holdout), "test_results")
        self._artifacts.save_table(
            evaluator.threshold_effect_table(holdout), "test_threshold_effect"
        )
        self._artifacts.save_figure(
            plots.plot_test_heatmap(evaluator.metrics_matrix(holdout)), "test_metrics_heatmap"
        )
        self._artifacts.save_figure(
            plots.plot_test_pr_curves(
                pr_curves, self._model_labels(), self._scenario_labels(), test_share
            ),
            "test_pr_curves",
        )
        self._artifacts.save_figure(
            plots.plot_test_roc_curves(roc_curves, self._model_labels(), self._scenario_labels()),
            "test_roc_curves",
        )
        self._artifacts.save_figure(
            plots.plot_confusion_matrices(
                evaluator.confusion_matrices(
                    holdout, best_model, (ScenarioId.B, ScenarioId.C_SMOTE)
                ),
                ModelFactory.short_name(best_model),
            ),
            "test_confusion_matrices",
        )
        self._artifacts.save_metrics(
            {
                "test": {
                    **evaluator.to_metrics(holdout),
                    "minority_share": test_share,
                    "confusion_matrix_model": best_model.value,
                }
            }
        )
        self._test_flat = evaluator.flat_metrics(holdout)

    # ------------------------------------------------------------------ 8. Підсумок

    def _summarize(self) -> None:
        """Порівняльна таблиця (R9), профіль моделей (R7), ефект аугментації (R10), ЛР3 (P4)."""
        config = self._config
        effect = AugmentationEffect(self._catalog, config.models)
        profile = ModelProfileBuilder(self._catalog, config.models)

        self._artifacts.save_table(
            ComparisonTableBuilder(self._catalog, config.models).build(self._grid, self._holdout),
            "comparison",
        )
        self._artifacts.save_table(profile.build(self._grid), "model_profile")
        self._artifacts.save_table(
            effect.table(self._grid, self._test_intervals), "augmentation_effect"
        )
        self._artifacts.save_table(
            effect.threshold_table(self._grid), "augmentation_threshold_effect"
        )
        self._artifacts.save_metrics(
            {
                "profile": profile.to_metrics(self._grid),
                "augmentation": {
                    "effect": effect.to_metrics(self._grid, self._test_intervals, self._test_flat),
                    "overfitting": merge_overfitting_metrics(
                        self._sweep_metrics, self._curve_metrics, config.models
                    ),
                },
                "lab3_handoff": lab3_handoff(self._grid, config.cv_folds, config.models),
            }
        )

    # ------------------------------------------------------------------ допоміжне

    def _model_labels(self, models: Sequence[ModelKind] | None = None) -> dict[str, str]:
        """Ключ моделі → коротка назва (у порядку конфігурації)."""
        return {
            model.value: ModelFactory.short_name(model) for model in models or self._config.models
        }

    def _scenario_labels(self, scenarios: Sequence[ScenarioId] | None = None) -> dict[str, str]:
        """Ключ сценарію → підпис; за замовчуванням — сценарії завдання."""
        chosen = scenarios or [spec.scenario_id for spec in self._catalog.main()]
        return {scenario.value: self._catalog.label(scenario) for scenario in chosen}
