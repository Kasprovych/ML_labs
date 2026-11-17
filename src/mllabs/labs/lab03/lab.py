"""Лабораторна робота №3: Оптимізація гіперпараметрів і витік даних.

Цей модуль лише ОРКЕСТРУЄ кроки: що і в якому порядку виконується.
Обчислення живуть в окремих модулях пакета (пошук — ``search.py``, вкладена
CV — ``nested.py``, витік — ``leakage.py``, test — ``holdout.py``), у спільних
пакетах ``mllabs.preprocessing`` (препроцесинг ЛР1) і ``mllabs.modeling``
(модель і протокол оцінки ЛР2), а параметри — у ``config.py``.

Головне правило: кроки 1–4 працюють лише з train. Test з'являється в кроці 3
(перевірка «реальністю» двох пайплайнів витоку) і в кроці 5 (фіксований
перелік моделей); жодне рішення за ним не ухвалюється.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.base import BaseEstimator

from mllabs.core import Lab, LabContext, Step
from mllabs.data import DatasetSchema, DatasetSplit, ProcessedDatasetRepository
from mllabs.labs.lab03 import plots, summary
from mllabs.labs.lab03.baseline import (
    BaselineEvaluator,
    BaselineResult,
    Lab2Reference,
    read_lab2_reference,
)
from mllabs.labs.lab03.config import Lab3Config
from mllabs.labs.lab03.holdout import BASELINE_KEY, FinalModelEvaluator
from mllabs.labs.lab03.leakage import (
    LeakageExperiment,
    LeakageHoldoutCheck,
)
from mllabs.labs.lab03.nested import NestedCrossValidation
from mllabs.labs.lab03.search import (
    HyperparameterSearch,
    OptunaSearchStrategy,
    SearchOutcome,
    SearchStrategyFactory,
)
from mllabs.labs.lab03.timing import StepTimer
from mllabs.modeling import CrossValidator, MetricsCalculator, ModelFactory, PairedBootstrap
from mllabs.preprocessing import (
    MODEL_STEP,
    PreprocessingScenario,
    PreprocessorFactory,
    SamplerFactory,
    SamplerKind,
    build_model_pipeline,
)


class Lab3(Lab):
    """Оптимізація гіперпараметрів, витік даних і вкладена крос-валідація.

    Args:
        context: Контекст виконання (шляхи, артефакти, логер).
        config: Параметри експерименту; за замовчуванням — стандартні.
    """

    NUMBER = 3
    TITLE = "Оптимізація гіперпараметрів і витік даних"

    # Стан, який кроки передають один одному (лише оголошення типів, як поля в Java).
    _schema: DatasetSchema
    _dev: DatasetSplit
    _full: DatasetSplit
    _preprocessors: PreprocessorFactory
    _models: ModelFactory
    _pipeline: BaseEstimator
    _calculator: MetricsCalculator
    _inner_cv: CrossValidator
    _outer_cv: CrossValidator
    _baseline: BaselineResult
    _lab2: Lab2Reference | None
    _strategies: list[HyperparameterSearch]
    _outcomes: list[SearchOutcome]
    _reports: list[summary.MethodReport]
    _test: summary.HoldoutSummary

    def __init__(self, context: LabContext, config: Lab3Config | None = None) -> None:
        super().__init__(context)
        self._config = config or Lab3Config()
        self._artifacts = context.artifacts
        self._seed = context.seed
        self._timer = StepTimer()

    def steps(self) -> Sequence[Step]:
        """Кроки відповідають етапам завдання з ASSIGNMENT.md."""
        timed = self._timer.timed
        return [
            Step("1. Дані, метрика та baseline", timed("data_baseline", self._prepare)),
            Step("2. Дослідження 1: Grid, Random, Bayesian", timed("search", self._search)),
            Step("3. Дослідження 2: витік даних", timed("leakage", self._study_leakage)),
            Step("4. Дослідження 3: вкладена CV 5×3", timed("nested", self._nested_cv)),
            Step("5. Фінальна оцінка на test", timed("test", self._evaluate_on_test)),
            Step("6. Зведені таблиці та графіки", self._summarize),
        ]

    # ------------------------------------------------------------------ 1. Дані й baseline

    def _prepare(self) -> None:
        """Dev-підвибірка, повний train, звірка препроцесингу, baseline (R1–R5)."""
        config = self._config
        repository = ProcessedDatasetRepository(
            self._context.project.data_processed_dir, config.dataset_name
        )
        self._schema = repository.load_schema()
        self._dev = repository.load(
            train_sample_size=config.dev_sample_size, random_state=self._seed
        )
        self._full = repository.load()
        self._preprocessors = PreprocessorFactory(self._schema, self._seed, config.preprocessing)
        self._preprocessors.check_manifest(repository.load_preprocessing())
        self._models = ModelFactory(config.model_settings, self._seed)
        self._pipeline = build_model_pipeline(
            self._preprocessors.build(PreprocessingScenario.FULL), self._models.create(config.model)
        )
        positive = self._schema.positive_label
        self._calculator = MetricsCalculator(positive)
        self._inner_cv = CrossValidator(
            config.inner_folds,
            self._seed,
            positive,
            config.optuna_fold_n_jobs,
            config.optuna_threads_per_worker,
        )
        self._outer_cv = CrossValidator(
            config.outer_folds,
            self._seed,
            positive,
            config.cv_n_jobs,
            config.cv_threads_per_worker,
        )
        self._baseline = BaselineEvaluator(
            self._inner_cv, self._outer_cv, self._calculator
        ).evaluate(self._pipeline, self._dev.x_train, self._dev.y_train)
        self._lab2 = read_lab2_reference(
            self._context.project.lab(2).metrics_file, config.model.value
        )
        if self._lab2 is None:
            self._log.warning("Метрики ЛР2 не знайдено — довідку lab2_reference пропущено")

        # Параметри ЛР2 з describe() + фактичні значення всіх параметрів простору пошуку.
        model_params = self._pipeline.named_steps[MODEL_STEP].get_params()
        params = self._models.describe(config.model) | {
            name: model_params[name] for name in config.search_space.names
        }
        self._artifacts.save_table(
            summary.data_state_frame(
                {
                    "dev (пошук, Nested CV)": self._dev.y_train,
                    "повний train (витік)": self._full.y_train,
                    "test": self._full.y_test,
                },
                positive,
            ),
            "data_state",
        )
        self._artifacts.save_table(config.search_space.describe(), "search_space")
        self._artifacts.save_table(
            summary.baseline_frame(
                params, self._baseline, self._lab2.train_cv_gap if self._lab2 else None
            ),
            "baseline",
        )
        self._artifacts.save_metrics(
            summary.preparation_metrics(
                config,
                self._seed,
                self._dev.y_train,
                self._full,
                positive,
                params,
                self._baseline,
                self._lab2.train_cv_gap if self._lab2 else None,
            )
        )

    # ------------------------------------------------------------------ 2. Пошук

    def _search(self) -> None:
        """Самостійний пошук кожним методом на dev, ті самі 3 внутрішні фолди (R6–R11)."""
        x_dev, y_dev = self._dev.x_train, self._dev.y_train
        splits = self._inner_cv.splits(x_dev, y_dev)
        self._strategies = SearchStrategyFactory(self._config, MODEL_STEP, self._seed).create_all()
        self._outcomes = []
        for strategy in self._strategies:
            outcome = strategy.run(self._pipeline, x_dev, y_dev, splits)
            self._log.info(
                "    %s: Best Score %.4f за %.1f с",
                outcome.label,
                outcome.best_score,
                outcome.wall_seconds,
            )
            self._outcomes.append(outcome)
        self._artifacts.save_table(summary.search_history_frame(self._outcomes), "search_history")

    # ------------------------------------------------------------------ 3. Витік

    def _study_leakage(self) -> None:
        """П'ять пайплайнів на повному train, розріз за обсягом, перевірка на test (R12–R14)."""
        config = self._config
        experiment = self._leakage_experiment()
        full_x, full_y = self._full.x_train, self._full.y_train
        results = experiment.run(full_x, full_y)

        repository = ProcessedDatasetRepository(
            self._context.project.data_processed_dir, config.dataset_name
        )
        samples = {
            size: repository.load(train_sample_size=size, random_state=self._seed)
            for size in config.leakage_sample_sizes
        }
        targets = {size: sample.y_train for size, sample in samples.items()}
        targets[len(full_y)] = full_y
        # «|» об'єднує словники: до менших обсягів додається вже порахований повний train.
        by_size = experiment.run_by_size(
            {size: (sample.x_train, sample.y_train) for size, sample in samples.items()}
        ) | {len(full_y): results}
        positives = {
            size: int(self._calculator.is_positive(target).sum())
            for size, target in targets.items()
        }

        holdout = LeakageHoldoutCheck(
            self._preprocessors.build(PreprocessingScenario.FULL),
            self._models.create(config.model),
            self._calculator,
            PairedBootstrap(config.bootstrap_repeats, config.confidence_level, self._seed),
            config.main_process_threads,
        ).evaluate((full_x, full_y), (self._full.x_test, self._full.y_test))

        self._artifacts.save_table(summary.leakage_comparison_frame(results), "leakage_comparison")
        self._artifacts.save_table(summary.leakage_folds_frame(results), "leakage_folds")
        self._artifacts.save_table(
            summary.leakage_holdout_frame(results, holdout), "leakage_holdout"
        )
        self._artifacts.save_table(
            summary.leakage_by_size_frame(by_size, positives), "leakage_by_size"
        )
        self._artifacts.save_figure(
            plots.plot_leakage_comparison(results, holdout), "leakage_comparison"
        )
        self._artifacts.save_figure(plots.plot_leakage_by_size(by_size), "leakage_by_size")
        self._artifacts.save_metrics(
            {"leakage": summary.leakage_metrics(results, holdout, by_size)}
        )

    def _leakage_experiment(self) -> LeakageExperiment:
        """Збирає дослідження витоку з препроцесора Б, моделі ЛР2 і SMOTE ЛР1–2."""
        config = self._config
        samplers = SamplerFactory(
            config.sampling_strategy, config.smote_neighbors, config.noise_scale, self._seed
        )
        return LeakageExperiment(
            preprocessor=self._preprocessors.build(PreprocessingScenario.FULL),
            model=self._models.create(config.model),
            sampler=samplers.create(SamplerKind.SMOTE),
            cv=CrossValidator(
                config.leakage_folds,
                self._seed,
                self._schema.positive_label,
                config.cv_n_jobs,
                config.cv_threads_per_worker,
            ),
            calculator=self._calculator,
            random_state=self._seed,
            n_jobs=config.cv_n_jobs,
            threads_per_worker=config.cv_threads_per_worker,
        )

    # ------------------------------------------------------------------ 4. Nested CV

    def _nested_cv(self) -> None:
        """Вкладена CV 5×3 для кожного методу на тих самих зовнішніх фолдах (R15–R17)."""
        nested = NestedCrossValidation(
            self._outer_cv,
            self._inner_cv,
            self._calculator,
            MODEL_STEP,
            self._config.main_process_threads,
        )
        self._reports = []
        for strategy, outcome in zip(self._strategies, self._outcomes, strict=True):
            result = nested.run(strategy, self._pipeline, self._dev.x_train, self._dev.y_train)
            self._log.info(
                "    %s: Nested %.4f ± %.4f за %.1f с",
                result.label,
                *result.mean_std,
                result.wall_seconds,
            )
            self._reports.append(summary.MethodReport(outcome, result))
        self._artifacts.save_table(
            summary.nested_folds_frame(self._baseline, self._reports), "nested_folds"
        )

    # ------------------------------------------------------------------ 5. Test

    def _evaluate_on_test(self) -> None:
        """Baseline і найкращі параметри кожного методу: refit на dev → test (D8, P6)."""
        config = self._config
        evaluator = FinalModelEvaluator(
            self._calculator,
            PairedBootstrap(config.bootstrap_repeats, config.confidence_level, self._seed),
            MODEL_STEP,
            config.main_process_threads,
        )
        candidates = {BASELINE_KEY: {}} | {
            outcome.key: outcome.best_params for outcome in self._outcomes
        }
        holdout = evaluator.evaluate(
            self._pipeline,
            candidates,
            (self._dev.x_train, self._dev.y_train),
            (self._full.x_test, self._full.y_test),
        )
        self._test = summary.HoldoutSummary(holdout, evaluator.intervals(holdout))
        labels = {BASELINE_KEY: summary.BASELINE_LABEL} | {
            outcome.key: outcome.label for outcome in self._outcomes
        }
        self._artifacts.save_table(
            summary.holdout_results_frame(self._test, labels), "test_results"
        )
        metrics = self._artifacts.load_metrics()
        baseline_metrics = metrics["baseline"] | summary.baseline_test_metrics(self._test)
        payload: dict = {"baseline": baseline_metrics}
        if self._lab2 is not None:
            payload["lab2_reference"] = {
                "hgb_full_cv_pr_auc": self._lab2.cv_pr_auc,
                "hgb_full_test_pr_auc": self._lab2.test_pr_auc,
            }
        self._artifacts.save_metrics(payload)

    # ------------------------------------------------------------------ 6. Підсумок

    def _summarize(self) -> None:
        """Таблиця завдання (R18), деталі (R10, R17, R21), графіки (R20, P5), час (P4)."""
        baseline, reports = self._baseline, self._reports
        save_table = self._artifacts.save_table
        save_table(summary.methods_table(baseline, reports), "optimization_methods")
        save_table(summary.cost_table(reports), "optimization_cost")
        save_table(summary.quality_table(baseline, reports), "optimization_quality")
        save_table(summary.pairwise_table(baseline, reports), "nested_pairwise")
        save_table(
            summary.best_params_frame(
                # Фактичні параметри моделі baseline (усі, зокрема max_features).
                self._pipeline.named_steps[MODEL_STEP].get_params(),
                reports,
                self._config.search_space.names,
            ),
            "best_params",
        )

        save_figure = self._artifacts.save_figure
        save_figure(plots.plot_quality_vs_time(baseline, reports), "quality_vs_time")
        save_figure(
            plots.plot_search_convergence(self._outcomes, self._optuna_startup()),
            "search_convergence",
        )
        save_figure(plots.plot_nested_inner_vs_outer(baseline, reports), "nested_inner_vs_outer")
        save_figure(
            plots.plot_search_space_samples(
                self._space_sample_outcomes(), self._config.search_space
            ),
            "search_space_samples",
        )
        self._artifacts.save_metrics(
            summary.methods_metrics(baseline, reports, self._test)
            | {"timing": self._timer.to_metrics()}
        )

    # ------------------------------------------------------------------ допоміжне

    def _optuna_startup(self) -> int:
        """Кількість стартових випадкових спроб TPE (для риски на графіку збіжності)."""
        optuna = next(s for s in self._strategies if isinstance(s, OptunaSearchStrategy))
        return optuna.startup_trials

    def _space_sample_outcomes(self) -> list[SearchOutcome]:
        """Grid, найбільший Random і Optuna — три різні «манери» перебору простору."""
        grid, *randoms, optuna = self._outcomes
        return [grid, randoms[-1], optuna]
