"""Лабораторна робота №1: EDA, попередня обробка та аугментація даних.

Цей модуль лише ОРКЕСТРУЄ кроки: що і в якому порядку виконується.
Самі обчислення живуть в окремих модулях цього пакета (по одному на
відповідальність) і в спільних пакетах ``mllabs.data`` та
``mllabs.preprocessing``, а параметри — у ``config.py``.

Головне правило роботи: після кроку 1.3 усе рахується ЛИШЕ на навчальній
вибірці. Тестова вибірка тільки записується на диск для ЛР2–3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pandas as pd

from mllabs.core import Lab, LabContext, Step
from mllabs.data import (
    KICK_SCHEMA,
    CleaningResult,
    DatasetSchema,
    DatasetSource,
    DatasetSplit,
    OpenMlDatasetSource,
    ProcessedDatasetRepository,
    RawDataCleaner,
    StratifiedSplitter,
)
from mllabs.labs.lab01 import plots
from mllabs.labs.lab01.augmentation import (
    KIND_MAJORITY,
    KIND_MINORITY,
    KIND_SYNTHETIC,
    AugmentationBenchmark,
)
from mllabs.labs.lab01.config import Lab1Config
from mllabs.labs.lab01.correlation import CorrelationAnalyzer, ExtraCollinearDropCheck
from mllabs.labs.lab01.encoding import EncodingBenchmark
from mllabs.labs.lab01.feature_engineering import GROUP_ORIGINAL, FeatureEngineeringBenchmark
from mllabs.labs.lab01.imputation import ImputationBenchmark
from mllabs.labs.lab01.normality import NormalityTester
from mllabs.labs.lab01.outliers import (
    IqrDetector,
    IsolationForestDetector,
    LofDetector,
    OutlierComparison,
    ZScoreDetector,
    project_to_plane,
)
from mllabs.labs.lab01.overview import DatasetProfiler
from mllabs.labs.lab01.scaling import (
    METHOD_RAW,
    METHOD_ROBUST,
    METHOD_YEO_JOHNSON,
    ScalingBenchmark,
)
from mllabs.labs.lab01.summary import PreprocessingSummaryBuilder
from mllabs.preprocessing import (
    PreprocessingOptions,
    PreprocessingScenario,
    PreprocessorFactory,
    ProxyModelEvaluator,
    SamplerFactory,
)


class Lab1(Lab):
    """EDA, попередня обробка та аугментація даних.

    Args:
        context: Контекст виконання (шляхи, артефакти, логер).
        config: Параметри експерименту; за замовчуванням — стандартні.
        source: Джерело сирих даних; за замовчуванням — OpenML із кешем у ``data/raw/``.
    """

    NUMBER = 1
    TITLE = "EDA, попередня обробка та аугментація даних"

    # Стан, який кроки передають один одному. Тут лише ОГОЛОШЕННЯ типів (як поля
    # класу в Java без ініціалізації); значення з'являються під час виконання кроків.
    _raw: pd.DataFrame
    _cleaning: CleaningResult
    _schema: DatasetSchema
    _split: DatasetSplit
    _factory: PreprocessorFactory
    _evaluator: ProxyModelEvaluator

    def __init__(
        self,
        context: LabContext,
        config: Lab1Config | None = None,
        source: DatasetSource | None = None,
    ) -> None:
        super().__init__(context)
        self._config = config or Lab1Config()
        # Джерело даних — залежність через конструктор: у тестах підставляється
        # фальшиве джерело з маленькою синтетичною таблицею замість мережі.
        self._source = source or OpenMlDatasetSource(
            self._config.openml_data_id, context.project.data_raw_dir / "openml"
        )
        self._artifacts = context.artifacts
        self._seed = context.seed

    def steps(self) -> Sequence[Step]:
        """Кроки відповідають пунктам завдання з ASSIGNMENT.md."""
        return [
            Step("1.1 Завантаження датасету", self._load_dataset),
            Step("1.2 Очищення та опис датасету", self._clean_and_describe),
            Step("1.3 Стратифіковане розбиття", self._split_dataset),
            Step("2.1 Кореляції та мультиколінеарність", self._analyze_correlations),
            Step("2.2 Перевірка нормальності", self._test_normality),
            Step("2.3 Виявлення викидів", self._detect_outliers),
            Step("3.1 Імп'ютація пропусків", self._compare_imputation),
            Step("3.2 Кодування категоріальних ознак", self._compare_encodings),
            Step("3.3 Масштабування та степеневі перетворення", self._compare_scaling),
            Step("4.1 Аугментація", self._compare_augmentation),
            Step("4.2 Інженерія ознак", self._engineer_features),
            Step("5. Підсумок препроцесингу та збереження даних", self._summarize_and_save),
        ]

    # ------------------------------------------------------------------ 1. Датасет

    def _load_dataset(self) -> None:
        """Завантажує сиру таблицю з OpenML (або з кешу в ``data/raw/``)."""
        self._raw = self._source.load()

    def _clean_and_describe(self) -> None:
        """Очищує дані детермінованими правилами й описує датасет (R1–R4)."""
        self._cleaning = RawDataCleaner(KICK_SCHEMA, self._config.max_missing_share).clean(
            self._raw
        )
        self._schema = self._cleaning.schema
        data = self._cleaning.data
        profiler = DatasetProfiler(self._schema, self._config.high_cardinality_threshold)
        profile = profiler.profile(data)
        class_balance = profiler.class_balance(data[self._schema.target])

        self._artifacts.save_table(
            profiler.overview_table(profile, self._raw.shape, self._cleaning.hidden_missing_cells),
            "dataset_overview",
        )
        self._artifacts.save_table(
            profiler.feature_summary(data), "feature_summary", float_format="%.2f"
        )
        self._artifacts.save_table(
            self._cleaning.missing_report, "missing_values", float_format="%.2f"
        )
        self._artifacts.save_table(self._cleaning.log, "cleaning_log")
        self._artifacts.save_figure(
            plots.plot_target_distribution(class_balance), "target_distribution"
        )
        self._artifacts.save_figure(
            plots.plot_missing_values(
                self._cleaning.missing_report, self._config.max_missing_share
            ),
            "missing_values",
        )
        self._artifacts.save_metrics(
            {
                "dataset": {
                    "name": self._schema.name,
                    "openml_id": self._config.openml_data_id,
                    "n_raw_features": self._raw.shape[1] - 1,
                    "hidden_missing_cells": self._cleaning.hidden_missing_cells,
                    # ** розпаковує словник усередину іншого словника (як putAll).
                    **profile.to_metrics(),
                }
            }
        )

    def _split_dataset(self) -> None:
        """Ділить дані на train/test; далі всі кроки працюють лише з train (P2)."""
        splitter = StratifiedSplitter(self._config.test_size, self._seed)
        self._split = splitter.split(self._cleaning.data, self._schema.target)
        self._factory = PreprocessorFactory(self._schema, self._seed, self._config.preprocessing)
        self._evaluator = ProxyModelEvaluator(
            n_splits=self._config.proxy_cv_folds,
            max_iter=self._config.proxy_max_iter,
            random_state=self._seed,
            positive_label=self._schema.positive_label,
        )
        self._artifacts.save_metrics({"split": self._split.to_metrics(self._schema.positive_label)})

    # ------------------------------------------------------------------ 2. EDA

    def _analyze_correlations(self) -> None:
        """Матриці Пірсона і Спірмена, сильні пари, VIF до/після відбору (R5, R6)."""
        config = self._config
        analyzer = CorrelationAnalyzer(config.correlation_threshold)
        numeric = self._numeric_train()
        pearson = analyzer.matrix(numeric, "pearson")
        spearman = analyzer.matrix(numeric, "spearman")
        pairs = analyzer.high_pairs(pearson, spearman)
        vif = analyzer.vif_table(
            numeric,
            dropped=self._labels(self._schema.collinear_columns),
            extra_dropped=self._labels(config.extra_collinear_candidates),
        )
        extra_check = ExtraCollinearDropCheck(
            self._schema, config.extra_collinear_candidates, config.preprocessing, self._seed
        )
        extra_scores = extra_check.compare(
            self._evaluator,
            self._factory.options_for(PreprocessingScenario.FULL),
            self._split.x_train,
            self._split.y_train,
        )

        self._artifacts.save_table(pearson, "correlation_pearson", index=True)
        self._artifacts.save_table(spearman, "correlation_spearman", index=True)
        self._artifacts.save_table(pairs, "high_correlation_pairs")
        self._artifacts.save_table(vif, "vif", float_format="%.2f")
        self._artifacts.save_figure(
            plots.plot_correlation_heatmap(pearson, "Пірсона"), "correlation_pearson"
        )
        self._artifacts.save_figure(
            plots.plot_correlation_heatmap(spearman, "Спірмена"), "correlation_spearman"
        )
        self._artifacts.save_metrics(
            {
                "correlation": {
                    **analyzer.to_metrics(pairs, vif, config.vif_threshold),
                    **extra_check.to_metrics(extra_scores),
                }
            }
        )

    def _test_normality(self) -> None:
        """Гістограми, Q-Q графіки та тести Шапіро-Уїлка і Д'Агостіно (R7, R8)."""
        numeric = self._numeric_train()
        tester = NormalityTester(
            self._config.normality_alpha, self._config.shapiro_sample_size, self._seed
        )
        table = tester.test(numeric)

        self._artifacts.save_table(table, "normality_tests", float_format="%.3f")
        self._artifacts.save_figure(
            plots.plot_numeric_distributions(numeric), "numeric_distributions"
        )
        qq_features = tester.features_for_qq(table, self._config.qq_plot_features)
        self._artifacts.save_figure(
            plots.plot_qq(tester.qq_sample(numeric)[qq_features]), "qq_plots"
        )
        self._artifacts.save_metrics({"normality": tester.to_metrics(table)})

    def _detect_outliers(self) -> None:
        """Порівнює IQR, Z-score, Isolation Forest і LOF (R9)."""
        config = self._config
        comparison = OutlierComparison(
            univariate=[IqrDetector(config.iqr_factor), ZScoreDetector(config.zscore_threshold)],
            multivariate=[
                IsolationForestDetector(
                    config.outlier_contamination, config.isolation_forest_estimators, self._seed
                ),
                LofDetector(config.outlier_contamination, config.lof_neighbors),
            ],
        )
        numeric = self._numeric_train()
        report = comparison.run(numeric, self._is_positive_train())
        projection, explained = project_to_plane(numeric, config.pca_plot_sample_size, self._seed)

        self._artifacts.save_table(report.by_feature, "outliers_by_feature", float_format="%.2f")
        self._artifacts.save_table(report.by_method, "outliers_by_method", float_format="%.2f")
        self._artifacts.save_table(report.overlap, "outlier_overlap", index=True)
        self._artifacts.save_figure(plots.plot_outlier_overlap(report.overlap), "outlier_overlap")
        self._artifacts.save_figure(
            plots.plot_outliers_pca(projection, report.masks, explained), "outliers_pca"
        )
        self._artifacts.save_metrics({"outliers": report.to_metrics()})

    # ------------------------------------------------------------------ 3. Обробка

    def _compare_imputation(self) -> None:
        """Медіана проти KNN і MICE: маскування та проксі-модель (R10)."""
        config = self._config
        benchmark = ImputationBenchmark(
            mask_share=config.imputation_mask_share,
            sample_size=config.imputation_sample_size,
            knn_neighbors=config.preprocessing.knn_neighbors,
            mice_max_iter=config.preprocessing.mice_max_iter,
            random_state=self._seed,
        )
        masking = benchmark.run_masking(self._numeric_train())
        numeric_scores, categorical_scores = benchmark.compare_on_proxy(
            self._factory,
            self._evaluator,
            self._full_options(),
            self._split.x_train,
            self._split.y_train,
        )
        wheel_type_rates = benchmark.positive_rate_by_missingness(
            self._split.x_train["WheelType"], self._is_positive_train()
        )

        self._artifacts.save_table(
            benchmark.comparison_table(masking, numeric_scores, categorical_scores),
            "imputation_comparison",
        )
        self._artifacts.save_figure(
            plots.plot_imputation_rmse(masking.rmse_by_feature), "imputation_comparison"
        )
        self._artifacts.save_figure(
            plots.plot_imputation_distributions(
                masking.restored, benchmark.features_to_show(masking)
            ),
            "imputation_distributions",
        )
        self._artifacts.save_metrics(
            {
                "imputation": benchmark.to_metrics(
                    masking, numeric_scores, categorical_scores, wheel_type_rates
                )
            }
        )

    def _compare_encodings(self) -> None:
        """План кодування за кардинальністю та порівняння чотирьох кодувань (R11, R12)."""
        benchmark = EncodingBenchmark(
            self._schema, self._config.high_cardinality_threshold, self._config.rare_category_size
        )
        # Поріг кардинальності і поділ ознак у схемі мають збігатися: фабрика бере
        # перелік високопотужних ознак зі схеми, а не з порогу.
        benchmark.check_schema(self._split.x_train)
        scores = benchmark.compare(
            self._factory,
            self._evaluator,
            self._full_options(),
            self._split.x_train,
            self._split.y_train,
        )
        comparison = benchmark.comparison_table(scores)
        plan = benchmark.plan(self._split.x_train)

        self._artifacts.save_table(
            benchmark.cardinality(self._split.x_train), "cardinality", float_format="%.2f"
        )
        self._artifacts.save_table(plan, "encoding_plan")
        self._artifacts.save_table(comparison, "encoding_comparison")
        self._artifacts.save_figure(
            plots.plot_encoding_comparison(comparison), "encoding_comparison"
        )
        self._artifacts.save_metrics({"encoding": benchmark.to_metrics(scores, plan)})

    def _compare_scaling(self) -> None:
        """RobustScaler проти степеневих перетворень: асиметрія до і після (R13)."""
        benchmark = ScalingBenchmark()
        result = benchmark.run(self._numeric_train())
        shown = benchmark.most_skewed(result, self._config.distribution_plot_features)

        self._artifacts.save_table(
            benchmark.skewness_table(result), "scaling_skewness", float_format="%.3f"
        )
        self._artifacts.save_figure(
            plots.plot_scaling_before_after(
                result.transformed,
                result.skewness,
                shown,
                methods=[METHOD_RAW, METHOD_ROBUST, METHOD_YEO_JOHNSON],
            ),
            "scaling_before_after",
        )
        self._artifacts.save_metrics({"scaling": result.to_metrics()})

    # ------------------------------------------------------------------ 4. Аугментація

    def _compare_augmentation(self) -> None:
        """SMOTE, ADASYN і гаусів шум: що вони роблять і чи допомагають (R14)."""
        config = self._config
        sampler_factory = SamplerFactory(
            sampling_strategy=config.sampling_strategy,
            k_neighbors=config.smote_neighbors,
            noise_scale=config.noise_scale,
            random_state=self._seed,
        )
        benchmark = AugmentationBenchmark(
            sampler_factory,
            config.pca_plot_sample_size,
            self._seed,
            positive_label=self._schema.positive_label,
        )
        preprocessor = self._factory.build_from_options(self._baseline_options())
        illustration = benchmark.illustrate(preprocessor, self._split.x_train, self._split.y_train)
        scores = benchmark.compare(
            self._evaluator, preprocessor, self._split.x_train, self._split.y_train
        )

        self._artifacts.save_table(
            illustration.class_balance, "augmentation_class_balance", float_format="%.2f"
        )
        self._artifacts.save_table(benchmark.comparison_table(scores), "augmentation_comparison")
        self._artifacts.save_figure(
            plots.plot_augmentation_class_balance(illustration.class_balance),
            "augmentation_class_balance",
        )
        self._artifacts.save_figure(
            plots.plot_augmentation_pca(
                illustration.projection,
                illustration.explained_variance,
                kinds=[KIND_MAJORITY, KIND_MINORITY, KIND_SYNTHETIC],
            ),
            "augmentation_pca",
        )
        self._artifacts.save_metrics({"augmentation": benchmark.to_metrics(illustration, scores)})

    def _engineer_features(self) -> None:
        """Нові ознаки: взаємна інформація з ціллю та вплив на проксі-модель (R15)."""
        config = self._config
        benchmark = FeatureEngineeringBenchmark(
            numeric_features=self._schema.numeric,
            polynomial_features=config.preprocessing.polynomial_features,
            polynomial_degree=config.preprocessing.polynomial_degree,
            sample_size=config.mutual_information_sample_size,
            random_state=self._seed,
            label=self._schema.label,  # довгі назви цін MMR → скорочення зі схеми
        )
        mutual_information = benchmark.mutual_information(self._split.x_train, self._split.y_train)
        new_features = benchmark.new_features_table(mutual_information)
        scores = benchmark.compare(
            self._factory,
            self._evaluator,
            self._baseline_options(),
            self._full_options(),
            self._split.x_train,
            self._split.y_train,
        )
        full_vif = benchmark.full_pipeline_vif(
            self._factory,
            self._factory.options_for(PreprocessingScenario.FULL),
            self._split.x_train,
            self._split.y_train,
            CorrelationAnalyzer(config.correlation_threshold),
        )

        self._artifacts.save_table(new_features, "engineered_features")
        self._artifacts.save_table(full_vif, "full_pipeline_vif", float_format="%.1f")
        self._artifacts.save_table(
            benchmark.comparison_table(scores), "feature_engineering_comparison"
        )
        self._artifacts.save_figure(
            plots.plot_feature_mutual_information(mutual_information, GROUP_ORIGINAL),
            "engineered_features_mi",
        )
        self._artifacts.save_metrics(
            {
                "feature_engineering": benchmark.to_metrics(
                    new_features,
                    scores,
                    full_vif,
                    config.vif_threshold,
                )
            }
        )

    # ------------------------------------------------------------------ 5. Підсумок

    def _summarize_and_save(self) -> None:
        """Підсумкова таблиця рішень і запис train/test для ЛР2–3 (R17, P1)."""
        full_options = self._factory.options_for(PreprocessingScenario.FULL)
        builder = PreprocessingSummaryBuilder(
            self._schema, full_options, self._factory.indicator_columns(full_options)
        )
        metrics = self._artifacts.load_metrics()
        self._artifacts.save_table(builder.build(metrics), "preprocessing_summary")
        # Δ, s і висновки рішень — окремим розділом, щоб звіт підставляв їх як {{metrics...}}.
        self._artifacts.save_metrics({"selection": builder.decisions_to_metrics(metrics)})

        repository = ProcessedDatasetRepository(
            self._context.project.data_processed_dir, self._schema.name
        )
        # Поруч із даними — фактичні налаштування препроцесингу: ЛР2–3 звіряють
        # із ними власну фабрику, щоб сценарій FULL не розійшовся з обґрунтованим тут.
        repository.save(self._split, self._schema, self._factory.manifest())
        self._log.info("Дані для ЛР2–3 записано: %s", repository.directory)

    # ------------------------------------------------------------------ допоміжне

    def _numeric_train(self) -> pd.DataFrame:
        """Числові ознаки train із короткими підписами колонок (для таблиць і графіків)."""
        numeric = self._split.x_train[list(self._schema.numeric)]
        # rename приймає функцію: кожна назва колонки проходить через schema.label.
        return numeric.rename(columns=self._schema.label)

    def _labels(self, columns: Sequence[str]) -> list[str]:
        """Короткі підписи колонок зі схеми (як у таблицях і на графіках)."""
        return [self._schema.label(column) for column in columns]

    def _is_positive_train(self) -> pd.Series:
        """Булева серія: чи належить рядок train до позитивного («поганого») класу."""
        return self._split.y_train == self._schema.positive_label

    def _full_options(self) -> PreprocessingOptions:
        """Остаточний сценарій FULL: відносно нього перевіряється кожне рішення.

        Імп'ютація, кодування, вилучення колінеарних ознак і групи нових ознак
        порівнюються як «FULL» проти «FULL зі зміненим одним елементом».
        """
        return self._factory.options_for(PreprocessingScenario.FULL)

    def _baseline_options(self) -> PreprocessingOptions:
        """Сценарій FULL БЕЗ згенерованих ознак.

        Від нього відраховується внесок кожної групи нових ознак у таблиці R15,
        і на ньому ілюструється аугментація (R14).
        """
        return replace(
            self._factory.options_for(PreprocessingScenario.FULL),
            add_row_features=False,
            add_group_aggregates=False,
            add_polynomial_features=False,
        )
