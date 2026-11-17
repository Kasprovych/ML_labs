"""Лабораторна робота №4: детекція аномалій на зображеннях без учителя (VisA).

Цей модуль лише ОРКЕСТРУЄ кроки. Обчислення — у спільному пакеті ``mllabs.anomaly``
(дані VisA, backbone, FAE, PaDiM, PatchCore, метрики, профілювання — його використає ЛР5)
та в модулях цього пакета (дослідження, оцінка, бенчмарк, графіки), параметри — у
``config.py``.

Правила ізоляції: усі моделі навчаються ЛИШЕ на нормальних зображеннях train; конфігурації
й пороги обираються на розміченому val; test використовується один раз — у кроці 8.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd

from mllabs.anomaly.backbones import BACKBONE_SPECS
from mllabs.anomaly.detector import save_bundle
from mllabs.anomaly.metrics import HistogramAuc, ImageBootstrap
from mllabs.anomaly.patchcore import PatchCore
from mllabs.anomaly.runtime import configure_tf_threads, device_info
from mllabs.anomaly.splits import SplitPart
from mllabs.anomaly.visa import VisaDatasetSource
from mllabs.anomaly.workspace import FeatureKey, VisaWorkspace
from mllabs.core import Lab, LabContext, Step
from mllabs.core.timing import StepTimer
from mllabs.labs.lab04 import data_overview, plots, summary
from mllabs.labs.lab04.benchmark import (
    PADIM,
    PATCHCORE,
    SotaBenchmark,
    SotaFit,
    published_lookup,
    published_table,
)
from mllabs.labs.lab04.config import Lab4Config
from mllabs.labs.lab04.evaluation import (
    BenchmarkResult,
    MethodMaps,
    TestEvaluator,
    evaluate_benchmark,
    ground_truth,
    map_examples,
    metrics_row,
    prediction_table,
    summary_metrics,
)
from mllabs.labs.lab04.final_training import FinalTrainer, detector_config
from mllabs.labs.lab04.profiling_study import (
    ProfilingStudy,
    accuracy_vs_cost,
    backbone_costs,
    end_to_end_latency,
    flops_table,
    latency_frame,
    latency_lookup,
    latency_table,
    profiling_metrics,
)
from mllabs.labs.lab04.studies import (
    FaeExperiment,
    FaeRun,
    StudyRunner,
    select_best,
    unique_runs,
)

#: Назва власної моделі в таблицях.
FAE = "FAE"
#: Еталонні методи (реімплементації).
REFERENCES = (PADIM, PATCHCORE)


class Lab4(Lab):
    """Детекція аномалій: Feature Autoencoder проти PaDiM і PatchCore на VisA.

    Args:
        context: Контекст виконання (шляхи, артефакти, логер).
        config: Параметри експерименту; за замовчуванням — стандартні.
        source: Джерело VisA (у тестах підміняється мікро-архівом).
        weights: Ваги backbone (у тестах — ``None``, без мережі).
    """

    NUMBER = 4
    TITLE = "Побудова та навчання нейронної мережі"

    # Стан, який кроки передають один одному (лише оголошення типів, як поля в Java).
    _splits: pd.DataFrame
    _backbones: pd.DataFrame
    _backbone_runs: tuple[FaeRun, ...]
    _patch_runs: tuple[FaeRun, ...]
    _regularization_runs: tuple[FaeRun, ...]
    _selected: FaeRun
    _final: dict[str, FaeRun]
    _maps: dict[tuple[str, str], MethodMaps]
    _sota: dict[tuple[str, str], SotaFit]
    _cost_quality: dict[tuple[str, str], float]
    _benchmark_result: BenchmarkResult

    def __init__(
        self,
        context: LabContext,
        config: Lab4Config | None = None,
        source: VisaDatasetSource | None = None,
        weights: str | None = "imagenet",
    ) -> None:
        super().__init__(context)
        config = self._config = config or Lab4Config()
        # Потоки TF задаються до першої операції TF (Lab.run далі фіксує seed).
        configure_tf_threads(config.intra_op_threads, config.inter_op_threads)
        source = source or VisaDatasetSource(
            context.project.data_raw_dir / "visa",
            config.visa_url,
            config.visa_sha256,
            config.visa_archive,
        )
        self._artifacts = context.artifacts
        self._seed = context.seed
        self._workspace = VisaWorkspace(
            context.project, config.workspace_settings(), source, weights
        )
        self._experiment = FaeExperiment(self._workspace, config, context.seed)
        self._benchmark = SotaBenchmark(self._workspace, self._experiment, config, context.seed)
        self._timer = StepTimer()

    def steps(self) -> Sequence[Step]:
        """Кроки відповідають пунктам завдання з ASSIGNMENT.md."""
        timed = self._timer.timed
        return [
            Step("1. Дані VisA: розбиття, стан даних", timed("data", self._prepare_data)),
            Step("2. Патчі та ознаки backbone (кеш)", timed("features", self._extract_features)),
            Step("3. Дослідження екстракторів", timed("study_backbones", self._study_backbones)),
            Step("4. Дослідження розміру патчів", timed("study_patch_size", self._study_patches)),
            Step("5. Дослідження регуляризації", timed("study_regularization", self._study_reg)),
            Step("6. Навчання фінальної моделі", timed("final_training", self._train_final)),
            Step("7. SOTA: PaDiM і PatchCore (реімплементації)", timed("sota", self._fit_sota)),
            Step("8. Оцінка на test", timed("evaluation", self._evaluate)),
            Step("9. Профілювання FLOPs і latency", timed("profiling", self._profile)),
            Step("10. Зведення", self._summarize),
        ]

    # ------------------------------------------------------------------ 1. Дані

    def _prepare_data(self) -> None:
        """Розпакування, розбиття, стан даних, дисперсія тла, площі дефектів (R1, R2, R13)."""
        config, workspace = self._config, self._workspace
        self._splits = workspace.prepare(self._seed)
        samples = {
            c: workspace.background_images(c, config.background_sample)
            for c in config.background_categories
        }
        background = data_overview.background_variance(
            samples, config.categories, config.brightness_cells
        )
        areas = data_overview.collect_defect_areas(workspace, config.categories)
        pixel_share = data_overview.collect_pixel_shares(workspace, config.categories)
        grid = config.image_size // 8
        transformations = data_overview.transformations(
            config.image_size,
            grid,
            config.feature_dim,
            config.patch_sizes,
            config.study_tile_grid,
            config.blur_sigma,
        )
        self._artifacts.save_table(data_overview.data_overview(self._splits), "data_overview")
        self._artifacts.save_table(data_overview.split_counts(self._splits), "splits")
        self._artifacts.save_table(background, "background_variance", float_format="%.2f")
        self._artifacts.save_table(
            data_overview.defect_area_summary(areas, pixel_share),
            "defect_area",
            float_format="%.3f",
        )
        self._artifacts.save_table(transformations, "transformations")
        examples = data_overview.sample_examples(workspace, config.categories, self._seed)
        self._artifacts.save_figure(plots.plot_samples_grid(examples), "samples_grid")
        self._artifacts.save_figure(
            plots.plot_background_variance(background), "background_variance"
        )
        self._artifacts.save_figure(plots.plot_defect_area_hist(areas), "defect_area_hist")
        study_image = np.asarray(workspace.images(config.study_category).images[0])
        self._artifacts.save_figure(
            plots.plot_patch_grid(
                study_image, grid, config.default_patch_size, config.study_tile_grid
            ),
            "patch_grid",
        )
        source = {
            "url": config.visa_url,
            "sha256": config.visa_sha256,
            "license": config.visa_license,
        }
        splits_csv = workspace.splits_path.relative_to(self._context.project.root)
        per_category = data_overview.data_metrics(
            self._splits, background, areas, pixel_share, config.categories
        )
        self._artifacts.save_metrics(
            {"data": {"source": source, "splits_csv": str(splits_csv), **per_category}}
        )

    # ------------------------------------------------------------------ 2. Ознаки

    def _extract_features(self) -> None:
        """Ознаки study_category для трьох backbone; таблиця backbone (R2, R3)."""
        config = self._config
        for backbone in config.backbones:
            self._workspace.raw_features(config.study_category, FeatureKey(backbone, 1))
        self._backbones = backbone_costs(config)
        self._artifacts.save_table(
            summary.backbones_table(self._backbones), "backbones", float_format="%.3f"
        )
        self._artifacts.save_metrics(summary.feature_metrics(self._backbones, config))

    # ------------------------------------------------------------------ 3–5. Дослідження

    def _study_backbones(self) -> None:
        """R7: FAE на трьох екстракторах (таблиця; графік із latency — у кроці 9)."""
        self._backbone_runs = self._runner().run_backbones()
        self._artifacts.save_table(summary.study_table(self._backbone_runs), "study_backbones")
        self._update_studies("backbones", summary.study_metrics(self._backbone_runs, "backbone"))

    def _study_patches(self) -> None:
        """R6: агрегація p і тайли."""
        base = select_best(self._backbone_runs, self._config.selection_tolerance)
        self._patch_runs = self._runner().run_patch_sizes(base)
        best = select_best(self._patch_runs, self._config.selection_tolerance)
        table = summary.study_table(self._patch_runs)
        self._artifacts.save_table(table, "study_patch_size")
        self._artifacts.save_figure(
            plots.plot_study_two_metrics(table, best.variant.label, "Pixel AP", "Pixel AP (val)"),
            "study_patch_size",
        )
        self._update_studies("patch_size", summary.study_metrics(self._patch_runs))

    def _study_reg(self) -> None:
        """R5: латент, Dropout, L2, шум входу; вибір фінальної конфігурації."""
        tolerance = self._config.selection_tolerance
        self._regularization_runs = self._runner().run_regularization(
            select_best(self._patch_runs, tolerance)
        )
        self._selected = select_best(self._regularization_runs, tolerance)
        table = summary.study_table(self._regularization_runs)
        self._artifacts.save_table(table, "study_regularization")
        self._artifacts.save_figure(
            plots.plot_study_two_metrics(
                table,
                self._selected.variant.label,
                "Розрив втрат",
                "Розрив MSE val_normal − train, ×10⁻³",
                second_digits=2,
                second_scale=1e3,
            ),
            "study_regularization",
        )
        runs = self._study_runs()
        selected_id = next(r.run_id for r in runs if r.training is self._selected.training)
        self._artifacts.save_figure(
            plots.plot_learning_curves_study(
                {r.run_id: r.training.history for r in runs}, selected_id
            ),
            "learning_curves_study",
        )
        self._artifacts.save_table(
            pd.DataFrame([summary.run_record(r) for r in runs]), "study_runs_full"
        )
        self._update_studies("regularization", summary.study_metrics(self._regularization_runs))
        self._update_studies("selected", summary.selected_metrics(self._selected, len(runs)))

    def _runner(self) -> StudyRunner:
        return StudyRunner(self._experiment, self._config)

    def _study_runs(self) -> list[FaeRun]:
        """Унікальні навчання трьох досліджень."""
        return unique_runs(self._backbone_runs, self._patch_runs, self._regularization_runs)

    def _update_studies(self, key: str, value: dict) -> None:
        """Доповнює розділ ``metrics.studies`` (save_metrics зливає лише верхній рівень)."""
        studies = self._artifacts.load_metrics().get("studies", {})
        self._artifacts.save_metrics({"studies": studies | {key: value}})

    # ------------------------------------------------------------------ 6. Фінальна модель

    def _train_final(self) -> None:
        """R4, R8: обрана конфігурація на всіх категоріях; історії; архітектура."""
        config = self._config
        self._final = FinalTrainer(self._experiment, config.categories).train_all(self._selected)
        others = [run for c, run in self._final.items() if c != self._selected.category]
        self._artifacts.save_table(
            summary.history_table([*self._study_runs(), *others]), "training_history"
        )
        self._artifacts.save_figure(
            plots.plot_learning_curves(
                {c: run.training.history for c, run in self._final.items()},
                {c: run.training.best_epoch for c, run in self._final.items()},
            ),
            "learning_curves",
        )
        head = self._selected.training.head
        settings = self._selected.variant.settings
        channels = sum(BACKBONE_SPECS[self._selected.variant.backbone].channels)
        self._artifacts.save_table(summary.architecture_table(head), "model_architecture")
        self._artifacts.save_figure(
            plots.plot_model_diagram(
                config.image_size,
                config.image_size // 8,
                channels,
                config.feature_dim,
                settings.latent,
                settings.hidden,
                config.blur_sigma,
            ),
            "model_diagram",
        )
        self._artifacts.save_metrics(summary.training_metrics(self._final))

    # ------------------------------------------------------------------ 7. SOTA

    def _fit_sota(self) -> None:
        """R12, P2: PaDiM і PatchCore на тих самих ознаках; якість усіх пар на val."""
        self._maps, self._sota = self._benchmark.run_all(self._final, FAE)
        self._cost_quality = self._benchmark.cost_quality(self._selected, FAE)
        self._artifacts.save_metrics(
            {
                "sota_fit": summary.sota_fit_metrics(self._sota),
                "cost_quality": {f"{h}.{b}": v for (h, b), v in self._cost_quality.items()},
            }
        )

    # ------------------------------------------------------------------ 8. Test

    def _evaluate(self) -> None:
        """R9, R12: пороги з val → test; бутстреп; прогнози й моделі для ЛР5."""
        config, workspace = self._config, self._workspace
        evaluator = TestEvaluator(
            ImageBootstrap(config.bootstrap_repeats, config.confidence_level, self._seed),
            HistogramAuc(config.histogram_bins),
        )
        truths = {c: ground_truth(workspace, c) for c in config.categories}
        result = evaluate_benchmark(evaluator, self._maps, truths, FAE, REFERENCES)
        self._benchmark_result = result
        evaluations = result.evaluations
        predictions = pd.concat(
            [prediction_table(workspace, e, config.brightness_cells) for e in evaluations.values()],
            ignore_index=True,
        )
        for category, run in self._final.items():
            metrics = evaluations[(FAE, category)].metrics
            thresholds = (metrics.threshold_image, metrics.threshold_pixel)
            detector = detector_config(config, run.variant, category, thresholds, self._seed)
            save_bundle(self._context.paths.models_dir / category, run.training.head, detector)
        full = pd.DataFrame([metrics_row(e) for e in evaluations.values()])
        published = published_lookup(config.published_results)
        examples = map_examples(workspace, predictions, result, self._maps, FAE, (FAE, *REFERENCES))
        save_table = self._artifacts.save_table
        save_table(summary.test_metrics_table(evaluations), "test_metrics", float_format="%.3f")
        save_table(full, "test_metrics_full")
        save_table(
            summary.sota_table(evaluations, published), "sota_benchmark", float_format="%.3f"
        )
        save_table(summary.bootstrap_table(result.deltas), "sota_bootstrap", float_format="%.3f")
        save_table(
            published_table(config.published_results), "published_reference", float_format="%.3f"
        )
        save_table(predictions, "test_predictions", float_format="%.5f")
        self._artifacts.save_figure(plots.plot_roc_pr(result.curves), "roc_pr_pixel")
        self._artifacts.save_figure(plots.plot_sota_benchmark(full), "sota_benchmark")
        self._artifacts.save_figure(plots.plot_anomaly_maps(examples), "anomaly_maps_test")
        self._artifacts.save_metrics(
            {
                "test": summary_metrics(evaluations),
                "sota": {
                    "delta": summary.delta_metrics(result.deltas),
                    "published": [asdict(r) for r in config.published_results],
                },
            }
        )

    # ------------------------------------------------------------------ 9. Профілювання

    def _profile(self) -> None:
        """R10, R11, P2: FLOPs, latency (окремі процеси), компроміс точність — вартість."""
        config, variant = self._config, self._selected.variant
        patchcore = self._sota[(PATCHCORE, config.study_category)].scorer
        bank = patchcore.bank_size if isinstance(patchcore, PatchCore) else 0
        study = ProfilingStudy(
            config, variant.settings, bank, variant.patch_size, variant.tile_grid, self._seed
        )
        load_before = device_info()["load_average"]
        raw = latency_frame(study.latency())
        head_costs = study.head_costs()
        backbones = backbone_costs(config, variant.tile_grid)
        gflops = dict(zip(backbones["backbone"], backbones["gflops"], strict=True))
        cost = accuracy_vs_cost(self._cost_quality, gflops, head_costs, end_to_end_latency(raw))
        study_table = summary.with_backbone_latency(
            summary.study_table(self._backbone_runs),
            self._backbone_runs,
            latency_lookup(raw, "all", "backbone", "", 1),
        )
        selected_label = select_best(self._backbone_runs, config.selection_tolerance).variant.label
        fae_profiled, fae_analytic = study.fae_flops_check()
        rows = self._workspace.category_rows(config.study_category)
        raw_images = [
            self._workspace.source.root / path
            for path in rows.loc[rows["part"] == str(SplitPart.TEST), "image"].head(
                config.decode_latency_images
            )
        ]
        self._artifacts.save_table(raw, "latency_raw")
        self._artifacts.save_table(latency_table(raw), "latency", float_format="%.2f")
        self._artifacts.save_table(flops_table(backbones, head_costs), "flops", float_format="%.3f")
        self._artifacts.save_table(cost, "accuracy_vs_cost")
        self._artifacts.save_table(study_table, "study_backbones")
        self._artifacts.save_figure(
            plots.plot_study_backbones(study_table, selected_label), "study_backbones"
        )
        self._artifacts.save_figure(plots.plot_accuracy_vs_cost(cost), "accuracy_vs_cost")
        self._artifacts.save_metrics(
            {
                "profiling": {
                    "device": device_info() | {"load_average_before_latency": load_before},
                    "patch_size": variant.patch_size,
                    "tile_grid": variant.tile_grid,
                    "patchcore_bank": bank,
                    "postprocess_gflops": study.postprocess_gflops(),
                    "decode_resize_ms": study.decode_latency(raw_images),
                    "fae_gflops_profiler": fae_profiled,
                    "fae_gflops_analytic": fae_analytic,
                    **profiling_metrics(raw, head_costs, gflops, config.latency_batch),
                }
            }
        )

    # ------------------------------------------------------------------ 10. Зведення

    def _summarize(self) -> None:
        """Параметри запуску й тривалість кроків."""
        self._artifacts.save_metrics(
            {
                "config": summary.config_metrics(self._config, self._seed),
                "timing": self._timer.to_metrics(),
            }
        )
