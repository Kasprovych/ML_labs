"""Лабораторна робота №5: пояснюваність (XAI) і межі надійності детектора аномалій ЛР4.

Цей модуль лише ОРКЕСТРУЄ кроки. Обчислення — у модулях пакета (``weights_analysis``,
``attribution``, ``perturbation``, ``error_cases``, ``drift``, ``edge`` і їхні сценарії
``*_study``), параметри — у ``config.py``. Модель ЛР4 не перенавчається: вона відновлюється
з ``labs/lab04/artifacts/models`` через спільний пакет ``mllabs.anomaly``.

Правило ізоляції: усе, що обирається (база IG, шар Grad-CAM, пороги TFLite і «рамки»),
обирається на розміченому val; FP/FN і звітні числа — лише test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from mllabs.anomaly.metrics import HistogramAuc
from mllabs.anomaly.runtime import configure_tf_threads, device_info
from mllabs.anomaly.visa import VISA_ARCHIVE, VISA_SHA256, VISA_URL, VisaDatasetSource
from mllabs.anomaly.workspace import VisaWorkspace, WorkspaceSettings
from mllabs.core import Lab, LabContext, Step
from mllabs.core.timing import StepTimer
from mllabs.labs.lab05 import figure_data, plots, summary
from mllabs.labs.lab05.attribution import IntegratedGradients
from mllabs.labs.lab05.checks import (
    LocalizationStudy,
    PerturbationRunner,
    SanityStudy,
    mean_curves,
    perturbation_metrics,
    perturbation_summary,
)
from mllabs.labs.lab05.config import Lab5Config
from mllabs.labs.lab05.drift import FAMILY_LABELS, DriftEvaluator, DriftStudy
from mllabs.labs.lab05.edge_study import VARIANT_LABELS, EdgeStudy, lr4_summary
from mllabs.labs.lab05.error_cases import DESCRIPTOR_LABELS
from mllabs.labs.lab05.errors_study import ErrorAnalysis, ErrorStudy, cause_label_map, error_metrics
from mllabs.labs.lab05.handoff import Lab4Handoff
from mllabs.labs.lab05.weights_analysis import spectra_table
from mllabs.labs.lab05.weights_study import CategoryWeights, WeightsStudy
from mllabs.labs.lab05.workbench import BenchBuilder, CategoryBench
from mllabs.labs.lab05.xai_study import (
    BASELINE_LABELS,
    METHOD_LABELS,
    BaselineSelection,
    Selection,
    TestExplanations,
    XaiStudy,
    select_images,
)

#: Ключові дескриптори для рисунка «FP vs TN, FN vs TP».
_FIGURE_DESCRIPTORS = ("global_error", "edge_strength", "border_distance", "brightness_std")


class Lab5(Lab):
    """Пояснюваність і межі надійності детектора аномалій ЛР4 (VisA, FAE).

    Args:
        context: Контекст виконання (шляхи, артефакти, логер).
        config: Параметри експерименту; за замовчуванням — стандартні.
        handoff: Доступ до артефактів ЛР4 (у тестах — мікро-ЛР4 у тимчасовому каталозі).
        weights: Ваги backbone (у тестах — ``None``, без мережі).
    """

    NUMBER = 5
    TITLE = "Пояснюваність (XAI) та межі надійності моделі"

    # Стан, який кроки передають один одному (оголошення типів, як поля класу в Java).
    _benches: dict[str, CategoryBench]
    _igs: dict[str, IntegratedGradients]
    _baseline: BaselineSelection
    _cam_layers: dict[str, str]
    _selections: dict[str, Selection]
    _explanations: dict[str, TestExplanations]
    _errors: ErrorAnalysis
    _drift: pd.DataFrame
    _drift_limits: pd.DataFrame
    _edge_variants: pd.DataFrame
    _edge_resolution: pd.DataFrame

    def __init__(
        self,
        context: LabContext,
        config: Lab5Config | None = None,
        handoff: Lab4Handoff | None = None,
        weights: str | None = "imagenet",
    ) -> None:
        super().__init__(context)
        config = self._config = config or Lab5Config()
        configure_tf_threads(config.intra_op_threads, config.inter_op_threads)
        self._artifacts = context.artifacts
        self._seed = context.seed
        self._weights = weights
        self._handoff = handoff or Lab4Handoff(
            context.project,
            VisaWorkspace(
                context.project,
                WorkspaceSettings(categories=config.categories),
                VisaDatasetSource(
                    context.project.data_raw_dir / "visa", VISA_URL, VISA_SHA256, VISA_ARCHIVE
                ),
                weights,
            ),
            config.categories,
            config.prediction_method,
            config.lab4_number,
            weights,
        )
        self._histogram = HistogramAuc(config.histogram_bins)
        self._timer = StepTimer()

    def steps(self) -> Sequence[Step]:
        """Кроки відповідають пунктам завдання з ASSIGNMENT.md і розділу 3.3 плану."""
        timed = self._timer.timed
        return [
            Step("1. Модель і дані ЛР4: перевірка узгодженості", timed("handoff", self._load)),
            Step("2. Інтерпретованість ваг FAE", timed("weights", self._analyze_weights)),
            Step(
                "3. Integrated Gradients: вибір бази на val",
                timed("ig_baseline", self._select_baseline),
            ),
            Step(
                "4. Пояснення на test: IG, Grad-CAM, HiResCAM", timed("explanations", self._explain)
            ),
            Step("5. Адекватність пояснень", timed("sanity", self._check_explanations)),
            Step("6. Граничні випадки FP/FN", timed("errors", self._analyze_errors)),
            Step("7. Дрейф розподілу", timed("drift", self._study_drift)),
            Step("8. Edge: TFLite, квантування, роздільність", timed("edge", self._study_edge)),
            Step("9. Межі придатності та зведення", self._summarize),
        ]

    # ------------------------------------------------------------------ 1

    def _load(self) -> None:
        """P3: детектори ЛР4, прогони val/test/train, звірка з ``test_predictions.csv``."""
        self._handoff.prepare()
        builder = BenchBuilder(self._handoff, self._config)
        self._benches = {category: builder.build(category) for category in self._config.categories}
        self._artifacts.save_table(summary.handoff_table(self._benches.values()), "handoff_check")
        self._artifacts.save_metrics({"handoff": summary.handoff_metrics(self._benches.values())})

    # ------------------------------------------------------------------ 2

    def _analyze_weights(self) -> None:
        """R1: розклад помилки, спектри ваг, латент, PCA, якобіан."""
        config = self._config
        study = WeightsStudy(config, self._seed)
        results: dict[str, CategoryWeights] = {
            category: study.run(bench.assets, bench.scorer)
            for category, bench in self._benches.items()
        }
        save = self._artifacts.save_table
        layer_rows = [row for result in results.values() for row in result.layer_rows]
        save(pd.DataFrame(layer_rows), "error_by_layer")
        save(
            pd.DataFrame([r for res in results.values() for r in res.map_rows]),
            "error_by_layer_maps",
        )
        topk = pd.DataFrame([row for result in results.values() for row in result.topk_rows])
        save(topk, "top_channels")
        spectra = {category: result.spectra for category, result in results.items()}
        save(spectra_table(spectra), "weight_spectra")
        save(summary.latent_table(figure_data.latent_rows(results)), "latent_analysis")
        save(
            summary.jacobian_table(figure_data.jacobian_rows(results, config.singular_threshold)),
            "jacobian_spectrum",
        )
        figures = self._artifacts.save_figure
        figures(plots.plot_error_by_layer(pd.DataFrame(layer_rows)), "error_by_layer")
        figures(plots.plot_layer_error_maps(figure_data.layer_rows(results)), "layer_error_maps")
        figures(plots.plot_top_channels(topk), "top_channels")
        figures(plots.plot_weight_spectra(spectra), "weight_spectra")
        figures(
            plots.plot_latent_pca({category: r.scatter for category, r in results.items()}),
            "latent_pca",
        )
        figures(
            plots.plot_jacobian_spectrum(
                {category: r.jacobians for category, r in results.items()},
                config.singular_threshold,
            ),
            "jacobian_spectrum",
        )
        maps = pd.DataFrame([row for result in results.values() for row in result.map_rows])
        figures(plots.plot_linear_vs_fae(maps), "linear_vs_fae")
        self._artifacts.save_metrics(
            {"weights": figure_data.weights_metrics(results, config.singular_threshold)}
        )

    # ------------------------------------------------------------------ 3

    def _select_baseline(self) -> None:
        """R2 (D3): усі бази IG на аномаліях val, вибір за AUROC за умови повноти."""
        config = self._config
        self._igs = {
            category: bench.integrated_gradients(config.ig_steps, config.ig_max_steps, config)
            for category, bench in self._benches.items()
        }
        self._baseline = XaiStudy(config, self._seed).select_baseline(
            list(self._benches.values()), self._igs
        )
        self._artifacts.save_table(self._baseline.runs, "ig_baselines_val_runs")
        self._artifacts.save_table(
            summary.baseline_table(self._baseline.summary), "ig_baselines_val"
        )
        labels = {str(kind): label for kind, label in BASELINE_LABELS.items()}
        self._artifacts.save_figure(
            plots.plot_ig_baselines(self._baseline.example, self._baseline.summary, labels),
            "ig_baselines",
        )
        pooled = self._baseline.summary[self._baseline.summary["category"] != "усі"]
        self._artifacts.save_metrics(
            {
                "xai": {
                    "ig_baseline": str(self._baseline.chosen),
                    "ig_baseline_reason": self._baseline.reason,
                    "baselines_score": {
                        baseline: dict(zip(group["category"], group["baseline_score"], strict=True))
                        for baseline, group in pooled.groupby("baseline")
                    },
                }
            }
        )

    # ------------------------------------------------------------------ 4

    def _explain(self) -> None:
        """R2 (D4, D5): шар CAM на val; IG/CAM на test; метрики локалізації."""
        config, benches = self._config, list(self._benches.values())
        study = XaiStudy(config, self._seed)
        cam_table, self._cam_layers = study.select_cam_layers(benches)
        self._artifacts.save_table(summary.cam_layer_table(cam_table), "gradcam_layers_val")
        self._selections = {
            bench.category: select_images(bench, config, self._seed) for bench in benches
        }
        self._explanations = {
            bench.category: study.explain_test(
                bench,
                self._igs[bench.category],
                self._baseline.chosen,
                self._selections[bench.category],
                self._cam_layers,
            )
            for bench in benches
        }
        completeness = summary.completeness_frame(benches, self._explanations)
        self._artifacts.save_table(completeness, "ig_completeness")
        localization = LocalizationStudy(config, self._seed)
        per_image = pd.concat(
            [
                localization.per_image(
                    bench, self._explanations[bench.category], self._selections[bench.category]
                )
                for bench in benches
            ],
            ignore_index=True,
        )
        subset = localization.summary(per_image, subset_only=True)
        complete = localization.summary(
            per_image[per_image["ig_completeness_ok"] == True],  # noqa: E712 — колонка з NaN
            subset_only=True,
        )
        self._artifacts.save_table(
            summary.localization_table(complete), "localization_summary_complete_ig"
        )
        everything = localization.summary(per_image[per_image["method"] != "ig"], subset_only=False)
        self._artifacts.save_table(per_image, "localization_test")
        self._artifacts.save_table(summary.localization_table(subset), "localization_summary")
        self._artifacts.save_table(
            summary.localization_table(everything), "localization_all_anomalies"
        )
        self._artifacts.save_table(localization.by_area(per_image), "localization_by_area")
        cam_titles = self._cam_titles()
        for bench in benches:
            rows = figure_data.xai_rows(
                bench,
                self._explanations[bench.category],
                self._selections[bench.category],
                config.xai_panel_images,
            )
            self._artifacts.save_figure(
                plots.plot_xai_panels(rows, cam_titles), f"xai_panels_{bench.category}"
            )
        methods = ("model_map", "ig", "gradcam", "hirescam", "random")
        self._artifacts.save_figure(
            plots.plot_localization_summary(subset, methods, self._method_labels()),
            "localization_summary",
        )
        xai = summary.xai_test_metrics(
            self._cam_layers,
            completeness,
            config.completeness_tolerance,
            {
                "localization": subset,
                "localization_all_anomalies": everything,
                "localization_complete_ig": complete,
            },
        )
        self._artifacts.save_metrics({"xai": self._artifacts.load_metrics()["xai"] | xai})

    # ------------------------------------------------------------------ 5

    def _check_explanations(self) -> None:
        """R3: perturbation (два режими), рандомізація моделі, стійкість до шуму."""
        config, benches = self._config, list(self._benches.values())
        runner = PerturbationRunner(config, self._seed)
        curves = pd.concat(
            [
                runner.run(
                    bench, self._explanations[bench.category], self._selections[bench.category]
                )
                for bench in benches
            ],
            ignore_index=True,
        )
        per_image = perturbation_metrics(curves)
        result = perturbation_summary(per_image, config, self._seed)
        self._artifacts.save_table(mean_curves(curves), "perturbation_curves")
        self._artifacts.save_table(per_image, "perturbation_per_image")
        self._artifacts.save_table(result, "perturbation_summary_full")
        self._artifacts.save_table(
            summary.perturbation_table(result, "anomaly"), "perturbation_summary"
        )
        self._artifacts.save_table(
            summary.perturbation_table(result, "false_positive"), "perturbation_fp"
        )
        methods = ("model_map", "ig", "gradcam", "hirescam", "oracle")
        self._artifacts.save_figure(
            plots.plot_perturbation_curves(mean_curves(curves), methods, self._method_labels()),
            "perturbation_curves",
        )
        example_bench = benches[0]
        steps = config.perturbation_example_steps
        states, scores = figure_data.perturbation_states(
            example_bench,
            self._explanations[example_bench.category],
            self._selections[example_bench.category],
            steps,
        )
        self._artifacts.save_figure(
            plots.plot_perturbation_example(
                states, steps, scores, example_bench.assets.config.threshold_image
            ),
            "perturbation_example",
        )
        sanity = SanityStudy(config, self._seed, self._weights)
        randomization, stability, examples = [], [], []
        for bench in benches:
            rows, stable, example = sanity.run(
                bench, self._selections[bench.category], self._baseline.chosen, self._cam_layers
            )
            randomization.append(rows)
            stability.append(stable)
            examples.append(example)
        randomization_frame = pd.concat(randomization, ignore_index=True)
        stability_frame = pd.concat(stability, ignore_index=True)
        self._artifacts.save_table(
            summary.similarity_table(randomization_frame), "sanity_randomization"
        )
        self._artifacts.save_table(
            summary.similarity_table(stability_frame), "explanation_stability"
        )
        self._artifacts.save_figure(
            plots.plot_sanity(examples[0], randomization_frame, self._method_labels()),
            "sanity_randomization",
        )
        self._artifacts.save_metrics(
            {
                "sanity": {
                    "perturbation": summary.perturbation_metrics(result),
                    "randomization": summary.similarity_metrics(randomization_frame),
                    "stability": summary.similarity_metrics(stability_frame).get("noise", {}),
                }
            }
        )

    # ------------------------------------------------------------------ 6

    def _analyze_errors(self) -> None:
        """R4: дескриптори, порівняння груп, причини, крайові артефакти, панелі."""
        config, benches = self._config, list(self._benches.values())
        self._errors = errors = ErrorStudy(config).run(benches)
        save = self._artifacts.save_table
        save(summary.cases_table(errors.cases), "error_cases")
        save(summary.groups_table(errors.groups), "error_groups")
        save(summary.causes_table(errors.causes), "error_causes")
        save(summary.causes_table(errors.causes_plan_order), "error_causes_plan_order")
        save(summary.defect_type_table(errors.defect_types), "error_by_defect_type")
        save(summary.overlap_table(errors.overlap), "error_overlap_methods")
        save(summary.border_table(errors.border), "border_artifacts")
        save(summary.exclusion_table(errors.exclusion), "border_exclusion")
        cam_titles = self._cam_titles()
        figures = self._artifacts.save_figure
        for outcome, name in (("FP", "fp_panels"), ("FN", "fn_panels")):
            rows = figure_data.error_rows(
                benches, self._explanations, errors.cases, outcome, config.error_panel_images
            )
            if rows:
                figures(plots.plot_xai_panels(rows, cam_titles), name)
        descriptors = {key: DESCRIPTOR_LABELS[key] for key in _FIGURE_DESCRIPTORS}
        figures(plots.plot_error_descriptors(errors.cases, descriptors), "error_descriptors")
        figures(
            plots.plot_fp_locations(errors.border_data, config.border_px), "fp_argmax_locations"
        )
        figures(plots.plot_border_artifacts(errors.border_data), "border_artifacts")
        figures(plots.plot_border_exclusion(errors.exclusion), "border_exclusion")
        self._artifacts.save_metrics(
            {
                "errors": error_metrics(errors, config.categories, config.significance_level)
                | {"cause_rules": cause_label_map(errors.rules)}
            }
        )

    # ------------------------------------------------------------------ 7

    def _study_drift(self) -> None:
        """P1 (D8): контрольований дрейф і монітор дрейфу без міток."""
        config = self._config
        study = DriftStudy(
            config.drift_levels,
            lambda threshold: DriftEvaluator(
                self._histogram, threshold, config.pointing_tolerance_px
            ),
            config.bootstrap_repeats,
            config.confidence_level,
            self._seed,
        )
        rows, examples = [], []
        for bench in self._benches.values():
            labels = bench.assets.labels(bench.test_rows)
            category_rows, example = study.run(
                bench.category,
                bench.runner.run,
                np.asarray(bench.assets.images[bench.test_rows]),
                bench.assets.masks[bench.test_rows],
                labels,
                bench.assets.config.threshold_image,
                self._histogram,
                int(np.flatnonzero(labels == 1)[0]),
            )
            rows += category_rows
            examples.append(example)
        self._drift = pd.DataFrame(rows)
        self._drift_limits = summary.drift_limits(self._drift, config)
        self._artifacts.save_table(self._drift, "drift_full")
        self._artifacts.save_table(summary.drift_table(self._drift), "drift")
        self._artifacts.save_table(self._drift_limits, "drift_limits")
        families = list(dict.fromkeys(level.family for level in config.drift_levels))
        self._artifacts.save_figure(
            plots.plot_drift_examples(examples[0], FAMILY_LABELS), "drift_examples"
        )
        self._artifacts.save_figure(
            plots.plot_drift_curves(self._drift, families, FAMILY_LABELS), "drift_curves"
        )
        self._artifacts.save_figure(
            plots.plot_drift_monitor(self._drift, families, FAMILY_LABELS), "drift_score_shift"
        )
        self._artifacts.save_metrics(
            {"drift": summary.drift_metrics(self._drift, self._drift_limits)}
        )

    # ------------------------------------------------------------------ 8

    def _study_edge(self) -> None:
        """R5 (D9): TFLite fp32/fp16/int8, latency, роздільність входу, зведення ЛР4."""
        config = self._config
        study = EdgeStudy(config, self._seed, self._context.paths.models_dir, self._weights)
        rows, examples = [], []
        for bench in self._benches.values():
            category_rows, example = study.variants(bench)
            rows += category_rows
            examples.append(example)
        self._edge_variants = pd.DataFrame(rows)
        self._edge_resolution = pd.DataFrame(study.resolution(list(self._benches.values())))
        save = self._artifacts.save_table
        save(self._edge_variants, "edge_variants_full")
        save(summary.edge_table(self._edge_variants), "edge_variants")
        save(self._edge_resolution, "edge_resolution_full")
        save(summary.resolution_table(self._edge_resolution), "edge_resolution")
        lr4 = lr4_summary(self._handoff.lab4_tables_dir)
        save(summary.lr4_summary_table(lr4), "edge_lr4_summary")
        postprocess = study.postprocess_latency(next(iter(self._benches.values())))
        points = figure_data.tradeoff_points(self._edge_variants, self._edge_resolution, config)
        self._artifacts.save_figure(
            plots.plot_edge_tradeoff(points, config.frame_budgets_ms), "edge_tradeoff"
        )
        self._artifacts.save_figure(
            plots.plot_quantization_maps(examples[0], VARIANT_LABELS), "edge_quantization_maps"
        )
        self._artifacts.save_metrics(
            {
                "edge": summary.edge_metrics(
                    self._edge_variants, self._edge_resolution, postprocess
                )
                | {"lr4_backbones": summary.lr4_metrics(lr4), "device": device_info()}
            }
        )

    # ------------------------------------------------------------------ 9

    def _summarize(self) -> None:
        """R5, P7: межі придатності, конфігурація, час кроків."""
        config = self._config
        edge = summary.edge_limits_table(self._edge_variants, self._edge_resolution, config)
        limits = summary.limits_table(edge, self._drift_limits, self._errors.cases, config)
        self._artifacts.save_table(summary.edge_limits_display(edge), "limits_edge")
        self._artifacts.save_table(limits, "limits_of_applicability")
        self._artifacts.save_metrics(
            {
                "limits": summary.limits_metrics(limits),
                "config": summary.config_metrics(self._config, self._seed),
                "timing": self._timer.to_metrics(),
            }
        )

    # ------------------------------------------------------------------ підписи

    def _cam_titles(self) -> dict[str, str]:
        layers = {"shallow": "шар 8", "deep": "шар 16", "fused": "злиття 8+16"}
        return {
            variant: f"{METHOD_LABELS[variant]}\n({layers[layer]})"
            for variant, layer in self._cam_layers.items()
        }

    @staticmethod
    def _method_labels() -> dict[str, str]:
        return METHOD_LABELS
