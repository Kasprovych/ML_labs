"""Сценарій edge-дослідження (R5): TFLite-варіанти, роздільність входу, межі придатності.

Пороги для перекалібрування (квантування зсуває шкалу помилки) обираються на val;
test — лише звітні числа. Latency вимірюється на цьому комп'ютері (Apple M1 Pro, CPU,
TFLite/XNNPACK): абсолютні мілісекунди на ARM-платах інші, переносяться співвідношення.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from mllabs.anomaly.detector import build_detector
from mllabs.anomaly.metrics import HistogramAuc
from mllabs.anomaly.postprocess import AnomalyMapPostprocessing
from mllabs.anomaly.profiling import LatencyBenchmark
from mllabs.anomaly.splits import SplitPart
from mllabs.labs.lab05.config import Lab5Config, TfliteVariant
from mllabs.labs.lab05.edge import TfliteExporter, TfliteScorer, resized, scoring_model
from mllabs.labs.lab05.inference import DetectorRunner, PartOutputs
from mllabs.labs.lab05.localization import downsample
from mllabs.labs.lab05.weights_analysis import spearman
from mllabs.labs.lab05.workbench import CategoryBench
from mllabs.modeling import F1ThresholdSelector

_logger = logging.getLogger(__name__)

#: Підписи варіантів.
VARIANT_LABELS = {
    "keras": "Keras fp32 (еталон)",
    TfliteVariant.FP32: "TFLite fp32",
    TfliteVariant.FP16: "TFLite fp16-ваги",
    TfliteVariant.DYNAMIC_INT8: "TFLite динамічне int8",
    TfliteVariant.FULL_INT8: "TFLite повне int8",
}


def thread_label(threads: int | None) -> str:
    """Підпис кількості потоків."""
    return "усі" if threads is None else str(threads)


class QualityEvaluator:
    """Якість карт/оцінок на test з порогом ЛР4 і з порогом, перекаліброваним на val.

    Args:
        histogram: Швидкі піксельні метрики.
    """

    def __init__(self, histogram: HistogramAuc) -> None:
        self._histogram = histogram

    def evaluate(
        self,
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        test_maps: np.ndarray,
        test_scores: np.ndarray,
        test_labels: np.ndarray,
        test_masks: np.ndarray,
        lab4_threshold: float,
    ) -> dict[str, float]:
        """Pixel AUROC/AP, Image AUROC, F1 за обома порогами."""
        pixel_auroc, pixel_ap = self._histogram.metrics(test_maps, test_masks)
        recalibrated = F1ThresholdSelector().select(val_labels, val_scores)
        return {
            "pixel_auroc": pixel_auroc,
            "pixel_ap": pixel_ap,
            "image_auroc": float(roc_auc_score(test_labels, test_scores)),
            "image_f1_lr4_threshold": float(f1_score(test_labels, test_scores >= lab4_threshold)),
            "image_f1_recalibrated": float(f1_score(test_labels, test_scores >= recalibrated)),
            "recalibrated_threshold": recalibrated,
        }


class EdgeStudy:
    """TFLite-варіанти та роздільність для всіх категорій.

    Args:
        config: Параметри ЛР5.
        seed: Зерно калібрувального набору.
        models_dir: Куди класти ``.tflite`` (``artifacts/models``, у ``.gitignore``).
        weights: Ваги backbone для моделей іншої роздільності.
    """

    def __init__(
        self, config: Lab5Config, seed: int, models_dir: Path, weights: str | None
    ) -> None:
        self._config = config
        self._seed = seed
        self._models_dir = models_dir
        self._weights = weights
        self._quality = QualityEvaluator(HistogramAuc(config.histogram_bins))
        self._benchmark = LatencyBenchmark(config.latency_warmup, config.latency_runs)

    def variants(self, bench: CategoryBench) -> tuple[list[dict], dict[str, np.ndarray]]:
        """Рядки «варіант × метрики» категорії й карти одного зображення для рисунка."""
        config = self._config
        assets = bench.assets
        val_rows, test_rows = bench.val_rows, bench.test_rows
        val_images = np.asarray(assets.images[val_rows])
        test_images = np.asarray(assets.images[test_rows])
        val_labels, test_labels = assets.labels(val_rows), assets.labels(test_rows)
        test_masks = assets.masks[test_rows]
        threshold = assets.config.threshold_image
        keras_val = bench.runner.run(val_images, keep_maps=False)
        keras_test = bench.runner.run(test_images)
        example = int(np.flatnonzero(test_labels == 1)[0])
        example_maps = {"keras": keras_test.maps[example]}
        rows = [
            {"category": assets.category, "variant": "keras", "size_mb": np.nan}
            | self._quality.evaluate(
                keras_val.scores,
                val_labels,
                keras_test.maps,
                keras_test.scores,
                test_labels,
                test_masks,
                threshold,
            )
        ]
        calibration = self._calibration(bench)
        exporter = TfliteExporter(calibration)
        model = scoring_model(assets.detector)
        directory = self._models_dir / assets.category
        directory.mkdir(parents=True, exist_ok=True)
        for variant in config.tflite_variants:
            blob = exporter.export(model, variant)
            (directory / f"{variant}.tflite").write_bytes(blob)
            scorer = TfliteScorer(blob, None)
            _, val_scores = scorer.run_all(val_images)
            test_maps, test_scores = scorer.run_all(test_images)
            row = {
                "category": assets.category,
                "variant": str(variant),
                "size_mb": len(blob) / 1e6,
            } | self._quality.evaluate(
                val_scores, val_labels, test_maps, test_scores, test_labels, test_masks, threshold
            )
            row |= self._deviation(test_maps, test_scores, keras_test)
            for threads in config.latency_threads:
                result = TfliteScorer(blob, threads).latency(test_images[0], self._benchmark)
                row[f"latency_ms_{thread_label(threads)}"] = result.median_ms
                row[f"latency_p90_ms_{thread_label(threads)}"] = result.p90_ms
            rows.append(row)
            example_maps[str(variant)] = test_maps[example]
            _logger.info("%s / %s: %.2f МБ", assets.category, variant, len(blob) / 1e6)
        example_maps["image"] = test_images[example]
        example_maps["mask"] = test_masks[example]
        return rows, example_maps

    def postprocess_latency(self, bench: CategoryBench) -> dict[str, float]:
        """Latency самої пост-обробки (збільшення карти патчів + гаусове згладжування), мс.

        Моделі ``.tflite`` з :meth:`variants` уже МІСТЯТЬ пост-обробку (вихід — готова
        карта аномалій), тож їхня latency — це весь конвеєр. Тут той самий шар конвертується
        окремо у TFLite fp32 і вимірюється тим самим таймером і тими самими режимами
        потоків: видно, яку частку кадру займає пост-обробка. Від категорії вимір не
        залежить (однакові форми тензорів), тому достатньо одного ``bench``.

        Returns:
            ``latency_ms_<потоки>`` (медіана) і ``latency_p90_ms_<потоки>``.
        """
        detector_config = bench.assets.config
        patch_scores = bench.test.patch_scores[0]  # (h, w) — карта патчів одного зображення
        patch_input = keras.Input(patch_scores.shape, name="image")
        layer = AnomalyMapPostprocessing(detector_config.eval_size, detector_config.blur_sigma)
        model = keras.Model(patch_input, layer(patch_input), name="edge_postprocess")
        # Калібрувальні дані потрібні лише повному int8 — для fp32 достатньо порожнього набору.
        blob = TfliteExporter(np.empty((0, *patch_scores.shape))).export(model, TfliteVariant.FP32)
        result: dict[str, float] = {}
        for threads in self._config.latency_threads:
            latency = TfliteScorer(blob, threads).latency(patch_scores, self._benchmark)
            result[f"latency_ms_{thread_label(threads)}"] = latency.median_ms
            result[f"latency_p90_ms_{thread_label(threads)}"] = latency.p90_ms
        return result

    def _calibration(self, bench: CategoryBench) -> np.ndarray:
        """Калібрувальний набір: випадкові (за seed) нормальні зображення train."""
        generator = np.random.default_rng(self._seed)
        train_rows = bench.assets.indices(SplitPart.TRAIN)
        chosen = generator.choice(train_rows, self._config.calibration_images, replace=False)
        return np.asarray(bench.assets.images[np.sort(chosen)])

    def _deviation(
        self, maps: np.ndarray, scores: np.ndarray, reference: PartOutputs
    ) -> dict[str, float]:
        grid = self._config.correlation_grid
        small = np.stack([downsample(image_map, grid) for image_map in maps])
        small_reference = np.stack([downsample(image_map, grid) for image_map in reference.maps])
        return {
            "map_max_abs_diff": float(np.max(np.abs(maps - reference.maps))),
            "score_max_abs_diff": float(np.max(np.abs(scores - reference.scores))),
            "map_spearman": spearman(small, small_reference),
        }

    def resolution(self, benches: Sequence[CategoryBench]) -> list[dict]:
        """Якість (Keras) і latency TFLite fp32 для меншої сторони входу backbone."""
        config = self._config
        rows = []
        for size in config.edge_input_sizes:
            latency: dict[str, float] = {}
            for number, bench in enumerate(benches):
                assets = bench.assets
                detector_config = replace(assets.config, image_size=size)
                detector = build_detector(detector_config, assets.head, self._weights)
                runner = DetectorRunner(detector, config.inference_batch)
                val_images = resized(assets.images[bench.val_rows], size)
                test_images = resized(assets.images[bench.test_rows], size)
                val = runner.run(val_images, keep_maps=False)
                test = runner.run(test_images)
                quality = self._quality.evaluate(
                    val.scores,
                    assets.labels(bench.val_rows),
                    test.maps,
                    test.scores,
                    assets.labels(bench.test_rows),
                    assets.masks[bench.test_rows],
                    assets.config.threshold_image,
                )
                if number == 0:  # швидкість не залежить від значень ваг — міряємо один раз
                    calibration = resized(self._calibration(bench), size)
                    blob = TfliteExporter(calibration).export(
                        scoring_model(detector), TfliteVariant.FP32
                    )
                    for threads in config.latency_threads:
                        result = TfliteScorer(blob, threads).latency(
                            test_images[0], self._benchmark
                        )
                        latency[f"latency_ms_{thread_label(threads)}"] = result.median_ms
                rows.append({"category": assets.category, "input_size": size} | quality | latency)
        return rows


def lr4_summary(tables_dir: Path, head: str = "FAE") -> pd.DataFrame:
    """Зведення ЛР4 для edge: GFLOPs, latency (1 потік і всі, пакет 1) і якість backbone.

    Числа копіюються з таблиць ЛР4 під час запуску ЛР5 (жодних «чисел з голови»).
    """
    latency = pd.read_csv(tables_dir / "latency.csv")
    flops = pd.read_csv(tables_dir / "flops.csv")
    study = pd.read_csv(tables_dir / "study_backbones.csv")
    end_to_end = latency[
        (latency["Компонент"] == "наскрізно")
        & (latency["Головка"] == head)
        & (latency["Пакет"] == 1)
    ]
    one = end_to_end[end_to_end["Потоки"].astype(str) == "1"].set_index("Backbone")
    every = end_to_end[end_to_end["Потоки"].astype(str) == "усі"].set_index("Backbone")
    flops = flops[flops["Головка"] == head].set_index("Backbone")
    study = study.set_index("Варіант")
    median = "Медіана, мс/зобр."
    return pd.DataFrame(
        {
            "Backbone": flops.index,
            "GFLOPs (backbone + FAE)": flops["GFLOPs разом"].to_numpy(),
            "Latency Keras, 1 потік, мс": one.loc[flops.index, median].to_numpy(),
            "Latency Keras, усі потоки, мс": every.loc[flops.index, median].to_numpy(),
            "Pixel AUROC (val, macaroni2)": study.loc[flops.index, "Pixel AUROC"].to_numpy(),
        }
    )
