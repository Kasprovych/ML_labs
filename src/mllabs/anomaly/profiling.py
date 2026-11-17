"""Профілювання: FLOPs (профайлер TensorFlow) і затримка (latency, мс/зображення).

* **FLOPs** — кількість операцій з рухомою комою на одне зображення (множення + додавання =
  2 операції, тобто FLOPs = 2 × MAC). Від пристрою не залежать. Рахуються профайлером TF на
  «замороженому» графі моделі (змінні перетворено на константи), для пакета з 1 зображення.
* **Latency** — реальний час на CPU: ``time.perf_counter`` навколо виклику скомпільованої
  ``tf.function``; спершу кілька прогрівальних запусків (компіляція графа, кеші), потім
  серія вимірювань — медіана й 90-й перцентиль. Пакет 1 — сценарій реального часу, пакет 16 —
  пропускна здатність (час / 16).

Кількість потоків TF можна задати лише до першої операції в процесі, тому кожен режим
потоків вимірюється в ОКРЕМОМУ процесі (``spawn``) — так виміри не залежать від того, що
вже робив головний процес. Ваги в такому процесі випадкові: швидкість від значень ваг не
залежить, лише від форм тензорів (розмір банку PatchCore, кількість каналів PaDiM тощо).
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np


class FlopsCounter:
    """Рахує FLOPs моделі Keras профайлером TensorFlow (на одне зображення)."""

    def count(self, model: Callable, input_shape: Sequence[int]) -> int:
        """FLOPs одного виклику ``model`` на тензорі форми ``input_shape`` (з batch = 1).

        Args:
            model: Модель Keras (або будь-який викличний об'єкт з TF-операціями).
            input_shape: Форма входу, напр. ``(1, 256, 256, 3)``.
        """
        import tensorflow as tf
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2,
        )
        from tensorflow.python.profiler import model_analyzer, option_builder

        function = tf.function(lambda inputs: model(inputs, training=False))
        concrete = function.get_concrete_function(tf.TensorSpec(tuple(input_shape), tf.float32))
        frozen = convert_variables_to_constants_v2(concrete)
        options = option_builder.ProfileOptionBuilder.float_operation()
        options["output"] = "none"  # без друку таблиці в консоль
        profile = model_analyzer.profile(graph=frozen.graph, options=options)
        return int(profile.total_float_ops)


@dataclass(frozen=True)
class LatencyResult:
    """Затримка в мілісекундах на ОДНЕ зображення.

    Attributes:
        median_ms: Медіана.
        p90_ms: 90-й перцентиль.
        runs: Кількість вимірювань.
    """

    median_ms: float
    p90_ms: float
    runs: int


class LatencyBenchmark:
    """Вимірює затримку викличного об'єкта.

    Args:
        warmup: Прогрівальних запусків (не враховуються).
        runs: Вимірювань.
    """

    def __init__(self, warmup: int, runs: int) -> None:
        self._warmup = warmup
        self._runs = runs

    def measure(self, function: Callable[[], object], batch_size: int) -> LatencyResult:
        """Вимірює ``function`` (вона має повертати результат, уже обчислений до кінця)."""
        for _ in range(self._warmup):
            function()
        durations = np.empty(self._runs)
        for run in range(self._runs):
            started = time.perf_counter()
            function()
            durations[run] = time.perf_counter() - started
        per_image = durations * 1000 / batch_size
        return LatencyResult(
            float(np.median(per_image)), float(np.percentile(per_image, 90)), self._runs
        )


@dataclass(frozen=True)
class HeadShape:
    """Форма головки для замірів швидкодії (значення ваг неважливі).

    Attributes:
        kind: ``"fae"``, ``"padim"`` або ``"patchcore"``.
        fae_hidden: Ширина прихованих шарів FAE.
        fae_latent: Латент FAE.
        padim_channels: ``d`` для PaDiM.
        patchcore_bank: Розмір банку PatchCore.
    """

    kind: str
    fae_hidden: int = 256
    fae_latent: int = 64
    padim_channels: int = 100
    patchcore_bank: int = 1


@dataclass(frozen=True)
class LatencyJob:
    """Завдання для окремого процесу: усі виміри одного режиму потоків.

    Attributes:
        threads: Потоки ``intra_op`` (``inter_op`` = 1 для 1 потоку); ``None`` — усі ядра.
        backbones: Значення ``BackboneKind``.
        heads: Форми головок.
        image_size: Сторона зображення.
        tile_grid: Тайли ``g``.
        feature_dim: ``D``.
        patch_size: ``p``.
        eval_size: Сторона карти.
        blur_sigma: σ згладжування.
        batch_sizes: Розміри пакетів.
        warmup: Прогрівальних запусків.
        runs: Вимірювань для кожного розміру пакета (той самий порядок, що ``batch_sizes``).
        seed: Зерно випадкових ваг і входів.
    """

    threads: int | None
    backbones: tuple[str, ...]
    heads: tuple[HeadShape, ...]
    image_size: int
    tile_grid: int
    feature_dim: int
    patch_size: int
    eval_size: int
    blur_sigma: float
    batch_sizes: tuple[int, ...]
    warmup: int
    runs: tuple[int, ...]
    seed: int


@dataclass(frozen=True)
class LatencyRecord:
    """Один рядок таблиці затримок.

    Attributes:
        threads: ``"1"`` або ``"all"``.
        backbone: Backbone.
        component: ``"backbone"``, ``"head"`` (головка + пост-обробка) або ``"end_to_end"``.
        head: Головка (``""`` для backbone).
        batch_size: Розмір пакета.
        median_ms: Медіана, мс/зображення.
        p90_ms: 90-й перцентиль, мс/зображення.
        runs: Вимірювань.
        load_average: Навантаження системи (1 хв) на момент виміру.
    """

    threads: str
    backbone: str
    component: str
    head: str
    batch_size: int
    median_ms: float
    p90_ms: float
    runs: int
    load_average: float


def run_latency_job(job: LatencyJob) -> list[LatencyRecord]:
    """Виконує ``job`` в окремому процесі (``spawn``) і повертає рядки результатів."""
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return pool.submit(_latency_worker, job).result()


def build_random_head(shape: HeadShape, feature_dim: int, grid: int, seed: int) -> object:
    """Шар головки з випадковими параметрами потрібної форми (лише для замірів швидкодії)."""
    from mllabs.anomaly.fae import FaeSettings, FeatureAutoencoderBuilder
    from mllabs.anomaly.padim import MahalanobisScoring
    from mllabs.anomaly.patchcore import NearestNeighborScoring

    generator = np.random.default_rng(seed)
    if shape.kind == "fae":
        builder = FeatureAutoencoderBuilder(
            FaeSettings(feature_dim=feature_dim, hidden=shape.fae_hidden, latent=shape.fae_latent)
        )
        ones = np.ones(feature_dim, np.float32)
        return builder.build_head(builder.build_core(), 0 * ones, ones)
    if shape.kind == "padim":
        channels = np.sort(generator.choice(feature_dim, shape.padim_channels, replace=False))
        layer = MahalanobisScoring(channels.tolist(), (grid, grid))
        layer.inverse_covariance.assign(
            np.tile(np.eye(shape.padim_channels, dtype=np.float32), (grid * grid, 1, 1))
        )
        return layer
    layer = NearestNeighborScoring(shape.patchcore_bank, feature_dim)
    layer.bank.assign(generator.normal(size=(shape.patchcore_bank, feature_dim)).astype(np.float32))
    return layer


def _latency_worker(job: LatencyJob) -> list[LatencyRecord]:
    """Тіло окремого процесу: налаштувати потоки ДО першої операції TF, потім виміряти."""
    from mllabs.anomaly.runtime import configure_tf_threads

    single = job.threads == 1
    configure_tf_threads(job.threads, 1 if single else None)

    import keras

    from mllabs.anomaly.backbones import BackboneKind, FeatureExtractorFactory

    keras.utils.set_random_seed(job.seed)
    factory = FeatureExtractorFactory(job.image_size, job.feature_dim, weights=None)
    grid = factory.grid_size(job.tile_grid)
    heads = {
        shape.kind: build_random_head(shape, job.feature_dim, grid, job.seed) for shape in job.heads
    }
    records: list[LatencyRecord] = []
    for backbone in job.backbones:
        extractor = factory.build(BackboneKind(backbone), job.tile_grid)
        for batch, runs in zip(job.batch_sizes, job.runs, strict=True):
            records.extend(
                _measure_pipeline(
                    job, backbone, extractor, heads, batch, LatencyBenchmark(job.warmup, runs)
                )
            )
    return records


def _measure_pipeline(
    job: LatencyJob,
    backbone: str,
    extractor: object,
    heads: dict[str, object],
    batch: int,
    bench: LatencyBenchmark,
) -> list[LatencyRecord]:
    """Виміри одного backbone при одному розмірі пакета: backbone, головки, наскрізний шлях."""
    import tensorflow as tf

    from mllabs.anomaly.features import PatchAggregator
    from mllabs.anomaly.postprocess import AnomalyMapPostprocessing

    label = "1" if job.threads == 1 else "all"
    side = job.tile_grid * job.image_size
    aggregation = PatchAggregator(job.patch_size).layer()
    postprocess = AnomalyMapPostprocessing(job.eval_size, job.blur_sigma)
    images = tf.random.uniform((batch, side, side, 3), 0, 255, seed=job.seed)
    backbone_fn = tf.function(lambda pixels: extractor(pixels, training=False))
    features = backbone_fn(images)

    def row(component: str, head: str, result: LatencyResult) -> LatencyRecord:
        return LatencyRecord(
            label,
            backbone,
            component,
            head,
            batch,
            result.median_ms,
            result.p90_ms,
            result.runs,
            os.getloadavg()[0],
        )

    records = [row("backbone", "", bench.measure(lambda: backbone_fn(images).numpy(), batch))]
    for kind, scorer in heads.items():
        head_fn = _head_function(scorer, aggregation, postprocess)
        full_fn = tf.function(lambda pixels, head=head_fn: head(backbone_fn(pixels)))
        records.append(
            row("head", kind, bench.measure(lambda f=head_fn: f(features).numpy(), batch))
        )
        records.append(
            row("end_to_end", kind, bench.measure(lambda f=full_fn: f(images).numpy(), batch))
        )
    return records


def _head_function(scorer: object, aggregation: object, postprocess: object) -> Callable:
    """``tf.function`` «ознаки → агрегація → головка → пост-обробка»."""
    import tensorflow as tf

    return tf.function(lambda maps: postprocess(scorer(aggregation(maps), training=False)))


def records_to_rows(records: Sequence[LatencyRecord]) -> list[dict]:
    """Рядки для ``pandas.DataFrame``."""
    return [asdict(record) for record in records]
