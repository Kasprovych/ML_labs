"""Edge-розгортання (R5, D9): TensorFlow Lite, квантування, latency, роздільність входу.

TensorFlow Lite — формат і інтерпретатор для мобільних та вбудованих пристроїв. Модель
Keras конвертується у плоский буфер ``.tflite``; при цьому ваги можна **квантувати**:

* ``fp32`` — без змін (еталон точності);
* ``fp16`` — ваги в 16-бітних числах з рухомою комою (розмір ×½, обчислення — fp32);
* ``dynamic_int8`` — ваги в 8-бітних цілих, активації квантуються «на льоту»;
* ``full_int8`` — ваги й активації в int8; масштаби активацій оцінюються на
  калібрувальному наборі (train-норми). Найшвидше, але найбільша похибка.

Повна модель (backbone + FAE + пост-обробка) конвертується ``TFLiteConverter.from_keras_
model`` (перевірено в ЛР4; ``from_concrete_functions`` дає NaN — не використовується).
API ``tf.lite`` застарілий (Google переносить його в пакет ``ai_edge_litert``) — нових
залежностей ЛР5 не додає, попередження в журналі очікуване.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import keras
import numpy as np
import tensorflow as tf

from mllabs.anomaly.profiling import LatencyBenchmark, LatencyResult
from mllabs.labs.lab05.config import TfliteVariant

_logger = logging.getLogger(__name__)


def scoring_model(detector: keras.Model) -> keras.Model:
    """Підмодель детектора з виходами ``anomaly_map`` і ``image_score`` (без backbone-виходів)."""
    outputs = detector.output  # словник символьних тензорів виходів
    return keras.Model(
        detector.inputs,
        {"anomaly_map": outputs["anomaly_map"], "image_score": outputs["image_score"]},
        name="edge_detector",
    )


class TfliteExporter:
    """Конвертує модель Keras у TFLite обраного варіанта.

    Args:
        calibration_images: ``(N, S, S, 3)`` train-норми для калібрування повного int8.
    """

    def __init__(self, calibration_images: np.ndarray) -> None:
        self._calibration = np.asarray(calibration_images, np.float32)

    def export(self, model: keras.Model, variant: TfliteVariant) -> bytes:
        """Буфер ``.tflite`` (вхід і виходи лишаються float32)."""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if variant is not TfliteVariant.FP32:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if variant is TfliteVariant.FP16:
            converter.target_spec.supported_types = [tf.float16]
        if variant is TfliteVariant.FULL_INT8:
            converter.representative_dataset = self._representative
        return converter.convert()

    def _representative(self) -> Iterator[list[np.ndarray]]:
        """Генератор калібрувальних прикладів по одному (формат, якого чекає конвертер)."""
        for image in self._calibration:
            yield [image[None]]


class TfliteScorer:
    """Запуск моделі ``.tflite`` по одному кадру (сценарій реального часу, batch = 1).

    Args:
        model: Буфер ``.tflite``.
        threads: Потоки інтерпретатора (``None`` — усі ядра).
    """

    def __init__(self, model: bytes, threads: int | None) -> None:
        self._interpreter = tf.lite.Interpreter(
            model_content=model, num_threads=threads or os.cpu_count()
        )
        self._interpreter.allocate_tensors()
        # Сигнатурний «runner» тримає посилання на внутрішні буфери інтерпретатора, після чого
        # прямий invoke() заборонено — тому він створюється лише на вимогу в run().
        self._runner = None

    def run(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Карта ``(E, E)`` і оцінка одного зображення."""
        if self._runner is None:
            self._runner = self._interpreter.get_signature_runner()
        outputs = self._runner(image=np.asarray(image, np.float32)[None])
        return outputs["anomaly_map"][0], float(np.asarray(outputs["image_score"]).reshape(-1)[0])

    def run_all(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Карти й оцінки набору зображень (по одному)."""
        maps, scores = [], []
        for image in images:
            image_map, score = self.run(image)
            maps.append(image_map)
            scores.append(score)
        return np.stack(maps), np.asarray(scores)

    def latency(self, image: np.ndarray, benchmark: LatencyBenchmark) -> LatencyResult:
        """Затримка одного кадру (мс): лише ``invoke``, без копіювання результатів.

        Викликати на «свіжому» об'єкті (до :meth:`run`), див. коментар у конструкторі.
        """
        input_index = self._interpreter.get_input_details()[0]["index"]
        self._interpreter.set_tensor(input_index, np.asarray(image, np.float32)[None])
        return benchmark.measure(self._interpreter.invoke, batch_size=1)


def resized(images: np.ndarray, size: int, batch: int = 64) -> np.ndarray:
    """Зображення ``uint8`` іншої роздільності (білінійно з антиаліасингом), ``float32``."""
    if images.shape[1] == size:
        return np.asarray(images, np.float32)
    parts = [
        tf.image.resize(
            np.asarray(images[start : start + batch], np.float32), (size, size), antialias=True
        ).numpy()
        for start in range(0, len(images), batch)
    ]
    return np.concatenate(parts)
