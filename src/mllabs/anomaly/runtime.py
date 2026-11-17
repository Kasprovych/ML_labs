"""Налаштування виконання TensorFlow (потоки) і опис пристрою для протоколу вимірювань."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from typing import Any

import tensorflow as tf

_logger = logging.getLogger(__name__)


def configure_tf_threads(intra_op: int | None, inter_op: int | None) -> bool:
    """Задає кількість потоків TensorFlow; діє лише ДО першої операції TF у процесі.

    ``intra_op`` — потоки всередині однієї операції (множення матриць, згортки);
    ``inter_op`` — скільки незалежних операцій графа виконуються паралельно.
    ``None`` або 0 — TensorFlow обирає сам (усі ядра).

    Returns:
        ``True``, якщо налаштування застосовано; ``False``, якщо TF уже ініціалізовано
        (тоді діють попередні значення — запис у журнал).
    """
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_op or 0)
        tf.config.threading.set_inter_op_parallelism_threads(inter_op or 0)
    except RuntimeError:
        _logger.warning("Потоки TF уже ініціалізовано — налаштування intra/inter-op не змінено")
        return False
    return True


def cpu_name() -> str:
    """Назва процесора (на macOS — через ``sysctl``)."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip() or platform.processor()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def device_info() -> dict[str, Any]:
    """Опис пристрою: CPU, ядра, версії, наявність GPU для TF, поточне навантаження."""
    import keras

    return {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "tf_version": tf.__version__,
        "keras_version": keras.__version__,
        "gpu_available": bool(tf.config.list_physical_devices("GPU")),
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "load_average": list(os.getloadavg()),
    }
