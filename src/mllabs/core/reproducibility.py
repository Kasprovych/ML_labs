"""Відтворюваність експериментів: фіксація генераторів випадкових чисел.

У машинному навчанні багато випадковості (розбиття даних, початкові ваги
нейромережі, семплінг). Якщо не зафіксувати «зерно» (seed), кожен запуск
даватиме трохи інші цифри, і звіт не збігатиметься з кодом.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

#: Зерно за замовчуванням для всіх лабораторних.
DEFAULT_SEED = 42

_logger = logging.getLogger(__name__)


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Фіксує seed для Python, NumPy та TensorFlow.

    TensorFlow імпортується ліниво (усередині функції), бо його імпорт триває
    кілька секунд, а лабораторним №1-3 він не потрібен.

    Args:
        seed: Ціле число; однаковий seed => однакові результати.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:
        _logger.debug("TensorFlow не встановлено — seed для нього пропущено")
        return
    tf.keras.utils.set_random_seed(seed)
