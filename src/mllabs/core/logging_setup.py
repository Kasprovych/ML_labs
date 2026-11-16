"""Налаштування журналювання (logging) для запуску з командного рядка."""

from __future__ import annotations

import logging
import os

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_TIME_FORMAT = "%H:%M:%S"


def configure_logging(*, verbose: bool = False) -> None:
    """Вмикає вивід журналу в консоль.

    Args:
        verbose: Якщо ``True`` — показувати й налагоджувальні повідомлення.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_TIME_FORMAT,
        force=True,
    )
    # TensorFlow та matplotlib дуже «балакучі» — залишаємо від них лише попередження.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for noisy in ("matplotlib", "PIL", "tensorflow", "absl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
