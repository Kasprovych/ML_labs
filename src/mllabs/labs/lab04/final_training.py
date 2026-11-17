"""Навчання фінальної FAE на всіх категоріях з конфігурацією, обраною на ``study_category``.

Конфігурація переноситься на інші категорії БЕЗ повторного підбору — так перевіряється,
чи узагальнюється вибір. Для ``study_category`` перевикористовується вже навчена в
дослідженні модель (те саме навчання з тим самим seed — повторювати немає сенсу).
"""

from __future__ import annotations

from dataclasses import replace

import keras
import tensorflow as tf

from mllabs.anomaly.backbones import BACKBONE_SPECS
from mllabs.anomaly.detector import DetectorConfig
from mllabs.labs.lab04.config import Lab4Config
from mllabs.labs.lab04.studies import FaeExperiment, FaeRun, FaeVariant


class FinalTrainer:
    """Навчає обрану конфігурацію на кожній категорії.

    Args:
        experiment: Запуск навчань.
        categories: Категорії.
    """

    def __init__(self, experiment: FaeExperiment, categories: tuple[str, ...]) -> None:
        self._experiment = experiment
        self._categories = categories

    def train_all(self, selected: FaeRun) -> dict[str, FaeRun]:
        """Категорія → навчання з конфігурацією ``selected``."""
        variant = replace(selected.variant, study="final", label="FAE (обрана конфігурація)")
        runs = {}
        for category in self._categories:
            if category == selected.category:
                runs[category] = replace(selected, variant=variant)
            else:
                runs[category] = self._experiment.run(variant, category)
        return runs


def detector_config(
    config: Lab4Config,
    variant: FaeVariant,
    category: str,
    thresholds: tuple[float, float],
    seed: int,
) -> DetectorConfig:
    """Конфіг детектора для ``detector.json`` (пороги — з розміченого val).

    Args:
        config: Параметри роботи.
        variant: Обрана конфігурація FAE.
        category: Категорія.
        thresholds: Пороги (зображення, піксель).
        seed: Зерно.
    """
    return DetectorConfig(
        category=category,
        head="fae",
        backbone=str(variant.backbone),
        layers=BACKBONE_SPECS[variant.backbone].layers,
        feature_dim=config.feature_dim,
        patch_size=variant.patch_size,
        tile_grid=variant.tile_grid,
        image_size=config.image_size,
        eval_size=config.eval_size,
        blur_sigma=config.blur_sigma,
        threshold_image=thresholds[0],
        threshold_pixel=thresholds[1],
        seed=seed,
        tf_version=tf.__version__,
        keras_version=keras.__version__,
    )
