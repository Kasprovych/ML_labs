"""SOTA-бенчмарк: реімплементації PaDiM і PatchCore на тих самих ознаках, що й FAE.

Порівняння «за інших рівних»: той самий backbone, агрегація патчів, розбиття, пост-обробка
і метрики — відрізняється лише головка. Тому різниця якості пояснюється саме головкою, а не
конвеєром (на відміну від порівняння з числами з інших статей — вони в окремій довідковій
таблиці). Обидва методи навчаються лише на нормальних зображеннях train.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mllabs.anomaly.backbones import BACKBONE_SPECS
from mllabs.anomaly.detector import PatchScorer
from mllabs.anomaly.fae import FaeScorer
from mllabs.anomaly.padim import Padim
from mllabs.anomaly.patchcore import GreedyCoresetSampler, PatchCore
from mllabs.anomaly.splits import SplitPart
from mllabs.anomaly.workspace import FeatureKey, VisaWorkspace
from mllabs.labs.lab04.config import Lab4Config, PublishedResult
from mllabs.labs.lab04.evaluation import MethodMaps
from mllabs.labs.lab04.studies import FaeExperiment, FaeRun

_logger = logging.getLogger(__name__)

#: Назви методів-реімплементацій.
PADIM = "PaDiM"
PATCHCORE = "PatchCore"


@dataclass(frozen=True)
class SotaFit:
    """Навчена реімплементація та її карти.

    Attributes:
        maps: Карти val і test.
        scorer: Головка.
        fit_seconds: Час навчання (fit).
        memory_bytes: Пам'ять головки.
        head_flops: FLOPs головки на зображення (аналітично).
    """

    maps: MethodMaps
    scorer: PatchScorer
    fit_seconds: float
    memory_bytes: int
    head_flops: int


class SotaBenchmark:
    """Навчає PaDiM і PatchCore та рахує їхні карти.

    Args:
        workspace: Дані.
        experiment: Спільна пост-обробка (та сама, що для FAE).
        config: Параметри.
        seed: Зерно (канали PaDiM, проєкція і старт coreset PatchCore).
    """

    def __init__(
        self, workspace: VisaWorkspace, experiment: FaeExperiment, config: Lab4Config, seed: int
    ) -> None:
        self._workspace = workspace
        self._experiment = experiment
        self._config = config
        self._seed = seed

    def build(self, method: str) -> Padim | PatchCore:
        """Новий ненавчений метод."""
        config = self._config
        if method == PADIM:
            return Padim(config.padim_channels, config.padim_epsilon, self._seed)
        sampler = GreedyCoresetSampler(
            config.patchcore_coreset_ratio, config.patchcore_projection_dim, self._seed
        )
        return PatchCore(sampler, config.patchcore_max_train_vectors, self._seed)

    def fit(self, method: str, category: str, key: FeatureKey, patch_size: int) -> SotaFit:
        """Навчання на train і карти val / test."""
        workspace = self._workspace
        train = workspace.features(category, key, patch_size, SplitPart.TRAIN)
        model = self.build(method)
        started = time.perf_counter()
        model.fit(train)
        seconds = time.perf_counter() - started
        positions = train.shape[1] * train.shape[2]
        del train
        gc.collect()
        _logger.info("%s / %s: fit %.1f с", method, category, seconds)
        maps = MethodMaps(
            method,
            category,
            self._experiment.maps(
                model, workspace.features(category, key, patch_size, SplitPart.VAL)
            ),
            self._experiment.maps(
                model, workspace.features(category, key, patch_size, SplitPart.TEST)
            ),
        )
        return SotaFit(maps, model, seconds, model.memory_bytes(), model.head_flops(positions))

    def run_all(
        self, final_runs: Mapping[str, FaeRun], model_name: str
    ) -> tuple[dict[tuple[str, str], MethodMaps], dict[tuple[str, str], SotaFit]]:
        """Карти val/test власної моделі та обох реімплементацій для кожної категорії.

        Args:
            final_runs: Категорія → фінальна FAE (її конфігурація задає ознаки для всіх).
            model_name: Назва власної моделі в ключах словника.
        """
        maps: dict[tuple[str, str], MethodMaps] = {}
        fits: dict[tuple[str, str], SotaFit] = {}
        for category, run in final_runs.items():
            key, p = run.variant.feature_key, run.variant.patch_size
            scorer = FaeScorer(run.training.head)
            maps[(model_name, category)] = MethodMaps(
                model_name,
                category,
                self._experiment.maps(
                    scorer, self._workspace.features(category, key, p, SplitPart.VAL)
                ),
                self._experiment.maps(
                    scorer, self._workspace.features(category, key, p, SplitPart.TEST)
                ),
            )
            for method in (PADIM, PATCHCORE):
                fits[(method, category)] = self.fit(method, category, key, p)
                maps[(method, category)] = fits[(method, category)].maps
        return maps, fits

    def cost_quality(self, selected: FaeRun, model_name: str) -> dict[tuple[str, str], float]:
        """Pixel AUROC на val для всіх пар «головка × backbone» з обраними p і g (P2).

        FAE для backbone, відмінного від обраного, навчається заново з обраними
        гіперпараметрами (той самий протокол, лише норма train), для обраного — береться
        вже навчена модель.
        """
        variant, category = selected.variant, selected.category
        quality: dict[tuple[str, str], float] = {}
        for backbone in self._config.backbones:
            if backbone == variant.backbone:
                run = selected
            else:
                label = f"{BACKBONE_SPECS[backbone].label} (обрані p, g, FAE)"
                cost_variant = replace(variant, study="cost", label=label, backbone=backbone)
                run = self._experiment.run(cost_variant, category)
            key = FeatureKey(backbone, variant.tile_grid)
            quality[(model_name, str(backbone))] = run.val_pixel_auroc
            for method in (PADIM, PATCHCORE):
                quality[(method, str(backbone))] = self.val_pixel_auroc(
                    method, category, key, variant.patch_size
                )
        return quality

    def val_pixel_auroc(
        self, method: str, category: str, key: FeatureKey, patch_size: int
    ) -> float:
        """Pixel AUROC на val (для графіка «точність — вартість» на всіх backbone)."""
        workspace = self._workspace
        model = self.build(method)
        model.fit(workspace.features(category, key, patch_size, SplitPart.TRAIN))
        gc.collect()
        maps = self._experiment.maps(
            model, workspace.features(category, key, patch_size, SplitPart.VAL)
        )
        return float(
            roc_auc_score(workspace.masks(category, SplitPart.VAL).reshape(-1), maps.reshape(-1))
        )


def published_table(results: tuple[PublishedResult, ...]) -> pd.DataFrame:
    """Довідкові опубліковані числа (≤ 7 колонок)."""
    labels = {"image_auroc": "Image AUROC", "pixel_auroc": "Pixel AUROC", "pixel_ap": "Pixel AP"}
    return pd.DataFrame(
        {
            "Метод": [r.method for r in results],
            "Категорія": [r.category for r in results],
            "Метрика": [labels[r.metric] for r in results],
            "Значення": [r.value for r in results],
            "Джерело": [r.source for r in results],
            "Протокол": [r.protocol for r in results],
        }
    )


def published_lookup(results: tuple[PublishedResult, ...]) -> dict[tuple[str, str, str], float]:
    """Словник «(метод, категорія, метрика) → значення»."""
    return {(r.method, r.category, r.metric): r.value for r in results}


def nan_if_missing(lookup: dict[tuple[str, str, str], float], key: tuple[str, str, str]) -> float:
    """Опубліковане значення або NaN, якщо його немає в джерелах."""
    return lookup.get(key, np.nan)
