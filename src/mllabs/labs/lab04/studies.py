"""Дослідження гіперпараметрів FAE (R5–R7) і спільний «запуск одного навчання».

Протокол (D7 плану): лише на категорії ``study_category``, вибір за Pixel AUROC на
розміченому val, жадібний координатний пошук — кожне дослідження стартує з найкращого
варіанта попереднього:

1. екстрактори: MobileNetV2 / EfficientNetB0 / ResNet50 (база p = 3, g = 1);
2. розмір патча: p ∈ {1, 3, 5} і тайли g = 2 при найкращому p;
3. регуляризація: по одному фактору від бази (латент, Dropout, L2, шум входу).

Різниця меншу за ``selection_tolerance`` вважається нічиєю — тоді обирається варіант,
що стоїть раніше в переліку (дешевший backbone, менший p, без тайлів, база без змін).
Жадібний пошук дешевий, але може пропустити взаємодії факторів — це обмеження методу.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from mllabs.anomaly.backbones import BACKBONE_SPECS, BackboneKind
from mllabs.anomaly.detector import PatchScorer, score_patches
from mllabs.anomaly.fae import (
    EpochMetricsCallback,
    FaeScorer,
    FaeSettings,
    FaeTrainer,
    FaeTrainingResult,
    LabelledFeatures,
)
from mllabs.anomaly.metrics import HistogramAuc
from mllabs.anomaly.postprocess import AnomalyMapPostprocessor, image_scores
from mllabs.anomaly.splits import SplitPart
from mllabs.anomaly.workspace import FeatureKey, VisaWorkspace
from mllabs.labs.lab04.config import Lab4Config

_logger = logging.getLogger(__name__)

#: Скільки карт train використовувати для p99 оцінок по епохах.
_TRAIN_SAMPLE_MAPS = 16


@dataclass(frozen=True)
class FaeVariant:
    """Конфігурація одного навчання FAE.

    Attributes:
        study: Дослідження (``backbones``, ``patch_size``, ``regularization``, ``final``).
        label: Підпис варіанта для таблиць.
        backbone: Backbone.
        patch_size: ``p``.
        tile_grid: ``g``.
        settings: Гіперпараметри FAE.
    """

    study: str
    label: str
    backbone: BackboneKind
    patch_size: int
    tile_grid: int
    settings: FaeSettings

    @property
    def feature_key(self) -> FeatureKey:
        """Ключ кешу ознак."""
        return FeatureKey(self.backbone, self.tile_grid)

    def same_model(self, other: FaeVariant) -> bool:
        """Чи це те саме навчання (без огляду на підпис і дослідження)."""
        return (self.backbone, self.patch_size, self.tile_grid, self.settings) == (
            other.backbone,
            other.patch_size,
            other.tile_grid,
            other.settings,
        )


@dataclass(frozen=True)
class FaeRun:
    """Результат навчання FAE та його якість на розміченому val.

    Attributes:
        variant: Конфігурація.
        category: Категорія.
        training: Навчена головка й історія.
        val_pixel_auroc: Pixel AUROC на val (точно).
        val_pixel_ap: Pixel AP на val.
        val_image_auroc: Image AUROC на val.
    """

    variant: FaeVariant
    category: str
    training: FaeTrainingResult
    val_pixel_auroc: float
    val_pixel_ap: float
    val_image_auroc: float

    @property
    def run_id(self) -> str:
        """Унікальний ідентифікатор для ``training_history.csv``."""
        return f"{self.variant.study}:{self.category}:{self.variant.label}"

    def best_row(self) -> pd.Series:
        """Рядок історії з найкращою епохою (її ваги відновлено)."""
        history = self.training.history
        return history[history["epoch"] == self.training.best_epoch].iloc[0]


class FaeExperiment:
    """Навчає FAE на ознаках категорії й оцінює її на розміченому val.

    Args:
        workspace: Дані.
        config: Параметри.
        seed: Зерно.
    """

    def __init__(self, workspace: VisaWorkspace, config: Lab4Config, seed: int) -> None:
        self._workspace = workspace
        self._config = config
        self._seed = seed
        self._postprocessor = AnomalyMapPostprocessor(config.eval_size, config.blur_sigma)
        self._histogram = HistogramAuc(config.histogram_bins)

    def run(self, variant: FaeVariant, category: str) -> FaeRun:
        """Одне навчання (лише норма train) + точні метрики на val."""
        workspace, key, p = self._workspace, variant.feature_key, variant.patch_size
        _logger.info("FAE %s / %s", category, variant.label)
        train = workspace.features(category, key, p, SplitPart.TRAIN)
        val_normal = workspace.features(category, key, p, SplitPart.VAL_NORMAL)
        validation = LabelledFeatures(
            workspace.features(category, key, p, SplitPart.VAL),
            workspace.masks(category, SplitPart.VAL),
            workspace.labels(category, SplitPart.VAL),
        )
        # Копія (а не зріз-«вид»), щоб колбек не тримав у пам'яті весь масив train.
        train_sample = np.array(train[:_TRAIN_SAMPLE_MAPS])

        def metrics_callback(head: keras.Model) -> EpochMetricsCallback:
            return EpochMetricsCallback(
                head, validation, train_sample, self._postprocessor, self._histogram
            )

        training = FaeTrainer(variant.settings, self._seed).train(
            train, val_normal, metrics_factory=metrics_callback
        )
        del train, val_normal
        gc.collect()  # звільняє сотні МБ ознак до наступного навчання
        maps = self.maps(FaeScorer(training.head), validation.features)
        truth, pixels = validation.masks.reshape(-1), maps.reshape(-1)
        return FaeRun(
            variant,
            category,
            training,
            float(roc_auc_score(truth, pixels)),
            float(average_precision_score(truth, pixels)),
            float(roc_auc_score(validation.labels, image_scores(maps))),
        )

    def maps(self, scorer: PatchScorer, features: np.ndarray) -> np.ndarray:
        """Карти аномалій ``(N, S, S)`` для агрегованих ознак будь-якою головкою."""
        patch = score_patches(scorer, features, self._config.scoring_batch_size)
        return self._postprocessor.maps(patch)


def _comma(label: str) -> str:
    """Десяткова кома в підписі варіанта (українська норма)."""
    return label.replace(".", ",")


def select_best(runs: Sequence[FaeRun], tolerance: float) -> FaeRun:
    """Найкращий за Pixel AUROC на val; при нічиїй (< ``tolerance``) — перший у переліку."""
    best = max(run.val_pixel_auroc for run in runs)
    return next(run for run in runs if run.val_pixel_auroc >= best - tolerance)


@dataclass(frozen=True)
class StudyOutcome:
    """Результати трьох досліджень і обрана конфігурація.

    Attributes:
        backbones: Запуски дослідження екстракторів.
        patch_sizes: Запуски дослідження розміру патча.
        regularization: Запуски дослідження регуляризації.
        selected: Обраний запуск (його головка — фінальна модель ``study_category``).
    """

    backbones: tuple[FaeRun, ...]
    patch_sizes: tuple[FaeRun, ...]
    regularization: tuple[FaeRun, ...]
    selected: FaeRun

    def all_runs(self) -> list[FaeRun]:
        """Усі унікальні навчання (повторно використані не дублюються)."""
        return unique_runs(self.backbones, self.patch_sizes, self.regularization)


class StudyRunner:
    """Виконує три дослідження D7 послідовно (жадібно).

    Args:
        experiment: Запуск навчань.
        config: Параметри.
    """

    def __init__(self, experiment: FaeExperiment, config: Lab4Config) -> None:
        self._experiment = experiment
        self._config = config
        self._category = config.study_category

    def run_backbones(self) -> tuple[FaeRun, ...]:
        """R7: однакова FAE на трьох екстракторах."""
        config = self._config
        return tuple(
            self._experiment.run(
                FaeVariant(
                    "backbones",
                    BACKBONE_SPECS[backbone].label,
                    backbone,
                    config.default_patch_size,
                    1,
                    config.fae,
                ),
                self._category,
            )
            for backbone in config.backbones
        )

    def run_patch_sizes(self, base: FaeRun) -> tuple[FaeRun, ...]:
        """R6: p ∈ patch_sizes (g = 1), потім тайли g×g при найкращому p."""
        config = self._config
        runs = []
        for size in config.patch_sizes:
            variant = replace(
                base.variant, study="patch_size", label=f"p = {size}", patch_size=size
            )
            runs.append(self._reuse_or_run(variant, base))
        if config.study_tile_grid > 1:
            best = select_best(runs, config.selection_tolerance)
            grid = config.study_tile_grid
            variant = replace(
                best.variant,
                study="patch_size",
                label=f"p = {best.variant.patch_size}, тайли {grid}×{grid}",
                tile_grid=grid,
            )
            runs.append(self._experiment.run(variant, self._category))
        return tuple(runs)

    def run_regularization(self, base: FaeRun) -> tuple[FaeRun, ...]:
        """R5: один фактор за раз від бази (база — перший рядок)."""
        config = self._config
        settings = base.variant.settings
        alternatives: list[tuple[str, FaeSettings]] = [
            *((f"латент k = {k}", replace(settings, latent=k)) for k in config.latent_grid),
            *(
                (_comma(f"Dropout {r:g}"), replace(settings, dropout=r))
                for r in config.dropout_grid
            ),
            *((_comma(f"L2 = {value:g}"), replace(settings, l2=value)) for value in config.l2_grid),
            *(
                (_comma(f"шум σ = {s:g}"), replace(settings, input_noise=s))
                for s in config.noise_grid
            ),
        ]
        base_variant = replace(
            base.variant, study="regularization", label=f"база (k = {settings.latent})"
        )
        runs = [self._reuse_or_run(base_variant, base)]
        for label, alternative in alternatives:
            variant = replace(
                base.variant, study="regularization", label=label, settings=alternative
            )
            runs.append(self._experiment.run(variant, self._category))
        return tuple(runs)

    def run_all(self) -> StudyOutcome:
        """Три дослідження й вибір конфігурації."""
        tolerance = self._config.selection_tolerance
        backbones = self.run_backbones()
        patch_sizes = self.run_patch_sizes(select_best(backbones, tolerance))
        regularization = self.run_regularization(select_best(patch_sizes, tolerance))
        return StudyOutcome(
            backbones, patch_sizes, regularization, select_best(regularization, tolerance)
        )

    def _reuse_or_run(self, variant: FaeVariant, base: FaeRun) -> FaeRun:
        """Те саме навчання, що вже є (``base``), не повторюється — лише перепідписується."""
        if variant.same_model(base.variant):
            return replace(base, variant=variant)
        return self._experiment.run(variant, self._category)


def unique_runs(*groups: Sequence[FaeRun]) -> list[FaeRun]:
    """Унікальні навчання (повторно використані в кількох дослідженнях — один раз)."""
    unique: list[FaeRun] = []
    for run in (run for group in groups for run in group):  # генератор «розгортає» групи
        if all(run.training is not other.training for other in unique):
            unique.append(run)
    return unique
