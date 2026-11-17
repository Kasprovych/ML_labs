"""Передача моделі й даних із ЛР4 у ЛР5: єдине місце, що знає, звідки що береться.

ЛР5 нічого не навчає наново: вона відновлює детектор ЛР4 (``load_bundle``), читає те саме
розбиття VisA, кеш зображень і кеш ознак backbone, а також таблицю прогнозів test
(``test_predictions.csv``), на якій ЛР4 рахувала FP/FN. Якщо ЛР4 ще не запускали, клас
кидає :class:`Lab4HandoffError` із порадою, що зробити.

Імена шарів детектора (``fae_head`` → ``normalization``, ``fae_core`` → ``latent`` …)
теж згадуються лише тут — решта модулів отримує готові об'єкти.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from mllabs.anomaly.backbones import BackboneKind
from mllabs.anomaly.detector import CONFIG_FILE, HEAD_FILE, DetectorConfig, load_bundle
from mllabs.anomaly.splits import SplitPart
from mllabs.anomaly.workspace import FeatureKey, VisaWorkspace
from mllabs.core import ProjectPaths

_logger = logging.getLogger(__name__)

#: Таблиця прогнозів test, яку пише ЛР4.
PREDICTIONS_FILE = "test_predictions.csv"
#: Імена шарів головки FAE (стабільний API ЛР4, див. docstring ``mllabs.anomaly``).
_HEAD_LAYER = "fae_head"
_NORMALIZATION_LAYER = "normalization"
_CORE_LAYER = "fae_core"
_LATENT_LAYER = "latent"
_DENSE_LAYERS = ("encoder_hidden", "latent", "decoder_hidden", "reconstruction")


class Lab4HandoffError(RuntimeError):
    """Артефакти ЛР4 відсутні або не узгоджені з даними."""


@dataclass(frozen=True)
class FaeParts:
    """Складові головки FAE для аналізу ваг.

    Attributes:
        mean: ``(D,)`` середнє каналів train (нормалізація).
        std: ``(D,)`` стандартне відхилення каналів train.
        core: Автоенкодер «нормалізовані ознаки → відновлені» (вхід ``(h, w, D)``).
        encoder: Підмодель «нормалізовані ознаки → латентний код» (``k`` вимірів).
        dense_kernels: Матриці ваг ``Dense`` у порядку проходу: ім'я → ``(входи, виходи)``.
    """

    mean: np.ndarray
    std: np.ndarray
    core: keras.Model
    encoder: keras.Model
    dense_kernels: dict[str, np.ndarray]

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """``z = (F − μ) / σ`` — те саме, що робить шар ``Normalization`` головки."""
        return (np.asarray(features, np.float32) - self.mean) / self.std


#: Нижня межа дисперсії каналу — як ``epsilon`` шару ``keras.layers.Normalization``.
NORMALIZATION_EPSILON = 1e-7


def fae_parts(head: keras.Model) -> FaeParts:
    """Розбирає головку FAE на нормалізацію, ядро, енкодер і матриці ваг."""
    normalization = head.get_layer(_NORMALIZATION_LAYER)
    core = head.get_layer(_CORE_LAYER)
    mean = np.asarray(normalization.mean, np.float32).reshape(-1)
    variance = np.asarray(normalization.variance, np.float32).reshape(-1)
    encoder = keras.Model(core.input, core.get_layer(_LATENT_LAYER).output, name="fae_encoder")
    kernels = {name: np.asarray(core.get_layer(name).kernel) for name in _DENSE_LAYERS}
    std = np.sqrt(np.maximum(variance, NORMALIZATION_EPSILON))
    return FaeParts(mean, std, core, encoder, kernels)


@dataclass(frozen=True)
class CategoryAssets:
    """Усе про одну категорію: детектор ЛР4, зображення, маски, ознаки, прогнози.

    Індекси (``indices``) — номери рядків у порядку кешів ЛР4 (``category_rows``).

    Attributes:
        category: Категорія VisA.
        config: Конфіг детектора ЛР4 (пороги, backbone, шари).
        detector: Наскрізна модель із виходами backbone (для Grad-CAM).
        head: Головка FAE «ознаки → помилка відновлення патчів».
        rows: Рядки ``splits.csv`` категорії.
        images: ``(N, S, S, 3)`` ``uint8`` (memmap).
        masks: ``(N, E, E)`` ``bool``.
        features: ``(N, h, w, D)`` ``float16`` ознаки backbone (memmap кешу ЛР4).
        predictions: Рядки FAE з ``test_predictions.csv`` у порядку рядків test.
        all_predictions: Усі методи ЛР4 (FAE, PaDiM, PatchCore) для цієї категорії.
        model_dir: Каталог збереженого детектора.
    """

    category: str
    config: DetectorConfig
    detector: keras.Model
    head: keras.Model
    rows: pd.DataFrame
    images: np.ndarray
    masks: np.ndarray
    features: np.ndarray
    predictions: pd.DataFrame
    all_predictions: pd.DataFrame
    model_dir: Path

    def indices(self, part: SplitPart, label: int | None = None) -> np.ndarray:
        """Номери рядків частини ``part`` (за потреби — лише з міткою ``label``)."""
        selected = self.rows["part"] == str(part)
        if label is not None:
            selected &= self.rows["label"] == label
        return np.flatnonzero(selected.to_numpy())

    def labels(self, indices: np.ndarray) -> np.ndarray:
        """Мітки зображень (0 — норма, 1 — аномалія)."""
        return self.rows["label"].to_numpy()[indices]

    @property
    def grid(self) -> int:
        """Сторона сітки патчів ``h`` (32 для входу 256)."""
        return int(self.features.shape[1])


class Lab4Handoff:
    """Завантажує артефакти ЛР4 для ЛР5.

    Args:
        project: Шляхи проєкту.
        workspace: Робочий простір VisA (той самий кеш, що в ЛР4).
        categories: Категорії з детекторами.
        method: Назва FAE в ``test_predictions.csv``.
        lab4_number: Номер лабораторної-джерела.
        weights: Ваги backbone (``"imagenet"``; у тестах — ``None``).
    """

    def __init__(
        self,
        project: ProjectPaths,
        workspace: VisaWorkspace,
        categories: tuple[str, ...],
        method: str,
        lab4_number: int = 4,
        weights: str | None = "imagenet",
    ) -> None:
        self._lab4 = project.lab(lab4_number)
        self._workspace = workspace
        self._categories = categories
        self._method = method
        self._weights = weights
        self._predictions: pd.DataFrame | None = None

    @property
    def lab4_tables_dir(self) -> Path:
        """Каталог таблиць ЛР4 (latency, FLOPs, дослідження backbone)."""
        return self._lab4.tables_dir

    @property
    def lab4_metrics_file(self) -> Path:
        """``metrics.json`` ЛР4."""
        return self._lab4.metrics_file

    def model_dir(self, category: str) -> Path:
        """Каталог детектора категорії."""
        return self._lab4.models_dir / category

    def prepare(self) -> None:
        """Перевіряє наявність артефактів ЛР4 і читає збережене розбиття VisA."""
        self.check_available()
        try:
            self._workspace.load_splits()
        except FileNotFoundError as error:
            raise Lab4HandoffError(str(error)) from error

    def check_available(self) -> None:
        """Перевіряє, що всі файли ЛР4 на місці, інакше — зрозуміла помилка."""
        missing = [
            path
            for category in self._categories
            for path in (
                self.model_dir(category) / CONFIG_FILE,
                self.model_dir(category) / HEAD_FILE,
            )
            if not path.is_file()
        ]
        if not (self._lab4.tables_dir / PREDICTIONS_FILE).is_file():
            missing.append(self._lab4.tables_dir / PREDICTIONS_FILE)
        if missing:
            listed = ", ".join(str(path) for path in missing)
            raise Lab4HandoffError(
                f"Бракує артефактів ЛР4 ({listed}). Спершу виконайте `uv run mllabs run 4`."
            )

    def predictions(self) -> pd.DataFrame:
        """Уся таблиця ``test_predictions.csv`` ЛР4 (усі методи й категорії)."""
        if self._predictions is None:
            frame = pd.read_csv(self._lab4.tables_dir / PREDICTIONS_FILE)
            self._predictions = frame.fillna({"defect_types": ""})
        return self._predictions

    def load(self, category: str) -> CategoryAssets:
        """Детектор, дані й прогнози однієї категорії (після :meth:`prepare`)."""
        config, detector = load_bundle(
            self.model_dir(category), weights=self._weights, expose_backbone=True
        )
        if config.tile_grid != 1:
            raise Lab4HandoffError(
                f"ЛР5 підтримує лише детектор без тайлів (tile_grid = 1), а не {config.tile_grid}"
            )
        workspace = self._workspace
        rows = workspace.category_rows(category)
        batch = workspace.images(category, config.image_size)
        masks = workspace.images(category, config.eval_size).masks
        features = workspace.raw_features(category, FeatureKey(BackboneKind(config.backbone), 1))
        if config.patch_size != 1:
            raise Lab4HandoffError("ЛР5 очікує детектор без агрегації патчів (patch_size = 1)")
        everything = self.predictions()
        mine = everything[everything["category"] == category]
        predictions = self._ordered_predictions(mine, rows)
        return CategoryAssets(
            category=category,
            config=config,
            detector=detector,
            head=detector.get_layer(_HEAD_LAYER),
            rows=rows,
            images=batch.images,
            masks=np.asarray(masks),
            features=features,
            predictions=predictions,
            all_predictions=mine.reset_index(drop=True),
            model_dir=self.model_dir(category),
        )

    def _ordered_predictions(self, frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Рядки FAE у тому самому порядку, що й зображення test у кешах."""
        fae = frame[frame["method"] == self._method].set_index("image")
        test_images = rows.loc[rows["part"] == str(SplitPart.TEST), "image"]
        missing = set(test_images) - set(fae.index)
        if missing:
            raise Lab4HandoffError(
                f"У {PREDICTIONS_FILE} немає {len(missing)} зображень test — "
                "таблиця не відповідає splits.csv; перезапустіть `uv run mllabs run 4`."
            )
        return fae.loc[test_images.to_list()].reset_index()


def check_scores(
    category: str, recomputed: np.ndarray, predictions: pd.DataFrame, tolerance: float
) -> float:
    """Звіряє перераховані оцінки test з оцінками ЛР4.

    Returns:
        Максимальна абсолютна розбіжність.

    Raises:
        Lab4HandoffError: Якщо розбіжність перевищує ``tolerance``.
    """
    difference = float(np.max(np.abs(recomputed - predictions["score"].to_numpy())))
    if difference > tolerance:
        raise Lab4HandoffError(
            f"{category}: оцінки test розходяться з ЛР4 на {difference:.2e} > {tolerance:.0e}. "
            "Модель і таблиці ЛР4 не узгоджені — перезапустіть `uv run mllabs run 4`."
        )
    _logger.info("%s: оцінки test збігаються з ЛР4 (max |Δ| = %.1e)", category, difference)
    return difference
