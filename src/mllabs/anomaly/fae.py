"""Feature Autoencoder (FAE) — власна модель: автоенкодер відновлення ознак патчів.

Ідея (DFR, Shi et al., 2021): мережа вчиться стискати вектор ознаки НОРМАЛЬНОГО патча
через вузьке «пляшкове горло» (латент розміру ``k``) і відновлювати його назад. Нормальні
патчі вона відновлює добре, а незнайомі (дефекти) — погано. Помилка відновлення
(середній квадрат різниці по каналах) і є оцінкою аномальності патча.

Будова::

    ознаки (…, D) → Normalization (середнє/дисперсія з train) ─┐
                    → GaussianNoise σ → Dense 256 → Dropout → Dense k → Dense 256 → Dense D
                    → помилка відновлення = mean((відновлене − нормалізоване)²) по каналах

Усі шари діють на ОСТАННЮ вісь (``Dense`` = згортка 1×1), тому та сама модель навчається
на окремих векторах і застосовується до цілої карти ``(B, h, w, D)``. Регуляризація:
ширина латенту, Dropout, L2 на ваги, шум на вході (денойзинговий автоенкодер), раннє
зупинення за втратою на ``val_normal``.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from mllabs.anomaly.metrics import HistogramAuc
from mllabs.anomaly.postprocess import AnomalyMapPostprocessor, image_scores

#: Ім'я підмоделі-автоенкодера всередині головки.
CORE_NAME = "fae_core"


@dataclass(frozen=True)
class FaeSettings:
    """Гіперпараметри FAE і навчання.

    Attributes:
        feature_dim: Розмірність вхідних ознак ``D``.
        hidden: Ширина прихованих шарів.
        latent: Ширина «пляшкового горла» ``k``.
        dropout: Частка Dropout після першого шару.
        l2: Коефіцієнт L2-регуляризації ваг.
        input_noise: σ гаусового шуму на вході (у нормалізованих одиницях).
        learning_rate: Крок Adam.
        batch_size: Векторів у пакеті.
        vectors_per_epoch: Скільки випадкових векторів train бачить одна епоха.
        validation_vectors: Скільки векторів val_normal (і стільки ж train) для ``val_mse`` /
            ``train_mse`` — фіксовані підвибірки для порівнянних втрат і раннього зупинення.
        max_epochs: Максимум епох.
        patience: Епох без покращення ``val_mse`` до зупинки.
    """

    feature_dim: int = 512
    hidden: int = 256
    latent: int = 64
    dropout: float = 0.0
    l2: float = 0.0
    input_noise: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 512
    vectors_per_epoch: int = 50_000
    validation_vectors: int = 20_000
    max_epochs: int = 30
    patience: int = 5

    def dense_sizes(self) -> list[tuple[int, int]]:
        """Пари (входи, виходи) усіх ``Dense`` — для аналітичного підрахунку FLOPs і параметрів."""
        widths = [self.feature_dim, self.hidden, self.latent, self.hidden, self.feature_dim]
        return list(itertools.pairwise(widths))


@keras.saving.register_keras_serializable(package="mllabs")
class PatchReconstructionError(keras.layers.Layer):
    """Середній квадрат різниці по останній осі: ``(…, D), (…, D) → (…)``."""

    @override
    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        reconstruction, target = inputs  # розпакування списку з двох тензорів
        return tf.reduce_mean(tf.square(reconstruction - target), axis=-1)

    @override
    def compute_output_shape(
        self, input_shape: list[tuple[int | None, ...]]
    ) -> tuple[int | None, ...]:
        return tuple(input_shape[0][:-1])


class FeatureAutoencoderBuilder:
    """Будує ядро FAE (для навчання) і головку (для оцінювання).

    Args:
        settings: Гіперпараметри.
    """

    def __init__(self, settings: FaeSettings) -> None:
        self._settings = settings

    def build_core(self) -> keras.Model:
        """Автоенкодер «нормалізовані ознаки → відновлені нормалізовані ознаки»."""
        settings = self._settings
        regularizer = keras.regularizers.L2(settings.l2) if settings.l2 > 0 else None

        def dense(units: int, activation: str | None, name: str) -> keras.layers.Dense:
            return keras.layers.Dense(
                units, activation=activation, kernel_regularizer=regularizer, name=name
            )

        inputs = keras.Input((None, None, settings.feature_dim), name="normalized_features")
        hidden = keras.layers.GaussianNoise(settings.input_noise, name="input_noise")(inputs)
        hidden = dense(settings.hidden, "leaky_relu", "encoder_hidden")(hidden)
        hidden = keras.layers.Dropout(settings.dropout, name="dropout")(hidden)
        latent = dense(settings.latent, None, "latent")(hidden)
        hidden = dense(settings.hidden, "leaky_relu", "decoder_hidden")(latent)
        outputs = dense(settings.feature_dim, None, "reconstruction")(hidden)
        return keras.Model(inputs, outputs, name=CORE_NAME)

    def build_head(self, core: keras.Model, mean: np.ndarray, variance: np.ndarray) -> keras.Model:
        """Головка «сирі ознаки ``(B, h, w, D)`` → помилка відновлення ``(B, h, w)``».

        Args:
            core: Ядро з :meth:`build_core` (його шари спільні з головкою).
            mean: Середнє кожного каналу на train.
            variance: Дисперсія кожного каналу на train.
        """
        features = keras.Input((None, None, self._settings.feature_dim), name="features")
        normalized = keras.layers.Normalization(mean=mean, variance=variance, name="normalization")(
            features
        )
        reconstruction = core(normalized)
        errors = PatchReconstructionError(name="reconstruction_error")([reconstruction, normalized])
        return keras.Model(features, errors, name="fae_head")


class RandomVectorDataset(keras.utils.PyDataset):
    """Щоепохи нова випадкова підвибірка векторів train (вхід = ціль автоенкодера).

    ``PyDataset`` — аналог ``Iterator`` для Keras: ``__len__`` — кількість пакетів в епосі,
    ``__getitem__`` — пакет за номером, ``on_epoch_end`` — Keras викликає після епохи.

    Args:
        vectors: ``(N, D)`` усі вектори train.
        mean: Середнє каналів (для нормалізації).
        std: Стандартне відхилення каналів.
        settings: Розмір пакета і кількість векторів на епоху.
        seed: Зерно; епоха ``e`` використовує ``seed + e``.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        settings: FaeSettings,
        seed: int,
    ) -> None:
        super().__init__()
        self._vectors = vectors
        self._mean = mean.astype(np.float32)
        self._std = std.astype(np.float32)
        self._batch = settings.batch_size
        self._per_epoch = min(settings.vectors_per_epoch, len(vectors))
        self._seed = seed
        self._epoch = 0
        self._indices = self._draw()

    def _draw(self) -> np.ndarray:
        generator = np.random.default_rng(self._seed + self._epoch)
        # Сортування індексів пришвидшує вибірку з memmap/великого масиву (послідовний доступ).
        return np.sort(generator.choice(len(self._vectors), self._per_epoch, replace=False))

    def __len__(self) -> int:
        return -(-self._per_epoch // self._batch)  # ділення зі стелею

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        rows = self._indices[index * self._batch : (index + 1) * self._batch]
        batch = (np.asarray(self._vectors[rows], np.float32) - self._mean) / self._std
        batch = batch[:, None, None, :]  # (n, 1, 1, D): «карта» 1×1 для ядра з входом (h, w, D)
        return batch, batch

    @override
    def on_epoch_end(self) -> None:
        self._epoch += 1
        self._indices = self._draw()


#: Зсуви зерна для фіксованих підвибірок val_normal і train.
_VAL_OFFSET = 10_000
_TRAIN_OFFSET = 20_000


class ReconstructionMseCallback(keras.callbacks.Callback):
    """Після епохи рахує MSE відновлення на фіксованих підвибірках train і val_normal.

    Keras-втрата ``loss`` рахується в режимі НАВЧАННЯ (з шумом входу, Dropout і штрафом
    L2), тому її різниця з валідаційною втратою спотворена регуляризацією. Тут обидві
    величини — однакова оцінка: прогін без шуму й Dropout (``training=False``), чиста MSE
    без штрафу. Їхня різниця ``val_mse − train_mse`` — чесна міра перенавчання.

    Args:
        core: Ядро автоенкодера, що навчається.
        train_sample: Нормалізовані вектори train ``(n, 1, 1, D)``.
        validation: Нормалізовані вектори val_normal ``(m, 1, 1, D)``.
        batch_size: Пакет прогону.
    """

    def __init__(
        self,
        core: keras.Model,
        train_sample: np.ndarray,
        validation: np.ndarray,
        batch_size: int = 4096,
    ) -> None:
        super().__init__()
        self._core = core
        self._train = train_sample
        self._validation = validation
        self._batch = batch_size

    def _mse(self, vectors: np.ndarray) -> float:
        total = 0.0
        for start in range(0, len(vectors), self._batch):
            batch = vectors[start : start + self._batch]
            reconstruction = self._core(batch, training=False).numpy()
            total += float(np.square(reconstruction - batch).sum())
        return total / vectors.size

    @override
    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            return
        logs["train_mse"] = self._mse(self._train)
        logs["val_mse"] = self._mse(self._validation)


@dataclass(frozen=True)
class LabelledFeatures:
    """Розмічений val для метрик по епохах.

    Attributes:
        features: ``(N, h, w, D)`` агреговані ознаки.
        masks: ``(N, S, S)`` маски.
        labels: ``(N,)`` мітки зображень.
    """

    features: np.ndarray
    masks: np.ndarray
    labels: np.ndarray


class EpochMetricsCallback(keras.callbacks.Callback):
    """Після кожної епохи рахує метрики детекції на розміченому val і p99 оцінок train.

    На train-нормі AUROC не визначений (лише один клас), тому для train відстежується
    99-й перцентиль оцінок патчів — «наскільки погано відновлюється найгірша норма».

    Args:
        head: Головка FAE (спільні шари з ядром, що навчається).
        validation: Розмічений val.
        train_sample: ``(n, h, w, D)`` кілька карт train.
        postprocessor: Пост-обробка карт.
        histogram: Швидкі піксельні метрики.
        batch_size: Пакет для прогону головки.
    """

    def __init__(
        self,
        head: keras.Model,
        validation: LabelledFeatures,
        train_sample: np.ndarray,
        postprocessor: AnomalyMapPostprocessor,
        histogram: HistogramAuc,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._head = head
        self._validation = validation
        self._train_sample = train_sample
        self._postprocessor = postprocessor
        self._histogram = histogram
        self._batch = batch_size

    def _patch_scores(self, features: np.ndarray) -> np.ndarray:
        parts = [
            self._head(
                np.asarray(features[start : start + self._batch], np.float32), training=False
            ).numpy()
            for start in range(0, len(features), self._batch)
        ]
        return np.concatenate(parts)

    @override
    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            return
        maps = self._postprocessor.maps(self._patch_scores(self._validation.features))
        pixel_auroc, pixel_ap = self._histogram.metrics(maps, self._validation.masks)
        logs["val_pixel_auroc"] = pixel_auroc
        logs["val_pixel_ap"] = pixel_ap
        logs["val_image_auroc"] = float(roc_auc_score(self._validation.labels, image_scores(maps)))
        logs["train_patch_p99"] = float(np.percentile(self._patch_scores(self._train_sample), 99))


@dataclass(frozen=True)
class FaeTrainingResult:
    """Результат навчання FAE.

    Attributes:
        head: Навчена головка «ознаки → помилка відновлення».
        history: По рядку на епоху: ``loss``, ``train_mse``, ``val_mse``, метрики val,
            ``train_patch_p99``.
        best_epoch: Номер епохи (з 1) з найменшим ``val_mse`` — її ваги відновлено.
        seconds: Час навчання.
    """

    head: keras.Model
    history: pd.DataFrame
    best_epoch: int
    seconds: float


@dataclass(frozen=True)
class FaeScorer:
    """Навчена FAE як головка детектора (реалізує :class:`PatchScorer`).

    Attributes:
        head: Головка FAE.
    """

    head: keras.Model

    def scoring_layer(self) -> keras.layers.Layer:
        """Модель Keras — теж шар, її можна вставити в наскрізну модель."""
        return self.head

    def memory_bytes(self) -> int:
        """Ваги головки (``float32``), включно з параметрами нормалізації."""
        return int(sum(np.prod(weight.shape) for weight in self.head.weights) * 4)


class FaeTrainer:
    """Навчає FAE на векторах нормальних патчів train.

    Args:
        settings: Гіперпараметри.
        seed: Зерно (ініціалізація ваг, вибір векторів, шум, Dropout).
    """

    def __init__(self, settings: FaeSettings, seed: int) -> None:
        self._settings = settings
        self._seed = seed

    def train(
        self,
        train_features: np.ndarray,
        val_normal_features: np.ndarray,
        callbacks: list[keras.callbacks.Callback] | None = None,
        metrics_factory: Callable[[keras.Model], keras.callbacks.Callback] | None = None,
    ) -> FaeTrainingResult:
        """Навчання з раннім зупиненням за ``val_mse`` і відновленням найкращих ваг.

        Історія містить ``loss`` (втрата Keras у режимі навчання: з шумом, Dropout і штрафом
        L2 — саме її мінімізує оптимізатор) і порівнянні між собою ``train_mse`` / ``val_mse``
        (див. :class:`ReconstructionMseCallback`).

        Args:
            train_features: ``(N, h, w, D)`` агреговані ознаки train (лише норма).
            val_normal_features: ``(M, h, w, D)`` ознаки val_normal (лише норма).
            callbacks: Додаткові колбеки.
            metrics_factory: Функція ``head -> Callback`` для метрик по епохах (потрібна
                головка, яка з'являється лише всередині цього методу); ``None`` — без них.
        """
        settings = self._settings
        keras.utils.set_random_seed(self._seed)
        dim = settings.feature_dim
        train_vectors = train_features.reshape(-1, dim)
        mean, variance = channel_statistics(train_vectors)
        std = np.sqrt(variance)
        builder = FeatureAutoencoderBuilder(settings)
        core = builder.build_core()
        head = builder.build_head(core, mean, variance)
        core.compile(optimizer=keras.optimizers.Adam(settings.learning_rate), loss="mse")

        validation = self._sample(val_normal_features.reshape(-1, dim), mean, std, _VAL_OFFSET)
        train_sample = self._sample(train_vectors, mean, std, _TRAIN_OFFSET)
        # Перший колбек: чисті MSE train і val_normal (однаковий режим, без штрафу L2) —
        # за val_mse працює раннє зупинення, тож він має бути в logs раніше за EarlyStopping.
        all_callbacks: list[keras.callbacks.Callback] = [
            ReconstructionMseCallback(core, train_sample, validation)
        ]
        if metrics_factory is not None:
            all_callbacks.append(metrics_factory(head))
        all_callbacks.extend(callbacks or [])
        all_callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_mse",
                mode="min",
                patience=settings.patience,
                restore_best_weights=True,
            )
        )
        started = time.perf_counter()
        history = core.fit(
            RandomVectorDataset(train_vectors, mean, std, settings, self._seed),
            epochs=settings.max_epochs,
            callbacks=all_callbacks,
            verbose=0,
        )
        seconds = time.perf_counter() - started
        frame = pd.DataFrame(history.history)
        frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
        best_epoch = int(frame["epoch"].iloc[int(np.argmin(frame["val_mse"].to_numpy()))])
        return FaeTrainingResult(head, frame, best_epoch, seconds)

    def _sample(
        self, vectors: np.ndarray, mean: np.ndarray, std: np.ndarray, offset: int
    ) -> np.ndarray:
        """Фіксована (за seed) нормалізована підвибірка векторів форми ``(n, 1, 1, D)``."""
        generator = np.random.default_rng(self._seed + offset)
        count = min(self._settings.validation_vectors, len(vectors))
        rows = np.sort(generator.choice(len(vectors), count, replace=False))
        normalized = (np.asarray(vectors[rows], np.float32) - mean) / std
        return normalized[:, None, None, :].astype(np.float32)


def channel_statistics(vectors: np.ndarray, chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Середнє й дисперсія кожного каналу (лише train) шматками, у ``float64``.

    Мала нижня межа дисперсії (1e-6) захищає від ділення на нуль для «мертвих» каналів.
    """
    total = np.zeros(vectors.shape[1])
    squares = np.zeros(vectors.shape[1])
    for start in range(0, len(vectors), chunk):
        part = np.asarray(vectors[start : start + chunk], np.float64)
        total += part.sum(axis=0)
        squares += (part**2).sum(axis=0)
    mean = total / len(vectors)
    variance = np.maximum(squares / len(vectors) - mean**2, 1e-6)
    return mean.astype(np.float32), variance.astype(np.float32)


def fae_analytic_flops(settings: FaeSettings, n_positions: int) -> int:
    """FLOPs головки FAE на зображення: ``Σ 2·in·out`` на позицію (без зсувів і активацій)."""
    return int(
        sum(2 * fan_in * fan_out for fan_in, fan_out in settings.dense_sizes()) * n_positions
    )
