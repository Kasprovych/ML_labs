"""Аугментація табличних даних: синтез нових рядків рідкісного класу.

Усі семплери мають інтерфейс imbalanced-learn (``fit_resample``) і працюють
лише всередині ``imblearn.pipeline.Pipeline``. Такий пайплайн викликає семплер
тільки під час ``fit``, тож валідаційні й тестові дані ніколи не
аугментуються — оцінка моделі лишається чесною.
"""

from __future__ import annotations

from enum import StrEnum
from numbers import Real

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval


class GaussianNoiseSampler(BaseOverSampler):
    """Дублює рядки рідкісного класу, додаючи до них малий гаусів шум.

    Найпростіша аугментація: беремо випадковий реальний рядок міноритарного
    класу і трохи «розмиваємо» кожну ознаку: ``x' = x + N(0, (noise_scale * σ)²)``,
    де ``σ`` — стандартне відхилення ознаки в навчальній вибірці. На відміну
    від SMOTE, нова точка лежить не на відрізку між двома сусідами, а в малому
    околі однієї реальної точки.

    Реальні рядки не змінюються й ідуть у результаті першими; синтетичні
    дописуються в кінець.

    Args:
        sampling_strategy: Бажане відношення «міноритарний / мажоритарний» після
            аугментації (``0.5`` — удвічі менше, ніж мажоритарного) або ``"auto"``
            для повного вирівнювання.
        noise_scale: Стандартне відхилення шуму в частках ``σ`` ознаки.
        random_state: Зерно генератора.
    """

    # Опис допустимих значень параметрів: imblearn перевіряє їх перед fit_resample.
    _parameter_constraints: dict = {  # noqa: RUF012 - так вимагає API imblearn
        **BaseOverSampler._parameter_constraints,
        "noise_scale": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        sampling_strategy: float | str = "auto",
        noise_scale: float = 0.05,
        random_state: int | None = None,
    ) -> None:
        super().__init__(sampling_strategy=sampling_strategy)
        self.noise_scale = noise_scale
        self.random_state = random_state

    def _fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
        """Створює синтетичні рядки (викликається базовим класом із ``fit_resample``).

        Імена ``X``/``y`` тут вимушені: метод перевизначає метод imblearn.
        """
        generator = check_random_state(self.random_state)
        noise_std = self.noise_scale * X.std(axis=0)
        synthetic_rows = [X]
        synthetic_labels = [y]
        # sampling_strategy_ обчислює базовий клас: {мітка класу: скільки рядків додати}.
        for label, n_new in self.sampling_strategy_.items():
            if n_new == 0:
                continue
            class_rows = np.flatnonzero(y == label)  # номери рядків цього класу
            picked = generator.choice(class_rows, size=n_new, replace=True)
            # Broadcasting: вектор noise_std (по одному σ на колонку) автоматично
            # «розтягується» на всі n_new рядків матриці шуму.
            noise = generator.normal(size=(n_new, X.shape[1])) * noise_std
            synthetic_rows.append(X[picked] + noise)
            synthetic_labels.append(np.full(n_new, label, dtype=y.dtype))
        return np.vstack(synthetic_rows), np.concatenate(synthetic_labels)


class SamplerKind(StrEnum):
    """Метод аугментації. ``StrEnum`` — enum, значення якого є рядками."""

    NONE = "none"
    SMOTE = "smote"
    ADASYN = "adasyn"
    GAUSSIAN_NOISE = "gaussian_noise"


class SamplerFactory:
    """Створює семплери з однаковими налаштуваннями — щоб порівняння було чесним.

    * **SMOTE** — нова точка на відрізку між рядком рідкісного класу та одним
      із його ``k`` найближчих сусідів того самого класу.
    * **ADASYN** — як SMOTE, але більше точок генерується там, де рідкісний
      клас оточений чужим (на «складній» межі класів).
    * **Гаусів шум** — див. :class:`GaussianNoiseSampler`.

    Args:
        sampling_strategy: Цільове відношення «міноритарний / мажоритарний».
        k_neighbors: Кількість сусідів для SMOTE та ADASYN.
        noise_scale: Масштаб шуму для :class:`GaussianNoiseSampler`.
        random_state: Зерно генератора.
    """

    def __init__(
        self, sampling_strategy: float, k_neighbors: int, noise_scale: float, random_state: int
    ) -> None:
        self._sampling_strategy = sampling_strategy
        self._k_neighbors = k_neighbors
        self._noise_scale = noise_scale
        self._random_state = random_state

    def create(self, kind: SamplerKind) -> BaseOverSampler | None:
        """Повертає новий нефітований семплер або ``None`` для ``SamplerKind.NONE``."""
        # match — аналог switch у Java (з Python 3.10).
        match kind:
            case SamplerKind.NONE:
                return None
            case SamplerKind.SMOTE:
                return SMOTE(
                    sampling_strategy=self._sampling_strategy,
                    k_neighbors=self._k_neighbors,
                    random_state=self._random_state,
                )
            case SamplerKind.ADASYN:
                return ADASYN(
                    sampling_strategy=self._sampling_strategy,
                    n_neighbors=self._k_neighbors,
                    random_state=self._random_state,
                )
            case SamplerKind.GAUSSIAN_NOISE:
                return GaussianNoiseSampler(
                    sampling_strategy=self._sampling_strategy,
                    noise_scale=self._noise_scale,
                    random_state=self._random_state,
                )
