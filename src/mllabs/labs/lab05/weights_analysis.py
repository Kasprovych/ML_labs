"""Інтерпретованість ваг FAE (R1): розклад помилки, спектри ваг, латентний простір, лінійність.

Модель ЛР4 — ланцюг ``x → φ(x) → F = φ·P → z = (F − μ)/σ → ẑ = g(h(z)) → e = ‖ẑ − z‖²/D``:

* ``φ`` — активації двох шарів backbone (крок 8: 240 каналів, крок 16: 672 канали);
* ``P`` — фіксована матриця усереднення каналів 912 → 512 (``ChannelPooling``);
* ``h``/``g`` — енкодер і декодер FAE (лише вони навчалися);
* ``e`` — помилка відновлення патча = оцінка аномальності.

Звідси чотири точні (без апроксимацій) погляди на «що вивчили ваги»:

1. ``e = Σ_d e_d`` — адитивний розклад за каналами, а через ``P`` — за шарами backbone;
2. сингулярні числа матриць ``Dense`` — скільки напрямів реально використано;
3. латентний код ``h(z)`` — розмірність і відстань Махаланобіса як альтернативна оцінка;
4. PCA-k — оптимальний ЛІНІЙНИЙ автоенкодер; якобіан FAE (мережа з leaky ReLU кусково-
   лінійна, тож локально ``ẑ ≈ J z + b``) показує, наскільки FAE — «проектор на многовид норм».
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from mllabs.anomaly.backbones import channel_pooling_matrix
from mllabs.anomaly.metrics import HistogramAuc
from mllabs.labs.lab05.handoff import FaeParts

#: Групи патчів test: патчі нормальних зображень, дефектні патчі й решта патчів аномалій.
NORMAL, DEFECT, CONTEXT = "normal", "defect", "context"
PATCH_GROUPS = (NORMAL, CONTEXT, DEFECT)


def patch_coverage(masks: np.ndarray, grid: int) -> np.ndarray:
    """Частка пікселів маски в кожному патчі сітки ``grid×grid``: ``(N, grid, grid)``."""
    side = masks.shape[1] // grid
    # reshape (N, grid, side, grid, side) і середнє по осях патча = частка дефекту в ньому.
    blocks = np.asarray(masks, np.float32).reshape(len(masks), grid, side, grid, side)
    return blocks.mean(axis=(2, 4))


def patch_groups(coverage: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Група кожного патча (``NORMAL``/``DEFECT``/``CONTEXT``) як масив рядків ``(N, h, w)``."""
    groups = np.full(coverage.shape, CONTEXT, dtype=object)
    groups[labels == 0] = NORMAL
    groups[(labels[:, None, None] == 1) & (coverage >= threshold)] = DEFECT
    # Патчі аномалій, частково зачеплені маскою (0 < частка < поріг), — ні дефект, ні контекст.
    groups[(labels[:, None, None] == 1) & (coverage > 0) & (coverage < threshold)] = ""
    return groups


class LayerAttribution:
    """Частка кожного шару backbone у вихідному каналі (з фіксованої матриці ``P``).

    Канал ``d`` після ``ChannelPooling`` — середнє 1–2 сусідніх вхідних каналів. Вхідні
    канали ``0 … C₁−1`` — шар кроку 8, решта — шар кроку 16. Частка шару 8 у каналі ``d`` =
    сума ваг ``P[i, d]`` по ``i < C₁`` (стовпці ``P`` сумуються до 1).

    Args:
        shallow_channels: ``C₁`` (240 для EfficientNetB0).
        deep_channels: ``C₂`` (672).
        feature_dim: ``D``.
    """

    def __init__(self, shallow_channels: int, deep_channels: int, feature_dim: int) -> None:
        pooling = channel_pooling_matrix(shallow_channels + deep_channels, feature_dim)
        self._shallow = pooling[:shallow_channels].sum(axis=0)

    @property
    def shallow_share(self) -> np.ndarray:
        """``(D,)`` частка шару 8 у кожному каналі."""
        return self._shallow

    @property
    def weights(self) -> np.ndarray:
        """``(D, 2)``: стовпці — частки шару 8 і шару 16."""
        return np.stack([self._shallow, 1 - self._shallow], axis=1)


class ErrorDecomposition:
    """Канальні внески ``e_d = (ẑ_d − z_d)² / D`` у помилку відновлення патча.

    Сума внесків по ``d`` дорівнює виходу головки FAE (точна тотожність).

    Args:
        parts: Складові FAE.
        batch_size: Зображень у пакеті.
    """

    def __init__(self, parts: FaeParts, batch_size: int) -> None:
        self._parts = parts
        self._batch = batch_size
        self._core = tf.function(
            lambda batch: parts.core(batch, training=False), reduce_retracing=True
        )

    def channel_errors(self, features: np.ndarray) -> np.ndarray:
        """``(n, h, w, D)`` ознак → ``(n, h, w, D)`` внесків каналів."""
        normalized = self._parts.normalize(features)
        reconstruction = self._core(tf.constant(normalized)).numpy()
        return np.square(reconstruction - normalized) / normalized.shape[-1]

    def batches(
        self, features: np.ndarray, rows: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Генератор пар (номери рядків пакета, внески каналів пакета).

        ``yield`` робить функцію генератором (аналог ``Iterator`` у Java): наступний пакет
        рахується лише тоді, коли його попросили, тож усі внески не лежать у пам'яті разом.
        """
        for start in range(0, len(rows), self._batch):
            chunk = rows[start : start + self._batch]
            yield chunk, self.channel_errors(np.asarray(features[chunk]))


@dataclass(frozen=True)
class DecompositionResult:
    """Розклад помилки на одній частині розбиття.

    Attributes:
        group_sums: Група патчів → ``(D,)`` сума внесків каналів.
        group_counts: Група → кількість патчів.
        layer_maps: ``(N, h, w, 2)`` помилка «лише шар 8» і «лише шар 16».
        topk_maps: k → ``(N, h, w)`` помилка з top-k каналів.
    """

    group_sums: dict[str, np.ndarray]
    group_counts: dict[str, int]
    layer_maps: np.ndarray
    topk_maps: dict[int, np.ndarray]

    def channel_ratio(self, floor: float = 1e-12) -> np.ndarray:
        """``R_d = E[e_d | дефект] / E[e_d | норма]`` — наскільки канал «реагує» на дефект."""
        defect = self.group_sums[DEFECT] / max(self.group_counts[DEFECT], 1)
        normal = self.group_sums[NORMAL] / max(self.group_counts[NORMAL], 1)
        return defect / np.maximum(normal, floor)


def decompose(
    decomposition: ErrorDecomposition,
    features: np.ndarray,
    rows: np.ndarray,
    groups: np.ndarray,
    layers: LayerAttribution,
    channel_order: np.ndarray | None,
    top_k: Sequence[int],
) -> DecompositionResult:
    """Проходить частину розбиття пакетами й накопичує суми та карти шарів / top-k.

    Args:
        decomposition: Розклад помилки.
        features: Кеш ознак.
        rows: Рядки частини.
        groups: ``(len(rows), h, w)`` групи патчів (з :func:`patch_groups`).
        layers: Частки шарів у каналах.
        channel_order: Порядок каналів за важливістю (з val) або ``None`` — без top-k карт.
        top_k: Кількості каналів.
    """
    dim = features.shape[-1]
    sums = {group: np.zeros(dim) for group in PATCH_GROUPS}
    counts = dict.fromkeys(PATCH_GROUPS, 0)
    layer_maps, topk = [], {k: [] for k in top_k}
    offset = 0
    for chunk, errors in decomposition.batches(features, rows):
        chunk_groups = groups[offset : offset + len(chunk)]
        offset += len(chunk)
        for group in PATCH_GROUPS:
            selected = errors[chunk_groups == group]
            sums[group] += selected.sum(axis=0)
            counts[group] += len(selected)
        layer_maps.append(errors @ layers.weights)
        if channel_order is not None:
            for k in top_k:
                topk[k].append(errors[..., channel_order[:k]].sum(axis=-1))
    return DecompositionResult(
        sums,
        counts,
        np.concatenate(layer_maps),
        {k: np.concatenate(parts) for k, parts in topk.items() if parts},
    )


def layer_share_table(
    category: str, result: DecompositionResult, layers: LayerAttribution
) -> list[dict[str, object]]:
    """Рядки таблиці «частка шару 8 / 16 у помилці» по групах патчів."""
    labels = {NORMAL: "нормальні", CONTEXT: "контекст дефекту", DEFECT: "дефектні"}
    rows = []
    for group in PATCH_GROUPS:
        total = result.group_sums[group].sum()
        count = result.group_counts[group]
        shallow = float(result.group_sums[group] @ layers.shallow_share / total) if total else 0.0
        rows.append(
            {
                "Категорія": category,
                "Група патчів": labels[group],
                "Патчів": count,
                "Частка шару 8": shallow,
                "Частка шару 16": 1 - shallow if total else 0.0,
                "Середня помилка": float(total / count) if count else float("nan"),
            }
        )
    return rows


def pixel_metrics(
    histogram: HistogramAuc, maps: np.ndarray, masks: np.ndarray
) -> tuple[float, float]:
    """Pixel AUROC і AP (гістограмне наближення, похибка ~1e-4)."""
    return histogram.metrics(maps, masks)


# ---------------------------------------------------------------- спектри ваг


@dataclass(frozen=True)
class SpectrumSummary:
    """Сингулярні числа однієї матриці ваг.

    Attributes:
        layer: Ім'я шару ``Dense``.
        shape: Форма матриці ``(входи, виходи)``.
        singular_values: Сингулярні числа за спаданням.
    """

    layer: str
    shape: tuple[int, int]
    singular_values: np.ndarray

    @property
    def effective_rank(self) -> float:
        """``exp(H(p))``, ``p_i = σ_i² / Σσ²`` — «скільки напрямів реально працює»."""
        energy = self.singular_values**2
        share = energy / energy.sum()
        share = share[share > 0]
        return float(np.exp(-(share * np.log(share)).sum()))


def weight_spectra(kernels: Mapping[str, np.ndarray]) -> list[SpectrumSummary]:
    """SVD кожної матриці ``Dense`` (``compute_uv=False`` — лише сингулярні числа)."""
    return [
        SpectrumSummary(name, kernel.shape, np.linalg.svd(kernel, compute_uv=False))
        for name, kernel in kernels.items()
    ]


def input_sensitivity(encoder_kernel: np.ndarray) -> np.ndarray:
    """``‖W_enc[d, :]‖`` — наскільки сильно перший шар енкодера «слухає» канал ``d``."""
    return np.linalg.norm(encoder_kernel, axis=1)


# ---------------------------------------------------------------- латентний простір


def sample_vectors(
    features: np.ndarray, rows: np.ndarray, count: int, generator: np.random.Generator
) -> np.ndarray:
    """Випадкова підвибірка ``count`` векторів ознак із рядків ``rows`` (``float32``).

    Індекси сортуються, а кожне зображення читається з memmap один раз — послідовний доступ.
    """
    per_image = features.shape[1] * features.shape[2]
    total = len(rows) * per_image
    flat = np.sort(generator.choice(total, size=min(count, total), replace=False))
    images, positions = flat // per_image, flat % per_image
    vectors = np.empty((len(flat), features.shape[-1]), np.float32)
    for image in np.unique(images):
        selected = images == image
        block = np.asarray(features[rows[image]], np.float32).reshape(per_image, -1)
        vectors[selected] = block[positions[selected]]
    return vectors


def participation_ratio(vectors: np.ndarray) -> float:
    """``(Σλ)² / Σλ²`` власних чисел коваріації — ефективна кількість вимірів хмари точок."""
    eigenvalues = np.linalg.eigvalsh(np.cov(vectors, rowvar=False))
    return float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


class LatentAnalyzer:
    """Латентні коди FAE: розмірність і відстань Махаланобіса до норми.

    Велика помилка відновлення означає «точка поза многовидом норм»; велика відстань
    Махаланобіса в латенті — «на многовиді, але в рідкісному місці». Порівняння двох
    оцінок показує, яку з властивостей дефектів використовує FAE.

    Args:
        parts: Складові FAE.
        batch_size: Векторів у пакеті кодування.
    """

    def __init__(self, parts: FaeParts, batch_size: int) -> None:
        self._parts = parts
        self._batch = batch_size
        self._encode = tf.function(
            lambda batch: parts.encoder(batch, training=False), reduce_retracing=True
        )
        self._mean: np.ndarray | None = None
        self._precision: np.ndarray | None = None

    def codes(self, normalized: np.ndarray) -> np.ndarray:
        """``(n, D)`` нормалізованих векторів → ``(n, k)`` латентних кодів."""
        parts = []
        for start in range(0, len(normalized), self._batch):
            chunk = normalized[start : start + self._batch][:, None, None, :]
            parts.append(self._encode(tf.constant(chunk)).numpy()[:, 0, 0, :])
        return np.concatenate(parts)

    def fit(self, train_codes: np.ndarray) -> LatentAnalyzer:
        """Середнє й обернена коваріація кодів train (``pinv`` — стійко до виродженості)."""
        self._mean = train_codes.mean(axis=0)
        self._precision = np.linalg.pinv(np.cov(train_codes, rowvar=False))
        return self

    def mahalanobis(self, codes: np.ndarray) -> np.ndarray:
        """Квадрат відстані Махаланобіса кожного коду до середнього train."""
        if self._mean is None or self._precision is None:
            raise RuntimeError("Спершу викличте fit()")
        centered = codes - self._mean
        # einsum: для кожного рядка i — cᵢᵀ Σ⁻¹ cᵢ (без явного циклу).
        return np.einsum("ij,jk,ik->i", centered, self._precision, centered)


# ---------------------------------------------------------------- лінійна проекція


class LinearProjectionBaseline:
    """PCA-k як «лінійний автоенкодер»: оцінка патча = залишок проекції на k компонент.

    Серед усіх лінійних автоенкодерів з латентом k саме PCA дає найменшу середню помилку
    (теорема Еккарта–Янга), тож це повністю прозорий еталон для FAE.

    Args:
        components: k.
        seed: Зерно рандомізованого SVD.
    """

    def __init__(self, components: int, seed: int) -> None:
        self._pca = PCA(components, svd_solver="randomized", random_state=seed)

    def fit(self, normalized_train: np.ndarray) -> LinearProjectionBaseline:
        """Головні компоненти нормалізованих train-векторів."""
        self._pca.fit(normalized_train)
        return self

    @property
    def basis(self) -> np.ndarray:
        """``(D, k)`` ортонормований базис підпростору PCA."""
        return self._pca.components_.T

    def residual(self, normalized: np.ndarray) -> np.ndarray:
        """``‖z − proj(z)‖² / D`` для ``(..., D)``."""
        flat = normalized.reshape(-1, normalized.shape[-1])
        restored = self._pca.inverse_transform(self._pca.transform(flat))
        errors = np.square(flat - restored).mean(axis=1)
        return errors.reshape(normalized.shape[:-1])


def principal_angle_cosines(basis: np.ndarray, decoder_kernel: np.ndarray) -> np.ndarray:
    """Косинуси головних кутів між підпростором PCA і вихідним підпростором декодера.

    Вихід останнього ``Dense`` (``y = x·W``, ``W`` — ``(256, D)``) лежить у лінійній
    оболонці рядків ``W``; її головні k напрямів — праві сингулярні вектори. Косинуси кутів
    між двома ортонормованими базисами — сингулярні числа ``Aᵀ·B`` (1 — напрям спільний).
    """
    _, _, right = np.linalg.svd(decoder_kernel, full_matrices=False)
    decoder_basis = right[: basis.shape[1]].T
    return np.linalg.svd(basis.T @ decoder_basis, compute_uv=False)


# ---------------------------------------------------------------- якобіан


@dataclass(frozen=True)
class JacobianSummary:
    """Спектри якобіанів однієї групи патчів.

    Attributes:
        singular_values: ``(n, D)`` сингулярні числа кожного якобіана за спаданням.
        projection_error: ``(n,)`` ``‖J² − J‖_F / ‖J‖_F`` (0 — ідеальний проектор).
    """

    singular_values: np.ndarray
    projection_error: np.ndarray

    def rank(self, threshold: float) -> np.ndarray:
        """Кількість «збережених» напрямів (σ_i > ``threshold``) для кожного патча."""
        return (self.singular_values > threshold).sum(axis=1)


class JacobianAnalyzer:
    """Локальний якобіан ``J = ∂ẑ/∂z`` ядра FAE у точках-патчах.

    Для лінійного проектора на k-вимірний підпростір ``J² = J``, k сингулярних чисел = 1,
    решта = 0. Наскільки FAE відхиляється від цього — видно зі спектра.

    Args:
        core: Ядро FAE (вхід ``(h, w, D)``).
        batch_size: Патчів у пакеті ``batch_jacobian``.
    """

    def __init__(
        self, core: keras.Model | Callable[[tf.Tensor], tf.Tensor], batch_size: int
    ) -> None:
        self._batch = batch_size

        @tf.function
        def jacobian(vectors: tf.Tensor) -> tf.Tensor:
            with tf.GradientTape() as tape:
                tape.watch(vectors)
                output = core(vectors[:, None, None, :], training=False)[:, 0, 0, :]
            # batch_jacobian: для кожного рядка пакета окремо — (n, D_out, D_in).
            return tape.batch_jacobian(output, vectors)

        self._jacobian = jacobian

    def jacobians(self, normalized: np.ndarray) -> np.ndarray:
        """``(n, D)`` → ``(n, D, D)``."""
        parts = [
            self._jacobian(tf.constant(normalized[start : start + self._batch], tf.float32))
            for start in range(0, len(normalized), self._batch)
        ]
        return np.concatenate([part.numpy() for part in parts])

    def summarize(self, normalized: np.ndarray) -> JacobianSummary:
        """Спектр і «проекційність» для кожного патча."""
        singular, errors = [], []
        for start in range(0, len(normalized), self._batch):
            jac = self.jacobians(normalized[start : start + self._batch])
            singular.append(np.linalg.svd(jac, compute_uv=False))
            square = jac @ jac  # @ — множення матриць (для кожного патча пакета окремо)
            errors.append(
                np.linalg.norm(square - jac, axis=(1, 2)) / np.linalg.norm(jac, axis=(1, 2))
            )
        return JacobianSummary(np.concatenate(singular), np.concatenate(errors))


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Рангова кореляція Спірмена двох однакових за формою масивів."""
    return float(spearmanr(first.reshape(-1), second.reshape(-1)).statistic)


def spectra_table(spectra: Mapping[str, Sequence[SpectrumSummary]]) -> pd.DataFrame:
    """Таблиця «категорія, шар, форма, σ_max, ефективний ранг, макс. ранг»."""
    return pd.DataFrame(
        [
            {
                "Категорія": category,
                "Шар": spectrum.layer,
                "Форма": f"{spectrum.shape[0]}×{spectrum.shape[1]}",
                "σ_max": float(spectrum.singular_values[0]),
                "σ_min": float(spectrum.singular_values[-1]),
                "Ефективний ранг": spectrum.effective_rank,
                "Макс. ранг": min(spectrum.shape),
            }
            for category, items in spectra.items()
            for spectrum in items
        ]
    )
