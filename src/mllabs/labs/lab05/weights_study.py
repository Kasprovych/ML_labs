"""Сценарій аналізу ваг FAE для однієї категорії (R1): збирає частини з ``weights_analysis``.

Дані: train — підвибірка векторів для PCA, латентних статистик і середніх внесків каналів;
val (розмічений) — лише порядок каналів за реакцією на дефект; test — звітні числа.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from mllabs.anomaly.backbones import BACKBONE_SPECS, BackboneKind
from mllabs.anomaly.metrics import HistogramAuc
from mllabs.anomaly.splits import SplitPart
from mllabs.labs.lab05.config import Lab5Config
from mllabs.labs.lab05.handoff import CategoryAssets, FaeParts, fae_parts
from mllabs.labs.lab05.inference import CachedScorer
from mllabs.labs.lab05.weights_analysis import (
    DEFECT,
    NORMAL,
    ErrorDecomposition,
    JacobianAnalyzer,
    JacobianSummary,
    LatentAnalyzer,
    LayerAttribution,
    LinearProjectionBaseline,
    SpectrumSummary,
    decompose,
    input_sensitivity,
    layer_share_table,
    participation_ratio,
    patch_coverage,
    patch_groups,
    principal_angle_cosines,
    sample_vectors,
    spearman,
    weight_spectra,
)

#: Скільки точок train і дефектних патчів показувати на PCA-2D (щоб рисунок не «заливало»).
_SCATTER_TRAIN = 3000
_SCATTER_DEFECT = 1500


@dataclass(frozen=True)
class LayerExample:
    """Приклад для рисунка «карта моделі = шар 8 + шар 16».

    Attributes:
        image: ``(S, S, 3)`` зображення.
        mask: ``(E, E)`` маска.
        full: Карта моделі.
        shallow: Карта «лише шар 8».
        deep: Карта «лише шар 16».
    """

    image: np.ndarray
    mask: np.ndarray
    full: np.ndarray
    shallow: np.ndarray
    deep: np.ndarray


@dataclass(frozen=True)
class CategoryWeights:
    """Результати аналізу ваг однієї категорії.

    Attributes:
        layer_rows: Рядки таблиці часток шарів по групах патчів.
        map_rows: Pixel AUROC/AP карт «повна / шар 8 / шар 16 / Махаланобіс / PCA».
        topk_rows: Pixel AUROC від кількості top-k каналів.
        spectra: Спектри матриць ``Dense``.
        sensitivity_spearman: ρ(‖W_enc[d,:]‖, середній внесок каналу на train).
        participation_ratio: Ефективна розмірність латентних кодів train.
        pca_fae_spearman: ρ оцінок патчів test FAE і PCA.
        angle_cosines: Косинуси головних кутів PCA ↔ декодер.
        jacobians: Група патчів → спектри якобіанів.
        scatter: Точки PCA-2D латенту: ``train`` і ``defect``.
        example: Приклад карт по шарах.
    """

    layer_rows: list[dict]
    map_rows: list[dict]
    topk_rows: list[dict]
    spectra: list[SpectrumSummary]
    sensitivity_spearman: float
    participation_ratio: float
    pca_fae_spearman: float
    angle_cosines: np.ndarray
    jacobians: dict[str, JacobianSummary]
    scatter: dict[str, np.ndarray]
    example: LayerExample


class WeightsStudy:
    """Повний аналіз ваг FAE однієї категорії.

    Args:
        config: Параметри ЛР5.
        seed: Зерно (підвибірки, PCA).
    """

    def __init__(self, config: Lab5Config, seed: int) -> None:
        self._config = config
        self._seed = seed
        self._histogram = HistogramAuc(config.histogram_bins)

    def run(self, assets: CategoryAssets, scorer: CachedScorer) -> CategoryWeights:
        """Виконує всі чотири аналізи D1 для категорії."""
        config = self._config
        generator = np.random.default_rng(self._seed)
        parts = fae_parts(assets.head)
        spec = BACKBONE_SPECS[BackboneKind(assets.config.backbone)]
        layers = LayerAttribution(*spec.channels, assets.config.feature_dim)
        decomposition = ErrorDecomposition(parts, config.head_batch)

        val_rows = assets.indices(SplitPart.VAL)
        val = decompose(
            decomposition,
            assets.features,
            val_rows,
            self._groups(assets, val_rows),
            layers,
            None,
            (),
        )
        order = np.argsort(-val.channel_ratio(), kind="stable")  # порядок — лише з val

        test_rows = assets.indices(SplitPart.TEST)
        test_masks = assets.masks[test_rows]
        test_groups = self._groups(assets, test_rows)
        test = decompose(
            decomposition,
            assets.features,
            test_rows,
            test_groups,
            layers,
            order,
            config.top_k_channels,
        )
        full_patch = test.layer_maps.sum(axis=-1)
        full_maps = scorer.maps(full_patch)
        shallow_maps = scorer.maps(test.layer_maps[..., 0])
        deep_maps = scorer.maps(test.layer_maps[..., 1])
        map_rows = [
            self._map_row(assets.category, "FAE (повна)", full_maps, test_masks),
            self._map_row(assets.category, "лише шар 8", shallow_maps, test_masks),
            self._map_row(assets.category, "лише шар 16", deep_maps, test_masks),
        ]
        topk_rows = []
        for k, maps in test.topk_maps.items():
            auroc, ap = self._histogram.metrics(scorer.maps(maps), test_masks)
            topk_rows.append(
                {
                    "Категорія": assets.category,
                    "k каналів": k,
                    "Pixel AUROC": auroc,
                    "Pixel AP": ap,
                    "Частка шару 8 у top-k": float(layers.shallow_share[order[:k]].mean()),
                }
            )

        train_rows = assets.indices(SplitPart.TRAIN)
        train_vectors = sample_vectors(assets.features, train_rows, config.train_vectors, generator)
        train_normalized = parts.normalize(train_vectors)
        mean_contribution = decomposition.channel_errors(train_vectors[:, None, None, :])
        sensitivity = spearman(
            input_sensitivity(parts.dense_kernels["encoder_hidden"]),
            mean_contribution.reshape(-1, mean_contribution.shape[-1]).mean(axis=0),
        )

        latent = LatentAnalyzer(parts, config.mahalanobis_batch)
        train_codes = latent.codes(train_normalized)
        latent.fit(train_codes)
        linear = LinearProjectionBaseline(config.pca_components, self._seed).fit(train_normalized)
        mahalanobis_patch, pca_patch = self._latent_and_linear_maps(
            assets, test_rows, parts, latent, linear
        )
        mahalanobis_maps = scorer.maps(mahalanobis_patch)
        pca_maps = scorer.maps(pca_patch)
        map_rows.append(
            self._map_row(assets.category, "Махаланобіс у латенті", mahalanobis_maps, test_masks)
        )
        map_rows.append(
            self._map_row(assets.category, "PCA-64 (лінійний АЕ)", pca_maps, test_masks)
        )

        jacobians = self._jacobians(assets, parts, test_rows, test_groups, generator)
        scatter = self._scatter(
            train_codes, assets, parts, latent, test_rows, test_groups, generator
        )
        example_index = self._example_index(test_masks, assets.labels(test_rows))
        example = LayerExample(
            np.asarray(assets.images[test_rows[example_index]]),
            test_masks[example_index],
            full_maps[example_index],
            shallow_maps[example_index],
            deep_maps[example_index],
        )
        return CategoryWeights(
            layer_rows=layer_share_table(assets.category, test, layers),
            map_rows=map_rows,
            topk_rows=topk_rows,
            spectra=weight_spectra(parts.dense_kernels),
            sensitivity_spearman=sensitivity,
            participation_ratio=participation_ratio(train_codes),
            pca_fae_spearman=spearman(full_patch, pca_patch),
            angle_cosines=self._angles(linear, parts.dense_kernels["reconstruction"]),
            jacobians=jacobians,
            scatter=scatter,
            example=example,
        )

    def _groups(self, assets: CategoryAssets, rows: np.ndarray) -> np.ndarray:
        coverage = patch_coverage(assets.masks[rows], assets.grid)
        return patch_groups(coverage, assets.labels(rows), self._config.defect_patch_coverage)

    def _map_row(
        self, category: str, name: str, maps: np.ndarray, masks: np.ndarray
    ) -> dict[str, object]:
        auroc, ap = self._histogram.metrics(maps, masks)
        return {"Категорія": category, "Карта": name, "Pixel AUROC": auroc, "Pixel AP": ap}

    def _latent_and_linear_maps(
        self,
        assets: CategoryAssets,
        rows: np.ndarray,
        parts: FaeParts,
        latent: LatentAnalyzer,
        linear: LinearProjectionBaseline,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Карти патчів test: Махаланобіс у латенті й залишок PCA (пакетами по зображеннях)."""
        grid, dim = assets.grid, assets.config.feature_dim
        mahalanobis, residual = [], []
        for start in range(0, len(rows), self._config.head_batch):
            chunk = rows[start : start + self._config.head_batch]
            normalized = parts.normalize(np.asarray(assets.features[chunk])).reshape(-1, dim)
            mahalanobis.append(latent.mahalanobis(latent.codes(normalized)))
            residual.append(linear.residual(normalized))
        shape = (len(rows), grid, grid)
        return np.concatenate(mahalanobis).reshape(shape), np.concatenate(residual).reshape(shape)

    def _patch_vectors(
        self,
        assets: CategoryAssets,
        rows: np.ndarray,
        groups: np.ndarray,
        group: str,
        count: int,
        generator: np.random.Generator,
    ) -> np.ndarray:
        """Випадкові ``count`` векторів ознак test заданої групи."""
        image, row, column = np.nonzero(groups == group)
        if len(image) > count:
            chosen = np.sort(generator.choice(len(image), count, replace=False))
            image, row, column = image[chosen], row[chosen], column[chosen]
        return np.asarray(assets.features[rows[image], row, column], np.float32)

    def _jacobians(
        self,
        assets: CategoryAssets,
        parts: FaeParts,
        rows: np.ndarray,
        groups: np.ndarray,
        generator: np.random.Generator,
    ) -> dict[str, JacobianSummary]:
        analyzer = JacobianAnalyzer(parts.core, self._config.jacobian_batch)
        count = self._config.jacobian_patches
        return {
            group: analyzer.summarize(
                parts.normalize(self._patch_vectors(assets, rows, groups, group, count, generator))
            )
            for group in (NORMAL, DEFECT)
        }

    def _scatter(
        self,
        train_codes: np.ndarray,
        assets: CategoryAssets,
        parts: FaeParts,
        latent: LatentAnalyzer,
        rows: np.ndarray,
        groups: np.ndarray,
        generator: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """PCA-2D латентних кодів: площина задається лише train-кодами."""
        projection = PCA(2, random_state=self._seed).fit(train_codes)
        shown = generator.choice(
            len(train_codes), min(_SCATTER_TRAIN, len(train_codes)), replace=False
        )
        defect = self._patch_vectors(assets, rows, groups, DEFECT, _SCATTER_DEFECT, generator)
        return {
            "train": projection.transform(train_codes[shown]),
            "defect": projection.transform(latent.codes(parts.normalize(defect))),
        }

    @staticmethod
    def _angles(linear: LinearProjectionBaseline, decoder_kernel: np.ndarray) -> np.ndarray:
        return principal_angle_cosines(linear.basis, decoder_kernel)

    @staticmethod
    def _example_index(masks: np.ndarray, labels: np.ndarray) -> int:
        """Аномалія з медіанною площею дефекту — «типовий» приклад."""
        areas = masks.reshape(len(masks), -1).mean(axis=1)
        anomalies = np.flatnonzero(labels == 1)
        ordered = anomalies[np.argsort(areas[anomalies])]
        return int(ordered[len(ordered) // 2])
