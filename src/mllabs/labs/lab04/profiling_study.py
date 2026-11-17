"""Профілювання детекторів: FLOPs, параметри, пам'ять, latency; компроміс «точність — вартість».

Усі 9 пар «головка × backbone» профілюються в однаковій конфігурації (вхід 256×256, сітка
32×32, базовий p) — так їх можна чесно порівнювати між собою. GPU для TensorFlow на цьому
Mac недоступний (tensorflow-metal несумісний з TF 2.21), тож latency — лише CPU; FLOPs від
пристрою не залежать.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mllabs.anomaly.backbones import BACKBONE_SPECS, BackboneKind, FeatureExtractorFactory
from mllabs.anomaly.fae import FaeSettings, FeatureAutoencoderBuilder, fae_analytic_flops
from mllabs.anomaly.images import load_resized_image
from mllabs.anomaly.postprocess import AnomalyMapPostprocessing
from mllabs.anomaly.profiling import (
    FlopsCounter,
    HeadShape,
    LatencyBenchmark,
    LatencyJob,
    LatencyRecord,
    records_to_rows,
    run_latency_job,
)
from mllabs.labs.lab04.config import Lab4Config

_logger = logging.getLogger(__name__)

#: Головки в порядку виведення.
HEADS = ("FAE", "PaDiM", "PatchCore")
_HEAD_KIND = {"FAE": "fae", "PaDiM": "padim", "PatchCore": "patchcore"}


@dataclass(frozen=True)
class HeadCost:
    """Аналітична вартість головки.

    Attributes:
        gflops: GFLOPs на зображення.
        params: Кількість чисел у параметрах головки.
        memory_mb: Пам'ять параметрів, МБ.
    """

    gflops: float
    params: int
    memory_mb: float


class ProfilingStudy:
    """FLOPs і latency.

    Args:
        config: Параметри.
        fae_settings: Обрані гіперпараметри FAE.
        patchcore_bank: Розмір банку PatchCore (з навченої моделі). Він залежить лише від
            кількості векторів train (частка coreset), тому однаковий для всіх backbone.
        patch_size: Обраний ``p`` (агрегація входить у вимірюваний конвеєр).
        tile_grid: Обраний ``g``.
        seed: Зерно.
    """

    def __init__(
        self,
        config: Lab4Config,
        fae_settings: FaeSettings,
        patchcore_bank: int,
        patch_size: int,
        tile_grid: int,
        seed: int,
    ) -> None:
        self._config = config
        self._fae = fae_settings
        self._bank = patchcore_bank
        self._patch_size = patch_size
        self._tile_grid = tile_grid
        self._seed = seed
        self._factory = FeatureExtractorFactory(config.image_size, config.feature_dim, weights=None)
        self._grid = self._factory.grid_size(tile_grid)

    def head_costs(self) -> dict[str, HeadCost]:
        """GFLOPs, параметри і пам'ять трьох головок (на сітці ознак обраної конфігурації)."""
        config, positions = self._config, self._grid**2
        builder = FeatureAutoencoderBuilder(self._fae)
        ones = np.ones(config.feature_dim, np.float32)
        head = builder.build_head(builder.build_core(), 0 * ones, ones)
        fae_flops = FlopsCounter().count(head, (1, self._grid, self._grid, config.feature_dim))
        fae_params = int(sum(np.prod(w.shape) for w in head.trainable_weights))
        depth = min(config.padim_channels, config.feature_dim)
        padim_params = positions * (depth + depth * depth)
        patchcore_params = self._bank * config.feature_dim
        return {
            "FAE": HeadCost(fae_flops / 1e9, fae_params, fae_params * 4 / 2**20),
            "PaDiM": HeadCost(
                (2 * depth * depth + 2 * depth) * positions / 1e9,
                padim_params,
                padim_params * 4 / 2**20,
            ),
            "PatchCore": HeadCost(
                2 * self._bank * config.feature_dim * positions / 1e9,
                patchcore_params,
                patchcore_params * 4 / 2**20,
            ),
        }

    def fae_flops_check(self) -> tuple[float, float]:
        """GFLOPs FAE: профайлер і аналітична формула ``Σ 2·in·out`` (перевірка)."""
        analytic = fae_analytic_flops(self._fae, self._grid**2) / 1e9
        return self.head_costs()["FAE"].gflops, analytic

    def postprocess_gflops(self) -> float:
        """GFLOPs пост-обробки (збільшення + згладжування) на зображення."""
        layer = AnomalyMapPostprocessing(self._config.eval_size, self._config.blur_sigma)
        return FlopsCounter().count(layer, (1, self._grid, self._grid)) / 1e9

    def latency(self) -> list[LatencyRecord]:
        """Latency в окремих процесах: 1 потік (пакет 1) і всі ядра (пакети 1 і 16)."""
        config = self._config
        heads = (
            HeadShape("fae", self._fae.hidden, self._fae.latent),
            HeadShape("padim", padim_channels=min(config.padim_channels, config.feature_dim)),
            HeadShape("patchcore", patchcore_bank=self._bank),
        )
        common = {
            "backbones": tuple(str(kind) for kind in config.backbones),
            "heads": heads,
            "image_size": config.image_size,
            "tile_grid": self._tile_grid,
            "feature_dim": config.feature_dim,
            "patch_size": self._patch_size,
            "eval_size": config.eval_size,
            "blur_sigma": config.blur_sigma,
            "warmup": config.latency_warmup,
            "seed": self._seed,
        }
        single = LatencyJob(threads=1, batch_sizes=(1,), runs=(config.latency_runs,), **common)
        parallel = LatencyJob(
            threads=None,
            batch_sizes=(1, config.latency_batch),
            runs=(config.latency_runs, config.latency_runs_batch),
            **common,
        )
        records = []
        for job in (single, parallel):
            _logger.info("Latency: потоки = %s", job.threads or "усі")
            records.extend(run_latency_job(job))
        return records

    def decode_latency(self, raw_images: Sequence[Path]) -> float:
        """Медіана часу «прочитати JPG + зменшити до 256», мс/зображення (1 потік Python)."""
        paths = list(raw_images)
        state = {"index": 0}

        def decode_next() -> np.ndarray:
            path = paths[state["index"] % len(paths)]
            state["index"] += 1
            return load_resized_image(path, self._config.image_size)

        bench = LatencyBenchmark(warmup=2, runs=len(paths))
        return bench.measure(decode_next, batch_size=1).median_ms


def backbone_costs(config: Lab4Config, tile_grid: int = 1) -> pd.DataFrame:
    """GFLOPs і параметри обрізаних backbone (профайлер TF, ваги не потрібні)."""
    factory = FeatureExtractorFactory(config.image_size, config.feature_dim, weights=None)
    counter = FlopsCounter()
    side = config.image_size * tile_grid
    rows = []
    for kind in config.backbones:
        extractor = factory.build(kind, tile_grid)
        backbone = extractor.get_layer("backbone")
        rows.append(
            {
                "backbone": str(kind),
                "label": BACKBONE_SPECS[kind].label,
                "tile_grid": tile_grid,
                "gflops": counter.count(extractor, (1, side, side, 3)) / 1e9,
                "params": int(backbone.count_params()),
                "layers": " + ".join(BACKBONE_SPECS[kind].layers),
                "channels": sum(BACKBONE_SPECS[kind].channels),
            }
        )
    return pd.DataFrame(rows)


def latency_frame(records: Sequence[LatencyRecord]) -> pd.DataFrame:
    """Сирі рядки latency (усі поля)."""
    return pd.DataFrame(records_to_rows(records))


def latency_lookup(
    raw: pd.DataFrame,
    threads: str,
    component: str,
    head: str,
    batch: int,
    column: str = "median_ms",
) -> dict[str, float]:
    """Значення ``column`` (медіана або p90) для кожного backbone при заданих умовах."""
    rows = raw[
        (raw["threads"] == threads)
        & (raw["component"] == component)
        & (raw["head"] == head)
        & (raw["batch_size"] == batch)
    ]
    return dict(zip(rows["backbone"], rows[column], strict=True))


def end_to_end_latency(raw: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Наскрізна медіана latency (усі ядра, пакет 1) для кожної пари «головка × backbone»."""
    return {
        (head, backbone): value
        for head in HEADS
        for backbone, value in latency_lookup(raw, "all", "end_to_end", head_kind(head), 1).items()
    }


def latency_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Компактна таблиця latency для звіту (7 колонок)."""
    labels = {str(kind): spec.label for kind, spec in BACKBONE_SPECS.items()}
    components = {"backbone": "backbone", "head": "головка + пост-обр.", "end_to_end": "наскрізно"}
    heads = {"": "—", **{kind: head for head, kind in _HEAD_KIND.items()}}
    return pd.DataFrame(
        {
            "Потоки": raw["threads"].map({"1": "1", "all": "усі"}),
            "Backbone": raw["backbone"].map(labels),
            "Компонент": raw["component"].map(components),
            "Головка": raw["head"].map(heads),
            "Пакет": raw["batch_size"],
            "Медіана, мс/зобр.": raw["median_ms"],
            "p90, мс/зобр.": raw["p90_ms"],
        }
    )


def flops_table(backbones: pd.DataFrame, head_costs: Mapping[str, HeadCost]) -> pd.DataFrame:
    """FLOPs, параметри і пам'ять для всіх пар «backbone × головка» (7 колонок)."""
    rows = [
        {
            "Backbone": backbone["label"],
            "Головка": head,
            "GFLOPs backbone": backbone["gflops"],
            "GFLOPs головки": head_costs[head].gflops,
            "GFLOPs разом": backbone["gflops"] + head_costs[head].gflops,
            "Параметри головки, тис.": head_costs[head].params / 1e3,
            "Пам'ять головки, МБ": head_costs[head].memory_mb,
        }
        for _, backbone in backbones.iterrows()
        for head in HEADS
    ]
    return pd.DataFrame(rows)


def profiling_metrics(
    raw: pd.DataFrame, head_costs: Mapping[str, HeadCost], gflops: Mapping[str, float], batch: int
) -> dict[str, dict]:
    """Розділ ``metrics.profiling.<головка>.<backbone>``."""
    result: dict[str, dict] = {}
    for head in HEADS:
        kind = head_kind(head)
        median = latency_lookup(raw, "all", "end_to_end", kind, 1)
        p90 = latency_lookup(raw, "all", "end_to_end", kind, 1, column="p90_ms")
        batched = latency_lookup(raw, "all", "end_to_end", kind, batch)
        single = latency_lookup(raw, "1", "end_to_end", kind, 1)
        cost = head_costs[head]
        result[head] = {
            backbone: {
                "gflops_backbone": gflops[backbone],
                "gflops_head": cost.gflops,
                "params_head": cost.params,
                "memory_mb": cost.memory_mb,
                "latency_ms_b1_median": median[backbone],
                "latency_ms_b1_p90": p90[backbone],
                "latency_ms_b16": batched[backbone],
                "latency_ms_b1_1thread": single[backbone],
            }
            for backbone in gflops
        }
    return result


def pareto_mask(cost: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """Точки фронту Парето: немає іншої, що не дорожча й не гірша (і краща хоч в одному)."""
    mask = np.ones(len(cost), dtype=bool)
    for index in range(len(cost)):
        dominated = (cost <= cost[index]) & (quality >= quality[index])
        strictly = (cost < cost[index]) | (quality > quality[index])
        mask[index] = not np.any(dominated & strictly)
    return mask


def accuracy_vs_cost(
    quality: Mapping[tuple[str, str], float],
    backbone_gflops: Mapping[str, float],
    head_costs: Mapping[str, HeadCost],
    latency_b1: Mapping[tuple[str, str], float],
) -> pd.DataFrame:
    """Таблиця «головка × backbone»: якість на val, GFLOPs, latency і фронт Парето."""
    rows = []
    for (head, backbone), value in quality.items():
        rows.append(
            {
                "Головка": head,
                "Backbone": BACKBONE_SPECS[BackboneKind(backbone)].label,
                "Pixel AUROC (val)": value,
                "GFLOPs": backbone_gflops[backbone] + head_costs[head].gflops,
                "Latency, мс": latency_b1[(head, backbone)],
            }
        )
    table = pd.DataFrame(rows)
    quality_values = table["Pixel AUROC (val)"].to_numpy()
    table["Парето (latency)"] = pareto_mask(table["Latency, мс"].to_numpy(), quality_values)
    table["Парето (FLOPs)"] = pareto_mask(table["GFLOPs"].to_numpy(), quality_values)
    return table


def head_kind(head: str) -> str:
    """Назва головки → ключ у таблиці latency."""
    return _HEAD_KIND[head]
