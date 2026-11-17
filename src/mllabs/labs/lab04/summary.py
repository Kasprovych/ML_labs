"""Зведені таблиці й розділи ``metrics.json`` ЛР4 (лише перетворення результатів у таблиці).

Таблиці, що йдуть у звіт, мають не більше 7 колонок; повні версії з усіма полями
зберігаються окремо (``*_full.csv``) — для перевірки й для ЛР5.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace

import keras
import numpy as np
import pandas as pd

from mllabs.labs.lab04.benchmark import SotaFit, nan_if_missing
from mllabs.labs.lab04.config import Lab4Config
from mllabs.labs.lab04.evaluation import Evaluation
from mllabs.labs.lab04.studies import FaeRun
from mllabs.modeling import BootstrapInterval


def run_record(run: FaeRun) -> dict[str, object]:
    """Усі поля одного навчання FAE (для ``study_runs_full.csv`` і ``metrics.json``)."""
    best = run.best_row()
    settings = run.variant.settings
    return {
        "run_id": run.run_id,
        "study": run.variant.study,
        "label": run.variant.label,
        "category": run.category,
        "backbone": str(run.variant.backbone),
        "patch_size": run.variant.patch_size,
        "tile_grid": run.variant.tile_grid,
        "latent": settings.latent,
        "dropout": settings.dropout,
        "l2": settings.l2,
        "input_noise": settings.input_noise,
        "pixel_auroc": run.val_pixel_auroc,
        "pixel_ap": run.val_pixel_ap,
        "image_auroc": run.val_image_auroc,
        "train_loss": float(best["train_mse"]),
        "val_loss": float(best["val_mse"]),
        "loss_gap": float(best["val_mse"] - best["train_mse"]),
        "epochs": len(run.training.history),
        "best_epoch": run.training.best_epoch,
        "train_seconds": run.training.seconds,
    }


def study_table(runs: Sequence[FaeRun]) -> pd.DataFrame:
    """Таблиця дослідження для звіту (7 колонок)."""
    records = [run_record(run) for run in runs]
    return pd.DataFrame(
        {
            "Варіант": [r["label"] for r in records],
            "Pixel AUROC": [r["pixel_auroc"] for r in records],
            "Pixel AP": [r["pixel_ap"] for r in records],
            "Image AUROC": [r["image_auroc"] for r in records],
            "Розрив втрат": [r["loss_gap"] for r in records],
            "Епохи": [r["epochs"] for r in records],
            "Час, с": [r["train_seconds"] for r in records],
        }
    )


def study_metrics(runs: Sequence[FaeRun], key: str = "label") -> dict[str, dict]:
    """Розділ ``metrics.studies.<дослідження>``: варіант → метрики."""
    result = {}
    for run in runs:
        record = run_record(run)
        result[str(record[key])] = {
            name: record[name]
            for name in (
                "pixel_auroc",
                "pixel_ap",
                "image_auroc",
                "loss_gap",
                "epochs",
                "best_epoch",
                "train_seconds",
            )
        }
    return result


def history_table(runs: Sequence[FaeRun]) -> pd.DataFrame:
    """Історія навчання всіх запусків (ключ ``run_id``)."""
    frames = [run.training.history.assign(run_id=run.run_id) for run in runs]
    table = pd.concat(frames, ignore_index=True)
    return table.loc[:, ["run_id", *[c for c in table.columns if c != "run_id"]]]


def architecture_table(head: keras.Model) -> pd.DataFrame:
    """Шари головки FAE: ім'я, тип, вихідна форма, параметри."""
    core = head.get_layer("fae_core")
    rows = [
        (
            "normalization",
            "Normalization",
            f"(…, {head.input_shape[-1]})",
            0,
            "z-оцінка каналів (з train)",
        )
    ]
    purposes = {
        "input_noise": "шум входу (лише навчання)",
        "encoder_hidden": "кодувальник",
        "dropout": "Dropout (лише навчання)",
        "latent": "«пляшкове горло»",
        "decoder_hidden": "декодувальник",
        "reconstruction": "відновлення ознаки",
    }
    for layer in core.layers[1:]:
        rows.append(
            (
                layer.name,
                type(layer).__name__,
                f"(…, {layer.output.shape[-1]})",
                int(layer.count_params()),
                purposes.get(layer.name, ""),
            )
        )
    rows.append(
        (
            "reconstruction_error",
            "PatchReconstructionError",
            "(…)",
            0,
            "MSE по каналах = оцінка патча",
        )
    )
    return pd.DataFrame(rows, columns=["Шар", "Тип", "Вихід", "Параметри", "Призначення"])


def test_metrics_table(evaluations: Mapping[tuple[str, str], Evaluation]) -> pd.DataFrame:
    """Головна таблиця якості на test (7 колонок)."""
    rows = []
    for (method, category), evaluation in evaluations.items():
        metrics = evaluation.metrics
        rows.append(
            {
                "Категорія": category,
                "Метод": method,
                "Image AUROC": metrics.image_auroc,
                "Pixel AUROC": metrics.pixel_auroc,
                "Pixel AP": metrics.pixel_ap,
                "Pixel F1": metrics.pixel_f1,
                "Image F1": metrics.image_f1,
            }
        )
    return pd.DataFrame(rows).sort_values(["Категорія", "Метод"], kind="stable", ignore_index=True)


def _published_pair(
    lookup: Mapping[tuple[str, str, str], float], method: str, category: str
) -> str:
    """Рядок «Image / Pixel AUROC» з літератури (або «—»)."""
    values = [
        nan_if_missing(dict(lookup), (method, category, m)) for m in ("image_auroc", "pixel_auroc")
    ]
    return " / ".join("—" if np.isnan(v) else f"{v:.3f}".replace(".", ",") for v in values)


def sota_table(
    evaluations: Mapping[tuple[str, str], Evaluation],
    published: Mapping[tuple[str, str, str], float],
) -> pd.DataFrame:
    """Порівняння з SOTA (наші реімплементації + опубліковані Image / Pixel AUROC)."""
    rows = []
    for (method, category), evaluation in evaluations.items():
        metrics = evaluation.metrics
        rows.append(
            {
                "Категорія": category,
                "Метод": method,
                "Image AUROC": metrics.image_auroc,
                "Pixel AUROC": metrics.pixel_auroc,
                "Pixel AP": metrics.pixel_ap,
                "Pixel F1": metrics.pixel_f1,
                "Опубл. Image / Pixel AUROC": _published_pair(published, method, category),
            }
        )
    return pd.DataFrame(rows).sort_values(["Категорія", "Метод"], kind="stable", ignore_index=True)


def bootstrap_table(deltas: Mapping[tuple[str, str, str], BootstrapInterval]) -> pd.DataFrame:
    """Парні ДІ різниць (7 колонок)."""
    labels = {"pixel_auroc": "Pixel AUROC", "pixel_ap": "Pixel AP", "image_auroc": "Image AUROC"}
    rows = [
        {
            "Категорія": category,
            "Порівняння": comparison,
            "Метрика": labels[metric],
            "Δ": interval.delta,
            "Нижня межа": interval.low,
            "Верхня межа": interval.high,
            "ДІ без нуля": interval.excludes_zero,
        }
        for (category, comparison, metric), interval in deltas.items()
    ]
    return pd.DataFrame(rows)


def delta_metrics(deltas: Mapping[tuple[str, str, str], BootstrapInterval]) -> dict[str, dict]:
    """Розділ ``metrics.sota.delta.<категорія>.<порівняння>.<метрика>``."""
    result: dict[str, dict] = {}
    for (category, comparison, metric), interval in deltas.items():
        key = comparison.lower().replace(" − ", "_minus_").replace("-", "_")
        entry = asdict(interval) | {"excludes_zero": interval.excludes_zero}
        result.setdefault(category, {}).setdefault(key, {})[metric] = entry
    return result


def backbones_table(backbones: pd.DataFrame) -> pd.DataFrame:
    """Таблиця екстракторів для звіту."""
    return pd.DataFrame(
        {
            "Backbone": backbones["label"],
            "Шари (крок 8 + 16)": backbones["layers"],
            "Канали": backbones["channels"],
            "Параметри, млн": backbones["params"] / 1e6,
            "GFLOPs (256×256)": backbones["gflops"],
        }
    )


def feature_metrics(backbones: pd.DataFrame, config: Lab4Config) -> dict[str, dict]:
    """Розділи ``metrics.features`` і ``metrics.patches``."""
    grid = config.image_size // 8
    return {
        "features": {
            row["backbone"]: {
                "layers": row["layers"],
                "channels_raw": int(row["channels"]),
                "feature_dim": config.feature_dim,
                "grid": grid,
                "gflops": float(row["gflops"]),
                "params": int(row["params"]),
            }
            for _, row in backbones.iterrows()
        },
        "patches": {
            "image_size": config.image_size,
            "grid": grid,
            "stride_px": config.image_size // grid,
            "patch_sizes": list(config.patch_sizes),
            "tile_grids": [1, config.study_tile_grid],
        },
    }


def selected_metrics(selected: FaeRun, n_trainings: int) -> dict[str, object]:
    """Розділ ``metrics.studies.selected``: обрана конфігурація."""
    variant, settings = selected.variant, selected.variant.settings
    return {
        "backbone": str(variant.backbone),
        "patch_size": variant.patch_size,
        "tile_grid": variant.tile_grid,
        "latent": settings.latent,
        "dropout": settings.dropout,
        "l2": settings.l2,
        "input_noise": settings.input_noise,
        "val_pixel_auroc": selected.val_pixel_auroc,
        "n_trainings": n_trainings,
    }


def training_metrics(final_runs: Mapping[str, FaeRun]) -> dict[str, dict]:
    """Розділи ``metrics.training.<категорія>`` і ``metrics.model``."""
    any_run = next(iter(final_runs.values()))
    head = any_run.training.head
    return {
        "training": {
            category: {
                "epochs": len(run.training.history),
                "best_epoch": run.training.best_epoch,
                "early_stopped": len(run.training.history) < run.variant.settings.max_epochs,
                "train_mse": float(run.best_row()["train_mse"]),
                "val_mse": float(run.best_row()["val_mse"]),
                "val_pixel_auroc": run.val_pixel_auroc,
                "val_image_auroc": run.val_image_auroc,
                "seconds": run.training.seconds,
            }
            for category, run in final_runs.items()
        },
        "model": {
            "head_params": int(head.get_layer("fae_core").count_params()),
            **asdict(any_run.variant.settings),
        },
    }


def sota_fit_metrics(fits: Mapping[tuple[str, str], SotaFit]) -> dict[str, dict]:
    """Розділ ``metrics.sota_fit``: час навчання, пам'ять, FLOPs головок."""
    return {
        f"{method}.{category}": {
            "fit_seconds": fit.fit_seconds,
            "memory_mb": fit.memory_bytes / 2**20,
            "head_gflops": fit.head_flops / 1e9,
        }
        for (method, category), fit in fits.items()
    }


def config_metrics(config: Lab4Config, seed: int) -> dict[str, object]:
    """Розділ ``metrics.config``: усі параметри запуску (кортежі — списками для JSON)."""
    plain = asdict(replace(config, published_results=()))
    return {
        "seed": seed,
        **{
            key: list(value) if isinstance(value, tuple) else value
            for key, value in plain.items()
            if key not in {"fae", "published_results"}
        },
        "fae_base": asdict(config.fae),
    }


def with_backbone_latency(
    table: pd.DataFrame, runs: Sequence[FaeRun], latency: Mapping[str, float]
) -> pd.DataFrame:
    """Таблиця дослідження екстракторів із latency backbone замість часу навчання."""
    values = [latency[str(run.variant.backbone)] for run in runs]
    return table.drop(columns=["Час, с"]).assign(**{"Latency backbone, мс": values})
