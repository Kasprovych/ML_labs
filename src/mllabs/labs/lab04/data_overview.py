"""Стан даних: кількості по частинах, дисперсія тла, площа дефектів, перелік перетворень.

Тут обґрунтовується вибір датасету (R1): «висока дисперсія тла» і «незбалансованість»
не декларуються, а вимірюються.

* Дисперсія тла: (а) σ яскравості між нормальними зображеннями в кожному пікселі
  (наскільки сцена змінюється від кадру до кадру — положення об'єктів, освітлення);
  (б) σ яскравості на грубій сітці 16×16 усередині зображення (неоднорідність тла).
* Незбалансованість: частка площі дефекту на аномальному зображенні та частка аномальних
  пікселів серед усіх пікселів val+test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from mllabs.anomaly.images import ImageBatch, coarse_brightness_std, pixelwise_std
from mllabs.anomaly.splits import SplitPart
from mllabs.anomaly.workspace import VisaWorkspace
from mllabs.labs.lab04.plots import Example

_PARTS = (SplitPart.TRAIN, SplitPart.VAL_NORMAL, SplitPart.VAL, SplitPart.TEST)


def split_counts(splits: pd.DataFrame) -> pd.DataFrame:
    """Довга таблиця «категорія × частина → норма, аномалії» (``tables/splits.csv``)."""
    grouped = splits.groupby(["category", "part"], sort=False)["label"]
    table = grouped.agg(normal=lambda labels: int((labels == 0).sum()), anomalous="sum")
    table = table.reset_index().rename(
        columns={
            "category": "Категорія",
            "part": "Частина",
            "normal": "Норма",
            "anomalous": "Аномалії",
        }
    )
    order = {str(part): index for index, part in enumerate(_PARTS)}
    return table.sort_values(
        ["Категорія", "Частина"], key=lambda c: c.map(order) if c.name == "Частина" else c
    )


def data_overview(splits: pd.DataFrame) -> pd.DataFrame:
    """Широка таблиця кількостей зображень по категоріях (≤ 7 колонок)."""
    rows = []
    for category, frame in splits.groupby("category", sort=False):
        count = frame.groupby(["part", "label"]).size()
        rows.append(
            {
                "Категорія": category,
                "train (норма)": int(count.get((str(SplitPart.TRAIN), 0), 0)),
                "val_normal": int(count.get((str(SplitPart.VAL_NORMAL), 0), 0)),
                "val: норма": int(count.get((str(SplitPart.VAL), 0), 0)),
                "val: аномалії": int(count.get((str(SplitPart.VAL), 1), 0)),
                "test: норма": int(count.get((str(SplitPart.TEST), 0), 0)),
                "test: аномалії": int(count.get((str(SplitPart.TEST), 1), 0)),
            }
        )
    return pd.DataFrame(rows)


def background_variance(
    samples: Mapping[str, np.ndarray], selected: Sequence[str], cells: int
) -> pd.DataFrame:
    """Дві міри неоднорідності тла для кожної категорії.

    Args:
        samples: Категорія → ``(n, S, S, 3)`` нормальні зображення.
        selected: Категорії, обрані для роботи (позначаються в таблиці).
        cells: Сітка для σ яскравості всередині зображення.
    """
    rows = [
        {
            "Категорія": category,
            "σ між кадрами": pixelwise_std(images),
            "σ яскравості в кадрі": float(coarse_brightness_std(images, cells).mean()),
            "Обрана": category in selected,
        }
        for category, images in samples.items()
    ]
    table = pd.DataFrame(rows).sort_values("σ між кадрами", ascending=False, ignore_index=True)
    table.insert(1, "Ранг", np.arange(1, len(table) + 1))
    return table


def defect_areas(category: str, rows: pd.DataFrame, batch: ImageBatch) -> pd.DataFrame:
    """Частка площі дефекту на кожному аномальному зображенні (на сітці оцінювання)."""
    anomalous = (rows["label"] == 1).to_numpy()
    masks = np.asarray(batch.masks[anomalous])
    return pd.DataFrame(
        {
            "category": category,
            "image": rows.loc[anomalous, "image"].to_numpy(),
            "defect_types": rows.loc[anomalous, "defect_types"].to_numpy(),
            "area_share": masks.reshape(len(masks), -1).mean(axis=1),
        }
    )


def defect_area_summary(areas: pd.DataFrame, pixel_share: Mapping[str, float]) -> pd.DataFrame:
    """Зведення площ дефектів по категоріях (у відсотках площі зображення)."""
    rows = []
    for category, frame in areas.groupby("category", sort=False):
        share = frame["area_share"] * 100
        rows.append(
            {
                "Категорія": category,
                "Аномальних зобр.": len(frame),
                "Медіана площі, %": share.median(),
                "p10, %": share.quantile(0.1),
                "p90, %": share.quantile(0.9),
                "Аном. пікселів (val+test), %": pixel_share[category] * 100,
                "Порожніх масок": int((frame["area_share"] == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def transformations(
    image_size: int,
    grid: int,
    feature_dim: int,
    patch_sizes: Sequence[int],
    tile_grid: int,
    sigma: float,
) -> pd.DataFrame:
    """Перелік перетворень даних із параметрами і тим, на чому вони «вчаться»."""
    rows = [
        ("Сирі дані", "JPG ~1500×1000, маски PNG (індекси дефектів)", "—", "—"),
        (
            "Розбиття",
            "1cls → train / val_normal / val / test",
            "85/15 % train; 50/50 % test",
            "seed",
        ),
        (
            "Зменшення",
            f"усе зображення → {image_size}×{image_size}, білінійно",
            f"{image_size}×{image_size}×3 uint8",
            "—",
        ),
        ("Маски", "найближчий сусід, бінаризація > 0", f"{image_size}×{image_size} bool", "—"),
        (
            "Тайли",
            f"{tile_grid}×{tile_grid} тайли по {image_size} пікс. (дослідження)",
            f"{tile_grid * grid}×{tile_grid * grid}×{feature_dim}",
            "—",
        ),
        (
            "Backbone",
            "2 проміжні шари (крок 8 і 16), апсемплінг, конкатенація",
            f"{grid}×{grid}×C",
            "ImageNet (заморожено)",
        ),
        (
            "Зведення каналів",
            f"усереднення сусідніх каналів до D = {feature_dim}",
            f"{grid}×{grid}×{feature_dim}",
            "фіксоване",
        ),
        (
            "Агрегація патчів",
            f"середнє вікна p×p, p ∈ {tuple(patch_sizes)}",
            f"{grid}×{grid}×{feature_dim}",
            "фіксоване",
        ),
        ("Нормалізація", "z-оцінка кожного каналу (у головці FAE)", "те саме", "train"),
        ("Головка", "FAE / PaDiM / PatchCore", f"{grid}×{grid} оцінок", "train (лише норма)"),
        (
            "Пост-обробка",
            f"білінійно до {image_size}, гаус σ = {sigma:g}",
            f"{image_size}×{image_size}",
            "—",
        ),
        ("Пороги", "max F1 (зображення і піксель)", "2 числа", "val (розмічений)"),
    ]
    return pd.DataFrame(rows, columns=["Етап", "Перетворення", "Результат", "Вчиться на"])


def anomalous_pixel_share(rows: pd.DataFrame, batch: ImageBatch) -> float:
    """Частка аномальних пікселів серед усіх пікселів val+test."""
    evaluated = rows["part"].isin([str(SplitPart.VAL), str(SplitPart.TEST)]).to_numpy()
    return float(np.asarray(batch.masks[evaluated]).mean())


def example_indices(rows: pd.DataFrame, per_class: int, seed: int) -> list[int]:
    """Позиції ``per_class`` нормальних і ``per_class`` аномальних зображень test (за seed)."""
    generator = np.random.default_rng(seed)
    test = rows["part"] == str(SplitPart.TEST)
    picks = []
    for label in (0, 1):
        candidates = np.flatnonzero((test & (rows["label"] == label)).to_numpy())
        picks.extend(sorted(generator.choice(candidates, per_class, replace=False).tolist()))
    return picks


def collect_defect_areas(workspace: VisaWorkspace, categories: Sequence[str]) -> pd.DataFrame:
    """Площі дефектів усіх аномальних зображень обраних категорій."""
    frames = [defect_areas(c, workspace.category_rows(c), workspace.images(c)) for c in categories]
    return pd.concat(frames, ignore_index=True)


def collect_pixel_shares(workspace: VisaWorkspace, categories: Sequence[str]) -> dict[str, float]:
    """Частка аномальних пікселів val+test кожної категорії."""
    return {
        c: anomalous_pixel_share(workspace.category_rows(c), workspace.images(c))
        for c in categories
    }


def sample_examples(
    workspace: VisaWorkspace, categories: Sequence[str], seed: int
) -> dict[str, list[Example]]:
    """По 2 нормальних і 2 аномальних зображення test кожної категорії (для сітки)."""
    examples = {}
    for category in categories:
        rows, batch = workspace.category_rows(category), workspace.images(category)
        examples[category] = [
            Example(
                np.asarray(batch.images[i]),
                batch.masks[i],
                "норма"
                if rows.loc[i, "label"] == 0
                else rows.loc[i, "defect_types"].replace(",", ", "),
            )
            for i in example_indices(rows, 2, seed)
        ]
    return examples


def data_metrics(
    splits: pd.DataFrame,
    background: pd.DataFrame,
    areas: pd.DataFrame,
    pixel_share: Mapping[str, float],
    categories: Sequence[str],
) -> dict[str, dict]:
    """Розділ ``metrics.data.<категорія>``: кількості, незбалансованість, тло."""
    overview = data_overview(splits).set_index("Категорія")
    sigma = background.set_index("Категорія")
    result = {}
    for category in categories:
        row = overview.loc[category]
        share = areas.loc[areas["category"] == category, "area_share"]
        result[category] = {
            "n_train": int(row["train (норма)"]),
            "n_val_normal": int(row["val_normal"]),
            "n_val_labelled_normal": int(row["val: норма"]),
            "n_val_labelled_anomalous": int(row["val: аномалії"]),
            "n_test_normal": int(row["test: норма"]),
            "n_test_anomalous": int(row["test: аномалії"]),
            "anomalous_pixel_share": pixel_share[category],
            "defect_area_median": float(share.median()),
            "background_std": float(sigma.loc[category, "σ між кадрами"]),
            "brightness_std": float(sigma.loc[category, "σ яскравості в кадрі"]),
            "background_rank": int(sigma.loc[category, "Ранг"]),
            "masks_empty_after_resize": int((share == 0).sum()),
        }
    return result
