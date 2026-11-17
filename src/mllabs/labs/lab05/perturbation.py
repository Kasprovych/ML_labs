"""Perturbation analysis (R3, D6): чи справді «важливі» патчі визначають рішення моделі.

Ідея (Samek et al., 2017): вилучаємо патчі в порядку важливості за поясненням і дивимося,
як змінюється відгук моделі. Добре пояснення — оцінка падає ШВИДКО, якщо вилучати
найважливіші першими (**MoRF**, most relevant first), і ПОВІЛЬНО, якщо найменш важливі
(**LeRF**). Метрика **ABPC** (area between perturbation curves) = середнє ``f_LeRF − f_MoRF``.

Головна проблема для детектора аномалій: будь-яке втручання в зображення (сірий квадрат,
шум, розмиття) саме є аномалією. Тому:

* основний режим — «вилучення» як **заміна патча вмістом схожої норми** з train (та сама
  позиція) — єдиний спосіб лишити зображення близько до розподілу навчання;
* режим «зашумлення» (гаусів шум у патчі) — бо його прямо вимагає завдання, але його криві
  тлумачаться лише ВІДНОСНО випадкового порядку (артефакт однаковий для всіх порядків);
* нижня межа — випадкові порядки, верхня — «маска-оракул» (патчі за покриттям маски).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from mllabs.labs.lab05.inference import DetectorRunner
from mllabs.labs.lab05.references import NormalReferenceFinder

#: Напрямки вилучення.
MORF, LERF = "morf", "lerf"
#: Назва «методу» випадкових порядків.
RANDOM = "random"


class SubstituteSource(Protocol):
    """Звідки береться вміст, яким замінюються вилучені патчі."""

    def substitute(self, image: np.ndarray, key: int) -> np.ndarray:
        """Повне зображення-замінник ``(S, S, 3)`` ``uint8`` для ``image``."""
        ...


class ReferenceReplacement:
    """«Вилучення»: вміст найближчого нормального зображення train у тій самій позиції.

    Args:
        finder: Пошук норм.
    """

    def __init__(self, finder: NormalReferenceFinder) -> None:
        self._finder = finder

    def substitute(self, image: np.ndarray, key: int) -> np.ndarray:
        """Еталонна норма для ``image``."""
        return self._finder.reference(image)


class NoiseReplacement:
    """«Зашумлення»: те саме зображення з гаусовим шумом σ (одне шумове поле на зображення).

    Args:
        sigma: σ шуму в одиницях яскравості.
        seed: Зерно; для зображення ``key`` використовується ``seed + key``.
    """

    def __init__(self, sigma: float, seed: int) -> None:
        self._sigma = sigma
        self._seed = seed

    def substitute(self, image: np.ndarray, key: int) -> np.ndarray:
        """Зашумлена копія."""
        generator = np.random.default_rng(self._seed + key)
        noisy = np.asarray(image, np.float32) + generator.normal(0, self._sigma, image.shape)
        return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def patch_importance(explanation: np.ndarray, grid: int) -> np.ndarray:
    """Середнє карти в кожному патчі сітки ``grid×grid``, сплющене в ``(grid²,)``."""
    block = explanation.shape[0] // grid
    return explanation.reshape(grid, block, grid, block).mean(axis=(1, 3)).reshape(-1)


def ranking(importance: np.ndarray, direction: str) -> np.ndarray:
    """Порядок патчів: MoRF — за спаданням важливості, LeRF — за зростанням."""
    sign = -1.0 if direction == MORF else 1.0
    return np.argsort(sign * importance, kind="stable")


def perturbed(
    image: np.ndarray, substitute: np.ndarray, patches: np.ndarray, grid: int
) -> np.ndarray:
    """Копія ``image``, у якій патчі ``patches`` (номери на сітці) взято з ``substitute``."""
    selected = np.zeros(grid * grid, bool)
    selected[patches] = True
    block = image.shape[0] // grid
    # np.repeat двічі «розтягує» сітку патчів до пікселів: (grid, grid) → (S, S).
    pixels = np.repeat(np.repeat(selected.reshape(grid, grid), block, 0), block, 1)
    return np.where(pixels[..., None], substitute, image)


@dataclass(frozen=True)
class CurveSpec:
    """Одна крива: метод, напрямок, порядок патчів.

    Attributes:
        method: Назва пояснення (або ``random``).
        direction: ``morf`` / ``lerf``.
        repeat: Номер випадкового порядку (0 для пояснень).
        order: Порядок патчів.
    """

    method: str
    direction: str
    repeat: int
    order: np.ndarray


class PerturbationStudy:
    """Криві MoRF/LeRF для одного зображення.

    Args:
        runner: Наскрізний прогін детектора.
        steps: Кількості вилучених патчів (перший — 0).
        grid: Сторона сітки патчів.
        random_orders: Кількість випадкових порядків.
        seed: Зерно випадкових порядків.
    """

    def __init__(
        self,
        runner: DetectorRunner,
        steps: Sequence[int],
        grid: int,
        random_orders: int,
        seed: int,
    ) -> None:
        if steps[0] != 0:
            raise ValueError("Перший стан perturbation має бути без вилучень (k = 0)")
        self._runner = runner
        self._steps = tuple(steps)
        self._grid = grid
        self._random_orders = random_orders
        self._seed = seed

    def curve_specs(self, importances: Mapping[str, np.ndarray], key: int) -> list[CurveSpec]:
        """MoRF і LeRF кожного пояснення + випадкові порядки (відтворювані за seed і key)."""
        specs = [
            CurveSpec(method, direction, 0, ranking(importance, direction))
            for method, importance in importances.items()
            for direction in (MORF, LERF)
        ]
        generator = np.random.default_rng((self._seed, key))
        specs += [
            CurveSpec(RANDOM, MORF, repeat, generator.permutation(self._grid**2))
            for repeat in range(self._random_orders)
        ]
        return specs

    def run(
        self,
        image: np.ndarray,
        substitute: np.ndarray,
        importances: Mapping[str, np.ndarray],
        mask_patches: np.ndarray | None,
        key: int,
    ) -> pd.DataFrame:
        """Відгуки моделі вздовж усіх кривих одного зображення.

        Args:
            image: ``(S, S, 3)`` ``uint8``.
            substitute: Зображення-замінник.
            importances: Метод → ``(grid²,)`` важливість патчів.
            mask_patches: ``(grid²,)`` булева маска «патч у зоні дефекту» (``None`` — норма).
            key: Номер зображення (для відтворюваних випадкових порядків).

        Returns:
            Рядки ``method, direction, repeat, k, score_rel, error_rel, mask_error_rel`` —
            відгуки, нормовані на значення без вилучень.
        """
        specs = self.curve_specs(importances, key)
        states = [image] + [
            perturbed(image, substitute, spec.order[:k], self._grid)
            for spec in specs
            for k in self._steps[1:]
        ]
        outputs = self._runner.run(np.stack(states), keep_maps=False)
        patch = outputs.patch_scores.reshape(len(states), -1)
        score = outputs.scores / outputs.scores[0]
        error = patch.mean(axis=1) / patch[0].mean()
        if mask_patches is not None and mask_patches.any():
            zone = patch[:, mask_patches].mean(axis=1)
            mask_error = zone / zone[0]
        else:
            mask_error = np.full(len(states), np.nan)
        rows = []
        per_curve = len(self._steps) - 1
        for number, spec in enumerate(specs):
            for position, k in enumerate(self._steps):
                state = 0 if k == 0 else 1 + number * per_curve + position - 1
                rows.append(
                    {
                        "method": spec.method,
                        "direction": spec.direction,
                        "repeat": spec.repeat,
                        "k": k,
                        "score_rel": float(score[state]),
                        "error_rel": float(error[state]),
                        "mask_error_rel": float(mask_error[state]),
                    }
                )
        return pd.DataFrame(rows)


#: Відгуки моделі, для яких рахуються метрики кривих.
RESPONSES = ("score_rel", "error_rel", "mask_error_rel")


def curve_metrics(curves: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """AUC MoRF, AUC LeRF, ABPC і Δ проти випадкового для кожного зображення.

    AUC — середнє нормованого відгуку по станах (рівномірно за номером кроку, як у Samek
    et al.); ABPC = AUC LeRF − AUC MoRF; ``delta_vs_random`` = AUC випадкових − AUC MoRF
    (> 0 — пояснення краще за випадковий порядок).

    Args:
        curves: Рядки з :meth:`PerturbationStudy.run` + ідентифікатори ``keys``.
        keys: Колонки, що ідентифікують зображення (``category``, ``image``, ``mode`` …).
    """
    tables = []
    for response in RESPONSES:
        # Площа під кривою = середній відгук по станах; спершу для кожної кривої окремо,
        # потім середнє по випадкових повторах (для пояснень повтор лише один).
        per_curve = curves.groupby([*keys, "method", "direction", "repeat"], as_index=False)[
            response
        ].mean()
        per_method = per_curve.groupby([*keys, "method", "direction"], as_index=False)[
            response
        ].mean()
        wide = per_method.pivot_table(
            index=[*keys, "method"], columns="direction", values=response, dropna=False
        ).reset_index()
        random = wide.loc[wide["method"] == RANDOM, [*keys, MORF]].rename(
            columns={MORF: "auc_random"}
        )
        explained = wide[wide["method"] != RANDOM].merge(random, on=list(keys))
        tables.append(
            explained.assign(
                response=response,
                auc_morf=explained[MORF],
                auc_lerf=explained[LERF],
                abpc=explained[LERF] - explained[MORF],
                delta_vs_random=explained["auc_random"] - explained[MORF],
            ).drop(columns=[MORF, LERF])
        )
    return pd.concat(tables, ignore_index=True)
