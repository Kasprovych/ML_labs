"""«Робочий стіл» категорії: детектор ЛР4, його прогони й інструменти пояснень разом.

Крок 1 ЛР5 будує :class:`CategoryBench` для кожної категорії один раз; наступні кроки
беруть із нього готові прогони (val/test/train) і об'єкти (IG, Grad-CAM, еталонні норми),
не перераховуючи їх.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mllabs.anomaly.splits import SplitPart
from mllabs.labs.lab05.attribution import BaselineFactory, GradCam, IgSettings, IntegratedGradients
from mllabs.labs.lab05.config import Lab5Config
from mllabs.labs.lab05.handoff import CategoryAssets, Lab4Handoff, check_scores
from mllabs.labs.lab05.inference import CachedScorer, DetectorRunner, PartOutputs
from mllabs.labs.lab05.references import NormalReferenceFinder

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandoffCheck:
    """Звірка з ЛР4.

    Attributes:
        n_val: Зображень val.
        n_test: Зображень test.
        max_score_diff: max |Δ| оцінок test (кеш ознак) проти ``test_predictions.csv``.
        end_to_end_diff: max |Δ| наскрізної моделі проти кешу на кількох зображеннях.
    """

    n_val: int
    n_test: int
    max_score_diff: float
    end_to_end_diff: float


@dataclass(frozen=True)
class CategoryBench:
    """Усе підготовлене для однієї категорії.

    Attributes:
        assets: Детектор і дані ЛР4.
        scorer: Головка на кешованих ознаках.
        runner: Наскрізний прогін.
        val: Прогін розміченого val (з картами).
        test: Прогін test (з картами).
        train: Прогін train (лише оцінки й карти патчів).
        references: Пошук нормальних train-зображень.
        baselines: Фабрика баз IG.
        cam: Grad-CAM / HiResCAM.
        check: Звірка з ЛР4.
    """

    assets: CategoryAssets
    scorer: CachedScorer
    runner: DetectorRunner
    val: PartOutputs
    test: PartOutputs
    train: PartOutputs
    references: NormalReferenceFinder
    baselines: BaselineFactory
    cam: GradCam
    check: HandoffCheck

    @property
    def category(self) -> str:
        """Категорія."""
        return self.assets.category

    @property
    def val_rows(self) -> np.ndarray:
        """Рядки val."""
        return self.assets.indices(SplitPart.VAL)

    @property
    def test_rows(self) -> np.ndarray:
        """Рядки test."""
        return self.assets.indices(SplitPart.TEST)

    def integrated_gradients(
        self, steps: int, max_steps: int, config: Lab5Config
    ) -> IntegratedGradients:
        """IG відносно ``image_score`` навченого детектора."""
        detector = self.assets.detector
        settings = IgSettings(
            steps=steps,
            max_steps=max_steps,
            alpha_batch=config.ig_alpha_batch,
            tolerance=config.completeness_tolerance,
            scale=self.assets.config.threshold_image,
            blur_sigma=config.attribution_blur_sigma,
            output_size=self.assets.config.eval_size,
        )
        return IntegratedGradients(
            lambda batch: detector(batch, training=False)["image_score"], settings
        )


class BenchBuilder:
    """Будує :class:`CategoryBench` і звіряє оцінки з ЛР4.

    Args:
        handoff: Доступ до артефактів ЛР4.
        config: Параметри ЛР5.
    """

    def __init__(self, handoff: Lab4Handoff, config: Lab5Config) -> None:
        self._handoff = handoff
        self._config = config

    def build(self, category: str) -> CategoryBench:
        """Завантажує детектор, рахує val/test/train і перевіряє узгодженість з ЛР4."""
        config = self._config
        assets = self._handoff.load(category)
        detector_config = assets.config
        scorer = CachedScorer(
            assets.head, detector_config.eval_size, detector_config.blur_sigma, config.head_batch
        )
        runner = DetectorRunner(assets.detector, config.inference_batch)
        val_rows = assets.indices(SplitPart.VAL)
        test_rows = assets.indices(SplitPart.TEST)
        train_rows = assets.indices(SplitPart.TRAIN)
        val = scorer.score(assets.features, val_rows)
        test = scorer.score(assets.features, test_rows)
        train = scorer.score(assets.features, train_rows, keep_maps=False)
        difference = check_scores(category, test.scores, assets.predictions, config.score_tolerance)
        probe = test_rows[: config.handoff_end_to_end_images]
        end_to_end = runner.run(np.asarray(assets.images[probe]), keep_maps=False)
        end_to_end_diff = float(np.max(np.abs(end_to_end.scores - test.scores[: len(probe)])))
        _logger.info(
            "%s: наскрізна модель проти кешу ознак: max |Δ| = %.1e", category, end_to_end_diff
        )
        references = NormalReferenceFinder(
            np.asarray(assets.images[train_rows]),
            train.scores,
            config.reference_candidates,
            config.mean_image_batch,
        )
        baselines = BaselineFactory(
            references,
            references.mean_image(),
            config.baseline_blur_sigma,
            config.expected_gradients_k,
        )
        cam = GradCam(assets.detector, detector_config.eval_size, config.inference_batch)
        check = HandoffCheck(len(val_rows), len(test_rows), difference, end_to_end_diff)
        return CategoryBench(
            assets, scorer, runner, val, test, train, references, baselines, cam, check
        )
