"""Тести перевірок ЛР5: локалізація, perturbation, подібність карт, рандомізація головки."""

import keras
import numpy as np
import pandas as pd
import pytest

from mllabs.anomaly.fae import FaeSettings, FeatureAutoencoderBuilder
from mllabs.labs.lab05.inference import PartOutputs
from mllabs.labs.lab05.localization import LocalizationEvaluator, area_terciles, bootstrap_mean
from mllabs.labs.lab05.perturbation import (
    LERF,
    MORF,
    RANDOM,
    NoiseReplacement,
    PerturbationStudy,
    curve_metrics,
    patch_importance,
    perturbed,
)
from mllabs.labs.lab05.sanity import add_noise, map_similarity, randomized_head, reinitialize

_SIDE, _GRID = 16, 4


def _mask() -> np.ndarray:
    mask = np.zeros((_SIDE, _SIDE), bool)
    mask[4:8, 8:12] = True
    return mask


def test_perfect_map_gets_perfect_localization_scores():
    mask = _mask()
    evaluator = LocalizationEvaluator(pointing_tolerance_px=0, correlation_grid=4)

    metrics = evaluator.evaluate(mask.astype(float), mask, mask.astype(float))

    assert metrics.pixel_auroc == pytest.approx(1.0)
    assert metrics.pointing == 1.0
    assert metrics.iou_topk == pytest.approx(1.0)
    assert metrics.mass_ratio == pytest.approx(1 / mask.mean())
    assert metrics.spearman_model == pytest.approx(1.0)


def test_random_map_has_chance_level_auroc():
    generator = np.random.default_rng(0)
    mask = np.zeros((64, 64), bool)
    mask[10:40, 10:40] = True
    evaluator = LocalizationEvaluator(pointing_tolerance_px=0, correlation_grid=8)

    metrics = evaluator.evaluate(generator.random((64, 64)), mask, generator.random((64, 64)))

    assert metrics.pixel_auroc == pytest.approx(0.5, abs=0.05)
    assert metrics.mass_ratio == pytest.approx(1.0, abs=0.1)


def test_bootstrap_mean_brackets_the_mean_and_is_reproducible():
    values = np.arange(20, dtype=float)

    first = bootstrap_mean(values, 200, 0.95, seed=1)
    second = bootstrap_mean(values, 200, 0.95, seed=1)

    assert first == second
    assert first.value == pytest.approx(9.5)
    assert first.low < first.value < first.high


def test_area_terciles_split_into_three_groups():
    strata = area_terciles(np.arange(9, dtype=float), 3)

    assert list(strata) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


class _SumRunner:
    """Підмінений детектор: оцінка = середня яскравість зони дефекту (патчі з маски)."""

    def __init__(self, zone: np.ndarray) -> None:
        self._zone = zone

    def run(self, images: np.ndarray, keep_maps: bool = True) -> PartOutputs:
        block = _SIDE // _GRID
        patches = (
            images.astype(float).mean(axis=-1).reshape(len(images), _GRID, block, _GRID, block)
        )
        patch_scores = patches.mean(axis=(2, 4))
        scores = patch_scores.reshape(len(images), -1)[:, self._zone].mean(axis=1)
        return PartOutputs(patch_scores, None, scores)


def _perturbation_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((_SIDE, _SIDE, 3), 10, np.uint8)
    mask = _mask()
    image[mask] = 250
    zone = patch_importance(mask.astype(float), _GRID) > 0
    return image, mask, zone


def test_oracle_order_gives_monotone_morf_and_beats_random():
    image, mask, zone = _perturbation_case()
    substitute = np.full_like(image, 10)
    study = PerturbationStudy(_SumRunner(zone), (0, 1, 2, 4, 8), _GRID, random_orders=3, seed=0)

    curves = study.run(
        image, substitute, {"oracle": patch_importance(mask.astype(float), _GRID)}, zone, key=0
    )
    morf = curves[(curves["method"] == "oracle") & (curves["direction"] == MORF)]
    metrics = curve_metrics(curves.assign(image="x"), ["image"])
    score = metrics[metrics["response"] == "score_rel"].iloc[0]

    assert np.all(np.diff(morf["score_rel"].to_numpy()) <= 1e-12)
    assert morf["score_rel"].iloc[-1] == pytest.approx(10 / 250)
    assert score["abpc"] > 0
    assert score["delta_vs_random"] > 0
    assert set(curves["method"]) == {"oracle", RANDOM}
    assert set(curves["direction"]) == {MORF, LERF}


def test_random_orders_are_reproducible_for_same_seed_and_key():
    study = PerturbationStudy(_SumRunner(np.ones(_GRID**2, bool)), (0, 1), _GRID, 2, seed=5)

    first = [spec.order for spec in study.curve_specs({}, key=3)]
    second = [spec.order for spec in study.curve_specs({}, key=3)]
    other = [spec.order for spec in study.curve_specs({}, key=4)]

    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert not np.array_equal(first[0], other[0])


def test_perturbed_replaces_only_selected_patches():
    image = np.zeros((_SIDE, _SIDE, 3), np.uint8)
    substitute = np.full_like(image, 9)

    result = perturbed(image, substitute, np.array([0, 5]), _GRID)

    assert result[:4, :4].min() == 9  # патч 0
    assert result[4:8, 4:8].min() == 9  # патч 5 = рядок 1, стовпець 1
    assert result.sum() == 9 * 2 * 16 * 3


def test_noise_replacement_is_deterministic_per_key():
    image = np.full((_SIDE, _SIDE, 3), 128, np.uint8)
    source = NoiseReplacement(32.0, seed=1)

    assert np.array_equal(source.substitute(image, 2), source.substitute(image, 2))
    assert not np.array_equal(source.substitute(image, 2), source.substitute(image, 3))


def test_map_similarity_is_one_for_identical_maps_and_low_for_unrelated():
    generator = np.random.default_rng(0)
    first = generator.random((32, 32))

    rho, ssim = map_similarity(first, first, 8)
    rho_other, _ = map_similarity(first, generator.random((32, 32)), 8)

    assert rho == pytest.approx(1.0)
    assert ssim == pytest.approx(1.0, abs=1e-4)
    assert abs(rho_other) < 0.3


def test_add_noise_keeps_dtype_and_range():
    images = np.full((2, 4, 4, 3), 255, np.uint8)

    noisy = add_noise(images, 5.0, seed=0)

    assert noisy.dtype == np.uint8
    assert noisy.max() <= 255


def _tiny_head():
    keras.utils.set_random_seed(0)
    builder = FeatureAutoencoderBuilder(FaeSettings(feature_dim=4, hidden=3, latent=2))
    mean, variance = np.arange(4, dtype=np.float32), np.full(4, 2.0, np.float32)
    return builder.build_head(builder.build_core(), mean, variance)


def test_randomized_head_keeps_or_resets_normalization_and_is_deterministic():
    head = _tiny_head()
    vectors = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)

    kept = randomized_head(head, seed=3)
    again = randomized_head(head, seed=3)
    reset = randomized_head(head, seed=3, reset_normalization=True)

    original = head.get_layer("normalization")(vectors).numpy()
    np.testing.assert_allclose(kept.get_layer("normalization")(vectors).numpy(), original)
    np.testing.assert_allclose(
        reset.get_layer("normalization")(vectors).numpy().reshape(3, 4), vectors, rtol=1e-5
    )
    kernel = head.get_layer("fae_core").get_layer("latent").kernel.numpy()
    new_kernel = kept.get_layer("fae_core").get_layer("latent").kernel.numpy()
    assert not np.allclose(kernel, new_kernel)
    np.testing.assert_allclose(
        new_kernel, again.get_layer("fae_core").get_layer("latent").kernel.numpy()
    )


def test_reinitialize_sets_batchnorm_to_identity_deterministically():
    inputs = keras.Input((8, 8, 3))
    hidden = keras.layers.Conv2D(4, 3)(inputs)
    hidden = keras.layers.DepthwiseConv2D(3)(hidden)
    outputs = keras.layers.BatchNormalization()(hidden)
    first, second = keras.Model(inputs, outputs), keras.Model(inputs, outputs)

    reinitialize(first, np.random.default_rng(1))
    kernel = first.layers[1].get_weights()[0].copy()
    reinitialize(second, np.random.default_rng(1))

    gamma, beta, mean, variance = first.layers[-1].get_weights()
    assert gamma.min() == 1 and beta.max() == 0 and mean.max() == 0 and variance.min() == 1
    np.testing.assert_allclose(second.layers[1].get_weights()[0], kernel)


def test_curve_metrics_handles_missing_mask_response():
    rows = []
    for method, direction in (("a", MORF), ("a", LERF), (RANDOM, MORF)):
        for k, value in ((0, 1.0), (1, 0.5)):
            rows.append(
                {
                    "method": method,
                    "direction": direction,
                    "repeat": 0,
                    "k": k,
                    "score_rel": value if direction == MORF else 1.0,
                    "error_rel": 1.0,
                    "mask_error_rel": np.nan,
                    "image": "x",
                }
            )
    metrics = curve_metrics(pd.DataFrame(rows), ["image"])

    score = metrics[metrics["response"] == "score_rel"].iloc[0]
    assert score["abpc"] == pytest.approx(0.25)
    assert score["delta_vs_random"] == pytest.approx(0.0)
