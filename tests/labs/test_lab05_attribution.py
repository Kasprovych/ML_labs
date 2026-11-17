"""Тести пояснень ЛР5: Integrated Gradients, Grad-CAM/HiResCAM, бази, еталонні норми."""

import keras
import numpy as np
import pytest
import tensorflow as tf

from mllabs.labs.lab05.attribution import (
    BaselineFactory,
    CamGraphError,
    GradCam,
    IgSettings,
    IntegratedGradients,
    cam_key,
    trapezoid_weights,
)
from mllabs.labs.lab05.config import BaselineKind
from mllabs.labs.lab05.references import NormalReferenceFinder

_SIZE = 8


def _settings(steps: int = 64, max_steps: int = 64, tolerance: float = 1e-3) -> IgSettings:
    return IgSettings(
        steps=steps,
        max_steps=max_steps,
        alpha_batch=33,
        tolerance=tolerance,
        scale=1.0,
        blur_sigma=0.0,
        output_size=_SIZE,
    )


def _quadratic(weights: np.ndarray):
    constant = tf.constant(weights, tf.float32)
    return lambda images: tf.reduce_sum(constant * tf.square(images), axis=(1, 2, 3))


def test_trapezoid_weights_sum_to_one_with_half_ends():
    weights = trapezoid_weights(4)

    np.testing.assert_allclose(weights, [0.125, 0.25, 0.25, 0.25, 0.125])
    assert weights.sum() == pytest.approx(1.0)


def test_ig_satisfies_completeness_for_quadratic_function():
    generator = np.random.default_rng(0)
    weights = generator.uniform(0.5, 1.5, (_SIZE, _SIZE, 3))
    image = generator.uniform(0, 2, (_SIZE, _SIZE, 3)).astype(np.float32)
    baseline = generator.uniform(0, 1, (1, _SIZE, _SIZE, 3)).astype(np.float32)
    ig = IntegratedGradients(_quadratic(weights), _settings())

    result = ig.explain(image, baseline)

    expected = float((weights * image**2).sum() - (weights * baseline[0] ** 2).sum())
    assert result.delta == pytest.approx(expected, rel=1e-4)
    assert result.completeness_error < 1e-3
    assert result.map.shape == (_SIZE, _SIZE)
    assert result.steps == 64


def test_ig_attribution_is_zero_when_input_equals_baseline():
    image = np.full((_SIZE, _SIZE, 3), 3.0, np.float32)
    ig = IntegratedGradients(_quadratic(np.ones((_SIZE, _SIZE, 3))), _settings())

    result = ig.explain(image, image[None])

    assert np.abs(result.signed).max() == 0.0
    assert result.delta == 0.0


def test_ig_escalates_steps_when_completeness_fails():
    # Сильно нелінійна функція: 8 кроків трапецій дають помітну похибку інтегрування.
    function = lambda images: tf.reduce_sum(tf.sin(3.0 * images), axis=(1, 2, 3))  # noqa: E731
    image = np.full((_SIZE, _SIZE, 3), 2.0, np.float32)
    baseline = np.zeros((1, _SIZE, _SIZE, 3), np.float32)
    ig = IntegratedGradients(function, _settings(steps=4, max_steps=64, tolerance=1e-4))

    result = ig.explain(image, baseline)

    assert result.steps > 4
    assert result.steps <= 64


def test_expected_gradients_average_several_baselines():
    generator = np.random.default_rng(1)
    weights = np.ones((_SIZE, _SIZE, 3))
    image = generator.uniform(0, 2, (_SIZE, _SIZE, 3)).astype(np.float32)
    baselines = generator.uniform(0, 1, (4, _SIZE, _SIZE, 3)).astype(np.float32)
    ig = IntegratedGradients(_quadratic(weights), _settings(steps=64))

    result = ig.explain(image, baselines)

    expected = float((image**2).sum() - (baselines**2).sum(axis=(1, 2, 3)).mean())
    assert result.delta == pytest.approx(expected, rel=1e-4)
    assert result.completeness_error < 1e-3


def _cam_model(position: tuple[int, int], channel: int, use_deep: bool = True) -> keras.Model:
    """Модель, ціль якої з шару «shallow» залежить від ОДНОГО каналу в ОДНІЙ позиції.

    Шар «deep» рахується з входу окремо (не з «shallow») і додає до цілі малий внесок —
    тоді градієнт за «shallow» не «протікає» через «deep».
    """
    inputs = keras.Input((_SIZE, _SIZE, 3))
    shallow = keras.layers.Conv2D(4, 1, activation="relu", name="shallow")(inputs)
    deep = keras.layers.AveragePooling2D(2)(keras.layers.Conv2D(4, 1, name="deep_conv")(inputs))
    row, column = position
    score = keras.layers.Lambda(lambda values: values[:, row, column, channel])(shallow)
    if use_deep:
        deep_term = keras.layers.Lambda(lambda values: 1e-3 * keras.ops.mean(values, (1, 2, 3)))
        score = keras.layers.Add()([score, deep_term(deep)])
    return keras.Model(
        inputs, {"image_score": score, "backbone_shallow": shallow, "backbone_deep": deep}
    )


def test_hirescam_is_nonzero_only_where_target_depends_on_activation():
    keras.utils.set_random_seed(0)
    model = _cam_model((2, 5), 1)
    images = np.random.default_rng(0).uniform(1, 255, (3, _SIZE, _SIZE, 3)).astype(np.float32)
    # Додатні ваги 1×1-згортки → активація каналу додатна для додатного входу.
    kernel, bias = model.get_layer("shallow").get_weights()
    model.get_layer("shallow").set_weights([np.abs(kernel), bias])

    maps = GradCam(model, _SIZE, batch_size=2).maps(images)

    hires = maps[cam_key("hirescam", "shallow")]
    assert hires.shape == (3, _SIZE, _SIZE)
    assert np.all(hires[:, 2, 5] > 0)
    hires[:, 2, 5] = 0
    assert np.abs(hires).max() == 0.0
    for key, value in maps.items():
        assert value.shape == (3, _SIZE, _SIZE), key
        assert value.min() >= 0, key
    assert maps[cam_key("gradcam", "fused")].max() <= 1.0 + 1e-6


def test_gradcam_refuses_layer_disconnected_from_target():
    model = _cam_model((1, 1), 0, use_deep=False)
    images = np.ones((1, _SIZE, _SIZE, 3), np.float32)

    with pytest.raises(CamGraphError, match="expose_backbone"):
        GradCam(model, _SIZE, batch_size=1).maps(images)


def test_exposed_backbone_gradients_match_staged_pass():
    """Градієнт за виходами ``expose_backbone`` = градієнт поетапного проходу (план, D4)."""
    from mllabs.anomaly.backbones import BACKBONE_SPECS, BackboneKind
    from mllabs.anomaly.detector import DetectorConfig, build_detector
    from mllabs.anomaly.fae import FaeSettings, FeatureAutoencoderBuilder

    keras.utils.set_random_seed(0)
    builder = FeatureAutoencoderBuilder(FaeSettings(feature_dim=8, hidden=8, latent=2))
    generator = np.random.default_rng(0)
    head = builder.build_head(
        builder.build_core(), generator.normal(size=8), generator.uniform(0.5, 2, size=8)
    )
    spec = BACKBONE_SPECS[BackboneKind.MOBILENET_V2]
    config = DetectorConfig(
        "toy", "fae", "mobilenet_v2", spec.layers, 8, 1, 1, 64, 64, 1.0, 0.0, 0.0, 0, "x", "y"
    )
    detector = build_detector(config, head, weights=None, expose_backbone=True)
    images = tf.constant(generator.uniform(0, 255, (2, 64, 64, 3)).astype(np.float32))

    with tf.GradientTape() as tape:
        outputs = detector(images, training=False)
    exposed = tape.gradient(
        outputs["image_score"], [outputs["backbone_shallow"], outputs["backbone_deep"]]
    )

    # Поетапно: шари за іменами в тому самому порядку, що в build_detector.
    def layer(name: str) -> keras.layers.Layer:
        return detector.get_layer(name)

    prepared = layer("preprocess")(layer("tile_split")(images))
    shallow, deep = layer("backbone")(prepared, training=False)
    with tf.GradientTape() as tape:
        tape.watch([shallow, deep])
        stacked = layer("concat")([shallow, layer("upsample")(deep)])
        features = layer("tile_merge")(layer("channel_pooling")(stacked))
        patch = layer("fae_head")(layer("patch_aggregation_p1")(features), training=False)
        score = keras.ops.max(layer("postprocess")(patch), axis=(1, 2))
    staged = tape.gradient(score, [shallow, deep])

    np.testing.assert_allclose(score.numpy(), outputs["image_score"].numpy(), rtol=1e-5)
    for first, second in zip(exposed, staged, strict=True):
        assert float(tf.reduce_sum(tf.abs(first))) > 0
        np.testing.assert_allclose(first.numpy(), second.numpy(), rtol=1e-4, atol=1e-5)


class _FakeReferences:
    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    def reference(self, image: np.ndarray) -> np.ndarray:
        return self._image

    def references(self, image: np.ndarray, count: int) -> np.ndarray:
        return np.repeat(self._image[None], count, axis=0)


def test_baseline_factory_shapes_and_kinds():
    image = np.full((_SIZE, _SIZE, 3), 100, np.uint8)
    reference = np.full((_SIZE, _SIZE, 3), 50, np.uint8)
    factory = BaselineFactory(_FakeReferences(reference), np.full((_SIZE, _SIZE, 3), 7.0), 1.0, 3)

    assert factory.make(BaselineKind.BLACK, image).max() == 0
    assert factory.make(BaselineKind.MEAN, image)[0, 0, 0, 0] == 7.0
    np.testing.assert_allclose(factory.make(BaselineKind.BLUR, image), 100.0, atol=1e-4)
    assert factory.make(BaselineKind.NEAREST_NORMAL, image)[0, 0, 0, 0] == 50
    assert factory.make(BaselineKind.EXPECTED, image).shape == (3, _SIZE, _SIZE, 3)


def test_reference_finder_takes_lowest_score_among_nearest():
    images = np.stack([np.full((32, 32, 3), value, np.uint8) for value in (10, 12, 200, 11)])
    scores = np.array([0.9, 0.1, 0.0, 0.5])
    finder = NormalReferenceFinder(images, scores, candidates=3)
    query = np.full((32, 32, 3), 10, np.uint8)

    # Найближчі за яскравістю: 10, 11, 12 → серед них мінімальна оцінка в «12».
    assert finder.reference_index(query) == 1
    assert list(finder.nearest(query, 2)) == [0, 3]
    assert finder.mean_image().shape == (32, 32, 3)
