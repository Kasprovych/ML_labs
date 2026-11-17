"""Тести моделей детекції: ознаки, головки, пост-обробка, метрики, наскрізна модель."""

import keras
import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

from mllabs.anomaly.backbones import (
    BackboneKind,
    ChannelPooling,
    FeatureExtractorFactory,
    TileMerge,
    TileSplit,
    channel_pooling_matrix,
)
from mllabs.anomaly.detector import DetectorConfig, build_detector, score_patches
from mllabs.anomaly.fae import FaeScorer, FaeSettings, FaeTrainer, fae_analytic_flops
from mllabs.anomaly.features import FeatureCache, PatchAggregator
from mllabs.anomaly.metrics import HistogramAuc, ImageBootstrap, PixelSummary
from mllabs.anomaly.padim import Padim
from mllabs.anomaly.patchcore import GreedyCoresetSampler, PatchCore
from mllabs.anomaly.postprocess import AnomalyMapPostprocessor
from mllabs.anomaly.profiling import FlopsCounter, LatencyBenchmark

_SMALL_FAE = FaeSettings(
    feature_dim=8,
    hidden=8,
    latent=2,
    batch_size=64,
    vectors_per_epoch=512,
    validation_vectors=128,
    max_epochs=3,
    patience=2,
)


def test_channel_pooling_matrix_averages_contiguous_groups():
    matrix = channel_pooling_matrix(6, 3)

    expected = np.zeros((6, 3), np.float32)
    for column in range(3):
        expected[2 * column : 2 * column + 2, column] = 0.5
    np.testing.assert_allclose(matrix, expected)
    np.testing.assert_allclose(channel_pooling_matrix(5, 3).sum(axis=0), 1.0)


def test_channel_pooling_layer_matches_reference_matrix():
    features = np.random.default_rng(0).normal(size=(2, 3, 3, 12)).astype(np.float32)

    pooled = ChannelPooling(4)(features).numpy()

    np.testing.assert_allclose(
        pooled, features @ channel_pooling_matrix(12, 4), rtol=1e-5, atol=1e-6
    )


def test_patch_aggregator_with_p1_is_identity_and_p3_averages_neighbours():
    features = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)

    assert np.array_equal(PatchAggregator(1).apply(features), features.astype(np.float16))
    aggregated = PatchAggregator(3).apply(features).astype(np.float32)
    assert aggregated[0, 1, 1, 0] == pytest.approx(features[0, :3, :3, 0].mean())
    # На краю усереднюються лише наявні сусіди (доповнення не нулями).
    assert aggregated[0, 0, 0, 0] == pytest.approx(features[0, :2, :2, 0].mean())


def test_tile_split_and_merge_are_inverse():
    images = np.random.default_rng(1).normal(size=(2, 8, 8, 3)).astype(np.float32)

    tiles = TileSplit(2)(images)
    merged = TileMerge(2)(tiles).numpy()

    assert tiles.shape == (8, 4, 4, 3)
    np.testing.assert_array_equal(tiles.numpy()[1], images[0, :4, 4:])
    np.testing.assert_array_equal(merged, images)


def test_postprocessor_keeps_constant_map_constant_and_upsamples():
    maps = np.full((2, 4, 4), 3.0, np.float32)

    result = AnomalyMapPostprocessor(32, sigma=2.0).maps(maps)

    assert result.shape == (2, 32, 32)
    np.testing.assert_allclose(result, 3.0, rtol=1e-5)


def test_flops_counter_counts_dense_as_two_in_out_per_position():
    inputs = keras.Input((None, None, 16))
    model = keras.Model(inputs, keras.layers.Dense(4, use_bias=False)(inputs))

    flops = FlopsCounter().count(model, (1, 3, 3, 16))

    assert flops == 2 * 16 * 4 * 9


def test_fae_analytic_flops_sum_dense_layers():
    settings = FaeSettings(feature_dim=10, hidden=6, latent=2)
    per_position = 2 * (10 * 6 + 6 * 2 + 2 * 6 + 6 * 10)

    assert fae_analytic_flops(settings, 4) == per_position * 4


def test_histogram_metrics_match_sklearn():
    generator = np.random.default_rng(0)
    masks = np.zeros((20, 32, 32), bool)
    masks[:10, 10:14, 10:14] = True
    maps = generator.normal(size=masks.shape).astype(np.float32) + 2.0 * masks

    auroc, ap = HistogramAuc(4096).metrics(maps, masks)

    assert auroc == pytest.approx(roc_auc_score(masks.ravel(), maps.ravel()), abs=1e-3)
    assert ap == pytest.approx(average_precision_score(masks.ravel(), maps.ravel()), abs=1e-3)


def test_image_bootstrap_is_reproducible_and_brackets_point_estimate():
    generator = np.random.default_rng(0)
    masks = np.zeros((30, 16, 16), bool)
    masks[:15, 4:8, 4:8] = True
    maps = generator.normal(size=masks.shape).astype(np.float32) + 1.5 * masks
    histogram = HistogramAuc(512)
    positive, negative = histogram.per_image_counts(maps, masks, histogram.edges(maps, masks))
    labels = masks.reshape(30, -1).any(axis=1).astype(int)
    summary = PixelSummary(positive, negative, labels, maps.reshape(30, -1).max(axis=1))
    bootstrap = ImageBootstrap(200, 0.95, random_state=1)

    first, second = bootstrap.samples(summary), bootstrap.samples(summary)
    point = HistogramAuc.auroc(positive.sum(0), negative.sum(0))
    interval = bootstrap.interval(first["pixel_auroc"], point)

    np.testing.assert_array_equal(first["pixel_auroc"], second["pixel_auroc"])
    assert interval.low <= point <= interval.high


def _normal_and_shifted_maps() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    train = generator.normal(size=(40, 4, 4, 8)).astype(np.float32)
    test = generator.normal(size=(2, 4, 4, 8)).astype(np.float32)
    test[1, 1, 1] += 6.0  # один «дефектний» патч
    return train, test


def test_padim_scores_shifted_patch_highest():
    train, test = _normal_and_shifted_maps()
    padim = Padim(n_channels=6, epsilon=0.01, random_state=0).fit(train)

    scores = score_patches(padim, test, batch_size=2)

    assert scores.shape == (2, 4, 4)
    assert np.unravel_index(scores.argmax(), scores.shape) == (1, 1, 1)
    assert padim.memory_bytes() == (16 * 6 + 16 * 36) * 4


def test_coreset_returns_unique_indices_of_requested_size():
    vectors = np.random.default_rng(0).normal(size=(500, 8)).astype(np.float32)

    indices = GreedyCoresetSampler(0.1, 4, random_state=0).select(vectors)

    assert len(indices) == 50
    assert len(np.unique(indices)) == 50


def test_patchcore_scores_shifted_patch_highest_and_caps_train_vectors():
    train, test = _normal_and_shifted_maps()
    patchcore = PatchCore(GreedyCoresetSampler(0.2, 4, 0), max_train_vectors=300, random_state=0)
    patchcore.fit(train)

    scores = score_patches(patchcore, test, batch_size=2)

    assert patchcore.n_train_vectors == 300
    assert patchcore.bank_size == 60
    assert np.unravel_index(scores.argmax(), scores.shape) == (1, 1, 1)


def test_fae_trained_on_vectors_applies_to_maps_and_is_seed_deterministic():
    train, test = _normal_and_shifted_maps()

    first = FaeTrainer(_SMALL_FAE, seed=5).train(train, train[:8])
    second = FaeTrainer(_SMALL_FAE, seed=5).train(train, train[:8])
    scores = score_patches(FaeScorer(first.head), test, batch_size=2)

    assert scores.shape == (2, 4, 4)
    assert {"loss", "train_mse", "val_mse", "epoch"} <= set(first.history.columns)
    for left, right in zip(first.head.weights, second.head.weights, strict=True):
        np.testing.assert_allclose(left.numpy(), right.numpy())


def test_fae_head_survives_keras_save_and_load(tmp_path):
    train, test = _normal_and_shifted_maps()
    head = FaeTrainer(_SMALL_FAE, seed=0).train(train, train[:8]).head
    path = tmp_path / "fae_head.keras"

    head.save(path)
    loaded = keras.models.load_model(path)

    np.testing.assert_allclose(loaded(test).numpy(), head(test).numpy(), rtol=1e-5)


@pytest.fixture(scope="module")
def tiny_detector_parts():
    """Малий backbone без завантажених ваг (MobileNetV2, вхід 64×64, D = 8)."""
    keras.utils.set_random_seed(0)
    factory = FeatureExtractorFactory(64, 8, weights=None)
    extractor = factory.build(BackboneKind.MOBILENET_V2)
    images = np.random.default_rng(0).uniform(0, 255, size=(6, 64, 64, 3)).astype(np.float32)
    features = extractor(images).numpy()
    settings = FaeSettings(
        feature_dim=8, hidden=8, latent=2, batch_size=32, vectors_per_epoch=256, max_epochs=2
    )
    head = FaeTrainer(settings, seed=0).train(features[:4], features[4:]).head
    config = DetectorConfig(
        category="toy",
        head="fae",
        backbone="mobilenet_v2",
        layers=("block_6_expand_relu", "block_13_expand_relu"),
        feature_dim=8,
        patch_size=3,
        tile_grid=1,
        image_size=64,
        eval_size=32,
        blur_sigma=1.0,
        threshold_image=0.0,
        threshold_pixel=0.0,
        seed=0,
        tf_version="x",
        keras_version="y",
    )
    return extractor, images, features, head, config


def test_end_to_end_detector_matches_cached_feature_path(tiny_detector_parts, tmp_path):
    extractor, images, _, head, config = tiny_detector_parts
    detector = build_detector(config, head, weights=None)
    # Ваги backbone випадкові — копіюємо їх з екстрактора, яким рахувався «кеш».
    detector.get_layer("backbone").set_weights(extractor.get_layer("backbone").get_weights())
    cache = FeatureCache(tmp_path)
    cached = cache.get_or_compute("toy", "set", {"v": 1}, extractor, images.astype(np.uint8), 4)

    aggregated = PatchAggregator(3).apply(cached)
    patch = score_patches(FaeScorer(head), aggregated[:2], batch_size=2)
    expected = AnomalyMapPostprocessor(32, 1.0).maps(patch)
    outputs = detector(images[:2].astype(np.uint8).astype(np.float32))

    assert outputs["anomaly_map"].shape == (2, 32, 32)
    np.testing.assert_allclose(outputs["anomaly_map"].numpy(), expected, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(outputs["image_score"].numpy(), expected.max(axis=(1, 2)), rtol=2e-2)


def test_detector_gives_nonzero_gradient_of_score_wrt_image(tiny_detector_parts):
    _, images, _, head, config = tiny_detector_parts
    detector = build_detector(config, head, weights=None)
    pixels = tf.Variable(images[:1])

    with tf.GradientTape() as tape:
        score = detector(pixels)["image_score"]
    gradient = tape.gradient(score, pixels)

    assert gradient is not None
    assert float(tf.reduce_sum(tf.abs(gradient))) > 0


def test_exposed_backbone_activations_support_grad_cam(tiny_detector_parts):
    _, images, _, head, config = tiny_detector_parts
    detector = build_detector(config, head, weights=None, expose_backbone=True)

    with tf.GradientTape() as tape:
        outputs = detector(images[:2])
    gradients = tape.gradient(
        outputs["image_score"], [outputs["backbone_shallow"], outputs["backbone_deep"]]
    )

    assert outputs["backbone_deep"].shape[-1] == 576  # block_13_expand_relu MobileNetV2
    assert detector.get_layer("fae_head") is head
    for gradient in gradients:
        assert gradient is not None
        assert float(tf.reduce_sum(tf.abs(gradient))) > 0


def test_saved_bundle_has_no_optimizer_state_and_reloads(tiny_detector_parts, tmp_path):
    import zipfile

    from mllabs.anomaly.detector import load_bundle, save_bundle

    _, images, _, head, config = tiny_detector_parts
    save_bundle(tmp_path, head, config)
    loaded_config, detector = load_bundle(tmp_path, weights=None)

    with zipfile.ZipFile(tmp_path / "fae_head.keras") as archive:
        assert not any("optimizer" in name for name in archive.namelist())
    assert loaded_config == config
    assert detector(images[:1])["anomaly_map"].shape == (1, 32, 32)


def test_detector_config_round_trips_through_json(tiny_detector_parts, tmp_path):
    config = tiny_detector_parts[-1]

    assert DetectorConfig.load(config.save(tmp_path / "d.json")) == config


def test_latency_benchmark_reports_per_image_time():
    result = LatencyBenchmark(warmup=1, runs=5).measure(lambda: sum(range(1000)), batch_size=4)

    assert result.runs == 5
    assert 0 <= result.median_ms <= result.p90_ms
