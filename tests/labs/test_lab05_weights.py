"""Тести аналізу ваг FAE (R1): розклад помилки, шари, PCA, якобіан, спектри."""

import keras
import numpy as np
import pytest

from mllabs.anomaly.backbones import channel_pooling_matrix
from mllabs.anomaly.fae import FaeSettings, FeatureAutoencoderBuilder
from mllabs.labs.lab05.handoff import fae_parts
from mllabs.labs.lab05.weights_analysis import (
    DEFECT,
    NORMAL,
    ErrorDecomposition,
    JacobianAnalyzer,
    LayerAttribution,
    LinearProjectionBaseline,
    SpectrumSummary,
    patch_coverage,
    patch_groups,
    principal_angle_cosines,
    sample_vectors,
)

_DIM = 8


@pytest.fixture(scope="module")
def tiny_head():
    keras.utils.set_random_seed(0)
    builder = FeatureAutoencoderBuilder(FaeSettings(feature_dim=_DIM, hidden=6, latent=3))
    core = builder.build_core()
    generator = np.random.default_rng(0)
    mean = generator.normal(size=_DIM).astype(np.float32)
    variance = generator.uniform(0.5, 2, size=_DIM).astype(np.float32)
    return builder.build_head(core, mean, variance)


def test_channel_contributions_sum_to_head_error(tiny_head):
    features = np.random.default_rng(1).normal(size=(2, 4, 4, _DIM)).astype(np.float32)
    decomposition = ErrorDecomposition(fae_parts(tiny_head), batch_size=2)

    contributions = decomposition.channel_errors(features)

    np.testing.assert_allclose(contributions.sum(axis=-1), tiny_head(features).numpy(), rtol=1e-4)


def test_fae_parts_expose_normalization_and_latent(tiny_head):
    parts = fae_parts(tiny_head)
    vectors = np.random.default_rng(2).normal(size=(5, _DIM)).astype(np.float32)

    normalized = parts.normalize(vectors)
    expected = tiny_head.get_layer("normalization")(vectors).numpy().reshape(normalized.shape)

    np.testing.assert_allclose(normalized, expected, rtol=1e-4, atol=1e-5)
    assert parts.encoder(normalized[:, None, None, :]).shape == (5, 1, 1, 3)
    assert list(parts.dense_kernels) == [
        "encoder_hidden",
        "latent",
        "decoder_hidden",
        "reconstruction",
    ]


def test_layer_attribution_on_real_pooling_matrix():
    layers = LayerAttribution(240, 672, 512)
    pooling = channel_pooling_matrix(912, 512)

    share = layers.shallow_share
    assert share.shape == (512,)
    assert np.all((share >= 0) & (share <= 1))
    pure = (pooling[:240] > 0).any(axis=0) != (pooling[240:] > 0).any(axis=0)
    assert set(np.unique(share[pure])) <= {0.0, 1.0}
    # Сумарна вага шару 8 по каналах = частка його вхідних каналів (усереднення зберігає масу).
    assert share.sum() == pytest.approx(512 * 240 / 912, rel=0.02)
    np.testing.assert_allclose(layers.weights.sum(axis=1), 1.0)


def test_patch_groups_split_normal_defect_and_context():
    masks = np.zeros((2, 8, 8), bool)
    masks[1, :4, :4] = True  # повністю покриває патч (0, 0) сітки 2×2
    coverage = patch_coverage(masks, 2)

    groups = patch_groups(coverage, np.array([0, 1]), threshold=0.25)

    assert coverage[1, 0, 0] == 1.0
    assert (groups[0] == NORMAL).all()
    assert groups[1, 0, 0] == DEFECT
    assert groups[1, 1, 1] == "context"


def test_linear_projection_has_zero_residual_inside_subspace():
    generator = np.random.default_rng(0)
    basis = np.linalg.qr(generator.normal(size=(_DIM, 3)))[0]
    data = generator.normal(size=(500, 3)) @ basis.T
    model = LinearProjectionBaseline(3, seed=0).fit(data)

    inside = generator.normal(size=(20, 3)) @ basis.T
    outside = inside + 5 * np.linalg.svd(basis, full_matrices=True)[0][:, -1]

    assert model.residual(inside).max() < 1e-8
    assert model.residual(outside).min() > 1.0
    np.testing.assert_allclose(principal_angle_cosines(model.basis, basis.T), 1.0, atol=1e-6)


def test_jacobian_of_linear_core_equals_its_matrix():
    inputs = keras.Input((None, None, _DIM))
    outputs = keras.layers.Dense(_DIM, use_bias=False)(inputs)
    core = keras.Model(inputs, outputs)
    kernel = core.layers[-1].get_weights()[0]
    points = np.random.default_rng(0).normal(size=(3, _DIM)).astype(np.float32)

    analyzer = JacobianAnalyzer(core, batch_size=2)
    jacobians = analyzer.jacobians(points)
    summary = analyzer.summarize(points)

    np.testing.assert_allclose(jacobians[0], kernel.T, rtol=1e-5, atol=1e-6)
    assert summary.singular_values.shape == (3, _DIM)


def test_jacobian_of_projector_is_idempotent():
    basis = np.linalg.qr(np.random.default_rng(0).normal(size=(_DIM, 2)))[0]
    projector = (basis @ basis.T).astype(np.float32)
    inputs = keras.Input((None, None, _DIM))
    dense = keras.layers.Dense(_DIM, use_bias=False)
    core = keras.Model(inputs, dense(inputs))
    dense.set_weights([projector])

    summary = JacobianAnalyzer(core, batch_size=4).summarize(np.ones((2, _DIM), np.float32))

    assert summary.projection_error.max() < 1e-5
    assert list(summary.rank(0.5)) == [2, 2]


def test_effective_rank_of_isotropic_matrix_equals_its_size():
    spectrum = SpectrumSummary("x", (4, 4), np.ones(4))

    assert spectrum.effective_rank == pytest.approx(4.0)


def test_sample_vectors_returns_requested_rows_from_selected_images():
    features = np.arange(4 * 2 * 2 * 3, dtype=np.float32).reshape(4, 2, 2, 3)
    generator = np.random.default_rng(0)

    vectors = sample_vectors(features, np.array([1, 3]), 5, generator)

    allowed = {tuple(row) for row in features[[1, 3]].reshape(-1, 3)}
    assert vectors.shape == (5, 3)
    assert all(tuple(row) in allowed for row in vectors)
