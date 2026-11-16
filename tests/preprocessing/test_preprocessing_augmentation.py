"""Тести семплерів аугментації (GaussianNoiseSampler, SamplerFactory)."""

import numpy as np
import pytest
from imblearn.over_sampling import ADASYN, SMOTE

from mllabs.preprocessing import GaussianNoiseSampler, SamplerFactory, SamplerKind


@pytest.fixture
def imbalanced():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(110, 3)) * [1.0, 10.0, 100.0]
    labels = np.array([0] * 100 + [1] * 10)
    return features, labels


def test_noise_sampler_keeps_real_rows_unchanged_and_first(imbalanced):
    features, labels = imbalanced

    resampled, resampled_labels = GaussianNoiseSampler(
        sampling_strategy=0.5, random_state=0
    ).fit_resample(features, labels)

    np.testing.assert_array_equal(resampled[: len(features)], features)
    np.testing.assert_array_equal(resampled_labels[: len(labels)], labels)


def test_noise_sampler_reaches_requested_class_ratio(imbalanced):
    features, labels = imbalanced

    _, resampled_labels = GaussianNoiseSampler(sampling_strategy=0.5, random_state=0).fit_resample(
        features, labels
    )

    assert (resampled_labels == 0).sum() == 100  # мажоритарний клас не змінюється
    assert (resampled_labels == 1).sum() == 50


def test_noise_sampler_adds_small_noise_scaled_by_feature_std(imbalanced):
    features, labels = imbalanced
    minority = features[labels == 1]

    resampled, _ = GaussianNoiseSampler(
        sampling_strategy=1.0, noise_scale=0.05, random_state=0
    ).fit_resample(features, labels)
    synthetic = resampled[len(features) :]

    # Кожна синтетична точка — поблизу якоїсь реальної точки рідкісного класу.
    distances = np.abs(synthetic[:, None, :] - minority[None, :, :]) / features.std(axis=0)
    nearest = distances.max(axis=2).min(axis=1)
    assert nearest.max() < 0.5
    assert not np.isin(synthetic, minority).all()  # але не точна копія


def test_noise_sampler_with_zero_noise_duplicates_minority_rows(imbalanced):
    features, labels = imbalanced
    minority_rows = {tuple(row) for row in features[labels == 1]}

    resampled, _ = GaussianNoiseSampler(
        sampling_strategy=0.5, noise_scale=0.0, random_state=0
    ).fit_resample(features, labels)

    assert all(tuple(row) in minority_rows for row in resampled[len(features) :])


def test_noise_sampler_is_reproducible(imbalanced):
    features, labels = imbalanced
    sampler = GaussianNoiseSampler(sampling_strategy=0.5, random_state=3)

    first, _ = sampler.fit_resample(features, labels)
    second, _ = sampler.fit_resample(features, labels)

    np.testing.assert_array_equal(first, second)


def test_sampler_factory_creates_configured_samplers():
    factory = SamplerFactory(sampling_strategy=0.5, k_neighbors=3, noise_scale=0.1, random_state=1)

    assert factory.create(SamplerKind.NONE) is None
    assert isinstance(factory.create(SamplerKind.SMOTE), SMOTE)
    assert isinstance(factory.create(SamplerKind.ADASYN), ADASYN)
    noise = factory.create(SamplerKind.GAUSSIAN_NOISE)
    assert noise.get_params()["noise_scale"] == 0.1
    assert noise.get_params()["sampling_strategy"] == 0.5
