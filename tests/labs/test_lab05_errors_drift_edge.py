"""Тести ЛР5: правила причин FP/FN, статистика груп, трансформації дрейфу, TFLite, ЛР4-handoff."""

import keras
import numpy as np
import pandas as pd
import pytest

from mllabs.anomaly.visa import VisaDatasetSource
from mllabs.anomaly.workspace import VisaWorkspace, WorkspaceSettings
from mllabs.core import ProjectPaths
from mllabs.labs.lab05.config import DriftLevel, TfliteVariant
from mllabs.labs.lab05.drift import (
    BrightnessGain,
    GaussianBlur,
    GaussianNoise,
    IlluminationGradient,
    JpegCompression,
    Translation,
    first_failing_level,
    make_transform,
)
from mllabs.labs.lab05.edge import TfliteExporter, TfliteScorer
from mllabs.labs.lab05.error_cases import (
    CauseRules,
    DescriptorCalculator,
    DescriptorSettings,
    border_exclusion,
    compare_groups,
    cropped_scores,
    holm,
    ring_errors,
)
from mllabs.labs.lab05.handoff import Lab4Handoff, Lab4HandoffError, check_scores

_RULES = CauseRules(
    global_error=1.0,
    glare_share=0.05,
    edge_strength=10.0,
    border_px=16,
    small_area=0.001,
    low_contrast=5.0,
    off_map_auroc=0.9,
    fp_order=("global", "glare", "object_edge", "frame_border"),
    fn_order=("small", "low_contrast", "off_map"),
)


def _row(**values: object) -> pd.Series:
    base = {
        "outcome": "FP",
        "global_error": 0.5,
        "window_saturated": 0.0,
        "edge_strength": 1.0,
        "border_distance": 100.0,
        "border_edges": 0,
        "defect_area": 0.01,
        "defect_contrast": 20.0,
        "argmax_in_mask": 1.0,
        "map_auroc": 0.95,
    }
    return pd.Series(base | values)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"global_error": 2.0, "window_saturated": 0.5}, "global"),
        ({"window_saturated": 0.5}, "glare"),
        ({"edge_strength": 20.0}, "object_edge"),
        ({"border_distance": 3.0, "border_edges": 1}, "frame_border"),
        ({"border_distance": 3.0, "border_edges": 2}, "frame_corner"),
        ({}, "undetermined"),
    ],
)
def test_fp_cause_rules_follow_configured_order(values, expected):
    assert _RULES.fp_cause(_row(**values)) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"defect_area": 0.0001}, "small"),
        ({"defect_contrast": 1.0}, "low_contrast"),
        ({"argmax_in_mask": 0.0}, "off_map"),
        ({"argmax_in_mask": 0.0, "map_auroc": 0.5}, "undetermined"),
    ],
)
def test_fn_cause_rules(values, expected):
    assert _RULES.fn_cause(_row(outcome="FN", **values)) == expected


def test_assign_leaves_correct_decisions_without_cause():
    cases = pd.DataFrame([_row(outcome="TP"), _row(outcome="FP", window_saturated=0.5)])

    assert list(_RULES.assign(cases)) == ["", "glare"]


def test_holm_matches_manual_computation():
    adjusted = holm([0.01, 0.04, 0.03])

    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_compare_groups_detects_shift_and_skips_small_groups():
    generator = np.random.default_rng(0)
    cases = pd.DataFrame(
        {
            "outcome": ["FP"] * 20 + ["TN"] * 20 + ["FN"] * 2,
            "x": np.r_[generator.normal(3, 1, 20), generator.normal(0, 1, 20), [0.0, 1.0]],
        }
    )

    shifted = compare_groups(cases, ["x"], "FP", "TN", min_group_size=5).iloc[0]
    small = compare_groups(cases, ["x"], "FN", "TN", min_group_size=5).iloc[0]

    assert shifted["r"] > 0.8
    assert shifted["p_holm"] < 0.001
    assert np.isnan(small["p"])


def test_descriptors_locate_peak_and_measure_defect():
    images = np.full((1, 32, 32, 3), 100, np.uint8)
    images[0, 10:14, 20:24] = 250
    maps = np.zeros((1, 32, 32), np.float32)
    maps[0, 12, 22] = 1.0
    masks = np.zeros((1, 32, 32), bool)
    masks[0, 10:14, 20:24] = True
    settings = DescriptorSettings(8, 250, 5, 4, 2, 0, 4)

    row = DescriptorCalculator(settings).compute(images, maps, np.ones((1, 4, 4)), masks).iloc[0]

    assert (row["peak_row"], row["peak_col"]) == (12, 22)
    assert row["argmax_in_mask"] == 1.0
    assert row["defect_contrast"] == pytest.approx(150.0)
    assert row["window_saturated"] > 0
    assert row["border_distance"] == 9


def test_border_exclusion_ignores_hot_frame_and_counts_masks_in_band():
    maps = np.zeros((4, 16, 16))
    maps[0, 0, 0] = 5.0  # норма з «гарячим кутом»
    maps[1, 8, 8] = 3.0  # аномалія в центрі
    maps[2, 1, 1] = 0.1
    maps[3, 7, 7] = 4.0
    labels = np.array([0, 1, 0, 1])
    masks = np.zeros((4, 16, 16), bool)
    masks[3, 0:3, 0:3] = True

    full = border_exclusion(maps, labels, maps, labels, masks, band=0)
    cropped = border_exclusion(maps, labels, maps, labels, masks, band=2)

    assert cropped_scores(maps, 2)[0] == 0.0
    assert cropped.image_auroc > full.image_auroc
    assert cropped.masks_touching_band == 1


def test_ring_errors_separate_border_ring_from_center():
    patches = np.ones((2, 6, 6))
    patches[:, 0, :] = patches[:, -1, :] = patches[:, :, 0] = patches[:, :, -1] = 3.0

    rings = ring_errors(patches)

    assert rings["ring1"] == 3.0
    assert rings["ring2"] == rings["center"] == 1.0


@pytest.mark.parametrize(
    "transform",
    [
        BrightnessGain(1.0),
        IlluminationGradient(0.0),
        GaussianNoise(0.0),
        GaussianBlur(0.0),
        Translation(0),
    ],
)
def test_zero_level_transforms_are_identity(transform):
    images = np.random.default_rng(0).integers(0, 256, (2, 16, 16, 3), dtype=np.uint8)
    masks = np.zeros((2, 16, 16), bool)

    changed, changed_masks = transform.apply(images, masks, np.random.default_rng(0))

    assert changed.dtype == np.uint8
    assert np.array_equal(changed, images)
    assert np.array_equal(changed_masks, masks)


def test_translation_moves_image_and_mask_together():
    images = np.zeros((1, 16, 16, 3), np.uint8)
    images[0, 5, 6] = 255
    masks = np.zeros((1, 16, 16), bool)
    masks[0, 5, 6] = True

    changed, changed_masks = Translation(3).apply(images, masks, np.random.default_rng(0))

    assert changed[0, 8, 9, 0] == 255
    assert changed_masks[0, 8, 9]
    assert changed.shape == images.shape


def test_noise_transform_is_deterministic_for_seed_and_jpeg_keeps_shape():
    images = np.full((1, 16, 16, 3), 128, np.uint8)
    masks = np.zeros((1, 16, 16), bool)
    noise = make_transform(DriftLevel("noise", 6.0, 1))

    first, _ = noise.apply(images, masks, np.random.default_rng(1))
    second, _ = noise.apply(images, masks, np.random.default_rng(1))
    jpeg, _ = JpegCompression(20).apply(images, masks, np.random.default_rng(0))

    assert np.array_equal(first, second)
    assert not np.array_equal(first, images)
    assert jpeg.shape == images.shape and jpeg.dtype == np.uint8


def test_first_failing_level_uses_both_tolerances():
    levels = [(1.0, 0.95, 0.10), (2.0, 0.94, 0.25), (3.0, 0.80, 0.30)]

    assert first_failing_level(levels, (0.95, 0.10), 0.02, 0.10) == 2.0
    assert first_failing_level(levels[:1], (0.95, 0.10), 0.02, 0.10) is None


def _tiny_edge_model() -> keras.Model:
    keras.utils.set_random_seed(0)
    image = keras.Input((16, 16, 3), name="image")
    features = keras.layers.Conv2D(4, 3, padding="same", activation="relu")(image / 255.0)
    anomaly_map = keras.layers.Conv2D(1, 1)(features)
    anomaly_map = keras.layers.Reshape((16, 16))(anomaly_map)
    score = keras.layers.GlobalMaxPooling1D()(anomaly_map)
    score = keras.layers.Reshape(())(keras.layers.Lambda(lambda x: x[:, :1])(score))
    return keras.Model(image, {"anomaly_map": anomaly_map, "image_score": score})


def test_tflite_fp32_matches_keras_and_int8_converts():
    model = _tiny_edge_model()
    images = np.random.default_rng(0).uniform(0, 255, (4, 16, 16, 3)).astype(np.float32)
    exporter = TfliteExporter(images)

    fp32 = TfliteScorer(exporter.export(model, TfliteVariant.FP32), threads=1)
    maps, _ = fp32.run_all(images)
    int8 = exporter.export(model, TfliteVariant.FULL_INT8)

    np.testing.assert_allclose(maps, model(images)["anomaly_map"].numpy(), atol=1e-4)
    assert len(int8) > 0
    assert TfliteScorer(int8, threads=1).run(images[0])[0].shape == (16, 16)


def test_handoff_without_lab4_models_explains_how_to_fix(tmp_path):
    project = ProjectPaths(root=tmp_path)
    source = VisaDatasetSource(tmp_path / "raw", "http://invalid", "0", "archive.tar")
    workspace = VisaWorkspace(project, WorkspaceSettings(categories=("toy",)), source)
    handoff = Lab4Handoff(project, workspace, ("toy",), "FAE")

    with pytest.raises(Lab4HandoffError, match="mllabs run 4"):
        handoff.prepare()


def test_check_scores_rejects_inconsistent_predictions():
    predictions = pd.DataFrame({"score": [0.1, 0.2]})

    assert check_scores("toy", np.array([0.1, 0.2001]), predictions, 1e-3) < 1e-3
    with pytest.raises(Lab4HandoffError, match="не узгоджені"):
        check_scores("toy", np.array([0.1, 0.5]), predictions, 1e-3)
