"""Наскрізний тест ЛР5 на мікро-«ЛР4»: синтетичні зображення, backbone без ваг, усі кроки.

Замість справжнього запуску ЛР4 (≈ 25 хв) у тимчасовому каталозі відтворюється все, що ЛР5
бере з ЛР4: розбиття, зображення, кеш ознак, збережений детектор, ``test_predictions.csv`` і
таблиці профілювання.
"""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib.use("Agg")  # малювання без вікна

from mllabs.anomaly.backbones import BACKBONE_SPECS, BackboneKind
from mllabs.anomaly.detector import DetectorConfig, save_bundle
from mllabs.anomaly.fae import FaeSettings, FaeTrainer
from mllabs.anomaly.splits import SPLIT_COLUMNS
from mllabs.anomaly.visa import VisaDatasetSource
from mllabs.anomaly.workspace import FeatureKey, VisaWorkspace, WorkspaceSettings
from mllabs.core import LabContext, ProjectPaths
from mllabs.labs.lab05.config import DriftLevel, Lab5Config, TfliteVariant
from mllabs.labs.lab05.handoff import Lab4Handoff
from mllabs.labs.lab05.inference import CachedScorer
from mllabs.labs.lab05.lab import Lab5

_CATEGORIES = ("alpha", "beta")
_SIZE = 64
_DIM = 16
_PARTS = (("train", 0, 12), ("val_normal", 0, 2), ("val", 0, 4), ("val", 1, 4), ("test", 0, 4))
_TEST_ANOMALIES = 6


def _write_images(root, category: str, rng: np.random.Generator) -> list[dict]:
    rows = []
    folder = root / category / "Data"
    (folder / "Images").mkdir(parents=True)
    (folder / "Masks").mkdir(parents=True)
    plan = [*_PARTS, ("test", 1, _TEST_ANOMALIES)]
    number = 0
    for part, label, count in plan:
        for _ in range(count):
            pixels = rng.normal(120, 12, size=(_SIZE, _SIZE, 3)).clip(0, 255).astype(np.uint8)
            mask = np.zeros((_SIZE, _SIZE), np.uint8)
            if label:
                side = int(rng.integers(6, 20))
                top, left = rng.integers(2, _SIZE - side - 2, size=2)
                pixels[top : top + side, left : left + side] = 250
                mask[top : top + side, left : left + side] = 255
            image_name = f"{category}/Data/Images/{number:03d}.JPG"
            Image.fromarray(pixels).save(root / image_name, format="JPEG", quality=95)
            mask_name = ""
            if label:
                mask_name = f"{category}/Data/Masks/{number:03d}.png"
                Image.fromarray(mask, mode="L").save(root / mask_name)
            rows.append(
                {
                    "category": category,
                    "official_split": "train" if part in ("train", "val_normal") else "test",
                    "part": part,
                    "label": label,
                    "image": image_name,
                    "mask": mask_name,
                    "defect_types": "square,bright" if label else "",
                }
            )
            number += 1
    return rows


def _detector_config(category: str) -> DetectorConfig:
    spec = BACKBONE_SPECS[BackboneKind.MOBILENET_V2]
    return DetectorConfig(
        category=category,
        head="fae",
        backbone=str(BackboneKind.MOBILENET_V2),
        layers=spec.layers,
        feature_dim=_DIM,
        patch_size=1,
        tile_grid=1,
        image_size=_SIZE,
        eval_size=_SIZE,
        blur_sigma=1.0,
        threshold_image=0.0,
        threshold_pixel=1.0,
        seed=0,
        tf_version="x",
        keras_version="y",
    )


def _lr4_tables(directory) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    backbones = ("MobileNetV2", "EfficientNetB0", "ResNet50")
    latency = [
        {
            "Потоки": threads,
            "Backbone": backbone,
            "Компонент": "наскрізно",
            "Головка": "FAE",
            "Пакет": 1,
            "Медіана, мс/зобр.": 10.0 + index,
            "p90, мс/зобр.": 11.0 + index,
        }
        for threads in ("1", "усі")
        for index, backbone in enumerate(backbones)
    ]
    pd.DataFrame(latency).to_csv(directory / "latency.csv", index=False)
    flops = [{"Backbone": b, "Головка": "FAE", "GFLOPs разом": 1.0} for b in backbones]
    pd.DataFrame(flops).to_csv(directory / "flops.csv", index=False)
    study = [{"Варіант": b, "Pixel AUROC": 0.9} for b in backbones]
    pd.DataFrame(study).to_csv(directory / "study_backbones.csv", index=False)


@pytest.fixture(scope="module")
def micro_lab4(tmp_path_factory):
    """Тимчасовий проєкт з артефактами «ЛР4» для двох синтетичних категорій."""
    project = ProjectPaths(root=tmp_path_factory.mktemp("lab5_project"))
    rng = np.random.default_rng(0)
    raw = project.data_raw_dir / "visa"
    rows = [row for category in _CATEGORIES for row in _write_images(raw, category, rng)]
    splits = pd.DataFrame(rows)[list(SPLIT_COLUMNS)]
    splits_path = project.data_processed_dir / "visa" / "splits.csv"
    splits_path.parent.mkdir(parents=True)
    splits.to_csv(splits_path, index=False)
    settings = WorkspaceSettings(
        categories=_CATEGORIES, image_size=_SIZE, eval_size=_SIZE, feature_dim=_DIM
    )
    source = VisaDatasetSource(raw, "http://invalid", "0", "archive.tar")
    workspace = VisaWorkspace(project, settings, source, weights=None)
    workspace.load_splits()
    lab4 = project.lab(4)
    predictions = []
    fae = FaeSettings(
        feature_dim=_DIM, hidden=8, latent=4, batch_size=64, vectors_per_epoch=512, max_epochs=2
    )
    for category in _CATEGORIES:
        features = workspace.raw_features(category, FeatureKey(BackboneKind.MOBILENET_V2, 1))
        parts = workspace.category_rows(category)["part"].to_numpy()
        head = (
            FaeTrainer(fae, seed=0)
            .train(
                np.asarray(features[parts == "train"]), np.asarray(features[parts == "val_normal"])
            )
            .head
        )
        test_rows = np.flatnonzero(parts == "test")
        scores = CachedScorer(head, _SIZE, 1.0, 4).score(features, test_rows).scores
        threshold = float(np.median(scores))
        config = _detector_config(category)
        config = DetectorConfig(**{**config.__dict__, "threshold_image": threshold})
        save_bundle(lab4.models_dir / category, head, config)
        frame = workspace.category_rows(category).iloc[test_rows].reset_index(drop=True)
        decision = (scores >= threshold).astype(int)
        outcome = np.where(
            frame["label"] == 1,
            np.where(decision == 1, "TP", "FN"),
            np.where(decision == 1, "FP", "TN"),
        )
        predictions.append(
            pd.DataFrame(
                {
                    "category": category,
                    "method": "FAE",
                    "image": frame["image"],
                    "label": frame["label"],
                    "defect_types": frame["defect_types"],
                    "defect_area_share": 0.01 + 0.01 * np.arange(len(frame)) * frame["label"],
                    "brightness_std": 1.0,
                    "score": scores,
                    "threshold": threshold,
                    "decision": decision,
                    "outcome": outcome,
                }
            )
        )
    _lr4_tables(lab4.tables_dir)
    pd.concat(predictions).to_csv(lab4.tables_dir / "test_predictions.csv", index=False)
    return project, workspace


_SMALL_CONFIG = Lab5Config(
    categories=_CATEGORIES,
    handoff_end_to_end_images=2,
    inference_batch=4,
    head_batch=4,
    histogram_bins=256,
    train_vectors=500,
    top_k_channels=(2, 4, 16),
    pca_components=4,
    jacobian_patches=8,
    jacobian_batch=4,
    ig_anomalies_per_category=3,
    perturbation_fp_per_category=1,
    val_images_for_selection=1,
    ig_steps=8,
    ig_max_steps=16,
    ig_alpha_batch=9,
    baseline_blur_sigma=2.0,
    reference_candidates=2,
    expected_gradients_k=2,
    attribution_blur_sigma=1.0,
    pointing_tolerance_px=2,
    correlation_grid=16,
    bootstrap_repeats=20,
    perturbation_steps=(0, 1, 2, 4),
    random_orders=2,
    sanity_images_per_category=1,
    sanity_ig_steps=8,
    descriptor_window=8,
    border_px=8,
    border_exclusion_px=(0, 4),
    brightness_cells=4,
    contrast_ring_px=2,
    drift_levels=(DriftLevel("noise", 6.0, 1), DriftLevel("shift", 4.0, 1)),
    tflite_variants=(TfliteVariant.FP32, TfliteVariant.DYNAMIC_INT8),
    calibration_images=2,
    latency_warmup=1,
    latency_runs=2,
    edge_input_sizes=(_SIZE, 48),
)


def test_lab5_runs_all_steps_on_micro_lab4(micro_lab4):
    project, workspace = micro_lab4
    context = LabContext(number=5, project=project, seed=0)
    handoff = Lab4Handoff(project, workspace, _CATEGORIES, "FAE", weights=None)

    Lab5(context, _SMALL_CONFIG, handoff=handoff, weights=None).run()

    metrics = json.loads(context.paths.metrics_file.read_text(encoding="utf-8"))
    for key in ("handoff", "weights", "xai", "sanity", "errors", "drift", "edge", "limits"):
        assert key in metrics, key
    assert metrics["handoff"]["alpha"]["max_score_diff"] < 1e-3
    assert metrics["edge"]["postprocess_ms"]["latency_ms_1"] > 0
    assert metrics["edge"]["relative_to_fp32"]["fp32"]["latency_ms_1"] == 1.0
    assert set(metrics["errors"]["total"]) == {"n_fp", "n_fn", "n_tp", "n_tn"}
    assert metrics["xai"]["ig_baseline"] in {"blur", "nearest_normal", "expected", "mean", "black"}
    tables = context.paths.tables_dir
    for name in ("localization_summary", "perturbation_summary", "error_cases", "drift"):
        assert (tables / f"{name}.csv").is_file(), name
    for name in ("xai_panels_alpha", "perturbation_curves", "edge_tradeoff", "drift_curves"):
        assert (context.paths.figures_dir / f"{name}.png").is_file(), name
    assert (context.paths.models_dir / "alpha" / "fp32.tflite").is_file()
