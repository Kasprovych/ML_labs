"""Наскрізний тест ЛР4 на мікро-VisA: синтетичні зображення, backbone без ваг, усі кроки."""

import hashlib
import io
import json
import tarfile

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib.use("Agg")  # малювання без вікна

from mllabs.anomaly.detector import load_bundle
from mllabs.anomaly.fae import FaeSettings
from mllabs.anomaly.visa import VisaDatasetSource
from mllabs.core import LabContext, ProjectPaths
from mllabs.labs.lab04.config import Lab4Config
from mllabs.labs.lab04.lab import Lab4

_CATEGORIES = ("alpha", "beta")
_SIZE = (80, 60)

_SMALL_CONFIG = Lab4Config(
    categories=_CATEGORIES,
    study_category="alpha",
    background_categories=(*_CATEGORIES, "gamma"),
    background_sample=3,
    brightness_cells=4,
    image_size=64,
    eval_size=64,
    blur_sigma=1.0,
    feature_dim=16,
    patch_sizes=(1, 3),
    default_patch_size=3,
    study_tile_grid=2,
    extraction_batch_size=8,
    fae=FaeSettings(
        feature_dim=16,
        hidden=16,
        latent=4,
        batch_size=64,
        vectors_per_epoch=512,
        validation_vectors=128,
        max_epochs=2,
        patience=1,
    ),
    latent_grid=(2,),
    padim_channels=8,
    patchcore_coreset_ratio=0.05,
    patchcore_projection_dim=8,
    patchcore_max_train_vectors=5000,
    scoring_batch_size=4,
    bootstrap_repeats=50,
    histogram_bins=256,
    latency_warmup=1,
    latency_runs=2,
    latency_runs_batch=1,
    latency_batch=2,
    decode_latency_images=2,
)


def _image(rng: np.random.Generator, defect: bool) -> tuple[bytes, bytes]:
    pixels = rng.normal(120, 10, size=(_SIZE[1], _SIZE[0], 3)).clip(0, 255).astype(np.uint8)
    mask = np.zeros((_SIZE[1], _SIZE[0]), np.uint8)
    if defect:
        top, left = rng.integers(5, 35), rng.integers(5, 55)
        pixels[top : top + 15, left : left + 15] = 250
        mask[top : top + 15, left : left + 15] = 1
    image_buffer, mask_buffer = io.BytesIO(), io.BytesIO()
    Image.fromarray(pixels).save(image_buffer, format="JPEG")
    Image.fromarray(mask, mode="L").save(mask_buffer, format="PNG")
    return image_buffer.getvalue(), mask_buffer.getvalue()


def _add(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _micro_visa(root) -> VisaDatasetSource:
    rng = np.random.default_rng(0)
    archive = root / "visa" / "micro.tar"
    archive.parent.mkdir(parents=True)
    split_rows = ["object,split,label,image,mask"]
    with tarfile.open(archive, "w") as tar:
        for category in (*_CATEGORIES, "gamma"):
            anno = ["image,label,mask"]
            for index in range(26):
                name = f"{category}/Data/Images/Normal/{index:03d}.JPG"
                _add(tar, name, _image(rng, defect=False)[0])
                split = "train" if index < 20 else "test"
                split_rows.append(f"{category},{split},normal,{name},")
                anno.append(f"{name},normal,")
            for index in range(6):
                name = f"{category}/Data/Images/Anomaly/{index:03d}.JPG"
                mask_name = f"{category}/Data/Masks/Anomaly/{index:03d}.png"
                image, mask = _image(rng, defect=True)
                _add(tar, name, image)
                _add(tar, mask_name, mask)
                split_rows.append(f"{category},test,anomaly,{name},{mask_name}")
                anno.append(f'{name},"spot,{"big" if index % 2 else "small"}",{mask_name}')
            _add(tar, f"{category}/image_anno.csv", "\n".join(anno).encode())
        _add(tar, "split_csv/1cls.csv", "\n".join(split_rows).encode())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return VisaDatasetSource(archive.parent, "http://unused", digest, archive.name)


class _TaggedFeatures(np.ndarray):
    """Масив ознак із позначкою частини розбиття, з якої його взято (для перевірки витоку)."""

    part: str = ""


def _record_parts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[str, ...]]]:
    """Підміняє методи так, щоб записувати, з яких частин розбиття «вчаться» моделі."""
    from mllabs.anomaly.fae import FaeTrainer
    from mllabs.anomaly.padim import Padim
    from mllabs.anomaly.patchcore import PatchCore
    from mllabs.anomaly.workspace import VisaWorkspace

    calls: list[tuple[str, tuple[str, ...]]] = []
    original_features = VisaWorkspace.features

    def tagged_features(self, category, key, patch_size, part):
        array = original_features(self, category, key, patch_size, part).view(_TaggedFeatures)
        array.part = str(part)
        return array

    def recording(name, method):
        def wrapper(self, *arrays, **kwargs):
            calls.append((name, tuple(getattr(a, "part", "?") for a in arrays)))
            return method(self, *arrays, **kwargs)

        return wrapper

    monkeypatch.setattr(VisaWorkspace, "features", tagged_features)
    monkeypatch.setattr(FaeTrainer, "train", recording("fae", FaeTrainer.train))
    monkeypatch.setattr(Padim, "fit", recording("padim", Padim.fit))
    monkeypatch.setattr(PatchCore, "fit", recording("patchcore", PatchCore.fit))
    return calls


@pytest.fixture(scope="module")
def finished_lab(tmp_path_factory):
    project = ProjectPaths(root=tmp_path_factory.mktemp("lab4_project"))
    source = _micro_visa(project.data_raw_dir)
    context = LabContext(number=4, project=project, seed=0)
    with pytest.MonkeyPatch.context() as monkeypatch:
        fit_calls = _record_parts(monkeypatch)
        Lab4(context, _SMALL_CONFIG, source=source, weights=None).run()
    return context, fit_calls


@pytest.fixture(scope="module")
def lab_context(finished_lab):
    return finished_lab[0]


def test_lab_writes_every_planned_figure_and_table(lab_context):
    figures = {path.stem for path in lab_context.paths.figures_dir.glob("*.png")}
    tables = {path.stem for path in lab_context.paths.tables_dir.glob("*.csv")}

    assert figures == {
        "samples_grid",
        "background_variance",
        "defect_area_hist",
        "patch_grid",
        "model_diagram",
        "study_backbones",
        "study_patch_size",
        "study_regularization",
        "learning_curves_study",
        "learning_curves",
        "roc_pr_pixel",
        "anomaly_maps_test",
        "sota_benchmark",
        "accuracy_vs_cost",
    }
    expected_tables = {
        "data_overview",
        "splits",
        "background_variance",
        "defect_area",
        "transformations",
        "backbones",
        "model_architecture",
        "study_backbones",
        "study_patch_size",
        "study_regularization",
        "training_history",
        "test_metrics",
        "test_predictions",
        "sota_benchmark",
        "sota_bootstrap",
        "published_reference",
        "flops",
        "latency",
        "accuracy_vs_cost",
    }
    assert expected_tables <= tables


def test_report_tables_have_at_most_seven_columns(lab_context):
    for name in (
        "data_overview",
        "test_metrics",
        "sota_benchmark",
        "sota_bootstrap",
        "flops",
        "latency",
    ):
        table = pd.read_csv(lab_context.paths.tables_dir / f"{name}.csv")
        assert table.shape[1] <= 7, name


def test_metrics_cover_every_method_and_category(lab_context):
    metrics = json.loads(lab_context.paths.metrics_file.read_text(encoding="utf-8"))

    for method in ("FAE", "PaDiM", "PatchCore"):
        for category in (*_CATEGORIES, "mean"):
            assert 0.0 <= metrics["test"][method][category]["pixel_auroc"] <= 1.0
    assert set(metrics["studies"]) >= {"backbones", "patch_size", "regularization", "selected"}
    assert metrics["profiling"]["FAE"]["resnet50"]["latency_ms_b1_1thread"] > 0
    assert metrics["data"]["alpha"]["n_train"] == 17


def test_models_learn_only_from_normal_train_rows(finished_lab):
    context, fit_calls = finished_lab
    splits = pd.read_csv(context.project.data_processed_dir / "visa" / "splits.csv")

    assert splits.loc[splits["part"].isin(["train", "val_normal"]), "label"].sum() == 0
    assert {name for name, _ in fit_calls} == {"fae", "padim", "patchcore"}
    for name, parts in fit_calls:
        expected = ("train", "val_normal") if name == "fae" else ("train",)
        assert parts == expected, name


def test_training_history_has_comparable_losses(lab_context):
    history = pd.read_csv(lab_context.paths.tables_dir / "training_history.csv")

    assert {"loss", "train_mse", "val_mse", "val_pixel_auroc", "train_patch_p99"} <= set(
        history.columns
    )


def test_predictions_have_outcomes_for_lab5(lab_context):
    predictions = pd.read_csv(lab_context.paths.tables_dir / "test_predictions.csv")

    assert set(predictions["outcome"]) <= {"TP", "FP", "FN", "TN"}
    assert len(predictions) == 3 * 2 * 6  # методи × категорії × зображення test


def test_saved_bundle_rebuilds_detector_for_lab5(lab_context):
    config, detector = load_bundle(lab_context.paths.models_dir / "alpha", weights=None)
    image = np.zeros((1, config.input_size, config.input_size, 3), np.float32)

    outputs = detector(image)

    assert outputs["anomaly_map"].shape == (1, config.eval_size, config.eval_size)
    assert config.threshold_pixel > 0
