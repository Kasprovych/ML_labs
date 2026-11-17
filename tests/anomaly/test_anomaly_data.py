"""Тести даних VisA: розбір CSV, безпечне розпакування, SHA-256, розбиття, кеш зображень."""

import hashlib
import io
import tarfile

import numpy as np
import pytest
from PIL import Image

from mllabs.anomaly.images import ImageStore, coarse_brightness_std
from mllabs.anomaly.splits import AnomalySplitter, SplitIntegrityError, SplitPart, validate_split
from mllabs.anomaly.visa import VisaDatasetSource, VisaSample, is_safe_member
from mllabs.data import DatasetDownloadError

_SPLIT_CSV = """object,split,label,image,mask
toy,train,normal,toy/Data/Images/Normal/000.JPG,
toy,train,normal,toy/Data/Images/Normal/001.JPG,
toy,test,normal,toy/Data/Images/Normal/002.JPG,
toy,test,anomaly,toy/Data/Images/Anomaly/000.JPG,toy/Data/Masks/Anomaly/000.png
other,train,normal,other/Data/Images/Normal/000.JPG,
"""
_ANNO_CSV = """image,label,mask
toy/Data/Images/Normal/000.JPG,normal,
toy/Data/Images/Anomaly/000.JPG,"crack,scratch",toy/Data/Masks/Anomaly/000.png
"""


def _jpeg_bytes(color: int, size: tuple[int, int] = (30, 20)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (color, color, color)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _png_mask(size: tuple[int, int] = (30, 20)) -> bytes:
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[5:15, 10:20] = 3  # індекс дефекту, не 255
    buffer = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _add(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


@pytest.fixture
def visa_archive(tmp_path):
    archive = tmp_path / "visa" / "VisA.tar"
    archive.parent.mkdir()
    with tarfile.open(archive, "w") as tar:
        _add(tar, "toy/image_anno.csv", _ANNO_CSV.encode())
        for index in range(3):
            _add(tar, f"toy/Data/Images/Normal/00{index}.JPG", _jpeg_bytes(100 + index))
        _add(tar, "toy/Data/Images/Anomaly/000.JPG", _jpeg_bytes(200))
        _add(tar, "toy/Data/Masks/Anomaly/000.png", _png_mask())
        _add(tar, "other/image_anno.csv", b"image,label,mask\n")
        for index in range(4):
            _add(tar, f"other/Data/Images/Normal/00{index}.JPG", _jpeg_bytes(50))
        _add(tar, "../evil.txt", b"x")
        _add(tar, "split_csv/1cls.csv", _SPLIT_CSV.encode())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return archive, digest


def test_is_safe_member_rejects_parent_and_absolute_paths():
    assert is_safe_member("toy/Data/a.JPG")
    assert not is_safe_member("../evil.txt")
    assert not is_safe_member("/etc/passwd")
    assert not is_safe_member("toy/../../evil")


def test_extract_takes_full_categories_and_limited_sample_and_skips_unsafe(visa_archive):
    archive, digest = visa_archive
    source = VisaDatasetSource(archive.parent, "http://unused", digest, archive.name)

    source.extract(["toy"], ["other"], normal_per_sample_category=2)

    assert len(source.normal_images("toy")) == 3
    assert len(source.normal_images("other")) == 2
    assert not (archive.parent.parent / "evil.txt").exists()


def test_read_samples_joins_split_with_defect_types(visa_archive):
    archive, digest = visa_archive
    source = VisaDatasetSource(archive.parent, "http://unused", digest, archive.name)
    source.extract(["toy"], [], 0)

    samples = source.read_samples("toy")

    assert len(samples) == 4
    anomaly = next(sample for sample in samples if sample.label == 1)
    assert anomaly.defect_types == "crack,scratch"
    assert anomaly.mask.endswith("000.png")
    assert all(sample.defect_types == "" for sample in samples if sample.label == 0)


def test_bad_checksum_raises_download_error(visa_archive):
    archive, _ = visa_archive
    source = VisaDatasetSource(archive.parent, "http://unused", "0" * 64, archive.name)

    with pytest.raises(DatasetDownloadError):
        source.ensure_archive()


def _samples(n_train: int, n_test_normal: int, n_test_anomalous: int) -> list[VisaSample]:
    samples = [VisaSample("c", "train", 0, f"tr{i}", "", "") for i in range(n_train)]
    samples += [VisaSample("c", "test", 0, f"tn{i}", "", "") for i in range(n_test_normal)]
    samples += [
        VisaSample("c", "test", 1, f"ta{i}", f"m{i}", "crack") for i in range(n_test_anomalous)
    ]
    return samples


def test_splitter_is_deterministic_disjoint_and_stratified():
    samples = _samples(100, 40, 40)
    first = AnomalySplitter(0.85, 0.5, random_state=3).split(samples)
    second = AnomalySplitter(0.85, 0.5, random_state=3).split(samples)

    assert first.equals(second)
    assert not first["image"].duplicated().any()
    counts = first.groupby(["part", "label"]).size()
    assert counts[(SplitPart.TRAIN, 0)] == 85
    assert counts[(SplitPart.VAL_NORMAL, 0)] == 15
    assert counts[(SplitPart.VAL, 1)] == counts[(SplitPart.TEST, 1)] == 20
    assert first.loc[first["part"].isin(["train", "val_normal"]), "label"].sum() == 0


def test_validate_split_rejects_anomaly_in_train():
    frame = AnomalySplitter(0.85, 0.5, 0).split(_samples(20, 4, 4))
    broken = frame.assign(label=np.where(frame["part"] == "train", 1, frame["label"]))

    with pytest.raises(SplitIntegrityError):
        validate_split(broken)


def test_image_store_caches_resized_images_and_binarized_masks(visa_archive, tmp_path):
    archive, digest = visa_archive
    source = VisaDatasetSource(archive.parent, "http://unused", digest, archive.name)
    source.extract(["toy"], [], 0)
    store = ImageStore(archive.parent, tmp_path / "cache")
    images = ["toy/Data/Images/Normal/000.JPG", "toy/Data/Images/Anomaly/000.JPG"]
    masks = ["", "toy/Data/Masks/Anomaly/000.png"]

    batch = store.load("toy", images, masks, size=16)
    again = store.load("toy", images, masks, size=16)

    assert batch.images.shape == (2, 16, 16, 3) and batch.images.dtype == np.uint8
    assert batch.masks.dtype == bool
    assert not batch.masks[0].any() and batch.masks[1].any()
    np.testing.assert_array_equal(np.asarray(batch.images), np.asarray(again.images))


def test_coarse_brightness_std_is_zero_for_flat_image_and_positive_for_gradient():
    flat = np.full((1, 16, 16, 3), 120, dtype=np.uint8)
    gradient = np.repeat(np.linspace(0, 255, 16, dtype=np.uint8)[None, :], 16, axis=0)
    gradient = np.stack([gradient] * 3, axis=-1)[None]

    assert coarse_brightness_std(flat, 4)[0] == pytest.approx(0.0, abs=1e-4)
    assert coarse_brightness_std(gradient, 4)[0] > 10


def _workspace_archive(root):
    """Мікро-VisA: 4 норми train, 2 норми й 2 аномалії test (для стратифікації)."""
    archive = root / "visa" / "ws.tar"
    archive.parent.mkdir(parents=True)
    rows = ["object,split,label,image,mask"]
    anno = ["image,label,mask"]
    with tarfile.open(archive, "w") as tar:
        for index in range(6):
            name = f"toy/Data/Images/Normal/{index:03d}.JPG"
            _add(tar, name, _jpeg_bytes(90 + index))
            rows.append(f"toy,{'train' if index < 4 else 'test'},normal,{name},")
        for index in range(2):
            name, mask = (
                f"toy/Data/Images/Anomaly/{index:03d}.JPG",
                f"toy/Data/Masks/Anomaly/{index:03d}.png",
            )
            _add(tar, name, _jpeg_bytes(200))
            _add(tar, mask, _png_mask())
            rows.append(f"toy,test,anomaly,{name},{mask}")
            anno.append(f"{name},crack,{mask}")
        _add(tar, "toy/image_anno.csv", "\n".join(anno).encode())
        _add(tar, "split_csv/1cls.csv", "\n".join(rows).encode())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return VisaDatasetSource(archive.parent, "http://unused", digest, archive.name)


def test_workspace_splits_and_returns_features_only_for_requested_part(tmp_path):
    from mllabs.anomaly.backbones import BackboneKind
    from mllabs.anomaly.splits import SplitPart
    from mllabs.anomaly.workspace import FeatureKey, VisaWorkspace, WorkspaceSettings
    from mllabs.core import ProjectPaths

    source = _workspace_archive(tmp_path / "raw")
    settings = WorkspaceSettings(
        categories=("toy",), train_fraction=0.5, image_size=32, eval_size=32, feature_dim=8
    )
    workspace = VisaWorkspace(ProjectPaths(root=tmp_path), settings, source, weights=None)

    splits = workspace.prepare(seed=0)
    key = FeatureKey(BackboneKind.MOBILENET_V2, 1)
    train = workspace.features("toy", key, 1, SplitPart.TRAIN)

    assert (tmp_path / "data" / "processed" / "visa" / "splits.csv").is_file()
    assert len(train) == int((splits["part"] == "train").sum()) == 2
    assert train.shape[1:] == (4, 4, 8)
    assert sorted(workspace.labels("toy", SplitPart.TEST).tolist()) == [0, 1]
