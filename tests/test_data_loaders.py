"""Tests for data loading utilities."""

import logging
import sys
import types
import unittest.mock as mock

import numpy as np
import pytest

from ndswin.config import DataConfig
from ndswin.training.data import (
    HuggingFaceDataLoader,
    SyntheticDataLoader,
    VolumeFolderDataLoader,
    create_data_loader,
    normalize_split,
)


def test_synthetic_data_loader():
    """Test functionality of the synthetic 2D loader."""
    DataConfig(dataset="synthetic", image_size=(3, 32, 32), task="classification")

    loader = SyntheticDataLoader(
        num_samples=16,
        input_shape=(3, 32, 32),
        num_classes=1000,
        batch_size=4,
    )

    assert loader.dataset_info.num_classes == 1000
    assert loader.dataset_info.num_train == 16

    batches = list(iter(loader))
    assert len(batches) == 4  # 16 / 4 = 4 batches

    batch = batches[0]
    assert "image" in batch
    assert "label" in batch
    assert batch["image"].shape == (4, 3, 32, 32)
    assert batch["label"].shape == (4,)

    # Test segmentation mode
    loader_seg = SyntheticDataLoader(
        num_samples=4,
        input_shape=(1, 16, 16),
        num_classes=2,
        batch_size=2,
    )
    assert loader_seg.dataset_info.num_classes == 2
    seg_batch = next(iter(loader_seg))
    assert seg_batch["label"].shape == (2,)


def test_synthetic_data_loader_3d():
    """Test synthetic data loader with 3D images."""
    loader = SyntheticDataLoader(
        num_samples=4,
        input_shape=(1, 8, 8, 8),
        num_classes=2,
        batch_size=2,
    )
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 1, 8, 8, 8)
    assert batch["label"].shape == (2,)


@mock.patch("ndswin.training.data.SyntheticDataLoader")
def test_create_data_loader_synthetic(mock_loader):
    """Test create_data_loader delegates to Synthetic correctly."""
    config = DataConfig(dataset="synthetic", image_size=(3, 16, 16))
    create_data_loader(config, batch_size=4)
    mock_loader.assert_called_once()


def test_create_data_loader_huggingface(monkeypatch):
    """Test correct fallback to HuggingFace dataset."""
    import types as _types

    config = DataConfig(dataset="hf:cifar10", image_size=(3, 32, 32))

    def _mock_load_dataset(hf_id, split, data_dir=None):
        return [
            {
                "img": np.zeros((32, 32, 3), dtype=np.uint8),
                "label": i % 10,
            }
            for i in range(100)
        ]

    fake_datasets = _types.SimpleNamespace(
        load_dataset=_mock_load_dataset,
        get_dataset_split_names=lambda hf_id, data_dir=None: ["train"],
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    loader_train = create_data_loader(config, batch_size=4)

    batch = next(iter(loader_train))
    assert batch["image"].shape == (4, 3, 32, 32)
    assert batch["label"].shape == (4,)

    # Check normalization boundaries roughly (should be mean 0 if input was 0 and we map roughly)
    assert loader_train.dataset_info.name == "cifar10"


def test_huggingface_loader_segmentation(monkeypatch):
    """Test 2D segmentation logic for HF loaders."""
    import types as _types

    def _mock_get_seg(idx):
        return {
            "image": np.ones((64, 64, 3), dtype=np.uint8),
            "label": np.zeros((64, 64), dtype=np.int32),
        }

    mock_dataset = [_mock_get_seg(i) for i in range(5)]

    fake_datasets = _types.SimpleNamespace(
        load_dataset=lambda *a, **kw: mock_dataset,
        get_dataset_split_names=lambda hf_id, data_dir=None: ["train"],
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    loader = HuggingFaceDataLoader(
        hf_id="some_seg_dataset", split="train", batch_size=2, image_size=(3, 64, 64)
    )

    # Will do feature finding via index 0
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["label"].shape == (2, 64, 64)


def test_missing_dataset_library():
    """Verify data loader falls back elegantly if datasets is uninstalled."""
    import sys

    # Temporarily hide datasets
    old_datasets = sys.modules.get("datasets")
    sys.modules["datasets"] = None

    try:
        with pytest.raises(RuntimeError, match="install 'datasets'"):
            HuggingFaceDataLoader(hf_id="test", split="train", batch_size=2)
    finally:
        if old_datasets is not None:
            sys.modules["datasets"] = old_datasets
        else:
            del sys.modules["datasets"]


def test_normalize_split_aliases():
    """Split aliases should normalize in one place."""
    assert normalize_split("train") == "train"
    assert normalize_split("val") == "validation"
    assert normalize_split("validation") == "validation"
    assert normalize_split("test") == "test"

    with pytest.raises(ValueError, match="Unsupported split"):
        normalize_split("dev")


def _install_fake_datasets_module(monkeypatch, split_names):
    calls = []

    def fake_load_dataset(hf_id, split, data_dir=None):
        calls.append({"hf_id": hf_id, "split": split, "data_dir": data_dir})
        return [
            {
                "image": np.zeros((8, 8, 3), dtype=np.uint8),
                "label": idx % 2,
            }
            for idx in range(4)
        ]

    fake_module = types.SimpleNamespace(
        load_dataset=fake_load_dataset,
        get_dataset_split_names=lambda hf_id, data_dir=None: list(split_names),
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    return calls


@pytest.mark.parametrize("requested_split", ["validation", "val"])
def test_huggingface_loader_routes_validation_aliases(monkeypatch, requested_split):
    """HF loaders should keep validation requests as validation when available."""
    calls = _install_fake_datasets_module(monkeypatch, ["train", "validation", "test"])

    loader = HuggingFaceDataLoader(
        hf_id="dummy/dataset",
        split=requested_split,
        batch_size=2,
        image_size=(3, 8, 8),
    )

    assert loader.requested_split == "validation"
    assert loader.split == "validation"
    assert calls[-1]["split"] == "validation"


def test_huggingface_loader_falls_back_from_validation_with_warning(monkeypatch, caplog):
    """HF validation requests should fall back explicitly when validation is absent."""
    calls = _install_fake_datasets_module(monkeypatch, ["train", "val", "test"])

    with caplog.at_level(logging.WARNING):
        loader = HuggingFaceDataLoader(
            hf_id="dummy/dataset",
            split="validation",
            batch_size=2,
            image_size=(3, 8, 8),
        )

    assert loader.split == "val"
    assert calls[-1]["split"] == "val"
    assert "using 'val' instead" in caplog.text


def test_create_data_loader_hf_routes_test_split(monkeypatch):
    """create_data_loader should stop remapping evaluation splits for HF datasets."""
    calls = _install_fake_datasets_module(monkeypatch, ["train", "validation", "test"])
    config = DataConfig(dataset="hf:demo", image_size=(3, 8, 8))

    loader = create_data_loader(config, split="test", batch_size=2)

    assert isinstance(loader, HuggingFaceDataLoader)
    assert loader.split == "test"
    assert calls[-1]["split"] == "test"


def _make_volume_class_dir(base, split_name, class_name="class0"):
    class_dir = base / split_name / class_name
    class_dir.mkdir(parents=True)
    np.save(class_dir / "sample.npy", np.zeros((4, 4, 4), dtype=np.float32))


@pytest.mark.parametrize("folder_name", ["validation", "val"])
def test_volume_folder_loader_supports_validation_layouts(tmp_path, folder_name):
    """Volume folder loader should support validation/ and val/ layouts."""
    _make_volume_class_dir(tmp_path, "train")
    _make_volume_class_dir(tmp_path, folder_name)

    loader = VolumeFolderDataLoader(
        name="volume",
        data_dir=str(tmp_path),
        split="validation",
        batch_size=1,
        in_channels=1,
        image_size=(4, 4, 4),
        mean=(0.0,),
        std=(1.0,),
    )

    assert loader.requested_split == "validation"
    assert loader.split == folder_name


def test_volume_folder_loader_validation_fallback_is_explicit(tmp_path, monkeypatch, caplog):
    """Validation fallback for folder datasets should be opt-in and logged."""
    _make_volume_class_dir(tmp_path, "train")
    _make_volume_class_dir(tmp_path, "test")

    config = DataConfig(
        dataset="volume",
        data_dir=str(tmp_path),
        image_size=(4, 4, 4),
        in_channels=1,
        mean=(0.0,),
        std=(1.0,),
    )

    with pytest.raises(FileNotFoundError, match="ALLOW_VALIDATION_SPLIT_FALLBACK=1"):
        create_data_loader(config, split="validation", batch_size=1)

    monkeypatch.setenv("ALLOW_VALIDATION_SPLIT_FALLBACK", "1")
    with caplog.at_level(logging.WARNING):
        loader = create_data_loader(config, split="validation", batch_size=1)

    assert isinstance(loader, VolumeFolderDataLoader)
    assert loader.split == "test"
    assert "falling back to 'test'" in caplog.text
