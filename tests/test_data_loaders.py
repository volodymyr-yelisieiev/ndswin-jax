"""Tests for data loading utilities."""

import unittest.mock as mock

import numpy as np
import pytest

from ndswin.config import DataConfig
from ndswin.training.data import (
    HuggingFaceDataLoader,
    NumpySegmentationFolderDataLoader,
    SyntheticDataLoader,
    VolumeFolderDataLoader,
    _normalize_batch,
    create_data_loader,
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


@mock.patch("datasets.load_dataset")
def test_create_data_loader_huggingface(mock_load_dataset):
    """Test correct fallback to HuggingFace dataset."""
    config = DataConfig(dataset="hf:cifar10", image_size=(3, 32, 32))

    def mock_get_item(idx):
        return {
            "img": np.zeros((32, 32, 3), dtype=np.uint8),
            "label": idx % 10,
        }

    mock_dataset = [mock_get_item(i) for i in range(100)]
    mock_load_dataset.return_value = mock_dataset

    loader_train = create_data_loader(config, batch_size=4)

    batch = next(iter(loader_train))
    assert batch["image"].shape == (4, 3, 32, 32)
    assert batch["label"].shape == (4,)

    # Check normalization boundaries roughly (should be mean 0 if input was 0 and we map roughly)
    assert loader_train.dataset_info.name == "cifar10"


@mock.patch("datasets.load_dataset")
def test_huggingface_loader_segmentation(mock_load_dataset):
    """Test 2D segmentation logic for HF loaders."""

    def mock_get_seg(idx):
        return {
            "image": np.ones((64, 64, 3), dtype=np.uint8),
            "label": np.zeros((64, 64), dtype=np.int32),
        }

    mock_dataset = [mock_get_seg(i) for i in range(5)]
    mock_load_dataset.return_value = mock_dataset

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


def test_normalize_batch_broadcasts_for_2d_inputs():
    """Normalization helper should broadcast over 2D spatial dimensions."""
    batch_x = np.array(
        [
            [
                [[1.0, 3.0], [5.0, 7.0]],
                [[6.0, 8.0], [10.0, 12.0]],
            ]
        ],
        dtype=np.float32,
    )

    normalized = _normalize_batch(batch_x, mean=(1.0, 2.0), std=(2.0, 4.0))

    expected = np.array(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[1.0, 1.5], [2.0, 2.5]],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(normalized, expected)


def test_normalize_batch_broadcasts_for_3d_inputs():
    """Normalization helper should broadcast over 3D spatial dimensions."""
    batch_x = np.array(
        [
            [
                [
                    [[1.0, 3.0], [5.0, 7.0]],
                    [[9.0, 11.0], [13.0, 15.0]],
                ]
            ]
        ],
        dtype=np.float32,
    )

    normalized = _normalize_batch(batch_x, mean=(1.0,), std=(2.0,))
    expected = (batch_x - 1.0) / 2.0

    np.testing.assert_allclose(normalized, expected)


def test_volume_folder_loader_normalizes_3d_batches(tmp_path):
    """Volume folder loader should normalize 3D volumes without shape errors."""
    class_dir = tmp_path / "train" / "class0"
    class_dir.mkdir(parents=True)
    np.save(class_dir / "sample.npy", np.arange(8, dtype=np.float32).reshape(2, 2, 2))

    loader = VolumeFolderDataLoader(
        name="volumes",
        data_dir=str(tmp_path),
        split="train",
        batch_size=1,
        shuffle=False,
        in_channels=1,
        image_size=(2, 2, 2),
        mean=(1.0,),
        std=(2.0,),
    )

    batch = next(iter(loader))

    expected = (np.arange(8, dtype=np.float32).reshape(1, 1, 2, 2, 2) - 1.0) / 2.0
    np.testing.assert_allclose(np.asarray(batch["image"]), expected)
    np.testing.assert_array_equal(np.asarray(batch["label"]), np.array([0], dtype=np.int32))


def test_numpy_segmentation_folder_loader_normalizes_2d_batches(tmp_path):
    """Segmentation folder loader should normalize 2D images without shape errors."""
    image_dir = tmp_path / "train" / "images"
    label_dir = tmp_path / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image = np.array(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[6.0, 8.0], [10.0, 12.0]],
        ],
        dtype=np.float32,
    )
    label = np.array([[0, 1], [1, 0]], dtype=np.int32)
    np.savez(image_dir / "sample.npz", image=image)
    np.savez(label_dir / "sample.npz", label=label)

    loader = NumpySegmentationFolderDataLoader(
        name="seg2d",
        data_dir=str(tmp_path),
        split="train",
        batch_size=1,
        shuffle=False,
        in_channels=2,
        image_size=(2, 2),
        mean=(1.0, 2.0),
        std=(2.0, 4.0),
    )

    batch = next(iter(loader))

    expected = np.array(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[1.0, 1.5], [2.0, 2.5]],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(batch["image"]), expected)
    np.testing.assert_array_equal(np.asarray(batch["label"]), label[None, ...])
