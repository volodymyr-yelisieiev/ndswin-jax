"""Data loading utilities for NDSwin-JAX.

This module provides data loading functionality for various datasets
and dimensionalities.
"""

import math
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ndswin.config import DataConfig
from ndswin.types import Batch
from ndswin.utils.logging import get_logger

logger = get_logger("data")


def normalize_split(split: str) -> str:
    """Normalize supported split names to canonical values.

    Args:
        split: Requested split name.

    Returns:
        Canonical split name: ``train``, ``validation``, or ``test``.

    Raises:
        ValueError: If the split name is unsupported.
    """
    normalized = split.strip().lower()
    if normalized == "train":
        return "train"
    if normalized in {"val", "validation"}:
        return "validation"
    if normalized == "test":
        return "test"
    raise ValueError(
        f"Unsupported split '{split}'. Supported splits: train, val, validation, test."
    )


def _is_enabled_env(var_name: str) -> bool:
    return os.getenv(var_name, "0").lower() in {"1", "true", "yes"}


def resolve_folder_split(
    data_dir: str,
    split: str,
    *,
    source_name: str,
    allow_validation_fallback: bool | None = None,
    allow_test_fallback: bool | None = None,
) -> str:
    """Resolve a canonical split to an on-disk folder name.

    Supported exact layouts:
      - train/
      - validation/
      - val/
      - test/

    Validation fallback to ``test`` or ``train`` is opt-in via
    ``ALLOW_VALIDATION_SPLIT_FALLBACK``. Test fallback to ``train`` remains
    opt-in via ``ALLOW_TEST_SPLIT_FALLBACK``.
    """
    normalized_split = normalize_split(split)
    available_dirs = (
        {path.name for path in Path(data_dir).iterdir() if path.is_dir()}
        if Path(data_dir).exists()
        else set()
    )

    if normalized_split == "train":
        if "train" in available_dirs:
            return "train"
        raise FileNotFoundError(f"{source_name}: train split directory not found under {data_dir}.")

    if normalized_split == "validation":
        if "validation" in available_dirs:
            return "validation"
        if "val" in available_dirs:
            logger.info(
                "%s: using 'val' directory as validation split under %s.",
                source_name,
                data_dir,
            )
            return "val"

        allow_validation_fallback = (
            _is_enabled_env("ALLOW_VALIDATION_SPLIT_FALLBACK")
            if allow_validation_fallback is None
            else allow_validation_fallback
        )
        if allow_validation_fallback:
            for fallback in ("test", "train"):
                if fallback in available_dirs:
                    logger.warning(
                        "%s: requested validation split, but no validation/ or val/ directory exists under %s; "
                        "falling back to '%s' because ALLOW_VALIDATION_SPLIT_FALLBACK is enabled.",
                        source_name,
                        data_dir,
                        fallback,
                    )
                    return fallback

        raise FileNotFoundError(
            f"{source_name}: validation split requested, but neither validation/ nor val/ exists under {data_dir}. "
            "To allow fallback to test/ or train/, set ALLOW_VALIDATION_SPLIT_FALLBACK=1."
        )

    if "test" in available_dirs:
        return "test"

    allow_test_fallback = (
        _is_enabled_env("ALLOW_TEST_SPLIT_FALLBACK")
        if allow_test_fallback is None
        else allow_test_fallback
    )
    if allow_test_fallback and "train" in available_dirs:
        logger.warning(
            "%s: requested test split, but no test/ directory exists under %s; "
            "falling back to 'train' because ALLOW_TEST_SPLIT_FALLBACK is enabled.",
            source_name,
            data_dir,
        )
        return "train"

    raise FileNotFoundError(
        f"{source_name}: test split requested, but test/ does not exist under {data_dir}. "
        "To allow fallback to train/, set ALLOW_TEST_SPLIT_FALLBACK=1."
    )


def resolve_hf_split(hf_id: str, split: str, *, data_dir: str | None = None) -> str:
    """Resolve a canonical split to an available Hugging Face dataset split."""
    normalized_split = normalize_split(split)

    try:
        from datasets import get_dataset_split_names
    except Exception as e:  # pragma: no cover - runtime optional
        raise RuntimeError(
            "To use Hugging Face datasets install 'datasets' (pip install datasets)"
        ) from e

    available_splits = set(get_dataset_split_names(hf_id, data_dir=data_dir) or [])

    if normalized_split == "train":
        if "train" in available_splits:
            return "train"
        raise ValueError(
            f"Hugging Face dataset '{hf_id}' does not provide a train split. "
            f"Available splits: {sorted(available_splits)}"
        )

    if normalized_split == "validation":
        if "validation" in available_splits:
            return "validation"
        if "val" in available_splits:
            logger.warning(
                "Hugging Face dataset '%s' does not provide a 'validation' split; using 'val' instead.",
                hf_id,
            )
            return "val"
        for fallback in ("test", "train"):
            if fallback in available_splits:
                logger.warning(
                    "Hugging Face dataset '%s' does not provide a validation split; using '%s' instead.",
                    hf_id,
                    fallback,
                )
                return fallback
        raise ValueError(
            f"Hugging Face dataset '{hf_id}' does not provide a validation split. "
            f"Available splits: {sorted(available_splits)}"
        )

    if "test" in available_splits:
        return "test"
    if "train" in available_splits and _is_enabled_env("ALLOW_TEST_SPLIT_FALLBACK"):
        logger.warning(
            "Hugging Face dataset '%s' does not provide a test split; using 'train' because "
            "ALLOW_TEST_SPLIT_FALLBACK is enabled.",
            hf_id,
        )
        return "train"

    raise ValueError(
        f"Hugging Face dataset '{hf_id}' does not provide a test split. "
        f"Available splits: {sorted(available_splits)}"
    )


@dataclass
class DatasetInfo:
    """Information about a dataset.

    Attributes:
        name: Dataset name.
        num_classes: Number of classes.
        num_train: Number of training samples.
        num_val: Number of validation samples.
        num_test: Number of test samples.
        input_shape: Shape of a single input sample.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.
    """

    name: str
    num_classes: int
    num_train: int
    num_val: int
    num_test: int
    input_shape: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    def __init__(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        """Initialize data loader."""
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @abstractmethod
    def __len__(self) -> int:
        """Return number of batches."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Batch]:
        """Iterate over batches."""
        pass

    @property
    @abstractmethod
    def dataset_info(self) -> DatasetInfo:
        """Return dataset information."""
        pass

    def reset(self) -> None:
        """Reset the data loader state."""
        self._rng = np.random.RandomState(self.seed)


def _reshape_channel_stats(
    stats: tuple[float, ...],
    num_channels: int,
    num_spatial_dims: int,
) -> np.ndarray:
    """Reshape per-channel stats for broadcasting over N-D spatial inputs."""
    stats_array = np.asarray(stats, dtype=np.float32).reshape(-1)
    if stats_array.size == 1 and num_channels != 1:
        stats_array = np.repeat(stats_array, num_channels)
    elif stats_array.size != num_channels:
        raise ValueError(
            f"Expected normalization stats with 1 or {num_channels} values, got {stats_array.size}"
        )
    return stats_array.reshape((1, num_channels) + (1,) * num_spatial_dims)


def _normalize_batch(
    batch_x: np.ndarray,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> np.ndarray:
    """Normalize a batch with shape (B, C, *spatial_dims)."""
    if batch_x.ndim < 3:
        raise ValueError(f"Expected batch with shape (B, C, *spatial_dims), got {batch_x.shape}")

    num_channels = batch_x.shape[1]
    num_spatial_dims = batch_x.ndim - 2
    mean_array = _reshape_channel_stats(mean, num_channels, num_spatial_dims)
    std_array = _reshape_channel_stats(std, num_channels, num_spatial_dims)
    normalized = (batch_x - mean_array) / std_array
    return np.asarray(normalized, dtype=batch_x.dtype)


class CIFARDataLoader(DataLoader):
    """Base class for CIFAR data loaders."""

    def __init__(
        self,
        name: str,
        data_dir: str,
        split: str,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        transform: Callable | None = None,
        download: bool = True,
        seed: int = 42,
        mean: tuple[float, ...] = (0.5, 0.5, 0.5),
        std: tuple[float, ...] = (0.5, 0.5, 0.5),
        num_classes: int = 10,
    ) -> None:
        super().__init__(batch_size, shuffle, drop_last, 0, seed)
        self.name = name
        self.data_dir = data_dir
        self.split = normalize_split(split)
        self.transform = transform
        self.download = download
        self.mean = mean
        self.std = std
        self.num_classes = num_classes

        self.x, self.y = self._load_data()

        # Implement 90/10 train/val split for CIFAR if needed
        if self.split in ["train", "validation"]:
            val_size = 5000 if self.num_classes == 10 else 5000  # 10% of 50k
            if self.split == "validation":
                self.x = self.x[-val_size:]
                self.y = self.y[-val_size:]
            else:
                self.x = self.x[:-val_size]
                self.y = self.y[:-val_size]

        self.num_samples = len(self.x)

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load CIFAR data using Hugging Face `datasets` or `tensorflow_datasets`.

        This loader prefers the `datasets` library (recommended for HF-first workflows).
        If `datasets` is not available, falls back to `tensorflow_datasets`.
        """
        ds_name = "cifar10" if self.num_classes == 10 else "cifar100"
        try:
            # Prefer Hugging Face datasets
            from datasets import load_dataset

            ds = load_dataset(ds_name, split=self.split)
            images, labels = [], []
            for item in ds:
                # Support multiple image/label field names used by datasets
                img = None
                for _k in ("image", "images", "img", "image0", "pixel_values"):
                    if _k in item:
                        img = item[_k]
                        break
                lab = None
                for _k in ("label", "labels", "fine_label", "coarse_label", "target"):
                    if _k in item:
                        lab = item[_k]
                        break

                if img is None or lab is None:
                    raise ValueError(f"Unexpected dataset format for {ds_name}")

                arr = np.array(img).astype(np.float32) / 255.0
                # Ensure channels-first (C, H, W)
                if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                    arr = arr.transpose(2, 0, 1)
                images.append(arr)
                labels.append(int(lab))
            return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
        except ImportError:
            try:
                import tensorflow_datasets as tfds

                ds = tfds.load(
                    ds_name, split=self.split, as_supervised=True, data_dir=self.data_dir
                )
                images, labels = [], []
                for img, label in ds:
                    img = np.transpose(img.numpy(), (2, 0, 1)) / 255.0
                    images.append(img.astype(np.float32))
                    labels.append(int(label.numpy()))
                return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)
            except ImportError:
                raise ImportError(
                    f"Loading {self.name} requires either 'datasets' (huggingface) or 'tensorflow-datasets'. "
                    "Install with: pip install datasets  or pip install tensorflow-datasets"
                )

    def _augment_batch(self, batch_x: np.ndarray) -> np.ndarray:
        """Apply vectorized batch augmentation efficiently using numpy.

        Replaces the per-sample for-loop with a single operation over
        the whole batch, eliminating repeated JAX JIT dispatch overhead.

        Args:
            batch_x: Float32 array of shape (B, C, H, W), already normalized.

        Returns:
            Augmented batch of same shape.
        """
        if self.transform is None:
            return batch_x

        B, C, H, W = batch_x.shape

        # --- Vectorized RandomCrop with padding=4 ---
        if getattr(getattr(self, "_do_random_crop", None), "__bool__", lambda: True)():
            pad = 4
            # Pad spatial dims: (B, C, H+2p, W+2p)
            padded = np.pad(batch_x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
            PH, PW = padded.shape[2], padded.shape[3]

            # Sample per-image crop offsets
            top = self._rng.randint(0, PH - H + 1, size=B)
            left = self._rng.randint(0, PW - W + 1, size=B)

            # Gather crops vectorized
            cropped = np.stack(
                [padded[i, :, top[i] : top[i] + H, left[i] : left[i] + W] for i in range(B)], axis=0
            )
            batch_x = cropped

        # --- Vectorized RandomHorizontalFlip ---
        flip_mask = self._rng.random(B) < 0.5  # (B,)
        if np.any(flip_mask):
            batch_x = batch_x.copy()
            batch_x[flip_mask] = batch_x[flip_mask, :, :, ::-1]

        return batch_x

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and end - start < self.batch_size:
                break

            batch_indices = indices[start:end]
            batch_x = self.x[batch_indices].astype(np.float32)
            batch_y = self.y[batch_indices]

            # Normalize
            batch_x = _normalize_batch(batch_x, self.mean, self.std)

            # Vectorized batch augmentation (numpy-based, no per-sample Python loop)
            batch_x = self._augment_batch(batch_x)
            yield {
                "image": jnp.array(batch_x),
                "label": jnp.array(batch_y),
            }


class CIFAR10DataLoader(CIFARDataLoader):
    """CIFAR-10 data loader."""

    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("mean", self.MEAN)
        kwargs.setdefault("std", self.STD)
        kwargs.setdefault("num_classes", 10)
        kwargs.setdefault("name", "CIFAR-10")
        super().__init__(**kwargs)

    @property
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="cifar10",
            num_classes=10,
            num_train=45000,
            num_val=5000,
            num_test=10000,
            input_shape=(3, 32, 32),
            mean=self.MEAN,
            std=self.STD,
        )


class CIFAR100DataLoader(CIFARDataLoader):
    """CIFAR-100 data loader."""

    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("mean", self.MEAN)
        kwargs.setdefault("std", self.STD)
        kwargs.setdefault("num_classes", 100)
        kwargs.setdefault("name", "CIFAR-100")
        super().__init__(**kwargs)

    @property
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="cifar100",
            num_classes=100,
            num_train=45000,
            num_val=5000,
            num_test=10000,
            input_shape=(3, 32, 32),
            mean=self.MEAN,
            std=self.STD,
        )


class VolumeFolderDataLoader(DataLoader):
    """Generic 3D volume folder data loader.

    Expects on-disk layout:
        data_dir/
            train/
                class0/
                    sample_a.npy
                    sample_b.npy
                class1/
                    sample_c.npy
            test/
                ...

    Each sample file should contain a numpy array of shape (D, H, W) or
    (C, D, H, W) where C equals `in_channels`. Normalization is applied per
    channel using `mean` and `std` from configuration.
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        transform: Callable | None = None,
        seed: int = 42,
        in_channels: int = 1,
        image_size: tuple[int, ...] = (8, 32, 32),
        mean: tuple[float, ...] = (0.5,),
        std: tuple[float, ...] = (0.5,),
        file_exts: tuple[str, ...] = (".npy", ".npz"),
        pad_to: tuple[int, ...] | None = None,
    ) -> None:
        """pad_to: optional minimum spatial size (per-dimension) to pad/crop samples to.

        If provided, the loader will pad (zero) or center-crop samples to meet these
        minimum spatial dimensions, which is useful to ensure compatibility with
        model patching/merging requirements.
        """
        super().__init__(batch_size, shuffle, drop_last, 0, seed)
        self.name = name
        self.data_dir = data_dir
        self.requested_split = normalize_split(split)
        self.split = resolve_folder_split(
            data_dir,
            self.requested_split,
            source_name=f"{self.__class__.__name__}({name})",
        )
        self.transform = transform
        self.in_channels = in_channels
        self.image_size = tuple(image_size)
        self.mean = mean
        self.std = std
        self.file_exts = file_exts
        self.pad_to = tuple(pad_to) if pad_to is not None else None

        self.file_paths: list[str] = []
        self.labels: list[int] = []
        self.class_to_idx: dict[str, int] = {}

        self._scan_files()
        self.num_samples = len(self.file_paths)

    def _scan_files(self) -> None:
        import os

        split_dir = os.path.join(self.data_dir, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        classes = [
            d for d in sorted(os.listdir(split_dir)) if os.path.isdir(os.path.join(split_dir, d))
        ]
        if not classes:
            raise ValueError(f"No class subdirectories found in {split_dir}")

        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if any(fname.endswith(ext) for ext in self.file_exts):
                    self.file_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def _load_file(self, path: str) -> np.ndarray:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # pick the first array in the archive
            arr = arr[list(arr.files)[0]]

        arr = np.array(arr)

        # Ensure shape: (C, D, H, W)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, 0)
        elif arr.ndim == 4:
            # assume (C, D, H, W)
            pass
        else:
            raise ValueError(f"Unsupported array shape {arr.shape} for volume file: {path}")

        # Validate channels
        if arr.shape[0] != self.in_channels:
            raise ValueError(
                f"File {path} has {arr.shape[0]} channels but expected {self.in_channels}"
            )

        # If sample spatial differs from expected image_size, allow resize by pad/crop
        spatial = arr.shape[1:]
        target_spatial = tuple(self.image_size)

        # If pad_to is specified, ensure final target is at least pad_to
        if self.pad_to is not None:
            if len(self.pad_to) != len(target_spatial):
                raise ValueError("pad_to must match number of spatial dims")
            target_spatial = tuple(max(ts, p) for ts, p in zip(target_spatial, self.pad_to))

        # If arr is larger than target -> center-crop; if smaller -> pad with zeros
        new_arr = arr
        if spatial != target_spatial:
            # Compute cropping or padding per axis
            pads = []  # list of (before, after) per spatial dim
            crops = []  # list of (start, end) per spatial dim
            for cur, tgt in zip(spatial, target_spatial):
                if cur < tgt:
                    total = tgt - cur
                    before = total // 2
                    after = total - before
                    pads.append((before, after))
                    crops.append((0, cur))
                else:
                    total = cur - tgt
                    start = total // 2
                    end = start + tgt
                    pads.append((0, 0))
                    crops.append((start, end))

            # Apply cropping first
            crop_slices = [slice(None)] + [slice(s, e) for s, e in crops]
            new_arr = new_arr[tuple(crop_slices)]

            # Apply padding
            pad_width = [(0, 0)] + pads
            new_arr = np.pad(new_arr, pad_width, mode="constant", constant_values=0)

        return new_arr.astype(np.float32)

    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and end - start < self.batch_size:
                break

            batch_indices = indices[start:end]

            batch_x = np.stack([self._load_file(self.file_paths[i]) for i in batch_indices])
            batch_y = np.array([self.labels[i] for i in batch_indices])

            batch_x = _normalize_batch(batch_x, self.mean, self.std)

            if self.transform is not None:
                # transform should accept (numpy array, rng) similar to CIFAR transforms
                batch_x = self.transform(batch_x, self._rng.randint(0, 2**31))

            yield {
                "image": jnp.array(batch_x),
                "label": jnp.array(batch_y),
            }

    @property
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name,
            num_classes=len(self.class_to_idx),
            num_train=0,
            num_val=0,
            num_test=0,
            input_shape=(self.in_channels,) + self.image_size,
            mean=self.mean,
            std=self.std,
        )


class NumpySegmentationFolderDataLoader(DataLoader):
    """Segmentation loader that reads paired NPZ files saved by the fetch script.

    Expects layout:
        data_dir/
            train/ images/, labels/
            validation/ images/, labels/
            test/ images/, labels/
    Each image npz contains 'image' with shape (C, D, H, W). Each label npz contains 'label' with shape (D, H, W).
    """

    def __init__(
        self,
        name: str,
        data_dir: str,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        transform: Callable | None = None,
        seed: int = 42,
        in_channels: int = 1,
        image_size: tuple[int, ...] = (64, 64, 64),
        mean: tuple[float, ...] = (0.0,),
        std: tuple[float, ...] = (1.0,),
    ) -> None:
        super().__init__(batch_size, shuffle, drop_last, 0, seed)
        self.name = name
        self.data_dir = data_dir
        self.requested_split = normalize_split(split)
        self.split = resolve_folder_split(
            data_dir,
            self.requested_split,
            source_name=f"{self.__class__.__name__}({name})",
        )
        self.transform = transform
        self.in_channels = in_channels
        self.image_size = tuple(image_size)
        self.mean = mean
        self.std = std

        self.image_paths: list[str] = []
        self.label_paths: list[str] = []

        self._scan_pairs()
        self.num_samples = len(self.image_paths)

    def _scan_pairs(self) -> None:
        import os

        split_dir = os.path.join(self.data_dir, self.split)
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            raise FileNotFoundError(f"Expected images/ and labels/ under {split_dir}")

        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npz")])
        for fname in img_files:
            base = os.path.splitext(fname)[0]
            img_p = os.path.join(img_dir, fname)
            lbl_p = os.path.join(lbl_dir, base + ".npz")
            if os.path.exists(lbl_p):
                self.image_paths.append(img_p)
                self.label_paths.append(lbl_p)

        if not self.image_paths:
            raise ValueError(f"No image/label pairs found under {split_dir}")

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def _load_npz(self, path: str) -> np.ndarray:
        data = np.load(path)
        keys = data.files
        if "image" in keys:
            arr = data["image"]
        elif "label" in keys:
            arr = data["label"]
        else:
            arr = data[keys[0]]
        return np.array(arr)

    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and end - start < self.batch_size:
                break

            batch_indices = indices[start:end]

            imgs = [self._load_npz(self.image_paths[i]) for i in batch_indices]
            lbls = [self._load_npz(self.label_paths[i]) for i in batch_indices]

            proc_imgs = []
            proc_lbls = []
            for img, lbl in zip(imgs, lbls):
                img = np.array(img, dtype=np.float32)
                if img.ndim == 3:
                    img = np.expand_dims(img, 0)
                elif img.ndim == 4 and img.shape[-1] < 5:
                    img = np.moveaxis(img, -1, 0)

                # crop/pad to target
                spatial = img.shape[1:]
                target = self.image_size
                if spatial != target:
                    pads = []
                    crops = []
                    for cur, tgt in zip(spatial, target):
                        if cur < tgt:
                            total = tgt - cur
                            before = total // 2
                            after = total - before
                            pads.append((before, after))
                            crops.append((0, cur))
                        elif cur > tgt:
                            start = (cur - tgt) // 2
                            end = start + tgt
                            pads.append((0, 0))
                            crops.append((start, end))
                        else:
                            pads.append((0, 0))
                            crops.append((0, cur))
                    slices = [slice(None)]
                    for s, e in crops:
                        slices.append(slice(s, e))
                    img = img[tuple(slices)]
                    pad_pattern = [(0, 0)] + pads
                    img = np.pad(img, pad_pattern, mode="constant", constant_values=0)

                lbl = np.array(lbl, dtype=np.int32)
                if lbl.ndim == 4 and lbl.shape[0] == 1:
                    lbl = lbl[0]
                spatial_lbl = lbl.shape
                if spatial_lbl != target:
                    pads = []
                    crops = []
                    for cur, tgt in zip(spatial_lbl, target):
                        if cur < tgt:
                            total = tgt - cur
                            before = total // 2
                            after = total - before
                            pads.append((before, after))
                            crops.append((0, cur))
                        elif cur > tgt:
                            start = (cur - tgt) // 2
                            end = start + tgt
                            pads.append((0, 0))
                            crops.append((start, end))
                        else:
                            pads.append((0, 0))
                            crops.append((0, cur))
                    slices = []
                    for s, e in crops:
                        slices.append(slice(s, e))
                    lbl = lbl[tuple(slices)]
                    pad_pattern = pads
                    lbl = np.pad(lbl, pad_pattern, mode="constant", constant_values=0)

                proc_imgs.append(img)
                proc_lbls.append(lbl)

            batch_x = np.stack(proc_imgs, axis=0)
            batch_y = np.stack(proc_lbls, axis=0)

            batch_x = _normalize_batch(batch_x, self.mean, self.std)

            yield {
                "image": jnp.array(batch_x),
                "label": jnp.array(batch_y),
            }

    @property
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name,
            num_classes=2,
            num_train=self.num_samples,
            num_val=0,
            num_test=0,
            input_shape=(self.in_channels,) + self.image_size,
            mean=self.mean,
            std=self.std,
        )


class HuggingFaceDataLoader(DataLoader):
    """Generic loader using Hugging Face `datasets`.

    Handles simple classification and segmentation datasets. Examples must contain
    at least 'image' and 'label' fields. For segmentation the label should be
    an array (per-voxel labels), for classification the label should be an int.
    """

    def __init__(
        self,
        hf_id: str,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
        in_channels: int = 1,
        image_size: tuple[int, ...] | None = None,
        mean: tuple[float, ...] = (0.0,),
        std: tuple[float, ...] = (1.0,),
        data_dir: str | None = None,
        transform: Callable | None = None,
    ) -> None:
        super().__init__(batch_size, shuffle, drop_last, 0, seed)
        self.hf_id = hf_id
        self.requested_split = normalize_split(split)
        self.split = resolve_hf_split(hf_id, self.requested_split, data_dir=data_dir)
        self.in_channels = in_channels
        self.image_size = tuple(image_size) if image_size is not None else None
        self.mean = mean
        self.std = std
        self.data_dir = data_dir
        self.transform = transform

        try:
            from datasets import load_dataset
        except Exception as e:  # pragma: no cover - runtime optional
            raise RuntimeError(
                "To use Hugging Face datasets install 'datasets' (pip install datasets)"
            ) from e

        # Load the split (may download the dataset)
        ds = load_dataset(hf_id, split=self.split, data_dir=self.data_dir)
        # Materialize into memory list for simpler iteration
        self._examples = list(ds)
        self.num_samples = len(self._examples)

        # Try to infer task type from the first example
        example = self._examples[0] if self.num_samples > 0 else {}
        # Support multiple possible label fields used by different HF datasets
        for _k in ("label", "labels", "fine_label", "coarse_label", "target"):
            if _k in example:
                lbl = example[_k]
                break
        else:
            lbl = None

        if isinstance(lbl, (int, float)):
            self.task = "classification"
        else:
            self.task = "segmentation"

        # For classification, we store image arrays and label ints
        # For segmentation, we expect image arrays and label arrays

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def _process_image(self, img: Any) -> np.ndarray:
        arr = np.array(img)
        # Scale integer images to [0, 1] float range
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        else:
            arr = arr.astype(np.float32)
        # Handle channel last -> first for common image types (H, W, C) -> (C, H, W)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            arr = np.moveaxis(arr, -1, 0)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        return arr

    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and end - start < self.batch_size:
                break

            batch_indices = indices[start:end]
            imgs: list[np.ndarray] = []
            lbls: list[Any] = []
            for i in batch_indices:
                ex = self._examples[i]
                # Try common image keys used across HF datasets
                img = None
                for _k in ("image", "images", "img", "image0", "pixel_values"):
                    if _k in ex:
                        img = ex[_k]
                        break

                # Try common label keys
                lab = None
                for _k in ("label", "labels", "fine_label", "coarse_label", "target"):
                    if _k in ex:
                        lab = ex[_k]
                        break
                if img is None or lab is None:
                    raise ValueError(f"Example {i} does not contain image/label fields")

                # pick first image if list
                if isinstance(img, (list, tuple)):
                    img = img[0]

                if self.task == "classification":
                    proc_img = self._process_image(img)
                    imgs.append(proc_img)
                    lbls.append(int(lab))
                else:
                    # segmentation
                    proc_img = self._process_image(img)
                    # label may already be an array
                    lbl_arr = np.array(lab)
                    if lbl_arr.ndim == 4 and lbl_arr.shape[0] == 1:
                        lbl_arr = lbl_arr[0]
                    imgs.append(proc_img)
                    lbls.append(lbl_arr.astype(np.int32))

            # Stack and normalize
            batch_x = np.stack(imgs, axis=0)

            # If channels are last (B, H, W, C) or (B, D, H, W, C), move them to front (B, C, ...)
            if batch_x.ndim >= 4:
                channels = self.in_channels
                # If channel location is last and doesn't match mean's channel dim, move it
                if batch_x.shape[-1] == channels and batch_x.shape[1] != channels:
                    batch_x = np.moveaxis(batch_x, -1, 1)

            # Normalize and yield
            # Normalization is now handled by the augmentation pipeline (if present)
            # or applied here if no pipeline is used.
            # If self.transform is not None, it is expected to include normalization.
            if self.transform is None:
                batch_x = _normalize_batch(batch_x, self.mean, self.std)

            if self.transform is not None:
                key = jax.random.PRNGKey(self._rng.randint(0, 2**31))
                # Apply per-sample transforms (RandomCrop expects (C,H,W))
                augmented = []
                for i in range(batch_x.shape[0]):
                    key, subkey = jax.random.split(key)
                    augmented.append(self.transform(jnp.array(batch_x[i]), subkey))
                batch_x = jnp.stack(augmented, axis=0)

            if self.task == "classification":
                yield {
                    "image": jnp.array(batch_x)
                    if not isinstance(batch_x, jnp.ndarray)
                    else batch_x,
                    "label": jnp.array(np.array(lbls, dtype=np.int32)),
                }
            else:
                yield {
                    "image": jnp.array(batch_x)
                    if not isinstance(batch_x, jnp.ndarray)
                    else batch_x,
                    "label": jnp.array(np.stack(lbls, axis=0)),
                }

    @property
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.hf_id,
            num_classes=2 if self.task == "segmentation" else 1000,
            num_train=self.num_samples,
            num_val=0,
            num_test=0,
            input_shape=(self.in_channels,)
            + (self.image_size if self.image_size is not None else ()),
            mean=self.mean,
            std=self.std,
        )


class SyntheticDataLoader(DataLoader):
    """Data loader for synthetic data."""

    def __init__(
        self,
        num_samples: int,
        input_shape: tuple[int, ...],
        num_classes: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(batch_size, shuffle, False, 0, seed)
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
        self._rng_data = np.random.RandomState(seed)
        self.x = self._rng_data.randn(num_samples, *input_shape).astype(np.float32)
        self.y = self._rng_data.randint(0, num_classes, num_samples)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, self.num_samples - self.batch_size + 1, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]
            yield {
                "image": jnp.array(self.x[batch_indices]),
                "label": jnp.array(self.y[batch_indices]),
            }

    @property
    def dataset_info(self) -> DatasetInfo:
        channels = self.input_shape[0]
        return DatasetInfo(
            name="synthetic",
            num_classes=self.num_classes,
            num_train=self.num_samples,
            num_val=0,
            num_test=0,
            input_shape=self.input_shape,
            mean=(0.0,) * channels,
            std=(1.0,) * channels,
        )


class SyntheticSegmentationDataLoader(DataLoader):
    """Synthetic segmentation loader that yields random volumes and label masks.

    Produces images shaped (B, C, D, H, W) and labels shaped (B, D, H, W).
    """

    def __init__(
        self,
        num_samples: int,
        input_shape: tuple[int, ...],  # (C, D, H, W)
        num_classes: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(batch_size, shuffle, False, 0, seed)
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.num_classes = num_classes
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, self.num_samples - self.batch_size + 1, self.batch_size):
            batch_size = self.batch_size
            imgs = self._rng.randn(batch_size, *self.input_shape).astype(np.float32)
            # labels are ints per voxel in [0, num_classes)
            spatial = self.input_shape[1:]
            lbls = self._rng.randint(0, self.num_classes, size=(batch_size, *spatial)).astype(
                np.int32
            )
            yield {"image": jnp.array(imgs), "label": jnp.array(lbls)}

    @property
    def dataset_info(self) -> DatasetInfo:
        channels = self.input_shape[0]
        return DatasetInfo(
            name="synthetic_segmentation",
            num_classes=self.num_classes,
            num_train=self.num_samples,
            num_val=0,
            num_test=0,
            input_shape=self.input_shape,
            mean=(0.0,) * channels,
            std=(1.0,) * channels,
        )


def create_data_loader(
    config: DataConfig,
    split: str = "train",
    batch_size: int = 32,
    pad_to: tuple[int, ...] | None = None,
) -> DataLoader:
    """Create a data loader from configuration."""
    from ndswin.training.augmentation import create_augmentation_pipeline

    normalized_split = normalize_split(split)
    bs = batch_size
    is_train = normalized_split == "train"

    ds = config.dataset.lower()
    # Certain loaders handle normalization internally (CIFAR, VolumeFolder)
    # We skip it in the pipeline to avoid double-normalization.
    internal_norm = ds in {"cifar10", "cifar100", "volume", "volume_folder", "3d"}

    # Build per-sample augmentation transform for training splits
    train_transform = None
    if is_train:
        train_transform = create_augmentation_pipeline(
            config, is_training=True, skip_normalize=internal_norm
        )
        # Only use the transform if it has actual transforms (not just empty Compose)
        if hasattr(train_transform, "transforms") and len(train_transform.transforms) == 0:
            train_transform = None

    kwargs = {
        "data_dir": config.data_dir,
        "split": normalized_split,
        "batch_size": bs,
        "shuffle": is_train,
        "download": config.download,
        "transform": train_transform,
    }

    if ds == "cifar10":
        return CIFAR10DataLoader(**kwargs)
    elif ds == "cifar100":
        return CIFAR100DataLoader(**kwargs)
    elif ds in {"volume", "volume_folder", "3d", "your_3d_dataset"}:
        # Volume folder data loader expects data dir layout with class subfolders per split
        return VolumeFolderDataLoader(
            name=config.dataset,
            data_dir=config.data_dir,
            split=normalized_split,
            batch_size=bs,
            shuffle=is_train,
            seed=config.download if hasattr(config, "download") else 42,
            in_channels=config.in_channels,
            image_size=config.image_size,
            mean=config.mean,
            std=config.std,
            pad_to=pad_to,
        )
    elif ds in {"medseg_msd", "medseg", "medseg_brain_tumour"}:
        # Numpy/NPZ-based segmentation dataset exported by train/fetch_medseg_msd.py
        requested_split = resolve_folder_split(
            config.data_dir,
            normalized_split,
            source_name=f"create_data_loader({config.dataset})",
        )

        return NumpySegmentationFolderDataLoader(
            name=config.dataset,
            data_dir=config.data_dir,
            split=requested_split,
            batch_size=bs,
            shuffle=is_train,
            seed=42,
            in_channels=config.in_channels,
            image_size=config.image_size,
            mean=config.mean,
            std=config.std,
        )
    elif getattr(config, "hf_id", None) is not None or (
        isinstance(config.dataset, str) and config.dataset.startswith("hf:")
    ):
        # Use Hugging Face datasets directly. DataConfig can set `hf_id` or set dataset to "hf:<id>".
        hf_id = getattr(config, "hf_id", None) or config.dataset.split("hf:", 1)[1]
        try:
            return HuggingFaceDataLoader(
                hf_id=hf_id,
                split=normalized_split,
                batch_size=bs,
                shuffle=is_train,
                seed=config.download if hasattr(config, "download") else 42,
                in_channels=config.in_channels,
                image_size=config.image_size,
                mean=config.mean,
                std=config.std,
                data_dir=config.data_dir if ds == "imagefolder" else None,
                transform=train_transform,
            )
        except ValueError as e:
            msg = str(e)
            # Detect common WebDataset-format archive error from `datasets` loader and fallback
            if "WebDataset" in msg or "TAR archives" in msg or "webdataset" in msg.lower():
                print(
                    f"Hugging Face dataset {hf_id} appears to be packaged as WebDataset and failed to load via `datasets`. Falling back to exporting via `train/fetch_hf_dataset.py` and using local NPZ loader. Error: {e}"
                )
                # Export the dataset to local disk once (per hf_id)
                outdir = Path("data") / (
                    hf_id.replace("/", "_") if hf_id is not None else "hf_dataset"
                )
                if not outdir.exists() or not any(outdir.iterdir()):
                    cmd = [
                        sys.executable,
                        "train/fetch_hf_dataset.py",
                        "--hf-id",
                        hf_id,
                        "--outdir",
                        str(outdir),
                    ]
                    print("Exporting dataset with:", " ".join(cmd))
                    try:
                        subprocess.check_call(cmd)
                    except subprocess.CalledProcessError as cpe:
                        print(
                            f"Export failed: {cpe}; falling back to synthetic data loader for testing."
                        )
                        # Fallback to synthetic data so sweep can continue in environments where HF webdataset isn't supported
                        if getattr(config, "task", "classification") == "segmentation":
                            input_shape = (config.in_channels,) + tuple(config.image_size)
                            return SyntheticSegmentationDataLoader(
                                num_samples=100,
                                input_shape=input_shape,
                                num_classes=getattr(config, "num_classes", 2),
                                batch_size=bs,
                                shuffle=(split == "train"),
                                seed=42,
                            )
                        else:
                            input_shape = (config.in_channels,) + tuple(config.image_size)
                            return SyntheticDataLoader(
                                num_samples=1000,
                                input_shape=input_shape,
                                num_classes=getattr(config, "num_classes", 10),
                                batch_size=bs,
                                shuffle=(split == "train"),
                                seed=42,
                            )
                # Use NumpySegmentationFolderDataLoader if segmentation
                return NumpySegmentationFolderDataLoader(
                    name=config.dataset,
                    data_dir=str(outdir),
                    split=normalized_split,
                    batch_size=bs,
                    shuffle=is_train,
                    seed=42,
                    in_channels=config.in_channels,
                    image_size=config.image_size,
                    mean=config.mean,
                    std=config.std,
                )
            # Re-raise other errors
            raise
    elif ds == "synthetic":
        return SyntheticDataLoader(
            num_samples=1000,
            input_shape=(config.in_channels,) + config.image_size,
            num_classes=10,
            batch_size=bs,
        )
    raise ValueError(f"Unknown dataset: {config.dataset}")


def prefetch_to_device(iterator: Iterator[Batch], size: int = 2) -> Generator[Batch, None, None]:
    """Prefetch batches to device."""
    queue = []
    for batch in iterator:
        queue.append(jax.device_put(batch))
        if len(queue) >= size:
            yield queue.pop(0)
    while queue:
        yield queue.pop(0)
