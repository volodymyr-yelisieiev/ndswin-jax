"""Data loading utilities for NDSwin-JAX.

This module provides data loading functionality for various datasets
and dimensionalities.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import subprocess
import sys
from pathlib import Path

from ndswin.config import DataConfig
from ndswin.types import Batch


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
        self.split = split
        self.transform = transform
        self.download = download
        self.mean = mean
        self.std = std
        self.num_classes = num_classes

        self.x, self.y = self._load_data()
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

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        mean = np.array(self.mean).reshape(1, 3, 1, 1)
        std = np.array(self.std).reshape(1, 3, 1, 1)

        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and end - start < self.batch_size:
                break

            batch_indices = indices[start:end]
            batch_x = self.x[batch_indices].astype(np.float32)
            batch_y = self.y[batch_indices]

            # Normalize
            batch_x = (batch_x - mean) / std

            if self.transform is not None:
                batch_x = self.transform(batch_x, self._rng.randint(0, 2**31))

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
            num_train=50000,
            num_val=0,
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
            num_train=50000,
            num_val=0,
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
        self.split = split
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

        classes = [d for d in sorted(os.listdir(split_dir)) if os.path.isdir(os.path.join(split_dir, d))]
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
            raise ValueError(f"File {path} has {arr.shape[0]} channels but expected {self.in_channels}")

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
                elif cur > tgt:
                    # crop
                    start = (cur - tgt) // 2
                    end = start + tgt
                    pads.append((0, 0))
                    crops.append((start, end))
                else:
                    pads.append((0, 0))
                    crops.append((0, cur))

            # Apply crop first
            slices = [slice(None)]  # channel dim
            for (s, e) in crops:
                slices.append(slice(s, e))
            new_arr = new_arr[tuple(slices)]

            # Apply pad
            # pads is for spatial dims; need pad pattern for np.pad: ((0,0), (before,after), ...)
            pad_pattern = [(0, 0)] + pads
            new_arr = np.pad(new_arr, pad_pattern, mode="constant", constant_values=0)

        return new_arr.astype(np.float32)

    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        mean = np.array(self.mean).reshape(1, -1, 1, 1)
        std = np.array(self.std).reshape(1, -1, 1, 1)

        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and end - start < self.batch_size:
                break

            batch_indices = indices[start:end]

            batch_x = np.stack([self._load_file(self.file_paths[i]) for i in batch_indices])
            batch_y = np.array([self.labels[i] for i in batch_indices])

            # Normalize: expect (B, C, D, H, W)
            # mean/std shapes: (1, C, 1, 1)
            # Bring to (B, C, D, H, W) and normalize along spatial dims
            batch_x = (batch_x - mean) / std

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
        self.split = split
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

        mean = np.array(self.mean).reshape(1, -1, 1, 1)
        std = np.array(self.std).reshape(1, -1, 1, 1)

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
                    for (s, e) in crops:
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
                    for (s, e) in crops:
                        slices.append(slice(s, e))
                    lbl = lbl[tuple(slices)]
                    pad_pattern = pads
                    lbl = np.pad(lbl, pad_pattern, mode="constant", constant_values=0)

                proc_imgs.append(img)
                proc_lbls.append(lbl)

            batch_x = np.stack(proc_imgs, axis=0)
            batch_y = np.stack(proc_lbls, axis=0)

            batch_x = (batch_x - mean) / std

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
    ) -> None:
        super().__init__(batch_size, shuffle, drop_last, 0, seed)
        self.hf_id = hf_id
        self.split = split
        self.in_channels = in_channels
        self.image_size = tuple(image_size) if image_size is not None else None
        self.mean = mean
        self.std = std

        try:
            from datasets import load_dataset
        except Exception as e:  # pragma: no cover - runtime optional
            raise RuntimeError("To use Hugging Face datasets install 'datasets' (pip install datasets)") from e

        # Load the split (may download the dataset)
        ds = load_dataset(hf_id, split=split)
        # Materialize into memory list for simpler iteration
        self._examples = [ex for ex in ds]
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
        # Handle channel last -> first for common image types (H, W, C) -> (C, H, W)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3) and (
            self.image_size is None or (len(self.image_size) > 0 and self.image_size[0] == arr.shape[-1])
        ):
            arr = np.moveaxis(arr, -1, 0)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        return arr.astype(np.float32)

    def __iter__(self) -> Iterator[Batch]:
        indices = np.arange(self.num_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        mean = np.array(self.mean).reshape(1, -1, 1, 1) if len(self.mean) == 1 else np.array(self.mean).reshape(1, -1, 1, 1)
        std = np.array(self.std).reshape(1, -1, 1, 1) if len(self.std) == 1 else np.array(self.std).reshape(1, -1, 1, 1)

        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and end - start < self.batch_size:
                break

            batch_indices = indices[start:end]
            imgs = []
            lbls = []
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
                channels = mean.shape[1]
                # If channel location is last and doesn't match mean's channel dim, move it
                if batch_x.shape[-1] == channels and batch_x.shape[1] != channels:
                    batch_x = np.moveaxis(batch_x, -1, 1)

            # Reshape mean/std to match number of spatial dimensions dynamically
            spatial_dims = batch_x.ndim - 2  # exclude batch and channel dims
            new_shape = (1, mean.shape[1]) + (1,) * spatial_dims
            mean = np.array(self.mean).reshape(1, -1)[:, : mean.shape[1]].reshape((1, mean.shape[1]))
            mean = mean.reshape(new_shape)
            std = np.array(self.std).reshape(1, -1)[:, : std.shape[1]].reshape((1, std.shape[1]))
            std = std.reshape(new_shape)

            # Normalize and yield
            batch_x = (batch_x - mean) / std
            if self.task == "classification":
                yield {"image": jnp.array(batch_x), "label": jnp.array(np.array(lbls, dtype=np.int32))}
            else:
                yield {"image": jnp.array(batch_x), "label": jnp.array(np.stack(lbls, axis=0))}

    @property
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.hf_id,
            num_classes=2 if self.task == "segmentation" else 1000,
            num_train=self.num_samples,
            num_val=0,
            num_test=0,
            input_shape=(self.in_channels,) + (self.image_size if self.image_size is not None else ()),
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
            end = start + self.batch_size
            batch_size = self.batch_size
            imgs = self._rng.randn(batch_size, *self.input_shape).astype(np.float32)
            # labels are ints per voxel in [0, num_classes)
            spatial = self.input_shape[1:]
            lbls = self._rng.randint(0, self.num_classes, size=(batch_size, *spatial)).astype(np.int32)
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
    config: DataConfig, split: str = "train", batch_size: int = 32, pad_to: tuple[int, ...] | None = None
) -> DataLoader:
    """Create a data loader from configuration."""
    bs = batch_size
    kwargs = {
        "data_dir": config.data_dir,
        "split": "train" if split in ["train", "val"] else "test",
        "batch_size": bs,
        "shuffle": (split == "train"),
        "download": config.download,
    }

    ds = config.dataset.lower()
    if ds == "cifar10":
        return CIFAR10DataLoader(**kwargs)
    elif ds == "cifar100":
        return CIFAR100DataLoader(**kwargs)
    elif ds in {"volume", "volume_folder", "3d", "your_3d_dataset"}:
        # Volume folder data loader expects data dir layout with class subfolders per split
        return VolumeFolderDataLoader(
            name=config.dataset,
            data_dir=config.data_dir,
            split="train" if split in ["train", "val"] else "test",
            batch_size=bs,
            shuffle=(split == "train"),
            seed=config.download if hasattr(config, "download") else 42,
            in_channels=config.in_channels,
            image_size=config.image_size,
            mean=config.mean,
            std=config.std,
            pad_to=pad_to,
        )
    elif ds in {"medseg_msd", "medseg", "medseg_brain_tumour"}:
        # Numpy/NPZ-based segmentation dataset exported by train/fetch_medseg_msd.py
        return NumpySegmentationFolderDataLoader(
            name=config.dataset,
            data_dir=config.data_dir,
            split="train" if split in ["train", "val"] else "test",
            batch_size=bs,
            shuffle=(split == "train"),
            seed=42,
            in_channels=config.in_channels,
            image_size=config.image_size,
            mean=config.mean,
            std=config.std,
        )
    elif getattr(config, "hf_id", None) is not None or (isinstance(config.dataset, str) and config.dataset.startswith("hf:")):
        # Use Hugging Face datasets directly. DataConfig can set `hf_id` or set dataset to "hf:<id>".
        hf_id = getattr(config, "hf_id", None) or config.dataset.split("hf:", 1)[1]
        try:
            return HuggingFaceDataLoader(
                hf_id=hf_id,
                split="train" if split in ["train", "val"] else "test",
                batch_size=bs,
                shuffle=(split == "train"),
                seed=config.download if hasattr(config, "download") else 42,
                in_channels=config.in_channels,
                image_size=config.image_size,
                mean=config.mean,
                std=config.std,
            )
        except ValueError as e:
            msg = str(e)
            # Detect common WebDataset-format archive error from `datasets` loader and fallback
            if "WebDataset" in msg or "TAR archives" in msg or "webdataset" in msg.lower():
                print(f"Hugging Face dataset {hf_id} appears to be packaged as WebDataset and failed to load via `datasets`. Falling back to exporting via `train/fetch_hf_dataset.py` and using local NPZ loader. Error: {e}")
                # Export the dataset to local disk once (per hf_id)
                outdir = Path("data") / (hf_id.replace("/", "_") if hf_id is not None else "hf_dataset")
                if not outdir.exists() or not any(outdir.iterdir()):
                    cmd = [sys.executable, "train/fetch_hf_dataset.py", "--hf-id", hf_id, "--outdir", str(outdir)]
                    print("Exporting dataset with:", " ".join(cmd))
                    try:
                        subprocess.check_call(cmd)
                    except subprocess.CalledProcessError as cpe:
                        print(f"Export failed: {cpe}; falling back to synthetic data loader for testing.")
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
                    split="train" if split in ["train", "val"] else "test",
                    batch_size=bs,
                    shuffle=(split == "train"),
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
