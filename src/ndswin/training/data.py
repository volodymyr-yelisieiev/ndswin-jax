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
        """Load CIFAR data using torchvision or tensorflow_datasets."""
        try:
            import torchvision
            import torchvision.transforms as T

            dataset_cls = (
                torchvision.datasets.CIFAR10
                if self.num_classes == 10
                else torchvision.datasets.CIFAR100
            )
            dataset = dataset_cls(
                root=self.data_dir,
                train=(self.split == "train"),
                download=self.download,
                transform=T.ToTensor(),
            )
            x = np.stack([np.array(img) for img, _ in dataset])
            y = np.array([label for _, label in dataset])
            return x, y
        except ImportError:
            try:
                import tensorflow_datasets as tfds

                ds_name = "cifar10" if self.num_classes == 10 else "cifar100"
                ds = tfds.load(
                    ds_name, split=self.split, as_supervised=True, data_dir=self.data_dir
                )
                images, labels = [], []
                for img, label in ds:
                    img = np.transpose(img.numpy(), (2, 0, 1)) / 255.0
                    images.append(img)
                    labels.append(label.numpy())
                return np.array(images, dtype=np.float32), np.array(labels)
            except ImportError:
                raise ImportError(
                    f"Loading {self.name} requires 'torchvision' or 'tensorflow-datasets'. "
                    f"Please install one of them: pip install torchvision"
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


def create_data_loader(
    config: DataConfig, split: str = "train", batch_size: int = 32
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
