"""Dataset reading utilities for ndswin-jax training.

This module uses Hugging Face `datasets` library for data loading,
avoiding TensorFlow dependencies entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator

import jax.numpy as jnp
import random
from datasets import load_dataset
from PIL import Image


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a dataset split."""

    examples: int
    steps_per_epoch: int


def get_dataset_info(config, split: str, batch_size: int) -> DatasetInfo:
    """Return metadata for the requested split using Hugging Face datasets."""
    ds = load_dataset(
        config.dataset,
        split=split,
        cache_dir=config.data_dir,
    )
    num_examples = len(ds)
    steps = max(1, num_examples // batch_size)
    return DatasetInfo(examples=num_examples, steps_per_epoch=steps)


def _hfds_batch_iterator(config, split: str, batch_size: int, shuffle: bool, seed: int, repeat: bool):
    """Create a batch iterator using Hugging Face datasets.

    Yields batches in the format: {'image': jnp.array, 'label': jnp.array}.
    Data is normalized and formatted according to the model's space dimension.
    """
    ds = load_dataset(
        config.dataset,
        split=split,
        cache_dir=config.data_dir,
    )

    # Convert to list for simple indexing
    data = list(ds)

    # Optionally shuffle
    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    def _get_data_and_label(example):
        """Extract and preprocess data based on model space dimension."""
        # Determine space dimension (default to 2D for backward compatibility)
        space = 2
        if hasattr(config, 'model') and hasattr(config.model, 'space'):
            space = config.model.space

        # Data column detection (flexible for different datasets)
        data_col = None
        for col in ["image", "img", "pixel_values", "points", "voxels", "data", "inputs"]:
            if col in example:
                data_col = col
                break

        if data_col is None:
            raise ValueError(f"Unrecognized data column in dataset. Available columns: {list(example.keys())}")

        arr = jnp.array(example[data_col])

        if space == 2:
            # 2D image processing
            arr = _process_2d_data(arr, config.input_resolution)
        elif space == 3:
            # 3D data processing
            arr = _process_3d_data(arr, config.input_resolution)
        else:
            raise ValueError(f"Unsupported space dimension: {space}")

        # Label extraction (same for all spaces)
        label = _extract_label(example)

        return arr, label

    def _process_2d_data(arr, target_resolution):
        """Process 2D image data."""
        target_h, target_w = target_resolution

        # Handle different input formats
        if arr.ndim == 2:  # Grayscale
            arr = arr.astype(jnp.float32) / 255.0
            arr = jnp.expand_dims(arr, axis=0)  # Add channel dimension
        elif arr.ndim == 3:  # RGB or HWC
            # Resize if needed
            if arr.shape[0] != target_h or arr.shape[1] != target_w:
                if arr.dtype != jnp.uint8:
                    arr = (arr * 255).astype(jnp.uint8)
                pil = Image.fromarray(arr)
                pil = pil.resize((target_w, target_h))
                arr = jnp.array(pil)

            # Normalize and convert HWC -> CHW
            arr = arr.astype(jnp.float32) / 255.0
            arr = jnp.moveaxis(arr, -1, 0)  # HWC -> CHW
        else:
            raise ValueError(f"Unexpected 2D data shape: {arr.shape}")

        return arr

    def _process_3d_data(arr, target_resolution):
        """Process 3D data (point clouds or voxels)."""
        target_d, target_h, target_w = target_resolution

        if arr.ndim == 2 and arr.shape[1] == 3:
            # Point cloud: (N, 3) -> voxelize to (D, H, W)
            arr = _voxelize_point_cloud(arr, target_resolution)
        elif arr.ndim == 3:
            # Already voxelized: (D, H, W) or (H, W, D)
            if arr.shape != target_resolution:
                # Simple resize for voxel grids (could be improved)
                arr = arr.astype(jnp.float32)
                # For now, assume input is close to target size
                # TODO: Implement proper 3D resizing
            arr = arr.astype(jnp.float32)
        elif arr.ndim == 4 and arr.shape[0] == 1:
            # Single channel voxel grid
            arr = arr[0].astype(jnp.float32)
        else:
            raise ValueError(f"Unexpected 3D data shape: {arr.shape}")

        # Ensure binary voxels (0 or 1)
        arr = jnp.clip(arr, 0, 1)

        # Add channel dimension
        arr = jnp.expand_dims(arr, 0)

        return arr

    def _voxelize_point_cloud(points, resolution):
        """Simple voxelization of point cloud."""
        # Normalize points to [0, 1] cube
        min_coords = jnp.min(points, axis=0)
        max_coords = jnp.max(points, axis=0)
        normalized_points = (points - min_coords) / (max_coords - min_coords + 1e-8)

        # Create voxel grid
        voxel_grid = jnp.zeros(resolution, dtype=jnp.float32)

        # Convert to voxel indices
        voxel_indices = (normalized_points * jnp.array(resolution)).astype(int)
        voxel_indices = jnp.clip(voxel_indices, 0, jnp.array(resolution) - 1)

        # Set occupied voxels
        voxel_grid = voxel_grid.at[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]].set(1.0)

        return voxel_grid

    def _extract_label(example):
        """Extract label from dataset example."""
        for col in ["label", "labels", "target", "fine_label", "category"]:
            if col in example:
                return int(example[col])
        raise ValueError(f"Unrecognized label column in dataset. Available columns: {list(example.keys())}")

    idx = 0
    n = len(data)
    while True:
        if idx + batch_size > n:
            if repeat:
                # start over
                if shuffle:
                    rng = np.random.default_rng(seed)
                    rng.shuffle(data)
                idx = 0
            else:
                break

        batch = data[idx: idx + batch_size]
        images = []
        labels = []
        for ex in batch:
            img, lab = _get_data_and_label(ex)
            images.append(img)
            labels.append(lab)

        images = jnp.stack(images)
        labels = jnp.array(labels, dtype=jnp.int32)
        yield {"image": jnp.array(images), "label": jnp.array(labels)}

        idx += batch_size


def create_input_iter(
    config,
    split: str,
    batch_size: int,
    *,
    seed: int,
    shuffle: bool,
    repeat: bool,
) -> Iterator[Dict[str, jnp.ndarray]]:
    """Create an iterator over batches for the given split using Hugging Face datasets.

    Supports both 2D images and 3D data (point clouds/voxels) based on config.model.space.
    """
    yield from _hfds_batch_iterator(config, split, batch_size, shuffle, seed, repeat)


__all__ = [
    "DatasetInfo",
    "create_input_iter",
    "get_dataset_info",
]
