"""Simple dataset loader for synthetic test data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a dataset split."""
    examples: int
    steps_per_epoch: int


def get_dataset_info_synthetic(dataset_name: str, split: str, batch_size: int, data_dir: str = "data/synthetic") -> DatasetInfo:
    """Return metadata for the requested split."""
    data_path = Path(data_dir)
    
    if dataset_name == "cifar100":
        file_name = f"cifar100_{split}.pkl"
    elif dataset_name == "modelnet40":
        file_name = f"modelnet40_{split}.pkl"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    with open(data_path / file_name, "rb") as f:
        data = pickle.load(f)
    
    # Get number of examples from the first key
    first_key = list(data.keys())[0]
    num_examples = len(data[first_key])
    steps = max(1, num_examples // batch_size)
    return DatasetInfo(examples=num_examples, steps_per_epoch=steps)


def create_input_iter_synthetic(
    dataset_name: str,
    split: str,
    batch_size: int,
    *,
    seed: int,
    shuffle: bool,
    repeat: bool,
    data_dir: str = "data/synthetic",
) -> Iterator[Dict[str, jnp.ndarray]]:
    """Create an iterator over batches for the given split."""
    
    data_path = Path(data_dir)
    
    if dataset_name == "cifar100":
        file_name = f"cifar100_{split}.pkl"
    elif dataset_name == "modelnet40":
        file_name = f"modelnet40_{split}.pkl"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    with open(data_path / file_name, "rb") as f:
        data = pickle.load(f)
    
    # Determine data and label keys
    if "images" in data:
        data_key = "images"
    elif "voxels" in data:
        data_key = "voxels"
    else:
        raise ValueError(f"Unknown data key in dataset")
    
    images = data[data_key]
    labels = data["labels"]
    n = len(images)
    
    # Create indices for shuffling
    indices = np.arange(n)
    
    while True:
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
            seed += 1  # Change seed for next epoch
        
        idx = 0
        while idx < n:
            # Get batch, handling the last partial batch
            end_idx = min(idx + batch_size, n)
            batch_indices = indices[idx:end_idx]
            batch_images = images[batch_indices]
            batch_labels = labels[batch_indices]
            
            # Process based on dataset type
            if dataset_name == "cifar100":
                # CIFAR-100: Convert HWC to CHW and normalize
                batch_images = batch_images.astype(np.float32) / 255.0
                batch_images = np.transpose(batch_images, (0, 3, 1, 2))  # BHWC -> BCHW
            elif dataset_name == "modelnet40":
                # ModelNet40: Add channel dimension for 3D voxels
                batch_images = batch_images.astype(np.float32)
                batch_images = batch_images[:, np.newaxis, :, :, :]  # BDHW -> BCDHW
            
            yield {
                "image": jnp.array(batch_images),
                "label": jnp.array(batch_labels, dtype=jnp.int32),
            }
            
            idx = end_idx
        
        if not repeat:
            break


__all__ = [
    "DatasetInfo",
    "get_dataset_info_synthetic",
    "create_input_iter_synthetic",
]
