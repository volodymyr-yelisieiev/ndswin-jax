"""Create synthetic datasets for testing the pipeline without internet access."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import pickle


def create_synthetic_cifar100(data_dir: str = "data/synthetic", num_train: int = 500, num_test: int = 100):
    """Create a synthetic CIFAR-100-like dataset for testing.
    
    Args:
        data_dir: Directory to save the dataset
        num_train: Number of training examples
        num_test: Number of test examples
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    print(f"Creating synthetic CIFAR-100 dataset with {num_train} train and {num_test} test examples...")
    
    train_data = {
        'images': np.random.randint(0, 256, (num_train, 32, 32, 3), dtype=np.uint8),
        'labels': np.random.randint(0, 100, num_train, dtype=np.int64),
    }
    
    test_data = {
        'images': np.random.randint(0, 256, (num_test, 32, 32, 3), dtype=np.uint8),
        'labels': np.random.randint(0, 100, num_test, dtype=np.int64),
    }
    
    # Save to disk
    with open(data_path / "cifar100_train.pkl", "wb") as f:
        pickle.dump(train_data, f)
    
    with open(data_path / "cifar100_test.pkl", "wb") as f:
        pickle.dump(test_data, f)
    
    print(f"Synthetic CIFAR-100 dataset saved to {data_path}")
    return data_path


def create_synthetic_modelnet40(data_dir: str = "data/synthetic", num_train: int = 500, num_test: int = 100):
    """Create a synthetic ModelNet40-like dataset for testing.
    
    Args:
        data_dir: Directory to save the dataset
        num_train: Number of training examples
        num_test: Number of test examples
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic 3D data (voxel grids)
    print(f"Creating synthetic ModelNet40 dataset with {num_train} train and {num_test} test examples...")
    
    train_data = {
        'voxels': np.random.rand(num_train, 32, 32, 32).astype(np.float32),
        'labels': np.random.randint(0, 40, num_train, dtype=np.int64),
    }
    
    test_data = {
        'voxels': np.random.rand(num_test, 32, 32, 32).astype(np.float32),
        'labels': np.random.randint(0, 40, num_test, dtype=np.int64),
    }
    
    # Save to disk
    with open(data_path / "modelnet40_train.pkl", "wb") as f:
        pickle.dump(train_data, f)
    
    with open(data_path / "modelnet40_test.pkl", "wb") as f:
        pickle.dump(test_data, f)
    
    print(f"Synthetic ModelNet40 dataset saved to {data_path}")
    return data_path


if __name__ == "__main__":
    # Create both datasets
    create_synthetic_cifar100()
    create_synthetic_modelnet40()
    print("All synthetic datasets created successfully!")
