"""Test configuration for ModelNet40 (2-epoch test run)."""

from __future__ import annotations

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """Return configuration for 2-epoch test on ModelNet40."""

    config = ConfigDict()

    # Data configuration
    config.seed = 42
    config.dataset = "jxie/modelnet40"  # Hugging Face dataset
    config.data_dir = "data"
    config.train_split = "train"
    config.eval_split = "test"
    config.input_resolution = (32, 32, 32)  # 3D voxel resolution
    config.prefetch = 4
    config.shuffle_buffer = 10_000

    # Training configuration - reduced for testing
    config.batch_size = 32  # Smaller batch for 3D data
    config.eval_batch_size = 64
    config.num_epochs = 2  # Only 2 epochs for testing
    config.warmup_epochs = 0  # No warmup for quick test
    config.learning_rate = 1e-3
    config.min_learning_rate = 1e-6
    config.weight_decay = 0.05
    config.label_smoothing = 0.1
    config.max_eval_batches = None
    config.log_every = 20
    config.keep_checkpoints = 2
    config.gradient_clip_norm = 1.0

    # Data augmentation (3D-specific) - reduced for testing
    config.use_augmentation = True
    config.mixup_alpha = 0.1
    config.cutmix_alpha = 0.1
    config.augmentation_prob = 0.5
    config.use_random_crop = False
    config.use_random_flip = True
    config.use_random_rotation = False
    config.use_random_scale = False
    config.use_point_dropout = False

    # Model configuration - Swin-Tiny adapted for 3D
    model = ConfigDict()
    model.space = 3  # 3D space
    model.dim = 96
    model.depths = (2, 2, 4)  # Reduced depth for faster testing
    model.num_heads = (3, 6, 12)
    model.patch_size = (4, 4, 4)
    model.window_size = (4, 4, 4)
    model.in_channels = 1  # Binary voxels
    model.num_classes = 40  # ModelNet40 has 40 classes
    model.mlp_ratio = 4.0
    model.drop_path_rate = 0.2
    model.head_drop = 0.2
    model.use_conv = False
    model.use_abs_pos = False

    config.model = model
    return config
