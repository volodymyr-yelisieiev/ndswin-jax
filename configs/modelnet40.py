"""Training configuration for ModelNet40 (3D shape classification)."""

from __future__ import annotations

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """Return configuration tuned for ModelNet40."""

    config = ConfigDict()

    # Data configuration
    config.seed = 42
    config.dataset = "jxie/modelnet40"  # Hugging Face dataset
    config.data_dir = "data"
    config.train_split = "train"
    config.eval_split = "test"
    config.input_resolution = (32, 32, 32)  # 3D voxel resolution
    config.prefetch = 4
    config.shuffle_buffer = 10_000  # Smaller dataset

    # Training configuration
    config.batch_size = 64  # Smaller batch size for 3D data
    config.eval_batch_size = 128
    config.num_epochs = 200  # Fewer epochs for convergence
    config.warmup_epochs = 5  # Shorter warmup
    config.learning_rate = 1e-3  # Conservative LR for 3D
    config.min_learning_rate = 1e-6
    config.weight_decay = 0.05  # Higher WD for regularization
    config.label_smoothing = 0.1
    config.max_eval_batches = None
    config.log_every = 50
    config.keep_checkpoints = 5
    config.gradient_clip_norm = 1.0  # Add gradient clipping

    # Data augmentation (3D-specific)
    config.use_augmentation = True
    config.mixup_alpha = 0.1  # Reduced for 3D
    config.cutmix_alpha = 0.1
    config.augmentation_prob = 0.8
    config.use_random_crop = True
    config.random_crop_padding = 2  # Smaller padding for 3D
    config.use_random_flip = True  # Random rotations/flips in 3D space
    config.use_random_rotation = True  # 3D rotations
    config.rotation_range = 45  # Degrees
    config.use_random_scale = True
    config.scale_range = (0.8, 1.2)
    config.use_point_dropout = True  # For point cloud augmentation
    config.dropout_prob = 0.1

    # Model configuration - Swin-Tiny adapted for 3D
    model = ConfigDict()
    model.space = 3  # 3D space
    model.dim = 96
    model.depths = (2, 2, 6, 2)  # Shallower for 3D
    model.num_heads = (3, 6, 12, 24)
    model.patch_size = (4, 4, 4)  # 3D patches
    model.window_size = (4, 4, 4)  # 3D windows
    model.in_channels = 1  # Binary voxels
    model.num_classes = 40  # ModelNet40 has 40 classes
    model.mlp_ratio = 4.0
    model.drop_path_rate = 0.3  # Higher dropout for 3D
    model.head_drop = 0.3
    model.use_conv = False
    model.use_abs_pos = False

    config.model = model
    return config