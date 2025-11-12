"""Test configuration for CIFAR-100 (2-epoch test run)."""

from __future__ import annotations

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """Return configuration for 2-epoch test on CIFAR-100."""

    config = ConfigDict()

    # Data configuration
    config.seed = 42
    config.dataset = "cifar100"
    config.data_dir = "data"
    config.train_split = "train"
    config.eval_split = "test"
    config.input_resolution = (32, 32)
    config.prefetch = 4
    config.shuffle_buffer = 50_000

    # Training configuration - reduced for testing
    config.batch_size = 128
    config.eval_batch_size = 256
    config.num_epochs = 2  # Only 2 epochs for testing
    config.warmup_epochs = 0  # No warmup for quick test
    config.learning_rate = 2e-3
    config.min_learning_rate = 1e-6
    config.weight_decay = 0.02
    config.label_smoothing = 0.1
    config.max_eval_batches = None
    config.log_every = 50
    config.keep_checkpoints = 2
    config.gradient_clip_norm = 1.0

    # Data augmentation - reduced for testing
    config.use_augmentation = True
    config.mixup_alpha = 0.2
    config.cutmix_alpha = 0.2
    config.augmentation_prob = 0.8
    config.use_random_crop = True
    config.random_crop_padding = 4
    config.use_random_flip = True
    config.use_color_jitter = False
    config.use_cutout = False

    # Model configuration - Swin-Tiny scaled for CIFAR-100
    model = ConfigDict()
    model.space = 2
    model.dim = 96
    model.depths = (2, 2, 6, 2)  # Slightly reduced depth
    model.num_heads = (3, 6, 12, 24)
    model.patch_size = (4, 4)
    model.window_size = (4, 4)
    model.in_channels = 3
    model.num_classes = 100
    model.mlp_ratio = 4.0
    model.drop_path_rate = 0.1
    model.head_drop = 0.2
    model.use_conv = False
    model.use_abs_pos = False

    config.model = model
    return config
