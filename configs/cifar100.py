"""Training configuration for CIFAR-100 (feasible in hours on Titan X Pascal)."""

from __future__ import annotations

from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """Return configuration tuned for CIFAR-100."""

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

    # Training configuration
    config.batch_size = 128
    config.eval_batch_size = 256
    config.num_epochs = 300  # More epochs for convergence
    config.warmup_epochs = 10  # Longer warmup
    config.learning_rate = 2e-3  # Lower LR for stability
    config.min_learning_rate = 1e-6
    config.weight_decay = 0.02  # Reduced WD
    config.label_smoothing = 0.2
    config.max_eval_batches = None
    config.log_every = 50
    config.keep_checkpoints = 5
    config.gradient_clip_norm = 1.0  # Add gradient clipping

    # Data augmentation
    config.use_augmentation = True
    config.mixup_alpha = 0.2
    config.cutmix_alpha = 0.2
    config.augmentation_prob = 1.0
    config.use_random_crop = True
    config.random_crop_padding = 4
    config.use_random_flip = True
    config.use_color_jitter = True
    config.color_jitter_brightness = 0.2
    config.color_jitter_contrast = 0.2
    config.use_cutout = True
    config.cutout_size = 8

    # Model configuration - Swin-Tiny scaled for CIFAR-100
    model = ConfigDict()
    model.space = 2
    model.dim = 96
    model.depths = (2, 2, 6, 3)  # Shallower for smaller dataset
    model.num_heads = (2, 4, 8, 16)
    model.patch_size = (4, 4)
    model.window_size = (4, 4)  # Smaller window for 32x32 images
    model.in_channels = 3
    model.num_classes = 100
    model.mlp_ratio = 4.0
    model.drop_path_rate = 0.2  # Lower for smaller model
    model.head_drop = 0.2
    model.use_conv = False
    model.use_abs_pos = False

    config.model = model
    return config