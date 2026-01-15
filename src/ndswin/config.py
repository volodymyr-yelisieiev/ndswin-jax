"""Configuration classes for NDSwin-JAX.

This module defines dataclasses for model, training, and data configurations
with validation and sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from ndswin.types import DType


def _validate_positive(value: int, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_tuple_length(value: tuple[int, ...], expected_length: int, name: str) -> None:
    """Validate that a tuple has the expected length."""
    if len(value) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}, got {len(value)}")


@dataclass
class NDSwinConfig:
    """Configuration for N-Dimensional Swin Transformer.

    This configuration class defines all hyperparameters needed to construct
    an NDSwinTransformer model. It supports arbitrary dimensionality (2D, 3D,
    4D, and beyond).

    Attributes:
        num_dims: Number of spatial dimensions (e.g., 2 for images, 3 for volumes).
        in_channels: Number of input channels.
        patch_size: Size of each patch in each dimension. Length must equal num_dims.
        embed_dim: Base embedding dimension.
        depths: Number of Swin blocks in each stage.
        num_heads: Number of attention heads in each stage.
        window_size: Window size for attention in each dimension.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Whether to use bias in QKV projections.
        drop_rate: Dropout rate for embeddings.
        attn_drop_rate: Dropout rate for attention weights.
        drop_path_rate: Stochastic depth rate (linearly increases with depth).
        norm_layer: Normalization layer type ('layernorm' or 'rmsnorm').
        use_abs_pos_embed: Whether to use absolute positional embeddings.
        use_rel_pos_bias: Whether to use relative position bias in attention.
        num_classes: Number of output classes (None for feature extraction only).
        pool_type: Pooling type for classification head ('avg', 'max', or 'token').
        head_init_scale: Scale factor for classification head initialization.
        dtype: Data type for computations.

    Example:
        >>> config = NDSwinConfig(
        ...     num_dims=2,
        ...     patch_size=(4, 4),
        ...     embed_dim=96,
        ...     depths=(2, 2, 6, 2),
        ...     num_heads=(3, 6, 12, 24),
        ...     window_size=(7, 7),
        ...     num_classes=1000,
        ... )
    """

    # Dimension configuration
    num_dims: int = 2
    in_channels: int = 3

    # Patch and embedding configuration
    patch_size: tuple[int, ...] = (4, 4)
    embed_dim: int = 96

    # Architecture configuration
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    window_size: tuple[int, ...] = (7, 7)

    # MLP configuration
    mlp_ratio: float = 4.0

    # Attention configuration
    qkv_bias: bool = True
    use_abs_pos_embed: bool = False
    use_rel_pos_bias: bool = True

    # Regularization
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    # Normalization
    norm_layer: str = "layernorm"
    norm_eps: float = 1e-6

    # Classification head
    num_classes: int | None = 1000
    pool_type: str = "avg"
    head_init_scale: float = 1.0

    # Computation settings
    dtype: DType = field(default_factory=lambda: jnp.float32)
    use_fast_attention: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration parameters."""
        _validate_positive(self.num_dims, "num_dims")
        _validate_positive(self.in_channels, "in_channels")
        _validate_positive(self.embed_dim, "embed_dim")

        # Validate tuples match num_dims
        _validate_tuple_length(self.patch_size, self.num_dims, "patch_size")
        _validate_tuple_length(self.window_size, self.num_dims, "window_size")

        # Validate patch sizes
        for i, ps in enumerate(self.patch_size):
            _validate_positive(ps, f"patch_size[{i}]")

        # Validate window sizes
        for i, ws in enumerate(self.window_size):
            _validate_positive(ws, f"window_size[{i}]")

        # Validate depths and heads have same length
        if len(self.depths) != len(self.num_heads):
            raise ValueError(
                f"depths and num_heads must have same length, "
                f"got {len(self.depths)} and {len(self.num_heads)}"
            )

        for i, (d, h) in enumerate(zip(self.depths, self.num_heads)):
            _validate_positive(d, f"depths[{i}]")
            _validate_positive(h, f"num_heads[{i}]")

        # Validate that embedding dims at each stage are divisible by num_heads
        for i in range(self.num_stages):
            stage_dim = self.get_stage_embed_dim(i)
            if stage_dim % self.num_heads[i] != 0:
                raise ValueError(
                    f"embed_dim at stage {i} ({stage_dim}) must be divisible by num_heads[{i}] ({self.num_heads[i]})"
                )

        # Validate ratios and rates
        _validate_non_negative(self.mlp_ratio, "mlp_ratio")
        _validate_non_negative(self.drop_rate, "drop_rate")
        _validate_non_negative(self.attn_drop_rate, "attn_drop_rate")
        _validate_non_negative(self.drop_path_rate, "drop_path_rate")

        if self.drop_rate > 1.0:
            raise ValueError(f"drop_rate must be <= 1.0, got {self.drop_rate}")
        if self.attn_drop_rate > 1.0:
            raise ValueError(f"attn_drop_rate must be <= 1.0, got {self.attn_drop_rate}")
        if self.drop_path_rate > 1.0:
            raise ValueError(f"drop_path_rate must be <= 1.0, got {self.drop_path_rate}")

        # Validate norm layer
        valid_norms = {"layernorm", "rmsnorm"}
        if self.norm_layer not in valid_norms:
            raise ValueError(f"norm_layer must be one of {valid_norms}, got {self.norm_layer}")

        # Validate pool type
        valid_pools = {"avg", "max", "token"}
        if self.pool_type not in valid_pools:
            raise ValueError(f"pool_type must be one of {valid_pools}, got {self.pool_type}")

    @property
    def num_stages(self) -> int:
        """Return the number of stages in the model."""
        return len(self.depths)

    @property
    def total_blocks(self) -> int:
        """Return the total number of Swin blocks."""
        return sum(self.depths)

    def get_stage_embed_dim(self, stage: int) -> int:
        """Get the embedding dimension for a specific stage.

        The embedding dimension doubles at each stage after patch merging.

        Args:
            stage: Stage index (0-indexed).

        Returns:
            Embedding dimension for the stage.
        """
        return int(self.embed_dim * (2**stage))

    def get_stage_num_heads(self, stage: int) -> int:
        """Get the number of attention heads for a specific stage.

        Args:
            stage: Stage index (0-indexed).

        Returns:
            Number of attention heads for the stage.
        """
        return self.num_heads[stage]

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "num_dims": self.num_dims,
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "use_abs_pos_embed": self.use_abs_pos_embed,
            "use_rel_pos_bias": self.use_rel_pos_bias,
            "drop_rate": self.drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "norm_layer": self.norm_layer,
            "norm_eps": self.norm_eps,
            "num_classes": self.num_classes,
            "pool_type": self.pool_type,
            "head_init_scale": self.head_init_scale,
            "dtype": str(self.dtype),
            "use_fast_attention": self.use_fast_attention,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "NDSwinConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters.

        Returns:
            NDSwinConfig instance.
        """
        # Handle dtype conversion
        if "dtype" in config_dict and isinstance(config_dict["dtype"], str):
            dtype_map = {
                "float32": jnp.float32,
                "float16": jnp.float16,
                "bfloat16": jnp.bfloat16,
            }
            dtype_str = config_dict["dtype"].replace("jnp.", "").replace("np.", "")
            config_dict["dtype"] = dtype_map.get(dtype_str, jnp.float32)

        # Convert lists to tuples
        for key in ["patch_size", "depths", "num_heads", "window_size"]:
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])

        return cls(**config_dict)

    # Preset configurations
    @classmethod
    def swin_tiny_2d(cls, num_classes: int = 1000) -> "NDSwinConfig":
        """Create Swin-Tiny configuration for 2D images.

        Args:
            num_classes: Number of output classes.

        Returns:
            NDSwinConfig for Swin-Tiny 2D.
        """
        return cls(
            num_dims=2,
            patch_size=(4, 4),
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=(7, 7),
            num_classes=num_classes,
        )

    @classmethod
    def swin_small_2d(cls, num_classes: int = 1000) -> "NDSwinConfig":
        """Create Swin-Small configuration for 2D images.

        Args:
            num_classes: Number of output classes.

        Returns:
            NDSwinConfig for Swin-Small 2D.
        """
        return cls(
            num_dims=2,
            patch_size=(4, 4),
            embed_dim=96,
            depths=(2, 2, 18, 2),
            num_heads=(3, 6, 12, 24),
            window_size=(7, 7),
            num_classes=num_classes,
        )

    @classmethod
    def swin_base_2d(cls, num_classes: int = 1000) -> "NDSwinConfig":
        """Create Swin-Base configuration for 2D images.

        Args:
            num_classes: Number of output classes.

        Returns:
            NDSwinConfig for Swin-Base 2D.
        """
        return cls(
            num_dims=2,
            patch_size=(4, 4),
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            window_size=(7, 7),
            num_classes=num_classes,
        )

    @classmethod
    def swin_large_2d(cls, num_classes: int = 1000) -> "NDSwinConfig":
        """Create Swin-Large configuration for 2D images.

        Args:
            num_classes: Number of output classes.

        Returns:
            NDSwinConfig for Swin-Large 2D.
        """
        return cls(
            num_dims=2,
            patch_size=(4, 4),
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            window_size=(7, 7),
            num_classes=num_classes,
        )

    @classmethod
    def swin_tiny_3d(cls, num_classes: int = 10) -> "NDSwinConfig":
        """Create Swin-Tiny configuration for 3D volumes.

        Args:
            num_classes: Number of output classes.

        Returns:
            NDSwinConfig for Swin-Tiny 3D.
        """
        return cls(
            num_dims=3,
            patch_size=(4, 4, 4),
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=(7, 7, 7),
            num_classes=num_classes,
        )

    @classmethod
    def swin_tiny_4d(cls, num_classes: int = 400) -> "NDSwinConfig":
        """Create Swin-Tiny configuration for 4D spatiotemporal data.

        Args:
            num_classes: Number of output classes.

        Returns:
            NDSwinConfig for Swin-Tiny 4D.
        """
        return cls(
            num_dims=4,
            patch_size=(2, 4, 4, 4),
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=(4, 7, 7, 7),
            num_classes=num_classes,
        )


@dataclass
class TrainingConfig:
    """Configuration for training NDSwin models.

    Attributes:
        learning_rate: Initial learning rate.
        min_learning_rate: Minimum learning rate for scheduling.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        num_epochs: Alias for epochs.
        warmup_epochs: Number of warmup epochs.
        warmup_steps: Number of warmup steps (overrides warmup_epochs if > 0).
        weight_decay: Weight decay coefficient.
        optimizer: Optimizer type ('adamw', 'sgd', 'lamb').
        scheduler: Learning rate scheduler ('cosine', 'linear', 'step').
        lr_schedule: Alias for scheduler.
        gradient_clip_norm: Maximum gradient norm for clipping.
        max_grad_norm: Alias for gradient_clip_norm.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        mixed_precision: Whether to use mixed precision training.
        label_smoothing: Label smoothing coefficient.
        use_ema: Whether to use exponential moving average of weights.
        ema_decay: EMA decay rate.
        seed: Random seed for reproducibility.
        log_interval: Steps between logging.
        eval_interval: Epochs between evaluation.
        save_interval: Epochs between checkpoint saves.
        checkpoint_dir: Directory for saving checkpoints.
        resume_from: Path to checkpoint to resume from.
        num_classes: Number of output classes.
    """

    # Optimization
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    batch_size: int = 32
    epochs: int = 100
    warmup_epochs: int = 5
    warmup_steps: int = 0  # Overrides warmup_epochs if > 0
    weight_decay: float = 0.05

    # Optimizer settings
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    momentum: float = 0.9  # For SGD
    nesterov: bool = True  # For SGD with momentum

    # Aliases for optimizer params
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Scheduler settings
    scheduler: str = "cosine"
    lr_schedule: str = "cosine"  # Alias for scheduler
    step_size: int = 30  # For step scheduler
    gamma: float = 0.1  # For step scheduler

    # Gradient handling
    gradient_clip_norm: float | None = 1.0
    max_grad_norm: float = 1.0  # Alias for gradient_clip_norm
    gradient_accumulation_steps: int = 1

    # Mixed precision
    mixed_precision: bool = False
    loss_scale: float | None = None  # Dynamic if None

    # Regularization
    label_smoothing: float = 0.1
    stochastic_depth_rate: float = 0.1

    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Logging
    log_interval: int = 50
    eval_interval: int = 1
    save_interval: int = 5
    use_tensorboard: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: str | None = None
    save_best_only: bool = False
    max_checkpoints: int = 5

    # Loss selection
    loss: str = "cross_entropy"
    bce_pos_weight: float | None = None

    # Early stopping
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4

    # Task settings
    num_classes: int = 10

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration parameters."""
        _validate_positive(self.batch_size, "batch_size")
        _validate_positive(self.epochs, "epochs")
        _validate_non_negative(self.warmup_epochs, "warmup_epochs")
        _validate_non_negative(self.weight_decay, "weight_decay")
        _validate_non_negative(self.learning_rate, "learning_rate")
        _validate_non_negative(self.min_learning_rate, "min_learning_rate")

        valid_optimizers = {"adamw", "sgd", "adam", "lamb"}
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}, got {self.optimizer}")

        valid_schedulers = {"cosine", "linear", "step", "constant"}
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}, got {self.scheduler}")

        if self.warmup_epochs >= self.epochs:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be less than epochs ({self.epochs})"
            )

    @property
    def num_epochs(self) -> int:
        """Alias for epochs."""
        return self.epochs

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "gradient_clip_norm": self.gradient_clip_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
            "label_smoothing": self.label_smoothing,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "seed": self.seed,
        }


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.

    Attributes:
        dataset: Name of the dataset.
        data_dir: Path to data directory.
        image_size: Size of input images/volumes per dimension.
        num_workers: Number of data loading workers.
        prefetch_size: Size of prefetch buffer.
        shuffle_buffer_size: Size of shuffle buffer.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        augmentation: Whether to apply data augmentation.
        normalize: Whether to normalize inputs.
        mean: Normalization mean per channel.
        std: Normalization std per channel.
    """

    # Dataset
    dataset: str = "cifar10"
    data_dir: str = "data"
    download: bool = True

    # Input size
    image_size: tuple[int, ...] = (224, 224)
    in_channels: int = 3

    # Data loading
    num_workers: int = 4
    prefetch_size: int = 2
    shuffle_buffer_size: int = 10000
    pin_memory: bool = True
    drop_last: bool = True

    # Splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Task type (classification | segmentation)
    task: str = "classification"

    # Preprocessing
    augmentation: bool = True
    normalize: bool = True
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    # Augmentation settings
    random_crop: bool = True
    random_flip: bool = True
    random_rotation: float = 0.0
    color_jitter: bool = False
    auto_augment: str | None = None  # 'v0', 'original', 'rand-m9-mstd0.5'
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    cutout_size: int = 0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration parameters."""
        _validate_non_negative(self.num_workers, "num_workers")
        _validate_positive(self.prefetch_size, "prefetch_size")

        # Validate splits sum to 1
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(
                f"train_split + val_split + test_split must equal 1.0, got {total_split}"
            )

        # Validate image size
        for i, size in enumerate(self.image_size):
            _validate_positive(size, f"image_size[{i}]")

        # Validate normalization parameters
        # Allow common convenient cases by broadcasting or selecting channels:
        # - If mean/std length is 1 and in_channels>1, replicate the single value
        # - If mean/std length is 3 (RGB defaults) and in_channels==1, take the first channel
        if len(self.mean) != self.in_channels:
            if len(self.mean) == 1:
                self.mean = tuple(self.mean[0] for _ in range(self.in_channels))
            elif len(self.mean) == 3 and self.in_channels == 1:
                self.mean = (self.mean[0],)
            else:
                raise ValueError(f"mean must have {self.in_channels} elements, got {len(self.mean)}")
        if len(self.std) != self.in_channels:
            if len(self.std) == 1:
                self.std = tuple(self.std[0] for _ in range(self.in_channels))
            elif len(self.std) == 3 and self.in_channels == 1:
                self.std = (self.std[0],)
            else:
                raise ValueError(f"std must have {self.in_channels} elements, got {len(self.std)}")

    @property
    def num_dims(self) -> int:
        """Return the number of spatial dimensions."""
        return len(self.image_size)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "dataset": self.dataset,
            "data_dir": self.data_dir,
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "num_workers": self.num_workers,
            "augmentation": self.augmentation,
            "normalize": self.normalize,
            "mean": self.mean,
            "std": self.std,
        }


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining model, training, and data configs.

    Attributes:
        name: Experiment name.
        model: Model configuration.
        training: Training configuration.
        data: Data configuration.
        output_dir: Directory for outputs.
        debug: Whether to run in debug mode.
    """

    name: str = "ndswin_experiment"
    model: NDSwinConfig = field(default_factory=NDSwinConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "outputs"
    debug: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "output_dir": self.output_dir,
            "debug": self.debug,
        }

    def get_config_hash(self) -> str:
        """Return a short hash for the configuration for reproducible naming."""
        import json
        import hashlib

        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def get_stamp(self) -> str:
        """Generate a standard stamp using name, dataset and short hash and timestamp."""
        from datetime import datetime

        dataset_name = getattr(self.data, "hf_id", None) or getattr(self.data, "dataset", None) or "dataset"
        dataset_name = str(dataset_name).replace("/", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.name}_{dataset_name}_{self.get_config_hash()}_{timestamp}"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        model_config = NDSwinConfig.from_dict(config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))

        return cls(
            name=config_dict.get("name", "ndswin_experiment"),
            model=model_config,
            training=training_config,
            data=data_config,
            output_dir=config_dict.get("output_dir", "outputs"),
            debug=config_dict.get("debug", False),
        )
