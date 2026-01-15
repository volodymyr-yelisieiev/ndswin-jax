"""Training infrastructure for NDSwin-JAX."""

from ndswin.training.augmentation import (
    Compose,
    Cutout,
    Cutmix,
    Mixup,
    MixupOrCutmix,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    create_augmentation_pipeline,
)
from ndswin.training.checkpoint import (
    CheckpointManager,
)
from ndswin.training.data import (
    CIFAR10DataLoader,
    CIFAR100DataLoader,
    DataLoader,
    SyntheticDataLoader,
    NumpySegmentationFolderDataLoader,
    create_data_loader,
)
from ndswin.training.losses import (
    cross_entropy_loss,
    focal_loss,
    label_smoothing_cross_entropy,
)
from ndswin.training.metrics import (
    MetricTracker,
    accuracy,
    top_k_accuracy,
    compute_segmentation_metrics,
)
from ndswin.training.optimizer import (
    create_learning_rate_schedule,
    create_optimizer,
)
from ndswin.training.scheduler import (
    CosineAnnealingSchedule,
    LinearSchedule,
    WarmupSchedule,
)
from ndswin.training.trainer import (
    Trainer,
    TrainState,
    create_train_state,
)

__all__ = [
    # Data
    "DataLoader",
    "CIFAR10DataLoader",
    "CIFAR100DataLoader",
    "SyntheticDataLoader",
    "NumpySegmentationFolderDataLoader",
    "create_data_loader",
    # Augmentation
    "Compose",
    "RandomHorizontalFlip",
    "RandomCrop",
    "Normalize",
    "Cutout",
    "Mixup",
    "Cutmix",
    "MixupOrCutmix",
    "create_augmentation_pipeline",
    # Losses
    "cross_entropy_loss",
    "label_smoothing_cross_entropy",
    "focal_loss",
    # Metrics
    "accuracy",
    "top_k_accuracy",
    "compute_segmentation_metrics",
    "MetricTracker",
    # Optimizer
    "create_optimizer",
    "create_learning_rate_schedule",
    # Scheduler
    "CosineAnnealingSchedule",
    "WarmupSchedule",
    "LinearSchedule",
    # Trainer
    "Trainer",
    "TrainState",
    "create_train_state",
    # Checkpoint
    "CheckpointManager",
]
