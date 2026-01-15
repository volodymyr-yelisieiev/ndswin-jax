"""NDSwin-JAX: N-Dimensional Swin Transformer in JAX/Flax.

A production-grade n-dimensional Swin Transformer implementation supporting
arbitrary dimensions (2D, 3D, 4D, and beyond).
"""

from typing import Any

from ndswin.config import DataConfig as DataConfig
from ndswin.config import ExperimentConfig as ExperimentConfig
from ndswin.config import NDSwinConfig as NDSwinConfig
from ndswin.config import TrainingConfig as TrainingConfig
from ndswin.models.classifier import SwinClassifier as SwinClassifier
from ndswin.models.classifier import SwinForSegmentation as SwinForSegmentation
from ndswin.models.swin import NDSwinTransformer as NDSwinTransformer
from ndswin.types import Array as Array
from ndswin.types import PRNGKey as PRNGKey
from ndswin.types import Shape as Shape


# Lazy imports for optional heavy modules
def __getattr__(name: str) -> Any:
    """Lazy import of submodules."""
    if name == "training":
        from ndswin import training

        return training
    elif name == "inference":
        from ndswin import inference

        return inference
    elif name == "core":
        from ndswin import core

        return core
    elif name == "utils":
        from ndswin import utils

        return utils
    raise AttributeError(f"module 'ndswin' has no attribute '{name}'")


__version__ = "0.1.0"
__author__ = "NDSwin-JAX Authors"

__all__ = [
    # Configuration
    "NDSwinConfig",
    "TrainingConfig",
    "DataConfig",
    "ExperimentConfig",
    # Models
    "NDSwinTransformer",
    "SwinClassifier",
    "SwinForSegmentation",
    # Types
    "Array",
    "PRNGKey",
    "Shape",
    # Submodules
    "training",
    "inference",
    "core",
    "utils",
    # Version
    "__version__",
]
