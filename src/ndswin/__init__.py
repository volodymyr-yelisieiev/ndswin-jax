"""NDSwin-JAX: N-Dimensional Swin Transformer in JAX/Flax.

A production-grade n-dimensional Swin Transformer implementation supporting
arbitrary dimensions (2D, 3D, 4D, and beyond).
"""

from typing import Any


# Lazy imports for optional and/or heavy modules.
def __getattr__(name: str) -> Any:
    """Lazily resolve public package attributes."""
    if name == "NDSwinConfig":
        from ndswin.config import NDSwinConfig

        return NDSwinConfig
    if name == "TrainingConfig":
        from ndswin.config import TrainingConfig

        return TrainingConfig
    if name == "DataConfig":
        from ndswin.config import DataConfig

        return DataConfig
    if name == "ExperimentConfig":
        from ndswin.config import ExperimentConfig

        return ExperimentConfig
    if name == "NDSwinTransformer":
        from ndswin.models.swin import NDSwinTransformer

        return NDSwinTransformer
    if name == "SwinClassifier":
        from ndswin.models.classifier import SwinClassifier

        return SwinClassifier
    if name == "SwinForSegmentation":
        from ndswin.models.classifier import SwinForSegmentation

        return SwinForSegmentation
    if name == "Array":
        from ndswin.types import Array

        return Array
    if name == "PRNGKey":
        from ndswin.types import PRNGKey

        return PRNGKey
    if name == "Shape":
        from ndswin.types import Shape

        return Shape
    if name == "training":
        from ndswin import training

        return training
    if name == "inference":
        from ndswin import inference

        return inference
    if name == "core":
        from ndswin import core

        return core
    if name == "utils":
        from ndswin import utils

        return utils
    raise AttributeError(f"module 'ndswin' has no attribute '{name}'")


__version__ = "0.1.0"
__author__ = "NDSwin-JAX Authors"

__all__ = [
    "NDSwinConfig",
    "TrainingConfig",
    "DataConfig",
    "ExperimentConfig",
    "NDSwinTransformer",
    "SwinClassifier",
    "SwinForSegmentation",
    "Array",
    "PRNGKey",
    "Shape",
    "training",
    "inference",
    "core",
    "utils",
    "__version__",
]
