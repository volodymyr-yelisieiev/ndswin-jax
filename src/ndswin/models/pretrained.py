"""Pretrained model loading and weight management.

This module provides utilities for loading pretrained weights,
saving/loading model parameters, and model registry.
"""

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax.core import unfreeze

from ndswin.config import NDSwinConfig
from ndswin.models.swin import NDSwinTransformer
from ndswin.types import Array, PyTree

# Registry of pretrained models
PRETRAINED_MODELS: dict[str, dict[str, Any]] = {
    "swin_tiny_2d_imagenet": {
        "config": NDSwinConfig.swin_tiny_2d,
        "url": None,  # Placeholder for model weights URL
        "description": "Swin-Tiny trained on ImageNet-1K",
        "input_size": (224, 224),
        "num_classes": 1000,
    },
    "swin_small_2d_imagenet": {
        "config": NDSwinConfig.swin_small_2d,
        "url": None,
        "description": "Swin-Small trained on ImageNet-1K",
        "input_size": (224, 224),
        "num_classes": 1000,
    },
    "swin_base_2d_imagenet": {
        "config": NDSwinConfig.swin_base_2d,
        "url": None,
        "description": "Swin-Base trained on ImageNet-1K",
        "input_size": (224, 224),
        "num_classes": 1000,
    },
}


def list_pretrained_models() -> list[str]:
    """List available pretrained models.

    Returns:
        List of pretrained model names.
    """
    return list(PRETRAINED_MODELS.keys())


def get_pretrained_config(model_name: str) -> NDSwinConfig:
    """Get configuration for a pretrained model.

    Args:
        model_name: Name of pretrained model.

    Returns:
        NDSwinConfig for the model.

    Raises:
        ValueError: If model_name is not found.
    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown pretrained model: {model_name}. Available: {list_pretrained_models()}"
        )

    model_info = PRETRAINED_MODELS[model_name]
    config_fn = model_info["config"]
    num_classes = model_info.get("num_classes", 1000)

    return cast(NDSwinConfig, config_fn(num_classes))


def load_pretrained(
    model_name: str,
    num_classes: int | None = None,
) -> tuple[NDSwinTransformer, PyTree]:
    """Load a pretrained model.

    Args:
        model_name: Name of pretrained model.
        num_classes: Number of output classes. If different from pretrained,
            the classification head will be reinitialized.

    Returns:
        Tuple of (model, variables).

    Raises:
        ValueError: If model_name is not found or weights are not available.
    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown pretrained model: {model_name}. Available: {list_pretrained_models()}"
        )

    model_info = PRETRAINED_MODELS[model_name]

    if model_info.get("url") is None:
        raise ValueError(
            f"Pretrained weights not available for {model_name}. "
            "Please train the model or provide weights manually."
        )

    # Get config
    config = get_pretrained_config(model_name)

    # Modify num_classes if needed
    if num_classes is not None and num_classes != config.num_classes:
        config_dict = config.to_dict()
        config_dict["num_classes"] = num_classes
        config = NDSwinConfig.from_dict(config_dict)

    # Create model (validates config is correct)
    _ = NDSwinTransformer(config)

    # Download and load weights
    # TODO: Implement actual weight downloading
    raise NotImplementedError("Pretrained weight loading not yet implemented")


def save_weights(
    variables: PyTree,
    path: str | Path,
    format: str = "safetensors",
) -> None:
    """Save model weights to a file.

    Args:
        variables: Model variables (params or full variables pytree).
        path: Path to save weights.
        format: Export format ('safetensors', 'npz', 'pickle').
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten variables to numpy
    flat_params: dict[str, Any] = {}

    def flatten_dict(d: dict[str, Any], prefix: str = "") -> None:
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flatten_dict(value, full_key)
            else:
                flat_params[full_key] = jax.device_get(value)

    # If it's a TrainState or similar, try to get params
    params = variables.get("params", variables) if hasattr(variables, "get") else variables
    flatten_dict(unfreeze(params))

    if format == "npz":
        jnp.savez(path, **flat_params)
    elif format == "pickle":
        import pickle

        with open(path, "wb") as f:
            pickle.dump(flat_params, f)
    elif format == "safetensors":
        try:
            from safetensors.numpy import save_file

            save_file(flat_params, str(path))
        except ImportError:
            # Fallback to npz if safetensors not available
            jnp.savez(path.with_suffix(".npz"), **flat_params)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_weights(
    path: str | Path,
) -> dict[str, Array]:
    """Load model weights from file.

    Args:
        path: Path to weights file.

    Returns:
        Dictionary of weights.
    """
    path = Path(path)

    if path.suffix == ".npz":
        return dict(jnp.load(path))
    elif path.suffix == ".pkl":
        import pickle

        with open(path, "rb") as f:
            return cast(dict[str, Array], pickle.load(f))
    elif path.suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file

            return {k: jnp.array(v) for k, v in load_file(str(path)).items()}
        except ImportError:
            raise ImportError("safetensors not installed for .safetensors file")
    else:
        # Try npz as default
        return dict(jnp.load(path))
