"""Weight management utilities.

NDSwin-JAX v0.1.0 does not ship bundled pretrained ImageNet checkpoints.
This module keeps the public helper names stable while making that limitation
explicit and providing supported local weight save/load helpers.
"""

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
from flax.core import unfreeze

from ndswin.config import NDSwinConfig
from ndswin.models.swin import NDSwinTransformer
from ndswin.types import Array, PyTree

BUNDLED_PRETRAINED_MODELS: dict[str, dict[str, Any]] = {}


def list_pretrained_models() -> list[str]:
    """List bundled pretrained models.

    Returns:
        List of bundled pretrained model names. Empty in v0.1.0.
    """
    return list(BUNDLED_PRETRAINED_MODELS.keys())


def get_pretrained_config(model_name: str) -> NDSwinConfig:
    """Get configuration for a bundled pretrained model.

    Args:
        model_name: Name of pretrained model.

    Returns:
        NDSwinConfig for the model.

    Raises:
        ValueError: Always in v0.1.0 because no bundled pretrained weights ship.
    """
    raise ValueError(
        f"No bundled pretrained models are available in NDSwin-JAX v0.1.0 "
        f"(requested {model_name!r}). Use load_weights() with a local export instead."
    )


def load_pretrained(
    model_name: str,
    num_classes: int | None = None,
) -> tuple[NDSwinTransformer, PyTree]:
    """Load a bundled pretrained model.

    Args:
        model_name: Name of pretrained model.
        num_classes: Number of output classes. If different from pretrained,
            the classification head will be reinitialized.

    Returns:
        Tuple of (model, variables).

    Raises:
        ValueError: Always in v0.1.0 because no bundled pretrained weights ship.
    """
    class_note = "" if num_classes is None else f" for {num_classes} classes"
    raise ValueError(
        f"No bundled pretrained weights are available in NDSwin-JAX v0.1.0 "
        f"(requested {model_name!r}{class_note}). Use export-weights or load_weights() "
        "with a local checkpoint bundle."
    )


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
