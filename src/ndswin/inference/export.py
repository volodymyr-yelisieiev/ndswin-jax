"""Model export utilities for NDSwin-JAX.

This module provides utilities to export models to various formats.
"""

import json
import os
from typing import Any, cast

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from ndswin.types import Array, PyTree


def export_to_numpy(
    model: nn.Module,
    params: PyTree,
    output_dir: str,
    batch_stats: PyTree | None = None,
    input_shape: tuple[int, ...] | None = None,
) -> str:
    """Export model weights to NumPy format.

    Args:
        model: Flax model.
        params: Model parameters.
        output_dir: Output directory.
        batch_stats: Optional batch statistics.
        input_shape: Optional input shape for metadata.

    Returns:
        Path to saved weights.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Flatten params
    flat_params: dict[str, Any] = {}
    _flatten_params(params, "", flat_params)

    # Add batch stats
    if batch_stats is not None:
        _flatten_params(batch_stats, "batch_stats/", flat_params)

    # Save as npz
    output_path = os.path.join(output_dir, "weights.npz")
    np.savez(output_path, **{k: np.array(v) for k, v in flat_params.items()})

    # Save metadata
    metadata = {
        "format": "numpy",
        "num_parameters": sum(v.size for v in flat_params.values()),
        "param_shapes": {k: list(v.shape) for k, v in flat_params.items()},
    }
    if input_shape is not None:
        metadata["input_shape"] = list(input_shape)

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def export_to_onnx(
    model: nn.Module,
    params: PyTree,
    output_path: str,
    input_shape: tuple[int, ...],
    batch_stats: PyTree | None = None,
    opset_version: int = 14,
) -> str:
    """Export model to ONNX format.

    Requires jax2onnx package.

    Args:
        model: Flax model.
        params: Model parameters.
        output_path: Output file path.
        input_shape: Input shape (including batch dimension).
        batch_stats: Optional batch statistics.
        opset_version: ONNX opset version.

    Returns:
        Path to saved ONNX model.
    """
    try:
        import jax2onnx
    except ImportError:
        raise ImportError(
            "jax2onnx is required for ONNX export. Install with: pip install jax2onnx"
        )

    # Create forward function
    def forward_fn(x: Array) -> Array:
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats
        return cast(Array, model.apply(variables, x, training=False))

    # Create dummy input
    dummy_input = jnp.zeros(input_shape)

    # Export
    onnx_model = jax2onnx.to_onnx(
        forward_fn,
        [dummy_input],
        opset_version=opset_version,
    )

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        import onnx

        onnx.save(onnx_model, output_path)
    except ImportError:
        # Fallback: save raw bytes
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

    return output_path


def export_to_saved_model(
    model: nn.Module,
    params: PyTree,
    output_dir: str,
    input_shape: tuple[int, ...],
    batch_stats: PyTree | None = None,
) -> str:
    """Export model to TensorFlow SavedModel format.

    Requires jax2tf package.

    Args:
        model: Flax model.
        params: Model parameters.
        output_dir: Output directory.
        input_shape: Input shape (including batch dimension).
        batch_stats: Optional batch statistics.

    Returns:
        Path to saved model directory.
    """
    try:
        import tensorflow as tf
        from jax.experimental import jax2tf
    except ImportError:
        raise ImportError(
            "jax2tf and tensorflow are required for SavedModel export. "
            "Install with: pip install tensorflow jax[tf]"
        )

    # Create forward function
    def forward_fn(x: Array) -> Array:
        variables = {"params": params}
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats
        return cast(Array, model.apply(variables, x, training=False))

    # Convert to TF function
    tf_fn = jax2tf.convert(
        forward_fn,
        polymorphic_shapes=["(batch, ...)"],
        enable_xla=True,
    )

    # Create TF module
    class TFModel(tf.Module):
        def __init__(self) -> None:
            super().__init__()

        @tf.function(  # type: ignore[untyped-decorator]
            input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input")]
        )
        def __call__(self, x: Any) -> Any:
            return tf_fn(x)

    tf_model = TFModel()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tf.saved_model.save(tf_model, output_dir)

    return output_dir


def export_for_serving(
    model: nn.Module,
    params: PyTree,
    output_dir: str,
    input_shape: tuple[int, ...],
    batch_stats: PyTree | None = None,
    formats: list[str] | None = None,
) -> dict[str, str]:
    """Export model for serving in multiple formats.

    Args:
        model: Flax model.
        params: Model parameters.
        output_dir: Output directory.
        input_shape: Input shape (including batch dimension).
        batch_stats: Optional batch statistics.
        formats: List of formats to export ('numpy', 'onnx', 'savedmodel').

    Returns:
        Dictionary mapping format to output path.
    """
    if formats is None:
        formats = ["numpy"]

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for fmt in formats:
        if fmt == "numpy":
            path = export_to_numpy(
                model,
                params,
                os.path.join(output_dir, "numpy"),
                batch_stats,
                input_shape,
            )
            results["numpy"] = path

        elif fmt == "onnx":
            path = export_to_onnx(
                model,
                params,
                os.path.join(output_dir, "model.onnx"),
                input_shape,
                batch_stats,
            )
            results["onnx"] = path

        elif fmt == "savedmodel":
            path = export_to_saved_model(
                model,
                params,
                os.path.join(output_dir, "savedmodel"),
                input_shape,
                batch_stats,
            )
            results["savedmodel"] = path

        else:
            raise ValueError(f"Unknown format: {fmt}")

    # Save export info
    info = {
        "input_shape": list(input_shape),
        "formats": list(results.keys()),
        "paths": results,
    }
    with open(os.path.join(output_dir, "export_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return results


def _flatten_params(
    params: PyTree,
    prefix: str,
    result: dict[str, Array],
) -> None:
    """Flatten nested parameter dictionary."""
    if isinstance(params, dict):
        for key, value in params.items():
            new_prefix = f"{prefix}{key}/" if prefix else f"{key}/"
            _flatten_params(value, new_prefix, result)
    else:
        # Remove trailing slash
        key = prefix.rstrip("/")
        result[key] = params


def load_exported_weights(
    path: str,
    format: str = "numpy",
) -> PyTree:
    """Load exported weights.

    Args:
        path: Path to weights file.
        format: Export format ('numpy').

    Returns:
        Model parameters.
    """
    if format == "numpy":
        data = dict(np.load(path))

        # Unflatten
        params: dict[str, Any] = {}
        for key, value in data.items():
            parts = key.split("/")
            _set_nested(params, parts, jnp.array(value))

        return params

    else:
        raise ValueError(f"Unknown format: {format}")


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """Set a value in a nested dictionary."""
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def create_inference_config(
    model_name: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    class_names: list[str] | None = None,
    preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create inference configuration.

    Args:
        model_name: Model name.
        input_shape: Expected input shape.
        num_classes: Number of output classes.
        class_names: Optional list of class names.
        preprocessing: Preprocessing configuration.

    Returns:
        Inference configuration dictionary.
    """
    config = {
        "model_name": model_name,
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "framework": "jax",
        "library": "ndswin",
    }

    if class_names is not None:
        config["class_names"] = class_names

    if preprocessing is not None:
        config["preprocessing"] = preprocessing
    else:
        config["preprocessing"] = {
            "normalize": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "channel_order": "CHW",
        }

    return config
