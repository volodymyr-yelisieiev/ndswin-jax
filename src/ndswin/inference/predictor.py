"""Inference predictor for NDSwin-JAX.

This module provides high-level prediction interfaces.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from ndswin.types import Array, PyTree


class Predictor(ABC):
    """Abstract base class for predictors.

    Provides a standardized interface for model inference.
    """

    def __init__(
        self,
        model: nn.Module,
        params: PyTree,
        batch_stats: PyTree | None = None,
        preprocess_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
    ) -> None:
        """Initialize predictor.

        Args:
            model: Flax model.
            params: Model parameters.
            batch_stats: Optional batch statistics.
            preprocess_fn: Optional preprocessing function.
            postprocess_fn: Optional postprocessing function.
        """
        self.model = model
        self.params = params
        self.batch_stats = batch_stats
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

        # JIT compile forward function
        self._forward_fn = self._create_forward_fn()

    def _create_forward_fn(self) -> Callable:
        """Create JIT-compiled forward function."""

        @jax.jit
        def forward(params: PyTree, batch_stats: PyTree | None, x: Array) -> Array:
            if batch_stats is not None:
                return cast(
                    Array,
                    self.model.apply(
                        {"params": params, "batch_stats": batch_stats},
                        x,
                        deterministic=True,
                    ),
                )
            else:
                return cast(
                    Array,
                    self.model.apply(
                        {"params": params},
                        x,
                        deterministic=True,
                    ),
                )

        return forward

    def preprocess(self, inputs: Any) -> Array:
        """Preprocess inputs.

        Args:
            inputs: Raw inputs.

        Returns:
            Preprocessed JAX array.
        """
        if self.preprocess_fn is not None:
            return cast(Array, self.preprocess_fn(inputs))
        return jnp.array(inputs)

    def postprocess(self, outputs: Array) -> Any:
        """Postprocess outputs.

        Args:
            outputs: Raw model outputs.

        Returns:
            Postprocessed outputs.
        """
        if self.postprocess_fn is not None:
            return self.postprocess_fn(outputs)
        return outputs

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Make predictions.

        Args:
            inputs: Input data.

        Returns:
            Predictions.
        """
        pass

    def __call__(self, inputs: Any) -> Any:
        """Make predictions.

        Args:
            inputs: Input data.

        Returns:
            Predictions.
        """
        return self.predict(inputs)


class ClassificationPredictor(Predictor):
    """Predictor for classification tasks.

    Example:
        >>> predictor = ClassificationPredictor(
        ...     model, params,
        ...     class_names=["cat", "dog"],
        ... )
        >>> result = predictor.predict(image)
        >>> print(result["class_name"])  # "cat"
    """

    def __init__(
        self,
        model: nn.Module,
        params: PyTree,
        batch_stats: PyTree | None = None,
        class_names: list[str] | None = None,
        preprocess_fn: Callable | None = None,
        top_k: int = 5,
        return_probabilities: bool = True,
    ) -> None:
        """Initialize classification predictor.

        Args:
            model: Flax model.
            params: Model parameters.
            batch_stats: Optional batch statistics.
            class_names: Optional list of class names.
            preprocess_fn: Optional preprocessing function.
            top_k: Number of top predictions to return.
            return_probabilities: Whether to return probabilities.
        """
        super().__init__(model, params, batch_stats, preprocess_fn)
        self.class_names = class_names
        self.top_k = top_k
        self.return_probabilities = return_probabilities

    def predict(
        self,
        inputs: Array | np.ndarray | list,
    ) -> dict[str, Any]:
        """Make classification predictions.

        Args:
            inputs: Input image(s) of shape (C, *spatial) or (B, C, *spatial).

        Returns:
            Dictionary with predictions:
                - class_id: Predicted class index.
                - class_name: Predicted class name (if class_names provided).
                - confidence: Confidence score.
                - top_k_classes: Top-k class indices.
                - top_k_confidences: Top-k confidence scores.
                - probabilities: Full probability distribution (if requested).
        """
        # Preprocess
        x = self.preprocess(inputs)

        # Add batch dimension if needed
        single_input = False
        if x.ndim == len(
            self.model.input_shape if hasattr(self.model, "input_shape") else (3, 32, 32)
        ):
            x = x[None, ...]
            single_input = True

        # Forward pass
        logits = self._forward_fn(self.params, self.batch_stats, x)

        # Convert to probabilities
        probs = jax.nn.softmax(logits, axis=-1)

        # Get predictions
        class_ids = jnp.argmax(probs, axis=-1)
        confidences = jnp.max(probs, axis=-1)

        # Get top-k
        top_k_indices = jnp.argsort(probs, axis=-1)[:, -self.top_k :][:, ::-1]
        top_k_probs = jnp.take_along_axis(probs, top_k_indices, axis=-1)

        # Build result
        result: dict[str, Any] = {
            "class_id": int(class_ids[0]) if single_input else np.array(class_ids),
            "confidence": float(confidences[0]) if single_input else np.array(confidences),
            "top_k_classes": np.array(top_k_indices[0])
            if single_input
            else np.array(top_k_indices),
            "top_k_confidences": np.array(top_k_probs[0])
            if single_input
            else np.array(top_k_probs),
        }

        # Add class names
        if self.class_names is not None:
            if single_input:
                class_id_val = int(class_ids[0])
                result["class_name"] = self.class_names[class_id_val]
                result["top_k_names"] = [
                    self.class_names[int(i)] for i in np.array(top_k_indices[0])
                ]
            else:
                result["class_names"] = [self.class_names[int(i)] for i in np.array(class_ids)]

        # Add full probabilities
        if self.return_probabilities:
            result["probabilities"] = np.array(probs[0]) if single_input else np.array(probs)

        return result

    def predict_batch(
        self,
        inputs: Array | np.ndarray | list,
    ) -> list[dict[str, Any]]:
        """Make predictions on a batch.

        Args:
            inputs: Batch of inputs of shape (B, C, *spatial).

        Returns:
            List of prediction dictionaries.
        """
        x = self.preprocess(inputs)
        logits = self._forward_fn(self.params, self.batch_stats, x)
        probs = jax.nn.softmax(logits, axis=-1)

        results = []
        for i in range(len(probs)):
            class_id = int(jnp.argmax(probs[i]))
            confidence = float(jnp.max(probs[i]))

            result: dict[str, Any] = {
                "class_id": class_id,
                "confidence": confidence,
            }

            if self.class_names is not None:
                result["class_name"] = self.class_names[class_id]

            results.append(result)

        return results


class SegmentationPredictor(Predictor):
    """Predictor for segmentation tasks.

    Example:
        >>> predictor = SegmentationPredictor(model, params)
        >>> mask = predictor.predict(image)
    """

    def __init__(
        self,
        model: nn.Module,
        params: PyTree,
        batch_stats: PyTree | None = None,
        class_names: list[str] | None = None,
        preprocess_fn: Callable | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Initialize segmentation predictor.

        Args:
            model: Flax model.
            params: Model parameters.
            batch_stats: Optional batch statistics.
            class_names: Optional list of class names.
            preprocess_fn: Optional preprocessing function.
            threshold: Threshold for binary segmentation.
        """
        super().__init__(model, params, batch_stats, preprocess_fn)
        self.class_names = class_names
        self.threshold = threshold

    def predict(
        self,
        inputs: Array | np.ndarray,
    ) -> dict[str, Any]:
        """Make segmentation predictions.

        Args:
            inputs: Input image of shape (C, *spatial).

        Returns:
            Dictionary with:
                - mask: Predicted segmentation mask.
                - probabilities: Per-pixel class probabilities.
        """
        x = self.preprocess(inputs)

        # Add batch dimension
        if x.ndim == 3:  # 2D image
            x = x[None, ...]

        # Forward pass
        logits = self._forward_fn(self.params, self.batch_stats, x)

        # Get predictions
        num_classes = logits.shape[1]

        if num_classes == 1:
            # Binary segmentation
            probs = jax.nn.sigmoid(logits)
            mask = (probs > self.threshold).astype(jnp.int32)
        else:
            # Multi-class segmentation
            probs = jax.nn.softmax(logits, axis=1)
            mask = jnp.argmax(probs, axis=1)

        return {
            "mask": np.array(mask[0]),
            "probabilities": np.array(probs[0]),
        }


class FeatureExtractor(Predictor):
    """Predictor that extracts features from intermediate layers.

    Example:
        >>> extractor = FeatureExtractor(model, params, layer_names=["stage_0", "stage_1"])
        >>> features = extractor.predict(image)
    """

    def __init__(
        self,
        model: nn.Module,
        params: PyTree,
        batch_stats: PyTree | None = None,
        preprocess_fn: Callable | None = None,
        return_features: bool = True,
    ) -> None:
        """Initialize feature extractor.

        Args:
            model: Flax model with return_features capability.
            params: Model parameters.
            batch_stats: Optional batch statistics.
            preprocess_fn: Optional preprocessing function.
            return_features: Whether to return intermediate features.
        """
        super().__init__(model, params, batch_stats, preprocess_fn)
        self.return_features = return_features

    def _create_forward_fn(self) -> Callable:
        """Create forward function that returns features."""

        @jax.jit
        def forward(params: PyTree, batch_stats: PyTree | None, x: Array) -> Any:
            variables = {"params": params}
            if batch_stats is not None:
                variables["batch_stats"] = batch_stats

            return self.model.apply(
                variables,
                x,
                training=False,
                return_features=self.return_features,
            )

        return forward

    def predict(
        self,
        inputs: Array | np.ndarray,
    ) -> dict[str, Any]:
        """Extract features.

        Args:
            inputs: Input of shape (C, *spatial) or (B, C, *spatial).

        Returns:
            Dictionary with features from each stage.
        """
        x = self.preprocess(inputs)

        # Add batch dimension
        if x.ndim == 3:
            x = x[None, ...]

        # Forward pass
        output = self._forward_fn(self.params, self.batch_stats, x)

        # Parse output
        if isinstance(output, tuple):
            logits, features = output
            result: dict[str, Any] = {"logits": np.array(logits)}
            if isinstance(features, dict):
                for name, feat in features.items():
                    result[name] = np.array(feat)
            elif isinstance(features, (list, tuple)):
                for i, feat in enumerate(features):
                    result[f"stage_{i}"] = np.array(feat)
        else:
            result = {"output": np.array(output)}

        return result


def create_predictor(
    model: nn.Module,
    params: PyTree,
    task: str = "classification",
    batch_stats: PyTree | None = None,
    **kwargs: Any,
) -> Predictor:
    """Create a predictor for the given task.

    Args:
        model: Flax model.
        params: Model parameters.
        task: Task type ('classification', 'segmentation', 'features').
        batch_stats: Optional batch statistics.
        **kwargs: Additional predictor arguments.

    Returns:
        Appropriate predictor instance.
    """
    if task == "classification":
        return ClassificationPredictor(model, params, batch_stats, **kwargs)
    elif task == "segmentation":
        return SegmentationPredictor(model, params, batch_stats, **kwargs)
    elif task == "features":
        return FeatureExtractor(model, params, batch_stats, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")
