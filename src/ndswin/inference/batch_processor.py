"""Batch processing utilities for NDSwin-JAX.

This module provides efficient batch processing for inference.
"""

import time
from collections.abc import Callable, Generator, Iterator
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from ndswin.inference.predictor import Predictor
from ndswin.types import Array, PyTree


class BatchProcessor:
    """Efficient batch processor for inference.

    Handles batching, parallel processing, and result aggregation.

    Example:
        >>> processor = BatchProcessor(predictor, batch_size=32)
        >>> results = processor.process(images)
    """

    def __init__(
        self,
        predictor: Predictor,
        batch_size: int = 32,
        prefetch: int = 2,
        show_progress: bool = True,
    ) -> None:
        """Initialize batch processor.

        Args:
            predictor: Predictor instance.
            batch_size: Batch size for processing.
            prefetch: Number of batches to prefetch.
            show_progress: Whether to show progress bar.
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.show_progress = show_progress

    def process(
        self,
        data: Any,
        num_samples: int | None = None,
    ) -> list[Any]:
        """Process data in batches.

        Args:
            data: Input data (list, array, or iterator).
            num_samples: Total number of samples (for progress).

        Returns:
            List of predictions.
        """
        results: list[Any] = []

        # Create batches
        batches = self._create_batches(data)

        # Get total for progress
        if num_samples is None:
            if isinstance(data, (list, np.ndarray)):
                num_samples = len(data)

        num_batches = (
            (num_samples + self.batch_size - 1) // self.batch_size if num_samples else None
        )

        # Process batches
        start_time = time.time()

        for i, batch in enumerate(batches):
            # Make predictions
            batch_results = self.predictor.predict(batch)

            # Handle different result types
            if isinstance(batch_results, dict):
                # Single result for batch
                results.append(batch_results)
            elif isinstance(batch_results, list):
                # List of results
                results.extend(batch_results)
            else:
                results.append(batch_results)

            # Show progress
            if self.show_progress and num_batches:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (num_batches - i - 1)
                print(f"\rProcessing: {i + 1}/{num_batches} batches, ETA: {eta:.1f}s", end="")

        if self.show_progress:
            print()

        return results

    def _create_batches(
        self,
        data: Any,
    ) -> Generator[Array, None, None]:
        """Create batches from data.

        Args:
            data: Input data.

        Yields:
            Batches as JAX arrays.
        """
        if isinstance(data, np.ndarray):
            # NumPy array
            for i in range(0, len(data), self.batch_size):
                batch_np = data[i : i + self.batch_size]
                yield jnp.array(batch_np)

        elif isinstance(data, list):
            # List of samples
            for i in range(0, len(data), self.batch_size):
                batch_list = data[i : i + self.batch_size]
                yield jnp.stack([jnp.array(x) for x in batch_list])

        elif isinstance(data, (Iterator, Generator)):
            # Iterator
            current_batch = []
            for item in data:
                current_batch.append(item)
                if len(current_batch) >= self.batch_size:
                    yield jnp.stack([jnp.array(x) for x in current_batch])
                    current_batch = []

            # Yield remaining
            if current_batch:
                yield jnp.stack([jnp.array(x) for x in current_batch])

    def process_generator(
        self,
        data: Iterator,
    ) -> Generator[Any, None, None]:
        """Process data lazily as a generator.

        Args:
            data: Input data iterator.

        Yields:
            Predictions for each batch.
        """
        for batch in self._create_batches(data):
            yield self.predictor.predict(batch)


def process_dataset(
    predictor: Predictor,
    dataset: Any,
    batch_size: int = 32,
    return_labels: bool = False,
) -> Any:
    """Process a dataset with a predictor.

    Args:
        predictor: Predictor instance.
        dataset: Dataset with 'image' and optionally 'label' keys.
        batch_size: Batch size.
        return_labels: Whether to return labels.

    Returns:
        Tuple of (predictions, labels) or just predictions.
    """
    predictions: list[Any] = []
    labels: list[Any] | None = [] if return_labels else None

    for batch in dataset:
        images = batch["image"]
        preds = predictor.predict(images)

        if isinstance(preds, dict):
            predictions.append(preds)
        else:
            predictions.extend(preds if isinstance(preds, list) else [preds])

        if labels is not None and "label" in batch:
            labels.extend(batch["label"].tolist())

    if return_labels:
        return predictions, labels
    return predictions


class StreamingProcessor:
    """Streaming processor for continuous inference.

    Useful for processing data streams in real-time.

    Example:
        >>> processor = StreamingProcessor(predictor)
        >>> for frame in video_stream:
        ...     result = processor.process_single(frame)
    """

    def __init__(
        self,
        predictor: Predictor,
        buffer_size: int = 1,
    ) -> None:
        """Initialize streaming processor.

        Args:
            predictor: Predictor instance.
            buffer_size: Buffer size for batching.
        """
        self.predictor = predictor
        self.buffer_size = buffer_size
        self._buffer: list[Array] = []
        self._results: list[Any] = []

    def process_single(self, input_data: Any) -> Any:
        """Process a single input immediately.

        Args:
            input_data: Single input.

        Returns:
            Prediction result.
        """
        x = self.predictor.preprocess(input_data)
        if x.ndim == 3:  # Add batch dimension
            x = x[None, ...]

        return self.predictor.predict(x)

    def add_to_buffer(self, input_data: Any) -> list[Any] | None:
        """Add input to buffer and process when full.

        Args:
            input_data: Single input.

        Returns:
            List of results if buffer was processed, None otherwise.
        """
        x = self.predictor.preprocess(input_data)
        self._buffer.append(x)

        if len(self._buffer) >= self.buffer_size:
            # Process buffer
            batch = jnp.stack(self._buffer)
            results = self.predictor.predict(batch)
            self._buffer = []

            if isinstance(results, dict):
                return [results]
            return cast(list[Any], results)

        return None

    def flush(self) -> list[Any] | None:
        """Process remaining items in buffer.

        Returns:
            List of results if buffer had items, None otherwise.
        """
        if not self._buffer:
            return None

        batch = jnp.stack(self._buffer)
        results = self.predictor.predict(batch)
        self._buffer = []

        if isinstance(results, dict):
            return [results]
        return cast(list[Any], results)


class ParallelProcessor:
    """Process data in parallel across devices.

    Example:
        >>> processor = ParallelProcessor(predictor)
        >>> results = processor.process(large_dataset)
    """

    def __init__(
        self,
        predictor: Predictor,
        devices: list[Any] | None = None,
    ) -> None:
        """Initialize parallel processor.

        Args:
            predictor: Predictor instance.
            devices: List of devices (default: all available).
        """
        self.predictor = predictor
        self.devices = devices or jax.devices()
        self.num_devices = len(self.devices)

        # Create pmap'd forward function
        self._forward_pmap = self._create_pmap_forward()

    def _create_pmap_forward(self) -> Callable:
        """Create pmapped forward function."""

        @jax.pmap
        def forward(params: PyTree, batch_stats: PyTree | None, x: Array) -> Array:
            if batch_stats is not None:
                return cast(
                    Array,
                    self.predictor.model.apply(
                        {"params": params, "batch_stats": batch_stats},
                        x,
                        training=False,
                    ),
                )
            else:
                return cast(
                    Array,
                    self.predictor.model.apply(
                        {"params": params},
                        x,
                        training=False,
                    ),
                )

        return cast(Callable, forward)

    def process(
        self,
        data: np.ndarray,
        batch_size_per_device: int = 32,
    ) -> np.ndarray:
        """Process data in parallel.

        Args:
            data: Input data of shape (N, C, *spatial).
            batch_size_per_device: Batch size per device.

        Returns:
            Predictions.
        """
        total_batch_size = batch_size_per_device * self.num_devices
        num_samples = len(data)

        # Replicate parameters
        params = jax.device_put_replicated(self.predictor.params, self.devices)
        batch_stats = None
        if self.predictor.batch_stats is not None:
            batch_stats = jax.device_put_replicated(self.predictor.batch_stats, self.devices)

        results = []

        for i in range(0, num_samples, total_batch_size):
            batch_np = data[i : i + total_batch_size]

            # Pad if needed
            if len(batch_np) < total_batch_size:
                pad_size = total_batch_size - len(batch_np)
                batch_np = np.concatenate(
                    [
                        batch_np,
                        np.zeros((pad_size,) + batch_np.shape[1:], dtype=batch_np.dtype),
                    ]
                )

            # Reshape for pmap: (num_devices, batch_per_device, ...)
            batch_np = batch_np.reshape(
                self.num_devices, batch_size_per_device, *batch_np.shape[1:]
            )
            batch_jax = jnp.array(batch_np)

            # Forward pass
            logits = self._forward_pmap(params, batch_stats, batch_jax)

            # Reshape back
            logits_reshaped = logits.reshape(-1, logits.shape[-1])

            # Remove padding
            valid_count = min(total_batch_size, num_samples - i)
            results.append(np.array(logits_reshaped[:valid_count]))

        return np.concatenate(results)
