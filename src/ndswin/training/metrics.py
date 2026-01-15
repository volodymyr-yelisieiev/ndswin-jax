"""Metrics for training and evaluation.

This module provides various metrics for monitoring training progress
and evaluating model performance.
"""

import jax.numpy as jnp

from ndswin.types import Array, MetricValue


def accuracy(
    logits: Array,
    labels: Array,
) -> MetricValue:
    """Compute classification accuracy.

    Args:
        logits: Predicted logits of shape (B, num_classes).
        labels: Ground truth labels of shape (B,) or (B, num_classes).

    Returns:
        Accuracy as a float.
    """
    predictions = jnp.argmax(logits, axis=-1)

    if labels.ndim > 1:
        labels = jnp.argmax(labels, axis=-1)

    return jnp.mean(predictions == labels)


def top_k_accuracy(
    logits: Array,
    labels: Array,
    k: int = 5,
) -> MetricValue:
    """Compute top-k accuracy.

    Args:
        logits: Predicted logits of shape (B, num_classes).
        labels: Ground truth labels of shape (B,) or (B, num_classes).
        k: Number of top predictions to consider.

    Returns:
        Top-k accuracy as a float.
    """
    if labels.ndim > 1:
        labels = jnp.argmax(labels, axis=-1)

    # Get top-k predictions
    top_k_preds = jnp.argsort(logits, axis=-1)[:, -k:]

    # Check if true label is in top-k
    correct = jnp.any(top_k_preds == labels[:, None], axis=-1)

    return jnp.mean(correct)


def precision(
    predictions: Array,
    labels: Array,
    num_classes: int | None = None,
    average: str = "macro",
) -> MetricValue:
    """Compute precision.

    Args:
        predictions: Predicted class indices of shape (B,).
        labels: Ground truth labels of shape (B,).
        num_classes: Number of classes (inferred if None).
        average: 'macro', 'micro', or 'weighted'.

    Returns:
        Precision score.
    """
    if num_classes is None:
        num_classes = int(jnp.maximum(predictions.max(), labels.max()) + 1)

    precisions = []
    weights = []

    for c in range(num_classes):
        pred_mask = predictions == c
        true_mask = labels == c

        true_positives = jnp.sum(pred_mask & true_mask)
        predicted_positives = jnp.sum(pred_mask)

        prec = jnp.where(
            predicted_positives > 0,
            true_positives / predicted_positives,
            0.0,
        )
        precisions.append(prec)
        weights.append(jnp.sum(true_mask))

    precisions_arr = jnp.array(precisions)
    weights_arr = jnp.array(weights)

    if average == "macro":
        return jnp.mean(precisions_arr)
    elif average == "micro":
        total_tp = sum(jnp.sum((predictions == c) & (labels == c)) for c in range(num_classes))
        total_pred = len(predictions)
        return total_tp / total_pred
    elif average == "weighted":
        return jnp.sum(precisions_arr * weights_arr) / jnp.sum(weights_arr)
    else:
        raise ValueError(f"Unknown average: {average}")


def recall(
    predictions: Array,
    labels: Array,
    num_classes: int | None = None,
    average: str = "macro",
) -> MetricValue:
    """Compute recall.

    Args:
        predictions: Predicted class indices of shape (B,).
        labels: Ground truth labels of shape (B,).
        num_classes: Number of classes (inferred if None).
        average: 'macro', 'micro', or 'weighted'.

    Returns:
        Recall score.
    """
    if num_classes is None:
        num_classes = int(jnp.maximum(predictions.max(), labels.max()) + 1)

    recalls = []
    weights = []

    for c in range(num_classes):
        pred_mask = predictions == c
        true_mask = labels == c

        true_positives = jnp.sum(pred_mask & true_mask)
        actual_positives = jnp.sum(true_mask)

        rec = jnp.where(
            actual_positives > 0,
            true_positives / actual_positives,
            0.0,
        )
        recalls.append(rec)
        weights.append(actual_positives)

    recalls_arr = jnp.array(recalls)
    weights_arr = jnp.array(weights)

    if average == "macro":
        return jnp.mean(recalls_arr)
    elif average == "weighted":
        return jnp.sum(recalls_arr * weights_arr) / jnp.sum(weights_arr)
    else:
        raise ValueError(f"Unknown average: {average}")


def f1_score(
    predictions: Array,
    labels: Array,
    num_classes: int | None = None,
    average: str = "macro",
) -> MetricValue:
    """Compute F1 score.

    Args:
        predictions: Predicted class indices of shape (B,).
        labels: Ground truth labels of shape (B,).
        num_classes: Number of classes (inferred if None).
        average: 'macro', 'micro', or 'weighted'.

    Returns:
        F1 score.
    """
    prec = precision(predictions, labels, num_classes, average)
    rec = recall(predictions, labels, num_classes, average)

    return 2 * prec * rec / (prec + rec + 1e-10)


def confusion_matrix(
    predictions: Array,
    labels: Array,
    num_classes: int,
) -> Array:
    """Compute confusion matrix.

    Args:
        predictions: Predicted class indices of shape (B,).
        labels: Ground truth labels of shape (B,).
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    cm = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)

    for i in range(len(predictions)):
        cm = cm.at[labels[i], predictions[i]].add(1)

    return cm


class MetricTracker:
    """Tracks and averages metrics over batches.

    Example:
        >>> tracker = MetricTracker()
        >>> for batch in data:
        ...     loss = compute_loss(batch)
        ...     tracker.update({"loss": loss, "accuracy": acc})
        >>> print(tracker.compute())  # {"loss": avg_loss, "accuracy": avg_acc}
    """

    def __init__(self) -> None:
        """Initialize metric tracker."""
        self._metrics: dict[str, list[float]] = {}
        self._counts: dict[str, list[int]] = {}

    def update(
        self,
        metrics: dict[str, float],
        count: int = 1,
    ) -> None:
        """Update metrics with new values.

        Args:
            metrics: Dictionary of metric names to values.
            count: Number of samples (for weighted averaging).
        """
        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = []
                self._counts[name] = []

            self._metrics[name].append(float(value))
            self._counts[name].append(count)

    def compute(self) -> dict[str, float]:
        """Compute averaged metrics.

        Returns:
            Dictionary of metric names to averaged values.
        """
        result = {}
        for name in self._metrics:
            values = jnp.array(self._metrics[name])
            counts = jnp.array(self._counts[name])

            # Weighted average
            result[name] = float(jnp.sum(values * counts) / jnp.sum(counts))

        return result

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._counts.clear()

    def __str__(self) -> str:
        """Return string representation of current metrics."""
        metrics = self.compute()
        parts = [f"{name}: {value:.4f}" for name, value in metrics.items()]
        return " | ".join(parts)


class EarlyStopping:
    """Early stopping callback.

    Stops training when a metric stops improving.

    Example:
        >>> early_stopping = EarlyStopping(patience=5, mode="min")
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     if early_stopping(val_loss):
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' (lower is better) or 'max' (higher is better).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value: float | None = None
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, value: float, epoch: int = 0) -> bool:
        """Check if training should stop.

        Args:
            value: Current metric value.
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_value = None
        self.counter = 0
        self.best_epoch = 0


def dice_coefficient_from_probs(probs: Array, labels_onehot: Array, smooth: float = 1e-6) -> Array:
    """Compute Dice coefficient per class given soft probabilities and one-hot labels.

    Args:
        probs: Soft probabilities with shape (B, C, *spatial).
        labels_onehot: One-hot labels with shape (B, C, *spatial).
    Returns:
        Dice per class averaged over batch.
    """
    batch_size = probs.shape[0]
    num_classes = probs.shape[1]
    probs_flat = probs.reshape(batch_size, num_classes, -1)
    labels_flat = labels_onehot.reshape(batch_size, num_classes, -1)

    intersection = jnp.sum(probs_flat * labels_flat, axis=-1)
    union = jnp.sum(probs_flat + labels_flat, axis=-1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    # average over batch and classes
    return jnp.mean(dice, axis=0)


def compute_segmentation_metrics(logits: Array, labels: Array, prefix: str = "") -> dict[str, float]:
    """Compute segmentation metrics: mean Dice and voxel-wise accuracy.

    Args:
        logits: Logits with shape (B, C, *spatial) or (B, 1, *spatial) for binary.
        labels: Ground truth labels with shape (B, *spatial) or one-hot (B, C, *spatial).
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metrics.
    """
    # Convert labels to one-hot if necessary
    if labels.ndim == logits.ndim - 1:
        # labels: (B, *spatial) -> one-hot
        num_classes = logits.shape[1]
        onehot = jax.nn.one_hot(labels, num_classes)
        # move class dim to position 1
        onehot = jnp.moveaxis(onehot, -1, 1)
    else:
        onehot = labels

    # softmax probabilities
    probs = jax.nn.softmax(logits, axis=1)

    dice_per_class = dice_coefficient_from_probs(probs, onehot)
    mean_dice = float(jnp.mean(dice_per_class))

    # Voxel-wise accuracy (argmax)
    preds = jnp.argmax(probs, axis=1)
    if labels.ndim > 1:
        true_labels = jnp.argmax(onehot, axis=1)
    else:
        true_labels = labels

    voxel_acc = float(jnp.mean(preds == true_labels))

    metrics = {
        f"{prefix}dice": mean_dice,
        f"{prefix}voxel_accuracy": voxel_acc,
    }
    return metrics


def compute_metrics(
    logits: Array,
    labels: Array,
    prefix: str = "",
) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        logits: Predicted logits of shape (B, num_classes).
        labels: Ground truth labels.
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metrics.
    """
    if labels.ndim > 1:
        labels = jnp.argmax(labels, axis=-1)

    metrics = {
        f"{prefix}accuracy": float(accuracy(logits, labels)),
        f"{prefix}top5_accuracy": float(top_k_accuracy(logits, labels, k=5)),
    }

    return metrics
