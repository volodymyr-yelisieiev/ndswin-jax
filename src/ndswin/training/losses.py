"""Loss functions for NDSwin-JAX.

This module provides various loss functions for training.
"""

import jax
import jax.numpy as jnp

from ndswin.types import Array, LossValue


def cross_entropy_loss(
    logits: Array,
    labels: Array,
    label_smoothing: float = 0.0,
) -> LossValue:
    """Compute cross-entropy loss.

    Args:
        logits: Predicted logits of shape (B, num_classes).
        labels: Ground truth labels of shape (B,) (class indices) or
            (B, num_classes) (one-hot or soft labels).
        label_smoothing: Label smoothing coefficient.

    Returns:
        Scalar loss value.
    """
    num_classes = logits.shape[-1]

    # Convert labels to one-hot if needed
    if labels.ndim == 1:
        labels = jax.nn.one_hot(labels, num_classes)

    # Apply label smoothing
    labels = labels * (1 - label_smoothing) + label_smoothing / num_classes

    # Compute log softmax
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Cross-entropy
    loss = -jnp.sum(labels * log_probs, axis=-1)

    return jnp.mean(loss)


def label_smoothing_cross_entropy(
    logits: Array,
    labels: Array,
    smoothing: float = 0.1,
) -> LossValue:
    """Compute cross-entropy loss with label smoothing.

    Args:
        logits: Predicted logits of shape (B, num_classes).
        labels: Ground truth labels of shape (B,) or (B, num_classes).
        smoothing: Label smoothing coefficient.

    Returns:
        Scalar loss value.
    """
    return cross_entropy_loss(logits, labels, label_smoothing=smoothing)


def focal_loss(
    logits: Array,
    labels: Array,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> LossValue:
    """Compute focal loss for handling class imbalance.

    Focal loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: Predicted logits of shape (B, num_classes).
        labels: Ground truth labels of shape (B,) or (B, num_classes).
        gamma: Focusing parameter (higher = more focus on hard examples).
        alpha: Weighting factor (None for no weighting).

    Returns:
        Scalar loss value.
    """
    num_classes = logits.shape[-1]

    # Convert labels to one-hot if needed
    if labels.ndim == 1:
        labels = jax.nn.one_hot(labels, num_classes)

    # Compute softmax probabilities
    probs = jax.nn.softmax(logits, axis=-1)

    # Compute log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Get probability of correct class
    p_t = jnp.sum(labels * probs, axis=-1)

    # Focal weight
    focal_weight = (1 - p_t) ** gamma

    # Cross-entropy
    ce = -jnp.sum(labels * log_probs, axis=-1)

    # Apply focal weight
    loss = focal_weight * ce

    # Apply alpha weighting if specified
    if alpha is not None:
        loss = alpha * loss

    return jnp.mean(loss)


def binary_cross_entropy_with_logits(
    logits: Array,
    labels: Array,
    pos_weight: float | None = None,
) -> LossValue:
    """Compute binary cross-entropy with logits.

    Args:
        logits: Predicted logits of shape (B,) or (B, num_labels).
        labels: Ground truth labels of same shape as logits.
        pos_weight: Positive class weight for handling imbalance.

    Returns:
        Scalar loss value.
    """
    # Numerically stable BCE
    max_val = jnp.clip(logits, 0, None)
    loss = max_val - logits * labels + jnp.log(1 + jnp.exp(-jnp.abs(logits)))

    if pos_weight is not None:
        # Weight positive samples
        loss = labels * loss * pos_weight + (1 - labels) * loss

    return jnp.mean(loss)


def dice_loss(
    logits: Array,
    labels: Array,
    smooth: float = 1e-6,
) -> LossValue:
    """Compute Dice loss for segmentation.

    Args:
        logits: Predicted logits of shape (B, C, *spatial).
        labels: Ground truth masks of same shape.
        smooth: Smoothing factor.

    Returns:
        Scalar loss value.
    """
    # Apply softmax
    probs = jax.nn.softmax(logits, axis=1)

    # Flatten spatial dimensions
    batch_size = probs.shape[0]
    num_classes = probs.shape[1]
    probs_flat = probs.reshape(batch_size, num_classes, -1)
    labels_flat = labels.reshape(batch_size, num_classes, -1)

    # Compute intersection and union
    intersection = jnp.sum(probs_flat * labels_flat, axis=-1)
    union = jnp.sum(probs_flat + labels_flat, axis=-1)

    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Dice loss (1 - dice)
    return 1.0 - jnp.mean(dice)


def mse_loss(
    predictions: Array,
    targets: Array,
    reduction: str = "mean",
) -> LossValue:
    """Compute mean squared error loss.

    Args:
        predictions: Predicted values.
        targets: Target values.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Loss value.
    """
    squared_error = (predictions - targets) ** 2

    if reduction == "mean":
        return jnp.mean(squared_error)
    elif reduction == "sum":
        return jnp.sum(squared_error)
    else:
        return squared_error


def l1_loss(
    predictions: Array,
    targets: Array,
    reduction: str = "mean",
) -> LossValue:
    """Compute L1 (mean absolute error) loss.

    Args:
        predictions: Predicted values.
        targets: Target values.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Loss value.
    """
    abs_error = jnp.abs(predictions - targets)

    if reduction == "mean":
        return jnp.mean(abs_error)
    elif reduction == "sum":
        return jnp.sum(abs_error)
    else:
        return abs_error


def huber_loss(
    predictions: Array,
    targets: Array,
    delta: float = 1.0,
    reduction: str = "mean",
) -> LossValue:
    """Compute Huber loss (smooth L1 loss).

    Args:
        predictions: Predicted values.
        targets: Target values.
        delta: Threshold for L1 vs L2.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Loss value.
    """
    abs_error = jnp.abs(predictions - targets)
    quadratic = jnp.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear

    if reduction == "mean":
        return jnp.mean(loss)
    elif reduction == "sum":
        return jnp.sum(loss)
    else:
        return loss


def kl_divergence(
    p_logits: Array,
    q_logits: Array,
    temperature: float = 1.0,
) -> LossValue:
    """Compute KL divergence between two distributions.

    KL(P || Q) = sum(P * log(P/Q))

    Args:
        p_logits: Logits for distribution P.
        q_logits: Logits for distribution Q.
        temperature: Temperature for softmax.

    Returns:
        Scalar KL divergence.
    """
    p = jax.nn.softmax(p_logits / temperature, axis=-1)
    q = jax.nn.softmax(q_logits / temperature, axis=-1)

    kl = p * (jnp.log(p + 1e-10) - jnp.log(q + 1e-10))

    return jnp.mean(jnp.sum(kl, axis=-1))


def distillation_loss(
    student_logits: Array,
    teacher_logits: Array,
    labels: Array,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> LossValue:
    """Compute knowledge distillation loss.

    Combines soft (KL divergence with teacher) and hard (CE with labels) losses.

    Args:
        student_logits: Student model logits.
        teacher_logits: Teacher model logits.
        labels: Ground truth labels.
        temperature: Temperature for softening distributions.
        alpha: Weight for soft loss (1-alpha for hard loss).

    Returns:
        Combined distillation loss.
    """
    # Soft loss (KL divergence with teacher)
    soft_loss = kl_divergence(student_logits, teacher_logits, temperature)

    # Hard loss (cross-entropy with labels)
    hard_loss = cross_entropy_loss(student_logits, labels)

    # Combine
    return alpha * (temperature**2) * soft_loss + (1 - alpha) * hard_loss
