"""Data augmentation utilities for training."""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict


def random_crop_and_flip(
    rng: jax.Array,
    image: jnp.ndarray,
    crop_size: tuple[int, int],
    padding: int = 4,
    *,
    apply_crop: bool = True,
    apply_flip: bool = True,
) -> jnp.ndarray:
    """Apply random crop and optional horizontal flip augmentation.

    Args:
        rng: Random number generator key.
        image: Input image of shape (C, H, W).
        crop_size: Target crop size (H, W).
        padding: Padding size for random crop when ``apply_crop`` is True.
        apply_crop: Whether to perform random cropping.
        apply_flip: Whether to perform random horizontal flipping.

    Returns:
        Augmented image with the same shape as ``image``.
    """
    rng_crop, rng_flip = jax.random.split(rng)
    out = image

    if apply_crop:
        rng_top, rng_left = jax.random.split(rng_crop)

        padded = out
        if padding > 0:
            padded = jnp.pad(
                out,
                ((0, 0), (padding, padding), (padding, padding)),
                mode="reflect",
            )

        channels, height, width = padded.shape
        crop_height, crop_width = crop_size

        max_top = max(1, height - crop_height + 1)
        max_left = max(1, width - crop_width + 1)

        top = jax.random.randint(rng_top, (), 0, max_top)
        left = jax.random.randint(rng_left, (), 0, max_left)

        out = jax.lax.dynamic_slice(
            padded,
            (0, top, left),
            (channels, crop_height, crop_width),
        )

    if apply_flip:
        should_flip = jax.random.bernoulli(rng_flip)
        out = jax.lax.cond(should_flip, lambda x: jnp.flip(x, axis=2), lambda x: x, out)

    return out


def color_jitter(
    rng: jax.Array,
    image: jnp.ndarray,
    *,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> jnp.ndarray:
    """Apply simple color jitter (brightness/contrast) in-place."""
    if brightness <= 0.0 and contrast <= 0.0:
        return image

    rng_brightness, rng_contrast = jax.random.split(rng)
    out = image

    if brightness > 0.0:
        delta = jax.random.uniform(
            rng_brightness,
            shape=(1, 1, 1),
            minval=-brightness,
            maxval=brightness,
        )
        out = out + delta

    if contrast > 0.0:
        factor = 1.0 + jax.random.uniform(
            rng_contrast,
            shape=(1, 1, 1),
            minval=-contrast,
            maxval=contrast,
        )
        mean = jnp.mean(out, axis=(1, 2), keepdims=True)
        out = (out - mean) * factor + mean

    return jnp.clip(out, 0.0, 1.0)


def mixup(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply mixup augmentation.
    
    Args:
        rng: Random number generator key
        images: Batch of images
        labels: Batch of labels (one-hot encoded)
        alpha: Mixup interpolation strength
        
    Returns:
        Mixed images and labels
    """
    if alpha <= 0:
        return images, labels
    
    batch_size = images.shape[0]
    
    # Sample lambda from Beta distribution
    rng1, rng2 = jax.random.split(rng)
    lam = jax.random.beta(rng1, alpha, alpha)
    
    # Shuffle indices
    indices = jax.random.permutation(rng2, batch_size)
    
    # Mix images and labels
    mixed_images = lam * images + (1 - lam) * images[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    return mixed_images, mixed_labels


def cutmix(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply cutmix augmentation.
    
    Args:
        rng: Random number generator key
        images: Batch of images (B, C, H, W)
        labels: Batch of labels (one-hot encoded)
        alpha: Cutmix interpolation strength
        
    Returns:
        Mixed images and labels
    """
    if alpha <= 0:
        return images, labels
    
    batch_size, _, height, width = images.shape
    
    rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 5)
    
    # Sample lambda from Beta distribution
    lam = jax.random.beta(rng1, alpha, alpha)
    
    # Compute bounding box
    cut_ratio = jnp.sqrt(1.0 - lam)
    cut_h = jnp.round(height * cut_ratio).astype(jnp.int32)
    cut_w = jnp.round(width * cut_ratio).astype(jnp.int32)
    
    # Random center point
    cx = jax.random.randint(rng2, (), 0, width)
    cy = jax.random.randint(rng3, (), 0, height)
    
    # Compute box coordinates
    x1 = jnp.clip(cx - cut_w // 2, 0, width)
    y1 = jnp.clip(cy - cut_h // 2, 0, height)
    x2 = jnp.clip(cx + cut_w // 2, 0, width)
    y2 = jnp.clip(cy + cut_h // 2, 0, height)
    
    # Shuffle indices
    indices = jax.random.permutation(rng4, batch_size)
    
    # Create mask
    mask = jnp.ones((height, width))
    mask = mask.at[y1:y2, x1:x2].set(0.0)
    mask = mask[None, None, :, :]  # (1, 1, H, W)
    
    # Mix images
    mixed_images = images * mask + images[indices] * (1 - mask)
    
    # Adjust lambda based on actual box area
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (width * height)
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    return mixed_images, mixed_labels


def apply_augmentation(
    rng: jax.Array,
    batch: Dict[str, jnp.ndarray],
    config: ConfigDict,
    num_classes: int,
) -> Dict[str, jnp.ndarray]:
    """Apply data augmentation to a batch.
    
    Args:
        rng: Random number generator key
        batch: Batch dictionary with 'image' and 'label' keys
        config: Configuration with augmentation parameters
        num_classes: Number of classes for one-hot encoding
        
    Returns:
        Augmented batch
    """
    if not getattr(config, "use_augmentation", False):
        return batch

    if len(config.input_resolution) != 2:
        labels = jax.nn.one_hot(batch["label"], num_classes)
        return {"image": batch["image"], "label_onehot": labels, "label": batch["label"]}
    
    images = batch["image"]
    labels = jax.nn.one_hot(batch["label"], num_classes)
    
    rng, crop_rng = jax.random.split(rng)
    rng, jitter_rng = jax.random.split(rng)
    rng, prob_rng = jax.random.split(rng)
    rng, mix_rng = jax.random.split(rng)

    # Spatial augmentations (crop/flip)
    apply_crop = bool(getattr(config, "use_random_crop", False))
    apply_flip = bool(getattr(config, "use_random_flip", False))
    if apply_crop or apply_flip:
        if len(config.input_resolution) == 2:
            padding = int(getattr(config, "random_crop_padding", 4))
            crop_size = tuple(config.input_resolution)
            crop_keys = jax.random.split(crop_rng, images.shape[0])
            images = jax.vmap(
                lambda key, img: random_crop_and_flip(
                    key,
                    img,
                    crop_size,
                    padding=padding,
                    apply_crop=apply_crop,
                    apply_flip=apply_flip,
                )
            )(crop_keys, images)

    # Color jitter
    if bool(getattr(config, "use_color_jitter", False)):
        brightness = float(getattr(config, "color_jitter_brightness", 0.2))
        contrast = float(getattr(config, "color_jitter_contrast", 0.2))
        jitter_keys = jax.random.split(jitter_rng, images.shape[0])
        images = jax.vmap(
            lambda key, img: color_jitter(
                key,
                img,
                brightness=brightness,
                contrast=contrast,
            )
        )(jitter_keys, images)

    # Apply mixup or cutmix
    aug_prob = getattr(config, "augmentation_prob", 0.5)
    should_augment = jax.random.uniform(prob_rng) < aug_prob

    if should_augment:
        mixup_alpha = getattr(config, "mixup_alpha", 0.0)
        cutmix_alpha = getattr(config, "cutmix_alpha", 0.0)

        if mixup_alpha > 0 and cutmix_alpha > 0:
            mix_rng, choice_rng = jax.random.split(mix_rng)
            use_mixup = jax.random.uniform(choice_rng) < 0.5
            if use_mixup:
                images, labels = mixup(mix_rng, images, labels, mixup_alpha)
            else:
                images, labels = cutmix(mix_rng, images, labels, cutmix_alpha)
        elif mixup_alpha > 0:
            images, labels = mixup(mix_rng, images, labels, mixup_alpha)
        elif cutmix_alpha > 0:
            images, labels = cutmix(mix_rng, images, labels, cutmix_alpha)
    
    return {"image": images, "label_onehot": labels, "label": batch["label"]}


__all__ = [
    "random_crop_and_flip",
    "color_jitter",
    "mixup",
    "cutmix",
    "apply_augmentation",
]
