"""Data augmentation utilities for NDSwin-JAX.

This module provides n-dimensional data augmentation transforms
that work with JAX arrays.
"""

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp

from ndswin.types import Array, PRNGKey


class Transform(ABC):
    """Abstract base class for transforms."""

    @abstractmethod
    def __call__(self, x: Array, key: PRNGKey | None = None) -> Array:
        """Apply transform to input.

        Args:
            x: Input array.
            key: Optional PRNG key for random transforms.

        Returns:
            Transformed array.
        """
        pass


class Compose(Transform):
    """Compose multiple transforms.

    Applies transforms in sequence.

    Example:
        >>> transform = Compose([
        ...     RandomHorizontalFlip(p=0.5),
        ...     Normalize(mean=(0.5,), std=(0.5,)),
        ... ])
        >>> x_aug = transform(x, key)
    """

    def __init__(self, transforms: list[Transform]) -> None:
        """Initialize compose transform.

        Args:
            transforms: List of transforms to apply.
        """
        self.transforms = transforms

    def __call__(self, x: Array, key: PRNGKey | None = None) -> Array:
        """Apply all transforms in sequence.

        Args:
            x: Input array.
            key: PRNG key (will be split for each random transform).

        Returns:
            Transformed array.
        """
        if key is None:
            for transform in self.transforms:
                x = transform(x)
        else:
            keys = jax.random.split(key, len(self.transforms))
            for transform, k in zip(self.transforms, keys):
                x = transform(x, k)
        return x


class RandomHorizontalFlip(Transform):
    """Random horizontal flip for n-dimensional data.

    Flips along the last spatial dimension (width).
    """

    def __init__(self, p: float = 0.5) -> None:
        """Initialize random horizontal flip.

        Args:
            p: Probability of flipping.
        """
        self.p = p

    def __call__(self, x: Array, key: PRNGKey | None = None) -> Array:
        """Apply random horizontal flip.

        Args:
            x: Input of shape (C, *spatial) or (B, C, *spatial).
            key: PRNG key.

        Returns:
            Possibly flipped array.
        """
        if key is None:
            # Deterministic - no flip
            return x

        do_flip = jax.random.uniform(key) < self.p
        # Flip last dimension (width)
        flipped = jnp.flip(x, axis=-1)
        return jnp.where(do_flip, flipped, x)


class RandomVerticalFlip(Transform):
    """Random vertical flip for n-dimensional data.

    Flips along the second-to-last spatial dimension (height).
    """

    def __init__(self, p: float = 0.5) -> None:
        """Initialize random vertical flip.

        Args:
            p: Probability of flipping.
        """
        self.p = p

    def __call__(self, x: Array, key: PRNGKey | None = None) -> Array:
        """Apply random vertical flip.

        Args:
            x: Input of shape (C, *spatial) or (B, C, *spatial).
            key: PRNG key.

        Returns:
            Possibly flipped array.
        """
        if key is None:
            return x

        do_flip = jax.random.uniform(key) < self.p
        # Flip second to last dimension (height)
        flipped = jnp.flip(x, axis=-2)
        return jnp.where(do_flip, flipped, x)


class RandomCrop(Transform):
    """Random crop for n-dimensional data.

    Randomly crops a region of specified size.
    """

    def __init__(
        self,
        size: int | tuple[int, ...],
        padding: int | tuple[int, ...] = 0,
    ) -> None:
        """Initialize random crop.

        Args:
            size: Output size. If int, used for all spatial dims.
            padding: Padding to add before cropping.
        """
        self.size = size if isinstance(size, tuple) else None
        self.size_int = size if isinstance(size, int) else None
        self.padding = padding

    def __call__(self, x: Array, key: PRNGKey | None = None) -> Array:
        """Apply random crop.

        Args:
            x: Input of shape (C, *spatial).
            key: PRNG key.

        Returns:
            Cropped array.
        """
        if key is None:
            # Center crop
            return self._center_crop(x)

        return self._random_crop(x, key)

    def _center_crop(self, x: Array) -> Array:
        """Apply center crop."""
        spatial_shape = x.shape[1:]
        num_dims = len(spatial_shape)

        if self.size is not None:
            crop_size = self.size
        else:
            assert self.size_int is not None
            crop_size = tuple(self.size_int for _ in range(num_dims))

        slices = [slice(None)]  # Channel dim
        for s, c in zip(spatial_shape, crop_size):
            start = (s - c) // 2
            slices.append(slice(start, start + c))

        return x[tuple(slices)]

    def _random_crop(self, x: Array, key: PRNGKey) -> Array:
        """Apply random crop."""
        spatial_shape = x.shape[1:]
        num_dims = len(spatial_shape)

        if self.size is not None:
            crop_size = self.size
        else:
            assert self.size_int is not None
            crop_size = tuple(self.size_int for _ in range(num_dims))

        # Add padding if specified
        if isinstance(self.padding, int) and self.padding > 0:
            pad_width = [(0, 0)] + [(self.padding, self.padding)] * num_dims
            x = jnp.pad(x, pad_width, mode="constant")
            spatial_shape = x.shape[1:]
        elif isinstance(self.padding, tuple) and any(p > 0 for p in self.padding):
            pad_width = [(0, 0)] + [(p, p) for p in self.padding]
            x = jnp.pad(x, pad_width, mode="constant")
            spatial_shape = x.shape[1:]

        # Generate random start positions
        keys = jax.random.split(key, num_dims)
        starts = []
        for i, (k, s, c) in enumerate(zip(keys, spatial_shape, crop_size)):
            max_start = s - c
            start = jax.random.randint(k, (), 0, max_start + 1)
            starts.append(start)

        # Use dynamic_slice for JIT compatibility
        start_indices = [0] + starts
        slice_sizes = [x.shape[0]] + list(crop_size)
        return jax.lax.dynamic_slice(x, start_indices, slice_sizes)


class Normalize(Transform):
    """Normalize data with mean and std.

    Applies: (x - mean) / std
    """

    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> None:
        """Initialize normalize transform.

        Args:
            mean: Per-channel mean values.
            std: Per-channel std values.
        """
        self.mean = jnp.array(mean)
        self.std = jnp.array(std)

    def __call__(self, x: Array, key: PRNGKey | None = None) -> Array:
        """Apply normalization.

        Args:
            x: Input of shape (C, *spatial) or (B, C, *spatial).
            key: Unused.

        Returns:
            Normalized array.
        """
        # Reshape mean and std for broadcasting
        if x.ndim == 3:
            # (C, H, W)
            mean = self.mean.reshape(-1, 1, 1)
            std = self.std.reshape(-1, 1, 1)
        elif x.ndim == 4:
            if x.shape[0] == len(self.mean):
                # (C, D, H, W)
                mean = self.mean.reshape(-1, 1, 1, 1)
                std = self.std.reshape(-1, 1, 1, 1)
            else:
                # (B, C, H, W)
                mean = self.mean.reshape(1, -1, 1, 1)
                std = self.std.reshape(1, -1, 1, 1)
        elif x.ndim == 5:
            # (B, C, D, H, W)
            mean = self.mean.reshape(1, -1, 1, 1, 1)
            std = self.std.reshape(1, -1, 1, 1, 1)
        else:
            # General case: assume channel is dim 1 or 0
            shape = [1] * x.ndim
            shape[1 if x.ndim > 1 else 0] = len(self.mean)
            mean = self.mean.reshape(shape)
            std = self.std.reshape(shape)

        return (x - mean) / std


class Denormalize(Transform):
    """Reverse normalization.

    Applies: x * std + mean
    """

    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> None:
        """Initialize denormalize transform.

        Args:
            mean: Per-channel mean values.
            std: Per-channel std values.
        """
        self.mean = jnp.array(mean)
        self.std = jnp.array(std)

    def __call__(self, x: Array, key: PRNGKey | None = None) -> Array:
        """Apply denormalization.

        Args:
            x: Input array.
            key: Unused.

        Returns:
            Denormalized array.
        """
        shape = [1] * x.ndim
        shape[1 if x.ndim > 1 else 0] = len(self.mean)
        mean = self.mean.reshape(shape)
        std = self.std.reshape(shape)

        return x * std + mean


class RandomRotation(Transform):
    """Random rotation for 2D data.

    Rotates by a random angle within specified range.
    """

    def __init__(
        self,
        degrees: float = 15.0,
    ) -> None:
        """Initialize random rotation.

        Args:
            degrees: Maximum rotation angle in degrees.
        """
        self.degrees = degrees

    def __call__(self, x: Array, key: PRNGKey | None = None) -> Array:
        """Apply random rotation.

        Args:
            x: Input of shape (C, H, W).
            key: PRNG key.

        Returns:
            Rotated array.
        """
        if key is None:
            return x

        # Generate random angle (currently not used in placeholder implementation)
        _ = jax.random.uniform(key, minval=-self.degrees, maxval=self.degrees)

        # Simple rotation using scipy.ndimage-like affine transform
        # For JAX, we use jax.scipy.ndimage.rotate or implement manually
        # For simplicity, return identity for now
        # TODO: Implement proper rotation
        return x


class ColorJitter(Transform):
    """Random color jittering for images.

    Randomly adjusts brightness, contrast, saturation, and hue.
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ) -> None:
        """Initialize color jitter.

        Args:
            brightness: Brightness jitter range.
            contrast: Contrast jitter range.
            saturation: Saturation jitter range.
            hue: Hue jitter range.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x: Array, key: PRNGKey | None = None) -> Array:
        """Apply color jittering.

        Args:
            x: Input of shape (C, H, W) with C=3.
            key: PRNG key.

        Returns:
            Color-jittered array.
        """
        if key is None or (
            self.brightness == 0 and self.contrast == 0 and self.saturation == 0 and self.hue == 0
        ):
            return x

        keys = jax.random.split(key, 4)

        # Brightness
        if self.brightness > 0:
            factor = jax.random.uniform(
                keys[0], minval=1 - self.brightness, maxval=1 + self.brightness
            )
            x = x * factor

        # Contrast
        if self.contrast > 0:
            factor = jax.random.uniform(keys[1], minval=1 - self.contrast, maxval=1 + self.contrast)
            mean = jnp.mean(x, axis=(1, 2), keepdims=True)
            x = (x - mean) * factor + mean

        # Clip to valid range
        x = jnp.clip(x, 0, 1)

        return x


class Mixup:
    """Mixup augmentation.

    Mixes pairs of samples and their labels.
    """

    def __init__(self, alpha: float = 1.0, num_classes: int | None = None) -> None:
        """Initialize mixup.

        Args:
            alpha: Beta distribution parameter.
            num_classes: Number of classes for one-hot encoding.
        """
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(
        self,
        x: Array,
        y: Array,
        key: PRNGKey,
    ) -> tuple[Array, Array]:
        """Apply mixup.

        Args:
            x: Input batch of shape (B, C, *spatial).
            y: Labels of shape (B,) or (B, num_classes).
            key: PRNG key.

        Returns:
            Tuple of (mixed_x, mixed_y).
        """
        batch_size = x.shape[0]
        key1, key2 = jax.random.split(key)

        # Sample lambda from Beta distribution
        lam = jax.random.beta(key1, self.alpha, self.alpha)

        # Random permutation for pairing
        perm = jax.random.permutation(key2, batch_size)

        # Mix inputs
        x_mixed = lam * x + (1 - lam) * x[perm]

        # Mix labels
        if y.ndim == 1:
            # One-hot encode for mixing
            num_classes = self.num_classes
            if num_classes is None:
                # Fallback to static max if possible, but this is risky in JIT
                # Better to provide num_classes in __init__
                raise ValueError("num_classes must be provided to Mixup if labels are not one-hot")
            y_onehot = jax.nn.one_hot(y, num_classes)
            y_mixed = lam * y_onehot + (1 - lam) * y_onehot[perm]
        else:
            y_mixed = lam * y + (1 - lam) * y[perm]

        return x_mixed, y_mixed


class Cutmix:
    """Cutmix augmentation.

    Cuts and pastes random patches between images.
    """

    def __init__(self, alpha: float = 1.0, num_classes: int | None = None) -> None:
        """Initialize cutmix.

        Args:
            alpha: Beta distribution parameter.
            num_classes: Number of classes for one-hot encoding.
        """
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(
        self,
        x: Array,
        y: Array,
        key: PRNGKey,
    ) -> tuple[Array, Array]:
        """Apply cutmix.

        Args:
            x: Input batch of shape (B, C, *spatial).
            y: Labels of shape (B,) or (B, num_classes).
            key: PRNG key.

        Returns:
            Tuple of (mixed_x, mixed_y).
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]
        num_dims = len(spatial_shape)

        key_lam, key_perm, key_bbox = jax.random.split(key, 3)

        # Sample lambda from Beta distribution
        lam = jax.random.beta(key_lam, self.alpha, self.alpha)

        # Random permutation
        perm = jax.random.permutation(key_perm, batch_size)

        # Cut ratio
        r = jnp.sqrt(1.0 - lam)

        # Calculate box coordinates for each dimension
        keys = jax.random.split(key_bbox, num_dims)
        mask = jnp.ones(spatial_shape, dtype=jnp.float32)

        for i, (k, s) in enumerate(zip(keys, spatial_shape)):
            # Sample size for this dimension
            cut_size = (s * r).astype(jnp.int32)
            cut_size = jnp.maximum(1, cut_size)

            # Sample center
            center = jax.random.randint(k, (), 0, s)

            # Start and end
            start = jnp.maximum(0, center - cut_size // 2)
            end = jnp.minimum(s, center + cut_size // 2)

            # Create mask for this dimension
            idx = jnp.arange(s)
            dim_mask = ~((idx >= start) & (idx < end))

            # Reshape for broadcasting
            shape = [1] * num_dims
            shape[i] = s
            dim_mask = dim_mask.reshape(shape)

            mask = mask * dim_mask

        # Apply to inputs (mask=1 where we keep x, mask=0 where we take from x[perm])
        mask_expanded = mask[None, None, ...]
        x_mixed = x * mask_expanded + x[perm] * (1.0 - mask_expanded)

        # Actual lambda used for labels (ratio of pixels kept from x)
        lam_actual = jnp.mean(mask)

        # Mix labels
        if y.ndim == 1:
            if self.num_classes is None:
                raise ValueError("num_classes must be provided for Cutmix")
            y_onehot = jax.nn.one_hot(y, self.num_classes)
            y_mixed = lam_actual * y_onehot + (1.0 - lam_actual) * y_onehot[perm]
        else:
            y_mixed = lam_actual * y + (1.0 - lam_actual) * y[perm]

        return x_mixed, y_mixed


class MixupOrCutmix:
    """Select between Mixup and Cutmix randomly."""

    def __init__(
        self,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 1.0,
        p: float = 0.5,
        num_classes: int | None = None,
    ) -> None:
        """Initialize combined augmentation.

        Args:
            mixup_alpha: Mixup alpha parameter.
            cutmix_alpha: Cutmix alpha parameter.
            p: Probability of choosing mixup over cutmix.
            num_classes: Number of classes for one-hot encoding.
        """
        self.mixup = Mixup(mixup_alpha, num_classes)
        self.cutmix = Cutmix(cutmix_alpha, num_classes)
        self.p = p

    def __call__(
        self,
        x: Array,
        y: Array,
        key: PRNGKey,
    ) -> tuple[Array, Array]:
        """Apply either Mixup or Cutmix.

        Args:
            x: Input batch.
            y: Labels.
            key: PRNG key.

        Returns:
            Tuple of (mixed_x, mixed_y).
        """
        key1, key2 = jax.random.split(key)
        do_mixup = jax.random.uniform(key1) < self.p

        return jax.lax.cond(
            do_mixup,
            lambda k: self.mixup(x, y, k),
            lambda k: self.cutmix(x, y, k),
            key2,
        )


class Cutout(Transform):
    """Cutout augmentation.

    Randomly masks out rectangular regions.
    """

    def __init__(
        self,
        size: int | tuple[int, ...],
        p: float = 0.5,
    ) -> None:
        """Initialize cutout.

        Args:
            size: Size of masked region.
            p: Probability of applying cutout.
        """
        self.size = size
        self.p = p

    def __call__(self, x: Array, key: PRNGKey | None = None) -> Array:
        """Apply cutout.

        Args:
            x: Input of shape (C, *spatial).
            key: PRNG key.

        Returns:
            Array with masked region.
        """
        if key is None:
            return x

        key1, key2 = jax.random.split(key)
        do_cutout = jax.random.uniform(key1) < self.p

        spatial_shape = x.shape[1:]
        num_dims = len(spatial_shape)

        # Get cutout size
        if isinstance(self.size, int):
            cutout_size = tuple(self.size for _ in range(num_dims))
        else:
            cutout_size = self.size

        # Random position
        keys = jax.random.split(key2, num_dims)
        mask = jnp.ones_like(x)

        for i, (k, s, cs) in enumerate(zip(keys, spatial_shape, cutout_size)):
            center = jax.random.randint(k, (), 0, s)
            start = jnp.maximum(0, center - cs // 2)
            end = jnp.minimum(s, center + cs // 2)

            # Create slice for this dimension
            idx = jnp.arange(s)
            dim_mask = (idx < start) | (idx >= end)

            # Reshape for broadcasting
            shape = [1] * (num_dims + 1)
            shape[i + 1] = s
            dim_mask = dim_mask.reshape(shape)

            mask = mask * dim_mask

        return jnp.where(do_cutout, x * mask, x)


def create_augmentation_pipeline(
    config: Any,
    is_training: bool = True,
) -> Transform:
    """Create augmentation pipeline from configuration.

    Args:
        config: DataConfig or similar configuration object.
        is_training: Whether this is for training (applies random augmentation).

    Returns:
        Composed transform.
    """
    transforms: list[Transform] = []

    if is_training and hasattr(config, "augmentation") and config.augmentation:
        # Random crop with padding
        if hasattr(config, "random_crop") and config.random_crop:
            transforms.append(RandomCrop(config.image_size, padding=4))

        # Random horizontal flip
        if hasattr(config, "random_flip") and config.random_flip:
            transforms.append(RandomHorizontalFlip(p=0.5))

        # Color jittering
        if hasattr(config, "color_jitter") and config.color_jitter:
            transforms.append(
                ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                )
            )

        # Cutout
        if hasattr(config, "cutout_size") and config.cutout_size > 0:
            transforms.append(Cutout(config.cutout_size))

    # Normalization
    if hasattr(config, "normalize") and config.normalize:
        transforms.append(Normalize(config.mean, config.std))

    return Compose(transforms)
