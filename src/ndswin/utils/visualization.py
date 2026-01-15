"""Visualization utilities for NDSwin-JAX.

This module provides utilities for visualizing attention maps,
feature maps, and model predictions.
"""

from typing import Any

import numpy as np

from ndswin.types import Array

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _check_matplotlib() -> None:
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for visualization. Install with: pip install matplotlib"
        )


def visualize_attention_2d(
    attention: Array,
    image: Array | None = None,
    head_idx: int = 0,
    layer_idx: int = 0,
    figsize: tuple[int, int] = (10, 10),
    cmap: str = "viridis",
    save_path: str | None = None,
    show: bool = True,
) -> Any | None:
    """Visualize 2D attention maps.

    Args:
        attention: Attention weights of shape (num_heads, H, W) or (B, num_heads, H, W).
        image: Optional input image of shape (H, W, C) or (C, H, W).
        head_idx: Attention head index to visualize.
        layer_idx: Layer index (for labeling).
        figsize: Figure size.
        cmap: Colormap for attention.
        save_path: Path to save figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure if show is False, None otherwise.
    """
    _check_matplotlib()

    # Handle batch dimension
    if attention.ndim == 4:
        attention = attention[0]  # Take first sample
    if attention.ndim == 3:
        attention = attention[head_idx]  # Take specified head

    attn_np = np.array(attention)

    fig, axes = plt.subplots(1, 2 if image is not None else 1, figsize=figsize)

    if image is not None:
        img_np = np.array(image)
        # Handle channel-first format
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
            img_np = np.transpose(img_np, (1, 2, 0))
        # Normalize if needed
        if img_np.max() > 1:
            img_np = img_np / 255.0

        axes[0].imshow(img_np)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        im = axes[1].imshow(attn_np, cmap=cmap)
        axes[1].set_title(f"Attention (Layer {layer_idx}, Head {head_idx})")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])
    else:
        im = axes.imshow(attn_np, cmap=cmap)
        axes.set_title(f"Attention (Layer {layer_idx}, Head {head_idx})")
        axes.axis("off")
        plt.colorbar(im)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_attention_3d_slices(
    attention: Array,
    slice_axis: int = 0,
    num_slices: int = 4,
    head_idx: int = 0,
    figsize: tuple[int, int] = (12, 3),
    cmap: str = "viridis",
    save_path: str | None = None,
    show: bool = True,
) -> Any | None:
    """Visualize 3D attention maps as 2D slices.

    Args:
        attention: Attention weights of shape (num_heads, D, H, W).
        slice_axis: Axis along which to take slices.
        num_slices: Number of slices to visualize.
        head_idx: Attention head index to visualize.
        figsize: Figure size.
        cmap: Colormap for attention.
        save_path: Path to save figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure if show is False, None otherwise.
    """
    _check_matplotlib()

    if attention.ndim == 5:
        attention = attention[0]  # Take first sample
    if attention.ndim == 4:
        attention = attention[head_idx]  # Take specified head

    attn_np = np.array(attention)
    depth = attn_np.shape[slice_axis]
    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    fig, axes = plt.subplots(1, num_slices, figsize=figsize)

    for i, idx in enumerate(slice_indices):
        if slice_axis == 0:
            slice_data = attn_np[idx]
        elif slice_axis == 1:
            slice_data = attn_np[:, idx]
        else:
            slice_data = attn_np[:, :, idx]

        axes[i].imshow(slice_data, cmap=cmap)
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis("off")

    plt.suptitle(f"3D Attention Slices (Head {head_idx})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def visualize_feature_maps(
    features: Array,
    num_features: int = 16,
    figsize: tuple[int, int] = (12, 12),
    cmap: str = "viridis",
    save_path: str | None = None,
    show: bool = True,
) -> Any | None:
    """Visualize feature maps.

    Args:
        features: Feature maps of shape (C, H, W) or (B, C, H, W).
        num_features: Number of feature channels to visualize.
        figsize: Figure size.
        cmap: Colormap for features.
        save_path: Path to save figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure if show is False, None otherwise.
    """
    _check_matplotlib()

    if features.ndim == 4:
        features = features[0]  # Take first sample

    feat_np = np.array(features)
    num_channels = min(feat_np.shape[0], num_features)
    grid_size = int(np.ceil(np.sqrt(num_channels)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_channels:
            axes[i].imshow(feat_np[i], cmap=cmap)
            axes[i].set_title(f"Ch {i}")
        axes[i].axis("off")

    plt.suptitle(f"Feature Maps ({num_channels} channels)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_training_curves(
    metrics: dict[str, list[float]],
    figsize: tuple[int, int] = (12, 4),
    save_path: str | None = None,
    show: bool = True,
) -> Any | None:
    """Plot training curves.

    Args:
        metrics: Dictionary mapping metric names to lists of values.
        figsize: Figure size.
        save_path: Path to save figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure if show is False, None otherwise.
    """
    _check_matplotlib()

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig


def plot_confusion_matrix(
    confusion_matrix: Array,
    class_names: list[str] | None = None,
    figsize: tuple[int, int] = (10, 10),
    cmap: str = "Blues",
    normalize: bool = True,
    save_path: str | None = None,
    show: bool = True,
) -> Any | None:
    """Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes).
        class_names: List of class names.
        figsize: Figure size.
        cmap: Colormap.
        normalize: Whether to normalize the matrix.
        save_path: Path to save figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure if show is False, None otherwise.
    """
    _check_matplotlib()

    cm = np.array(confusion_matrix)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
        return None
    else:
        return fig
