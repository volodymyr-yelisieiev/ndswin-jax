"""Core components for NDSwin-JAX."""

from ndswin.core.attention import (
    MultiHeadAttention,
    WindowAttention,
)
from ndswin.core.blocks import (
    DropPath,
    MLPBlock,
    SwinTransformerBlock,
)
from ndswin.core.patch_embed import (
    PatchEmbed,
    PatchMerging,
)
from ndswin.core.utils import (
    compute_num_patches,
    get_relative_position_index,
    to_ntuple,
)
from ndswin.core.window_ops import (
    create_attention_mask,
    cyclic_shift,
    get_window_grid_shape,
    partition_windows,
    reverse_cyclic_shift,
    reverse_partition_windows,
)

__all__ = [
    # Window operations
    "partition_windows",
    "reverse_partition_windows",
    "cyclic_shift",
    "reverse_cyclic_shift",
    "create_attention_mask",
    "get_window_grid_shape",
    # Attention
    "WindowAttention",
    "MultiHeadAttention",
    # Blocks
    "SwinTransformerBlock",
    "MLPBlock",
    "PatchMerging",
    "DropPath",
    # Patch embedding
    "PatchEmbed",
    # Utilities
    "to_ntuple",
    "compute_num_patches",
    "get_relative_position_index",
]
