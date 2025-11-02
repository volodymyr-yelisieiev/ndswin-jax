"""Window-based multi-head self-attention with relative position bias."""

from __future__ import annotations

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ...utils.mlp import MLP
from .relative import relative_coords_table, relative_position_index


class WindowAttention(nn.Module):
    """Window based multi-head self-attention with continuous relative position bias.
    
    This implements the attention mechanism from Swin Transformer V2, which uses
    log-spaced continuous relative position biases for better transferability
    across different window sizes.
    
    Attributes:
        dim: Number of input channels
        num_heads: Number of attention heads
        window_size: Size of the attention window for each dimension
        qkv_bias: If True, add a learnable bias to query, key, value projections
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
    """

    dim: int
    num_heads: int
    window_size: Sequence[int]
    qkv_bias: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    def setup(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError("`dim` must be divisible by `num_heads`.")
        self.head_dim = self.dim // self.num_heads

        # QKV projection
        self.qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias)
        self.proj = nn.Dense(self.dim)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(self.attn_drop)
        self.proj_dropout = nn.Dropout(self.proj_drop)
        
        # Continuous relative position bias MLP
        self.cpb_mlp = MLP(
            features=[len(self.window_size), 512, self.num_heads],
            activation=nn.relu
        )

        # Learnable temperature parameter
        self.logit_scale = self.param(
            "logit_scale",
            lambda rng: jnp.log(jnp.ones((self.num_heads, 1, 1)) * 10.0)
        )
        self.max_logits = jnp.log(jnp.array(1.0 / 0.01, dtype=jnp.float32))
        
        # Pre-compute relative position tables
        self.rpb = relative_coords_table(self.window_size)
        self.rpb_idx = relative_position_index(self.window_size)

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray],
        *,
        deterministic: bool
    ) -> jnp.ndarray:
        """Apply window attention.
        
        Args:
            x: Input tensor of shape (batch_windows, num_tokens, dim)
            mask: Optional attention mask of shape (num_windows, num_tokens, num_tokens)
            deterministic: If True, disable dropout
            
        Returns:
            Output tensor of shape (batch_windows, num_tokens, dim)
        """
        batch_windows, num_tokens, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_windows, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, batch_windows, num_heads, num_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Normalize Q and K (cosine attention)
        def _normalize(t):
            norm_sq = jnp.sum(t * t, axis=-1, keepdims=True)
            inv_norm = jax.lax.rsqrt(jnp.maximum(norm_sq, 1e-12))
            return t * inv_norm

        q = _normalize(q)
        k = _normalize(k)
        
        # Compute attention scores
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k)

        # Apply learnable temperature
        logit_scale = jnp.exp(jnp.minimum(self.logit_scale, self.max_logits))
        attn = attn * logit_scale

        # Add continuous relative position bias
        rpb = self.cpb_mlp(self.rpb, deterministic=deterministic)
        rpb = jnp.take(rpb, self.rpb_idx.reshape(-1), axis=0)
        rpb = rpb.reshape(num_tokens, num_tokens, self.num_heads)
        rpb = jnp.transpose(rpb, (2, 0, 1))  # (num_heads, num_tokens, num_tokens)
        attn = attn + 16.0 * nn.sigmoid(rpb)[None, ...]

        # Apply attention mask if provided
        if mask is not None:
            n_w = mask.shape[0]
            batches = batch_windows // n_w
            attn = attn.reshape(batches, n_w, self.num_heads, num_tokens, num_tokens)
            attn = attn + mask[None, :, None, :, :]
            attn = attn.reshape(batch_windows, self.num_heads, num_tokens, num_tokens)

        # Apply softmax and dropout
        attn = nn.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn, deterministic=deterministic)

        # Compute output
        x = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        x = x.transpose(0, 2, 1, 3).reshape(batch_windows, num_tokens, self.dim)
        x = self.proj(x)
        x = self.proj_dropout(x, deterministic=deterministic)
        
        return x


__all__ = ["WindowAttention"]
