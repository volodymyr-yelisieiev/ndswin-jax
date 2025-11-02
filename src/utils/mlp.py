"""Feed-forward helper modules."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import flax.linen as nn


class MLP(nn.Module):
    """Configurable multi-layer perceptron used across the project."""

    features: Sequence[int]
    activation: Callable[[Any], Any] = nn.gelu
    dropout_rate: float = 0.0
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: Any, *, deterministic: bool) -> Any:
        if len(self.features) < 2:
            raise ValueError("`features` must provide at least an input and output size.")

        for idx, (_, out_dim) in enumerate(zip(self.features, self.features[1:])):
            x = nn.Dense(out_dim, use_bias=self.use_bias, name=f"dense_{idx}")(x)
            is_last = idx == len(self.features) - 2
            if not is_last:
                x = self.activation(x)
                if self.dropout_rate > 0.0:
                    x = nn.Dropout(rate=self.dropout_rate, name=f"dropout_{idx}")(x, deterministic=deterministic)
        return x
