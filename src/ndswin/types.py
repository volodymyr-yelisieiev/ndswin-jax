"""Type definitions for NDSwin-JAX.

This module defines type aliases used throughout the codebase for improved
type safety and documentation.
"""

from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

# =============================================================================
# JAX Array Types
# =============================================================================

# Generic JAX array type
Array = jax.Array

# Numpy array for interop
NpArray = jnp.ndarray

# Array-like types that can be converted to JAX arrays
ArrayLike = Union[Array, jnp.ndarray, list, tuple, float, int]

# =============================================================================
# JAX PRNG Types
# =============================================================================

# PRNG key type
PRNGKey = jax.Array

# PRNG key array (for vmap operations)
PRNGKeyArray = jax.Array

# =============================================================================
# Shape Types
# =============================================================================

# Single dimension
Dim = int

# Shape tuple
Shape = tuple[int, ...]

# Variable length shape (for n-dimensional operations)
SpatialShape = tuple[int, ...]

# Window size specification
WindowSize = tuple[int, ...]

# Patch size specification
PatchSize = tuple[int, ...]

# =============================================================================
# Model Parameter Types
# =============================================================================

# Flax parameter dictionary
Params = FrozenDict[str, Any]

# Mutable state (e.g., batch norm statistics)
State = FrozenDict[str, Any]

# Combined variables dictionary
Variables = FrozenDict[str, Any]

# Gradient type (same structure as Params)
Gradients = Params

# =============================================================================
# Training Types
# =============================================================================

# Learning rate type
LearningRate = Union[float, Array]

# Loss value type
LossValue = Union[float, Array]

# Metric value type
MetricValue = Union[float, Array]

# Batch of data
Batch = dict[str, Array]

# Single data sample
Sample = dict[str, Array]

# Labels type
Labels = Array

# Logits type
Logits = Array

# Predictions type
Predictions = Array

# =============================================================================
# Function Types
# =============================================================================

# Generic function type
F = TypeVar("F", bound=Callable[..., Any])

# Loss function type
LossFn = Callable[[Logits, Labels], LossValue]

# Metric function type
MetricFn = Callable[[Predictions, Labels], MetricValue]

# Activation function type
ActivationFn = Callable[[Array], Array]

# Initializer function type
Initializer = Callable[[PRNGKey, Shape, jnp.dtype], Array]

# =============================================================================
# Configuration Types
# =============================================================================

# Depth specification per stage
Depths = tuple[int, ...]

# Number of heads per stage
NumHeads = tuple[int, ...]

# Dimension multipliers
DimMultipliers = tuple[int, ...]

# Drop path rate per block
DropPathRates = tuple[float, ...]

# =============================================================================
# Optional Types
# =============================================================================

# Optional array
OptionalArray = Optional[Array]

# Optional shape
OptionalShape = Optional[Shape]

# Optional integer
OptionalInt = Optional[int]

# Optional float
OptionalFloat = Optional[float]

# =============================================================================
# Device Types
# =============================================================================

# Device specification
Device = Any  # jax.Device

# Device array (for multi-device)
DeviceArray = Sequence[Device]

# =============================================================================
# Data Types
# =============================================================================

# Supported dtypes
DType = jnp.dtype

# Default floating point dtype
DEFAULT_DTYPE: DType = jnp.float32

# Half precision dtype
HALF_DTYPE: DType = jnp.float16

# Brain float dtype
BFLOAT16_DTYPE: DType = jnp.bfloat16

# =============================================================================
# Checkpoint Types
# =============================================================================

# Checkpoint state dictionary
CheckpointState = dict[str, Any]

# Training step counter
Step = int

# Epoch counter
Epoch = int

# =============================================================================
# Learning Rate Schedule Types
# =============================================================================

# Learning rate schedule function
LearningRateSchedule = Callable[[int], Union[float, Array]]

# PyTree type for nested structures
PyTree = Any
