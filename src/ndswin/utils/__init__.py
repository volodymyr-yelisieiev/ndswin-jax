"""Utility modules for NDSwin-JAX."""

from ndswin.utils.device import (
    DeviceInfo,
    get_default_device,
    get_device_info,
    is_gpu_available,
    is_tpu_available,
    set_default_device,
)
from ndswin.utils.logging import (
    LogLevel,
    get_logger,
    setup_logging,
)
from ndswin.utils.reproducibility import (
    create_rng_keys,
    get_deterministic_key,
    set_seed,
)

__all__ = [
    # Device utilities
    "DeviceInfo",
    "get_device_info",
    "get_default_device",
    "set_default_device",
    "is_gpu_available",
    "is_tpu_available",
    # Logging
    "get_logger",
    "setup_logging",
    "LogLevel",
    # Reproducibility
    "set_seed",
    "create_rng_keys",
    "get_deterministic_key",
]
