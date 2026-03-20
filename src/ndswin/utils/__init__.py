"""Utility modules for NDSwin-JAX."""

from typing import Any


# Lazily expose utility helpers so CLI commands can import lightweight pieces.
def __getattr__(name: str) -> Any:
    if name in {
        "DeviceInfo",
        "get_default_device",
        "get_device_info",
        "is_gpu_available",
        "is_tpu_available",
        "set_default_device",
    }:
        from ndswin.utils.device import (
            DeviceInfo,
            get_default_device,
            get_device_info,
            is_gpu_available,
            is_tpu_available,
            set_default_device,
        )

        mapping = {
            "DeviceInfo": DeviceInfo,
            "get_default_device": get_default_device,
            "get_device_info": get_device_info,
            "is_gpu_available": is_gpu_available,
            "is_tpu_available": is_tpu_available,
            "set_default_device": set_default_device,
        }
        return mapping[name]
    if name in {"get_logger", "setup_logging", "LogLevel"}:
        from ndswin.utils.logging import LogLevel, get_logger, setup_logging

        mapping = {
            "get_logger": get_logger,
            "setup_logging": setup_logging,
            "LogLevel": LogLevel,
        }
        return mapping[name]
    if name in {"set_seed", "create_rng_keys", "get_deterministic_key"}:
        from ndswin.utils.reproducibility import create_rng_keys, get_deterministic_key, set_seed

        mapping = {
            "set_seed": set_seed,
            "create_rng_keys": create_rng_keys,
            "get_deterministic_key": get_deterministic_key,
        }
        return mapping[name]
    raise AttributeError(f"module 'ndswin.utils' has no attribute '{name}'")


__all__ = [
    "DeviceInfo",
    "get_device_info",
    "get_default_device",
    "set_default_device",
    "is_gpu_available",
    "is_tpu_available",
    "get_logger",
    "setup_logging",
    "LogLevel",
    "set_seed",
    "create_rng_keys",
    "get_deterministic_key",
]
