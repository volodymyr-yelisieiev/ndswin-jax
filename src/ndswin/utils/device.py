"""Device detection and management utilities for JAX.

This module provides utilities for detecting available compute devices (CPU, GPU, TPU)
and managing device placement for JAX computations.
"""

import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class DeviceInfo:
    """Information about a compute device.

    Attributes:
        device_type: Type of device ('cpu', 'gpu', 'tpu').
        device_id: Device index.
        platform: Backend platform name.
        device_kind: Specific device kind (e.g., 'NVIDIA A100').
        memory_bytes: Available memory in bytes (if available).
    """

    device_type: str
    device_id: int
    platform: str
    device_kind: str
    memory_bytes: int | None = None

    def __str__(self) -> str:
        """Return string representation of device."""
        mem_str = ""
        if self.memory_bytes is not None:
            mem_gb = self.memory_bytes / (1024**3)
            mem_str = f", {mem_gb:.1f}GB"
        return f"{self.device_type.upper()}:{self.device_id} ({self.device_kind}{mem_str})"


def get_device_info(device: jax.Device | None = None) -> DeviceInfo:
    """Get information about a device.

    Args:
        device: JAX device to query. If None, uses default device.

    Returns:
        DeviceInfo object with device details.
    """
    if device is None:
        device = jax.devices()[0]

    device_type = device.platform
    device_id = device.id
    platform = device.platform
    device_kind = str(device.device_kind) if hasattr(device, "device_kind") else platform

    return DeviceInfo(
        device_type=device_type,
        device_id=device_id,
        platform=platform,
        device_kind=device_kind,
    )


def get_all_devices() -> list[jax.Device]:
    """Get list of all available devices.

    Returns:
        List of JAX devices.
    """
    return jax.devices()


def get_devices_by_type(device_type: str) -> list[jax.Device]:
    """Get devices of a specific type.

    Args:
        device_type: Type of device ('cpu', 'gpu', 'tpu').

    Returns:
        List of devices matching the type.
    """
    return jax.devices(device_type.lower())


def get_default_device() -> jax.Device:
    """Get the default compute device.

    Returns the first available device, preferring GPU over TPU over CPU.

    Returns:
        Default JAX device.
    """
    devices = jax.devices()
    return devices[0]


def set_default_device(device_type: str, device_id: int = 0) -> None:
    """Set the default device for JAX computations.

    This sets environment variables that affect JAX device selection.
    Must be called before any JAX operations.

    Args:
        device_type: Type of device ('cpu', 'gpu', 'tpu').
        device_id: Device index.

    Raises:
        ValueError: If device type is not available.
    """
    device_type = device_type.lower()

    if device_type == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device_type == "gpu":
        if not is_gpu_available():
            raise ValueError("GPU is not available")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    elif device_type == "tpu":
        if not is_tpu_available():
            raise ValueError("TPU is not available")
        # TPU configuration handled by JAX
    else:
        raise ValueError(f"Unknown device type: {device_type}")


def is_gpu_available() -> bool:
    """Check if GPU is available.

    Returns:
        True if GPU is available, False otherwise.
    """
    try:
        devices = jax.devices("gpu")
        return len(devices) > 0
    except RuntimeError:
        return False


def is_tpu_available() -> bool:
    """Check if TPU is available.

    Returns:
        True if TPU is available, False otherwise.
    """
    try:
        devices = jax.devices("tpu")
        return len(devices) > 0
    except RuntimeError:
        return False


def get_num_devices(device_type: str | None = None) -> int:
    """Get number of available devices.

    Args:
        device_type: Type of device to count. If None, counts all devices.

    Returns:
        Number of available devices.
    """
    if device_type is None:
        return len(jax.devices())
    try:
        return len(jax.devices(device_type.lower()))
    except RuntimeError:
        return 0


def get_memory_info(device: jax.Device | None = None) -> tuple[int, int]:
    """Get memory information for a device.

    Args:
        device: JAX device to query. If None, uses default device.

    Returns:
        Tuple of (used_memory, total_memory) in bytes.
        Returns (0, 0) if memory info is not available.
    """
    if device is None:
        device = get_default_device()

    try:
        # This is backend-specific and may not be available
        stats = device.memory_stats()
        if stats is not None:
            return (
                stats.get("bytes_in_use", 0),
                stats.get("bytes_limit", 0),
            )
    except (AttributeError, RuntimeError):
        pass

    return (0, 0)


def clear_caches() -> None:
    """Clear JAX compilation caches.

    This can help free memory after training or when switching models.
    """
    jax.clear_caches()


def synchronize_devices() -> None:
    """Synchronize all devices.

    Blocks until all computations on all devices are complete.
    Useful for accurate timing measurements.
    """
    for device in jax.devices():
        jax.device_get(jnp.zeros(1, device=device))


def device_put_sharded(arrays: list[jnp.ndarray]) -> list[jnp.ndarray]:
    """Put arrays on multiple devices (one per device).

    Args:
        arrays: List of arrays, one per device.

    Returns:
        List of device-placed arrays.
    """
    devices = jax.devices()
    if len(arrays) != len(devices):
        raise ValueError(
            f"Number of arrays ({len(arrays)}) must match number of devices ({len(devices)})"
        )
    return [jax.device_put(arr, device) for arr, device in zip(arrays, devices)]


def replicate(array: jnp.ndarray) -> list[jnp.ndarray]:
    """Replicate an array across all devices.

    Args:
        array: Array to replicate.

    Returns:
        List of arrays, one per device.
    """
    devices = jax.devices()
    return [jax.device_put(array, device) for device in devices]


def print_device_info() -> None:
    """Print information about all available devices."""
    print("=" * 60)
    print("JAX Device Information")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Number of devices: {get_num_devices()}")
    print("-" * 60)

    for i, device in enumerate(jax.devices()):
        info = get_device_info(device)
        print(f"Device {i}: {info}")

    print("-" * 60)
    print(f"GPU available: {is_gpu_available()}")
    print(f"TPU available: {is_tpu_available()}")
    print("=" * 60)


# Convenience constants
CPU = "cpu"
GPU = "gpu"
TPU = "tpu"
