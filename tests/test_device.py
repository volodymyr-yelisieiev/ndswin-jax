"""Tests for device detection and management utilities."""

import os
import unittest.mock as mock

import jax
import jax.numpy as jnp
import pytest

from ndswin.utils.device import (
    DeviceInfo,
    clear_caches,
    device_put_sharded,
    get_all_devices,
    get_default_device,
    get_device_info,
    get_devices_by_type,
    get_memory_info,
    get_num_devices,
    is_gpu_available,
    is_tpu_available,
    print_device_info,
    replicate,
    set_default_device,
    synchronize_devices,
)


def test_device_info_dataclass():
    """Test DeviceInfo dataclass formatting."""
    info = DeviceInfo(device_type="cpu", device_id=0, platform="cpu", device_kind="mock_cpu")
    assert str(info) == "CPU:0 (mock_cpu)"

    info_with_mem = DeviceInfo(
        device_type="gpu", device_id=1, platform="gpu", device_kind="A100", memory_bytes=42949672960
    )
    assert str(info_with_mem) == "GPU:1 (A100, 40.0GB)"


def test_get_device_info():
    """Test get_device_info function."""
    info = get_device_info()
    assert isinstance(info, DeviceInfo)
    assert info.device_type in ("cpu", "gpu", "tpu")

    devices = get_all_devices()
    info2 = get_device_info(devices[0])
    assert info.device_id == info2.device_id


def test_get_all_devices():
    """Test getting all devices."""
    devices = get_all_devices()
    assert len(devices) > 0
    assert isinstance(devices[0], jax.Device)


def test_get_devices_by_type():
    """Test getting devices by type."""
    cpu_devices = get_devices_by_type("cpu")
    assert len(cpu_devices) >= 1

    if is_gpu_available():
        gpu_devices = get_devices_by_type("gpu")
        assert len(gpu_devices) >= 1


def test_get_default_device():
    """Test getting default device."""
    device = get_default_device()
    assert isinstance(device, jax.Device)


@mock.patch.dict(os.environ, {"JAX_PLATFORMS": ""})
def test_set_default_device_cpu():
    """Test setting default device to CPU ensures OS env is configured."""
    set_default_device("cpu", 0)
    assert os.environ.get("JAX_PLATFORMS") == "cpu"


@mock.patch("ndswin.utils.device.is_gpu_available", return_value=True)
@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""})
def test_set_default_device_gpu(mock_is_gpu):
    """Test setting default device to GPU configures CUDA variable."""
    set_default_device("gpu", 2)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"


@mock.patch("ndswin.utils.device.is_gpu_available", return_value=False)
def test_set_default_device_gpu_unavailable(mock_is_gpu):
    with pytest.raises(ValueError, match="GPU is not available"):
        set_default_device("gpu", 0)


@mock.patch("ndswin.utils.device.is_tpu_available", return_value=False)
def test_set_default_device_tpu_unavailable(mock_is_tpu):
    with pytest.raises(ValueError, match="TPU is not available"):
        set_default_device("tpu", 0)


def test_set_default_device_invalid():
    with pytest.raises(ValueError, match="Unknown device type: unknown"):
        set_default_device("unknown", 0)


def test_is_gpu_tpu_available():
    """Test that availability handlers do not crash."""
    assert isinstance(is_gpu_available(), bool)
    assert isinstance(is_tpu_available(), bool)


@mock.patch("jax.devices")
def test_availability_runtime_error(mock_devices):
    mock_devices.side_effect = RuntimeError("Mock error")
    assert not is_gpu_available()
    assert not is_tpu_available()


def test_get_num_devices():
    """Test get_num_devices counting."""
    assert get_num_devices() >= 1
    assert get_num_devices("cpu") >= 1


@mock.patch("jax.devices")
def test_get_num_devices_error(mock_devices):
    mock_devices.side_effect = RuntimeError("Mock error")
    assert get_num_devices("unsupported") == 0


def test_get_memory_info():
    """Test memory info fetches (may securely return 0 if no backend stat support)."""
    used, total = get_memory_info()
    assert isinstance(used, int)
    assert isinstance(total, int)
    assert used >= 0
    assert total >= 0


def test_clear_caches():
    """Test that clear_caches doesn't crash."""
    clear_caches()


def test_synchronize_devices():
    """Test synchronization logic."""
    synchronize_devices()


def test_device_put_sharded():
    """Test sharding arrays across devices."""
    devices = get_all_devices()
    arrays = [jnp.array([i]) for i in range(len(devices))]
    sharded = device_put_sharded(arrays)
    assert len(sharded) == len(devices)

    with pytest.raises(ValueError, match="Number of arrays"):
        # Guarantee a mismatch by passing one more array than the number of devices
        device_put_sharded([jnp.array([1])] * (len(devices) + 1))


def test_replicate():
    """Test array replication."""
    arr = jnp.array([1, 2, 3])
    rep = replicate(arr)
    assert len(rep) == get_num_devices()


def test_print_device_info(capsys):
    """Test printing logic generates text securely."""
    print_device_info()
    captured = capsys.readouterr()
    assert "JAX Device Information" in captured.out
    assert "Backend:" in captured.out
