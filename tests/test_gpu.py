"""Tests for intelligent GPU allocator."""

import os
import subprocess
import unittest.mock as mock

from ndswin.utils.gpu import get_available_gpus, setup_optimal_gpus


@mock.patch("ndswin.utils.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
@mock.patch("subprocess.check_output")
def test_get_available_gpus_success(mock_check_output, _mock_which):
    """Test successful parsing of nvidia-smi results."""
    mock_check_output.return_value = (
        "0, 15000, 0, 10\n1, 2000, 0, 90\n2, [Not Supported], 0, 40\n3, 8000, 5, 45\n"
    )

    # Needs 4MB free, under 50%
    gpus = get_available_gpus(min_free_mb=4000, max_utilization=50)

    # GPU 0 has 15000 > 4000, used 0, util 10% < 50% -> YES
    # GPU 1 has 2000 < 4000 -> NO
    # GPU 2 has "[Not Supported]" free -> errors out silently -> NO
    # Let's see GPU 3: 8000 > 4000, util 45% < 50% -> YES
    assert gpus == [0, 3]


@mock.patch("ndswin.utils.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
@mock.patch("subprocess.check_output")
def test_get_available_gpus_not_supported_util(mock_check_output, _mock_which):
    mock_check_output.return_value = "0, 8000, 0, [Not Supported]\n"
    # It should assume 0% if util is "[Not Supported]"
    gpus = get_available_gpus(min_free_mb=4000, max_utilization=50)
    assert gpus == [0]


@mock.patch("ndswin.utils.gpu.shutil.which", return_value=None)
def test_get_available_gpus_no_smi(_mock_which):
    """Test when nvidia-smi is missing."""
    assert get_available_gpus() == []


@mock.patch("ndswin.utils.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
@mock.patch("subprocess.check_output")
def test_get_available_gpus_command_error(mock_check_output, _mock_which):
    mock_check_output.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")
    assert get_available_gpus() == []


@mock.patch("ndswin.utils.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
@mock.patch("ndswin.utils.gpu.subprocess.check_output")
@mock.patch("ndswin.utils.gpu.warnings.warn")
def test_get_available_gpus_unexpected_error(mock_warn, mock_check_output, _mock_which):
    mock_check_output.side_effect = Exception("Unexpected error")
    assert get_available_gpus() == []
    mock_warn.assert_called_once()


@mock.patch("ndswin.utils.gpu.shutil.which", return_value="/usr/bin/nvidia-smi")
@mock.patch("subprocess.check_output")
def test_get_available_gpus_excludes_busy_gpu_by_used_memory(mock_check_output, _mock_which):
    mock_check_output.return_value = "0, 5000, 6100, 0\n1, 9000, 5, 0\n"
    assert get_available_gpus(min_free_mb=4000, max_utilization=50, max_used_mb=512) == [1]


@mock.patch("ndswin.utils.gpu.get_available_gpus", return_value=[1, 2])
def test_setup_optimal_gpus_binds(mock_get, monkeypatch, caplog):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    with caplog.at_level("INFO", logger="ndswin.gpu"):
        setup_optimal_gpus()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "Bound CUDA_VISIBLE_DEVICES to available GPUs" in caplog.text


@mock.patch("ndswin.utils.gpu.get_available_gpus", return_value=[])
def test_setup_optimal_gpus_empty_no_bind(mock_get, monkeypatch, caplog):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    with caplog.at_level("INFO", logger="ndswin.gpu"):
        setup_optimal_gpus()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "No GPUs satisfied allocation thresholds" in caplog.text


@mock.patch("ndswin.utils.gpu.get_available_gpus", return_value=[0])
def test_setup_optimal_gpus_respects_existing(mock_get, monkeypatch, caplog):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with caplog.at_level("INFO", logger="ndswin.gpu"):
        setup_optimal_gpus()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert "Respecting existing CUDA_VISIBLE_DEVICES=3" in caplog.text
