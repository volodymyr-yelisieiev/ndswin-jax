"""Tests for intelligent GPU allocator."""

import os
import subprocess
import unittest.mock as mock

from ndswin.utils.gpu import get_available_gpus, setup_optimal_gpus


@mock.patch("subprocess.check_output")
def test_get_available_gpus_success(mock_check_output):
    """Test successful parsing of nvidia-smi results."""
    mock_check_output.side_effect = [
        b"",  # which nvidia-smi check
        "0, 15000, 10\n1, 2000, 90\n2, [Not Supported], 40\n3, 8000, 45\n",
    ]

    # Needs 4MB free, under 50%
    gpus = get_available_gpus(min_free_mb=4000, max_utilization=50)

    # GPU 0 has 15000 > 4000, util 10% < 50% -> YES
    # GPU 1 has 2000 < 4000 -> NO
    # GPU 2 has "[Not Supported]" free -> errors out silently -> NO
    # But wait, GPU 2 says it's free. If it is int parsing error it skips.
    # Actually wait: format is "index, free, util" -> "2, [Not Supported], 40" -> ValueError on float('[Not Supported]')
    # Let's see GPU 3: 8000 > 4000, util 45% < 50% -> YES
    assert gpus == [0, 3]


@mock.patch("subprocess.check_output")
def test_get_available_gpus_not_supported_util(mock_check_output):
    mock_check_output.side_effect = [b"", "0, 8000, [Not Supported]\n"]
    # It should assume 0% if util is "[Not Supported]"
    gpus = get_available_gpus(min_free_mb=4000, max_utilization=50)
    assert gpus == [0]


@mock.patch("subprocess.check_output")
def test_get_available_gpus_no_smi(mock_check_output):
    """Test when nvidia-smi is missing."""
    mock_check_output.side_effect = FileNotFoundError()
    assert get_available_gpus() == []


@mock.patch("subprocess.check_output")
def test_get_available_gpus_command_error(mock_check_output):
    mock_check_output.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")
    assert get_available_gpus() == []


@mock.patch("ndswin.utils.gpu.subprocess.check_output")
@mock.patch("ndswin.utils.gpu.warnings.warn")
def test_get_available_gpus_unexpected_error(mock_warn, mock_check_output):
    # allow 'which nvidia-smi'
    # error on the actual command
    mock_check_output.side_effect = [b"", Exception("Unexpected error")]
    assert get_available_gpus() == []
    mock_warn.assert_called_once()


@mock.patch("ndswin.utils.gpu.get_available_gpus", return_value=[1, 2])
def test_setup_optimal_gpus_binds(mock_get, monkeypatch, caplog):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with caplog.at_level("INFO", logger="ndswin.gpu"):
        setup_optimal_gpus()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert "Bound CUDA_VISIBLE_DEVICES to available GPUs" in caplog.text


@mock.patch("ndswin.utils.gpu.get_available_gpus", return_value=[])
def test_setup_optimal_gpus_empty_no_bind(mock_get, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    setup_optimal_gpus()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@mock.patch("ndswin.utils.gpu.get_available_gpus", return_value=[0])
def test_setup_optimal_gpus_respects_existing(mock_get, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    setup_optimal_gpus()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
