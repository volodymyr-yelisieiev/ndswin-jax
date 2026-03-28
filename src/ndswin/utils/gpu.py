"""Utilities for robust GPU allocation."""

import logging
import os
import shutil
import subprocess
import warnings

logger = logging.getLogger("ndswin.gpu")


def get_available_gpus(
    min_free_mb: int = 4000,
    max_utilization: int = 50,
    max_used_mb: int = 512,
) -> list[int]:
    """Query nvidia-smi to find GPUs meeting free memory and utilization thresholds.

    Args:
        min_free_mb: Minimum free memory required on the GPU in megabytes.
        max_utilization: Maximum allowed GPU utilization percentage.
        max_used_mb: Maximum used memory allowed on the GPU in megabytes.

    Returns:
        List of integer GPU IDs that meet the criteria. Returns a list of all GPUs (or empty)
        if nvidia-smi fails or is unavailable.
    """
    if shutil.which("nvidia-smi") is None:
        # nvidia-smi not available (e.g. CPU or TPU environment, or different OS)
        return []

    try:
        # Query free/used memory plus utilization so we can avoid partially occupied GPUs.
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.check_output(cmd, universal_newlines=True)

        candidates: list[tuple[int, float, float, float]] = []
        for line in result.strip().split("\n"):
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue

            idx_str, free_mem_str, used_mem_str, util_str = parts

            # Handle cases where utilization counter might be unsupported ("[Not Supported]")
            try:
                idx = int(idx_str)
                free_mem = float(free_mem_str)
                used_mem = float(used_mem_str)
                # If util is strictly "[Not Supported]", assume 0% to allow memory-based allocation
                util = float(util_str) if util_str.replace(".", "").isdigit() else 0.0

                if free_mem >= min_free_mb and used_mem <= max_used_mb and util <= max_utilization:
                    candidates.append((idx, free_mem, used_mem, util))
            except ValueError:
                continue

        candidates.sort(key=lambda item: (-item[1], item[2], item[3], item[0]))
        return [idx for idx, _free_mem, _used_mem, _util in candidates]

    except Exception as e:
        warnings.warn(f"Failed to query nvidia-smi for optimal GPUs: {e}", stacklevel=2)
        return []


def setup_optimal_gpus(
    min_free_mb: int = 4000,
    max_utilization: int = 50,
    max_used_mb: int = 512,
) -> None:
    """Configures CUDA_VISIBLE_DEVICES intelligently to only utilize non-busy GPUs.

    This function MUST be called BEFORE importing `jax` or any library that initializes
    the XLA backend.

    Args:
        min_free_mb: Minimum free memory required in megabytes (default 4000).
        max_utilization: Maximum percent utilization acceptable (default 50%).
        max_used_mb: Maximum used memory tolerated on a candidate GPU (default 512).
    """
    if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # If CUDA_VISIBLE_DEVICES is already explicitly set by the user/scheduler, respect it
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.info(
            "Respecting existing CUDA_VISIBLE_DEVICES=%s",
            os.environ["CUDA_VISIBLE_DEVICES"],
        )
        return

    available_gpus = get_available_gpus(min_free_mb, max_utilization, max_used_mb)

    if available_gpus:
        # Map intelligently to only the free GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
        logger.info(
            "Bound CUDA_VISIBLE_DEVICES to available GPUs: %s "
            "(min_free_mb=%d, max_used_mb=%d, max_utilization=%d)",
            available_gpus,
            min_free_mb,
            max_used_mb,
            max_utilization,
        )
    else:
        # If the query returned empty (e.g. no Nvidia driver, or genuinely 0 GPUs met threshold)
        # we don't set CUDA_VISIBLE_DEVICES, letting JAX fallback to default behavior (CPU/TPU/all-GPUs)
        logger.info(
            "No GPUs satisfied allocation thresholds; leaving CUDA_VISIBLE_DEVICES unset "
            "(min_free_mb=%d, max_used_mb=%d, max_utilization=%d).",
            min_free_mb,
            max_used_mb,
            max_utilization,
        )
