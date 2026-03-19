"""Utilities for robust GPU allocation."""

import os
import subprocess
import warnings
from typing import List


def get_available_gpus(min_free_mb: int = 4000, max_utilization: int = 50) -> List[int]:
    """Query nvidia-smi to find GPUs meeting free memory and utilization thresholds.
    
    Args:
        min_free_mb: Minimum free memory required on the GPU in megabytes.
        max_utilization: Maximum allowed GPU utilization percentage.
        
    Returns:
        List of integer GPU IDs that meet the criteria. Returns a list of all GPUs (or empty)
        if nvidia-smi fails or is unavailable.
    """
    try:
        # Check if nvidia-smi exists
        subprocess.check_output(["which", "nvidia-smi"], stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # nvidia-smi not available (e.g. CPU or TPU environment, or different OS)
        return []

    try:
        # Query memory.free, utilization.gpu
        # Output format: "index, memory.free, utilization.gpu" (e.g., "0, 15000 MiB, 10 %")
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.check_output(cmd, universal_newlines=True)
        
        available_gpus = []
        for line in result.strip().split("\n"):
            if not line:
                continue
            
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
                
            idx_str, free_mem_str, util_str = parts
            
            # Handle cases where utilization counter might be unsupported ("[Not Supported]")
            try:
                idx = int(idx_str)
                free_mem = float(free_mem_str)
                # If util is strictly "[Not Supported]", assume 0% to allow memory-based allocation
                util = float(util_str) if util_str.replace(".", "").isdigit() else 0.0
                
                if free_mem >= min_free_mb and util <= max_utilization:
                    available_gpus.append(idx)
            except ValueError:
                continue
                
        return available_gpus

    except Exception as e:
        warnings.warn(f"Failed to query nvidia-smi for optimal GPUs: {e}")
        return []


def setup_optimal_gpus(min_free_mb: int = 4000, max_utilization: int = 50) -> None:
    """Configures CUDA_VISIBLE_DEVICES intelligently to only utilize non-busy GPUs.
    
    This function MUST be called BEFORE importing `jax` or any library that initializes
    the XLA backend.
    
    Args:
        min_free_mb: Minimum free memory required in megabytes (default 4000).
        max_utilization: Maximum percent utilization acceptable (default 50%).
    """
    # If CUDA_VISIBLE_DEVICES is already explicitly set by the user/scheduler, respect it
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return

    available_gpus = get_available_gpus(min_free_mb, max_utilization)
    
    if available_gpus:
        # Map intelligently to only the free GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
        print(f"[GPU Allocation] Bound CUDA_VISIBLE_DEVICES to free GPUs: {available_gpus}")
    else:
        # If the query returned empty (e.g. no Nvidia driver, or genuinely 0 GPUs met threshold)
        # we don't set CUDA_VISIBLE_DEVICES, letting JAX fallback to default behavior (CPU/TPU/all-GPUs)
        pass
