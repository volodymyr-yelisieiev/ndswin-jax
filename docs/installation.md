# Installation Guide

This guide covers the installation of NDSwin-JAX for different platforms and use cases.

## Quick Start

### Using pip (recommended)

```bash
pip install ndswin-jax
```

### From source

```bash
git clone https://github.com/yourusername/ndswin-jax.git
cd ndswin-jax
pip install -e .
```

## Prerequisites

- Python 3.11 or higher
- CUDA 12.0+ (for GPU support)

## Detailed Installation

### 1. Create a virtual environment

We recommend using conda for managing environments:

```bash
# Create a new conda environment
conda create -n ndswin python=3.11
conda activate ndswin
```

Or using venv:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

### 2. Install JAX

JAX installation varies by platform:

#### CPU only

```bash
pip install "jax[cpu]"
```

#### CUDA (NVIDIA GPU)

```bash
# For CUDA 12 (recommended)
pip install "jax[cuda12_pip]==0.7.1" jax-cuda12-plugin==0.7.1 jax-cuda12-pjrt==0.7.1
# Also ensure CUDA compiler tools are available for PTX compilation:
pip install nvidia-cuda-nvcc-cu12
```

#### TPU

```bash
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 3. Install NDSwin-JAX

```bash
# Basic installation
pip install ndswin-jax

# With all optional dependencies
pip install "ndswin-jax[all]"

# Development installation
pip install "ndswin-jax[dev]"
```

### 4. Verify installation

```python
import jax
import ndswin

print(f"JAX version: {jax.__version__}")
print(f"NDSwin version: {ndswin.__version__}")
print(f"Available devices: {jax.devices()}")

# Quick test
from ndswin import NDSwinConfig, NDSwinTransformer
import jax.numpy as jnp

config = NDSwinConfig.swin_tiny_2d()
model = NDSwinTransformer(config=config)
print("Installation successful!")
```

## Using Conda Environment File

We provide an environment file for reproducible installations:

```bash
# Clone the repository
git clone https://github.com/yourusername/ndswin-jax.git
cd ndswin-jax

# Create environment from file
conda env create -f environment.yml

# Activate
conda activate ndswin-jax

# Install the package
pip install -e .
```

## Platform-Specific Notes

### Linux

Most straightforward installation. Ensure CUDA drivers are installed for GPU support.

### macOS

- Apple Silicon (M1/M2): JAX supports Metal GPU acceleration
- Intel Macs: CPU only is recommended

```bash
# Apple Silicon
pip install jax jaxlib
```

### Windows

- Native Windows support is limited
- Recommended: Use WSL2 with Ubuntu

```bash
# In WSL2
pip install "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Troubleshooting

### JAX not finding GPU

```python
import jax
print(jax.devices())  # Should show GPU devices
```

If only CPU is shown:
1. Check CUDA installation: `nvidia-smi`
2. Verify CUDA version matches JAX requirements
3. Reinstall JAX with correct CUDA version

### Import errors

```bash
# Reinstall dependencies
pip install --upgrade jax jaxlib flax optax
```

### Memory issues

```python
# Limit GPU memory growth
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [Configuration Guide](configuration.md)
