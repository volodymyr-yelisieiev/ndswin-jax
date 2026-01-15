# Configuration Files

This directory contains training configuration files for NDSwin-JAX experiments.

## Usage

To train with a specific configuration:

```bash
# Using make (recommended)
make train CONFIG=configs/cifar100.json

# Using train script directly
python train/train.py --config configs/cifar100.json

# With overrides
python train/train.py --config configs/cifar100.json --epochs 200 --batch-size 256
```

## Available Configurations

| Config File | Dataset | Dims | Description |
|-------------|---------|------|-------------|
| `cifar100.json` | CIFAR-100 | 2D | CIFAR-100 training with stronger augmentation and regularization |

## Configuration Structure

Each configuration file contains the following sections:

### Metadata
- `name`: Unique identifier for the experiment
- `description`: Human-readable description

### Dataset
```json
{
    "dataset": {
        "name": "cifar100",           // Dataset name: cifar10, cifar100, synthetic
        "data_dir": "data",           // Directory for dataset files
        "download": true,             // Auto-download if missing
        "num_classes": 100,           // Number of classes
        "input_shape": [3, 32, 32],   // Input shape (C, *spatial)
        "mean": [0.5, 0.5, 0.5],      // Normalization mean
        "std": [0.25, 0.25, 0.25]     // Normalization std
    }
}
```

### Model
```json
{
    "model": {
        "num_dims": 2,                // Number of spatial dimensions
        "patch_size": [4, 4],         // Patch embedding size
        "window_size": [4, 4],        // Attention window size
        "in_channels": 3,             // Input channels
        "embed_dim": 96,              // Base embedding dimension
        "depths": [2, 2, 6, 2],       // Blocks per stage
        "num_heads": [3, 6, 12, 24],  // Heads per stage
        "mlp_ratio": 4.0,             // MLP expansion ratio
        "drop_path_rate": 0.1,        // Stochastic depth rate
        "drop_rate": 0.0,             // Dropout rate
        "attn_drop_rate": 0.0         // Attention dropout rate
    }
}
```

### Training
```json
{
    "training": {
        "epochs": 100,                // Number of epochs
        "batch_size": 128,            // Batch size
        "learning_rate": 1e-3,        // Initial learning rate
        "min_learning_rate": 1e-6,    // Minimum LR for schedule
        "weight_decay": 0.05,         // Weight decay
        "optimizer": "adamw",         // Optimizer: adamw, sgd
        "lr_schedule": "cosine",      // Schedule: cosine, linear, step
        "warmup_epochs": 5,           // Warmup epochs
        "label_smoothing": 0.1,       // Label smoothing
        "gradient_clip_norm": 1.0     // Gradient clipping
    }
}
```

### Augmentation
```json
{
    "augmentation": {
        "random_crop": true,          // Random cropping
        "crop_padding": 4,            // Padding for random crop
        "random_flip": true,          // Horizontal flip
        "color_jitter": false,        // Color jittering
        "mixup_alpha": 0.0,           // Mixup alpha (0 = disabled)
        "cutmix_alpha": 0.0,          // CutMix alpha (0 = disabled)
        "cutout_size": 0              // Cutout size (0 = disabled)
    }
}
```

### Checkpointing and Logging
```json
{
    "checkpointing": {
        "save_every": 5000,           // Steps between checkpoints
        "max_to_keep": 5              // Max checkpoints to keep
    },
    "logging": {
        "log_every": 50               // Steps between log entries
    },
    "seed": 42                        // Random seed
}
```

## Creating Custom Configurations

1. Copy an existing config as a template
2. Modify parameters as needed
3. Give it a unique `name`
4. Run training with your new config

## Output Structure

Training creates timestamped outputs:
```
checkpoints/
└── cifar100_20260115_143022/
    ├── config.json          # Copy of config used
    ├── checkpoint_*.npz     # Model checkpoints
    └── metrics.json         # Training metrics

logs/
└── cifar100_20260115_143022.log
```

The folder name includes the config name and timestamp for easy identification.
