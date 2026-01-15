# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Nothing yet

## [0.1.0] - 2026-01-13

### Added
- Initial release
- Core N-dimensional Swin Transformer implementation
- Support for 2D, 3D, 4D, and higher dimensional inputs
- Window partitioning and shifted window attention
- Relative position bias for attention
- Patch embedding and patch merging layers
- Hierarchical multi-stage architecture
- Classification head with global pooling
- Complete training infrastructure with Trainer class
- Multiple optimizers: AdamW, SGD, LAMB
- Learning rate schedules: Cosine, Linear, Step, OneCycle
- Data augmentation: RandomFlip, RandomCrop, Mixup, Cutout
- Loss functions: Cross-entropy, Focal, Dice, Distillation
- Metrics: Accuracy, Top-k, Precision, Recall, F1
- Checkpointing and early stopping
- ClassificationPredictor for easy inference
- BatchProcessor for efficient batch processing
- CIFAR-10 and CIFAR-100 training examples
- Comprehensive test suite (72 tests)
- Documentation and examples
- CI/CD with GitHub Actions

### Requirements
- Python 3.11+
- JAX 0.4.35+
- Flax 0.10.0+
