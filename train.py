"""Training entrypoint and helpers for the ndswin classifier."""

from __future__ import annotations

import argparse
import jax
from pathlib import Path

jax.config.update('jax_platform_name', 'gpu')

from training import CheckpointManager, apply_overrides, load_config, train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the ndswin-jax classifier")
    parser.add_argument(
        "--config",
        default="configs.cifar10",
        help="Python module path to a get_config() function.",
    )
    parser.add_argument("--workdir", required=True, help="Directory to store checkpoints and logs.")
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        help="Override configuration values, e.g. --config_override model.num_classes=3",
    )
    return parser.parse_args()


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.config_override)
    
    checkpoint_manager = CheckpointManager(Path(args.workdir), keep=config.keep_checkpoints)
    train(config, checkpoint_manager)


if __name__ == "__main__":
    main()
