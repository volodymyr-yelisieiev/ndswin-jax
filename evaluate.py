#!/usr/bin/env python3
"""Evaluate a trained model on the test set."""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from loader import create_input_iter, get_dataset_info
from training import CheckpointManager, load_config
from training.state import create_model, create_train_state
from training.trainer import _build_eval_step


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ndswin-jax classifier")
    parser.add_argument(
        "--config",
            default="configs.cifar10",
        help="Python module path to get_config() function",
    )
    parser.add_argument(
        "--workdir",
        required=True,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--use_best",
        action="store_true",
        help="Use best checkpoint instead of latest",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate on",
    )
    return parser.parse_args()


def evaluate(config, checkpoint_manager, use_best=False, split="test"):
    """Run evaluation on the specified split.
    
    Args:
        config: Configuration dictionary
        checkpoint_manager: CheckpointManager instance
        use_best: Use best checkpoint instead of latest
        split: Dataset split to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Initialize model
    model = create_model(config)
    rng = jax.random.PRNGKey(config.seed)
    
    # Calculate steps_per_epoch for learning rate schedule
    train_info = get_dataset_info(config, config.train_split, config.batch_size)
    steps_per_epoch = train_info.steps_per_epoch
    
    # Create dummy state for structure (same as training)
    state = create_train_state(model, config, rng, steps_per_epoch)
    
    # Load checkpoint
    if use_best:
        checkpoint_dir = checkpoint_manager.best_checkpoint_dir
        print(f"Loading best checkpoint from {checkpoint_dir}")
        from flax.training import checkpoints
        state = checkpoints.restore_checkpoint(checkpoint_dir, target=state)
    else:
        state = checkpoint_manager.restore(state)
        print(f"Loaded checkpoint at step {state.step}")
    
    # Get dataset info
    eval_info = get_dataset_info(config, split, config.eval_batch_size)
    eval_steps = eval_info.steps_per_epoch
    print(f"Evaluating on {eval_info.examples} examples ({eval_steps} steps)")
    
    # Create evaluation function
    eval_step_fn = _build_eval_step(model.num_classes)
    
    # Create data iterator
    eval_iter = create_input_iter(
        config,
        split,
        config.eval_batch_size,
        seed=config.seed,
        shuffle=False,
        repeat=False,
    )
    
    # Run evaluation
    all_metrics = []
    progress = tqdm(range(eval_steps), desc="Evaluating")
    for _ in progress:
        try:
            batch = next(eval_iter)
            metrics = jax.device_get(eval_step_fn(state, batch))
            all_metrics.append(metrics)
            
            # Update progress bar
            current_acc = sum(m["accuracy"] for m in all_metrics) / len(all_metrics)
            current_loss = sum(m["loss"] for m in all_metrics) / len(all_metrics)
            progress.set_postfix(
                loss=f"{current_loss:.4f}",
                acc=f"{current_acc:.4f}",
            )
        except StopIteration:
            break
    
    # Compute final metrics
    final_loss = float(jnp.mean(jnp.array([m["loss"] for m in all_metrics])))
    final_accuracy = float(jnp.mean(jnp.array([m["accuracy"] for m in all_metrics])))
    
    return {
        "loss": final_loss,
        "accuracy": final_accuracy,
        "num_examples": eval_info.examples,
        "num_batches": len(all_metrics),
    }


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(Path(args.workdir))
    
    # Run evaluation
    results = evaluate(config, checkpoint_manager, args.use_best, args.split)
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Dataset split: {args.split}")
    print(f"Number of examples: {results['num_examples']}")
    print(f"Number of batches: {results['num_batches']}")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("=" * 50)
    
    # Save results
    import json
    results_path = Path(args.workdir) / f"eval_results_{args.split}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
