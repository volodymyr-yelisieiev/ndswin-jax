from pathlib import Path

from train.train import setup_output_dirs

from ndswin.config import ExperimentConfig


def test_setup_output_dirs(tmp_path: Path):
    # This also guarantees no syntax errors in train.py itself
    config = ExperimentConfig()

    config_file = tmp_path / "test_config.json"
    config_file.touch()

    # Check if we can create output directories
    checkdir, cfgdir = setup_output_dirs(config, str(config_file), stamp="test_stamp")

    assert str(checkdir).endswith("checkpoints")
    assert str(cfgdir).endswith(".log")


def test_trainer_tensorboard_init(tmp_path: Path):
    """Verify Trainer with use_tensorboard=True doesn't crash."""
    from ndswin import NDSwinTransformer
    from ndswin.config import NDSwinConfig, TrainingConfig
    from ndswin.training.trainer import Trainer

    config = TrainingConfig(batch_size=4, num_classes=10, epochs=10)
    model_config = NDSwinConfig.swin_tiny_2d(num_classes=10)
    model = NDSwinTransformer(config=model_config)

    checkpoint_dir = str(tmp_path / "checkpoints")

    # This should not raise
    trainer = Trainer(
        model=model,
        config=config,
        checkpoint_dir=checkpoint_dir,
        use_tensorboard=True,
    )
    assert hasattr(trainer, "tb_writer")
