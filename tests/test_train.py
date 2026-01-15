from train.train import setup_output_dirs
from ndswin.config import ExperimentConfig
from pathlib import Path

def test_setup_output_dirs(tmp_path: Path):
    # This also guarantees no syntax errors in train.py itself
    config = ExperimentConfig()
    
    config_file = tmp_path / "test_config.json"
    config_file.touch()
    
    # Check if we can create output directories
    checkdir, cfgdir = setup_output_dirs(config, str(config_file), stamp="test_stamp")
    
    assert str(checkdir).endswith("checkpoints")
    assert str(cfgdir).endswith(".log")
