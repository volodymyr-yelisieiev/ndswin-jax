"""Tests for the queue runner."""

import json
import sys
import subprocess
from pathlib import Path

import pytest

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@pytest.fixture
def sample_queue(tmp_path: Path) -> Path:
    """Create a sample queue YAML file."""
    queue_data = {
        "jobs": [
            {
                "name": "dry_sweep_1",
                "type": "sweep",
                "sweep": "configs/sweeps/cifar100_hyperparam_sweep.yaml",
                "trials": 2,
                "outdir": str(tmp_path / "sweep_out_1"),
            },
            {
                "name": "dry_sweep_2",
                "type": "sweep",
                "sweep": "configs/sweeps/cifar100_hyperparam_sweep.yaml",
                "trials": 1,
                "outdir": str(tmp_path / "sweep_out_2"),
            },
        ]
    }
    queue_path = tmp_path / "test_queue.yaml"
    if HAS_YAML:
        queue_path.write_text(yaml.dump(queue_data))
    else:
        # Fallback to JSON
        queue_path = tmp_path / "test_queue.json"
        queue_path.write_text(json.dumps(queue_data))
    return queue_path


@pytest.fixture
def sample_queue_json(tmp_path: Path) -> Path:
    """Create a sample queue JSON file."""
    queue_data = [
        {
            "name": "dry_train",
            "type": "train",
            "config": "configs/cifar100.json",
            "epochs": 1,
        }
    ]
    queue_path = tmp_path / "test_queue.json"
    queue_path.write_text(json.dumps(queue_data))
    return queue_path


def test_load_queue_yaml(sample_queue: Path):
    """Test loading a YAML queue file."""
    from train.queue_runner import load_queue

    jobs = load_queue(str(sample_queue))
    assert len(jobs) == 2
    assert jobs[0]["name"] == "dry_sweep_1"
    assert jobs[1]["name"] == "dry_sweep_2"


def test_load_queue_json(sample_queue_json: Path):
    """Test loading a JSON queue file."""
    from train.queue_runner import load_queue

    jobs = load_queue(str(sample_queue_json))
    assert len(jobs) == 1
    assert jobs[0]["name"] == "dry_train"


def test_load_queue_missing(tmp_path: Path):
    """Test loading a non-existent queue file."""
    from train.queue_runner import load_queue

    with pytest.raises(FileNotFoundError):
        load_queue(str(tmp_path / "nonexistent.yaml"))


def test_load_queue_empty(tmp_path: Path):
    """Test loading an empty queue file."""
    from train.queue_runner import load_queue

    queue_path = tmp_path / "empty.json"
    queue_path.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="No jobs found"):
        load_queue(str(queue_path))


def test_build_command_auto_sweep():
    """Test building auto-sweep command."""
    from train.queue_runner import build_command

    job = {
        "type": "auto-sweep",
        "sweep": "configs/sweeps/cifar100_hyperparam_sweep.yaml",
        "trials": 5,
        "train_epochs": 50,
    }
    cmd = build_command(job, "python")
    assert "train/auto_sweep_and_train.py" in cmd
    assert "--sweep" in cmd
    assert "--trials" in cmd
    assert "5" in cmd
    assert "--train-epochs" in cmd
    assert "50" in cmd


def test_build_command_sweep():
    """Test building sweep command."""
    from train.queue_runner import build_command

    job = {
        "type": "sweep",
        "sweep": "configs/sweeps/cifar100_hyperparam_sweep.yaml",
        "trials": 3,
        "outdir": "/tmp/sweep_out",
    }
    cmd = build_command(job, "python")
    assert "train/run_sweep.py" in cmd
    assert "--outdir" in cmd


def test_build_command_train():
    """Test building train command."""
    from train.queue_runner import build_command

    job = {"type": "train", "config": "configs/cifar100.json", "epochs": 10}
    cmd = build_command(job, "python")
    assert "train/train.py" in cmd
    assert "--config" in cmd
    assert "--epochs" in cmd


def test_build_command_unknown_type():
    """Test building command with unknown type."""
    from train.queue_runner import build_command

    with pytest.raises(ValueError, match="Unknown job type"):
        build_command({"type": "invalid"}, "python")


def test_build_command_missing_fields():
    """Test building command with missing required fields."""
    from train.queue_runner import build_command

    with pytest.raises(ValueError, match="requires 'sweep'"):
        build_command({"type": "auto-sweep"}, "python")

    with pytest.raises(ValueError, match="requires 'config'"):
        build_command({"type": "train"}, "python")


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_queue_runner_dry_run(tmp_path: Path):
    """Test running queue in dry-run mode via subprocess."""
    queue_data = {
        "jobs": [
            {
                "name": "dry_sweep",
                "type": "sweep",
                "sweep": "configs/sweeps/cifar100_hyperparam_sweep.yaml",
                "trials": 1,
                "outdir": str(tmp_path / "sweep_out"),
            }
        ]
    }
    queue_path = tmp_path / "queue.yaml"
    queue_path.write_text(yaml.dump(queue_data))

    # The dry-run only prints commands, doesn't execute them
    cmd = [
        sys.executable,
        "train/queue_runner.py",
        "--queue",
        str(queue_path),
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "QUEUE RUNNER" in result.stdout
    assert "dry_sweep" in result.stdout


def test_config_3d_classification_valid():
    """Test that the ModelNet10 3D config is valid."""
    config_path = Path("configs/modelnet10.json")
    if not config_path.exists():
        pytest.skip("modelnet10.json not found")

    from ndswin.config import ExperimentConfig

    with open(config_path) as f:
        config_dict = json.load(f)

    exp = ExperimentConfig.from_dict(config_dict)
    assert exp.model.num_dims == 3
    assert len(exp.model.patch_size) == 3
    assert len(exp.model.window_size) == 3
    assert exp.model.num_classes == 40
    assert exp.model.embed_dim > 0


def test_config_2d_classification_valid():
    """Test that the CIFAR-100 2D config is valid."""
    config_path = Path("configs/cifar100.json")
    if not config_path.exists():
        pytest.skip("cifar100.json not found")

    from ndswin.config import ExperimentConfig

    with open(config_path) as f:
        config_dict = json.load(f)

    exp = ExperimentConfig.from_dict(config_dict)
    assert exp.model.num_dims == 2
    assert len(exp.model.patch_size) == 2
    assert exp.model.num_classes == 100
    assert exp.model.embed_dim == 48  # verify we use the smaller size


def test_sweep_config_embed_dim_divisible():
    """Test that all embed_dim choices in sweep configs are divisible by num_heads[0]."""
    sweep_path = Path("configs/sweeps/cifar100_hyperparam_sweep.yaml")
    if not sweep_path.exists():
        pytest.skip("cifar100 sweep not found")

    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")

    with open(sweep_path) as f:
        sweep = yaml.safe_load(f)

    embed_choices = sweep.get("param_space", {}).get("model.embed_dim", {}).get("values", [])
    # Default num_heads[0] for Swin is 3
    for dim in embed_choices:
        assert dim % 3 == 0, f"embed_dim {dim} not divisible by num_heads[0]=3"
