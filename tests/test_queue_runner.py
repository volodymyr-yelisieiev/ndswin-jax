"""Tests for the queue runner."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = (
        src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    )
    return env


@pytest.fixture
def sample_queue(tmp_path: Path) -> Path:
    """Create a sample queue YAML file."""
    queue_data = {
        "jobs": [
            {
                "name": "dry_sweep_1",
                "type": "sweep",
                "sweep": "configs/sweeps/cifar10_hyperparam_sweep.yaml",
                "trials": 2,
                "outdir": str(tmp_path / "sweep_out_1"),
            },
            {
                "name": "dry_sweep_2",
                "type": "sweep",
                "sweep": "configs/sweeps/cifar10_hyperparam_sweep.yaml",
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
            "config": "configs/cifar10.json",
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
        "sweep": "configs/sweeps/cifar10_hyperparam_sweep.yaml",
        "trials": 5,
        "train_epochs": 50,
    }
    cmd = build_command(job, "python")
    assert "-m" in cmd
    assert "ndswin.cli" in cmd
    assert "auto-sweep" in cmd
    assert "--sweep" in cmd
    assert "--trials" in cmd
    assert "5" in cmd
    assert "--train-epochs" in cmd
    assert "50" in cmd


def test_build_command_auto_sweep_omits_trials_when_not_specified():
    """Queue auto-sweep jobs should defer to sweep-file trials when unset."""
    from ndswin.cli import build_command

    job = {
        "type": "auto-sweep",
        "sweep": "configs/sweeps/cifar10_hyperparam_sweep.yaml",
        "train_epochs": 25,
    }
    cmd = build_command(job, "python")
    assert "auto-sweep" in cmd
    assert "--trials" not in cmd


def test_build_command_sweep():
    """Test building sweep command."""
    from train.queue_runner import build_command

    job = {
        "type": "sweep",
        "sweep": "configs/sweeps/cifar10_hyperparam_sweep.yaml",
        "trials": 3,
        "outdir": "/tmp/sweep_out",
    }
    cmd = build_command(job, "python")
    assert "-m" in cmd
    assert "ndswin.cli" in cmd
    assert "sweep" in cmd
    assert "--outdir" in cmd


def test_build_command_train():
    """Test building train command."""
    from train.queue_runner import build_command

    job = {"type": "train", "config": "configs/cifar10.json", "epochs": 10}
    cmd = build_command(job, "python")
    assert "-m" in cmd
    assert "ndswin.cli" in cmd
    assert "train" in cmd
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
                "sweep": "configs/sweeps/cifar10_hyperparam_sweep.yaml",
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
        "-m",
        "ndswin.cli",
        "queue",
        "--queue",
        str(queue_path),
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=cli_env())
    assert result.returncode == 0
    assert "QUEUE RUNNER" in result.stdout
    assert "dry_sweep" in result.stdout


def test_load_completed_jobs(tmp_path: Path):
    """Test loading completed jobs from previous queue results."""
    from train.queue_runner import load_completed_jobs

    # Create some fake queue result files
    (tmp_path / "queue_20260101_120000.json").write_text(
        json.dumps(
            [
                {"name": "job_a", "status": "completed"},
                {"name": "job_b", "status": "failed"},
            ]
        )
    )
    (tmp_path / "queue_20260102_120000.json").write_text(
        json.dumps(
            [
                {"name": "job_c", "status": "completed"},
                {"name": "job_d", "status": "dry"},
            ]
        )
    )

    completed = load_completed_jobs(tmp_path)
    assert completed == {"job_a", "job_c"}
    assert "job_b" not in completed
    assert "job_d" not in completed


def test_load_completed_jobs_empty(tmp_path: Path):
    """Test loading completed jobs from empty directory."""
    from train.queue_runner import load_completed_jobs

    completed = load_completed_jobs(tmp_path)
    assert completed == set()


def test_load_completed_jobs_corrupt_file(tmp_path: Path):
    """Test loading completed jobs gracefully handles corrupt files."""
    from train.queue_runner import load_completed_jobs

    (tmp_path / "queue_bad.json").write_text("not valid json{{{")
    completed = load_completed_jobs(tmp_path)
    assert completed == set()


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
    assert exp.model.num_classes == 10
    assert exp.model.embed_dim > 0


def test_config_3d_modelnet40_valid():
    """Test that the ModelNet40 3D config is valid."""
    config_path = Path("configs/modelnet40.json")
    if not config_path.exists():
        pytest.skip("modelnet40.json not found")

    from ndswin.config import ExperimentConfig

    with open(config_path) as f:
        config_dict = json.load(f)

    exp = ExperimentConfig.from_dict(config_dict)
    assert exp.model.num_dims == 3
    assert exp.model.num_classes == 40
    assert len(exp.model.patch_size) == 3
    assert len(exp.model.window_size) == 3


def test_config_2d_classification_valid():
    """Test that the CIFAR-10 2D config is valid."""
    config_path = Path("configs/cifar10.json")
    if not config_path.exists():
        pytest.skip("cifar10.json not found")

    from ndswin.config import ExperimentConfig

    with open(config_path) as f:
        config_dict = json.load(f)

    exp = ExperimentConfig.from_dict(config_dict)
    assert exp.model.num_dims == 2
    assert len(exp.model.patch_size) == 2
    assert exp.model.num_classes == 10
    assert exp.model.embed_dim == 48  # verify we use the smaller size


def test_config_descriptions_accurate():
    """Test that config name and description fields match the file."""
    config_path = Path("configs/cifar10.json")
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        assert data["name"] == "cifar10"
        assert "CIFAR-10" in data["description"]
        assert "10 classes" in data["description"]

    config_path = Path("configs/modelnet10.json")
    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        assert data["name"] == "modelnet10"
        assert "ModelNet10" in data["description"]
        assert "10 classes" in data["description"]


def test_sweep_config_embed_dim_divisible():
    """Test that all embed_dim choices in sweep configs are divisible by num_heads[0]."""
    sweep_path = Path("configs/sweeps/cifar10_hyperparam_sweep.yaml")
    if not sweep_path.exists():
        pytest.skip("cifar10 sweep not found")

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


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_example_queue_valid():
    """Test that the example queue file is valid and loadable."""
    from train.queue_runner import load_queue

    queue_path = Path("configs/queues/example_queue.yaml")
    if not queue_path.exists():
        pytest.skip("example_queue.yaml not found")

    jobs = load_queue(str(queue_path))
    assert len(jobs) >= 2

    # All jobs must have names
    for job in jobs:
        assert "name" in job
        assert "type" in job
        assert job["type"] in ("auto-sweep", "sweep", "train")
