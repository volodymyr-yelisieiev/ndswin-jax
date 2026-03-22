"""Tests for logging utilities."""

import logging
import tempfile

import pytest

from ndswin.utils.logging import (
    LogConfig,
    LogLevel,
    MetricLogger,
    ProgressLogger,
    get_logger,
    log_dict,
    setup_logging,
)


@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_log_config():
    """Test log config."""
    config = LogConfig(level=LogLevel.DEBUG, log_to_file=False)
    assert config.level == LogLevel.DEBUG
    assert not config.log_to_file


def test_setup_logging(temp_log_dir):
    """Test basic logging setup."""
    setup_logging(
        level="debug",
        log_dir=temp_log_dir,
        log_to_file=True,
        log_to_console=True,
    )

    logger = get_logger("test_logger")
    assert logger.getEffectiveLevel() == logging.DEBUG

    # Need to check if there is a file handler
    handlers = logging.getLogger().handlers
    assert len(handlers) >= 1

    # Try sending a log
    logger.info("Test message")


def test_setup_logging_with_config(temp_log_dir):
    config = LogConfig(
        level=LogLevel.WARNING,
        log_dir=temp_log_dir,
        log_to_file=False,
    )
    setup_logging(config=config)
    get_logger("test_cfg")
    # Will not raise error


def test_metric_logger(temp_log_dir):
    """Test MetricLogger averaging."""
    setup_logging(level=LogLevel.INFO, log_dir=temp_log_dir, log_to_file=False)

    ml = MetricLogger(name="metrics_test")
    ml.update({"loss": 1.0, "acc": 0.5})
    ml.update({"loss": 0.5, "acc": 0.7})

    assert ml.get_average("loss") == 0.75
    assert ml.get_average("acc") == 0.6

    assert ml.get_averages() == {"loss": 0.75, "acc": 0.6}
    assert ml.get_average("unknown") == 0.0

    # Log it
    ml.log()
    ml.log(prefix="Valid", step=10)

    # Reset
    ml.reset()
    assert ml.get_average("loss") == 0.0


def test_progress_logger(temp_log_dir):
    """Test ProgressLogger execution logic."""
    setup_logging(level=LogLevel.INFO, log_dir=temp_log_dir, log_to_file=False)

    pl = ProgressLogger(total=10, log_interval=2)
    pl.start()

    for i in range(10):
        pl.update(1)

    pl.finish()


def test_log_dict(temp_log_dir):
    """Test log dictionary mapping."""
    logger = get_logger()
    log_dict(logger, {"metric_int": 5, "metric_float": 0.333333})
