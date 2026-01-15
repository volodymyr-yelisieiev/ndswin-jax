"""Logging utilities for NDSwin-JAX.

This module provides structured logging utilities with support for
console output, file logging, and integration with TensorBoard and Weights & Biases.
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class LogLevel(Enum):
    """Log level enumeration."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogConfig:
    """Configuration for logging.

    Attributes:
        level: Logging level.
        log_dir: Directory for log files.
        log_to_file: Whether to log to file.
        log_to_console: Whether to log to console.
        use_rich: Whether to use rich formatting for console.
        filename_prefix: Prefix for log filenames.
        format_string: Format string for log messages.
    """

    level: LogLevel = LogLevel.INFO
    log_dir: str = "logs"
    log_to_file: bool = True
    log_to_console: bool = True
    use_rich: bool = True
    filename_prefix: str = "ndswin"
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# Global logger cache
_loggers: dict[str, logging.Logger] = {}
_log_config: LogConfig | None = None


def setup_logging(
    config: LogConfig | None = None,
    level: LogLevel | str | int = LogLevel.INFO,
    log_dir: str | None = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    use_rich: bool = True,
) -> None:
    """Set up global logging configuration.

    Args:
        config: LogConfig object. If provided, other arguments are ignored.
        level: Logging level.
        log_dir: Directory for log files.
        log_to_file: Whether to log to file.
        log_to_console: Whether to log to console.
        use_rich: Whether to use rich formatting for console.
    """
    global _log_config

    if config is not None:
        _log_config = config
    else:
        # Convert level to LogLevel
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        elif isinstance(level, int):
            level = LogLevel(level)

        _log_config = LogConfig(
            level=level,
            log_dir=log_dir or "logs",
            log_to_file=log_to_file,
            log_to_console=log_to_console,
            use_rich=use_rich and RICH_AVAILABLE,
        )

    # Create log directory if needed
    if _log_config.log_to_file:
        Path(_log_config.log_dir).mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(_log_config.level.value)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Add console handler
    if _log_config.log_to_console:
        if _log_config.use_rich and RICH_AVAILABLE:
            console_handler: logging.Handler = RichHandler(
                markup=True,
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(_log_config.format_string))
        root_logger.addHandler(console_handler)

    # Add file handler
    if _log_config.log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(_log_config.log_dir) / f"{_log_config.filename_prefix}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(_log_config.format_string))
        root_logger.addHandler(file_handler)


def get_logger(name: str = "ndswin") -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    global _loggers, _log_config

    if name in _loggers:
        return _loggers[name]

    # Ensure logging is set up
    if _log_config is None:
        setup_logging()

    logger = logging.getLogger(name)
    _loggers[name] = logger

    return logger


class MetricLogger:
    """Logger for training metrics.

    This class provides utilities for logging training metrics with support
    for averaging, formatting, and integration with experiment trackers.

    Attributes:
        name: Logger name.
        delimiter: Delimiter for metric output.
    """

    def __init__(self, name: str = "metrics", delimiter: str = "  ") -> None:
        """Initialize MetricLogger.

        Args:
            name: Logger name.
            delimiter: Delimiter for metric output.
        """
        self.name = name
        self.delimiter = delimiter
        self._logger = get_logger(name)
        self._metrics: dict[str, list[float]] = {}
        self._step = 0

    def update(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Update metrics.

        Args:
            metrics: Dictionary of metric name to value.
            step: Current step. If None, uses internal counter.
        """
        if step is not None:
            self._step = step
        else:
            self._step += 1

        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(value)

    def log(self, prefix: str = "", step: int | None = None) -> None:
        """Log current metrics.

        Args:
            prefix: Prefix for log message.
            step: Current step to include in log.
        """
        if not self._metrics:
            return

        step_str = f"[Step {step or self._step}]" if step or self._step else ""
        parts = [f"{prefix} {step_str}".strip()]

        for key, values in self._metrics.items():
            if values:
                avg = sum(values) / len(values)
                parts.append(f"{key}: {avg:.4f}")

        message = self.delimiter.join(parts)
        self._logger.info(message)

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()

    def get_average(self, key: str) -> float:
        """Get average value for a metric.

        Args:
            key: Metric name.

        Returns:
            Average value, or 0.0 if metric doesn't exist.
        """
        if key not in self._metrics or not self._metrics[key]:
            return 0.0
        return sum(self._metrics[key]) / len(self._metrics[key])

    def get_averages(self) -> dict[str, float]:
        """Get averages for all metrics.

        Returns:
            Dictionary of metric name to average value.
        """
        return {key: self.get_average(key) for key in self._metrics}


class ProgressLogger:
    """Logger for progress tracking.

    This class provides utilities for logging progress during training
    with ETA estimation and throughput calculation.
    """

    def __init__(
        self,
        total: int,
        desc: str = "",
        log_interval: int = 1,
    ) -> None:
        """Initialize ProgressLogger.

        Args:
            total: Total number of steps.
            desc: Description for the progress.
            log_interval: Interval between log updates.
        """
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self._logger = get_logger("progress")
        self._start_time: datetime | None = None
        self._current = 0

    def start(self) -> None:
        """Start progress tracking."""
        self._start_time = datetime.now()
        self._current = 0

    def update(self, n: int = 1) -> None:
        """Update progress.

        Args:
            n: Number of steps completed.
        """
        self._current += n

        if self._current % self.log_interval == 0 or self._current == self.total:
            self._log_progress()

    def _log_progress(self) -> None:
        """Log current progress."""
        if self._start_time is None:
            return

        elapsed = (datetime.now() - self._start_time).total_seconds()
        progress = self._current / self.total
        eta = (elapsed / progress - elapsed) if progress > 0 else 0

        self._logger.info(
            f"{self.desc} [{self._current}/{self.total}] "
            f"({progress * 100:.1f}%) - "
            f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
        )

    def finish(self) -> None:
        """Finish progress tracking."""
        if self._start_time is None:
            return

        elapsed = (datetime.now() - self._start_time).total_seconds()
        throughput = self.total / elapsed if elapsed > 0 else 0

        self._logger.info(
            f"{self.desc} Complete - "
            f"Total: {self.total}, "
            f"Time: {elapsed:.1f}s, "
            f"Throughput: {throughput:.1f} items/s"
        )


def log_dict(
    logger: logging.Logger,
    data: dict[str, Any],
    prefix: str = "",
    level: LogLevel = LogLevel.INFO,
) -> None:
    """Log a dictionary of values.

    Args:
        logger: Logger instance.
        data: Dictionary to log.
        prefix: Prefix for log message.
        level: Logging level.
    """
    parts = [prefix] if prefix else []
    for key, value in data.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")

    message = " | ".join(parts)
    logger.log(level.value, message)
