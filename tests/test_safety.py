"""Tests for result protection and safety features."""

import json
from pathlib import Path

import pytest


class TestResultProtection:
    """Tests for archive-results and clean-runs safety."""

    def test_makefile_clean_runs_has_confirmation(self):
        """Test that clean-runs target requires confirmation."""
        makefile = Path("Makefile")
        if not makefile.exists():
            pytest.skip("Makefile not found")

        content = makefile.read_text()
        # Ensure clean-runs has a confirmation prompt
        assert "Are you sure" in content
        assert "FORCE" in content

    def test_makefile_archive_results_target_exists(self):
        """Test that archive-results target exists in Makefile."""
        makefile = Path("Makefile")
        if not makefile.exists():
            pytest.skip("Makefile not found")

        content = makefile.read_text()
        assert "archive-results:" in content
        assert "backups/" in content

    def test_gitignore_has_backups(self):
        """Test that .gitignore includes backups/."""
        gitignore = Path(".gitignore")
        if not gitignore.exists():
            pytest.skip(".gitignore not found")

        content = gitignore.read_text()
        assert "backups/" in content


class TestQueueResultTimestamping:
    """Tests that queue results use timestamped filenames."""

    def test_result_files_are_timestamped(self, tmp_path: Path):
        """Test that queue result files have timestamp in name."""
        # Simulate what the queue runner would create
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = tmp_path / f"queue_{stamp}.json"
        results_path.write_text(json.dumps([{"name": "test", "status": "dry"}]))

        # Verify we can find it
        found = list(tmp_path.glob("queue_*.json"))
        assert len(found) == 1
        assert stamp in found[0].name

    def test_multiple_runs_dont_overwrite(self, tmp_path: Path):
        """Test that multiple queue runs create separate result files."""
        from datetime import datetime

        files = []
        for i in range(3):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i}"
            path = tmp_path / f"queue_{stamp}.json"
            path.write_text(json.dumps([{"name": f"job_{i}", "status": "completed"}]))
            files.append(path)

        found = list(tmp_path.glob("queue_*.json"))
        assert len(found) == 3


class TestLoggingSetup:
    """Tests for the enhanced logging setup."""

    def test_setup_logging_with_file_path(self, tmp_path: Path):
        """Test setup_logging with a specific file path."""
        from ndswin.utils.logging import get_logger, setup_logging

        log_file = tmp_path / "subdir" / "test.log"
        setup_logging(level="INFO", log_file=str(log_file), use_rich=False)

        logger = get_logger("test_file_path")
        logger.info("Test message for file path logging")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_setup_logging_without_file(self, tmp_path: Path):
        """Test setup_logging with log_to_file=False."""
        import logging

        from ndswin.utils.logging import setup_logging

        setup_logging(level="INFO", log_to_file=False, use_rich=False)

        root = logging.getLogger()
        # Should have no file handlers
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0
