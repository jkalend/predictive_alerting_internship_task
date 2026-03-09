"""Integration tests for the full pipeline (run_pipeline.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestRunPipeline:
    """Smoke tests for scripts/run_pipeline.py."""

    def test_help(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_pipeline.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "data" in result.stdout
        assert "classifier" in result.stdout

    def test_single_combination_synthetic_xgboost(self):
        """Run one data+classifier combo (synthetic + xgboost) to verify pipeline."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_pipeline.py",
                "--data",
                "synthetic",
                "--classifier",
                "xgboost",
                "--max-rows",
                "400",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        assert "synthetic_xgboost" in result.stdout or "EVALUATION" in result.stdout
