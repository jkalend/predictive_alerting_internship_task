"""Shared pytest fixtures for predictive alerting tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure outputs/ exists for evaluate tests that write PR curves
ROOT = Path(__file__).resolve().parents[1]
(ROOT / "outputs").mkdir(exist_ok=True)


@pytest.fixture
def sample_telemetry_df():
    """Minimal telemetry DataFrame with interval_start and feature columns."""
    n = 100
    start = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame(
        {
            "interval_start": timestamps,
            "rate_5xx": np.random.rand(n) * 0.1,
            "rate_4xx": np.random.rand(n) * 0.2,
            "total_count": np.random.randint(100, 1000, n),
            "mean_latency_avg": np.random.rand(n) * 100,
        }
    )


@pytest.fixture
def sample_anomaly_df():
    """Anomaly windows overlapping with sample_telemetry_df (100 steps)."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.DataFrame(
        [
            {"anomaly_start": start + pd.Timedelta(minutes=5 * 20), "anomaly_end": start + pd.Timedelta(minutes=5 * 35)},
            {"anomaly_start": start + pd.Timedelta(minutes=5 * 60), "anomaly_end": start + pd.Timedelta(minutes=5 * 75)},
        ]
    )


@pytest.fixture
def synthetic_df_and_anomaly():
    """Generate synthetic data for training tests (no external files)."""
    from src.data.synthetic import generate_synthetic
    return generate_synthetic(n_steps=500, seed=42)
