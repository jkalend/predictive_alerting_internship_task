"""Unit tests for src.data.synthetic."""

from __future__ import annotations

import pandas as pd

from src.data.synthetic import generate_synthetic


class TestGenerateSynthetic:
    """Tests for generate_synthetic()."""

    def test_returns_two_dataframes(self):
        df, anomaly_df = generate_synthetic(n_steps=200, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(anomaly_df, pd.DataFrame)

    def test_telemetry_shape_and_columns(self):
        df, _ = generate_synthetic(n_steps=500, seed=42)
        assert len(df) == 500
        assert "interval_start" in df.columns
        assert "total_count" in df.columns
        assert "rate_5xx" in df.columns
        # Expect dc1..dc7 columns
        for i in range(1, 8):
            assert f"dc{i}_5xx" in df.columns
            assert f"dc{i}_total" in df.columns
            assert f"dc{i}_latency" in df.columns

    def test_anomaly_columns(self):
        _, anomaly_df = generate_synthetic(n_steps=500, seed=42)
        assert "anomaly_start" in anomaly_df.columns
        assert "anomaly_end" in anomaly_df.columns
        assert len(anomaly_df) >= 1

    def test_reproducibility(self):
        df1, a1 = generate_synthetic(n_steps=300, seed=123)
        df2, a2 = generate_synthetic(n_steps=300, seed=123)
        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_frame_equal(a1, a2)

    def test_different_seeds_differ(self):
        df1, _ = generate_synthetic(n_steps=300, seed=1)
        df2, _ = generate_synthetic(n_steps=300, seed=2)
        assert not df1["rate_5xx"].equals(df2["rate_5xx"])

    def test_n_datacenters(self):
        df, _ = generate_synthetic(n_steps=100, n_datacenters=3, seed=42)
        for i in range(1, 4):
            assert f"dc{i}_5xx" in df.columns
        assert "dc4_5xx" not in df.columns
