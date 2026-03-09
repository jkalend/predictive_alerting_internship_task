"""Unit tests for src.features.windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.windows import (
    FEATURE_COLS,
    H_DEFAULT,
    W_DEFAULT,  # noqa: F401
    make_features,
    make_labels,
    make_mask_normal_at_t,
    make_raw_windows,
)


class TestMakeLabels:
    """Tests for make_labels()."""

    def test_shape_matches_df(self, sample_telemetry_df, sample_anomaly_df):
        labels = make_labels(sample_telemetry_df, sample_anomaly_df, H=H_DEFAULT)
        assert labels.shape == (len(sample_telemetry_df),)
        assert labels.dtype in (np.int8, np.int32, np.int64, int)

    def test_binary_labels(self, sample_telemetry_df, sample_anomaly_df):
        labels = make_labels(sample_telemetry_df, sample_anomaly_df, H=H_DEFAULT)
        assert set(np.unique(labels)) <= {0, 1}

    def test_onset_vs_active_mode(self, sample_telemetry_df, sample_anomaly_df):
        labels_active = make_labels(sample_telemetry_df, sample_anomaly_df, H=6, mode="active")
        labels_onset = make_labels(sample_telemetry_df, sample_anomaly_df, H=6, mode="onset")
        # Onset typically has fewer positives (only at anomaly start)
        assert labels_onset.sum() <= labels_active.sum()

    def test_empty_anomaly_all_zeros(self, sample_telemetry_df):
        empty_anomaly = pd.DataFrame(columns=["anomaly_start", "anomaly_end"])
        labels = make_labels(sample_telemetry_df, empty_anomaly, H=H_DEFAULT)
        assert np.all(labels == 0)


class TestMakeMaskNormalAtT:
    """Tests for make_mask_normal_at_t()."""

    def test_shape_matches_df(self, sample_telemetry_df, sample_anomaly_df):
        mask = make_mask_normal_at_t(sample_telemetry_df, sample_anomaly_df)
        assert mask.shape == (len(sample_telemetry_df),)
        assert mask.dtype == bool

    def test_excludes_anomaly_interior(self, sample_telemetry_df, sample_anomaly_df):
        mask = make_mask_normal_at_t(sample_telemetry_df, sample_anomaly_df)
        # Some rows should be excluded (inside anomaly windows)
        assert mask.sum() < len(mask)

    def test_empty_anomaly_all_true(self, sample_telemetry_df):
        empty_anomaly = pd.DataFrame(columns=["anomaly_start", "anomaly_end"])
        mask = make_mask_normal_at_t(sample_telemetry_df, empty_anomaly)
        assert np.all(mask)


class TestMakeFeatures:
    """Tests for make_features()."""

    def test_shape_and_columns(self, sample_telemetry_df):
        X = make_features(sample_telemetry_df, W=6)
        assert len(X) == len(sample_telemetry_df)
        # 6 stats per col: mean, std, max, last, deltaW, deltaW2
        avail = [c for c in FEATURE_COLS if c in sample_telemetry_df.columns]
        assert len(X.columns) == 6 * len(avail)
        for col in avail:
            assert f"{col}_mean6" in X.columns
            assert f"{col}_std6" in X.columns
            assert f"{col}_max6" in X.columns
            assert f"{col}_last" in X.columns

    def test_no_nan_in_output(self, sample_telemetry_df):
        X = make_features(sample_telemetry_df, W=6)
        assert not X.isna().any().any()

    def test_custom_feature_cols(self, sample_telemetry_df):
        cols = ["rate_5xx", "total_count"]
        X = make_features(sample_telemetry_df, W=6, feature_cols=cols)
        assert len(X.columns) == 6 * 2  # 2 cols × 6 stats

    def test_group_col_smoke(self):
        """Smoke test for per-group rolling (C-MAPSS style)."""
        n = 50
        df = pd.DataFrame(
            {
                "unit_number": np.repeat([1, 2], n // 2),
                "time_cycles": np.tile(np.arange(n // 2), 2),
                "sensor_1": np.random.randn(n),
            }
        )
        X = make_features(df, W=5, group_col="unit_number", feature_cols=["sensor_1"])
        assert len(X) == n
        assert "sensor_1_mean5" in X.columns


class TestMakeRawWindows:
    """Tests for make_raw_windows()."""

    def test_shape(self, sample_telemetry_df):
        W = 10
        X = make_raw_windows(sample_telemetry_df, W=W)
        n_cols = len([c for c in FEATURE_COLS if c in sample_telemetry_df.columns])
        assert X.shape == (len(sample_telemetry_df), W, n_cols)
        assert X.dtype == np.float32

    def test_left_padding(self, sample_telemetry_df):
        W = 10
        X = make_raw_windows(sample_telemetry_df, W=W)
        # First row should have zeros in left part (no history)
        assert X[0, 0, 0] == 0.0
        # Last row should have recent data
        assert X[-1, -1, 0] != 0.0 or np.any(X[-1] != 0)

    def test_group_col_smoke(self):
        """Smoke test for per-group windows."""
        n = 30
        df = pd.DataFrame(
            {
                "unit_number": np.repeat([1, 2], n // 2),
                "time_cycles": np.tile(np.arange(n // 2), 2),
                "sensor_1": np.random.randn(n).astype(float),
            }
        )
        X = make_raw_windows(df, W=5, group_col="unit_number", feature_cols=["sensor_1"])
        assert X.shape == (n, 5, 1)
