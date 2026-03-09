"""Unit tests for src.models.evaluate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.evaluate import evaluate_file


def _make_pred_npz(tmp_path: Path, n: int = 100, pos_frac: float = 0.2, optimal_threshold: float = 0.5):
    """Create a minimal predictions .npz for testing."""
    rng = np.random.default_rng(42)
    y_true = (rng.random(n) < pos_frac).astype(np.int32)
    proba = rng.random(n).astype(np.float32)
    # Slightly bias proba toward y_true for non-trivial metrics
    proba[y_true == 1] += 0.2
    proba = np.clip(proba, 0, 1)
    timestamps = np.arange(n, dtype=np.float64)
    path = tmp_path / "predictions_test_model.npz"
    np.savez(
        path,
        proba=proba,
        y_true=y_true,
        timestamps=timestamps,
        proba_val=proba[: n // 2],
        y_val=y_true[: n // 2],
        optimal_threshold=np.float64(optimal_threshold),
    )
    return path


class TestEvaluateFile:
    """Tests for evaluate_file()."""

    def test_returns_dataframe(self, tmp_path):
        pred_path = _make_pred_npz(tmp_path)
        df = evaluate_file(pred_path)
        assert isinstance(df, pd.DataFrame)
        assert "threshold" in df.columns
        assert "recall" in df.columns
        assert "precision" in df.columns
        assert "f1" in df.columns

    def test_dataframe_has_threshold_sweep(self, tmp_path):
        pred_path = _make_pred_npz(tmp_path)
        df = evaluate_file(pred_path)
        assert len(df) >= 1
        assert 0.5 in df["threshold"].values

    def test_handles_all_negative_labels(self, tmp_path):
        """When y_true is all zeros, metrics should not crash."""
        rng = np.random.default_rng(43)
        n = 50
        y_true = np.zeros(n, dtype=np.int32)
        proba = rng.random(n).astype(np.float32)
        path = tmp_path / "predictions_all_neg.npz"
        np.savez(
            path,
            proba=proba,
            y_true=y_true,
            timestamps=np.arange(n),
            proba_val=proba[:25],
            y_val=y_true[:25],
            optimal_threshold=np.float64(0.5),
        )
        df = evaluate_file(path)
        assert isinstance(df, pd.DataFrame)

    def test_legacy_npz_without_optimal_threshold(self, tmp_path):
        """Legacy .npz without optimal_threshold should still run (with warning)."""
        rng = np.random.default_rng(44)
        n = 80
        y_true = (rng.random(n) < 0.3).astype(np.int32)
        proba = rng.random(n).astype(np.float32)
        path = tmp_path / "predictions_legacy.npz"
        np.savez(
            path,
            proba=proba,
            y_true=y_true,
            timestamps=np.arange(n),
        )
        df = evaluate_file(path)
        assert isinstance(df, pd.DataFrame)
