"""Unit tests for src.models.train."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.train import (
    EnsembleRFXGB,
    _build_rf,
    _build_xgboost,
    _proba_positive,
    build_model,
    run,
)


class TestProbaPositive:
    """Tests for _proba_positive()."""

    def test_binary_shape(self):
        p = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
        out = _proba_positive(p)
        assert out.shape == (3,)
        np.testing.assert_array_almost_equal(out, [0.8, 0.1, 0.5])

    def test_single_class_shape(self):
        p = np.array([[1.0], [0.0], [0.5]])
        out = _proba_positive(p)
        assert out.shape == (3,)
        np.testing.assert_array_almost_equal(out, [1.0, 0.0, 0.5])


class TestEnsembleRFXGB:
    """Tests for EnsembleRFXGB."""

    def test_fit_predict_proba_shape(self):
        X = np.random.randn(200, 5)
        y = (np.random.rand(200) > 0.7).astype(int)
        rf = _build_rf(scale_pos_weight=1.0, n_estimators=10)
        xgb = _build_xgboost(scale_pos_weight=1.0, n_estimators=10)
        ens = EnsembleRFXGB(rf, xgb)
        ens.fit(X, y)
        proba = ens.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(10))


class TestBuildModel:
    """Tests for build_model()."""

    def test_xgboost(self):
        model = build_model("xgboost", scale_pos_weight=2.0)
        assert model is not None
        X = np.random.randn(50, 5)
        y = (np.random.rand(50) > 0.8).astype(int)
        model.fit(X, y)
        p = model.predict_proba(X[:5])
        assert p.shape[0] == 5

    def test_rf(self):
        model = build_model("rf", scale_pos_weight=2.0)
        assert model is not None
        X = np.random.randn(50, 5)
        y = (np.random.rand(50) > 0.8).astype(int)
        model.fit(X, y)
        p = model.predict_proba(X[:5])
        assert p.shape[0] == 5

    def test_ensemble(self):
        model = build_model("ensemble", scale_pos_weight=2.0)
        assert model is not None
        X = np.random.randn(100, 5)
        y = (np.random.rand(100) > 0.85).astype(int)
        model.fit(X, y)
        p = model.predict_proba(X[:5])
        assert p.shape == (5, 2)

    def test_unknown_classifier_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            build_model("unknown", scale_pos_weight=1.0)


class TestRunSynthetic:
    """Integration tests for run() using synthetic data (no external files)."""

    def test_run_synthetic_xgboost(self):
        info = run(
            data_source="synthetic",
            classifier="xgboost",
            max_rows=300,
            W=12,
            H=6,
        )
        assert info["tag"] == "synthetic_xgboost"
        assert "model_path" in info
        assert "pred_path" in info
        assert info["n_train"] > 0
        assert info["n_test"] > 0

    def test_run_synthetic_rf(self):
        info = run(
            data_source="synthetic",
            classifier="rf",
            max_rows=300,
            W=12,
            H=6,
        )
        assert info["tag"] == "synthetic_rf"
        assert info["n_train"] > 0

    def test_run_synthetic_ensemble(self):
        info = run(
            data_source="synthetic",
            classifier="ensemble",
            max_rows=400,
            W=12,
            H=6,
        )
        assert info["tag"] == "synthetic_ensemble"
