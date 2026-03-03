"""Training pipeline for sliding-window incident prediction.

Trains binary classifiers on rolling-statistics features (XGBoost, RF, ensemble)
or raw sequences (LSTM, TCN), predicting whether an incident will start within
the next H steps.

Supported data sources
----------------------
ibm       — IBM Cloud Console telemetry (39k rows, 5-min intervals)
synthetic — Controlled synthetic telemetry with injected incident ramp-ups
swat      — Secure Water Treatment dataset (1.44M rows at 1-sec, 51 sensors)
cmapss    — NASA C-MAPSS turbofan engine degradation (FD001, per-engine RUL)

Supported classifiers
--------------------
xgboost   — Gradient boosting (tabular features)
rf        — Random Forest (tabular features)
ensemble  — Average of RF + XGBoost probabilities
lstm      — LSTM on raw W-step sequences (requires PyTorch)
tcn       — Temporal Convolutional Network on raw sequences (requires PyTorch)

Usage
-----
    python -m src.models.train [--data ibm|synthetic|swat|cmapss|all]
                               [--classifier xgboost|rf|ensemble|lstm|tcn|both]
                               [--max-rows N] [--stride N]
                               [--W 24] [--H 6]

Outputs (written to outputs/)
------
    model_<data>_<classifier>.pkl
    predictions_<data>_<classifier>.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.load_ibm import load_anomaly_windows, load_telemetry
from src.data.load_swat import W_SWAT, H_SWAT, get_sensor_cols, load_swat
from src.data.load_cmapss import W_CMAPSS, H_CMAPSS, FEATURE_COLS as CMAPSS_FEATURES, load_cmapss
from src.data.synthetic import generate_synthetic
from src.features.windows import (
    H_DEFAULT,
    W_DEFAULT,
    make_features,
    make_labels,
    make_raw_windows,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def _build_xgboost(scale_pos_weight: float, n_estimators: int) -> object:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _build_rf(scale_pos_weight: float, n_estimators: int) -> object:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


class EnsembleRFXGB:
    """Ensemble of RandomForest + XGBoost: average of probability outputs."""

    def __init__(self, rf, xgb):
        self.rf = rf
        self.xgb = xgb

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        return self

    def predict_proba(self, X):
        p_rf = self.rf.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        proba = (p_rf + p_xgb) / 2
        return np.column_stack([1 - proba, proba])


def build_model(
    classifier: str,
    scale_pos_weight: float,
    n_estimators: int = 300,
    n_train: int = 0,
    n_features: int = 0,
    seq_len: int = 0,
) -> object:
    """Instantiate a classifier with class-imbalance handling.

    For RandomForest on large datasets (> 50k training samples), n_estimators
    is capped at 100 to keep wall-clock time reasonable.
    """
    if classifier in ("rf", "ensemble") and n_train > 50_000:
        n_estimators = min(n_estimators, 100)

    if classifier == "xgboost":
        return _build_xgboost(scale_pos_weight, n_estimators)
    elif classifier == "rf":
        return _build_rf(scale_pos_weight, n_estimators)
    elif classifier == "ensemble":
        return EnsembleRFXGB(
            _build_rf(scale_pos_weight, n_estimators),
            _build_xgboost(scale_pos_weight, n_estimators),
        )
    elif classifier == "lstm":
        from src.models.sequential import SequentialWrapper

        return SequentialWrapper(
            "lstm",
            n_features=n_features,
            seq_len=seq_len,
            scale_pos_weight=scale_pos_weight,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
        )
    elif classifier == "tcn":
        from src.models.sequential import SequentialWrapper

        return SequentialWrapper(
            "tcn",
            n_features=n_features,
            seq_len=seq_len,
            scale_pos_weight=scale_pos_weight,
            n_channels=32,
            kernel_size=3,
            n_levels=4,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier!r}")


def _prepare_ibm_or_synthetic(
    data_source: str,
    max_rows: int,
    W: int,
    H: int,
    label_mode: str,
):
    """Load data, build features and labels for IBM / Synthetic sources."""
    if data_source == "ibm":
        df = load_telemetry(max_rows=max_rows)
        anomaly_df = load_anomaly_windows()
    else:
        n = max_rows if max_rows else 10_000
        df, anomaly_df = generate_synthetic(n_steps=n)

    y = make_labels(df, anomaly_df, H=H, mode=label_mode)
    X = make_features(df, W=W)
    X_seq = make_raw_windows(df, W=W)

    split = int(len(X) * 0.8)
    X_train = X.iloc[:split].values
    X_test = X.iloc[split:].values
    X_train_seq = X_seq[:split]
    X_test_seq = X_seq[split:]
    y_train = y[:split]
    y_test = y[split:]
    timestamps = df["interval_start"].values[split:]
    feature_names = list(X.columns)
    n_raw_features = X_seq.shape[2]

    return X_train, X_test, y_train, y_test, timestamps, feature_names, X_train_seq, X_test_seq, n_raw_features


def _prepare_swat(max_rows: int, stride: int, W: int, H: int):
    """Load SWaT, build features and labels."""
    df = load_swat(max_rows=max_rows, stride=stride)
    sensor_cols = get_sensor_cols(df)

    # Label: attack active within next H seconds (already in df["label"])
    # For 'active' semantics we shift the label back H steps so that the
    # model at time t sees: "will an attack be active in [t, t+H]?"
    label_shifted = (
        df["label"]
        .rolling(H, min_periods=1)
        .max()
        .shift(-H + 1)
        .fillna(0)
        .astype(int)
        .values
    )

    X = make_features(df, W=W, feature_cols=sensor_cols)
    X_seq = make_raw_windows(df, W=W, feature_cols=sensor_cols)

    split = int(len(X) * 0.8)
    X_train = X.iloc[:split].values
    X_test = X.iloc[split:].values
    X_train_seq = X_seq[:split]
    X_test_seq = X_seq[split:]
    y_train = label_shifted[:split]
    y_test = label_shifted[split:]
    timestamps = df["Timestamp"].values[split:]
    feature_names = list(X.columns)
    n_raw_features = X_seq.shape[2]

    return X_train, X_test, y_train, y_test, timestamps, feature_names, X_train_seq, X_test_seq, n_raw_features


def _prepare_cmapss(W: int, H: int):
    """Load C-MAPSS FD001 train/test with per-engine rolling features."""
    train_df, test_df = load_cmapss(subset="FD001", H=H)

    X_train = make_features(
        train_df, W=W, group_col="unit_number", feature_cols=CMAPSS_FEATURES
    ).values
    X_test = make_features(
        test_df, W=W, group_col="unit_number", feature_cols=CMAPSS_FEATURES
    ).values
    X_train_seq = make_raw_windows(
        train_df, W=W, group_col="unit_number", feature_cols=CMAPSS_FEATURES
    )
    X_test_seq = make_raw_windows(
        test_df, W=W, group_col="unit_number", feature_cols=CMAPSS_FEATURES
    )

    y_train = train_df["label"].values
    y_test = test_df["label"].values
    timestamps = test_df["time_cycles"].values
    feature_names = [
        f"{c}_{stat}{W}"
        for c in CMAPSS_FEATURES
        for stat in ("mean", "std", "max", "last", f"delta{W}", f"delta{W//2}")
    ]
    n_raw_features = X_train_seq.shape[2]

    return X_train, X_test, y_train, y_test, timestamps, feature_names, X_train_seq, X_test_seq, n_raw_features


def run(
    data_source: str = "ibm",
    classifier: str = "xgboost",
    max_rows: int = 0,
    stride: int = 1,
    W: int | None = None,
    H: int | None = None,
    label_mode: str = "active",
) -> dict:
    """Full train -> predict -> save pipeline.

    Returns
    -------
    dict with keys: tag, model_path, pred_path, n_train, n_test, pos_rate_train
    """
    # Use dataset-appropriate defaults when W/H are not explicitly set
    if W is None:
        W = {
            "swat": W_SWAT,
            "cmapss": W_CMAPSS,
        }.get(data_source, W_DEFAULT)
    if H is None:
        H = {
            "swat": H_SWAT,
            "cmapss": H_CMAPSS,
        }.get(data_source, H_DEFAULT)

    print(f"\n[train] data={data_source}  classifier={classifier}  "
          f"max_rows={max_rows or 'all'}  W={W}  H={H}  label={label_mode}"
          + (f"  stride={stride}" if data_source == "swat" else ""), flush=True)

    # ------------------------------------------------------------------ data
    if data_source in ("ibm", "synthetic"):
        out = _prepare_ibm_or_synthetic(data_source, max_rows, W, H, label_mode)
    elif data_source == "swat":
        out = _prepare_swat(max_rows, stride, W, H)
    elif data_source == "cmapss":
        out = _prepare_cmapss(W, H)
    else:
        raise ValueError(f"Unknown data source: {data_source!r}")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        timestamps,
        feature_names,
        X_train_seq,
        X_test_seq,
        n_raw_features,
    ) = out

    print(f"[train] total  train: {len(X_train):,}  test: {len(X_test):,}")
    print(f"[train] positive (train): {y_train.sum():,}  ({100 * y_train.mean():.2f}%)")
    print(f"[train] positive (test) : {y_test.sum():,}  ({100 * y_test.mean():.2f}%)")

    # ----------------------------------------------- class-imbalance handling
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    scale_pos_weight = n_neg / n_pos
    print(f"[train] scale_pos_weight: {scale_pos_weight:.2f}")

    # ---------------------------- choose input format and train classifier
    is_sequential = classifier in ("lstm", "tcn")
    if is_sequential:
        X_tr, X_te = X_train_seq, X_test_seq
        model = build_model(
            classifier,
            scale_pos_weight,
            n_train=len(X_train),
            n_features=n_raw_features,
            seq_len=W,
        )
    else:
        X_tr, X_te = X_train, X_test
        model = build_model(classifier, scale_pos_weight, n_train=len(X_train))

    model.fit(X_tr, y_train)

    # ----------------------------------------------------------- save outputs
    if is_sequential:
        proba = model.predict_proba(X_te)
    else:
        proba = model.predict_proba(X_te)[:, 1]

    tag = f"{data_source}_{classifier}"
    model_path = OUTPUT_DIR / f"model_{tag}.pkl"
    pred_path = OUTPUT_DIR / f"predictions_{tag}.npz"

    with open(model_path, "wb") as fh:
        pickle.dump(
            {"model": model, "feature_names": feature_names, "W": W, "H": H}, fh
        )

    np.savez(pred_path, proba=proba, y_true=y_test, timestamps=timestamps)

    print(f"[train] model saved       -> {model_path}")
    print(f"[train] predictions saved -> {pred_path}")

    return {
        "tag": tag,
        "model_path": str(model_path),
        "pred_path": str(pred_path),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "pos_rate_train": float(y_train.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train incident prediction model")
    parser.add_argument(
        "--data",
        choices=["ibm", "synthetic", "swat", "cmapss", "both", "all"],
        default="both",
        help="Data source (default: both = ibm + synthetic)",
    )
    parser.add_argument(
        "--classifier",
        choices=["xgboost", "rf", "ensemble", "lstm", "tcn", "both", "all"],
        default="both",
        help="Classifier: xgboost|rf|ensemble|lstm|tcn|both(=xgboost+rf)|all(=all 5)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Row cap for IBM/SWaT datasets (0 = all)",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Downsample SWaT by keeping every Nth row (default 1 = full 1-Hz resolution)",
    )
    parser.add_argument("--W", type=int, default=None, help="Look-back window steps")
    parser.add_argument("--H", type=int, default=None, help="Prediction horizon steps")
    parser.add_argument(
        "--label-mode", choices=["active", "onset"], default="active",
        help="Label strategy for IBM/Synthetic",
    )
    args = parser.parse_args()

    if args.data == "both":
        data_sources = ["ibm", "synthetic"]
    elif args.data == "all":
        data_sources = ["ibm", "synthetic", "swat", "cmapss"]
    else:
        data_sources = [args.data]

    if args.classifier == "both":
        classifiers = ["xgboost", "rf"]
    elif args.classifier == "all":
        classifiers = ["xgboost", "rf", "ensemble", "lstm", "tcn"]
    else:
        classifiers = [args.classifier]

    for ds in data_sources:
        for clf in classifiers:
            run(
                data_source=ds,
                classifier=clf,
                max_rows=args.max_rows,
                stride=args.stride,
                W=args.W,
                H=args.H,
                label_mode=args.label_mode,
            )


if __name__ == "__main__":
    main()
