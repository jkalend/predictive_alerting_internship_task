"""Sliding-window feature engineering and label generation.

Formulation
-----------
For each time step t:
  - Features  : rolling statistics (mean, std, max) of each metric column
                over the last W steps — a compact, fixed-size feature vector
                that represents the recent behaviour of the system.
  - Target    : 1  if any anomaly *start* falls in the half-open interval
                    (t, t + H * 5 min]
                0  otherwise

This converts the time-series anomaly prediction problem into standard
supervised binary classification — compatible with any sklearn estimator.

Design choices
--------------
W = 24 steps  →  2 hours of history at the 5-minute IBM granularity.
H =  6 steps  →  30-minute prediction horizon (alert before the incident).

Rolling stats (mean, std, max) are preferred over raw window flattening
because they are:
  * dimensionality-efficient (3 × C features vs W × C for raw flattening)
  * invariant to short-term jitter, capturing the trend that precedes an incident
  * fast to compute and easy to interpret
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default look-back and horizon (in 5-minute steps)
W_DEFAULT: int = 24  # 2 hours
H_DEFAULT: int = 6   # 30 minutes

# Columns used for feature extraction (present in both IBM and synthetic data).
# Any column missing from a particular DataFrame is silently skipped.
FEATURE_COLS: list[str] = [
    "rate_5xx",
    "rate_4xx",
    "rate_2xx",
    "mean_latency_avg",
    "max_latency_max",
    "median_latency",
    "total_count",
    "count_5xx",
    "dc1_5xx", "dc1_total", "dc1_latency",
    "dc2_5xx", "dc2_total", "dc2_latency",
    "dc3_5xx", "dc3_total", "dc3_latency",
    "dc4_5xx", "dc4_total", "dc4_latency",
    "dc5_5xx", "dc5_total", "dc5_latency",
    "dc6_5xx", "dc6_total", "dc6_latency",
    "dc7_5xx", "dc7_total", "dc7_latency",
]


def make_labels(
    df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    H: int = H_DEFAULT,
    mode: str = "active",
) -> np.ndarray:
    """Create binary incident labels for each row of ``df``.

    Parameters
    ----------
    df:
        Telemetry DataFrame with an ``interval_start`` column (UTC-aware).
    anomaly_df:
        DataFrame with ``anomaly_start`` and ``anomaly_end`` columns.
    H:
        Prediction horizon in 5-minute steps.
    mode:
        ``"onset"``  — label=1 iff an anomaly *starts* in (t, t+H*5min].
                       Strict early-warning; few positives per anomaly window.
        ``"active"`` — label=1 iff any anomaly window is *active* (ongoing or
                       starting) within [t, t+H*5min].  Equivalent to asking
                       "will an anomaly be happening in the next H steps?"
                       Generates more positives and is more robust for datasets
                       with few labeled events.

    Returns
    -------
    np.ndarray of shape (len(df),) with dtype int (0 or 1).
    """
    horizon = np.timedelta64(H * 5, "m")
    ts = df["interval_start"].values.astype("datetime64[ns]")
    labels = np.zeros(len(df), dtype=np.int8)

    for _, row in anomaly_df.iterrows():
        a_start = np.datetime64(row["anomaly_start"].to_datetime64())
        a_end = np.datetime64(row["anomaly_end"].to_datetime64())

        if mode == "onset":
            # Anomaly must START within (t, t+horizon]
            mask = (ts < a_start) & (a_start <= ts + horizon)
        else:
            # Anomaly window [a_start, a_end] must OVERLAP [t, t+horizon]
            # Overlap condition: a_start <= t+horizon  AND  a_end >= t
            mask = (a_start <= ts + horizon) & (a_end >= ts)

        labels[mask] = 1

    return labels.astype(int)


def make_mask_normal_at_t(
    df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
) -> np.ndarray:
    """Boolean mask: True for rows where no anomaly is active at time t.

    Used for onset-only evaluation: exclude rows where an incident is already
    ongoing, forcing the model to predict onset rather than continuation.
    """
    ts = df["interval_start"].values.astype("datetime64[ns]")
    mask = np.ones(len(df), dtype=bool)
    for _, row in anomaly_df.iterrows():
        a_start = np.datetime64(row["anomaly_start"].to_datetime64())
        a_end = np.datetime64(row["anomaly_end"].to_datetime64())
        # Exclude rows where t is inside [a_start, a_end]
        inside = (ts >= a_start) & (ts <= a_end)
        mask &= ~inside
    return mask


def make_features(
    df: pd.DataFrame,
    W: int = W_DEFAULT,
    group_col: str | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a rolling-statistics + delta feature matrix from ``df``.

    For each available feature column computes:

    Rolling statistics over the last W steps:
      - mean, std, max

    Trend / rate-of-change features (captures rising trends before an incident):
      - ``_deltaW``      — change since W steps ago
      - ``_deltaW2``     — change since W//2 steps ago (shorter-term acceleration)

    Raw current value:
      - ``_last``

    Parameters
    ----------
    df:
        Telemetry DataFrame sorted chronologically (or per-group chronologically).
    W:
        Look-back window size in steps.
    group_col:
        If provided, rolling statistics are computed *per group* (e.g. per engine
        for C-MAPSS) so that the window never crosses group boundaries.
        The column must exist in ``df`` and the DataFrame must already be sorted
        by (group_col, time).
    feature_cols:
        Explicit list of column names to use as features.  If None, falls back to
        ``FEATURE_COLS`` intersected with ``df.columns``.  Pass an explicit list
        for datasets with different naming conventions (SWaT, C-MAPSS).

    Returns
    -------
    pd.DataFrame of shape (len(df), 6 × len(available_cols)).
    """
    if feature_cols is not None:
        avail = [c for c in feature_cols if c in df.columns]
    else:
        avail = [c for c in FEATURE_COLS if c in df.columns]

    out: dict[str, pd.Series] = {}

    for col in avail:
        s = df[col].fillna(0.0)

        if group_col is not None:
            # Per-group rolling — never mixes data across engines / entities
            grp = s.groupby(df[group_col], group_keys=False)
            r = grp.rolling(W, min_periods=1)
            mean_s = r.mean().reset_index(level=0, drop=True).sort_index()
            std_s = r.std().fillna(0.0).reset_index(level=0, drop=True).sort_index()
            max_s = r.max().reset_index(level=0, drop=True).sort_index()
            # Per-group shifts for delta features
            shifted_W = grp.apply(lambda x: x.shift(W)).reset_index(level=0, drop=True).sort_index()
            shifted_W2 = grp.apply(lambda x: x.shift(W // 2)).reset_index(level=0, drop=True).sort_index()
        else:
            r = s.rolling(W, min_periods=1)
            mean_s = r.mean()
            std_s = r.std().fillna(0.0)
            max_s = r.max()
            shifted_W = s.shift(W)
            shifted_W2 = s.shift(W // 2)

        out[f"{col}_mean{W}"] = mean_s
        out[f"{col}_std{W}"] = std_s
        out[f"{col}_max{W}"] = max_s
        out[f"{col}_last"] = s
        out[f"{col}_delta{W}"] = s - shifted_W.fillna(s)
        out[f"{col}_delta{W // 2}"] = s - shifted_W2.fillna(s)

    return pd.DataFrame(out, index=df.index)


def make_raw_windows(
    df: pd.DataFrame,
    W: int = W_DEFAULT,
    group_col: str | None = None,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Build raw sliding windows for LSTM/TCN (sequence input).

    For each row t, returns the last W steps of raw values for each feature column.
    Shape: (n_samples, W, n_cols). Earlier rows are left-padded with zeros when
    fewer than W steps of history exist.

    Parameters
    ----------
    df, W, group_col, feature_cols
        Same semantics as make_features().

    Returns
    -------
    np.ndarray of shape (len(df), W, n_cols) with dtype float32.
    """
    if feature_cols is not None:
        avail = [c for c in feature_cols if c in df.columns]
    else:
        avail = [c for c in FEATURE_COLS if c in df.columns]

    n_cols = len(avail)
    n_rows = len(df)
    out = np.zeros((n_rows, W, n_cols), dtype=np.float32)

    raw = df[avail].fillna(0.0).values.astype(np.float32)

    if group_col is not None:
        for _, grp in df.groupby(group_col):
            idx = grp.index
            positions = df.index.get_indexer(idx)
            vals = grp[avail].fillna(0.0).values.astype(np.float32)
            for i, pos in enumerate(positions):
                start = max(0, i - W + 1)
                seg_len = i - start + 1
                out[pos, W - seg_len :, :] = vals[start : i + 1, :]
    else:
        for i in range(n_rows):
            start = max(0, i - W + 1)
            seg_len = i - start + 1
            out[i, W - seg_len :, :] = raw[start : i + 1, :]

    return out
