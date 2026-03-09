"""Load the NASA C-MAPSS turbofan engine degradation dataset.

Dataset
-------
Source  : NASA Prognostics Data Repository / Kaggle palbha/cmapss-jet-engine-simulated-data
Files   : data_cmapss/train_FD001.txt, test_FD001.txt, RUL_FD001.txt
          (likewise for FD002 / FD003 / FD004)
Columns : unit_number, time_cycles, op_setting_1-3, sensor_1-21  (26 cols, space-separated)
Granularity: one row per operational cycle per engine

Label formulation
-----------------
This dataset has no per-row binary label out of the box. We derive one for the
predictive alerting task:

    label[t] = 1  if  RUL[t] <= H

where H is the prediction horizon in cycles. This answers the question:
"Will this engine fail within the next H cycles?"

RUL for training engines: RUL[i, t] = max_cycle_i - t
RUL for test engines    : provided by RUL_FDxxx.txt for the *last* cycle of each
                          engine; for earlier cycles we add back the offset.

Design choices
--------------
- FD001 is the simplest sub-dataset: 1 operating condition, 1 fault mode, 100 engines.
- W = 30 cycles of history (look-back window).
- H = 30 cycles horizon (label=1 if engine fails within 30 cycles).
- Features are computed PER ENGINE (group_col='unit_number') to avoid mixing
  sensor degradation profiles across different engines — this is critical to
  prevent data leakage and is the main difference from the IBM / SWaT pipelines.
- The official train/test file split is used (no percentage split).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data_cmapss")

# Window parameters matching the cycle scale of C-MAPSS
W_CMAPSS: int = 30   # 30 cycles look-back
H_CMAPSS: int = 30   # predict failure within 30 cycles

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
FEATURE_COLS = OP_COLS + SENSOR_COLS
COL_NAMES = ["unit_number", "time_cycles"] + OP_COLS + SENSOR_COLS


def _read_txt(path: Path) -> pd.DataFrame:
    """Read a space-separated C-MAPSS text file into a named DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # The files sometimes have trailing empty columns — drop them
    df = df.dropna(axis=1, how="all")
    df.columns = COL_NAMES[: df.shape[1]]
    return df


def _compute_rul_train(df: pd.DataFrame) -> pd.Series:
    """Compute Remaining Useful Life for each row of the training set."""
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    return max_cycle - df["time_cycles"]


def _compute_rul_test(df: pd.DataFrame, rul_file: Path) -> pd.Series:
    """Compute RUL for each row of the test set using the ground-truth file."""
    # RUL_FDxxx.txt contains one RUL value per test engine (at the last cycle)
    rul_at_end = pd.read_csv(rul_file, header=None, names=["rul_end"]).squeeze()

    # Map engine id → rul at last observed cycle
    last_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    rul_end_series = df["unit_number"].map(
        dict(enumerate(rul_at_end.values, start=1))
    )
    # RUL[t] = rul_end + (last_cycle - t)
    return rul_end_series + (last_cycle - df["time_cycles"])


def load_cmapss(
    subset: str = "FD001",
    H: int = H_CMAPSS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load C-MAPSS train and test DataFrames with binary RUL-derived labels.

    Parameters
    ----------
    subset:
        Which sub-dataset to load: 'FD001', 'FD002', 'FD003', or 'FD004'.
    H:
        Prediction horizon in cycles. Label = 1 if RUL <= H.

    Returns
    -------
    train_df, test_df — each with columns:
        unit_number, time_cycles, op_setting_1-3, sensor_1-21, rul, label
    """
    train_path = DATA_DIR / f"train_{subset}.txt"
    test_path = DATA_DIR / f"test_{subset}.txt"
    rul_path = DATA_DIR / f"RUL_{subset}.txt"

    train_df = _read_txt(train_path)
    test_df = _read_txt(test_path)

    train_df["rul"] = _compute_rul_train(train_df)
    test_df["rul"] = _compute_rul_test(test_df, rul_path)

    train_df["label"] = (train_df["rul"] <= H).astype(int)
    test_df["label"] = (test_df["rul"] <= H).astype(int)

    # Sort chronologically within each engine
    train_df = train_df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)
    test_df = test_df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

    return train_df, test_df
