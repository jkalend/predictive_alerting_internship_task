"""Load the SWaT (Secure Water Treatment) dataset.

Dataset
-------
Source  : Kaggle — vishala28/swat-dataset-secure-water-treatment-system
          (original: iTrust Labs, SUTD, Dec 2015)
Files   : data_swat/merged.csv  — 1,441,719 rows × 53 columns
          (normal.csv + attack.csv pre-merged; same schema)
Columns : Timestamp, 51 sensors (FIT/LIT/AIT/P/MV/DPIT/UV/PIT), Normal/Attack
Granularity: 1-second intervals
Label   : 'Normal' or 'Attack'

Design choices
--------------
W = 60 steps  → 1 minute of history at 1-second resolution
H = 60 steps  → predict whether an attack will be active within the next minute

stride : downsample every N rows (default 1 = full resolution)
         Use stride=5 to reduce to 5-second resolution (~288k rows) if memory
         is tight on a 32 GB system.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data_swat")

# Default window parameters for SWaT (1-second granularity)
W_SWAT: int = 60   # 1-minute look-back
H_SWAT: int = 60   # 1-minute prediction horizon

LABEL_COL = "Normal/Attack"
TIMESTAMP_COL = "Timestamp"

# Sensors to use as features (all 51 process variables)
EXCLUDE_COLS = {TIMESTAMP_COL, LABEL_COL}


def load_swat(
    max_rows: int = 0,
    stride: int = 1,
    use_merged: bool = True,
) -> pd.DataFrame:
    """Load SWaT telemetry and labels into a tidy DataFrame.

    Parameters
    ----------
    max_rows:
        Cap on total rows after striding (0 = no cap).
    stride:
        Keep every Nth row to reduce temporal resolution and memory usage.
        stride=1 → 1-second resolution (~1.44 M rows, ~610 MB)
        stride=5 → 5-second resolution (~288 k rows, ~120 MB)
    use_merged:
        If True (default), read merged.csv (normal + attack combined).
        If False, read normal.csv and attack.csv and concatenate.

    Returns
    -------
    pd.DataFrame with columns: Timestamp (datetime), <51 sensors>, label (int 0/1)
    """
    # Build a skiprows function so we never load the full 1.44M rows into RAM
    # when stride > 1.  Row 0 is the header (always kept); data rows i >= 1
    # are kept only when (i - 1) % stride == 0.
    if stride > 1:
        skip_fn = lambda i: i > 0 and (i - 1) % stride != 0
    else:
        skip_fn = None

    if use_merged:
        path = DATA_DIR / "merged.csv"
        df = pd.read_csv(path, skiprows=skip_fn, low_memory=False)
    else:
        norm = pd.read_csv(DATA_DIR / "normal.csv", skiprows=skip_fn, low_memory=False)
        atk = pd.read_csv(DATA_DIR / "attack.csv", skiprows=skip_fn, low_memory=False)
        df = pd.concat([norm, atk], ignore_index=True)

    # Strip leading/trailing whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Parse timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Row cap
    if max_rows and max_rows < len(df):
        df = df.iloc[:max_rows].copy()

    # Binary label: 1 = attack active
    df["label"] = (df[LABEL_COL].str.strip() != "Normal").astype(int)
    df = df.drop(columns=[LABEL_COL])

    # Coerce all sensor columns to float
    sensor_cols = [c for c in df.columns if c not in {TIMESTAMP_COL, "label"}]
    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df


def get_sensor_cols(df: pd.DataFrame) -> list[str]:
    """Return list of sensor feature column names (excludes Timestamp and label)."""
    return [c for c in df.columns if c not in {TIMESTAMP_COL, "label"}]
