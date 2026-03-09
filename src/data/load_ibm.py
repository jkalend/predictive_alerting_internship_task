"""Load pre-reduced IBM Cloud telemetry and anomaly ground truth."""

from pathlib import Path

import pandas as pd

from src.config import IBM_DATA_DIR


def _data_dir() -> Path:
    if IBM_DATA_DIR is None or str(IBM_DATA_DIR).strip() == "":
        raise RuntimeError(
            "IBM_DATA_DIR is not configured. Copy src/config/local.example.py to "
            "src/config/local.py and set IBM_DATA_DIR to your data path."
        )
    return Path(IBM_DATA_DIR)


def load_telemetry(max_rows: int = 0) -> pd.DataFrame:
    """Load the pre-reduced IBM Cloud telemetry feature file.

    The file is produced by scripts/reduce_features.py, which aggregates the
    raw unpivoted parquet (117k+ columns) into ~34 columns using DuckDB
    (global + per-datacenter 5xx / latency stats).  Only a row-limit is
    applied here; no additional column reduction is needed because the file
    is already compact (~34 columns, <<1 GB in RAM).
    """
    path = _data_dir() / "reduced_features_full.parquet"
    df = pd.read_parquet(path)
    df["interval_start"] = pd.to_datetime(df["interval_start"], unit="s", utc=True)
    df = df.sort_values("interval_start").reset_index(drop=True)
    if max_rows and max_rows < len(df):
        df = df.iloc[:max_rows].copy()
    return df


def load_anomaly_windows() -> pd.DataFrame:
    """Load labeled anomaly ground-truth windows."""
    path = _data_dir() / "anomaly_windows.csv"
    df = pd.read_csv(path)
    df["anomaly_start"] = pd.to_datetime(df["anomaly_start"], utc=True)
    df["anomaly_end"] = pd.to_datetime(df["anomaly_end"], utc=True)
    return df
