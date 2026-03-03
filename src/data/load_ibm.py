"""Load pre-reduced IBM Cloud telemetry and anomaly ground truth."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("E:/Side-Projects/chronos-experiments-3-done-NotOnGH/data")


def load_telemetry(max_rows: int = 0) -> pd.DataFrame:
    """Load the pre-reduced IBM Cloud telemetry feature file.

    The file was produced by the previous reduce_features step and contains 34
    columns (global + per-datacenter 5xx / latency stats).  Only a row-limit
    is applied here; no additional column reduction is needed because the file
    is already compact (~34 columns, <<1 GB in RAM).
    """
    path = DATA_DIR / "reduced_features_full.parquet"
    df = pd.read_parquet(path)
    df["interval_start"] = pd.to_datetime(df["interval_start"], unit="s", utc=True)
    df = df.sort_values("interval_start").reset_index(drop=True)
    if max_rows and max_rows < len(df):
        df = df.iloc[:max_rows].copy()
    return df


def load_anomaly_windows() -> pd.DataFrame:
    """Load labeled anomaly ground-truth windows."""
    path = DATA_DIR / "anomaly_windows.csv"
    df = pd.read_csv(path)
    df["anomaly_start"] = pd.to_datetime(df["anomaly_start"], utc=True)
    df["anomaly_end"] = pd.to_datetime(df["anomaly_end"], utc=True)
    return df
