"""Local config template — copy to local.py and fill in your paths.

    cp src/config/local.example.py src/config/local.py

Then edit local.py with your actual paths. local.py is gitignored.
"""
from __future__ import annotations

from pathlib import Path

# Directory containing reduced_features_full.parquet and anomaly_windows.csv
# (output of scripts/reduce_features.py, or your IBM Cloud data location)
IBM_DATA_DIR: Path = Path("")  # e.g. Path("E:/Side-Projects/chronos-experiments-3/data")
