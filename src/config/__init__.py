"""Configuration for data paths and other local settings.

Local overrides live in config/local.py (gitignored). Copy config/local.example.py
to config/local.py and fill in your paths.
"""
from __future__ import annotations

from pathlib import Path

from src.config.defaults import IBM_DATA_DIR as _default
IBM_DATA_DIR: Path | None = _default

try:
    from src.config.local import IBM_DATA_DIR as _local
    IBM_DATA_DIR = _local
except ImportError:
    pass
