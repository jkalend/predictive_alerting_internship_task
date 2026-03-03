"""Synthetic time-series generator with labeled incident intervals.

Produces a multivariate telemetry stream that mimics IBM Cloud structure:
- Normal baseline: low, stationary 5xx error rates per datacenter
- Pre-incident ramp-up: gradual error spike in the W steps before each anomaly
- Incident burst: large error / latency spike during the anomaly window
- Class imbalance: ~4-5% positive labels, matching real-world distribution
"""

from typing import Tuple

import numpy as np
import pandas as pd


def generate_synthetic(
    n_steps: int = 10_000,
    n_datacenters: int = 7,
    anomaly_fraction: float = 0.05,
    pre_anomaly_lead: int = 12,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic telemetry DataFrame and matching anomaly windows.

    Parameters
    ----------
    n_steps:
        Total number of 5-minute intervals to generate.
    n_datacenters:
        Number of simulated datacenters (dc1 … dcN).
    anomaly_fraction:
        Approximate fraction of steps that fall inside an anomaly window.
    pre_anomaly_lead:
        Steps before each anomaly where error rates begin to ramp up —
        this is what the model can learn to detect early.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Telemetry with columns matching the IBM reduced-features schema.
    anomaly_df : pd.DataFrame
        Anomaly windows with ``anomaly_start`` and ``anomaly_end`` columns
        (UTC-aware Timestamps).
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps = pd.date_range(start, periods=n_steps, freq="5min")

    # --- Normal baseline per datacenter ---
    dc_total = {}
    dc_5xx = {}
    dc_latency = {}
    for dc in range(1, n_datacenters + 1):
        dc_total[dc] = rng.poisson(lam=500, size=n_steps).astype(float)
        dc_5xx[dc] = rng.poisson(lam=5, size=n_steps).astype(float)
        dc_latency[dc] = rng.lognormal(mean=3.0, sigma=0.3, size=n_steps)

    # --- Inject anomalies ---
    avg_duration = 18  # steps (~90 min)
    n_anomalies = max(1, int(n_steps * anomaly_fraction / avg_duration))
    min_gap = pre_anomaly_lead + avg_duration + 6
    centers = _sample_centers(rng, n_steps, n_anomalies, min_gap, pre_anomaly_lead)

    anomaly_records = []
    for center in centers:
        duration = int(rng.integers(6, avg_duration * 2))
        end_idx = min(center + duration - 1, n_steps - 1)

        # Pre-incident ramp: pick one datacenter to show leading signal
        ramp_dc = rng.integers(1, n_datacenters + 1)
        for step in range(max(0, center - pre_anomaly_lead), center):
            ramp = (step - (center - pre_anomaly_lead)) / pre_anomaly_lead
            dc_5xx[ramp_dc][step] *= 1 + ramp * 8

        # Burst during incident
        for step in range(center, end_idx + 1):
            for dc in range(1, n_datacenters + 1):
                dc_5xx[dc][step] *= rng.uniform(5, 20)
                dc_latency[dc][step] *= rng.uniform(3, 8)

        anomaly_records.append(
            {
                "anomaly_start": timestamps[center],
                "anomaly_end": timestamps[end_idx],
            }
        )

    # --- Assemble DataFrame ---
    data: dict = {"interval_start": timestamps}
    for dc in range(1, n_datacenters + 1):
        data[f"dc{dc}_total"] = dc_total[dc]
        data[f"dc{dc}_5xx"] = dc_5xx[dc]
        data[f"dc{dc}_latency"] = dc_latency[dc]

    total_all = sum(dc_total[dc] for dc in range(1, n_datacenters + 1))
    err_all = sum(dc_5xx[dc] for dc in range(1, n_datacenters + 1))
    lat_all = np.mean([dc_latency[dc] for dc in range(1, n_datacenters + 1)], axis=0)

    data["total_count"] = total_all
    data["count_5xx"] = err_all
    data["count_2xx"] = total_all - err_all
    data["count_3xx"] = np.zeros(n_steps)
    data["count_4xx"] = rng.poisson(lam=20 * n_datacenters, size=n_steps).astype(float)
    data["count_other"] = np.zeros(n_steps)
    data["rate_5xx"] = err_all / np.maximum(total_all, 1)
    data["rate_4xx"] = data["count_4xx"] / np.maximum(total_all, 1)
    data["rate_2xx"] = data["count_2xx"] / np.maximum(total_all, 1)
    data["mean_latency_avg"] = lat_all
    data["max_latency_max"] = np.max([dc_latency[dc] for dc in range(1, n_datacenters + 1)], axis=0)
    data["median_latency"] = np.median([dc_latency[dc] for dc in range(1, n_datacenters + 1)], axis=0)

    df = pd.DataFrame(data)
    anomaly_df = pd.DataFrame(anomaly_records)
    return df, anomaly_df


def _sample_centers(
    rng: np.random.Generator,
    n_steps: int,
    n_anomalies: int,
    min_gap: int,
    lead: int,
) -> list:
    """Sample anomaly center indices with minimum spacing."""
    centers = []
    forbidden: set = set()
    attempts = 0
    while len(centers) < n_anomalies and attempts < n_anomalies * 50:
        c = int(rng.integers(lead + 5, n_steps - 30))
        if not any(abs(c - existing) < min_gap for existing in centers):
            centers.append(c)
        attempts += 1
    return sorted(centers)
