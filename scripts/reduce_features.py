"""
Memory-efficient dimensionality reduction using DuckDB.
Reduces 117k+ features from the IBM Cloud dataset to ~100 Lambda-friendly features
without loading data into RAM (stays under 32GB).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def get_data_dir() -> Path:
    """Resolve data directory relative to project root."""
    return Path(__file__).resolve().parents[2] / "data"


def reduce_features(
    unpivoted_path: Path | str,
    output_path: Path | str,
    max_features: int = 100,
    max_rows: int | None = None,
) -> None:
    """
    Aggregate unpivoted telemetry into a small, predictive feature set using DuckDB.
    All processing is done out-of-core to respect 32GB RAM limit.
    If max_rows > 0, only output the first max_rows rows (for smaller cached files).
    """
    unpivoted_path = Path(unpivoted_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")

    # Register parquet - DuckDB reads directly from disk, no full load into RAM
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{unpivoted_path.as_posix()}')")

    limit_clause = f"LIMIT {max_rows}" if max_rows and max_rows > 0 else ""

    # Build aggregated features per 5-minute interval (interval_start)
    # Status code groups: 2xx, 3xx, 4xx, 5xx, other (-1, etc.)
    # aggregated_stats_name: 'count', 'avg', 'max', 'median', 'min', 'std', 'kurtosis', 'skewness'
    query = f"""
    WITH base AS (
        SELECT
            interval_start,
            location,
            statusCode,
            aggregated_stats_name,
            aggregated_stats_value
        FROM raw
    ),
    counts AS (
        SELECT
            interval_start,
            -- Global counts by status group
            SUM(CASE WHEN aggregated_stats_name = 'count' AND statusCode >= 200 AND statusCode < 300 THEN aggregated_stats_value ELSE 0 END) AS count_2xx,
            SUM(CASE WHEN aggregated_stats_name = 'count' AND statusCode >= 300 AND statusCode < 400 THEN aggregated_stats_value ELSE 0 END) AS count_3xx,
            SUM(CASE WHEN aggregated_stats_name = 'count' AND statusCode >= 400 AND statusCode < 500 THEN aggregated_stats_value ELSE 0 END) AS count_4xx,
            SUM(CASE WHEN aggregated_stats_name = 'count' AND statusCode >= 500 THEN aggregated_stats_value ELSE 0 END) AS count_5xx,
            SUM(CASE WHEN aggregated_stats_name = 'count' AND (statusCode < 0 OR statusCode >= 600) THEN aggregated_stats_value ELSE 0 END) AS count_other,
            SUM(CASE WHEN aggregated_stats_name = 'count' THEN aggregated_stats_value ELSE 0 END) AS total_count,
            -- Latency aggregates (avg of avg, max of max as p99 proxy)
            AVG(CASE WHEN aggregated_stats_name = 'avg' THEN aggregated_stats_value END) AS mean_latency_avg,
            MAX(CASE WHEN aggregated_stats_name = 'max' THEN aggregated_stats_value END) AS max_latency_max,
            AVG(CASE WHEN aggregated_stats_name = 'median' THEN aggregated_stats_value END) AS median_latency
        FROM base
        GROUP BY interval_start
    ),
    per_location AS (
        SELECT
            interval_start,
            location,
            SUM(CASE WHEN aggregated_stats_name = 'count' AND statusCode >= 500 THEN aggregated_stats_value ELSE 0 END) AS dc_5xx,
            SUM(CASE WHEN aggregated_stats_name = 'count' THEN aggregated_stats_value ELSE 0 END) AS dc_total,
            AVG(CASE WHEN aggregated_stats_name = 'avg' THEN aggregated_stats_value END) AS dc_mean_latency
        FROM base
        GROUP BY interval_start, location
    )
    SELECT
        c.interval_start,
        c.count_2xx,
        c.count_3xx,
        c.count_4xx,
        c.count_5xx,
        c.count_other,
        c.total_count,
        CASE WHEN c.total_count > 0 THEN c.count_5xx / c.total_count ELSE 0 END AS rate_5xx,
        CASE WHEN c.total_count > 0 THEN c.count_4xx / c.total_count ELSE 0 END AS rate_4xx,
        CASE WHEN c.total_count > 0 THEN c.count_2xx / c.total_count ELSE 0 END AS rate_2xx,
        c.mean_latency_avg,
        c.max_latency_max,
        c.median_latency,
        p1.dc_5xx AS dc1_5xx,
        p1.dc_total AS dc1_total,
        p1.dc_mean_latency AS dc1_latency,
        p2.dc_5xx AS dc2_5xx,
        p2.dc_total AS dc2_total,
        p2.dc_mean_latency AS dc2_latency,
        p3.dc_5xx AS dc3_5xx,
        p3.dc_total AS dc3_total,
        p3.dc_mean_latency AS dc3_latency,
        p4.dc_5xx AS dc4_5xx,
        p4.dc_total AS dc4_total,
        p4.dc_mean_latency AS dc4_latency,
        p5.dc_5xx AS dc5_5xx,
        p5.dc_total AS dc5_total,
        p5.dc_mean_latency AS dc5_latency,
        p6.dc_5xx AS dc6_5xx,
        p6.dc_total AS dc6_total,
        p6.dc_mean_latency AS dc6_latency,
        p7.dc_5xx AS dc7_5xx,
        p7.dc_total AS dc7_total,
        p7.dc_mean_latency AS dc7_latency
    FROM counts c
    LEFT JOIN (SELECT * FROM per_location WHERE location = 'datacenter1') p1 ON c.interval_start = p1.interval_start
    LEFT JOIN (SELECT * FROM per_location WHERE location = 'datacenter2') p2 ON c.interval_start = p2.interval_start
    LEFT JOIN (SELECT * FROM per_location WHERE location = 'datacenter3') p3 ON c.interval_start = p3.interval_start
    LEFT JOIN (SELECT * FROM per_location WHERE location = 'datacenter4') p4 ON c.interval_start = p4.interval_start
    LEFT JOIN (SELECT * FROM per_location WHERE location = 'datacenter5') p5 ON c.interval_start = p5.interval_start
    LEFT JOIN (SELECT * FROM per_location WHERE location = 'datacenter6') p6 ON c.interval_start = p6.interval_start
    LEFT JOIN (SELECT * FROM per_location WHERE location = 'datacenter7') p7 ON c.interval_start = p7.interval_start
    ORDER BY c.interval_start
    {limit_clause}
    """

    # Export to parquet - streamed, no full materialization in Python
    con.execute(f"COPY ({query}) TO '{output_path.as_posix()}' (FORMAT PARQUET)")
    con.close()

    print(f"Reduced features written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce IBM Cloud dataset features using DuckDB")
    parser.add_argument(
        "--unpivoted",
        type=Path,
        default=get_data_dir() / "unpivoted_data.parquet",
        help="Path to unpivoted_data.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for reduced parquet (default: reduced_features_{n}.parquet or reduced_features_full.parquet)",
    )
    parser.add_argument("--max-features", type=int, default=100, help="Max features (informational)")
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to output (0 = all). Affects default output filename.")
    args = parser.parse_args()

    if args.output is None:
        suffix = str(args.max_rows) if args.max_rows > 0 else "full"
        args.output = get_data_dir() / f"reduced_features_{suffix}.parquet"

    reduce_features(args.unpivoted, args.output, args.max_features, args.max_rows or None)


if __name__ == "__main__":
    main()
