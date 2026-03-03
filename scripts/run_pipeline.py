"""End-to-end pipeline: train all model combinations, then evaluate.

Run from the project root:
    python scripts/run_pipeline.py [options]

Default runs all four datasets × two classifiers = 8 combinations:
    1. IBM       + XGBoost
    2. IBM       + RandomForest
    3. Synthetic + XGBoost
    4. Synthetic + RandomForest
    5. SWaT      + XGBoost   (W=60, H=60, stride=5 to keep memory < 3 GB)
    6. SWaT      + RandomForest
    7. C-MAPSS   + XGBoost   (W=30, H=30, official train/test split)
    8. C-MAPSS   + RandomForest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train import run as train_run
from src.models.evaluate import evaluate_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full predictive alerting pipeline")
    parser.add_argument(
        "--data",
        choices=["ibm", "synthetic", "swat", "cmapss", "both", "all"],
        default="all",
        help="Which dataset(s) to run (default: all)",
    )
    parser.add_argument(
        "--classifier",
        choices=["xgboost", "rf", "ensemble", "lstm", "tcn", "both", "all"],
        default="both",
        help="Classifier (default: both = xgboost+rf; all = all 5)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Row cap for IBM / SWaT (0 = all)",
    )
    parser.add_argument(
        "--stride", type=int, default=5,
        help="Downsample SWaT: keep every Nth row (default 5 = 5-second resolution)",
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

    results = []
    for ds in data_sources:
        for clf in classifiers:
            info = train_run(
                data_source=ds,
                classifier=clf,
                max_rows=args.max_rows,
                stride=args.stride,
                label_mode="active",
            )
            results.append(info)

    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)

    output_dir = Path("outputs")
    for info in results:
        pred_path = output_dir / f"predictions_{info['tag']}.npz"
        if pred_path.exists():
            evaluate_file(pred_path)


if __name__ == "__main__":
    main()
