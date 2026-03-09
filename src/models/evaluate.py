"""Evaluation script: threshold tuning and metric reporting.

Loads a saved predictions .npz file and prints a full threshold-sweep table,
highlighting the default (0.5) operating point and the optimal threshold for
≥80% recall with maximum precision (matching the evaluation methodology from
the previous Chronos experiments).

Usage
-----
    python -m src.models.evaluate [--predictions outputs/predictions_ibm_xgboost.npz]
                                  [--all]
                                  [--recall-target 0.80]

When --all is passed, all .npz files in outputs/ are evaluated in turn.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

OUTPUT_DIR = Path("outputs")


def evaluate_file(pred_path: str | Path, recall_target: float = 0.80) -> pd.DataFrame:
    """Evaluate one predictions .npz file.

    Uses validation-tuned threshold when available (optimal_threshold in npz)
    to avoid test-set data leakage. For legacy .npz files without it, falls
    back to sweep-on-test (with a warning).

    Returns
    -------
    pd.DataFrame with one row per evaluated threshold.
    """
    pred_path = Path(pred_path)
    data = np.load(pred_path, allow_pickle=True)
    proba: np.ndarray = data["proba"]
    y_true: np.ndarray = data["y_true"]
    optimal_threshold = float(data["optimal_threshold"]) if "optimal_threshold" in data else None

    tag = pred_path.stem.replace("predictions_", "")
    pos = int(y_true.sum())
    total = len(y_true)
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"  Test samples: {total:,}  |  Positives: {pos:,}  ({100*pos/total:.1f}%)")
    print(f"{'='*60}")

    # Global metrics (threshold-independent)
    roc_auc = roc_auc_score(y_true, proba) if pos > 0 else float("nan")
    pr_auc = average_precision_score(y_true, proba) if pos > 0 else float("nan")
    print(f"  ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")

    # Build threshold sweep (for display / PR curve only)
    _, _, thresholds = precision_recall_curve(y_true, proba)
    eval_thresholds = np.unique(np.concatenate([[0.5], thresholds]))

    rows = []
    for thresh in eval_thresholds:
        preds = (proba >= thresh).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        r = recall_score(y_true, preds, zero_division=0.0)
        p = precision_score(y_true, preds, zero_division=0.0)
        f1 = f1_score(y_true, preds, zero_division=0.0)
        rows.append(
            dict(threshold=thresh, recall=r, precision=p, f1=f1, tp=tp, fp=fp, fn=fn)
        )

    df = pd.DataFrame(rows)

    # ---- default operating point (0.5) ----
    row_05 = df[df["threshold"] == 0.5].iloc[0]
    print(f"\n  Threshold 0.5 (default)")
    _print_row(row_05)

    # ---- tuned threshold: validation-calibrated (no leakage) ----
    if optimal_threshold is not None:
        preds_tuned = (proba >= optimal_threshold).astype(int)
        r_tuned = recall_score(y_true, preds_tuned, zero_division=0.0)
        p_tuned = precision_score(y_true, preds_tuned, zero_division=0.0)
        f1_tuned = f1_score(y_true, preds_tuned, zero_division=0.0)
        tp = int(((preds_tuned == 1) & (y_true == 1)).sum())
        fp = int(((preds_tuned == 1) & (y_true == 0)).sum())
        fn = int(((preds_tuned == 0) & (y_true == 1)).sum())
        print(f"\n  Tuned threshold (calibrated on validation): {optimal_threshold:.4f}")
        print(
            f"    Recall={r_tuned*100:.2f}%  "
            f"Precision={p_tuned*100:.2f}%  "
            f"F1={f1_tuned*100:.2f}%  "
            f"TP={tp}  FP={fp}  FN={fn}"
        )
    else:
        # Legacy: sweep on test (data leakage — warn user)
        print(f"\n  [warn] No validation-tuned threshold in .npz; using test-set sweep (data leakage).")
        cands = df[df["recall"] >= recall_target]
        if not cands.empty:
            best = cands.loc[cands["precision"].idxmax()]
            print(f"  Best threshold for >={recall_target*100:.0f}% recall: {best.threshold:.4f}")
            _print_row(best)
        else:
            print(f"  [warn] No threshold achieves >={recall_target*100:.0f}% recall.")

    # ---- precision-recall curve plot ----
    _plot_pr_curve(y_true, proba, tag, recall_target)

    return df


def _print_row(row: pd.Series) -> None:
    print(
        f"    Recall={row.recall*100:.2f}%  "
        f"Precision={row.precision*100:.2f}%  "
        f"F1={row.f1*100:.2f}%  "
        f"TP={int(row.tp)}  FP={int(row.fp)}  FN={int(row.fn)}"
    )


def _plot_pr_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    tag: str,
    recall_target: float,
) -> None:
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_arr, precision_arr, lw=1.5, label=tag)
    ax.axvline(x=recall_target, color="gray", linestyle="--", lw=1,
               label=f"recall target = {recall_target:.0%}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall curve: {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / f"pr_curve_{tag}.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  PR curve saved -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate incident prediction model")
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to a predictions .npz file"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all .npz files in outputs/"
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.80,
        help="Recall target for threshold tuning (default: 0.80)"
    )
    args = parser.parse_args()

    if args.all:
        files = sorted(OUTPUT_DIR.glob("predictions_*.npz"))
        if not files:
            print("[eval] No prediction files found in outputs/")
            return
        for f in files:
            evaluate_file(f, recall_target=args.recall_target)
    elif args.predictions:
        evaluate_file(args.predictions, recall_target=args.recall_target)
    else:
        files = sorted(OUTPUT_DIR.glob("predictions_*.npz"))
        if not files:
            print("[eval] No prediction files found. Run train.py first.")
            return
        for f in files:
            evaluate_file(f, recall_target=args.recall_target)


if __name__ == "__main__":
    main()
