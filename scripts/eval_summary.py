"""Quick evaluation summary - load npz and compute metrics without plotting."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main():
    main_tags = ["ibm", "synthetic", "swat", "cmapss"]
    main_clfs = ["xgboost", "rf", "ensemble", "lstm", "tcn"]
    results = []
    for ds in main_tags:
        for clf in main_clfs:
            p = OUTPUT_DIR / f"predictions_{ds}_{clf}.npz"
            if not p.exists():
                continue
            data = np.load(p, allow_pickle=True)
            proba = data["proba"]
            y_true = data["y_true"]
            opt_thresh = float(data["optimal_threshold"]) if "optimal_threshold" in data else None
            pos = int(y_true.sum())
            total = len(y_true)
            roc_auc = roc_auc_score(y_true, proba) if pos > 0 else float("nan")
            pr_auc = average_precision_score(y_true, proba) if pos > 0 else float("nan")
            preds_05 = (proba >= 0.5).astype(int)
            r_05 = recall_score(y_true, preds_05, zero_division=0.0)
            p_05 = precision_score(y_true, preds_05, zero_division=0.0)
            f1_05 = f1_score(y_true, preds_05, zero_division=0.0)
            if opt_thresh is not None:
                preds_t = (proba >= opt_thresh).astype(int)
                r_t = recall_score(y_true, preds_t, zero_division=0.0)
                p_t = precision_score(y_true, preds_t, zero_division=0.0)
                f1_t = f1_score(y_true, preds_t, zero_division=0.0)
            else:
                r_t, p_t, f1_t = float("nan"), float("nan"), float("nan")
            results.append({
                "tag": f"{ds}_{clf}",
                "dataset": ds,
                "clf": clf,
                "n": total,
                "pos": pos,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "r_05": r_05,
                "p_05": p_05,
                "f1_05": f1_05,
                "r_tuned": r_t,
                "p_tuned": p_t,
                "f1_tuned": f1_t,
                "opt_thresh": opt_thresh,
            })
    # Print summary
    for r in results:
        print(f"\n{r['tag']}: n={r['n']}, pos={r['pos']} ({100*r['pos']/r['n']:.1f}%)")
        print(f"  ROC-AUC={r['roc_auc']:.4f}  PR-AUC={r['pr_auc']:.4f}")
        print(f"  @0.5: R={r['r_05']*100:.2f}% P={r['p_05']*100:.2f}% F1={r['f1_05']*100:.2f}%")
        if r["opt_thresh"] is not None:
            print(f"  Tuned: R={r['r_tuned']*100:.2f}% P={r['p_tuned']*100:.2f}% F1={r['f1_tuned']*100:.2f}%")


if __name__ == "__main__":
    main()
