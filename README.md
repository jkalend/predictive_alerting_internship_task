# Predictive Alerting — Task 1

Predicts whether an incident will occur within the next **H = 6 steps (30 minutes)** based on the previous **W = 24 steps (2 hours)** of cloud telemetry metrics, using a sliding-window binary classification approach.

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

**IBM Cloud data**: Copy `src/config/local.example.py` to `src/config/local.py` and set `IBM_DATA_DIR` to your data path. `local.py` is gitignored.

## Datasets

| Dataset | Location | Source |
|---|---|---|
| IBM Cloud | Configure in `src/config/local.py` | [Zenodo 14062900](https://zenodo.org/records/14062900) |
| SWaT | `data_swat/merged.csv` | [Kaggle vishala28/swat-dataset-secure-water-treatment-system](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system) |
| C-MAPSS | `data_cmapss/train_FD001.txt` etc. | [Kaggle palbha/cmapss-jet-engine-simulated-data](https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data) |
| Synthetic | Generated on-the-fly | — |

## Run

### Full pipeline (all 4 datasets, both classifiers)
```powershell
python scripts/run_pipeline.py
```

### All classifiers (xgboost, rf, ensemble, lstm, tcn)
```powershell
python scripts/run_pipeline.py --classifier all
```

### Individual datasets
```powershell
python scripts/run_pipeline.py --data ibm
python scripts/run_pipeline.py --data swat --stride 15   # stride=15 → 15-sec resolution
python scripts/run_pipeline.py --data cmapss
python scripts/run_pipeline.py --data synthetic
```

### Quick test (IBM, XGBoost, 5k rows)
```powershell
python scripts/run_pipeline.py --data ibm --classifier xgboost --max-rows 5000
```

### Train only
```powershell
python -m src.models.train --data swat --classifier xgboost --stride 15
python -m src.models.train --data cmapss --classifier xgboost
python -m src.models.train --data synthetic --classifier lstm --max-rows 5000
python -m src.models.train --data synthetic --classifier tcn --max-rows 5000
python -m src.models.train --data synthetic --classifier ensemble
```

### Evaluate only (after training)
```powershell
python -m src.models.evaluate --all
```

## Outputs

All outputs are written to `outputs/`:
- `model_<data>_<classifier>.pkl` — serialised fitted model
- `predictions_<data>_<classifier>.npz` — test-set probabilities and ground truth
- `pr_curve_<data>_<classifier>.png` — precision–recall curve plot

## Project Structure

```
src/
  data/
    load_ibm.py       — IBM Cloud pre-reduced parquet + anomaly windows
    load_swat.py      — SWaT merged.csv with stride-based efficient loading
    load_cmapss.py    — C-MAPSS FD001 with per-engine RUL-derived labels
    synthetic.py      — synthetic telemetry generator with injected incidents
  features/
    windows.py        — rolling-stats + delta features; per-group support for C-MAPSS
  models/
    train.py          — XGBoost / RF / ensemble / LSTM / TCN for all 4 data sources
    sequential.py     — LSTM and TCN PyTorch models (raw-window input)
    evaluate.py       — threshold sweep, metrics, PR-curve plots
scripts/
  run_pipeline.py     — end-to-end pipeline runner
  reduce_features.py  — IBM Cloud feature reduction (unpivoted → ~34 cols via DuckDB)
docs/
  task1_report.md     — full report with modeling choices, analysis, and results
outputs/              — saved models, predictions, PR-curve plots
data_swat/            — SWaT CSV files
data_cmapss/          — C-MAPSS text files
```

## Report

See [docs/task1_report.md](docs/task1_report.md) for the full description of modeling choices, evaluation setup, and results (to be filled after training).
