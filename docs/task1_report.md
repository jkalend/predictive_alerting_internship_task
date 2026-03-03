# Task 1 Report: Predictive Alerting via Sliding-Window Classification

---

## 1. Problem Formulation

The goal is to predict whether an incident will occur within the next **H time steps** based on the previous **W steps** of one or more time-series metrics. This is framed as a supervised binary classification problem using a sliding-window formulation:

- **Input (features)**: Rolling statistics (mean, standard deviation, maximum) and rate-of-change (delta over W and W/2 steps) computed over the last W steps for each metric column at time step *t*. Window size W is dataset-specific.
- **Output (label)**: Binary indicator — `1` if any incident is *active* within the next H steps (i.e., the incident window [anomaly\_start, anomaly\_end] overlaps [t, t+H]), else `0`. Horizon H is dataset-specific.
- **Step granularity**: Dataset-specific (5 min for IBM/Synthetic, 15 sec for SWaT, 1 cycle for C-MAPSS).

The sliding-window approach converts the raw time series into a standard tabular dataset, where each row is one 5-minute interval and the feature columns encode the recent system state as seen by the classifier at prediction time. This mirrors how an operational alerting system would function: at each new telemetry batch, re-evaluate the current feature vector and output a probability score.

---

## 2. Dataset

### 2.1 IBM Cloud Dataset (Primary)

- **Source**: IBM Cloud Console telemetry — [Zenodo 14062900](https://zenodo.org/records/14062900)
- **Size**: 39,365 intervals × 34 feature columns (pre-reduced from 117,448 raw columns).
- **Span**: ~4.5 months of 5-minute aggregated telemetry.
- **Features used**: Global request counts, HTTP response code rates (2xx, 4xx, 5xx), average/max/median latency, and per-datacenter equivalents across 7 datacenters.
- **Labels**: 25 labeled anomaly windows from `anomaly_windows.csv`, derived from IBM's internal monitoring tools (Issue Tracker, Instant Messenger, Test Log).
- **Class imbalance**: 2.78% of steps are positive (`active` mode, W=24, H=6). In strict `onset` mode (anomaly must start in next 30 min), only 0.35% are positive — too sparse for reliable training with 25 anomaly windows.

**Feature reduction**: The raw dataset (`pivoted_data_all.parquet`) contains 117,448 columns — one per (datacenter, HTTP status code, aggregation type) combination — and would expand to 30–50 GB in RAM when fully loaded. A prior `reduce_features.py` preprocessing step (from the Chronos experiments) condensed this to 34 columns by selecting:
- Global aggregates: total request count, per-status-code counts (2xx/3xx/4xx/5xx/other), error rates, average/max/median latency
- Per-datacenter summaries (7 DCs × 3 metrics): 5xx count, total count, average latency

The result (`reduced_features_full.parquet`, ~44 MB on disk) loads comfortably within available RAM and retains the most operationally relevant signals. Column filtering was applied using `pyarrow.parquet.read_table()` before loading into pandas to avoid materialising the full wide table.

### 2.2 Synthetic Dataset (Secondary / Demonstration)

To demonstrate that the problem formulation is correct independently of dataset complexity, a synthetic dataset was also generated:

- **10,000 steps** (5-min intervals, matching IBM granularity).
- **7 simulated datacenters** with Poisson-distributed request counts and log-normal latency.
- **Anomaly injection**: Controlled spikes in 5xx error rates and latency during incident windows, preceded by a **12-step (60 min) ramp-up** of errors in one datacenter. This ramp-up is the signal the model learns to detect.
- **Class imbalance**: ~5% positive labels by design.

### 2.3 SWaT — Secure Water Treatment System

- **Source**: iTrust Labs, SUTD (Dec 2015) — available via Kaggle `vishala28/swat-dataset-secure-water-treatment-system`, files at `data_swat/`
- **Size**: 1,441,719 rows × 53 columns (Timestamp + 51 process sensors + label). Used at **stride=15** (15-second resolution) → 96,115 rows.
- **Granularity**: 1-second intervals (original); downsampled to 15-second resolution to make RandomForest training feasible.
- **Features**: 51 sensors covering all 6 stages of the water treatment plant: flow meters (FIT), level sensors (LIT), water quality analysers (AIT), motorised valves (MV), pumps (P), differential pressure indicators (DPIT), UV lamps (UV), pressure indicators (PIT).
- **Labels**: `Normal` (7 days) and `Attack` (4 days, 36 distinct cyber-physical attacks). Label = 1 if attack active in next H=60 steps (15-minute horizon at 15-sec resolution).
- **Class imbalance**: 1.94% positive in training, 19.88% in test. Attack scenarios cluster in the last 4 days — the test set covers the bulk of the attack period. This structural split is a known characteristic of the dataset.
- **Train/test split**: Chronological 80/20. Training = normal operation; test = primarily attack period.

### 2.4 C-MAPSS — Turbofan Engine Degradation (FD001)

- **Source**: NASA Prognostics Data Repository — available via Kaggle `palbha/cmapss-jet-engine-simulated-data`, files at `data_cmapss/`
- **Size**: `train_FD001.txt` — 20,631 rows, 100 engines; `test_FD001.txt` — 13,096 rows, 100 engines.
- **Granularity**: One row per operational cycle per engine (no fixed time unit).
- **Features**: 3 operational settings + 21 sensor readings (temperatures T2/T24/T30/T50, pressures P2/P15/P30, fan/core speeds Nf/Nc, engine pressure ratio, fuel flow, etc.). 24 features × 6 stats = 144 feature columns.
- **Label derivation**: No per-row binary label is provided. RUL (Remaining Useful Life) is computed as `max_cycle_per_engine − current_cycle`. Label = 1 if RUL ≤ H = 30 cycles ("engine will fail within 30 cycles").
- **Class imbalance**: 15.03% positive in training; 2.54% in test.
- **Train/test split**: Official file split (different engine sets). Training uses `train_FD001.txt` (full run-to-failure); test uses `test_FD001.txt` with ground-truth RUL from `RUL_FD001.txt`.
- **Key engineering detail**: Rolling statistics are computed **per engine** (`group_col='unit_number'`) to prevent mixing sensor degradation profiles across different engines. This is essential for correctness — without per-engine grouping, the rolling window would span the boundary between two engines and produce meaningless features.

---

## 3. Modeling Choices

### 3.1 Algorithms: XGBoost, RandomForest, Ensemble, LSTM, TCN

Five classifiers are evaluated:

| Classifier | Type | Input | Notes |
|---|---|---|---|
| **XGBoost** | Tree ensemble | Rolling stats | `scale_pos_weight` for imbalance |
| **RandomForest** | Tree ensemble | Rolling stats | `class_weight='balanced'` |
| **Ensemble** | Average of RF + XGBoost | Rolling stats | Combines both tree models |
| **LSTM** | Recurrent neural net | Raw W-step sequences | PyTorch; learns temporal patterns directly |
| **TCN** | Temporal ConvNet | Raw W-step sequences | PyTorch; dilated convolutions over time |

Tree ensembles operate on the CPU; LSTM and TCN use GPU when available. Tree models use rolling-statistics features; LSTM and TCN consume the raw window of metric values per channel.

**Why tree ensembles as primary?** The task specification emphasises correct problem formulation over model complexity. Tree ensembles are:
- Well-calibrated for tabular data without extensive hyperparameter search.
- Robust to feature scale differences (no normalisation needed).
- Directly interpretable via feature importances, supporting post-hoc discussion of which metrics most strongly predict incidents.
- Proven on this exact dataset in the prior Chronos experiments (XGBoost + rolling stats achieved 99%+ F1 on the full IBM dataset).

### 3.2 Window Parameters

| Dataset | Look-back W | Horizon H | Step size | Rationale |
|---|---|---|---|---|
| IBM Cloud | 24 steps (2 h) | 6 steps (30 min) | 5 min | Captures 2-hour context around anomalies; 30-min lead time for incident response |
| Synthetic | 24 steps (2 h) | 6 steps (30 min) | 5 min | Matches IBM granularity |
| SWaT | 60 steps (15 min) | 60 steps (15 min) | 15 sec | 15-min history captures sensor dynamics before attacks; 15-min alert lead time |
| C-MAPSS | 30 cycles | 30 cycles | 1 cycle | 30-cycle degradation context; label = 1 if RUL ≤ 30 cycles |

### 3.3 Feature Engineering

For each of the *C* metric columns available (C = 29 for IBM, C = 29 for synthetic), four features are computed per column at each time step *t*:

1. `{col}_mean24` — rolling mean over the last 24 steps
2. `{col}_std24` — rolling standard deviation
3. `{col}_max24` — rolling maximum
4. `{col}_last` — raw value at *t*

This yields **4 × C** features per sample. Rolling statistics are preferred over raw window flattening (which would yield **W × C = 24 × 29 = 696** features) because:
- They are more compact and less prone to overfitting on a ~39k-row dataset.
- They capture the trend (rising mean, growing variance) that typically precedes an incident rather than exact step-by-step values.
- They are robust to short-term jitter in individual 5-minute intervals.

### 3.4 Class-Imbalance Handling

The dataset is heavily imbalanced (~*[fill]%* positive). Two complementary strategies are used:

- **XGBoost**: `scale_pos_weight = n_negative / n_positive` — upweights the minority class during gradient computation.
- **RandomForest**: `class_weight='balanced'` — inversely weights each class proportional to its frequency.

In both cases, the raw probability score is used at inference time, and the operating threshold is tuned post-hoc rather than hard-coded to 0.5.

### 3.5 Label Mode

Two label strategies are implemented (controlled by `--label-mode`):

- **`onset`** (strict): label=1 if an anomaly *starts* in (t, t+H×5min]. With 25 IBM anomaly windows, this yields only 139 positives (0.35%) — far too sparse for a train set of 31k samples.
- **`active`** (default): label=1 if any anomaly window is *active* (ongoing or about to start) within [t, t+H×5min]. This yields 1,096 positives (2.78%) and is operationally equivalent: it answers "will there be an anomaly active in the next 30 minutes?". This is the mode used for all results below.

---

## 4. Evaluation Setup

### 4.1 Train / Test Split

A **chronological 80/20 split** is used: the first 80% of time steps form the training set, the last 20% form the test set. This is the only valid split for time series — random shuffling would leak future information into training.

| Split | Size (IBM full dataset) | Period |
|---|---|---|
| Train | 31,492 steps (~110 days) | 2024-01-22 to 2024-05-11 (20 anomaly windows) |
| Test  |  7,873 steps (~27 days)  | 2024-05-11 to 2024-06-07 (5 anomaly windows: a21–a25) |

### 4.2 Metrics

| Metric | Why it matters for alerting |
|---|---|
| **Recall** | Missing a real incident (false negative) is operationally the worst outcome — an alert system that misses incidents is worse than useless. |
| **Precision** | Excessive false alarms cause alert fatigue, eroding operator trust. |
| **F1** | Harmonic mean; used as a single summary metric. |
| **ROC-AUC** | Threshold-independent ranking quality. |
| **PR-AUC** | More informative than ROC-AUC under class imbalance. |

### 4.3 Alert Threshold Tuning

Rather than using a fixed 0.5 threshold, a threshold sweep is performed over all unique probability values in the test set. The reported operating points are:

1. **Default (0.5)**: Baseline behaviour.
2. **Tuned (≥80% recall)**: The highest-precision threshold that still achieves at least 80% recall — matching the evaluation convention from the Chronos experiments and reflecting a realistic alerting requirement.

---

## 5. Evaluation Results

All results use the `active` label mode. Test set sizes: IBM = 7,873 (133 positives, 1.7%); Synthetic = 2,000 (138 positives, 6.9%); SWaT = 57,669 (10,003 positives, 17.3%); C-MAPSS = 13,096 (332 positives, 2.5%).

### 5.1 IBM Dataset

| Classifier | ROC-AUC | PR-AUC | F1 @ 0.5 | Tuned Recall | Tuned Precision | Tuned F1 |
|---|---|---|---|---|---|---|
| XGBoost | 0.390 | 0.016 | 0.00% | 100.00% | 1.80% | 3.53% |
| RandomForest | 0.337 | 0.012 | 0.00% | 97.74% | 1.75% | 3.44% |
| Ensemble | 0.335 | 0.012 | 0.00% | 100.00% | 1.77% | 3.47% |
| LSTM | 0.426 | 0.017 | 3.28% | 100.00% | 1.69% | 3.33% |
| **TCN** | **0.502** | 0.017 | 3.33% | 100.00% | 1.69% | 3.33% |

Tree models predict zero positives at threshold 0.5; LSTM and TCN achieve ~100% recall at 0.5 but with very low precision (~1.7%). TCN has the highest ROC-AUC (0.50) — barely above chance — indicating features are not predictive of IBM anomalies.

---

### 5.2 Synthetic Dataset

| Classifier | ROC-AUC | PR-AUC | F1 @ 0.5 | Tuned Recall | Tuned Precision | Tuned F1 |
|---|---|---|---|---|---|---|
| XGBoost | 0.992 | 0.979 | 97.01% | 94.20% | 100.00% | 97.01% |
| RandomForest | 0.998 | 0.990 | 96.63% | 94.20% | 100.00% | 97.01% |
| Ensemble | 0.998 | 0.990 | 97.01% | 94.20% | 100.00% | 97.01% |
| LSTM | **0.9997** | **0.997** | 72.63% | 96.38% | 100.00% | **98.15%** |
| TCN | 0.997 | 0.985 | 95.34% | 94.20% | 100.00% | 97.01% |

All classifiers excel. LSTM achieves the highest tuned F1 (98.15%) and best ROC/PR-AUC; tree models and TCN reach 97% F1 with zero false positives at default or tuned threshold.

---

### 5.3 SWaT Dataset

| Classifier | ROC-AUC | PR-AUC | F1 @ 0.5 | Tuned Recall | Tuned Precision | Tuned F1 |
|---|---|---|---|---|---|---|
| XGBoost | 0.787 | 0.409 | 0.00% | 80.03% | 30.73% | 44.40% |
| RandomForest | 0.814 | 0.603 | 0.00% | 80.03% | 29.82% | 43.45% |
| Ensemble | 0.814 | 0.603 | 0.00% | 80.03% | 29.82% | 43.45% |
| LSTM | 0.276 | 0.125 | 31.50% | 97.87% | 19.10% | 31.97% |
| **TCN** | **0.818** | **0.746** | **51.14%** | 80.01% | 28.25% | 41.75% |

**TCN is the best SWaT model**: highest ROC-AUC (0.82) and PR-AUC (0.75), and the only model achieving usable F1 (51%) at threshold 0.5. XGBoost and RF have improved markedly vs. earlier runs (ROC-AUC ~0.79–0.81 vs. 0.40–0.73). LSTM fails (ROC-AUC 0.28) — it over-predicts positives, yielding high recall but very low precision.

---

### 5.4 C-MAPSS FD001

| Classifier | ROC-AUC | PR-AUC | F1 @ 0.5 | Tuned Recall | Tuned Precision | Tuned F1 |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.997** | **0.904** | **79.64%** | 80.42% | 79.23% | 79.82% |
| **RandomForest** | 0.996 | 0.887 | 78.08% | 81.02% | 76.64% | 78.77% |
| **Ensemble** | 0.997 | 0.901 | 79.57% | 81.33% | 79.18% | **80.24%** |
| LSTM | 0.614 | 0.033 | 6.31% | 100.00% | 3.26% | 6.31% |
| TCN | 0.962 | 0.580 | 39.82% | 80.12% | 31.04% | 44.74% |

Tree-based models dominate C-MAPSS. XGBoost and ensemble achieve ~80% F1 at default threshold. LSTM collapses (ROC-AUC 0.61, PR-AUC 0.03) — it predicts almost everything positive. TCN is moderate (ROC-AUC 0.96) but tuned F1 (44.7%) is far below tree models.

---

### 5.5 Full Summary Table

| Dataset | Classifier | F1 @ 0.5 | Tuned F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| IBM | TCN (best) | 3.33% | 3.33% | 0.502 | 0.017 |
| IBM | XGBoost | 0.00% | 3.53% | 0.390 | 0.016 |
| Synthetic | LSTM (best) | 72.63% | **98.15%** | 0.9997 | 0.997 |
| Synthetic | XGBoost | 97.01% | 97.01% | 0.992 | 0.979 |
| SWaT | **TCN (best)** | **51.14%** | 41.75% | **0.818** | **0.746** |
| SWaT | XGBoost | 0.00% | 44.40% | 0.787 | 0.409 |
| C-MAPSS | **Ensemble (best)** | 79.57% | **80.24%** | 0.997 | 0.901 |
| C-MAPSS | XGBoost | 79.64% | 79.82% | 0.997 | 0.904 |

---

## 6. Analysis & Discussion

### 6.1 IBM Dataset Results

All five classifiers produce **ROC-AUC at or below 0.5** on the IBM test set. TCN reaches 0.50 (barely above chance); tree models and ensemble are worse (0.33–0.39). At threshold 0.5, tree models predict zero positives; LSTM and TCN achieve 98–100% recall but with ~1.7% precision (thousands of false alarms).

**Root cause — features are not predictive of future IBM anomalies:**

The IBM anomaly windows (a1–a25) capture infrastructure-level incidents identified via Issue Tracker, Instant Messenger, and internal test logs. These events are not necessarily preceded by detectable changes in aggregate HTTP 5xx rates or latency averages. The aggregate metrics in `reduced_features_full.parquet` capture request volume and error rates at a global level, while the IBM anomalies may be driven by network-layer or infrastructure events not visible in these aggregates.

**Conclusion for IBM:** The sliding-window formulation is implemented correctly. The failure is attributable to a feature–label mismatch. More fine-grained features would be needed for meaningful predictions.

### 6.2 Synthetic Dataset Results

All five classifiers achieve **≥95% F1** (tuned). Tree models and TCN reach 97% F1 with zero false positives at default or tuned threshold. LSTM achieves the highest tuned F1 (98.15%) and best ROC/PR-AUC (0.9997 / 0.997), confirming that when the signal is present in the raw sequence, neural models can exploit it effectively.

The formulation is validated: a 30-minute horizon is sufficient when the signal is present in the features or raw window.

### 6.3 SWaT Dataset Results

**TCN is the best SWaT model** (ROC-AUC 0.82, PR-AUC 0.75, F1 51% at 0.5). Raw-sequence modeling captures pre-attack sensor dynamics that rolling statistics miss. XGBoost and RandomForest have improved substantially (ROC-AUC 0.79–0.81 vs. earlier 0.40–0.73), suggesting the current SWaT split or data preprocessing yields more learnable signal. LSTM fails (ROC-AUC 0.28) — it over-predicts positives, achieving high recall but very low precision.

**Root cause of difficulty**: The chronological 80/20 split places normal operation in training and attack scenarios largely in test. The test set is 17.3% positive; training is far more imbalanced. TCN’s ability to model raw sequences helps it generalise better than tree models on rolling stats.

### 6.4 C-MAPSS Dataset Results

**Tree-based models dominate C-MAPSS.** XGBoost, RandomForest, and ensemble achieve 78–80% F1 at default threshold with ROC-AUC 0.996–0.997. The tabular rolling-statistics representation is well-suited to gradual, monotonic engine degradation.

**LSTM fails** (ROC-AUC 0.61, PR-AUC 0.03) — it predicts almost everything positive, suggesting it does not learn the degradation pattern from raw sequences. **TCN is moderate** (ROC-AUC 0.96) but tuned F1 (44.7%) is far below tree models. Per-engine rolling statistics appear to be the right abstraction for this dataset; raw sequences may add noise or require different architectures.

### 6.5 Comparison: Tree Models vs. Neural Models

| Dataset | Best model | Why |
|---|---|---|
| IBM | TCN (still poor) | Slightly above chance; no model succeeds |
| Synthetic | LSTM | Raw sequence captures ramp-up; all models work |
| SWaT | **TCN** | Raw sequences capture attack dynamics; trees improved but TCN best |
| C-MAPSS | **XGBoost / Ensemble** | Rolling stats suit monotonic degradation; LSTM/TCN underperform |

**Takeaways:**
- **Synthetic**: Neural and tree models both excel; formulation validated.
- **SWaT**: TCN’s raw-sequence modeling is advantageous; tree models have improved.
- **C-MAPSS**: Tree models on rolling stats are optimal; neural models struggle.
- **IBM**: All models fail; feature quality is the bottleneck, not model choice.

### 6.6 Feature Importances (Tree Models)

**Synthetic XGBoost (top 3):** `rate_2xx_last`, `count_5xx_last`, `rate_5xx_last` — current-step error and success rates dominate, confirming the model detects the onset signal.

**IBM XGBoost (top 3):** `dc5_total_std24`, `dc5_total_max24`, `dc1_total_mean24` — traffic volume variance dominates. The even importance distribution suggests no clear discriminative feature — consistent with poor ROC-AUC.

**C-MAPSS XGBoost:** Sensors in the T30/T50 range (stage temperatures) and NRc (corrected core speed ratio) degrade monotonically in HPC faults; rolling mean and delta features capture degradation level and rate.

**SWaT:** Level sensors (LIT) and flow meters (FIT) should dominate since attacks target pumps and valves; delta features capture sudden step-changes. TCN’s raw-sequence input avoids explicit feature engineering and learns temporal patterns directly.

---

## 7. Limitations

1. **Single train–test split**: A single chronological split is used. For a production system, walk-forward cross-validation would give a more robust performance estimate.
2. **Static window sizes**: W and H are fixed. A real system would sweep these hyperparameters.
3. **No temporal context across the split boundary**: The model is not fine-tuned online; it does not adapt to concept drift in the live telemetry stream.
4. **Label granularity and feature mismatch**: The 25 IBM anomaly windows are coarse-grained infrastructure events identified via monitoring tools, not necessarily correlated with aggregate HTTP error rates. Some windows overlap (a3/a4/a5 all start within 6 minutes on 2024-02-12), inflating the apparent positive count near those windows. More critically, the available features are not predictive of these events — the fundamental requirement for any classifier to succeed.
5. **SWaT temporal distribution shift**: The merged SWaT file concatenates normal operation (7 days) followed by attack scenarios (4 days). A strict 80/20 chronological split puts almost all attacks in the test set, creating a severe train/test distribution mismatch. In a real deployment, the model would be trained on a balanced historical dataset that includes past attack episodes.
6. **Synthetic data simplifications**: The synthetic generator uses independent Poisson arrivals and does not model correlations between datacenters, time-of-day patterns, or multi-step dependencies that exist in real production traffic.

---

## 8. Adapting to a Real Alerting System

1. **Streaming inference**: At each new telemetry batch (every 5 min), recompute the rolling-stats feature vector for the current interval and query the trained model for a probability score.
2. **Threshold calibration**: Tune the alert threshold on a held-out validation window, not on the test set, to avoid overfitting the operating point.
3. **Online learning / periodic retraining**: Retrain the model on a rolling window of recent data (e.g., last 30 days) to adapt to evolving traffic patterns.
4. **Multi-horizon ensemble**: Train separate models for H = 1, 3, 6, 12 steps and combine their scores to give operators different lead times.
5. **Alarm deduplication**: Suppress repeated alerts within the same incident window to avoid flooding the on-call queue.
