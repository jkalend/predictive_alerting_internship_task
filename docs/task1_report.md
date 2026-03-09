# Task 1 Report: Predictive Alerting via Sliding-Window Classification

---

## 1. Problem Formulation

The goal is to predict whether an incident will occur within the next **H time steps** based on the previous **W steps** of one or more time-series metrics. This is framed as a supervised binary classification problem using a sliding-window formulation:

- **Input (features)**: Rolling statistics (mean, standard deviation, maximum) and rate-of-change (delta over W and W/2 steps) computed over the last W steps for each metric column at time step *t*. Window size W is dataset-specific.
- **Output (label)**: Binary indicator: `1` if any incident is *active* within the next H steps (i.e., the incident window [anomaly\_start, anomaly\_end] overlaps [t, t+H]), else `0`. Horizon H is dataset-specific.
- **Step granularity**: Dataset-specific (5 min for IBM/Synthetic, 15 sec for SWaT, 1 cycle for C-MAPSS).

The sliding-window approach converts the raw time series into a standard tabular dataset, where each row is one 5-minute interval and the feature columns encode the recent system state as seen by the classifier at prediction time. This mirrors how an operational alerting system would function: at each new telemetry batch, re-evaluate the current feature vector and output a probability score.

---

## 2. Dataset

### 2.1 IBM Cloud Dataset (Primary)

- **Source**: IBM Cloud Console telemetry: [Zenodo 14062900](https://zenodo.org/records/14062900)
- **Size**: 39,365 intervals × 34 feature columns (pre-reduced from 117,448 raw columns).
- **Span**: ~4.5 months of 5-minute aggregated telemetry.
- **Features used**: Global request counts, HTTP response code rates (2xx, 4xx, 5xx), average/max/median latency, and per-datacenter equivalents across 7 datacenters.
- **Labels**: 25 labeled anomaly windows from `anomaly_windows.csv`, derived from IBM's internal monitoring tools (Issue Tracker, Instant Messenger, Test Log).
- **Class imbalance**: 2.78% of steps are positive (`active` mode, W=24, H=6). In strict `onset` mode (anomaly must start in next 30 min), only 0.35% are positive: too sparse for reliable training with 25 anomaly windows.

**Feature reduction**: The raw dataset (`pivoted_data_all.parquet`) contains 117,448 columns: one per (datacenter, HTTP status code, aggregation type) combination, and would expand to 30-50 GB in RAM when fully loaded. The `scripts/reduce_features.py` script in this project performs the feature reduction: it uses DuckDB for memory-efficient, out-of-core aggregation of the unpivoted parquet and condenses this to 34 columns by selecting:
- Global aggregates: total request count, per-status-code counts (2xx/3xx/4xx/5xx/other), error rates, average/max/median latency
- Per-datacenter summaries (7 DCs × 3 metrics): 5xx count, total count, average latency

The result (`reduced_features_full.parquet`, ~44 MB on disk) loads comfortably within available RAM and retains the most operationally relevant signals. `load_ibm.py` reads this parquet directly; no further filtering is needed.

### 2.2 Synthetic Dataset (Secondary / Demonstration)

To demonstrate that the problem formulation is correct independently of dataset complexity, a synthetic dataset was also generated:

- **10,000 steps** (5-min intervals, matching IBM granularity).
- **7 simulated datacenters** with Poisson-distributed request counts and log-normal latency.
- **Anomaly injection**: Controlled spikes in 5xx error rates and latency during incident windows, preceded by a **12-step (60 min) ramp-up** of errors in one datacenter. This ramp-up is the signal the model learns to detect.
- **Class imbalance**: ~5% positive labels by design.

### 2.3 SWaT: Secure Water Treatment System

- **Source**: iTrust Labs, SUTD (Dec 2015): available via Kaggle `vishala28/swat-dataset-secure-water-treatment-system`, files at `data_swat/`
- **Size**: 1,441,719 rows × 53 columns (Timestamp + 51 process sensors + label). The raw data is downsampled by stride for tractability; the **reported results use stride=5** (5-second resolution) → ~55k test samples. Alternative configurations (e.g. stride=15 for 15-second resolution) yield different sample counts.
- **Granularity**: 1-second intervals (original); stride controls the effective resolution.
- **Features**: 51 sensors covering all 6 stages of the water treatment plant: flow meters (FIT), level sensors (LIT), water quality analysers (AIT), motorised valves (MV), pumps (P), differential pressure indicators (DPIT), UV lamps (UV), pressure indicators (PIT).
- **Labels**: `Normal` (7 days) and `Attack` (4 days, 36 distinct cyber-physical attacks). Label = 1 if attack active in next H=60 steps (e.g. 5-min horizon at stride=5, 15-min at stride=15).
- **Class imbalance**: 1.94% positive in training, 19.88% in test. Attack scenarios cluster in the last 4 days: the test set covers the bulk of the attack period. This structural split is a known characteristic of the dataset.
- **Train/test split**: Chronological 80/20. Training = normal operation; test = primarily attack period.

### 2.4 C-MAPSS: Turbofan Engine Degradation (FD001)

- **Source**: NASA Prognostics Data Repository: available via Kaggle `palbha/cmapss-jet-engine-simulated-data`, files at `data_cmapss/`
- **Size**: `train_FD001.txt`: 20,631 rows, 100 engines; `test_FD001.txt`: 13,096 rows, 100 engines.
- **Granularity**: One row per operational cycle per engine (no fixed time unit).
- **Features**: 3 operational settings + 21 sensor readings (temperatures T2/T24/T30/T50, pressures P2/P15/P30, fan/core speeds Nf/Nc, engine pressure ratio, fuel flow, etc.). 24 features × 6 stats = 144 feature columns.
- **Label derivation**: No per-row binary label is provided. RUL (Remaining Useful Life) is computed as `max_cycle_per_engine - current_cycle`. Label = 1 if RUL ≤ H = 30 cycles ("engine will fail within 30 cycles").
- **Class imbalance**: 15.03% positive in training; 2.54% in test.
- **Train/test split**: Official file split (different engine sets). Training uses `train_FD001.txt` (full run-to-failure); test uses `test_FD001.txt` with ground-truth RUL from `RUL_FD001.txt`.
- **Key engineering detail**: Rolling statistics are computed **per engine** (`group_col='unit_number'`) to prevent mixing sensor degradation profiles across different engines. This is essential for correctness: without per-engine grouping, the rolling window would span the boundary between two engines and produce meaningless features.

---

## 3. Modeling Choices

### 3.1 Algorithms: XGBoost, RandomForest, Ensemble, LSTM, TCN

Five classifiers are evaluated:

| Classifier | Type | Input | Notes |
|---|---|---|---|
| **XGBoost** | Tree ensemble | Rolling stats | `scale_pos_weight` for imbalance |
| **RandomForest** | Tree ensemble | Rolling stats | `class_weight='balanced'` |
| **Ensemble** | Average of RF + XGBoost | Rolling stats | Combines both tree models |
| **LSTM** | Recurrent neural net | Raw W-step sequences | PyTorch; StandardScaler normalization; learns temporal patterns directly |
| **TCN** | Temporal ConvNet | Raw W-step sequences | PyTorch; StandardScaler normalization; dilated convolutions over time |

Tree ensembles operate on the CPU; LSTM and TCN use GPU when available. Tree models use rolling-statistics features; LSTM and TCN consume the raw window of metric values per channel. **Raw sequences are normalized with StandardScaler** (fit on train, transform train/val/test) before feeding to PyTorch - critical for C-MAPSS and SWaT where sensor scales differ wildly (e.g. core speeds ~8000 vs pressure ratios ~1.0); unscaled data causes gradient saturation and model collapse.

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
| SWaT | 60 steps | 60 steps | stride-dependent (5 sec at stride=5) | W/H in steps; effective duration depends on stride |
| C-MAPSS | 30 cycles | 30 cycles | 1 cycle | 30-cycle degradation context; label = 1 if RUL ≤ 30 cycles |

### 3.3 Feature Engineering

For each of the *C* metric columns available (C = 29 for IBM, C = 29 for synthetic), four features are computed per column at each time step *t*:

1. `{col}_mean24`: rolling mean over the last 24 steps
2. `{col}_std24`: rolling standard deviation
3. `{col}_max24`: rolling maximum
4. `{col}_last`: raw value at *t*

This yields **4 × C** features per sample. Rolling statistics are preferred over raw window flattening (which would yield **W × C = 24 × 29 = 696** features) because:
- They are more compact and less prone to overfitting on a ~39k-row dataset.
- They capture the trend (rising mean, growing variance) that typically precedes an incident rather than exact step-by-step values.
- They are robust to short-term jitter in individual 5-minute intervals.

### 3.4 Class-Imbalance Handling

The datasets are heavily imbalanced (~0.4-2.5% positive in the test sets, depending on dataset and masking). Two complementary strategies are used:

- **XGBoost**: `scale_pos_weight = n_negative / n_positive`: upweights the minority class during gradient computation.
- **RandomForest**: `class_weight='balanced'`: inversely weights each class proportional to its frequency.

In both cases, the raw probability score is used at inference time, and the operating threshold is tuned post-hoc rather than hard-coded to 0.5.

### 3.5 Label Mode

Two label strategies are implemented (controlled by `--label-mode`):

- **`onset`** (strict): label=1 if an anomaly *starts* in (t, t+H×5min]. With 25 IBM anomaly windows, this yields only 139 positives (0.35%): far too sparse for a train set of 31k samples.
- **`active`** (default): label=1 if any anomaly window is *active* (ongoing or about to start) within [t, t+H×5min]. This yields 1,096 positives (2.78%) and is operationally equivalent: it answers "will there be an anomaly active in the next 30 minutes?". For IBM, Synthetic, and SWaT, the reported results use `active` labels **combined with `--mask-active-only`**, which excludes rows where an anomaly is already active at time *t*, so evaluation strictly measures onset prediction rather than continuation. C-MAPSS uses RUL-derived labels and is not masked.

---

## 4. Evaluation Setup

### 4.1 Train / Validation / Test Split

A **64% train / 16% validation / 20% test** chronological split is used. The validation set is used exclusively for threshold calibration: the operating threshold is tuned on validation to achieve ≥80% recall with maximum precision, then applied to the test set. This avoids **threshold data leakage** (oracle evaluation) that would occur if the threshold were optimized on the test set itself.

| Split | Size (IBM full dataset) | Purpose |
|---|---|---|
| Train | ~25,194 steps | Model fitting |
| Validation | ~6,298 steps | Threshold calibration (≥80% recall) |
| Test | ~7,873 steps (→ 7,682 with masking) | Final evaluation only |

The reported IBM test size (7,682) is lower than the nominal 20% split (~7,873) because `--mask-active-only` excludes rows where an anomaly is already active, and rolling-window warm-up drops initial rows. For C-MAPSS, the official train file is split 80/20 per-engine (last 20% of each engine's cycles → validation); the test file remains the held-out test set.

### 4.2 Metrics

| Metric | Why it matters for alerting |
|---|---|
| **Recall** | Missing a real incident (false negative) is operationally the worst outcome: an alert system that misses incidents is worse than useless. |
| **Precision** | Excessive false alarms cause alert fatigue, eroding operator trust. |
| **F1** | Harmonic mean; used as a single summary metric. |
| **ROC-AUC** | Threshold-independent ranking quality. |
| **PR-AUC** | More informative than ROC-AUC under class imbalance. |

### 4.3 Alert Threshold Tuning

Rather than using a fixed 0.5 threshold, the operating point is calibrated as follows:

1. **Default (0.5)**: Baseline behaviour.
2. **Tuned (≥80% recall)**: A threshold sweep is performed on the **validation set** (not the test set). The highest-precision threshold that achieves at least 80% recall on validation is selected, then applied to the test set for reporting. This prevents data leakage: the test set is never used for threshold selection.

---

## 5. Evaluation Results

Results follow the protocol described in Section 3.5: IBM, Synthetic, and SWaT use `active` labels with `--mask-active-only`; C-MAPSS uses RUL-derived labels (no masking). Test set sizes (current run): IBM = 7,682 (30 positives, 0.4%); Synthetic = 1,894 (30 positives, 1.6%); SWaT = 55,484 (1,073 positives, 1.9%); C-MAPSS = 13,096 (332 positives, 2.5%).

### 5.1 IBM Dataset

| Classifier | ROC-AUC | PR-AUC | F1 @ 0.5 | Tuned Recall | Tuned Precision | Tuned F1 |
|---|---|---|---|---|---|---|
| **Ensemble** | **0.697** | 0.011 | 0.00% | 100.00% | 0.40% | 0.81% |
| XGBoost | 0.692 | 0.015 | 0.00% | 100.00% | 0.42% | 0.83% |
| RandomForest | 0.691 | 0.011 | 0.00% | 100.00% | 0.39% | 0.78% |
| TCN | 0.623 | 0.005 | 0.00% | 100.00% | 0.39% | 0.78% |
| LSTM | 0.566 | 0.004 | 0.00% | 100.00% | 0.42% | 0.84% |

Tree models and ensemble achieve the highest ROC-AUC (~0.69-0.70) but still barely above chance. All models predict zero positives at threshold 0.5. At tuned threshold, 100% recall is achieved but precision remains ~0.4%: thousands of false alarms for 30 true positives. Features are not predictive of IBM anomalies.

---

### 5.2 Synthetic Dataset

| Classifier | ROC-AUC | PR-AUC | F1 @ 0.5 | Tuned Recall | Tuned Precision | Tuned F1 |
|---|---|---|---|---|---|---|
| LSTM | **0.993** | 0.831 | 74.51% | 56.67% | 100.00% | 72.34% |
| TCN | 0.992 | 0.878 | 82.35% | 80.00% | 100.00% | **88.89%** |
| RandomForest | 0.980 | 0.942 | 82.35% | 63.33% | 100.00% | 77.55% |
| Ensemble | 0.967 | 0.941 | 84.62% | 73.33% | 100.00% | 84.62% |
| XGBoost | 0.924 | 0.802 | 84.62% | 73.33% | 100.00% | 84.62% |

All classifiers excel. **TCN achieves the best tuned F1 (88.89%)** with 80% recall and 100% precision. XGBoost and ensemble reach 84.62% F1 at default threshold with zero false positives. The formulation is validated: when the signal is present, models detect it effectively.

---

### 5.3 SWaT Dataset

| Classifier | ROC-AUC | PR-AUC | F1 @ 0.5 | Tuned Recall | Tuned Precision | Tuned F1 |
|---|---|---|---|---|---|---|
| LSTM | 0.565 | 0.032 | 0.00% | 87.42% | 1.86% | 3.64% |
| XGBoost | 0.500 | 0.019 | 0.00% | 100.00% | 1.93% | 3.79% |
| RandomForest | 0.500 | 0.019 | 3.79% | 100.00% | 1.93% | 3.79% |
| Ensemble | 0.500 | 0.019 | 3.79% | 100.00% | 1.93% | 3.79% |
| TCN | 0.500 | 0.019 | 0.00% | 100.00% | 1.93% | 3.79% |

**All models perform at or near chance** (ROC-AUC 0.50). Tree models and TCN achieve 100% recall at tuned threshold but with ~1.9% precision: no discriminative signal. LSTM has slightly higher ROC-AUC (0.57) but still poor. The current SWaT split (stride=5) or preprocessing yields minimal learnable signal; the train/test distribution may differ substantially.

---

### 5.4 C-MAPSS FD001

| Classifier | ROC-AUC | PR-AUC | F1 @ 0.5 | Tuned Recall | Tuned Precision | Tuned F1 |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.985** | **0.654** | 10.76% | 84.34% | 44.30% | **58.09%** |
| Ensemble | 0.981 | 0.574 | 4.12% | 79.52% | 37.24% | 50.72% |
| RandomForest | 0.955 | 0.492 | 0.00% | 77.11% | 36.99% | 50.00% |
| TCN | 0.627 | 0.037 | 0.58% | 100.00% | 2.54% | 4.94% |
| LSTM | 0.562 | 0.028 | 3.25% | 100.00% | 2.57% | 5.01% |

**Tree-based models dominate C-MAPSS.** XGBoost achieves the best tuned F1 (58.09%) with ROC-AUC 0.985 and PR-AUC 0.65. RF and ensemble reach ~50% tuned F1. LSTM and TCN collapse (ROC-AUC 0.56-0.63, PR-AUC ~0.03): they predict almost everything positive. Per-engine rolling statistics are the right abstraction for monotonic engine degradation.

---

### 5.5 Full Summary Table

| Dataset | Classifier | F1 @ 0.5 | Tuned F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| IBM | Ensemble (best) | 0.00% | 0.81% | 0.697 | 0.011 |
| IBM | XGBoost | 0.00% | 0.83% | 0.692 | 0.015 |
| Synthetic | **TCN (best)** | 82.35% | **88.89%** | 0.992 | 0.878 |
| Synthetic | XGBoost | 84.62% | 84.62% | 0.924 | 0.802 |
| SWaT | LSTM (best) | 0.00% | 3.64% | 0.565 | 0.032 |
| SWaT | XGBoost | 0.00% | 3.79% | 0.500 | 0.019 |
| C-MAPSS | **XGBoost (best)** | 10.76% | **58.09%** | **0.985** | **0.654** |
| C-MAPSS | Ensemble | 4.12% | 50.72% | 0.981 | 0.574 |

---

## 6. Analysis & Discussion

### 6.1 IBM Dataset Results

Tree models and ensemble achieve ROC-AUC ~0.69-0.70: modestly above chance: while LSTM and TCN are worse (0.56-0.62). At threshold 0.5, all models predict zero positives. At tuned threshold, 100% recall is achieved but precision remains ~0.4%, yielding thousands of false alarms for 30 true positives.

**Root cause: features are not predictive of future IBM anomalies:**

The IBM anomaly windows (a1-a25) capture infrastructure-level incidents identified via Issue Tracker, Instant Messenger, and internal test logs. These events are not necessarily preceded by detectable changes in aggregate HTTP 5xx rates or latency averages. The aggregate metrics in `reduced_features_full.parquet` capture request volume and error rates at a global level, while the IBM anomalies may be driven by network-layer or infrastructure events not visible in these aggregates.

**Conclusion for IBM:** The sliding-window formulation is implemented correctly. The failure is attributable to a feature-label mismatch. More fine-grained features would be needed for meaningful predictions.

### 6.2 Synthetic Dataset Results

All five classifiers achieve strong performance. **TCN achieves the best tuned F1 (88.89%)** with 80% recall and 100% precision. XGBoost and ensemble reach 84.62% F1 at default threshold with zero false positives. LSTM has the highest ROC-AUC (0.99) but lower tuned F1 due to recall-precision trade-off. The fact that TCN wins here validates that temporal convolutional networks are highly effective when a true sequential precursor (the ramp-up) exists. The formulation is validated: a 30-minute horizon is sufficient when the signal is present.

### 6.3 SWaT Dataset Results

**All models perform at or near chance** (ROC-AUC 0.50). LSTM has slightly higher ROC-AUC (0.57) but still poor. Tree models, ensemble, and TCN achieve 100% recall at tuned threshold but with ~1.9% precision: no discriminative signal.

The collapse of performance compared to earlier, unmasked runs *indicates* that the models were previously relying on active attack signatures. When forced to predict the onset of an attack from healthy precursor data, the models fail. This *suggests* SWaT attacks may be instantaneous or lack a measurable 15-minute precursor in the sensor data, though train/test distribution shift and preprocessing choices (e.g. stride) are alternative explanations. The current configuration (stride=5, 55k test samples, 1.9% positive) yields minimal learnable signal.

### 6.4 C-MAPSS Dataset Results

**Tree-based models dominate C-MAPSS.** XGBoost achieves the best tuned F1 (58.09%) with ROC-AUC 0.985 and PR-AUC 0.65. RF and ensemble reach ~50% tuned F1. The tabular rolling-statistics representation is well-suited to gradual, monotonic engine degradation.

**LSTM and TCN collapse** (ROC-AUC 0.56-0.63, PR-AUC ~0.03): they predict almost everything positive, suggesting they do not learn the degradation pattern from raw sequences. Despite StandardScaler normalization, out-of-the-box LSTMs and TCNs struggle compared to XGBoost given rolling stats. Per-engine rolling statistics appear to be the right abstraction; raw sequences may add noise or require different architectures.

### 6.5 Comparison: Tree Models vs. Neural Models

| Dataset | Best model | Why |
|---|---|---|
| IBM | Ensemble (still poor) | ROC-AUC ~0.70; no model succeeds |
| Synthetic | **TCN** | Best tuned F1 (89%); all models work |
| SWaT | LSTM (still poor) | ROC-AUC 0.57; all near chance |
| C-MAPSS | **XGBoost** | Rolling stats suit monotonic degradation; LSTM/TCN collapse |

**Takeaways:**
- **Synthetic**: Neural and tree models both excel; TCN best; formulation validated.
- **SWaT**: All models fail; configuration or distribution shift limits learnability.
- **C-MAPSS**: Tree models on rolling stats are optimal; neural models collapse.
- **IBM**: All models fail; feature quality is the bottleneck, not model choice.

### 6.6 Feature Importances (Tree Models)

**Synthetic XGBoost (top 3):** `rate_2xx_last`, `count_5xx_last`, `rate_5xx_last`: current-step error and success rates dominate, confirming the model detects the onset signal.

**IBM XGBoost (top 3):** `dc5_total_std24`, `dc5_total_max24`, `dc1_total_mean24`: traffic volume variance dominates. The even importance distribution suggests no clear discriminative feature: consistent with poor ROC-AUC.

**C-MAPSS XGBoost:** Sensors in the T30/T50 range (stage temperatures) and NRc (corrected core speed ratio) degrade monotonically in HPC faults; rolling mean and delta features capture degradation level and rate.

**SWaT:** Level sensors (LIT) and flow meters (FIT) should dominate since attacks target pumps and valves; delta features capture sudden step-changes. In the current run, all models perform at chance: no clear feature dominance was observed.

---

## 7. Limitations

1. **Single train-test split**: A single chronological split is used. For a production system, walk-forward cross-validation would give a more robust performance estimate.
2. **Static window sizes**: W and H are fixed. A real system would sweep these hyperparameters.
3. **No temporal context across the split boundary**: The model is not fine-tuned online; it does not adapt to concept drift in the live telemetry stream.
4. **Label granularity and feature mismatch**: The 25 IBM anomaly windows are coarse-grained infrastructure events identified via monitoring tools, not necessarily correlated with aggregate HTTP error rates. Some windows overlap (a3/a4/a5 all start within 6 minutes on 2024-02-12), inflating the apparent positive count near those windows. More critically, the available features are not predictive of these events: the fundamental requirement for any classifier to succeed.
5. **SWaT temporal distribution shift**: The merged SWaT file concatenates normal operation (7 days) followed by attack scenarios (4 days). A strict 80/20 chronological split puts almost all attacks in the test set, creating a severe train/test distribution mismatch. In a real deployment, the model would be trained on a balanced historical dataset that includes past attack episodes.
6. **Synthetic data simplifications**: The synthetic generator uses independent Poisson arrivals and does not model correlations between datacenters, time-of-day patterns, or multi-step dependencies that exist in real production traffic.

---

## 8. Adapting to a Real Alerting System

1. **Streaming inference**: At each new telemetry batch (every 5 min), recompute the rolling-stats feature vector for the current interval and query the trained model for a probability score.
2. **Threshold calibration**: Tune the alert threshold on a held-out validation window, not on the test set, to avoid overfitting the operating point.
3. **Online learning / periodic retraining**: Retrain the model on a rolling window of recent data (e.g., last 30 days) to adapt to evolving traffic patterns.
4. **Multi-horizon ensemble**: Train separate models for H = 1, 3, 6, 12 steps and combine their scores to give operators different lead times.
5. **Alarm deduplication**: Suppress repeated alerts within the same incident window to avoid flooding the on-call queue.

---

## 9. Methodology Updates (Post-Review)

Following an expert review, the following fixes were implemented to improve scientific rigor:

1. **LSTM/TCN normalization**: Raw sequences fed to PyTorch models are now scaled with `StandardScaler` (fit on train, transform train/val/test). C-MAPSS and SWaT sensors have wildly divergent scales; unscaled data caused gradient saturation and LSTM collapse (e.g. ROC-AUC 0.28 on SWaT, 0.61 on C-MAPSS). The scaler is saved with the model for inference.

2. **Threshold data leakage fix**: The "Tuned F1" was previously computed by sweeping thresholds on the test set (oracle evaluation). The pipeline now uses a 64/16/20 train/validation/test split. The threshold is tuned on validation to achieve ≥80% recall with maximum precision, then applied strictly to the test set. The `optimal_threshold` is saved in the predictions `.npz` for evaluation.

3. **Onset-only option** (`--mask-active-only`): For true predictive evaluation, the model can be restricted to samples where no incident is active at time *t*. This excludes rows where an anomaly is already ongoing, forcing the model to predict *onset* rather than *continuation*. Available for IBM, Synthetic, and SWaT. C-MAPSS uses RUL-derived labels and is not affected.

---

## 10. Test Suite

A pytest-based test suite was added to ensure correctness of the pipeline and to support future refactoring. The rationale is threefold:

1. **Regression prevention**: The pipeline involves multiple data loaders, feature engineering (rolling stats, per-group windows), and model training paths. Tests catch unintended breakage when changing window logic, label semantics, or classifier wiring.

2. **Documentation of expected behaviour**: Unit tests for `make_labels`, `make_features`, `make_raw_windows`, and `make_mask_normal_at_t` encode the intended semantics (e.g. onset vs active mode, per-engine grouping for C-MAPSS). They serve as executable specifications for the sliding-window formulation.

3. **CI-ready, data-independent coverage**: Tests use synthetic data or small fixtures only. No IBM, SWaT, or C-MAPSS files are required, so the suite runs in any environment (e.g. CI) without downloading datasets. Integration smoke tests run the pipeline on synthetic data with `max_rows=300-400` to verify end-to-end execution.

**Structure**: `tests/test_windows.py` (feature extraction and labels), `tests/test_synthetic.py` (synthetic generator), `tests/test_train.py` (build_model, ensemble, `run()` on synthetic), `tests/test_evaluate.py` (evaluate_file with mocked .npz), `tests/test_pipeline.py` (run_pipeline.py smoke tests). Run with `python -m pytest tests/ -v`.
