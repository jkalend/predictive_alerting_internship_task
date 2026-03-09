# model comparison notes

## setup
- sliding window → predict incident in next H steps from W-step rolling stats
- features: rolling mean/std/max + deltas per metric
- RF + XGBoost, datasets: IBM, Synthetic, SWaT, C-MAPSS

## might be better

**LightGBM / CatBoost** — same family as xgb, often faster or better on tabular. SWaT: xgb gets 0.40 ROC-AUC vs RF 0.73, so worth a shot here

**LSTM/TCN** — we're only using rolling summaries, not raw sequence. could try raw W-step window. but IBM/SWaT are bad anyway, probably features/labels not the model

**RF + Xgb ensemble** — SWaT RF beats xgb, maybe avg probs helps

**TabNet etc** — attention over features, transformers for tabular. probably overkill for 31k–96k rows

## probably worse

- **LogReg** — baseline only, linear won't cut it
- **k-NN** — 144–306 features, curse of dimensionality
- **Naive Bayes** — features are correlated, independence assumption breaks
- **untuned MLP** — trees usually beat these on tabular without heavy tuning

## by dataset
| dataset | note |
|---------|------|
| IBM | both bad — fix features/labels first |
| Synthetic | fine as is |
| SWaT | RF > xgb — try LightGBM or ensemble |
| C-MAPSS | fine as is |

## tl;dr
RF + XGBoost make sense for this. Try LightGBM/CatBoost/ensemble on SWaT. Skip logreg/knn/nb/mlp. Neural seq models = more complexity, might not fix the real bottlenecks.
