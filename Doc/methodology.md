# AFSE — Methodology

This document gives the mathematical formulation, pseudocode, architecture,
and complexity analysis behind AFSE, at the level of detail expected for an
undergraduate research write-up.

## 1. Problem statement

Given a dataset `D = (X, y)` with `n` samples and `d` features, existing
feature-selection methods each have a known failure mode:

| Method | Limitation |
|---|---|
| RFE | Accurate but computationally expensive; degrades on high-dimensional data |
| Mutual Information | Ignores redundancy between selected features |
| LASSO | Struggles with groups of correlated variables |
| Random Forest importance | Can be biased toward high-cardinality / high-variance features |

**Research question:** can a system automatically choose and combine
feature-selection strategies based on the measurable characteristics of the
dataset in front of it, rather than applying one fixed method to every
dataset?

## 2. Meta-Feature Vector

Every dataset is profiled (`src/dataset.py::profile_dataset`) into a
meta-feature vector `M`:

```
M = [ n_samples, n_features, feat_sample_ratio, missing_pct, duplicate_pct,
      mean_abs_corr, max_abs_corr, mean_variance, mean_entropy,
      class_imbalance, sparsity, noise_estimate ]
```

`feat_sample_ratio = n_features / n_samples` is the single strongest signal
for whether a dataset behaves like a high-dimensional problem (Madelon:
0.25) or a low-dimensional one (Breast Cancer: 0.05).

## 3. Adaptive weight generation: `w = f(M)`

Rather than hard thresholds (`if features > 300: use MI`), each meta-feature
is first mapped to a derived signal in `[0, 1]` using either a sigmoid ramp
(replacing a hard cutoff with a soft one) or a direct normalization:

```
HighDim      = sigmoid(feat_sample_ratio, midpoint=0.15)     # soft "features > 300"
Correlation  = min(1, mean_abs_corr / 0.6)                   # soft "corr > 0.8"
Noise        = min(1, noise_estimate)
Entropy      = mean_entropy
Dense        = 1 - sparsity
LowDim       = 1 - HighDim
SmallSample  = sigmoid(-n_samples, midpoint=-1000, steepness=0.004)
NonLinearity = 0.5*mean_variance + 0.5*Entropy
```

Each candidate strategy then gets a linear score over these signals:

```
Score_MI       = 0.5·HighDim + 0.3·Noise + 0.2·Entropy
Score_RF       = 0.45·Noise  + 0.35·NonLinearity + 0.2·LowDim
Score_LASSO    = 0.5·Correlation + 0.3·Dense + 0.2·LowDim
Score_Variance = 0.6·(1 - Correlation) + 0.4·SmallSample
```

Scores are normalized into weights that sum to 1:

```
w_i = Score_i / Σ_j Score_j
```

This is the direct answer to "why 300? why 0.8?": every coefficient above
is a named constant in `src/loss.py::COEF`, not a magic number buried in an
if-statement, and the same function is applied identically to every dataset.

## 4. Hybrid feature ranking with redundancy elimination

For each feature `i`, four component importances are computed and min-max
normalized to `[0, 1]`:

- `MI_i` — mutual information with the target
- `RF_i` — Random Forest feature importance
- `LASSO_i` — |coefficient| from L1-penalized logistic regression
- `Variance_i` — feature variance

The base score is the adaptive-weighted combination:

```
BaseScore_i = w_MI·MI_i + w_RF·RF_i + w_LASSO·LASSO_i + w_Var·Variance_i
```

A redundancy penalty discourages selecting features that duplicate
information already captured by other high-scoring features:

```
Redundancy_i = normalize( Σ_j |Corr(i,j)| · BaseScore_j  -  BaseScore_i )

FinalScore_i = BaseScore_i - λ · Redundancy_i        (λ = 0.15 by default)
```

Features are ranked by `FinalScore` descending; the top-k% are selected.

## 5. Architecture

```
Dataset
   │
   ▼
Dataset Profiler            (dataset.py::profile_dataset)     → M
   │
   ▼
Adaptive Weight Generator   (loss.py::compute_weights)        → w = f(M)
   │
   ▼
Component Importance Scores (loss.py::component_scores)       → MI, RF, LASSO, Variance
   │
   ▼
Hybrid Ranking + Redundancy (loss.py::final_score)             → FinalScore per feature
   │
   ▼
Top-k% Selection             (model.py::AFSEModel.select)
   │
   ▼
Model Training                (train.py::train_on_subset)      → LR, RF, SVM, XGBoost*
   │
   ▼
Evaluation                    (evaluate.py)                    → benchmark, stability,
   │                                                              ablation, significance
   ▼
Explainability                (model.py::AFSEModel.explain)    → per-feature reasons
```

## 6. Complexity analysis

| Component | Complexity | Notes |
|---|---|---|
| Mutual Information | O(n·d) | one pass per feature |
| Random Forest importance | O(t·n·log(n)·d) | `t` trees, dominant cost for large `d` |
| LASSO | O(d²·n) worst case (coordinate descent) | faster in practice with sparsity |
| RFE (baseline) | O(d²·n) or worse | iterative refit at every elimination step — the reason it's the slowest baseline in `Results/*_benchmark.csv` |
| AFSE overall | dominated by the Random Forest importance term | same order as running RF once, plus one MI pass and one LASSO fit |

AFSE runs each of MI, RF, and LASSO **once** per dataset (not once per
elimination step, unlike RFE), which is why it consistently has lower
runtime than RFE in the benchmark results while matching or beating its
accuracy — see `Figures/*_runtime_comparison.png`.

## 7. Ablation study design

To isolate which component actually drives performance (`evaluate.py::ablation_variants`):

| Variant | Components active |
|---|---|
| AFSE-Base | MI only |
| AFSE-R | MI + RF |
| AFSE-RL | MI + RF + LASSO |
| AFSE-Full | MI + RF + LASSO + Variance + redundancy penalty |

Each variant's weights are renormalized over only its active components, so
the comparison isolates the *contribution* of each addition rather than
just diluting the weight of the others.

## 8. Statistical validation

Because a single train/test split's accuracy difference can be noise, AFSE
is compared against each baseline using a paired Wilcoxon signed-rank test
(`evaluate.py::wilcoxon_test`) over per-model accuracy pairs (Logistic
Regression, Random Forest, SVM, XGBoost/GradientBoosting run on the same
split). This tests whether AFSE's improvement holds consistently across
model types, not just on average.

## 9. Stability analysis

A feature selected once in a lucky train/test split is not the same as a
feature selected reliably. `evaluate.py::stability_analysis` reruns the
selector across repeated stratified k-fold splits and reports each
feature's selection frequency — a feature selected 48/50 times is reported
as stable; one selected 2/50 times is not, regardless of its score on any
single run.

## 10. Future scope

- **Meta-learning**: instead of the fixed named scoring function in §3,
  learn `w = f(M)` from a library of datasets so the weighting itself
  improves with more data.
- **Reinforcement learning**: treat feature selection as a sequential
  decision process and learn a selection policy directly.
- **Streaming data support**: adapt the profiler and ranking to
  incrementally-arriving data rather than a fixed batch.
- **AutoML integration**: use AFSE's output as the feature-selection stage
  of a larger automated pipeline search.
