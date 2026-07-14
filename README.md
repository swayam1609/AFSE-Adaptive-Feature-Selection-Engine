# AFSE — Explainability-Guided Adaptive Feature Selection

A meta-feature-driven adaptive feature selection framework for high-dimensional
classification. Instead of applying one fixed feature-selection algorithm to
every dataset, AFSE profiles the dataset first, uses a transparent scoring
function to decide how much to trust Mutual Information vs. Random Forest
importance vs. LASSO vs. Variance for *this specific dataset*, combines them
into a single ranking with a redundancy penalty, and explains why each
feature was chosen.

> No single feature-selection method consistently outperforms the others
> across all datasets. AFSE adapts its strategy to each dataset's
> dimensionality, correlation structure, and noise instead of picking one
> method and hoping it generalizes.

## What makes this different from "I ran 4 feature selectors and compared them"

A fixed rule like `if features > 300: use Mutual Information` invites the
obvious question: *why 300?* AFSE replaces every hard threshold with a named,
inspectable scoring function over a **Meta-Feature Vector** `M` describing the
dataset (size, dimensionality ratio, correlation, entropy, sparsity, noise,
class imbalance). The adaptive weights `w = f(M)` are computed the same way
for every dataset — nothing is hand-tuned per dataset.

```
Score_MI       = 0.5·HighDim + 0.3·Noise   + 0.2·Entropy
Score_RF       = 0.45·Noise  + 0.35·NonLinearity + 0.2·LowDim
Score_LASSO    = 0.5·Correlation + 0.3·Dense + 0.2·LowDim
Score_Variance = 0.6·(1−Correlation) + 0.4·SmallSample

w_i = Score_i / Σ Score        # adaptive weights, sum to 1
```

Features are then ranked by:

```
FinalScore_i = w1·MI_i + w2·RF_i + w3·LASSO_i + w4·Variance_i − λ·Redundancy_i
```

where the redundancy term penalizes features that are highly correlated with
other already-important features — a direct fix for Mutual Information's
known tendency to select redundant variables.

## Pipeline

```
Dataset
   │
   ▼
Dataset Profiler  (dataset.py)         → Meta-Feature Vector M
   │
   ▼
Adaptive Weight Generator  (loss.py)   → w = f(M)
   │
   ▼
Hybrid Feature Ranking + Redundancy Elimination  (loss.py, model.py)
   │
   ▼
Feature Selection (top-k%)
   │
   ▼
Model Training  (train.py)             → LogisticRegression, RandomForest, SVM, XGBoost*
   │
   ▼
Evaluation  (evaluate.py)              → benchmark vs RFE/MI/LASSO/RF-importance,
   │                                      stability analysis, ablation study,
   │                                      Wilcoxon significance test
   ▼
Explainability  (model.py)             → per-feature reasons, not just scores
```

\* falls back to `GradientBoostingClassifier` automatically if `xgboost` isn't installed.

## Repository structure

```
AFSE-Explainability-Guided-Adaptive-Feature-Selection/
│
├── Data/           # dataset CSVs (Madelon included; see "Datasets" below)
├── Docs/           # write-ups, notes
├── Figures/        # generated plots (accuracy/runtime comparison, feature
│                     importance, correlation heatmap) — created by main.py
├── Model/          # pickled AFSEModel per dataset — created by main.py
├── Notebooks/       # exploratory notebooks
├── Results/        # CSV outputs: benchmark, ranking, stability, ablation,
│                     explanations — created by main.py
│
├── src/
│   ├── dataset.py    # dataset loading + Dataset Profiler (meta-feature extraction)
│   ├── model.py       # AFSEModel (adaptive weights + ranking + explainability),
│   │                    downstream classifier zoo
│   ├── loss.py        # the AFSE objective: w = f(M), component scores, FinalScore
│   ├── train.py       # trains downstream classifiers on a selected feature subset
│   └── evaluate.py    # baseline methods (RFE/MI/LASSO/RF), benchmarking,
│                        stability analysis, ablation study, significance testing
│
├── main.py           # end-to-end pipeline entry point
├── requirements.txt
└── README.md
```

## Datasets

| Dataset | Samples | Features | Source |
|---|---|---|---|
| Breast Cancer Wisconsin | 569 | 30 | `sklearn.datasets`, no download needed |
| Madelon | 2000 | 500 | included in `Data/madelon.csv` |
| Sonar | 208 | 60 | UCI ML Repository — place at `Data/sonar.csv` |
| Parkinson's | 195 | 22 | UCI ML Repository — place at `Data/parkinsons.csv` (label column `status`) |
| Arrhythmia | 452 | 279 | UCI ML Repository — place at `Data/arrhythmia.csv` |

Datasets not present in `Data/` are skipped automatically — `main.py` runs on
whatever is available without any code changes.

## Usage

```bash
pip install -r requirements.txt

# Run every available dataset at top-10% feature selection
python main.py

# Run a single dataset
python main.py --dataset madelon

# Select a different fraction of features
python main.py --dataset breast_cancer --k_pct 0.20
```

Each run prints the dataset profile, adaptive weights, benchmark table,
top stable features, and ablation results to the console, and writes:

- `Figures/<dataset>_accuracy_comparison.png`, `_runtime_comparison.png`,
  `_feature_importance.png`, `_correlation_heatmap.png`
- `Results/<dataset>_benchmark.csv`, `_afse_ranking.csv`, `_stability.csv`,
  `_ablation.csv`, `_explanations.txt`
- `Model/<dataset>_afse_model.pkl`

## Evaluation methodology

- **Baselines**: RFE, plain Mutual Information, plain LASSO, plain Random
  Forest importance — each given the same top-k% budget as AFSE.
- **Stability analysis**: feature selection is rerun across repeated
  stratified k-fold splits; each feature's selection frequency is reported
  (a feature selected in 48/50 runs is far more trustworthy than one
  selected once).
- **Ablation study**: AFSE-Base (MI only) → AFSE-R (+RF) → AFSE-RL (+LASSO)
  → AFSE-Full (+Variance, +redundancy penalty), isolating which component
  actually drives performance.
- **Significance testing**: paired Wilcoxon signed-rank test between AFSE
  and each baseline's per-model accuracy.
- **Metrics**: accuracy, precision, recall, F1, ROC-AUC, runtime, peak
  memory, and number of features selected.

## Future scope

- Learn the meta-feature → weight mapping (`w = f(M)`) directly from data
  across many datasets instead of a fixed named scoring function
  (meta-learning).
- Reinforcement learning for sequential feature selection.
- Streaming / online feature selection for non-stationary data.
- AutoML integration.
