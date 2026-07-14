"""
evaluate.py
-----------
Phase 8-9: Benchmark AFSE against standard baselines, plus the research-grade
extras: stability analysis, ablation study, statistical significance testing.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from train import train_on_subset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------
# Baseline feature-selection methods (compared against AFSE)
# ---------------------------------------------------------------------

def baseline_mutual_information(X: pd.DataFrame, y: pd.Series, k_pct: float) -> list:
    X_filled = X.fillna(X.median(numeric_only=True))
    mi = mutual_info_classif(X_filled, y, random_state=42)
    order = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    k = max(1, int(round(len(order) * k_pct)))
    return list(order.index[:k])


def baseline_lasso(X: pd.DataFrame, y: pd.Series, k_pct: float) -> list:
    X_filled = X.fillna(X.median(numeric_only=True))
    X_scaled = StandardScaler().fit_transform(X_filled)
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=42, max_iter=2000)
    lasso.fit(X_scaled, y)
    coefs = np.abs(lasso.coef_).mean(axis=0) if lasso.coef_.ndim > 1 else np.abs(lasso.coef_[0])
    order = pd.Series(coefs, index=X.columns).sort_values(ascending=False)
    k = max(1, int(round(len(order) * k_pct)))
    return list(order.index[:k])


def baseline_random_forest(X: pd.DataFrame, y: pd.Series, k_pct: float) -> list:
    X_filled = X.fillna(X.median(numeric_only=True))
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_filled, y)
    order = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    k = max(1, int(round(len(order) * k_pct)))
    return list(order.index[:k])


def baseline_rfe(X: pd.DataFrame, y: pd.Series, k_pct: float) -> list:
    X_filled = X.fillna(X.median(numeric_only=True))
    k = max(1, int(round(X.shape[1] * k_pct)))
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    step = max(1, X.shape[1] // 50)  # keep RFE tractable on high-dim sets like Madelon
    rfe = RFE(estimator, n_features_to_select=k, step=step)
    rfe.fit(X_filled, y)
    return list(X.columns[rfe.support_])


BASELINES = {
    "RFE": baseline_rfe,
    "MutualInformation": baseline_mutual_information,
    "LASSO": baseline_lasso,
    "RandomForestImportance": baseline_random_forest,
}


# ---------------------------------------------------------------------
# Metric computation from a train.py TrainingRun
# ---------------------------------------------------------------------

@dataclass
class EvalResult:
    method: str
    model: str
    n_features_selected: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    runtime_sec: float
    peak_memory_kb: float


def evaluate_run(run) -> list:
    results = []
    n_classes = run.y_train.nunique()
    avg = "binary" if n_classes == 2 else "macro"
    for tm in run.trained_models:
        preds = tm.estimator.predict(run.X_test)
        try:
            probs = tm.estimator.predict_proba(run.X_test)
            roc = roc_auc_score(run.y_test, probs[:, 1]) if probs.shape[1] == 2 \
                else roc_auc_score(run.y_test, probs, multi_class="ovr")
        except Exception:
            roc = float("nan")
        results.append(EvalResult(
            method=run.method, model=tm.name, n_features_selected=len(run.features),
            accuracy=accuracy_score(run.y_test, preds),
            precision=precision_score(run.y_test, preds, average=avg, zero_division=0),
            recall=recall_score(run.y_test, preds, average=avg, zero_division=0),
            f1=f1_score(run.y_test, preds, average=avg, zero_division=0),
            roc_auc=roc, runtime_sec=tm.runtime_sec, peak_memory_kb=tm.peak_memory_kb,
        ))
    return results


def results_to_dataframe(results: list) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def benchmark_all_methods(X: pd.DataFrame, y: pd.Series, afse_features: list, k_pct: float) -> pd.DataFrame:
    """Runs AFSE's already-selected features plus every baseline at the same k_pct, trains all
    models on each, and returns one combined results table."""
    all_results = []

    afse_run = train_on_subset(X, y, afse_features, method="AFSE")
    all_results += evaluate_run(afse_run)

    for name, fn in BASELINES.items():
        features = fn(X, y, k_pct)
        run = train_on_subset(X, y, features, method=name)
        all_results += evaluate_run(run)

    return results_to_dataframe(all_results)


# ---------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------

def stability_analysis(X: pd.DataFrame, y: pd.Series, selector_fn, n_splits: int = 5,
                        n_repeats: int = 10, k_pct: float = 0.10) -> pd.DataFrame:
    """selector_fn(X_fold, y_fold, k_pct) -> list[str]. Returns per-feature selection frequency."""
    counts = pd.Series(0, index=X.columns, dtype=int)
    total_runs = 0
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        for train_idx, _ in skf.split(X, y):
            X_fold, y_fold = X.iloc[train_idx], y.iloc[train_idx]
            selected = selector_fn(X_fold, y_fold, k_pct)
            counts[selected] += 1
            total_runs += 1
    freq = (counts / total_runs).sort_values(ascending=False)
    return pd.DataFrame({"feature": freq.index, "selection_frequency": freq.values, "runs": total_runs})


# ---------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------

def wilcoxon_test(scores_a: list, scores_b: list) -> dict:
    scores_a, scores_b = np.array(scores_a), np.array(scores_b)
    if len(scores_a) < 6:
        return {"note": "Wilcoxon needs >=6 paired samples for a meaningful p-value.", "n": len(scores_a)}
    stat, p = wilcoxon(scores_a, scores_b)
    return {"statistic": float(stat), "p_value": float(p), "significant_at_0.05": bool(p < 0.05), "n": len(scores_a)}


# ---------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------

def ablation_variants(weights_full) -> dict:
    """AFSE component ablation: MI-only, MI+RF, MI+RF+LASSO, Full."""
    from loss import AdaptiveWeights

    def renorm(mi, rf, lasso, var):
        total = mi + rf + lasso + var
        total = total if total > 0 else 1.0
        return AdaptiveWeights(mi / total, rf / total, lasso / total, var / total, raw_scores={}, derived_signals={})

    return {
        "AFSE-Base (MI only)": renorm(weights_full.w_mi, 0, 0, 0),
        "AFSE-R (MI+RF)": renorm(weights_full.w_mi, weights_full.w_rf, 0, 0),
        "AFSE-RL (MI+RF+LASSO)": renorm(weights_full.w_mi, weights_full.w_rf, weights_full.w_lasso, 0),
        "AFSE-Full (all modules)": weights_full,
    }


if __name__ == "__main__":
    from dataset import load_dataset
    from model import AFSEModel

    bundle = load_dataset("breast_cancer")
    afse = AFSEModel().fit(bundle.X, bundle.y, name=bundle.name)
    features = afse.select(0.10)

    table = benchmark_all_methods(bundle.X, bundle.y, features, k_pct=0.10)
    print(table[["method", "model", "accuracy", "f1", "roc_auc", "runtime_sec"]]
          .sort_values(["method", "model"]).to_string(index=False))
