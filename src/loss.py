"""
loss.py
-------
The AFSE objective: every "loss/score" function the model optimizes over
when ranking features.

1. compute_weights(profile)   -> w = f(M)
   Meta-feature-driven adaptive weights (replaces hard-coded thresholds
   like "if features > 300" with a transparent, named scoring function).

2. component_scores(X, y)     -> per-feature MI / RandomForest / LASSO / Variance
   The four raw importance signals that get combined.

3. final_score(components, weights, corr) -> FinalScore
   FinalScore_i = w1*MI_i + w2*RF_i + w3*LASSO_i + w4*Variance_i - lambda * Redundancy_i
   Redundancy_i penalizes features that are highly correlated with other
   already-important features (fixes MI's known tendency to pick redundant
   variables).
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dataset import DatasetProfile

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REDUNDANCY_LAMBDA = 0.15  # strength of the correlation penalty; named, not magic

# Named, inspectable coefficients -- the direct answer to "why 300? why 0.8?"
COEF = {
    "MI":       {"high_dim": 0.5, "noise": 0.3, "entropy": 0.2},
    "RF":       {"noise": 0.45, "nonlinearity": 0.35, "low_dim": 0.2},
    "LASSO":    {"correlation": 0.5, "dense": 0.3, "low_dim": 0.2},
    "VARIANCE": {"low_correlation": 0.6, "small_sample": 0.4},
}


# ---------------------------------------------------------------------
# 1. Adaptive weights: w = f(M)
# ---------------------------------------------------------------------

@dataclass
class AdaptiveWeights:
    w_mi: float
    w_rf: float
    w_lasso: float
    w_variance: float
    raw_scores: dict
    derived_signals: dict

    def as_dict(self):
        return {"MI": self.w_mi, "RandomForest": self.w_rf, "LASSO": self.w_lasso, "Variance": self.w_variance}

    def pretty_print(self) -> str:
        lines = ["Adaptive Weights (w = f(M))", "-" * 30]
        for k, v in self.as_dict().items():
            lines.append(f"{k:<14}: {v:.3f}")
        return "\n".join(lines)


def _sigmoid_scale(x: float, midpoint: float, steepness: float = 6.0) -> float:
    """Soft ramp on [0,1], replacing a hard threshold with a continuous function."""
    return float(1.0 / (1.0 + np.exp(-steepness * (x - midpoint))))


def compute_weights(profile: DatasetProfile) -> AdaptiveWeights:
    high_dim = _sigmoid_scale(profile.feat_sample_ratio, midpoint=0.15)      # soft "features > 300"
    noise = min(1.0, profile.noise_estimate)
    entropy = profile.mean_entropy
    correlation = min(1.0, profile.mean_abs_corr / 0.6)                     # soft "corr > 0.8"
    dense = 1.0 - profile.sparsity
    low_dim = 1.0 - high_dim
    small_sample = _sigmoid_scale(-profile.n_samples, midpoint=-1000, steepness=0.004)
    nonlinearity = float(np.clip(0.5 * profile.mean_variance + 0.5 * entropy, 0, 1))

    derived = {"high_dim": high_dim, "noise": noise, "entropy": entropy, "correlation": correlation,
               "dense": dense, "low_dim": low_dim, "small_sample": small_sample, "nonlinearity": nonlinearity}

    score_mi = COEF["MI"]["high_dim"] * high_dim + COEF["MI"]["noise"] * noise + COEF["MI"]["entropy"] * entropy
    score_rf = COEF["RF"]["noise"] * noise + COEF["RF"]["nonlinearity"] * nonlinearity + COEF["RF"]["low_dim"] * low_dim
    score_lasso = COEF["LASSO"]["correlation"] * correlation + COEF["LASSO"]["dense"] * dense + COEF["LASSO"]["low_dim"] * low_dim
    score_variance = COEF["VARIANCE"]["low_correlation"] * (1 - correlation) + COEF["VARIANCE"]["small_sample"] * small_sample

    raw = np.clip(np.array([score_mi, score_rf, score_lasso, score_variance], dtype=float), 1e-6, None)
    weights = raw / raw.sum()

    return AdaptiveWeights(
        w_mi=float(weights[0]), w_rf=float(weights[1]), w_lasso=float(weights[2]), w_variance=float(weights[3]),
        raw_scores={"MI": score_mi, "RF": score_rf, "LASSO": score_lasso, "Variance": score_variance},
        derived_signals=derived,
    )


# ---------------------------------------------------------------------
# 2. Component importance scores (MI, RF, LASSO, Variance)
# ---------------------------------------------------------------------

def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def component_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    numeric_X = X.select_dtypes(include=[np.number])
    X_filled = numeric_X.fillna(numeric_X.median(numeric_only=True))

    mi = pd.Series(mutual_info_classif(X_filled, y, random_state=42), index=X_filled.columns)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_filled, y)
    rf_scores = pd.Series(rf.feature_importances_, index=X_filled.columns)

    X_scaled = StandardScaler().fit_transform(X_filled)
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=42, max_iter=2000)
    lasso.fit(X_scaled, y)
    coefs = np.abs(lasso.coef_).mean(axis=0) if lasso.coef_.ndim > 1 else np.abs(lasso.coef_[0])
    lasso_scores = pd.Series(coefs, index=X_filled.columns)

    var_scores = X_filled.var(numeric_only=True).fillna(0.0)

    return pd.DataFrame({
        "MI": _minmax(mi), "RandomForest": _minmax(rf_scores),
        "LASSO": _minmax(lasso_scores), "Variance": _minmax(var_scores),
    })


# ---------------------------------------------------------------------
# 3. Final score: weighted combination minus redundancy penalty
# ---------------------------------------------------------------------

def final_score(components: pd.DataFrame, weights: AdaptiveWeights, X: pd.DataFrame,
                 redundancy_lambda: float = REDUNDANCY_LAMBDA) -> pd.DataFrame:
    base_score = (
        weights.w_mi * components["MI"]
        + weights.w_rf * components["RandomForest"]
        + weights.w_lasso * components["LASSO"]
        + weights.w_variance * components["Variance"]
    )

    numeric_X = X[components.index].select_dtypes(include=[np.number])
    corr = numeric_X.corr().abs().fillna(0.0)
    redundancy = corr.mul(base_score, axis=0).sum(axis=1) - base_score
    redundancy = redundancy / max(len(base_score) - 1, 1)
    redundancy = _minmax(redundancy)

    result = components.copy()
    result["BaseScore"] = base_score
    result["Redundancy"] = redundancy
    result["FinalScore"] = base_score - redundancy_lambda * redundancy
    return result.sort_values("FinalScore", ascending=False)


if __name__ == "__main__":
    from dataset import load_dataset, profile_dataset

    bundle = load_dataset("breast_cancer")
    profile = profile_dataset(bundle.X, bundle.y, bundle.name)
    weights = compute_weights(profile)
    print(weights.pretty_print())

    comps = component_scores(bundle.X, bundle.y)
    scored = final_score(comps, weights, bundle.X)
    print(scored.head(10))
