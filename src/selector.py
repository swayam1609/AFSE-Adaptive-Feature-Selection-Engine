"""
selector.py
-----------
Phase 5 + 6: Hybrid Feature Ranking, Redundancy Elimination, and Selection.

FinalScore_i = w1*MI_i + w2*RF_i + w3*LASSO_i + w4*Variance_i - lambda * sum_j Corr(i,j)

where w1..w4 come from adaptive_engine.compute_weights(profile), and the
redundancy term penalizes features that are highly correlated with other
already-important features (addresses the known MI weakness of picking
redundant variables, per review notes).
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from adaptive_engine import AdaptiveWeights


REDUNDANCY_LAMBDA = 0.15  # strength of the correlation penalty; named, not magic


@dataclass
class RankingResult:
    scores: pd.DataFrame        # per-feature MI/RF/LASSO/Variance/Final scores
    ranked_features: list       # feature names, best first (post redundancy penalty)

    def top_k_pct(self, pct: float) -> list:
        k = max(1, int(round(len(self.ranked_features) * pct)))
        return self.ranked_features[:k]


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def _mutual_information_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    X_filled = X.fillna(X.median(numeric_only=True))
    mi = mutual_info_classif(X_filled, y, random_state=42)
    return pd.Series(mi, index=X.columns)


def _random_forest_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    X_filled = X.fillna(X.median(numeric_only=True))
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_filled, y)
    return pd.Series(rf.feature_importances_, index=X.columns)


def _lasso_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    X_filled = X.fillna(X.median(numeric_only=True))
    X_scaled = StandardScaler().fit_transform(X_filled)
    lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, random_state=42, max_iter=2000)
    lasso.fit(X_scaled, y)
    coefs = np.abs(lasso.coef_).mean(axis=0) if lasso.coef_.ndim > 1 else np.abs(lasso.coef_[0])
    return pd.Series(coefs, index=X.columns)


def _variance_scores(X: pd.DataFrame) -> pd.Series:
    return X.var(numeric_only=True).fillna(0.0)


def rank_features(X: pd.DataFrame, y: pd.Series, weights: AdaptiveWeights,
                   redundancy_lambda: float = REDUNDANCY_LAMBDA) -> RankingResult:
    numeric_X = X.select_dtypes(include=[np.number])

    mi = _minmax(_mutual_information_scores(numeric_X, y))
    rf = _minmax(_random_forest_scores(numeric_X, y))
    lasso = _minmax(_lasso_scores(numeric_X, y))
    var = _minmax(_variance_scores(numeric_X))

    base_score = (
        weights.w_mi * mi
        + weights.w_rf * rf
        + weights.w_lasso * lasso
        + weights.w_variance * var
    )

    # Redundancy penalty: for each feature, how correlated is it (on average)
    # with the other top-scoring features? Penalize features that mostly
    # duplicate information another strong feature already provides.
    corr = numeric_X.corr().abs().fillna(0.0)
    # weight correlations by the partner feature's base_score so a feature
    # is only "penalized" for overlapping with something that already matters
    redundancy = corr.mul(base_score, axis=0).sum(axis=1) - base_score  # exclude self-corr(=1)*self-score
    redundancy = redundancy / max(len(base_score) - 1, 1)
    redundancy = _minmax(redundancy)

    final_score = base_score - redundancy_lambda * redundancy

    scores_df = pd.DataFrame({
        "MI": mi, "RandomForest": rf, "LASSO": lasso, "Variance": var,
        "BaseScore": base_score, "Redundancy": redundancy, "FinalScore": final_score,
    }).sort_values("FinalScore", ascending=False)

    return RankingResult(scores=scores_df, ranked_features=list(scores_df.index))


if __name__ == "__main__":
    from dataset import load_dataset
    from profiler import profile_dataset
    from adaptive_engine import compute_weights

    bundle = load_dataset("breast_cancer")
    profile = profile_dataset(bundle.X, bundle.y, bundle.name)
    weights = compute_weights(profile)
    result = rank_features(bundle.X, bundle.y, weights)
    print(result.scores.head(10))
    print("\nTop 10% features:", result.top_k_pct(0.10))
