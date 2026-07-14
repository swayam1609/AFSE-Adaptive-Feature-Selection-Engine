"""
model.py
--------
The AFSE model itself: given a dataset, it profiles it, derives adaptive
weights, ranks features via the hybrid score in loss.py, and can explain
why each feature was chosen.

Also defines the downstream classifier zoo used by train.py/evaluate.py
to actually test how good the selected features are.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from profiler import DatasetProfile, profile_dataset
from loss import AdaptiveWeights, compute_weights, component_scores, final_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def get_models() -> dict:
    """Downstream classifier zoo. Falls back to GradientBoosting if xgboost
    isn't installed, so the repo runs with only requirements.txt."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM": SVC(probability=True, random_state=42),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
    else:
        models["GradientBoosting"] = GradientBoostingClassifier(random_state=42)
    return models


@dataclass
class FeatureExplanation:
    feature: str
    importance: float
    reasons: list

    def pretty_print(self) -> str:
        lines = [f"Feature    : {self.feature}", f"Importance : {self.importance:.3f}", "Reason:"]
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def _relative_label(value: float, high: float = 0.66, low: float = 0.33) -> str:
    if value >= high:
        return "High"
    if value <= low:
        return "Low"
    return "Moderate"


class AFSEModel:
    """
    Adaptive Feature Selection Engine.

    Usage:
        model = AFSEModel().fit(X, y, name="Madelon")
        top_features = model.select(k_pct=0.10)
        explanations = model.explain(top_n=10)
    """

    def __init__(self, redundancy_lambda: float = 0.15):
        self.redundancy_lambda = redundancy_lambda
        self.profile_: DatasetProfile | None = None
        self.weights_: AdaptiveWeights | None = None
        self.ranking_: pd.DataFrame | None = None
        self._stability_freq: dict | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, name: str = "dataset") -> "AFSEModel":
        self.profile_ = profile_dataset(X, y, name)
        self.weights_ = compute_weights(self.profile_)
        comps = component_scores(X, y)
        self.ranking_ = final_score(comps, self.weights_, X, redundancy_lambda=self.redundancy_lambda)
        return self

    def select(self, k_pct: float = 0.10) -> list:
        if self.ranking_ is None:
            raise RuntimeError("Call .fit(X, y) before .select().")
        k = max(1, int(round(len(self.ranking_) * k_pct)))
        return list(self.ranking_.index[:k])

    def set_stability(self, stability_df: pd.DataFrame) -> None:
        """Attach repeated-CV selection frequency (from evaluate.py) so
        explain() can cite stability as a reason."""
        self._stability_freq = dict(zip(stability_df["feature"], stability_df["selection_frequency"]))

    def explain(self, top_n: int = 15) -> list:
        if self.ranking_ is None:
            raise RuntimeError("Call .fit(X, y) before .explain().")
        rows = self.ranking_.head(top_n)
        explanations = []
        for feature, row in rows.iterrows():
            reasons = [
                f"{_relative_label(row['MI'])} Mutual Information ({row['MI']:.2f})",
                f"{_relative_label(row['RandomForest'])} Random Forest importance ({row['RandomForest']:.2f})",
                f"{_relative_label(row['LASSO'])} LASSO coefficient magnitude ({row['LASSO']:.2f})",
                f"{_relative_label(row['Redundancy'])} redundancy with other top features "
                f"({row['Redundancy']:.2f} penalty applied)",
            ]
            if self._stability_freq and feature in self._stability_freq:
                freq = self._stability_freq[feature]
                stability_word = "stable" if freq >= 0.7 else "moderately stable" if freq >= 0.4 else "unstable"
                reasons.append(f"Selected in {freq:.0%} of repeated cross-validation runs ({stability_word})")
            explanations.append(FeatureExplanation(feature=feature, importance=float(row["FinalScore"]), reasons=reasons))
        return explanations


if __name__ == "__main__":
    from dataset import load_dataset

    bundle = load_dataset("breast_cancer")
    model = AFSEModel().fit(bundle.X, bundle.y, name=bundle.name)

    print(model.weights_.pretty_print())
    print("\nTop 10% features:", model.select(0.10))
    print()

    for exp in model.explain(top_n=3):
        print(exp.pretty_print())
        print()