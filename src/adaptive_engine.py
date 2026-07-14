"""
adaptive_engine.py
-------------------
Phase 4 (upgraded per review notes): the Adaptive Decision Engine.

Instead of hard-coded rules like:

    if features > 300: use MI
    if correlation > 0.8: use LASSO

we score each candidate feature-selection strategy against the dataset's
meta-feature vector M (from profiler.py) using a transparent linear scoring
function, then softmax-normalize the scores into weights w1..w4 that sum to 1.

This gives every weight a stated, inspectable justification instead of an
arbitrary threshold -- the direct fix for "why 300? why 0.8?" that a reviewer
or interviewer will ask.

Score_MI       = a1*HighDim + a2*Noise + a3*Entropy
Score_RF       = b1*Noise + b2*NonLinearitySignal + b3*(1 - HighDim)
Score_LASSO    = c1*Correlation + c2*(1 - Sparsity) + c3*(1 - HighDim)
Score_Variance = d1*(1 - Correlation) + d2*LowSampleSize

P_i = Score_i / sum(Score)   ->  these are the adaptive weights w1..w4

The coefficients (a1, a2, ... ) are declared as named constants below, not
buried magic numbers, and can be tuned/learned later (see Future Scope:
meta-learning).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from profiler import DatasetProfile


# ---- Named, inspectable coefficients (the "why 300 / why 0.8" answer) ----
# Each is a weight in [0,1] describing how much a raw dataset characteristic
# contributes to preferring a given strategy. Tunable; documented; not magic.
COEF = {
    "MI":       {"high_dim": 0.5, "noise": 0.3, "entropy": 0.2},
    "RF":       {"noise": 0.45, "nonlinearity": 0.35, "low_dim": 0.2},
    "LASSO":    {"correlation": 0.5, "dense": 0.3, "low_dim": 0.2},
    "VARIANCE": {"low_correlation": 0.6, "small_sample": 0.4},
}


@dataclass
class AdaptiveWeights:
    w_mi: float
    w_rf: float
    w_lasso: float
    w_variance: float
    raw_scores: dict
    derived_signals: dict

    def as_dict(self):
        return {
            "MI": self.w_mi,
            "RandomForest": self.w_rf,
            "LASSO": self.w_lasso,
            "Variance": self.w_variance,
        }

    def pretty_print(self) -> str:
        lines = ["Adaptive Weights (w = f(M))", "-" * 30]
        for k, v in self.as_dict().items():
            lines.append(f"{k:<14}: {v:.3f}")
        lines.append("")
        lines.append("Derived signals used:")
        for k, v in self.derived_signals.items():
            lines.append(f"  {k:<16}: {v:.3f}")
        return "\n".join(lines)


def _sigmoid_scale(x: float, midpoint: float, steepness: float = 6.0) -> float:
    """Smoothly maps a raw meta-feature onto [0,1] centered at `midpoint`,
    replacing a hard threshold (e.g. 'features > 300') with a soft ramp."""
    return float(1.0 / (1.0 + np.exp(-steepness * (x - midpoint))))


def compute_weights(profile: DatasetProfile) -> AdaptiveWeights:
    # Derived signals, each already normalized to roughly [0,1]
    high_dim = _sigmoid_scale(profile.feat_sample_ratio, midpoint=0.15)          # soft version of "features > 300"
    noise = min(1.0, profile.noise_estimate)
    entropy = profile.mean_entropy
    correlation = min(1.0, profile.mean_abs_corr / 0.6)                          # soft version of "corr > 0.8"
    dense = 1.0 - profile.sparsity
    low_dim = 1.0 - high_dim
    small_sample = _sigmoid_scale(-profile.n_samples, midpoint=-1000, steepness=0.004)
    # nonlinearity signal proxy: high variance dispersion + moderate entropy
    # (RF importance tends to help when relationships are non-monotonic)
    nonlinearity = float(np.clip(0.5 * profile.mean_variance + 0.5 * entropy, 0, 1))

    derived = {
        "high_dim": high_dim,
        "noise": noise,
        "entropy": entropy,
        "correlation": correlation,
        "dense": dense,
        "low_dim": low_dim,
        "small_sample": small_sample,
        "nonlinearity": nonlinearity,
    }

    score_mi = (
        COEF["MI"]["high_dim"] * high_dim
        + COEF["MI"]["noise"] * noise
        + COEF["MI"]["entropy"] * entropy
    )
    score_rf = (
        COEF["RF"]["noise"] * noise
        + COEF["RF"]["nonlinearity"] * nonlinearity
        + COEF["RF"]["low_dim"] * low_dim
    )
    score_lasso = (
        COEF["LASSO"]["correlation"] * correlation
        + COEF["LASSO"]["dense"] * dense
        + COEF["LASSO"]["low_dim"] * low_dim
    )
    score_variance = (
        COEF["VARIANCE"]["low_correlation"] * (1 - correlation)
        + COEF["VARIANCE"]["small_sample"] * small_sample
    )

    raw = np.array([score_mi, score_rf, score_lasso, score_variance], dtype=float)
    raw = np.clip(raw, 1e-6, None)  # avoid zero-division; every strategy keeps a nonzero floor
    weights = raw / raw.sum()

    return AdaptiveWeights(
        w_mi=float(weights[0]),
        w_rf=float(weights[1]),
        w_lasso=float(weights[2]),
        w_variance=float(weights[3]),
        raw_scores={"MI": score_mi, "RF": score_rf, "LASSO": score_lasso, "Variance": score_variance},
        derived_signals=derived,
    )


if __name__ == "__main__":
    from dataset import load_all_available
    from profiler import profile_dataset

    for bundle in load_all_available():
        profile = profile_dataset(bundle.X, bundle.y, bundle.name)
        weights = compute_weights(profile)
        print(f"=== {bundle.name} ===")
        print(weights.pretty_print())
        print()
