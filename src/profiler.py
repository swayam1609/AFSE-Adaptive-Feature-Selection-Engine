"""
profiler.py
-----------
Phase 3 of AFSE: the Dataset Profiler.

Computes a Meta-Feature Vector M for a dataset:

    M = [n_samples, n_features, feat_sample_ratio, missing_pct, dup_pct,
         mean_abs_corr, max_abs_corr, mean_variance (normalized),
         mean_entropy, class_imbalance, sparsity, noise_estimate]

This vector is what the Adaptive Decision Engine (Phase 4) consumes to
choose feature-selection weights, instead of hard-coded if/else thresholds.
Keeping the profiler as its own module means the meta-features are computed
once and can be logged/inspected independently of the weighting logic.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


@dataclass
class DatasetProfile:
    name: str
    n_samples: int
    n_features: int
    feat_sample_ratio: float
    missing_pct: float
    duplicate_pct: float
    mean_abs_corr: float
    max_abs_corr: float
    mean_variance: float          # normalized 0-1 (relative to max feature variance)
    mean_entropy: float           # normalized 0-1 (binned entropy per feature)
    class_imbalance: float        # 0 = perfectly balanced, 1 = maximally imbalanced
    sparsity: float               # fraction of near-zero values
    noise_estimate: float         # proxy: mean coefficient of variation of low-corr features

    def as_meta_vector(self) -> np.ndarray:
        """Returns M, the numeric meta-feature vector fed to the adaptive engine."""
        return np.array([
            self.n_samples,
            self.n_features,
            self.feat_sample_ratio,
            self.missing_pct,
            self.duplicate_pct,
            self.mean_abs_corr,
            self.max_abs_corr,
            self.mean_variance,
            self.mean_entropy,
            self.class_imbalance,
            self.sparsity,
            self.noise_estimate,
        ], dtype=float)

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty_print(self) -> str:
        lines = [f"Dataset Profile: {self.name}", "-" * (18 + len(self.name))]
        lines.append(f"Samples             : {self.n_samples}")
        lines.append(f"Features            : {self.n_features}")
        lines.append(f"Feature/Sample ratio: {self.feat_sample_ratio:.4f}")
        lines.append(f"Missing             : {self.missing_pct:.2%}")
        lines.append(f"Duplicate rows      : {self.duplicate_pct:.2%}")
        lines.append(f"Mean |correlation|  : {self.mean_abs_corr:.3f}")
        lines.append(f"Max  |correlation|  : {self.max_abs_corr:.3f}")
        lines.append(f"Mean variance (norm): {self.mean_variance:.3f}")
        lines.append(f"Mean entropy (norm) : {self.mean_entropy:.3f}")
        lines.append(f"Class imbalance     : {self.class_imbalance:.3f}")
        lines.append(f"Sparsity            : {self.sparsity:.3f}")
        lines.append(f"Noise estimate      : {self.noise_estimate:.3f}")
        return "\n".join(lines)


def _binned_entropy(col: pd.Series, bins: int = 10) -> float:
    """Shannon entropy of a continuous column after equal-width binning, normalized to [0,1]."""
    col = col.dropna()
    if col.nunique() <= 1 or len(col) == 0:
        return 0.0
    counts, _ = np.histogram(col, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    h = scipy_entropy(probs, base=2)
    h_max = np.log2(bins)
    return float(h / h_max) if h_max > 0 else 0.0


def profile_dataset(X: pd.DataFrame, y: pd.Series, name: str) -> DatasetProfile:
    n_samples, n_features = X.shape

    missing_pct = float(X.isna().mean().mean()) if n_features > 0 else 0.0
    duplicate_pct = float(X.duplicated().mean())

    # Correlation structure (numeric columns only, pairwise, upper triangle)
    numeric_X = X.select_dtypes(include=[np.number])
    if numeric_X.shape[1] >= 2:
        corr = numeric_X.corr().abs()
        iu = np.triu_indices_from(corr, k=1)
        corr_vals = corr.values[iu]
        corr_vals = corr_vals[~np.isnan(corr_vals)]
        mean_abs_corr = float(corr_vals.mean()) if corr_vals.size else 0.0
        max_abs_corr = float(corr_vals.max()) if corr_vals.size else 0.0
    else:
        mean_abs_corr = 0.0
        max_abs_corr = 0.0

    # Variance, normalized against the max so it's comparable across datasets
    variances = numeric_X.var(numeric_only=True).fillna(0.0)
    mean_variance = float((variances / (variances.max() + 1e-12)).mean()) if len(variances) else 0.0

    # Per-feature entropy (binned), averaged
    if numeric_X.shape[1] > 0:
        sample_cols = numeric_X.columns
        # cap at 200 columns for speed on very high-dim sets like Madelon/Arrhythmia
        if len(sample_cols) > 200:
            sample_cols = pd.Index(np.random.RandomState(42).choice(sample_cols, 200, replace=False))
        entropies = [_binned_entropy(numeric_X[c]) for c in sample_cols]
        mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    else:
        mean_entropy = 0.0

    # Class imbalance: 0 = balanced, 1 = one class dominates completely
    class_counts = y.value_counts(normalize=True)
    if len(class_counts) >= 2:
        majority = class_counts.iloc[0]
        n_classes = len(class_counts)
        # distance from uniform (1/n_classes) scaled to [0,1]
        class_imbalance = float((majority - (1 / n_classes)) / (1 - (1 / n_classes)))
        class_imbalance = max(0.0, min(1.0, class_imbalance))
    else:
        class_imbalance = 0.0

    # Sparsity: fraction of values within 1e-6 of zero
    if numeric_X.shape[1] > 0:
        sparsity = float((numeric_X.abs() < 1e-6).mean().mean())
    else:
        sparsity = 0.0

    # Noise proxy: mean coefficient of variation among the *least* correlated
    # (with each other) features -- high CoV + low structure suggests noise.
    if numeric_X.shape[1] > 0:
        means = numeric_X.mean(numeric_only=True).replace(0, np.nan)
        stds = numeric_X.std(numeric_only=True)
        cov = (stds / means.abs()).replace([np.inf, -np.inf], np.nan).dropna()
        noise_estimate = float(min(1.0, cov.mean())) if len(cov) else 0.0
    else:
        noise_estimate = 0.0

    return DatasetProfile(
        name=name,
        n_samples=n_samples,
        n_features=n_features,
        feat_sample_ratio=float(n_features / max(n_samples, 1)),
        missing_pct=missing_pct,
        duplicate_pct=duplicate_pct,
        mean_abs_corr=mean_abs_corr,
        max_abs_corr=max_abs_corr,
        mean_variance=mean_variance,
        mean_entropy=mean_entropy,
        class_imbalance=class_imbalance,
        sparsity=sparsity,
        noise_estimate=noise_estimate,
    )


if __name__ == "__main__":
    from dataset import load_all_available
    for bundle in load_all_available():
        p = profile_dataset(bundle.X, bundle.y, bundle.name)
        print(p.pretty_print())
        print()
