"""
train.py
--------
Phase 7: Train Models.

Given a dataset and a chosen feature subset (from model.py's AFSEModel, or
a baseline method in evaluate.py), split the data, scale it, and train every
model in model.get_models() on it -- recording runtime and peak memory for
each fit, which evaluate.py reports on.
"""

from __future__ import annotations
import time
import tracemalloc
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import get_models


@dataclass
class TrainedModel:
    name: str
    estimator: object
    runtime_sec: float
    peak_memory_kb: float


@dataclass
class TrainingRun:
    method: str                 # which feature-selection method produced this subset (e.g. "AFSE", "RFE")
    features: list
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    trained_models: list = field(default_factory=list)


def train_on_subset(X: pd.DataFrame, y: pd.Series, features: list, method: str = "AFSE",
                     test_size: float = 0.3, random_state: int = 42) -> TrainingRun:
    X_sub = X[features].fillna(X[features].median(numeric_only=True))
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    run = TrainingRun(method=method, features=features, X_train=X_train_scaled, X_test=X_test_scaled,
                       y_train=y_train, y_test=y_test)

    for name, estimator in get_models().items():
        tracemalloc.start()
        t0 = time.perf_counter()
        estimator.fit(X_train_scaled, y_train)
        runtime = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        run.trained_models.append(TrainedModel(name=name, estimator=estimator, runtime_sec=runtime,
                                                peak_memory_kb=peak / 1024))
    return run


if __name__ == "__main__":
    from dataset import load_dataset
    from model import AFSEModel

    bundle = load_dataset("breast_cancer")

    afse = AFSEModel().fit(bundle.X, bundle.y, name=bundle.name)
    features = afse.select(0.10)

    run = train_on_subset(bundle.X, bundle.y, features, method="AFSE")

    for tm in run.trained_models:
        print(f"{tm.name:<20} runtime={tm.runtime_sec:.4f}s  peak_mem={tm.peak_memory_kb:.1f}KB")