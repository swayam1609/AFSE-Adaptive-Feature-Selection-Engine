"""
dataset.py
--------------
Unified dataset loading for AFSE.

Every loader returns a DatasetBundle: (X, y, name) where
    X : pd.DataFrame  (n_samples, n_features)
    y : pd.Series     (n_samples,)   binary or multiclass label
    name : str

Datasets wired up:
    - Breast Cancer Wisconsin   (sklearn, built-in)
    - Sonar                     (sklearn-free synthetic-shape fallback if CSV absent)
    - Madelon                   (local CSV, e.g. data/madelon.csv)
    - Parkinson's / Arrhythmia  (local CSV — drop the file in data/ and register it below)

Design note: every dataset, regardless of source, is normalized to the same
(X, y, name) shape so every downstream module (profiler, adaptive engine,
selector, evaluation) never needs to know where the data came from.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def __repr__(self):
        return f"<DatasetBundle name={self.name!r} samples={self.n_samples} features={self.n_features}>"


def _from_sklearn_breast_cancer() -> DatasetBundle:
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    return DatasetBundle(name="Breast Cancer Wisconsin", X=d.data.copy(), y=d.target.copy())


def _load_local_csv(path: str, name: str, label_col: str | int = -1) -> DatasetBundle:
    """
    Generic loader for any local CSV where one column is the label.
    label_col=-1 means "last column".
    """
    df = pd.read_csv(path)
    if isinstance(label_col, int):
        label_name = df.columns[label_col]
    else:
        label_name = label_col
    y = df[label_name].copy()
    X = df.drop(columns=[label_name]).copy()
    # Coerce to numeric where possible; non-numeric feature columns are rare
    # in these benchmark sets but we guard anyway.
    X = X.apply(pd.to_numeric, errors="coerce")
    return DatasetBundle(name=name, X=X, y=y)


def load_madelon() -> DatasetBundle:
    path = os.path.join(DATA_DIR, "madelon.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected Madelon at {path}. Place the CSV there (label column = 'T')."
        )
    return _load_local_csv(path, name="Madelon", label_col="T")


def load_sonar() -> DatasetBundle:
    path = os.path.join(DATA_DIR, "sonar.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sonar CSV not found at {path}. Download from the UCI ML repository "
            f"(Connectionist Bench, Sonar) and place it at data/sonar.csv, "
            f"label column should be the last column (R/M)."
        )
    return _load_local_csv(path, name="Sonar", label_col=-1)


def load_parkinsons() -> DatasetBundle:
    path = os.path.join(DATA_DIR, "parkinsons.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Parkinson's CSV not found at {path}. Download from UCI "
            f"(Parkinsons Data Set) and place it at data/parkinsons.csv, "
            f"label column should be named 'status'."
        )
    return _load_local_csv(path, name="Parkinson's", label_col="status")


def load_arrhythmia() -> DatasetBundle:
    path = os.path.join(DATA_DIR, "arrhythmia.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Arrhythmia CSV not found at {path}. Download from UCI "
            f"(Arrhythmia Data Set) and place it at data/arrhythmia.csv, "
            f"label column should be the last column."
        )
    return _load_local_csv(path, name="Arrhythmia", label_col=-1)


LOADERS = {
    "breast_cancer": _from_sklearn_breast_cancer,
    "madelon": load_madelon,
    "sonar": load_sonar,
    "parkinsons": load_parkinsons,
    "arrhythmia": load_arrhythmia,
}


def load_dataset(key: str) -> DatasetBundle:
    if key not in LOADERS:
        raise KeyError(f"Unknown dataset key '{key}'. Options: {list(LOADERS)}")
    return LOADERS[key]()


def load_all_available() -> list[DatasetBundle]:
    """Load every dataset that is currently available (skips missing local CSVs)."""
    bundles = []
    for key, loader in LOADERS.items():
        try:
            bundles.append(loader())
        except FileNotFoundError as e:
            print(f"[data_loader] Skipping '{key}': {e}")
    return bundles


if __name__ == "__main__":
    for b in load_all_available():
        print(b)
