"""
main.py
-------
End-to-end AFSE pipeline entry point.

For every available dataset:
    1. Load + profile it                              (dataset.py)
    2. Fit AFSEModel -> adaptive weights + ranking     (model.py, loss.py)
    3. Select top-k% features, train downstream models (train.py)
    4. Benchmark AFSE vs RFE / MI / LASSO / RF-importance (evaluate.py)
    5. Run stability analysis + ablation study + Wilcoxon test
    6. Save figures to Figures/, tables to Results/, a pickled model to Model/

Run:
    python main.py                      # all available datasets, top 10%
    python main.py --dataset madelon    # single dataset
    python main.py --k_pct 0.2          # top 20% instead
"""

from __future__ import annotations
import argparse
import os
import sys
import pickle
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from dataset import load_dataset, load_all_available
from model import AFSEModel
from evaluate import (
    benchmark_all_methods, stability_analysis, wilcoxon_test, ablation_variants,
    baseline_mutual_information,
)
from train import train_on_subset
from evaluate import evaluate_run, results_to_dataframe

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT, "Figures")
RESULTS_DIR = os.path.join(ROOT, "Results")
MODEL_DIR = os.path.join(ROOT, "Model")

sns.set_theme(style="whitegrid")


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "")


def save_figures(dataset_name: str, benchmark_df: pd.DataFrame, ranking: pd.DataFrame, X: pd.DataFrame) -> None:
    tag = _safe_name(dataset_name)

    # Accuracy comparison bar chart
    plt.figure(figsize=(8, 5))
    order = benchmark_df.groupby("method")["accuracy"].mean().sort_values(ascending=False).index
    sns.barplot(data=benchmark_df, x="method", y="accuracy", order=order, errorbar="sd")
    plt.title(f"Accuracy by Feature-Selection Method — {dataset_name}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{tag}_accuracy_comparison.png"), dpi=150)
    plt.close()

    # Runtime comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(data=benchmark_df, x="method", y="runtime_sec", order=order, errorbar="sd")
    plt.title(f"Training Runtime by Method — {dataset_name}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{tag}_runtime_comparison.png"), dpi=150)
    plt.close()

    # Feature importance (top 15 from AFSE ranking)
    plt.figure(figsize=(8, 6))
    top = ranking.head(15)
    sns.barplot(x=top["FinalScore"], y=top.index, orient="h")
    plt.title(f"AFSE Top-15 Feature Importance — {dataset_name}")
    plt.xlabel("FinalScore")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{tag}_feature_importance.png"), dpi=150)
    plt.close()

    # Correlation heatmap (top 15 features only, for readability)
    plt.figure(figsize=(8, 6))
    top_feats = list(top.index)
    corr = X[top_feats].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title(f"Correlation Heatmap (Top 15 Features) — {dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{tag}_correlation_heatmap.png"), dpi=150)
    plt.close()


def run_pipeline(dataset_key: str, k_pct: float = 0.10) -> dict:
    bundle = load_dataset(dataset_key)
    print(f"\n{'='*60}\n{bundle.name}  ({bundle.n_samples} samples, {bundle.n_features} features)\n{'='*60}")

    # 1-2. Profile + adaptive weights + ranking
    afse = AFSEModel().fit(bundle.X, bundle.y, name=bundle.name)
    print(afse.profile_.pretty_print())
    print()
    print(afse.weights_.pretty_print())

    # 3. Select + train
    features = afse.select(k_pct)
    print(f"\nAFSE selected {len(features)} features (top {k_pct:.0%})")

    # 4. Benchmark vs baselines
    benchmark_df = benchmark_all_methods(bundle.X, bundle.y, features, k_pct=k_pct)
    print("\nBenchmark (mean per method):")
    print(benchmark_df.groupby("method")[["accuracy", "f1", "roc_auc", "runtime_sec"]].mean().round(4))

    # 5a. Stability analysis (fewer repeats for very high-dim sets to stay fast)
    n_repeats = 3 if bundle.n_features > 200 else 5
    stability_df = stability_analysis(bundle.X, bundle.y, lambda X, y, k: AFSEModel()
                                       .fit(X, y, name=bundle.name).select(k),
                                       n_repeats=n_repeats, k_pct=k_pct)
    afse.set_stability(stability_df)
    print(f"\nTop-5 most stable features:\n{stability_df.head(5).to_string(index=False)}")

    # 5b. Ablation study
    ablation_results = []
    for variant_name, weights in ablation_variants(afse.weights_).items():
        from loss import component_scores, final_score
        comps = component_scores(bundle.X, bundle.y)
        ranked = final_score(comps, weights, bundle.X)
        variant_features = list(ranked.index[:len(features)])
        run = train_on_subset(bundle.X, bundle.y, variant_features, method=variant_name)
        for r in evaluate_run(run):
            ablation_results.append(r)
    ablation_df = results_to_dataframe(ablation_results)
    print(f"\nAblation study (mean accuracy per variant):")
    print(ablation_df.groupby("method")["accuracy"].mean().sort_values(ascending=False).round(4))

    # 5c. Significance test: AFSE vs its strongest baseline, per-model accuracy pairs
    afse_scores = benchmark_df[benchmark_df.method == "AFSE"].sort_values("model")["accuracy"].tolist()
    other_methods = [m for m in benchmark_df.method.unique() if m != "AFSE"]
    sig_results = {}
    for m in other_methods:
        other_scores = benchmark_df[benchmark_df.method == m].sort_values("model")["accuracy"].tolist()
        sig_results[m] = wilcoxon_test(afse_scores, other_scores)
    print("\nSignificance (AFSE vs baselines, Wilcoxon on paired per-model accuracy):")
    for m, res in sig_results.items():
        print(f"  vs {m:<24}: {res}")

    # 6. Explanations
    explanations = afse.explain(top_n=10)

    # Save figures + results
    save_figures(bundle.name, benchmark_df, afse.ranking_, bundle.X)
    tag = _safe_name(bundle.name)
    benchmark_df.to_csv(os.path.join(RESULTS_DIR, f"{tag}_benchmark.csv"), index=False)
    afse.ranking_.to_csv(os.path.join(RESULTS_DIR, f"{tag}_afse_ranking.csv"))
    stability_df.to_csv(os.path.join(RESULTS_DIR, f"{tag}_stability.csv"), index=False)
    ablation_df.to_csv(os.path.join(RESULTS_DIR, f"{tag}_ablation.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, f"{tag}_explanations.txt"), "w") as f:
        for exp in explanations:
            f.write(exp.pretty_print() + "\n\n")

    with open(os.path.join(MODEL_DIR, f"{tag}_afse_model.pkl"), "wb") as f:
        pickle.dump(afse, f)

    return {
        "dataset": bundle.name, "profile": afse.profile_, "weights": afse.weights_,
        "benchmark": benchmark_df, "stability": stability_df, "ablation": ablation_df,
        "significance": sig_results, "explanations": explanations,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the AFSE pipeline.")
    parser.add_argument("--dataset", type=str, default=None,
                         help="One of: breast_cancer, madelon, sonar, parkinsons, arrhythmia. "
                              "Omit to run every dataset currently available in Data/.")
    parser.add_argument("--k_pct", type=float, default=0.10, help="Top-k fraction of features to select.")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.dataset:
        run_pipeline(args.dataset, k_pct=args.k_pct)
    else:
        from dataset import LOADERS
        for key, loader in LOADERS.items():
            try:
                loader()  # cheap existence check before the full pipeline runs it again
            except FileNotFoundError as e:
                print(f"[main] Skipping '{key}': {e}")
                continue
            run_pipeline(key, k_pct=args.k_pct)


if __name__ == "__main__":
    main()
