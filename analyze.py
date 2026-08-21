#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


DEFAULT_COMPARISONS = [
    "dyna:avg",
    "dyna:att",
    "dyna_entropy:dyna",
    "static_mean:dyna",
    "equal:dyna",
    "dyna_att_only:dyna",
    "dyna_drop_avg:dyna",
    "dyna_drop_max:dyna",
    "dyna_drop_gem:dyna",
    "dyna_drop_att:dyna",
]


def load_runs(experiment_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(experiment_dir.glob("*/seed_*/result.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("status") != "completed":
            continue
        rows.append(
            {
                "method": result["method"],
                "seed": int(result["seed"]),
                "accuracy": float(result["validation_accuracy"]),
                "accuracy_percent": 100.0 * float(result["validation_accuracy"]),
                "validation_loss": float(result["validation_loss"]),
                "learned_gem_p": result.get("learned_gem_p"),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "run_dir": str(path.parent),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No completed result.json files under {experiment_dir}")
    return pd.DataFrame(rows)


def summarize_accuracy(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in runs.groupby("method", sort=True):
        values = group["accuracy_percent"].to_numpy(dtype=float)
        n = len(values)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else math.nan
        margin = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else math.nan
        rows.append(
            {
                "method": method,
                "n_seeds": n,
                "mean_accuracy_percent": mean,
                "std_accuracy_percent": sd,
                "ci95_low_percent": mean - margin if n > 1 else math.nan,
                "ci95_high_percent": mean + margin if n > 1 else math.nan,
                "seeds": ",".join(str(seed) for seed in sorted(group["seed"])),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_accuracy_percent", ascending=False)


def paired_tests(runs: pd.DataFrame, comparisons: list[str]) -> pd.DataFrame:
    rows = []
    for comparison in comparisons:
        method_a, method_b = comparison.split(":", 1)
        a = runs[runs.method == method_a][["seed", "accuracy_percent"]].rename(
            columns={"accuracy_percent": "a"}
        )
        b = runs[runs.method == method_b][["seed", "accuracy_percent"]].rename(
            columns={"accuracy_percent": "b"}
        )
        paired = a.merge(b, on="seed").sort_values("seed")
        if len(paired) < 2:
            continue
        differences = (paired["a"] - paired["b"]).to_numpy()
        t_result = stats.ttest_rel(paired["a"], paired["b"])
        try:
            wilcoxon_p = float(stats.wilcoxon(differences).pvalue) if np.any(differences != 0) else 1.0
        except ValueError:
            wilcoxon_p = math.nan
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n_paired_seeds": len(paired),
                "seeds": ",".join(map(str, paired.seed.tolist())),
                "mean_difference_percentage_points_a_minus_b": float(differences.mean()),
                "std_difference_percentage_points": float(differences.std(ddof=1)),
                "paired_t_statistic": float(t_result.statistic),
                "paired_t_pvalue_two_sided": float(t_result.pvalue),
                "wilcoxon_pvalue_two_sided": wilcoxon_p,
            }
        )
    return pd.DataFrame(rows)


def mcnemar_tests(runs: pd.DataFrame, comparisons: list[str]) -> pd.DataFrame:
    lookup = {(row.method, int(row.seed)): Path(row.run_dir) for row in runs.itertuples()}
    rows = []
    for comparison in comparisons:
        method_a, method_b = comparison.split(":", 1)
        common_seeds = sorted(
            {seed for method, seed in lookup if method == method_a}
            & {seed for method, seed in lookup if method == method_b}
        )
        for seed in common_seeds:
            a_path = lookup[(method_a, seed)] / "predictions_validation.npz"
            b_path = lookup[(method_b, seed)] / "predictions_validation.npz"
            if not a_path.exists() or not b_path.exists():
                continue
            a_data, b_data = np.load(a_path), np.load(b_path)
            if not np.array_equal(a_data["targets"], b_data["targets"]):
                raise RuntimeError(f"Validation target order differs for {comparison}, seed {seed}")
            target = a_data["targets"]
            a_correct = a_data["predictions"] == target
            b_correct = b_data["predictions"] == target
            a_only = int(np.sum(a_correct & ~b_correct))
            b_only = int(np.sum(~a_correct & b_correct))
            discordant = a_only + b_only
            exact_p = (
                float(stats.binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue)
                if discordant
                else 1.0
            )
            rows.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "seed": seed,
                    "a_correct_b_wrong": a_only,
                    "a_wrong_b_correct": b_only,
                    "discordant_total": discordant,
                    "exact_mcnemar_pvalue_two_sided": exact_p,
                    "role": "auxiliary per-seed evidence; multi-seed paired test is primary",
                }
            )
    return pd.DataFrame(rows)


def alpha_run_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs.itertuples():
        path = Path(run.run_dir) / "alpha_summary_validation.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        row: dict[str, Any] = {
            "method": run.method,
            "seed": run.seed,
            "sample_count": summary["sample_count"],
            "entropy_mean": summary["entropy"]["mean"],
            "entropy_std": summary["entropy"]["std"],
        }
        for branch, values in summary["coefficients"].items():
            for statistic in ("mean", "std", "min", "q25", "median", "q75", "max"):
                row[f"{branch}_{statistic}"] = values[statistic]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_accuracy(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("mean_accuracy_percent")
    lower = ordered["mean_accuracy_percent"] - ordered["ci95_low_percent"]
    upper = ordered["ci95_high_percent"] - ordered["mean_accuracy_percent"]
    errors = np.vstack([lower.fillna(0), upper.fillna(0)])
    fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * len(ordered))))
    ax.errorbar(
        ordered["mean_accuracy_percent"], ordered["method"], xerr=errors, fmt="o", capsize=4
    )
    ax.set_xlabel("Tiny-ImageNet validation Top-1 accuracy (%)")
    ax.set_ylabel("Method")
    ax.set_title("Mean accuracy with 95% t confidence intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_95ci.png", dpi=300)
    plt.close(fig)


def plot_learning_curves(runs: pd.DataFrame, output_dir: Path) -> None:
    frames = []
    for run in runs.itertuples():
        path = Path(run.run_dir) / "history.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["method"] = run.method
            frame["seed"] = run.seed
            frames.append(frame)
    if not frames:
        return
    history = pd.concat(frames, ignore_index=True)
    baseline_order = [name for name in ["avg", "max", "gem", "att", "dyna", "dyna_entropy"] if name in set(history.method)]
    selected = history[history.method.isin(baseline_order)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=selected,
        x="epoch",
        y="validation_accuracy",
        hue="method",
        estimator="mean",
        errorbar="sd",
        ax=ax,
    )
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Learning curves (mean ± SD across seeds)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=300)
    plt.close(fig)


def plot_gate_distributions(runs: pd.DataFrame, output_dir: Path) -> None:
    records = []
    rng = np.random.default_rng(2026)
    for run in runs[runs.method.isin(["dyna", "dyna_entropy"])].itertuples():
        path = Path(run.run_dir) / "alphas_validation.npz"
        if not path.exists():
            continue
        data = np.load(path)
        alphas = data["alphas"]
        branches = data["branches"].astype(str)
        indices = rng.choice(len(alphas), size=min(5_000, len(alphas)), replace=False)
        for branch_index, branch in enumerate(branches):
            for value in alphas[indices, branch_index]:
                records.append(
                    {"method": run.method, "seed": run.seed, "branch": branch, "coefficient": float(value)}
                )
    if not records:
        return
    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.violinplot(data=frame, x="branch", y="coefficient", hue="method", cut=0, inner="quart", ax=ax)
    ax.set_title("Per-sample gate coefficient distributions")
    ax.set_xlabel("Pooling branch")
    ax.set_ylabel("Gate coefficient")
    fig.tight_layout()
    fig.savefig(output_dir / "gate_coefficient_violin.png", dpi=300)
    plt.close(fig)


def plot_class_heatmap(runs: pd.DataFrame, output_dir: Path, method: str = "dyna") -> None:
    frames = []
    for run in runs[runs.method == method].itertuples():
        path = Path(run.run_dir) / "alpha_by_class_validation.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["seed"] = run.seed
            frames.append(frame)
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    mean_columns = [column for column in combined.columns if column.endswith("_mean")]
    class_mean = combined.groupby("class_name")[mean_columns].mean()
    class_mean.columns = [column.removesuffix("_mean") for column in class_mean.columns]
    # Sort by the dominant branch to make structure visible without cherry-picking classes.
    class_mean = class_mean.assign(_max=class_mean.max(axis=1), _arg=class_mean.idxmax(axis=1)).sort_values(
        ["_arg", "_max"], ascending=[True, False]
    ).drop(columns=["_max", "_arg"])
    fig, ax = plt.subplots(figsize=(8, 24))
    sns.heatmap(class_mean, cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "Mean coefficient"}, ax=ax)
    ax.set_title(f"Class-wise gate coefficients: {method} (mean across seeds)")
    ax.set_xlabel("Pooling branch")
    ax.set_ylabel("Tiny-ImageNet class")
    fig.tight_layout()
    fig.savefig(output_dir / f"class_gate_heatmap_{method}.png", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reviewer-facing statistics, tables, and figures")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--comparisons", nargs="+", default=DEFAULT_COMPARISONS)
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = experiment_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    runs = load_runs(experiment_dir)
    runs.drop(columns=["run_dir"]).to_csv(output_dir / "all_completed_runs.csv", index=False)
    summary = summarize_accuracy(runs)
    summary.to_csv(output_dir / "accuracy_summary.csv", index=False)
    paired = paired_tests(runs, args.comparisons)
    paired.to_csv(output_dir / "paired_seed_tests.csv", index=False)
    mcnemar = mcnemar_tests(runs, args.comparisons)
    mcnemar.to_csv(output_dir / "mcnemar_auxiliary.csv", index=False)
    alpha = alpha_run_summary(runs)
    alpha.to_csv(output_dir / "gate_summary_by_run.csv", index=False)
    plot_accuracy(summary, output_dir)
    plot_learning_curves(runs, output_dir)
    plot_gate_distributions(runs, output_dir)
    plot_class_heatmap(runs, output_dir, method="dyna")
    plot_class_heatmap(runs, output_dir, method="dyna_entropy")
    print(summary.to_string(index=False))
    print(f"\nAnalysis written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
