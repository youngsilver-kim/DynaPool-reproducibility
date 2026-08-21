#!/usr/bin/env python3
"""Reproduce the manuscript's Figure 3 from retained validation coefficients.

The script reads all 10,000 validation predictions for each requested seed.
It does not sample or hard-code the reported coefficient or entropy values.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("dyna", "(a) Unregularized DynaPool"),
    ("dyna_entropy", r"(b) DynaPool-ER ($\lambda_u=0.05$)"),
]
BRANCHES = ["avg", "max", "gem", "att"]
BRANCH_LABELS = ["Average", "Max", "GeM", "Attention"]
BRANCH_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
DEFAULT_SEEDS = [13, 42, 2026]


def default_experiment_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "paper_results" / "reviewer_revision_v1"


def load_method(
    experiment_dir: Path,
    method: str,
    seeds: list[int],
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for seed in seeds:
        path = (
            experiment_dir
            / method
            / f"seed_{seed:05d}"
            / "alphas_validation.npz"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing coefficient archive: {path}")

        with np.load(path, allow_pickle=False) as archive:
            required = {"alphas", "branches"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path} is missing keys: {sorted(missing)}")

            alphas = np.asarray(archive["alphas"], dtype=np.float64)
            branches = [str(value) for value in archive["branches"].tolist()]

        if alphas.ndim != 2:
            raise ValueError(f"Expected a 2-D alpha array in {path}; got {alphas.shape}")
        if alphas.shape[0] != 10_000:
            raise ValueError(f"Expected 10,000 validation rows in {path}; got {alphas.shape[0]}")
        if set(branches) != set(BRANCHES):
            raise ValueError(f"Unexpected branches in {path}: {branches}")

        order = [branches.index(branch) for branch in BRANCHES]
        alphas = alphas[:, order]
        if not np.isfinite(alphas).all():
            raise ValueError(f"Non-finite coefficient found in {path}")
        if np.any(alphas < 0.0) or np.any(alphas > 1.0):
            raise ValueError(f"Coefficient outside [0, 1] in {path}")
        if not np.allclose(alphas.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError(f"Coefficient rows do not sum to one in {path}")
        arrays.append(alphas)

    return np.vstack(arrays)


def normalized_entropy(alphas: np.ndarray) -> np.ndarray:
    safe = np.clip(alphas, np.finfo(np.float64).tiny, 1.0)
    return -(safe * np.log(safe)).sum(axis=1) / math.log(alphas.shape[1])


def write_statistics(
    path: Path,
    data: dict[str, np.ndarray],
    seeds: list[int],
) -> None:
    fields = [
        "method",
        "branch",
        "seeds",
        "sample_count",
        "mean",
        "sample_sd",
        "median",
        "q25",
        "q75",
        "minimum",
        "maximum",
        "normalized_entropy_mean",
        "normalized_entropy_sample_sd",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, _ in METHODS:
            alphas = data[method]
            entropy = normalized_entropy(alphas)
            for index, branch in enumerate(BRANCHES):
                values = alphas[:, index]
                q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
                writer.writerow(
                    {
                        "method": method,
                        "branch": branch,
                        "seeds": ",".join(map(str, seeds)),
                        "sample_count": len(values),
                        "mean": f"{values.mean():.9f}",
                        "sample_sd": f"{values.std(ddof=1):.9f}",
                        "median": f"{median:.9f}",
                        "q25": f"{q25:.9f}",
                        "q75": f"{q75:.9f}",
                        "minimum": f"{values.min():.9f}",
                        "maximum": f"{values.max():.9f}",
                        "normalized_entropy_mean": f"{entropy.mean():.9f}",
                        "normalized_entropy_sample_sd": f"{entropy.std(ddof=1):.9f}",
                    }
                )


def draw_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.35, 4.55),
        gridspec_kw={"hspace": 0.34},
    )

    positions = np.arange(1, len(BRANCHES) + 1)
    for ax, (method, title) in zip(axes, METHODS):
        alphas = data[method]
        distributions = [alphas[:, index] for index in range(len(BRANCHES))]
        violin = ax.violinplot(
            distributions,
            positions=positions,
            widths=0.74,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            points=300,
            bw_method="scott",
        )
        for body, color in zip(violin["bodies"], BRANCH_COLORS):
            body.set_facecolor(color)
            body.set_edgecolor("#263238")
            body.set_linewidth(0.55)
            body.set_alpha(0.82)

        q25 = np.array([np.quantile(values, 0.25) for values in distributions])
        median = np.array([np.median(values) for values in distributions])
        q75 = np.array([np.quantile(values, 0.75) for values in distributions])
        ax.vlines(positions, q25, q75, color="#1A1A1A", linewidth=1.05, zorder=3)
        ax.scatter(positions, median, color="white", edgecolor="#1A1A1A", s=12, zorder=4)

        entropy_mean = normalized_entropy(alphas).mean()
        attention_mean = alphas[:, BRANCHES.index("att")].mean()
        annotation = (
            f"Attention mean = {attention_mean:.6f}\n"
            f"Normalized entropy = {entropy_mean:.4f}"
        )
        annotation_x = 0.02 if method == "dyna" else 0.98
        annotation_alignment = "left" if method == "dyna" else "right"
        ax.text(
            annotation_x,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha=annotation_alignment,
            va="top",
            fontsize=7.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        )

        ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", pad=4)
        ax.set_ylabel("Mixture coefficient", fontsize=7.6)
        ax.set_xticks(positions, BRANCH_LABELS, rotation=12, ha="right")
        ax.tick_params(axis="both", labelsize=7.0, width=0.7, length=3)
        ax.grid(axis="y", color="#D7DEE5", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)

        if method == "dyna":
            # A symmetric logarithmic axis keeps zero representable while
            # expanding the near-zero Average, Max, and GeM branches.
            ax.set_yscale("symlog", linthresh=1e-4, linscale=1.0, base=10)
            ax.set_ylim(0.0, 1.05)
            ax.set_yticks([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])

            # The Attention coefficients are concentrated extremely close to
            # one and therefore collapse into a thin mark on the symlog axis.
            # Show the same unsampled values in a small linear-scale inset.
            attention_values = distributions[BRANCHES.index("att")]
            inset = ax.inset_axes([0.70, 0.48, 0.27, 0.34])
            inset_violin = inset.violinplot(
                [attention_values],
                positions=[1],
                widths=0.62,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                points=300,
                bw_method="scott",
            )
            inset_body = inset_violin["bodies"][0]
            inset_body.set_facecolor(BRANCH_COLORS[BRANCHES.index("att")])
            inset_body.set_edgecolor("#263238")
            inset_body.set_linewidth(0.55)
            inset_body.set_alpha(0.82)

            attention_q25, attention_median, attention_q75 = np.quantile(
                attention_values, [0.25, 0.50, 0.75]
            )
            inset.vlines(
                1,
                attention_q25,
                attention_q75,
                color="#1A1A1A",
                linewidth=1.05,
                zorder=3,
            )
            inset.scatter(
                1,
                attention_median,
                color="white",
                edgecolor="#1A1A1A",
                s=12,
                zorder=4,
            )
            inset.set_xlim(0.55, 1.45)
            inset.set_ylim(0.995, 1.0001)
            inset.set_xticks([1], ["Attention"])
            inset.set_yticks([0.995, 0.9975, 1.000])
            inset.set_title("Attention zoom", fontsize=6.5, pad=2)
            inset.tick_params(axis="both", labelsize=5.8, width=0.6, length=2)
            inset.grid(axis="y", color="#D7DEE5", linewidth=0.45)
            inset.spines[["top", "right"]].set_visible(False)
        else:
            ax.set_ylim(0.0, 0.90)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])

    fig.subplots_adjust(left=0.19, right=0.98, top=0.98, bottom=0.10)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 3 from retained validation gate coefficients."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=default_experiment_dir(),
        help="Experiment root containing dyna/ and dyna_entropy/ (default: bundled paper results).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_figures"),
        help="Directory for PNG, PDF, and statistics CSV outputs.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        method: load_method(experiment_dir, method, args.seeds)
        for method, _ in METHODS
    }
    statistics_path = output_dir / "figure3_statistics.csv"
    write_statistics(statistics_path, data, args.seeds)

    figure = draw_figure(data)
    png_path = output_dir / "figure3_gate_coefficients.png"
    pdf_path = output_dir / "figure3_gate_coefficients.pdf"
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Experiment data: {experiment_dir}")
    print(f"PNG: {png_path}")
    print(f"Vector PDF: {pdf_path}")
    print(f"Statistics: {statistics_path}")
    for method, _ in METHODS:
        alphas = data[method]
        entropy_mean = normalized_entropy(alphas).mean()
        attention_mean = alphas[:, BRANCHES.index("att")].mean()
        print(
            f"{method}: n={len(alphas)}, attention_mean={attention_mean:.6f}, "
            f"normalized_entropy_mean={entropy_mean:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
