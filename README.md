# DynaPool reproducibility benchmark

This repository contains a reproducible Tiny-ImageNet pooling benchmark built to answer the major reviewer concerns about DynaPool. It compares Average, Max, GeM, Attention, and DynaPool under identical ResNet-18 training conditions and adds the required multi-seed uncertainty, gate-collapse analysis, ablations, and transparent efficiency measurements.

The primary reported checkpoint is always the **final epoch**. A best-validation checkpoint is not used, so the official Tiny-ImageNet validation split is not reused for model selection.

## What was corrected

| Reviewer concern | Implementation/evidence |
|---|---|
| Standard ImageNet stem may reduce 64×64 inputs to a 2×2 final map | 3×3, stride-1 stem; initial max-pool removed; final feature map is tested as 8×8 |
| GeM implementation and learned exponent are unclear | Positive exponent parameterized with `p_min + softplus(raw_p)`; epsilon and final learned `p` are recorded |
| Single-seed accuracy cannot support a 0.25 percentage-point claim | Three seeds by default; mean, SD, 95% t-CI, paired t-test, Wilcoxon test, and auxiliary per-seed exact McNemar test |
| Gate mean of 0.999140 may indicate collapse, not adaptivity | Every validation-sample coefficient is saved; mean, SD, median, min/max, quartiles, entropy, class statistics, violin plots, and heatmaps are generated |
| DynaPool ablations are missing | Attention-only, static-mean, equal mixture, every branch-removal model, Attention+Average/Max/GeM, and entropy-regularized DynaPool |
| Latency/FLOP conditions are inconsistent | A separate FP32 measurement script records raw trials, hardware/software, input/batch size, warm-up, synchronization, eval/inference mode, MACs, and the explicit 2-FLOPs-per-MAC convention |
| Colab runtime can disconnect | Atomic checkpoint after every epoch to Google Drive; rerunning the same command resumes and completed runs are skipped |

## Colab workflow

Open [`colab/DynaPool_Reviewer_Revision.ipynb`](colab/DynaPool_Reviewer_Revision.ipynb). The notebook mounts Google Drive and uses the following persistent layout:

```text
MyDrive/DynaPool_Reviewer_Revision/
└── reviewer_revision_v1/
    ├── avg/seed_00013/
    ├── max/seed_00013/
    ├── ...
    └── analysis/
```

If the runtime disconnects, reconnect, run the setup/data cells, and rerun the same training cell. At most the currently unfinished epoch is lost. An old checkpoint is never loaded when the experiment configuration hash differs.

The Git commit is also part of the run hash. Do not pull new code in the middle of an experiment unless you intentionally start a new `--experiment-name`; mixing checkpoints produced by different code revisions is blocked.

## Local setup

PyTorch and torchvision are preinstalled on Colab. For the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python download_data.py --data-dir data --delete-archive
```

Run a short software-only smoke test first:

```bash
python train.py \
  --fake-data \
  --methods avg dyna \
  --seeds 13 \
  --epochs 2 \
  --warmup-epochs 1 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --experiment-name smoke_test
```

FakeData results are only for checking the pipeline and must never be reported in the paper.

## Phase 1: five-method, three-seed benchmark

```bash
python train.py \
  --suite baselines \
  --seeds 13 42 2026 \
  --epochs 50 \
  --warmup-epochs 5 \
  --data-root data/tiny-imagenet-200 \
  --output-root /content/drive/MyDrive/DynaPool_Reviewer_Revision \
  --experiment-name reviewer_revision_v1
```

All methods use the same epoch-derived data order and augmentation seed for each paired run. The default architecture and optimizer details are serialized to every `run_config.json`.

## Phase 2: reviewer-required ablations

Run this only after Phase 1 finishes. `static_mean` derives its fixed coefficients from the completed DynaPool model's deterministic **training-set** coefficient mean for the same seed, avoiding validation-set leakage.

```bash
python train.py \
  --suite reviewer \
  --seeds 13 42 2026 \
  --epochs 50 \
  --warmup-epochs 5 \
  --data-root data/tiny-imagenet-200 \
  --output-root /content/drive/MyDrive/DynaPool_Reviewer_Revision \
  --experiment-name reviewer_revision_v1
```

Important method names:

- `dyna`: original dense four-branch DynaPool, entropy coefficient 0
- `dyna_entropy`: same model with the configured uniformity penalty, default `lambda=0.05`
- `dyna_att_only`: attention-only mixture-head control
- `static_mean`: fixed train-set mean coefficients from `dyna`
- `equal`: fixed 0.25 coefficients
- `dyna_drop_*`: branch-removal controls
- `att_avg`, `att_max`, `att_gem`: two-branch dynamic controls

## Statistical tables and figures

```bash
python analyze.py \
  --experiment-dir /content/drive/MyDrive/DynaPool_Reviewer_Revision/reviewer_revision_v1
```

Generated evidence includes:

- `accuracy_summary.csv`: mean, SD, and 95% CI
- `paired_seed_tests.csv`: paired multi-seed comparisons
- `mcnemar_auxiliary.csv`: auxiliary per-seed exact McNemar tests
- `gate_summary_by_run.csv`: coefficient and entropy statistics
- `accuracy_95ci.png`, `learning_curves.png`
- `gate_coefficient_violin.png`, class-wise heatmaps

### Reproduce manuscript Figure 3

The repository includes the six original validation-coefficient archives used
for Figure 3. The plotting script recomputes the annotations from all 30,000
validation predictions per method and does not sample the data.

```bash
python scripts/plot_figure3.py
```

Outputs are written to `paper_figures/`:

- `figure3_gate_coefficients.png` (600 dpi by default)
- `figure3_gate_coefficients.pdf` (vector output)
- `figure3_statistics.csv` (recomputed means, dispersion, quartiles, and entropy)

Recommended manuscript caption:

> **Figure 3.** Validation coefficient distributions pooled over three seeds
> (30,000 image-level observations per method). The unregularized panel uses a
> symmetric logarithmic scale to expose the near-zero branches.

To use an external evidence directory instead of the bundled result subset:

```bash
python scripts/plot_figure3.py \
  --experiment-dir /path/to/reviewer_revision_v1 \
  --output-dir /path/to/figures
```

## Efficiency measurement

Run efficiency measurement in one uninterrupted Colab session on the same assigned GPU. Do not combine results from T4, L4, A100, CPU, or different batch sizes.

```bash
python measure_efficiency.py \
  --methods avg max gem att dyna \
  --batch-size 1 \
  --warmup 100 \
  --repetitions 300 \
  --trials 10 \
  --output-dir /content/drive/MyDrive/DynaPool_Reviewer_Revision/reviewer_revision_v1/efficiency
```

`efficiency_summary.csv`, `latency_raw_trials.csv`, and `measurement_environment.json` must be kept together. THOP's output is labelled as MACs. The separate FLOPs column explicitly uses two FLOPs per MAC; functional elementwise operations may not be included and this limitation is recorded.

## Final completeness audit

After both phases, analysis, and efficiency measurement finish:

```bash
python audit_experiment.py \
  --experiment-dir /content/drive/MyDrive/DynaPool_Reviewer_Revision/reviewer_revision_v1 \
  --seeds 13 42 2026 \
  --require-efficiency
```

The command exits with an error if a required run, seed, final-epoch history, 10,000-image validation result, paired prediction order, gate archive, GeM exponent, analysis table, or efficiency record is missing. The machine-readable report is saved as `analysis/audit_report.json`.

## Evidence policy

- Report Tiny-ImageNet **validation** accuracy, never test accuracy.
- Report final-epoch results unless the experimental protocol is redesigned with a separate held-out model-selection split.
- Do not describe DynaPool as input-adaptive if per-sample coefficients have negligible variation or collapse to Attention.
- Do not claim superiority when the paired multi-seed comparison is not statistically significant.
- Do not select only favorable seeds, epochs, ablations, or latency trials.
- Keep the original `dyna` result alongside `dyna_entropy`; the regularized variant is an ablation, not a silent replacement.

See [`REVIEWER_CHECKLIST.md`](REVIEWER_CHECKLIST.md) before revising the manuscript or response letter.

See [`PROVENANCE.md`](PROVENANCE.md) and
[`GITHUB_RELEASE_CHECKLIST.md`](GITHUB_RELEASE_CHECKLIST.md) before publishing
the code or evidence archive.
