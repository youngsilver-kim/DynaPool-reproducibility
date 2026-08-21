# Audited paper-result subset

This directory contains only the small result subset needed to regenerate the
paper figures and inspect the reported summary statistics.

- `reviewer_revision_v1/dyna/*/alphas_validation.npz`: unregularized DynaPool
  validation coefficients for seeds 13, 42, and 2026.
- `reviewer_revision_v1/dyna_entropy/*/alphas_validation.npz`: entropy-regularized
  collapse-control coefficients for the same seeds.
- `reviewer_revision_v1/analysis/`: audited accuracy, paired-test, gate, and run
  summary files.
- `reviewer_revision_v1/efficiency/`: summary, raw latency trials, and profiling
  environment.

Run Figure 3 from a fresh clone:

```bash
python -m pip install -r requirements.txt
python scripts/plot_figure3.py
```

The command writes a 600-dpi PNG, a vector PDF, and the statistics recomputed
from the coefficient archives to `paper_figures/`.

The complete audit archive is distributed as a GitHub Release asset. It is not
committed to the main branch because it contains prediction-level evidence for
all 48 runs. Model checkpoints and Tiny-ImageNet images are not distributed.

Verify the bundled subset from this directory with:

```bash
sha256sum -c SHA256SUMS
```
