# Reviewer-response evidence checklist

This checklist maps each empirical request to an output that must exist before the manuscript is revised. Code availability does not satisfy a reviewer request by itself; the completed multi-seed outputs must be reported.

## Major comment 1 — uncertainty and model selection

- [ ] `avg`, `max`, `gem`, `att`, and `dyna` completed for seeds 13, 42, and 2026
- [ ] `analysis/accuracy_summary.csv` reports mean ± SD and 95% CI
- [ ] `analysis/paired_seed_tests.csv` includes DynaPool–Average and DynaPool–Attention
- [ ] `analysis/mcnemar_auxiliary.csv` is described only as auxiliary per-seed evidence
- [ ] learning curves show whether 50 epochs are sufficient
- [ ] manuscript states that the final epoch is reported and no best-validation selection is used

## Major comment 2 — gate collapse and ablations

- [ ] per-sample alpha archives exist for every dynamic model and seed
- [ ] means, SDs, quartiles, ranges, and entropy are reported
- [ ] original `dyna` is interpreted as collapsed if its distribution is nearly constant/attention-dominant
- [ ] `dyna_att_only`, `static_mean`, and `equal` are completed
- [ ] all four `dyna_drop_*` runs are completed
- [ ] `att_avg`, `att_max`, and `att_gem` are completed if complementary-branch claims remain
- [ ] `dyna_entropy` is clearly labelled as a regularized variant rather than substituted for the original model

## Major comment 3 — compute and latency

- [ ] all five primary methods measured in one session on the same device
- [ ] batch size, input size, precision, warm-up, trials, synchronization, and software/hardware are reported
- [ ] Table, Figure, body text, and conclusion are regenerated from the same `efficiency_summary.csv`
- [ ] MACs and FLOPs are not used interchangeably
- [ ] 3×3/stride-1 stem, no initial max-pool, and 8×8 final feature map are stated

## Major comment 4 — implementation detail

- [ ] Attention uses a 1×1 spatial scoring convolution
- [ ] gate input is global-average-pooled 512-D features
- [ ] hidden dimension 256, ReLU, dropout 0.1, output dimension, and temperature are stated
- [ ] original DynaPool entropy coefficient is reported as 0; regularized variant coefficient is reported separately
- [ ] GeM `p_min`, epsilon, initialization, positivity parameterization, and final learned p are reported
- [ ] backbone stem and head initialization rules are reported
- [ ] Adam betas/epsilon, warm-up epochs, crop size/padding mode, flip probability, normalization, seeds, AMP, and versions are stated

## Claim gate

- [ ] “outperforms/superior/improves” is used only when paired evidence supports it
- [ ] “input-adaptive/interpretable preference” is used only when per-sample variation is meaningful
- [ ] complementary-information claims are backed by two-branch and branch-removal ablations
- [ ] efficiency claims use the corrected raw measurements
- [ ] limitations include single dataset/backbone, 64×64 resolution, dense execution of every active branch, and any remaining gate collapse
