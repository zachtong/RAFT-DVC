# RAFTcorr3D Experiment Plan

**Date**: 2026-04-15
**Target**: Experimental Mechanics submission, July 2026
**Paper title**: "RAFTcorr3D: A Feature-Robust Deep Learning Framework for Digital Volume Correlation"

---

## 1. Core Hypothesis

The accuracy of RAFT-based DVC models is highly coupled with feature characteristics
(bead density, bead radius) and architecture choice (downsampling factor). By
systematically studying this coupling and introducing feature-dropout training (cutout)
and uncertainty-aware loss, we can build a more feature-robust model.

## 2. Controlled Variables (Fixed)

> **Revised 2026-05-15**: feature map size lowered from 32³ to **16³**.
> Rationale:
> 1. **Inference accessibility** — at 16³ feature map, standard CorrBlock uses only
>    67 MB / sample, so any consumer GPU (RTX 3060+ / 4060+ / lab A4000) can deploy
>    the model.  32³ would require 4.29 GB per sample, restricting deployment to
>    H100/A100/GH200-class hardware.
> 2. **Matches VolRAFT baseline scale** — VolRAFT (CVPR 2024) trained at feature
>    map ~8×10×10 (60×80×80 input / 1/8 downsample).  Our previous 23 experiments
>    were at 16³ feature map.  Phase-1 staying at 16³ keeps direct comparability.
> 3. **Local-development viable** — 4070 Ti 12 GB can train at 16³ in reasonable
>    wall-clock; 32³ requires GH200 because consumer GPUs hit OOM-edge slowdown.
>
> Trade-off: the 1/1 encoder + size 16 input case has only ~4 beads per dense
> sample (16³ × 1.0/1000 = 4.1).  This row is kept but is expected to be one of
> the failure modes in Phase-1 (drives the "viability threshold" finding).

| Variable | Value | Rationale |
|----------|-------|-----------|
| Feature map size | **16³** (all experiments) | Consumer-GPU deployable; matches previous baselines; viable on 4070 Ti for development |
| Pyramid levels | 2 | Small displacement regime; primarily affects search range, not accuracy |
| Search radius | 4 | Standard RAFT setting |
| GRU iterations | 12 (train) / 24 (inference) | Standard RAFT practice |
| Displacement type | Small-magnitude affine or smooth B-spline | Isolate feature-architecture coupling from displacement complexity |
| Optimizer | AdamW (lr=1e-4, weight_decay=5e-5) | Consistent across all experiments |
| Scheduler | CyclicLR (triangular2) | Matching VolRAFT |
| Mixed precision | AMP (float16) | Standard |

## 3. Independent Variables

### Data Side (rows of experiment matrix)
| Level | Bead Radius (voxels) | Density | Notes |
|-------|---------------------|---------|-------|
| R2-sparse | 2 | sparse | TBD: exact density values |
| R2-medium | 2 | medium | |
| R2-dense | 2 | dense | |
| R4-sparse | 4 | sparse | |
| R4-medium | 4 | medium | |
| R4-dense | 4 | dense | |
| R6-sparse | 6 | sparse | |
| R6-medium | 6 | medium | |
| R6-dense | 6 | dense | |
| **mixed** | 2+4+6 mixed | sparse+medium+dense mixed | Samples from all 9 configs combined |

### Model Side (columns of experiment matrix)
| Downsampling | Input Size | Feature Map | 6D Corr Volume | Approx. bead count (dense) |
|-------------|-----------|-------------|----------------|---------------------------|
| 1/1 | 16³ | 16³ | 67 MB | ~4 (low, expected fail) |
| 1/2 | 32³ | 16³ | 67 MB | ~33 |
| 1/4 | 64³ | 16³ | 67 MB | ~262 |
| 1/8 | 128³ | 16³ | 67 MB | ~2,097 |

### Displacement magnitude (per encoder, at 16³ feature map)
`fm_target ~ U(0.3, 3.0)` voxels (constant across all encoders),
`input_max_disp = fm_target × downsample_factor`:

| Encoder | Input disp range | Generated data status |
|---------|------------------|----------------------|
| 1/1 + size 16 | [0.3, 3] | ✓ crop from `r*_medium_size32` at runtime |
| 1/2 + size 32 | [0.6, 6] | ❌ existing `size32` is at [0.3, 3] — regenerate needed |
| 1/4 + size 64 | [1.2, 12] | ❌ existing `size64` is at [0.6, 6] — regenerate needed |
| 1/8 + size 128 | [2.4, 24] | ❌ not yet generated |

## 4. Experiment Phases

### Phase 1: Core Matrix — Feature-Architecture Coupling (40 training runs)

**Goal**: Reveal coupling between feature characteristics and downsampling factor.

**Training**: 10 data configs × 4 downsampling factors = **40 models**

**Evaluation**: Each model tested on ALL 9 single-config test sets → **360 evaluations** (inference only)

**Key output**: Generalization matrix per downsampling factor:

```
              Test set
              R2-sp  R2-md  R2-dn  R4-sp  R4-md  R4-dn  R6-sp  R6-md  R6-dn
Train R2-sp  [ in  ] [ cross ...                                              ]
Train R2-md  [     ] [ in   ] [    ...                                        ]
...
Train mixed  [     ] [      ] [    ...                          ...    [ ... ] ]
```

- Diagonal (in-distribution) vs off-diagonal (cross-distribution) gap = fragility measure
- Mixed-trained models: are they universally better or jack-of-all-trades?
- Feature map visualizations: direct comparison across downsampling factors

**Effective feature scale analysis**:
| Radius | 1/1 effective | 1/2 effective | 1/4 effective | 1/8 effective |
|--------|--------------|--------------|--------------|--------------|
| 2 | 2.0 | 1.0 | 0.5 | 0.25 |
| 4 | 4.0 | 2.0 | 1.0 | 0.5 |
| 6 | 6.0 | 3.0 | 1.5 | 0.75 |

Expect a "viability threshold" in effective feature scale below which training fails.

### Phase 2: Training Strategy — Cutout & Uncertainty Loss (6-8 training runs)

**Goal**: Show that cutout and NLL loss improve robustness independently of data config.

**Assumption**: Cutout and loss function effects are approximately independent of
bead density/radius/downsampling (to be validated by spot checks).

**Design**: Select 2-3 representative data configs from Phase 1 results:
- One "easy" config (large beads, dense, moderate downsampling)
- One "hard" config (small beads, sparse, aggressive downsampling)
- One "mixed" config

For each, train with:
- Baseline (EPE loss, no cutout) — already from Phase 1
- + Cutout only
- + NLL loss only
- + Cutout + NLL

**Evaluation**: Same 9-config generalization matrix as Phase 1.

**Key output**: Does cutout/NLL reduce the diagonal vs off-diagonal gap?

### Phase 3: Comparison & Validation (3-5 runs)

**Goal**: Benchmark against baseline and test real-data generalization.

1. **VolRAFT baseline comparison**: Best RAFTcorr3D config vs vanilla VolRAFT architecture
   on the same test sets.

2. **Real data test**: Best config applied to Franck / JinYang experimental data
   (ground truth: synthetic displacement applied to real reference images).
   - Even if results are poor, this is valuable as an honest limitation discussion.

3. **Feature map visualization**: Side-by-side comparison of extracted features across
   downsampling factors to provide visual/interpretive understanding.

## 5. Compute Plan

**Platform**: TACC Vista GH200 (96 GB HBM3)

| Phase | Training Runs | Est. GPU-hours per run | Total GPU-hours |
|-------|--------------|----------------------|-----------------|
| Phase 1 | 40 | TBD (depends on epochs) | TBD |
| Phase 2 | 8 | TBD | TBD |
| Phase 3 | 5 | TBD | TBD |
| **Total** | **~53** | | |

**Evaluation**: 360+ inference runs (cheap, minutes each on GPU).

**Data generation**: 10 synthetic dataset configs × 4 input sizes = up to 40 datasets
(some may be shared across configs).

## 6. Metrics

| Metric | Purpose |
|--------|---------|
| EPE (End-Point Error) | Primary accuracy metric |
| Generalization gap (off-diag EPE / diag EPE) | Feature robustness measure |
| Uncertainty-error correlation | Quality of uncertainty map (Phase 2) |
| Training convergence speed | Epochs to reach target EPE |
| Inference memory & time | Practical deployment metric |

## 7. Paper Narrative Structure

1. **Introduction**: DVC challenges (feature diversity), motivation from RAFT-DIC
2. **Method**: RAFTcorr3D architecture, cutout strategy, uncertainty-aware loss
3. **Experiments**:
   - Phase 1 → "Feature-architecture coupling is real and significant"
   - Phase 2 → "Cutout and NLL improve robustness"
   - Phase 3 → "Comparison with VolRAFT + real data discussion"
4. **Discussion**: Effective feature scale hypothesis, limitations (domain gap),
   why MoL fails for DVC (unimodal vs bimodal), future directions
5. **Conclusion**: Practical recommendations for DVC practitioners

## 8. Open Questions

- [ ] Exact density values for sparse/medium/dense (to be determined empirically)
- [ ] Training epochs: 300 full or 100-150 screening first?
- [ ] 256³ data generation feasibility and storage for 1/8 downsampling experiments
- [ ] Pyramid levels: fix at 2, or briefly validate 2 vs 4?
- [ ] Batch size feasibility for 32³ feature map on GH200 (target: 4+)

## 9. Naming Convention

**Method**: RAFTcorr3D
**Naming lineage**: RAFTcorr (2D DIC, previous paper) → RAFTcorr3D (3D DVC, this paper)
**Future**: stereo-RAFTcorr (3D-DIC, reserved)

---

*This plan will be refined as Phase 1 results become available.*
