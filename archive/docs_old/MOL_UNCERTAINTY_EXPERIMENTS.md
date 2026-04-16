# Mixture-of-Laplace (MoL) Uncertainty Estimation Experiments

## Summary

This document records the experiments of adapting the Mixture-of-Laplace (MoL) loss from SEA-RAFT (ECCV 2024, Best Paper Award Candidate) to RAFT-DVC for 3D volumetric Digital Volume Correlation. All three parameter configurations of `b1` resulted in **alpha collapse**, revealing a fundamental incompatibility between the MoL formulation and the error distribution of sparse-feature volumetric data.

---

## 1. Background: SEA-RAFT's MoL Loss

### 1.1 Motivation

Standard L1/L2 loss treats all pixels equally. In optical flow, occluded pixels have inherently unpredictable flow, and forcing the network to minimize error on these pixels can hurt overall performance. SEA-RAFT proposes a Mixture-of-Laplace (MoL) loss that automatically separates "ordinary" pixels (reliable flow) from "ambiguous" pixels (occluded/featureless).

### 1.2 Mathematical Formulation

The MoL loss per voxel per flow component is:

$$\mathcal{L}_{\text{MoL}} = -\log\left[\alpha \cdot \text{Lap}(e; b_1) + (1 - \alpha) \cdot \text{Lap}(e; b_2)\right]$$

where $\text{Lap}(x; b) = \frac{1}{2b}\exp\left(-\frac{|x|}{b}\right)$ is the Laplace density, and:

- **$b_1 = \exp(\beta_1)$**: Fixed scale for the "ordinary" component. SEA-RAFT sets $\beta_1 = 0$, so $b_1 = 1.0$. This makes the ordinary component equivalent to **L1 loss**.
- **$b_2 = \exp(\beta_2)$**: Learned scale for the "ambiguous" component. $\beta_2$ is clamped to $[0, 10]$, ensuring $b_2 \geq 1$ (always wider than $b_1$).
- **$\alpha = \sigma(\text{logit}_\alpha)$**: Learned mixing weight. $\alpha \to 1$ means "ordinary" (use L1), $\alpha \to 0$ means "ambiguous" (use learned $b_2$).

All parameters ($\beta_2$, $\text{logit}_\alpha$) are predicted per-voxel by a dedicated uncertainty head, and are updated at every RAFT iteration. The final sequence loss is:

$$\mathcal{L}_{\text{all}} = \sum_{i=1}^{N} \gamma^{N-i} \cdot \mathcal{L}_{\text{MoL}}^{(i)}$$

### 1.3 SEA-RAFT Source Code vs Paper

We verified our implementation against both the paper and the [official source code](https://github.com/princeton-vl/SEA-RAFT). Key findings:

| Aspect | Paper Description | Actual Code | Our Implementation |
|---|---|---|---|
| $\beta_1$ | Fixed at 0 | `clamp(raw, min=0, max=0)` = 0 | `b1=1.0` (equivalent) |
| $\beta_2$ range | $[0, 10]$ | `clamp(raw, min=0, max=10)` | `log_b2.clamp(0, 10)` |
| Mixing weights | $\alpha$, $1-\alpha$ | 2 raw log-weights + softmax | 1 logit + sigmoid (equivalent) |
| $\beta_2$ per flow dim | Not specified | 1 shared (u,v) | 3 separate (dx,dy,dz) |
| Head architecture | Not specified | Single FlowHead (6ch: 2 flow + 4 info) | Separate FlowHead (3ch) + MoLUncertaintyHead (4ch) |
| NLL computation | In loss function | Inside model forward() | In loss function |

**Config**: `var_min=0, var_max=10, use_var=true` in all SEA-RAFT training configs.

---

## 2. Experiment Setup

### 2.1 Common Settings

All experiments share:

| Setting | Value |
|---|---|
| Dataset | `synthetic_confocal_128_v1` (128^3, ~1000 beads, affine+bspline deformations) |
| Model | RAFT-DVC, encoder 1/8, corr_levels=4, corr_radius=4 |
| Epochs | 300 |
| Optimizer | AdamW (lr=1e-4, weight_decay=5e-5) |
| Scheduler | CyclicLR (base_lr=5e-5, max_lr=1e-4, triangular2) |
| AMP | Enabled (float16) |
| Gamma | 0.8 (sequence loss weighting) |
| Uncertainty head | MoLUncertaintyHead: 4 channels (3 log_b2 + 1 logit_alpha) |
| logit_alpha init | bias = 2.0 (alpha starts at ~0.88) |
| log_b2 init | bias = 0.0 (b2 starts at 1.0) |

### 2.2 Experiment Matrix

| Experiment | Config File | b1 | Batch Size | GPU | Output Directory |
|---|---|---|---|---|---|
| MoL v1 | `confocal_128_v1_1_8_p4_r4_uncertainty_mol_v1.yaml` | 0.5 | 1 | 4070 Ti | `outputs/training/confocal_128_v1_1_8_p4_r4_uncertainty_mol_v1/` |
| MoL v2 | `confocal_128_v1_1_8_p4_r4_uncertainty_mol_v2.yaml` | 1.0 | 4 | 5090 | `outputs/training/confocal_128_v1_1_8_p4_r4_uncertainty_mol_v2/` |
| MoL v3 | `confocal_128_v1_1_8_p4_r4_uncertainty_mol_v3.yaml` | 0.8 | 4 | 5090 | `outputs/training/confocal_128_v1_1_8_p4_r4_uncertainty_mol_v3/` |

Note: v1 used batch_size=1 due to 4070 Ti VRAM limitation. v2/v3 used batch_size=4 on 5090 (batch_size=5 caused catastrophic slowdown due to MoL's extra computation).

### 2.3 NLL Baseline (for reference)

| Experiment | Config File | Output Directory |
|---|---|---|
| NLL (single Laplace) | `confocal_128_v1_1_8_p4_r4_uncertainty.yaml` | `outputs/training/confocal_128_v1_1_8_p4_r4_uncertainty/` |

The NLL baseline uses `UncertaintyHead` (3 channels, log_b only) with loss $\mathcal{L} = |e| \cdot \exp(-\log b) + \log b$.

---

## 3. Results

### 3.1 MoL v1: b1 = 0.5

**Observation**: Alpha collapsed to 1 rapidly (within ~100 epochs). log_b2 dropped to approximately -6 (raw, unclamped).

**Interpretation**: With $b_1 = 0.5$, the ordinary component is **always tighter** than any $b_2 \geq 1.0$ (the clamp minimum). The Laplace density $\text{Lap}(e; 0.5) > \text{Lap}(e; b_2 \geq 1)$ for all per-component errors $|e| < 2.77$. Since virtually all voxels have per-component errors well below this threshold, the ordinary component dominates everywhere, and alpha is driven to 1.

### 3.2 MoL v2: b1 = 1.0 (SEA-RAFT default)

**Observation**: Alpha collapsed to ~0.05 by halfway through training. log_b2 stabilized around -2 (raw). EPE ~0.8 at convergence. Uncertainty correlation ~0.5.

**Interpretation**: With $b_1 = 1.0 = b_{2,\min}$, the two components are **identical** at the clamp boundary. The model learns $b_2$ slightly above 1.0 for some voxels, and for any voxel where $b_2 > 1$, there exists a crossover error below which the components are essentially equivalent. Since the model's per-component errors are often in the range where $b_2 > 1$ provides slightly better likelihood (errors are large enough, ~1-2 vx per component), the gradient pushes alpha toward 0 for most voxels.

### 3.3 MoL v3: b1 = 0.8

**Observation**: Alpha initially stabilized around 0.5 (at ~100 epochs), then slowly drifted upward toward 0.9-1.0 over the remaining training. log_b2 continued decreasing below 0 (raw). The uncertainty map became **binary**: most voxels had b2 clamped to 1.0 (raw log_b2 < 0), with only a few small regions reaching b2 = exp(10).

**Interpretation**: The crossover point between $\text{Lap}(e; 0.8)$ and $\text{Lap}(e; 1.0)$ is at per-component $|e| \approx 0.89$. With EPE of 1-2 voxels, the per-component error for most voxels is $\approx 0.58$-$1.15$, straddling this crossover. Initially the mixture finds a balance, but as training progresses and errors decrease, more voxels fall below the crossover → alpha slowly drifts to 1.

### 3.4 Summary Table

| Experiment | b1 | Alpha Collapse Direction | Speed | Uncertainty Map Quality |
|---|---|---|---|---|
| v1 | 0.5 | $\alpha \to 1$ | Fast (~100 ep) | Degenerate (uniform) |
| v2 | 1.0 | $\alpha \to 0$ | Fast (~150 ep) | Degenerate (uniform) |
| v3 | 0.8 | $\alpha \to 1$ | Slow (~200+ ep) | Binary (most b2=1, few b2=exp(10)) |

**Key finding**: No value of $b_1$ produces stable, non-degenerate MoL training on our data.

---

## 4. Root Cause Analysis

### 4.1 The Impossible Dilemma of Fixed b1

The MoL formulation with fixed $b_1$ and clamped $b_2 \geq b_{2,\min}$ creates an inherent instability:

$$b_1 < b_{2,\min} \implies \text{Lap}(e; b_1) > \text{Lap}(e; b_2) \text{ for small } e \implies \alpha \to 1$$
$$b_1 = b_{2,\min} \implies \text{no preference} \implies \alpha \text{ drifts to 0 (when errors > crossover)}$$
$$b_1 > b_{2,\min} \implies \text{impossible (b2 is clamped to} \geq b_{2,\min}\text{)}$$

There is **no stable equilibrium** for $\alpha$ under this parameterization when the error distribution is unimodal.

### 4.2 Why SEA-RAFT Works but RAFT-DVC Doesn't

The fundamental difference lies in the **error distribution**:

#### Natural images (SEA-RAFT's domain):

```
Error distribution: BIMODAL

    ^
    |  ████
    |  █████
    |  ██████                           ██
    |  ███████                         ████
    |  ████████                       ██████
    +--+--------+--------------------+--------+--> error
       0     ~0.5                   ~10     ~50
       Textured pixels              Occluded pixels
       (majority, ~85%)             (minority, ~15%)
```

- **Hard occlusions** create a clear second mode: occluded pixels have **no valid correspondence** in the other frame. The model literally cannot predict the correct flow.
- The two Laplace components map cleanly onto the two modes.
- $\alpha \approx 1$ for textured pixels, $\alpha \approx 0$ for occluded pixels.

#### Confocal volumetric data (our domain):

```
Error distribution: UNIMODAL (skewed)

    ^
    |  ██
    |  ████
    |  ██████
    |  █████████
    |  ████████████████
    |  █████████████████████████████████
    +--+-----+-----+-----+-----+-----+--> error
       0   0.5   1.0   1.5   2.0   3.0+
       Near beads -----> Far from beads
```

- There are **no hard occlusions** in DVC. The GT deformation field is defined everywhere (it's synthetically generated from a continuous displacement field).
- Dark regions between beads lack features, but the model can **partially** predict flow through GRU propagation and correlation search radius (±32 voxels at full resolution).
- The error transitions **continuously** from near-bead regions (low error) to far-from-bead regions (moderate error). There is no sharp boundary.

### 4.3 Why Dark Regions ≠ Hard Occlusions

Despite both being "featureless", there are critical differences:

| Property | Hard Occlusion (2D flow) | Dark Region (3D DVC) |
|---|---|---|
| GT flow | Undefined or from inpainting | Well-defined (continuous deformation field) |
| Model information | **Zero** (object invisible) | **Partial** (GRU propagation, corr search radius) |
| Typical per-component error | 10-50 px | 0.5-2 vx |
| Spatial boundary | Sharp (object edge) | Gradual (intensity falloff) |
| Fraction of volume | ~10-15% (natural images) | ~96% (sparse beads at ~4% fill) |

The confocal data has a much higher fraction of featureless voxels (~96%), but their errors are much smaller than true occlusions because:
1. The correlation search radius (4 at 1/8 res = 32 voxels at full res) is large relative to inter-bead spacing (~12-15 voxels).
2. The GRU iterative refinement propagates flow information from bead regions into dark regions.
3. The deformations are smooth, so dark-region flow is largely predictable by interpolation.

### 4.4 The Clamp Gradient Problem

An additional issue: the `log_b2.clamp(min=0)` operation kills gradients for voxels where the model wants $b_2 < 1$ (i.e., raw $\beta_2 < 0$). During training:

1. Model predicts raw $\beta_2 = -2$ for a voxel (it "wants" $b_2 = 0.14$)
2. Clamp forces $\beta_2 = 0$ ($b_2 = 1.0$) for loss computation
3. Gradient w.r.t. $\beta_2$ is **zero** (clamp kills it)
4. The model cannot learn from this voxel's $\beta_2$ at all

This affects the **majority** of voxels (mean raw log_b2 was consistently negative across all experiments), creating a binary uncertainty map: most voxels are stuck at b2=1 (clamped), a few outlier voxels have b2>>1.

---

## 5. Implementation Notes

### 5.1 Float16 Subnormal Issue (v1)

The initial MoL implementation suffered from catastrophic GPU slowdown (~100x) in mixed precision. Root cause: computing `exp(-|e|/b) / (2b)` directly produces subnormal float16 values for large errors, triggering the GPU's slow subnormal handling path.

**Fix**: Rewrite the entire MoL loss in **log-space** using `logaddexp`:

```python
log_lap1 = -abs_err / b1 - log(2 * b1)
log_lap2 = -abs_err / b2 - log(2) - log_b2
nll = -torch.logaddexp(log_alpha + log_lap1, log_1m_alpha + log_lap2)
```

This keeps all intermediate values in regular float16 range. No float32 cast needed.

### 5.2 Batch Size Cliff (v1 → v2)

MoL adds ~20% computation overhead (extra head + logaddexp). On 5090:
- batch_size=4: ~3.5s/iter (acceptable)
- batch_size=5: catastrophic slowdown (memory thrashing)

All v2/v3 experiments used batch_size=4.

### 5.3 Validation Logging Discrepancy

The validation code logs **raw** (unclamped) model output for `mean_log_b`:
```python
total_log_b += unc[:, :3].mean().item()  # raw output, no clamp
```

While the loss function applies `clamp(min=0, max=10)`. This means TensorBoard shows `mean_log_b < 0`, which reflects the model's "desired" b2 (less than 1), not the value actually used in training.

---

## 6. Conclusions and Recommendations

### 6.1 MoL is Not Suitable for DVC Data (Without Modification)

The MoL formulation assumes a **bimodal** error distribution with a clear separation between "ordinary" and "ambiguous" pixels. This assumption holds for natural image optical flow (hard occlusions create a second mode) but fails for volumetric DVC data (continuous error transition, no hard boundary).

### 6.2 Proposed Solution: Cutout + MoL

To artificially create the bimodal distribution that MoL requires:

- **Cutout augmentation**: Randomly zero out 15-40% of the volume during training
- In cutout regions: correlation = 0, GRU has no information → large errors → $\alpha \to 0$
- In intact regions: normal matching → small errors → $\alpha \to 1$

This forces the model to learn a meaningful $\alpha$ map that transfers to real featureless regions at inference time. Experiment: `confocal_128_v1_1_8_p4_r4_cutout_uncertainty_mol_v4.yaml`.

### 6.3 Alternative: Plain NLL with Floor

If Cutout+MoL still fails, the fallback is single-component Laplace NLL with a minimum floor:

```python
log_b = log_b.clamp(min=-2)  # b >= 0.135, prevents collapse to 0
```

This produces a **continuous** uncertainty map without the bimodal assumption, at the cost of losing the explicit "ordinary vs ambiguous" classification.

### 6.4 Broader Implications for DVC Uncertainty

The failure of MoL on DVC data highlights a fundamental property of Digital Volume Correlation:

1. **DVC is NOT optical flow with a different modality.** The physics of the problem differs fundamentally — there are no occlusions, no scene changes, no lighting variations. The sole challenge is **sparse features** in a continuous deformation field.

2. **Uncertainty in DVC is inherently continuous**, not categorical. A voxel's reliability depends on its distance to the nearest feature, the local feature density, and the smoothness of the deformation — all continuous quantities. Binary "reliable/unreliable" classification is too coarse.

3. **Methods designed for natural image pathologies (occlusion, out-of-boundary) do not transfer directly to DVC.** Adaptations must account for the different error distribution and the absence of hard anomalies.

---

## 7. File References

### Config Files
- `configs/training/confocal_128_v1_1_8_p4_r4_uncertainty.yaml` — NLL baseline
- `configs/training/confocal_128_v1_1_8_p4_r4_uncertainty_mol_v1.yaml` — MoL v1 (b1=0.5)
- `configs/training/confocal_128_v1_1_8_p4_r4_uncertainty_mol_v2.yaml` — MoL v2 (b1=1.0)
- `configs/training/confocal_128_v1_1_8_p4_r4_uncertainty_mol_v3.yaml` — MoL v3 (b1=0.8)
- `configs/training/confocal_128_v1_1_8_p4_r4_cutout_uncertainty_mol_v4.yaml` — Cutout+MoL v4

### Source Code
- `src/training/loss.py` — `MoLSequenceLoss`, `NLLSequenceLoss`
- `src/core/update.py` — `MoLUncertaintyHead`, `UncertaintyHead`
- `src/core/raft_dvc.py` — `RAFTDVC.forward()` uncertainty output path
- `scripts/train_confocal.py` — `setup_strategies()`, `train_epoch()`, `validate()`

### Training Outputs
- `outputs/training/confocal_128_v1_1_8_p4_r4_uncertainty_mol_v1/` — v1 results
- `outputs/training/confocal_128_v1_1_8_p4_r4_uncertainty_mol_v2/` — v2 results
- `outputs/training/confocal_128_v1_1_8_p4_r4_uncertainty_mol_v3/` — v3 results

### Reference Paper
- `2024_ECCV_Princeton_Yihan_Wang_SEA-RAFT.pdf` — SEA-RAFT paper (in project root)
- GitHub: https://github.com/princeton-vl/SEA-RAFT

---

*Document created: 2026-02-11*
*Related to: RAFT-DVC uncertainty estimation, sparse feature handling*
