# Headless batch wrappers for ALDVC and FE-Global-DVC

Non-interactive (`matlab -batch`) wrappers around the two MATLAB DVC codebases
in this repo (`ALDVC\` and `Global_DVC\`, both by Jin Yang). Displacement
only — no strain, no plotting, no prompts. Original sources are **not
modified**; everything added lives in `benchmark\`.

Smoke-tested end-to-end on 2026-07-13 (R2025b, Windows 11): all three methods
reach mean EPE < 0.35 voxel on a clean synthetic 64^3 pair (see bottom).

## Quick start

```bat
cd traditional_DIC_codes\benchmark
"C:\Program Files\MATLAB\R2025b\bin\matlab.exe" -batch "aldvc_headless('cfgs/smoke_aldvc.json')"
"C:\Program Files\MATLAB\R2025b\bin\matlab.exe" -batch "globaldvc_headless('cfgs/smoke_global.json')"
```

Run from `benchmark\` (or any cwd if cfg paths are absolute; relative paths in
the cfg are resolved against MATLAB's cwd). Forward slashes work fine.

Full smoke-test pipeline (data gen -> both wrappers -> EPE evaluation):

```bat
python make_smoke_pair.py
matlab -batch "aldvc_headless('cfgs/smoke_aldvc.json')"
matlab -batch "globaldvc_headless('cfgs/smoke_global.json')"
python eval_smoke.py        & rem exits 0 on SUCCESS (all mean EPE < 0.5 vox)
```

Python: `C:\Users\zt3323\.conda\envs\raft-dvc-2\python.exe` (numpy+scipy).

## Config schemas (JSON, or .mat with a struct named `cfg`)

`aldvc_headless(cfgFile)`:

```json
{
  "refFile":     "data/smoke_ref.mat",
  "defFile":     "data/smoke_def.mat",
  "winsize":     [16, 16, 16],
  "winstepsize": [8, 8, 8],
  "outFile":     "data/smoke_out_aldvc.mat"
}
```

`globaldvc_headless(cfgFile)`:

```json
{
  "refFile":     "data/smoke_ref.mat",
  "defFile":     "data/smoke_def.mat",
  "winstepsize": [8, 8, 8],
  "alpha":       0.8,
  "outFile":     "data/smoke_out_global.mat"
}
```

Volume `.mat` files: the loader takes the FIRST variable in the file (cell
`{1}` or bare 3-D array), cast to double, z-score normalized over the ROI
(full volume). Use `npz_to_mat.py` to convert our npz samples
(`python npz_to_mat.py sample.npz out_dir` writes `<prefix>_ref.mat` /
`<prefix>_def.mat`, each with a single variable `vol`, no permutation).

## Outputs

Both wrappers `save(outFile, ...)`:

| Variable         | Meaning                                                        |
|------------------|----------------------------------------------------------------|
| `U_aldvc`        | ALDVC ADMM result, 3*N x 1 interleaved `[u1;v1;w1;u2;...]`     |
| `U_local`        | pure local subset IC-GN result (ALDVC Sec 4), same layout      |
| `U_global`       | FE-Global-DVC result (globaldvc only), same layout             |
| `coordinatesFEM` | N x 3 node coordinates, MATLAB 1-based voxel units, ndgrid at `winstepsize` spacing |
| `winstepsize`    | 1x3                                                            |
| `timing`         | struct, per-stage wall-clock seconds                           |

## Axis convention (measured — see AXIS_CONVENTION.md)

Identity mapping, positive sign:

- numpy axis 0 (MATLAB dim 1) -> `u = U(1:3:end)` -> `coordinatesFEM(:,1)`
- numpy axis 1 (MATLAB dim 2) -> `v = U(2:3:end)` -> `coordinatesFEM(:,2)`
- numpy axis 2 (MATLAB dim 3) -> `w = U(3:3:end)` -> `coordinatesFEM(:,3)`

Positive component = feature motion toward increasing numpy index.
`coordinatesFEM` is 1-based: numpy position = `coordinatesFEM - 1`.

Generator convention `def[x] = ref[x + u_gen(x)]` implies the DVC-measured
displacement obeys `U(X) = -u_gen(X + U(X))` (NOT simply `-u_gen(X)`;
first-order evaluation costs up to ~|u|*|grad u| voxels). `eval_smoke.py`
resolves this with a fixed-point iteration on the analytic field.

## Pinned parameters

`aldvc_headless`: interpMethod='cubic', clusterNo=1, initFFTMethod='bigxcorr',
trackingMode='cumulative', Subpb2FDOrFEM='finiteDifference', ICGNtol=1e-2,
Subpb1ICGNMaxIterNum=50, ADMMtol=1e-2, ALVarMu=1e-3, single
beta = 1e-2*mean(winstepsize)^2*ALVarMu (no L-curve sweep), 2 ADMM outer
iterations, gridRange = full volume, median-test threshold 2, no qDIC removal.

`globaldvc_headless`: winsize = winstepsize+[6,6,6] (ReadImage3 convention),
alphaList = [alpha] (kills the main-script line-126 L-curve override),
maxIter=20, tol=1e-2, GaussPtOrder=2, ClusterNo=1, InitFFTMethod='bigxcorr',
gridRange = full volume, median-test threshold 2.

## Parameter table per bead size (our synthetic datasets)

| Dataset | ALDVC winsize | ALDVC winstepsize | Global element size (winstepsize) | Global alpha (start) |
|---------|---------------|-------------------|-----------------------------------|----------------------|
| r2 (small beads)  | [12,12,12] | [6,6,6]   | 8  | 0.8 |
| r4 (medium beads) | [16,16,16] | [8,8,8]   | 10 | 1.0 |
| r8 (large beads)  | [24,24,24] | [12,12,12]| 12 | 1.2 |

alpha guidance (measured on z-score-normalized clean 64^3 blob volumes,
element size 8): alpha = 10*step = 80 shrinks the field ~6x (mean EPE 2.22),
alpha = 1*step = 8 still over-smooths (EPE 1.31), **alpha = 0.1*step = 0.8
passes (EPE 0.27)**. Start at 0.1*element size; if results look noisy/biased
sweep [1e-2 .. 1e1]*mean(winstepsize) (the original code's L-curve range).

## Known pitfalls

1. **`clusterNo`/`ClusterNo` must be 1.** `LocalICGN3.m:35` has a
   `case 0 || 1` bug (evaluates to `case true`), so 0 falls into the `parfor`
   branch and tries to open a parallel pool. Both wrappers pin 1.
2. **Do not use `strainCalculationMethod==3`** (`ComputeStrain3.m:91` typo
   crash). Not relevant here — wrappers never compute strain.
3. **`Global_DVC\ba_interp3.mexw64` is a broken build** on this machine
   ("DLL initialization routine failed" under R2025b). `benchmark\` carries a
   copy of the working `ALDVC\ba_interp3.mexw64` and is placed first on the
   path (`addpath` prepends, so the wrappers add `benchmark\` last). No
   `mex -O ba_interp3.cpp` / `setenv MW_MINGW64_LOC` needed at startup.
4. **`RemoveOutliers3` blocks headless runs.** The Global_DVC version has
   unconditional `input()` prompts; the ALDVC version prompts when the median
   threshold is 0 (which is what `main_ALDVC.m` passes in cumulative mode).
   Both wrappers call `benchmark\RemoveOutliers3_headless.m` — an exact port
   of the non-interactive code path (median/universal-outlier test with fixed
   threshold 2 + `inpaint_nans3`), no figures, no prompts.
5. **`main_FE_GlobalDVC.m:126` silently overrides `alphaList`** with an
   L-curve sweep — pinned to the cfg alpha in the wrapper.
6. **waitbars/figures are safe** in `matlab -batch` on R2025b/Windows
   (created invisibly); the internal waitbars in `funIntegerSearch3*`,
   `LocalICGN3`, `Subpb13`, `funGlobalICGN3` were left untouched.
7. **Border margins eat small volumes.**
   - ALDVC nodes start ~`0.5*winsize+4` voxels from each face
     (64^3, win 16/step 8 -> 6^3 = 216 nodes spanning [12,52]).
   - Global-DVC clips at ~`1.4*winsize` where winsize = step+6
     (64^3, step 8 -> only 4^3 = 64 nodes spanning [19,44]).
   Budget volume size accordingly; on 128^3 this is negligible.
8. **`np.roll` wraps content** across faces — fine for calibration
   (interior nodes + median stats), do not use rolled pairs for accuracy
   claims near boundaries.
9. **Interleaved output layout**: `U` is `[u1;v1;w1;u2;...]` on
   `coordinatesFEM` nodes (x fastest, ndgrid order). Reshape per component
   with `U(1:3:end)` etc.
10. **Requires Image Processing Toolbox** (`padarray`, `imgaussfilt3`) —
    present and verified on this machine.
11. **Global-DVC integer search degenerates on small volumes** (found
    2026-07-14 on 64^3 benchmark scenarios). In
    `Global_DVC\func\funIntegerSearch3Mg.m` ('bigxcorr' branch) the strict
    open-interval filter on multigrid block centers can leave `indxyz`
    EMPTY, after which `scatteredInterpolant` silently returns EMPTY grids
    and the run crashes later in `inpaint_nans3` (`NA(3)` index error).
    `benchmark\funIntegerSearch3Mg.m` is a patched shadow copy (benchmark
    is first on the path; ALDVC's copy of this function is byte-identical
    and its wrapper does not call it): when starved it relaxes the filter
    to all points and falls back to a constant median fill if a 3-D
    triangulation is impossible. Additionally `RemoveOutliers3_headless`
    now skips the median test (pass-through) when the node grid has a
    singleton/missing dimension (e.g. 64^3 with win 24) instead of
    crashing. Healthy paths are unchanged — smoke test reproduces the
    table below exactly.

## Measured smoke-test results (64^3 clean pair, sinusoidal field, max |u| = 3.46 vox)

| Method                        | mean EPE | median | max   | nodes | runtime (stages)                          |
|-------------------------------|----------|--------|-------|-------|-------------------------------------------|
| U_local  (ALDVC local IC-GN)  | 0.322    | 0.310  | 0.801 | 216   | integer 4.1 s, local IC-GN 1.7 s          |
| U_aldvc  (ALDVC ADMM)         | 0.308    | 0.310  | 0.455 | 216   | + global/ADMM 2.9 s (2 outer iterations)  |
| U_global (FE Global, a=0.8)   | 0.270    | 0.251  | 0.605 | 64    | integer 4.0 s, gradients 0.01 s, ICGN 3.4 s |

Axis-calibration runs (roll +3): recovered +3.000 on the matching component,
|cross-talk| < 0.001 (see `logs/calib_axis*.log`).

## Bonus: real r8 sample (128^3, data_paper1_axis2, max |flow| = 15.9 vox)

`sample_00000.npz` converted with `npz_to_mat.py`, run with the r8 row of the
parameter table (`cfgs/real_r8_*.json`, evaluated by `eval_real_r8.py`):

| Method                         | mean EPE | median | nodes | runtime (stages)                            |
|--------------------------------|----------|--------|-------|----------------------------------------------|
| U_local  (win 24 / step 12)    | 0.598    | 0.400  | 729   | integer 5.0 s, local 7.2 s                   |
| U_aldvc                        | 0.530    | 0.381  | 729   | + global/ADMM 10.3 s                         |
| U_global (step 12, alpha 1.2)  | 0.647    | 0.493  | 343   | integer 5.5 s, gradients 0.03 s, ICGN 15.7 s |

IMPORTANT (measured): the npz `flow` key is ALREADY the ref->def feature
displacement in DVC convention — `U_dvc ~= +flow` at the nodes. Do NOT apply
the backward-warp inversion `U=-flow(X+U)` to it (that hypothesis is off by
~5 vox median). See AXIS_CONVENTION.md. ~11-18% of local subsets fail IC-GN
on this large-|U| sparse-texture sample and get inpainted, which drives the
max EPE up (7.5 vox for U_local); median stays ~0.4.

## Files

| File | Purpose |
|------|---------|
| `aldvc_headless.m` | ALDVC Sections 1-6 as a function (cfg-driven) |
| `globaldvc_headless.m` | FE-Global-DVC Sections 1-4 as a function |
| `RemoveOutliers3_headless.m` | patched non-interactive outlier removal (shadows nothing; called explicitly) |
| `funIntegerSearch3Mg.m` | patched shadow copy fixing the small-volume degenerate-grid crash (pitfall 11) |
| `ba_interp3.mexw64` | working mex copy (from ALDVC) shadowing the broken Global_DVC build |
| `cfgs/bench/` | per-scenario cfg JSONs written by `scripts/paper1/benchmark_suite.py gen-cfgs` |
| `npz_to_mat.py` | RAFT-DVC npz (I1/I2) -> ref/def .mat (variable `vol`, no permutation) |
| `make_calibration_pair.py`, `check_calibration.py` | axis-convention measurement |
| `make_smoke_pair.py`, `eval_smoke.py` | smoke-test data gen + EPE evaluation |
| `eval_real_r8.py` | EPE evaluation of the real r8 sample against its npz `flow` GT |
| `AXIS_CONVENTION.md` | measured axis mapping details |
| `cfgs/`, `data/`, `logs/` | configs, volumes/outputs, MATLAB logs |
