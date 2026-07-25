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
  "outFile":     "data/smoke_out_aldvc.mat",
  "clusterNo":   12
}
```

`clusterNo` (OPTIONAL, default **12**): worker count for the per-node IC-GN
loops (LocalICGN3 / Subpb13 benchmark shadow copies). `0` or `1` = serial
(original behavior). `>1` opens a persistent `parpool('Processes',N)` ONCE
and reuses it across calls (workers' paths are re-synced to the client path
on every call). Parallel results are **bitwise identical** to serial
(validated; see "Performance optimizations" below).

`globaldvc_headless(cfgFile)`:

```json
{
  "refFile":     "data/smoke_ref.mat",
  "defFile":     "data/smoke_def.mat",
  "winstepsize": [8, 8, 8],
  "alpha":       0.8,
  "outFile":     "data/smoke_out_global.mat",
  "fast":        true
}
```

`fast` (OPTIONAL, default **true**): `true` = vectorized FE assembly
(`funGlobalICGN3_fast.m`, ~40x faster on the ICGN stage at 128^3,
equivalent to reduction-order rounding ~8e-15); `false` = original
per-voxel assembly (benchmark shadow of `funGlobalICGN3`, waitbars
stripped, numerics untouched).

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

`aldvc_headless`: interpMethod='cubic', clusterNo=cfg (default 12; 1=serial),
initFFTMethod='bigxcorr', trackingMode='cumulative',
Subpb2FDOrFEM='finiteDifference', ICGNtol=1e-2, Subpb1ICGNMaxIterNum=50,
ADMMtol=1e-2, ALVarMu=1e-3, single
beta = 1e-2*mean(winstepsize)^2*ALVarMu (no L-curve sweep), 2 ADMM outer
iterations, gridRange = full volume, median-test threshold 2, no qDIC removal.

`globaldvc_headless`: winsize = winstepsize+[6,6,6] (ReadImage3 convention),
alphaList = [alpha] (kills the main-script line-126 L-curve override),
maxIter=20, tol=1e-2, GaussPtOrder=2, ClusterNo=1 (integer search stays
serial; global ICGN speed comes from `fast`), InitFFTMethod='bigxcorr',
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

1. **`case 0 || 1` bug in the ORIGINAL `LocalICGN3.m:35`/`Subpb13.m:33`**
   (evaluates to `case true`, so 0 falls into the `parfor` branch). The
   benchmark shadow copies fix it to `case {0,1}`, so `clusterNo` is now a
   real switch: 0/1 = serial, >1 = parfor. If you ever bypass the shadows
   and call the ALDVC originals, clusterNo must be exactly 1 for serial.
   The original Subpb13 parfor branch additionally references an undefined
   `ConvItPerEle2` and preallocates `UtempPar2/...` instead of `UtempPar/...`
   -- both fixed in the shadow copy (the former breaks parfor entirely).
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
6. **waitbars stripped (2026-07-14).** waitbars/figures are technically safe
   in `matlab -batch` on R2025b/Windows (created invisibly) but they cost
   real time (~3 s in the ALDVC integer search alone at 128^3, 2 calls per
   subset). All waitbar/parfor_progressbar calls in the hot path are removed
   in the benchmark shadow copies of `LocalICGN3`, `Subpb13`,
   `funGlobalICGN3`, `funIntegerSearch3Multigrid`, `funIntegerSearch3Mg`.
   Verified bitwise no-op on results (see "Performance optimizations").
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
12. **parfor has fixed overhead** (~3-4 s/call: `parallel.pool.Constant`
    image transfer + scheduling + worker path sync). On tiny grids (64^3
    smoke, 216 nodes) `clusterNo: 1` is FASTER than 12. Break-even is
    around ~1000 nodes; the 128^3 cases win clearly with 12 workers.
    Also: the pool is persistent by design — the first wrapper call pays
    ~15-30 s `parpool` startup once per MATLAB session; batch many samples
    per session to amortize it (or pre-open the pool).
13. **`funGlobalICGN3_fast` assumes a uniform axis-aligned brick mesh**
    (true for all MeshSetUp3 grids here). It verifies this at run time and
    falls back to the original (waitbar-stripped) implementation with a
    warning if violated.

## Performance optimizations (2026-07-14, validated)

Hot-path shadow copies in `benchmark\` (originals untouched; benchmark is
first on the wrappers' path so the shadows always win — verified via
`which` at run time):

1. **Waitbar strip** (`LocalICGN3`, `Subpb13`, `funGlobalICGN3`,
   `funIntegerSearch3Multigrid`, `funIntegerSearch3Mg`): zero numerical
   impact — serial shadow output is **bitwise identical** (max diff exactly
   0.0) to the pristine originals on both test cases.
2. **ALDVC parfor** (`cfg.clusterNo`, default 12): enables the authors' own
   parfor branches (with their 3 latent bugs fixed, see pitfall 1). The
   per-node IC-GN solves are embarrassingly parallel; parallel output is
   **bitwise identical** to serial (max diff exactly 0.0).
3. **Vectorized global FE assembly** (`funGlobalICGN3_fast.m`, `cfg.fast`,
   default true): replaces the interpreted per-voxel loop with matrix
   products (`tempA = B*B' + AregSum`, `tempb = B*res - AregSum*U_ele`),
   precomputes the translation-invariant regularizer block once (uniform
   brick mesh verified at run time, falls back to the original otherwise),
   batches `ba_interp3` over element chunks, and preallocates the sparse
   triplets. Differences are reduction-order rounding only:
   max |dU| = 8.0e-15 at 128^3 (identical ICGN iteration counts).

Measured on the 128^3 S6 sample (`cfgs/bench/S6/sample_00000_*.json`) and
the 64^3 smoke pair, single run each, `matlab -batch`, quiet-ish machine
(two idle MATLAB processes from another session could not be killed —
contention caveat; single-run timings, not medians):

| method | stage             | before (pristine) | after (optimized) | speedup | max num. diff |
|--------|-------------------|-------------------|--------------------|---------|---------------|
| ALDVC  | integer search    | 4.26 s            | 1.14 s             | 3.7x    | (bitwise 0)   |
| ALDVC  | local IC-GN       | 8.60 s            | 0.93 s             | 9.2x    | 0.0           |
| ALDVC  | global+ADMM       | 13.57 s           | 2.48 s             | 5.5x    | 0.0           |
| ALDVC  | **wall-clock**    | **27.0 s**        | **7.8 s**          | **3.5x**| **0.0**       |
| Global | integer search    | 4.59 s            | 3.24 s             | 1.4x    | (bitwise 0)   |
| Global | global IC-GN      | 19.32 s           | 0.44 s             | 43.9x   | see wall row  |
| Global | **wall-clock**    | **24.6 s**        | **4.7 s**          | **5.2x**| **8.0e-15**   |

64^3 smoke: ALDVC 10.5 -> 5.9 s wall (parfor overhead dominates tiny
216-node grids — see pitfall 12), Global 5.4 -> 1.3 s wall (max diff
U_global 4.4e-15, U_local/U_aldvc 0.0).

Reproduce/validate: `matlab -batch "validate_optimizations('opt')"` in
`benchmark\` (compares serial-shadow AND optimized paths against pristine
references stored in `data/val/ref_*.mat`; hard-fails above 1e-6, strict
gate at 1e-9). The pristine references were generated by
`validate_optimizations('ref')` BEFORE the shadow copies existed; rerunning
'ref' now exercises the (bitwise-identical) serial shadows instead.

NOTE on the older profiler evidence (`profile_hotspots.m`: ALDVC 34 s,
Global 129 s): `profile('on')` inflates interpreted loops severely (the
Global per-voxel assembly showed 122 s under the profiler vs 19.3 s real).
Use plain wall-clock for before/after claims; profiler only for ranking.

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
| `funIntegerSearch3Mg.m` | patched shadow copy: small-volume degenerate-grid fix (pitfall 11) + waitbars stripped |
| `funIntegerSearch3Multigrid.m` | shadow copy of ALDVC integer search, waitbars stripped |
| `LocalICGN3.m` | shadow copy of ALDVC local IC-GN: waitbars stripped, `case 0||1` bug fixed, parfor enabled via `clusterNo` |
| `Subpb13.m` | shadow copy of ALDVC ADMM Subproblem 1: same + 2 latent parfor bugs fixed |
| `funGlobalICGN3.m` | shadow copy of Global-DVC FE ICGN, waitbars stripped (serial `fast:false` path) |
| `funGlobalICGN3_fast.m` | vectorized FE assembly (default `fast:true` path), ~44x on the ICGN stage at 128^3 |
| `validate_optimizations.m` | equivalence + timing gate ('ref' = pristine baseline, 'opt' = compare serial/optimized vs refs) |
| `ba_interp3.mexw64` | working mex copy (from ALDVC) shadowing the broken Global_DVC build |
| `cfgs/bench/` | per-scenario cfg JSONs written by `scripts/paper1/benchmark_suite.py gen-cfgs` |
| `npz_to_mat.py` | RAFT-DVC npz (I1/I2) -> ref/def .mat (variable `vol`, no permutation) |
| `make_calibration_pair.py`, `check_calibration.py` | axis-convention measurement |
| `make_smoke_pair.py`, `eval_smoke.py` | smoke-test data gen + EPE evaluation |
| `eval_real_r8.py` | EPE evaluation of the real r8 sample against its npz `flow` GT |
| `AXIS_CONVENTION.md` | measured axis mapping details |
| `cfgs/`, `data/`, `logs/` | configs, volumes/outputs, MATLAB logs |
