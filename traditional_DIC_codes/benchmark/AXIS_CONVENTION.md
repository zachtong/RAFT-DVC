# Axis convention: numpy arrays vs. ALDVC/Global-DVC displacement components

Measured 2026-07-13 with `make_calibration_pair.py` + `aldvc_headless.m`
(64^3 gaussian-blob volume, def = `np.roll(ref, +3, axis=K)`, winsize
[12,12,12], winstepsize [6,6,6]; see `logs/calib_axis*.log` and
`check_calibration.py`).

## Setup

- Volumes are saved from Python with `scipy.io.savemat(..., {"vol": arr})`.
  `savemat` preserves index semantics: `numpy vol[i,j,k] == MATLAB vol(i+1,j+1,k+1)`.
  **No permutation is applied anywhere.**
- `np.roll(ref, +3, axis=K)` moves features by **+3 voxels along numpy axis K**
  (`def[i] = ref[i-3]`, so a feature at index p appears at p+3).
- DVC solves `f(X) = g(X+U)` (f = ref, g = def), so U is the feature motion
  from ref to def.

## Measured result (medians over all nodes, U_aldvc)

| Shift applied                | u (comp 1) | v (comp 2) | w (comp 3) |
|------------------------------|-----------|-----------|-----------|
| +3 along numpy axis 0        | **+3.000** | +0.000    | -0.000    |
| +3 along numpy axis 1        | +0.000    | **+3.000** | -0.000    |
| +3 along numpy axis 2        | +0.000    | +0.000    | **+3.000** |

## Mapping (identity, positive sign)

| numpy axis | MATLAB array dim | DVC component in `U = [u1;v1;w1;u2;...]` | `coordinatesFEM` column |
|------------|------------------|------------------------------------------|--------------------------|
| axis 0     | dim 1 (rows)     | u = `U(1:3:end)`                         | column 1 ("x")           |
| axis 1     | dim 2 (cols)     | v = `U(2:3:end)`                         | column 2 ("y")           |
| axis 2     | dim 3 (pages)    | w = `U(3:3:end)`                         | column 3 ("z")           |

- Sign: a positive DVC component = feature motion in the +index direction of
  the corresponding numpy axis.
- `coordinatesFEM` is **1-based** (MATLAB voxel indices). To evaluate a numpy
  field at a node: `p_numpy = coordinatesFEM_row - 1.0`.
- Caveat: the internal "x/y" naming in the ALDVC plotting utilities swaps
  axes for display purposes; it does NOT affect the solver output layout
  measured above.

## Relation to the RAFT-DVC generator convention

If a pair is built as `def[x] = ref[x + u(x)]` (backward warp,
`scipy.ndimage.map_coordinates` sampling ref at `x + u`), a feature at `p`
in ref appears at `x ~= p - u(p)` in def, so the DVC-measured displacement
satisfies the implicit relation

    U_dvc(X) = -u(X + U_dvc(X))    (~= -u(X) to first order)

`eval_smoke.py` resolves this exactly with a fixed-point iteration when
comparing DVC output against the smoke-test warp field (wrong-sign check:
using `+u(X)` as GT gives mean EPE ~5 voxels; `-u(X+U)` gives ~0.3).

## Stored `flow` key in the real datasets (measured 2026-07-13)

For `G:\Zixiang_local_data\raft-dvc\data_paper1_axis2\r8_medium_size128\test\sample_00000.npz`
(`flow` shape (3,128,128,128), components along numpy axes 0,1,2), the DVC
outputs match the stored flow DIRECTLY:

    U_dvc(X) ~= +flow[:, X]     (median EPE 0.38-0.49 vox, max |flow| = 15.9)

whereas the backward-warp hypothesis `U = -flow(X+U)` is off by ~5 voxels
median. I.e., the npz `flow` field is already the ref->def feature
displacement in DVC convention — use it as GT without sign flips or
fixed-point inversion (see `eval_real_r8.py`).
