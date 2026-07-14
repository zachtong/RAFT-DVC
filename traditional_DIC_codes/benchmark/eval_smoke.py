"""Evaluate the smoke-test DVC outputs against analytic ground truth.

Reads data/smoke_out_aldvc.mat (U_local, U_aldvc) and
data/smoke_out_global.mat (U_global), maps node coordinates to numpy 0-based
coordinates, and computes EPE at the nodes for all three methods.

Ground truth: the generator uses def[x] = ref[x + u_gen(x)], so the
DVC-measured displacement satisfies U(X) = -u_gen(X + U(X)). This is resolved
exactly by fixed-point iteration on the analytic field (contraction:
max |grad u_gen| ~ 0.2).

Axis mapping (measured, see AXIS_CONVENTION.md): identity —
U components (u,v,w) = numpy axes (0,1,2); coordinatesFEM is 1-based.

SUCCESS criterion: mean EPE < 0.5 voxel for U_local, U_aldvc, U_global.

Usage:  python eval_smoke.py
"""

import os
import sys

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

GT = np.load(os.path.join(DATA, "smoke_gt.npz"))
L = float(GT["L"])
A = float(GT["A"])


def u_gen(p):
    """Analytic generator displacement at numpy coords p (N,3) -> (N,3)."""
    w = 2.0 * np.pi / L
    return np.stack(
        [A * np.sin(w * p[:, 1]), A * np.sin(w * p[:, 2]), A * np.sin(w * p[:, 0])],
        axis=1,
    )


def gt_dvc_displacement(p, n_iter=25):
    """Solve U = -u_gen(p + U) by fixed-point iteration (exact GT for DVC)."""
    U = -u_gen(p)
    for _ in range(n_iter):
        U_new = -u_gen(p + U)
        if np.max(np.abs(U_new - U)) < 1e-12:
            U = U_new
            break
        U = U_new
    return U


def epe_stats(U_flat, coords_1based):
    U = np.asarray(U_flat).ravel()
    Um = np.stack([U[0::3], U[1::3], U[2::3]], axis=1)  # (N,3), identity mapping
    p = np.asarray(coords_1based, dtype=np.float64) - 1.0  # MATLAB 1-based -> numpy
    Ugt = gt_dvc_displacement(p)
    epe = np.linalg.norm(Um - Ugt, axis=1)
    return {
        "mean": float(epe.mean()),
        "median": float(np.median(epe)),
        "max": float(epe.max()),
        "n_nodes": int(len(epe)),
    }


def fmt_timing(t):
    # scipy loads MATLAB struct as nested object arrays
    names = t.dtype.names
    return ", ".join(f"{n}={float(np.squeeze(t[n][0, 0])):.2f}s" for n in names)


def main():
    results = {}

    m = sio.loadmat(os.path.join(DATA, "smoke_out_aldvc.mat"))
    coords = m["coordinatesFEM"]
    results["U_local (ALDVC local subset IC-GN)"] = epe_stats(m["U_local"], coords)
    results["U_aldvc (ALDVC ADMM global)"] = epe_stats(m["U_aldvc"], coords)
    print(f"[aldvc ] nodes={coords.shape[0]}  timing: {fmt_timing(m['timing'])}")

    g = sio.loadmat(os.path.join(DATA, "smoke_out_global.mat"))
    coords_g = g["coordinatesFEM"]
    results["U_global (FE Global DVC)"] = epe_stats(g["U_global"], coords_g)
    print(f"[global] nodes={coords_g.shape[0]}  timing: {fmt_timing(g['timing'])}")

    print("\n=== Smoke-test EPE at nodes (voxels) ===")
    ok = True
    for name, s in results.items():
        status = "PASS" if s["mean"] < 0.5 else "FAIL"
        ok &= s["mean"] < 0.5
        print(
            f"  {status}  {name:38s} mean={s['mean']:.4f}  "
            f"median={s['median']:.4f}  max={s['max']:.4f}  (n={s['n_nodes']})"
        )

    print("\nSUCCESS" if ok else "\nFAILURE: at least one method has mean EPE >= 0.5")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
