"""Generate the smoke-test volume pair with a smooth sinusoidal displacement.

Convention (same as the RAFT-DVC generator):  def[x] = ref[x + u(x)],
implemented with scipy.ndimage.map_coordinates (cubic spline, i.e. def is ref
sampled at x + u(x)).

Displacement field (numpy 0-based voxel coordinates p0,p1,p2; L=64; A=2.0):
    u0(p) = A * sin(2*pi * p1 / L)
    u1(p) = A * sin(2*pi * p2 / L)
    u2(p) = A * sin(2*pi * p0 / L)
Max |component| = 2 voxels, max |u| ~ 3.5 voxels, max gradient ~ 0.2.
The analytic parameters are stored in smoke_gt.npz so eval_smoke.py can
evaluate ground truth exactly (no grid interpolation error).

Also writes cfg JSONs:
    cfgs/smoke_aldvc.json  : winsize [16,16,16], winstepsize [8,8,8]
    cfgs/smoke_global.json : winstepsize [8,8,8], alpha = 0.8

alpha note: 0.8 = 1e-1 * mean(winstepsize). The initially suggested
alpha = 10*step = 80 over-regularizes z-score-normalized clean volumes and
shrinks the recovered field ~6x (measured mean EPE 2.22); alpha = 8 still
over-smooths (EPE 1.31); alpha = 0.8 passes (EPE 0.27). The original
main_FE_GlobalDVC L-curve sweep spans [1e-2..1e2]*mean(winstepsize).

Usage:  python make_smoke_pair.py
"""

import json
import os

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter, map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CFGS = os.path.join(HERE, "cfgs")

L = 64
A = 2.0


def make_blob_volume(n=64, sigma=2.0, seed=1):
    rng = np.random.default_rng(seed)
    vol = gaussian_filter(rng.standard_normal((n, n, n)), sigma=sigma)
    vol = (vol - vol.min()) / (vol.max() - vol.min())
    vol = np.clip((vol - 0.5) * 4.0 + 0.5, 0.0, 1.0)
    return vol.astype(np.float64)


def u_gen(p0, p1, p2):
    """Generator displacement field, numpy-axis components (broadcastable)."""
    w = 2.0 * np.pi / L
    return (A * np.sin(w * p1), A * np.sin(w * p2), A * np.sin(w * p0))


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(CFGS, exist_ok=True)

    ref = make_blob_volume(n=L, sigma=2.0, seed=1)

    idx = np.indices((L, L, L), dtype=np.float64)
    u0, u1, u2 = u_gen(idx[0], idx[1], idx[2])
    coords = np.stack([idx[0] + u0, idx[1] + u1, idx[2] + u2])
    dfm = map_coordinates(ref, coords, order=3, mode="reflect")

    sio.savemat(os.path.join(DATA, "smoke_ref.mat"), {"vol": ref})
    sio.savemat(os.path.join(DATA, "smoke_def.mat"), {"vol": dfm})
    np.savez(
        os.path.join(DATA, "smoke_gt.npz"),
        L=np.array(L), A=np.array(A),
        u0=u0.astype(np.float32), u1=u1.astype(np.float32), u2=u2.astype(np.float32),
    )
    umag = np.sqrt(u0**2 + u1**2 + u2**2)
    print(f"smoke pair written: L={L}, A={A}, max|u|={umag.max():.3f} voxels")

    cfg_aldvc = {
        "refFile": "data/smoke_ref.mat",
        "defFile": "data/smoke_def.mat",
        "winsize": [16, 16, 16],
        "winstepsize": [8, 8, 8],
        "outFile": "data/smoke_out_aldvc.mat",
    }
    cfg_global = {
        "refFile": "data/smoke_ref.mat",
        "defFile": "data/smoke_def.mat",
        "winstepsize": [8, 8, 8],
        "alpha": 0.8,  # 1e-1*mean(step); 10*step over-smooths (see module docstring)
        "outFile": "data/smoke_out_global.mat",
    }
    with open(os.path.join(CFGS, "smoke_aldvc.json"), "w") as f:
        json.dump(cfg_aldvc, f, indent=2)
    with open(os.path.join(CFGS, "smoke_global.json"), "w") as f:
        json.dump(cfg_global, f, indent=2)
    print("wrote cfgs/smoke_aldvc.json and cfgs/smoke_global.json")


if __name__ == "__main__":
    main()
