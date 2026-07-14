"""Evaluate the headless DVC outputs on the real r8 128^3 sample against the
GT 'flow' stored in the npz (shape (3,128,128,128), components along numpy
axes 0,1,2).

Because the generator's flow sign convention is being cross-checked here, EPE
is reported under both hypotheses:
  H_direct : U_dvc(X) =  flow(X)            (flow = feature motion ref->def)
  H_bwarp  : U_dvc(X) = -flow(X + U)        (flow = u_gen of def[x]=ref[x+u])
The matching hypothesis is the one with small EPE.

Usage:  python eval_real_r8.py [npz_path]
"""

import os
import sys

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DEFAULT_NPZ = r"G:\Zixiang_local_data\raft-dvc\data_paper1_axis2\r8_medium_size128\test\sample_00000.npz"


def make_interp(flow):
    axes = [np.arange(s, dtype=np.float64) for s in flow.shape[1:]]
    return [
        RegularGridInterpolator(axes, flow[k], bounds_error=False, fill_value=None)
        for k in range(3)
    ]


def flow_at(interps, p):
    return np.stack([f(p) for f in interps], axis=1)


def epe_stats(Um, Ugt):
    epe = np.linalg.norm(Um - Ugt, axis=1)
    return f"mean={epe.mean():.4f} median={np.median(epe):.4f} max={epe.max():.4f}"


def eval_field(name, U_flat, coords, interps):
    U = np.asarray(U_flat).ravel()
    Um = np.stack([U[0::3], U[1::3], U[2::3]], axis=1)
    p = np.asarray(coords, dtype=np.float64) - 1.0

    gt_direct = flow_at(interps, p)

    Ub = -flow_at(interps, p)
    for _ in range(15):
        Ub = -flow_at(interps, p + Ub)

    print(f"  {name}:")
    print(f"    H_direct (U=+flow(X))    {epe_stats(Um, gt_direct)}")
    print(f"    H_bwarp  (U=-flow(X+U))  {epe_stats(Um, Ub)}")


def main():
    npz_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NPZ
    d = np.load(npz_path)
    flow = np.asarray(d["flow"], dtype=np.float64)
    print(f"flow stats: comp means {flow.mean(axis=(1,2,3)).round(3)}, "
          f"max |flow| = {np.abs(flow).max():.3f}")
    interps = make_interp(flow)

    m = sio.loadmat(os.path.join(DATA, "real_r8_out_aldvc.mat"))
    print(f"[aldvc ] nodes={m['coordinatesFEM'].shape[0]}")
    eval_field("U_local", m["U_local"], m["coordinatesFEM"], interps)
    eval_field("U_aldvc", m["U_aldvc"], m["coordinatesFEM"], interps)

    gpath = os.path.join(DATA, "real_r8_out_global.mat")
    if os.path.isfile(gpath):
        g = sio.loadmat(gpath)
        print(f"[global] nodes={g['coordinatesFEM'].shape[0]}")
        eval_field("U_global", g["U_global"], g["coordinatesFEM"], interps)


if __name__ == "__main__":
    main()
