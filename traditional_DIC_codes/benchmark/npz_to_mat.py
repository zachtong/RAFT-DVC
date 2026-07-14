"""Convert a RAFT-DVC sample .npz (keys I1, I2) into ref.mat / def.mat for the
headless MATLAB DVC wrappers (aldvc_headless.m / globaldvc_headless.m).

Each output .mat contains a single variable ``vol`` holding the 3-D array as
float64. IMPORTANT: no axis permutation is applied here. scipy.io.savemat
preserves index semantics: numpy vol[i,j,k] == MATLAB vol(i+1,j+1,k+1). The
measured mapping between numpy axes and the DVC (u,v,w) displacement
components is documented in benchmark/AXIS_CONVENTION.md.

Usage:
    python npz_to_mat.py sample.npz out_dir [--ref-key I1] [--def-key I2]
                                            [--prefix name]

Writes: out_dir/<prefix>_ref.mat and out_dir/<prefix>_def.mat
"""

import argparse
import os

import numpy as np
import scipy.io as sio


def npz_to_mat(npz_path, out_dir, ref_key="I1", def_key="I2", prefix=None):
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"npz file not found: {npz_path}")
    data = np.load(npz_path)
    for key in (ref_key, def_key):
        if key not in data:
            raise KeyError(
                f"key '{key}' not in {npz_path}; available: {sorted(data.keys())}"
            )
    ref = np.asarray(data[ref_key], dtype=np.float64)
    dfm = np.asarray(data[def_key], dtype=np.float64)
    if ref.ndim != 3 or dfm.ndim != 3:
        raise ValueError(f"expected 3-D arrays, got {ref.shape} / {dfm.shape}")

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(npz_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    ref_path = os.path.join(out_dir, f"{prefix}_ref.mat")
    def_path = os.path.join(out_dir, f"{prefix}_def.mat")
    sio.savemat(ref_path, {"vol": ref})
    sio.savemat(def_path, {"vol": dfm})
    print(f"wrote {ref_path}  shape={ref.shape}")
    print(f"wrote {def_path}  shape={dfm.shape}")
    return ref_path, def_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz_path")
    ap.add_argument("out_dir")
    ap.add_argument("--ref-key", default="I1")
    ap.add_argument("--def-key", default="I2")
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()
    npz_to_mat(args.npz_path, args.out_dir, args.ref_key, args.def_key, args.prefix)
